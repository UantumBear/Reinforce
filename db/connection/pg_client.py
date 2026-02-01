"""
@프로젝트: Reinforce
@경로: db/connection/pg_client.py
@설명: PostgreSQL 싱글톤 클라이언트 (SQLAlchemy sync + psycopg)
@작성자: Untumnbear


"""

from typing import Optional, Generator
from threading import Lock
from sqlalchemy import create_engine, text
from sqlalchemy.engine import URL, Engine
from sqlalchemy.orm import sessionmaker, Session, declarative_base
from sqlalchemy.exc import OperationalError
import time
from conf.config import Settings

# utils.log.logging이 없으면 기본 logging 사용
try:
    from utils.log.logging import logger
except ImportError:
    import logging
    logger = logging.getLogger(__name__)


class PgClient:
    _engine: Optional[Engine] = None
    _SessionLocal: Optional[sessionmaker] = None
    _lock: Lock = Lock()

    _max_retry_count: int = 3      # 요청 1번당 최대 3번 시도
    _retry_interval: float = 1.0   # 시도 사이 대기 시간(초)
    
    # SQLAlchemy Base 클래스 (클래스 속성으로 한 번만 생성)
    Base = declarative_base() # SQLAlchemy 관례상 클래스 속성이 더 적합.

    def __init__(self) -> None:
        # 지연 초기화(Lazy init) → 최초 사용 시 생성
        pass

    def _init_engine(self) -> None:

        if self._engine is not None:
            return
        
        # 초기화 경쟁 방지
        with self._lock:
            if self._engine is not None:
                return
        
        # Config 초기화
        Settings.setup()
        
        url = URL.create(
            "postgresql+psycopg",
            username=Settings.DB_USER,      # 자동 인코딩 처리
            password=Settings.DB_PASSWORD,  # 자동 인코딩 처리
            host=Settings.DB_HOST,
            port=Settings.DB_PORT,
            database=Settings.DB_NAME,
            query={
                "sslmode": Settings.DB_SSLMODE,
                "connect_timeout": str(Settings.DB_CONNECT_TIMEOUT), 
            },
        )
        engine = create_engine(
            url,
            pool_pre_ping=True, # 유휴 연결 체크
            pool_size=Settings.DB_POOL_SIZE,
            max_overflow=Settings.DB_MAX_OVERFLOW,
            pool_recycle=Settings.DB_POOL_RECYCLE,
            future=True,
        )
        self._engine = engine

        SessionLocal = sessionmaker(
            bind=self._engine,
            autoflush=False,
            autocommit=False,
            expire_on_commit=False,
            class_=Session,
            future=True,
        )        
        self._SessionLocal = SessionLocal

    def get_session(self) -> Session:
        """
            DB 세션 생성.
            - OperationalError(연결 문제)일 때만 몇 번 재시도.
            - 그 외 예외는 바로 raise 해서 디버깅 쉽게 유지.
        """
        # if self._SessionLocal is None:
        #     self._init_engine()
        # assert self._SessionLocal is not None
        # return self._SessionLocal()
        last_exc: Optional[Exception] = None

        for attempt in range(1, self._max_retry_count + 1):
            try:
                if self._SessionLocal is None:
                    self._init_engine()
                assert self._SessionLocal is not None
                return self._SessionLocal()

            except OperationalError as e:
                # 진짜 연결 이슈 → 재시도 대상
                last_exc = e
                logger.error(f"[PG_CLIENT] 세션 생성 실패 (연결 오류) - 시도 {attempt}/{self._max_retry_count}: {e}")

                # 풀과 엔진 초기화 후 재시도 준비
                self.dispose()

                if attempt < self._max_retry_count:
                    time.sleep(self._retry_interval)  # 1초 정도 대기 후 다시 시도
                else:
                    # 마지막 시도도 실패 → 루프 밖에서 raise
                    break

            except Exception:
                # 코드/설정 버그 같은 건 재시도해봐야 소용 없으니 바로 올려보내기
                raise

        # 여기까지 왔다는 건 OperationalError가 계속 발생한 경우
        raise last_exc or Exception("DB 세션 생성 실패")

    def ping(self) -> bool:
        # 간단한 헬스체크
        if self._engine is None:
            self._init_engine()
        assert self._engine is not None
        with self._engine.connect() as conn:
            # exec_driver_sql이 헬스체크에 가장 단순
            conn.exec_driver_sql("SELECT 1")
        return True
    
    def dispose(self) -> None:
        """
        엔진이 들고 있던 커넥션 풀을 정리(폐기)해서, 
        풀 안의 연결들을 닫고, 다음부터는 새 연결을 만들게 하는 함수. 
        보통 앱 종료 시점이나 DB가 리셋/장시간 끊겼을 때 호출한다.
        """
        # if self._engine is not None:
        #     self._engine.dispose()
        with self._lock:
            if self._engine is not None:
                try:
                    self._engine.dispose()
                except Exception:
                    pass
            self._engine = None
            self._SessionLocal = None


# 모듈 레벨 싱글톤
pg_client = PgClient()
