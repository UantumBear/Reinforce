"""
@경로 : conf/config.py
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv 


# 로거는 순환 참조 방지를 위해 여기서는 간단한 print나 logging 모듈을 직접 쓰는 게 안전할 수 있음
# 기존 utils.log.logging 사용 시 순환 참조가 없다면 그대로 사용 가능
try:
    from utils.log.logging import logger
except ImportError:
    import logging
    logger = logging.getLogger(__name__)

# 1. 경로 설정 (pathlib 사용)
BASE_DIR = Path(__file__).resolve().parent # conf/  (== os.path.dirname(os.path.abspath(__file__)))
PROJECT_ROOT = BASE_DIR.parent             # 상위 폴더, Project Root


class AppConfig:
    """
    환경변수 및 설정값을 관리하는 클래스
    실제 값은 Settings.setup() 호출 시 로드됨
    """

    """ 초기값 선언, 실제 값은 setup_environment() 실행 후 갱신 """
    # [경로]
    PROJECT_ROOT = PROJECT_ROOT
    DATA_DIR = PROJECT_ROOT / "data"
    LOG_DIR = PROJECT_ROOT / "logs"

    # [공통 설정]
    USE_AZURE = None

    # [현재 활성화된 모델 정보 (Agent는 이것만 참조하면 됨)]
    GENERATOR_MODEL = None  # 실제 사용할 모델명 (예: gpt-4, gemini-pro)
    EMBEDDING_MODEL = None  # 임베딩 모델명
    API_KEY = None          # 현재 활성화된 API 키
    API_BASE = None         # Endpoint
    API_VERSION = None
    OPTIMIZER_MODEL = None  # Optimizer LLM 모델명
    TESTER_MODEL = None     # Target/Tester LLM 모델명

    # [Azure Raw Config] 
    AZURE_OPENAI_API_KEY = None
    AZURE_OPENAI_ENDPOINT = None
    AZURE_OPENAI_API_VERSION = None
    AZURE_GPT5_CHAT_DEPLOYMENT = None
    AZURE_GPTO4_MINI_DEPLOYMENT = None
    AZURE_GPT5_NANO_DEPLOYMENT = None
    AZURE_GPT5_MINI_DEPLOYMENT = None
    
    # [Azure 임베딩 모델 설정]
    AZURE_EMBEDDING3_SMALL_DEPLOYMENT = None  # Azure 임베딩 deployment 이름 (예: text-embedding-3-small)
    
    # [LLM 분리 설정]
    OPTIMIZER_MODEL = None
    TESTER_MODEL = None
    # [Google Raw Config]
    GOOGLE_API_KEY = None
    # [EMBEDDING 모델 설정]
    LOCAL_EMBEDDING_MODEL_PATH = "model/embedding/ko-sroberta-multitask" # 이건 os 에서 읽지 않으므로 고정값으로 둠
    
    # [Database Config] DB 관련 설정 추가
    DB_HOST = None
    DB_PORT = None
    DB_USER = None
    DB_PASSWORD = None
    DB_NAME = None
    DB_SSLMODE = None
    DATABASE_URL = None
    DB_POOL_SIZE = None
    DB_MAX_OVERFLOW = None
    DB_POOL_RECYCLE = None
    DB_CONNECT_TIMEOUT = None


    @classmethod
    def setup(cls):
        """
        [하이브리드 설정 로드]
        1. 로컬 개발 환경: .env 및 key/*.env 파일에서 설정을 읽어옵니다.
        2. 서버 배포 환경: 이미 OS 환경변수에 설정된 값이 있다면 그걸 우선합니다.
        """
        # 1. 메인 .env 로드 (로컬/서버 공통)
        # override=False: 서버의 시스템 환경변수가 있다면 덮어쓰지 않음 (서버 우선)
        main_env_path = cls.PROJECT_ROOT / ".env"
        if main_env_path.exists():
            load_dotenv(main_env_path, override=False)

        # 2. USE_AZURE 플래그 확인
        use_azure_raw = os.getenv("USE_AZURE", "False")
        cls.USE_AZURE = str(use_azure_raw).lower() in ("true", "1", "yes", "on")

        # 3. 로컬 전용 키 파일 로드 시도 (key 폴더)
        if cls.USE_AZURE:
            azure_env_path = cls.PROJECT_ROOT / "key/azure.env"
            if azure_env_path.exists():
                logger.info(f">>> [Local] Azure 키 파일을 로드합니다: {azure_env_path}")
                load_dotenv(azure_env_path, override=True) # 로컬 파일이 우선
            else:
                logger.info(">>> [Server/Env] Azure 키 파일이 없습니다. 시스템 환경변수를 확인합니다.")
            
            # 매핑 실행
            cls._configure_azure()
            
        else:
            gemini_env_path = cls.PROJECT_ROOT / "key/gemini.env"
            if gemini_env_path.exists():
                logger.info(f">>> [Local] Gemini 키 파일을 로드합니다: {gemini_env_path}")
                load_dotenv(gemini_env_path, override=True)
            else:
                logger.info(">>> [Server/Env] Gemini 키 파일이 없습니다. 시스템 환경변수를 확인합니다.")
            
            # 매핑 실행
            cls._configure_google()

        # 4. DB 설정 파일 로드 (공통)
        db_env_path = cls.PROJECT_ROOT / "key/db.env"
        if db_env_path.exists():
            logger.info(f">>> [Local] DB 설정 파일을 로드합니다: {db_env_path}")
            load_dotenv(db_env_path, override=True)
            cls._configure_database()
        else:
            logger.warning(">>> [Warning] DB 설정 파일이 없습니다. 시스템 환경변수를 확인합니다.")
            cls._configure_database()  # 시스템 환경변수에서 로드 시도

        # 5. [최종 검증] 파일이 있든 없든, 결과적으로 API KEY가 세팅되었는가?
        if not cls.API_KEY:
            logger.error("!!! [CRITICAL] API Key가 로드되지 않았습니다. !!!")
            logger.error("로컬이라면 key/*.env 파일을, 서버라면 환경변수 설정을 확인해주세요.")
            sys.exit(1)

    @classmethod
    def _configure_azure(cls):
        """Azure 환경변수를 공통 변수에 매핑"""
        cls.AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
        cls.AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
        cls.AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION")
        
        # 모델명 매핑 (여기서 사용할 모델을 지정)
        # 예: .env에 AZURE_GPT5_NANO_DEPLOYMENT=gpt-4-turbo 라고 되어 있다면
        # cls.AZURE_DEPLOYMENT_NAME = os.getenv("AZURE_GPT5_NANO_DEPLOYMENT") 
        cls.AZURE_GPT5_NANO_DEPLOYMENT = os.getenv("AZURE_GPT5_NANO_DEPLOYMENT") 
        cls.AZURE_GPT5_MINI_DEPLOYMENT = os.getenv("AZURE_GPT5_MINI_DEPLOYMENT") 
        cls.AZURE_GPTO4_MINI_DEPLOYMENT = os.getenv("AZURE_GPTO4_MINI_DEPLOYMENT")

        # 임베딩 모델 설정
        cls.AZURE_EMBEDDING3_SMALL_DEPLOYMENT = os.getenv("AZURE_EMBEDDING3_SMALL_DEPLOYMENT") # text-embedding-3-small
        # For RAGAS
        cls.AZURE_EMBEDDING_DEPLOYMENT = cls.AZURE_EMBEDDING3_SMALL_DEPLOYMENT

        # [중요] Agent가 사용할 공통 인터페이스 설정
        cls.API_KEY = cls.AZURE_OPENAI_API_KEY
        cls.API_BASE = cls.AZURE_OPENAI_ENDPOINT
        cls.API_VERSION = cls.AZURE_OPENAI_API_VERSION
        # cls.GENERATOR_MODEL = cls.AZURE_GPT5_NANO_DEPLOYMENT
        cls.OPTIMIZER_MODEL = cls.AZURE_GPT5_NANO_DEPLOYMENT        
        # 최적화 모델, 현재는 가장 싼 nano 를 쓰고 있지만, 전체 설계 완료 후에는 일반 모델로 변경하기
        cls.TESTER_MODEL = cls.AZURE_GPT5_MINI_DEPLOYMENT
        # Tester 모델, 해당 모델은 따로 제약을 걸 이유가 없음, 아무 모델이나 사용하기.
        cls.RAGAS_CHAT_MODEL = os.getenv("AZURE_GPTO4_MINI_DEPLOYMENT")  # RAGAS 평가자 모델
        # RAGAS 평가자 모델은 gpt-5-mini, gpt-5-nano 를 지원하지 않음 (temperature 파라미터 문제)
        
        # 유효성 검사
        if not cls.API_KEY or not cls.API_BASE:
            logger.error("Azure 필수 설정(Key, Endpoint)이 누락되었습니다.")

    @classmethod
    def _configure_google(cls):
        """Google 환경변수를 공통 변수에 매핑"""
        cls.GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
        
        # [중요] Agent가 사용할 공통 인터페이스 설정
        cls.API_KEY = cls.GOOGLE_API_KEY
        cls.GENERATOR_MODEL = "gemini-1.5-pro" # 혹은 os.getenv로 받기
        cls.OPTIMIZER_MODEL = os.getenv("OPTIMIZER_MODEL", cls.GENERATOR_MODEL)
        cls.TESTER_MODEL = os.getenv("TESTER_MODEL", cls.GENERATOR_MODEL)
        
        
        if not cls.API_KEY:
            logger.warning("Google API Key가 설정되지 않았습니다.")
    
    @classmethod
    def _configure_database(cls):
        """Database 환경변수를 클래스 변수에 매핑"""
        cls.DB_HOST = os.getenv("DB_HOST", "localhost")
        cls.DB_PORT = int(os.getenv("DB_PORT", "5432"))
        cls.DB_USER = os.getenv("DB_USER", "postgres")
        cls.DB_PASSWORD = os.getenv("DB_PASSWORD", "password")
        cls.DB_NAME = os.getenv("DB_NAME", "reinforcement_learning")
        cls.DB_SSLMODE = os.getenv("DB_SSLMODE", "prefer")
        cls.DATABASE_URL = os.getenv("DATABASE_URL")
        cls.DB_POOL_SIZE = int(os.getenv("DB_POOL_SIZE", "5"))
        cls.DB_MAX_OVERFLOW = int(os.getenv("DB_MAX_OVERFLOW", "10"))
        cls.DB_POOL_RECYCLE = int(os.getenv("DB_POOL_RECYCLE", "1800"))
        cls.DB_CONNECT_TIMEOUT = int(os.getenv("DB_CONNECT_TIMEOUT", "30"))
        
        # DATABASE_URL이 없으면 개별 설정으로 생성
        if not cls.DATABASE_URL:
            cls.DATABASE_URL = f"postgresql://{cls.DB_USER}:{cls.DB_PASSWORD}@{cls.DB_HOST}:{cls.DB_PORT}/{cls.DB_NAME}?sslmode={cls.DB_SSLMODE}"
            
        logger.info(f">>> [DB Config] Host: {cls.DB_HOST}:{cls.DB_PORT}, DB: {cls.DB_NAME}")
    
    @classmethod
    def get_database_url(cls) -> str:
        """SQLAlchemy용 DATABASE_URL 반환"""
        if not cls.DATABASE_URL:
            cls._configure_database()
        return cls.DATABASE_URL


Settings = AppConfig



    # @staticmethod
    # def setup_environment():
    #     """환경변수 파일 로드 및 설정"""
    #     current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
    #     # 먼저 루트의 .env 파일에서 USE_AZURE 설정 로드
    #     main_env_path = os.path.join(current_dir, ".env")
    #     if os.path.exists(main_env_path):
    #         logger.info(f"메인 환경변수 파일 로드: {main_env_path}")
    #         Env.load_env_file(main_env_path)
            
    #     # USE_AZURE 값에 따라 적절한 환경변수 파일 로드
    #     use_azure_raw = os.getenv("USE_AZURE", "False")
    #     use_azure = use_azure_raw.lower() in ("true", "1", "yes", "on")
        
    #     if use_azure:
    #         logger.info("Azure 모드로 실행 - azure.env 로드")
    #         azure_env_path = os.path.join(current_dir, "key", "azure.env")
    #         if os.path.exists(azure_env_path):
    #             Env.load_env_file(azure_env_path)
    #             logger.info(f"Azure 환경변수 파일 로드: {azure_env_path}")
    #             # AZURE
    #             Env.USE_AZURE = os.getenv("USE_AZURE", "False").lower() in ("true", "1", "yes", "on")
    #             Env.AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
    #             Env.AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
    #             Env.AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION")
    #             Env.AZURE_GPT5_CHAT_DEPLOYMENT = os.getenv("AZURE_GPT5_CHAT_DEPLOYMENT")
    #             Env.AZURE_GPT4O_MINI_DEPLOYMENT = os.getenv("AZURE_GPT4O_MINI_DEPLOYMENT")
    #            
               
    #         else:
    #             logger.error(f"Azure 환경변수 파일을 찾을 수 없습니다: {azure_env_path}")
    #     else:
    #         logger.info("Gemini 모드로 실행 - gemini.env 로드")
    #         gemini_env_path = os.path.join(current_dir, "key", "gemini.env")
    #         if os.path.exists(gemini_env_path):
    #             Env.load_env_file(gemini_env_path)
    #             logger.info(f"Gemini 환경변수 파일 로드: {gemini_env_path}")
    #              # GOOGLE
    #             Env.GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    #         else:
    #             logger.error(f"Gemini 환경변수 파일을 찾을 수 없습니다: {gemini_env_path}")
            
    #     logger.info(f"최종 USE_AZURE 설정: {use_azure} (원본값: {use_azure_raw})")
    #     logger.info("환경변수 파일 로드 완료")

    # @staticmethod
    # def check_google_api_key():
    #     """Google API Key 확인 및 안내"""
    #     if "GOOGLE_API_KEY" not in os.environ:
    #         print("[GUIDE] 실행을 위해 Google AI Studio 무료 키가 필요합니다.")
    #         # os.environ["GOOGLE_API_KEY"] = input("Google API Key 입력: ") # Colab 등에서 사용 시
    #         return False
    #     return True
    
    # @staticmethod
    # def check_azure_api_key():
    #     """Azure API Key 확인"""
    #     # Azure 사용 시 필요한 필수 키들이 있는지 체크
    #     required_keys = ["AZURE_OPENAI_API_KEY", "AZURE_OPENAI_ENDPOINT"]
    #     missing_keys = [key for key in required_keys if key not in os.environ]
        
    #     if missing_keys:
    #         logger.error(f"[Env] Azure 관련 필수 키가 누락되었습니다: {missing_keys}")
    #         print("[GUIDE] key/azure.env 파일에 다음 키들이 정의되어 있어야 합니다:")
    #         print(" - AZURE_OPENAI_API_KEY")
    #         print(" - AZURE_OPENAI_ENDPOINT")
    #         print(" - AZURE_OPENAI_API_VERSION (Optional)")
    #         return False
    #     return True
    

