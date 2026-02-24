# """
# @경로: db/connection/pg_depends.py
# @설명: DB 세션 의존성 주입 유틸
# @작성자: Untumnbear
# @수정내역: 

# """
# from fastapi import HTTPException, Request
# from sqlalchemy.exc import OperationalError
# from sqlalchemy.orm import Session
# # from apps.backend.db.client import db_client # Local 기본 SQLite
# from db.connection.pg_client import pg_client # PostgreSQL

# def get_pg_db(request: Request):
#     """
#     PostgreSQL 의존성 (자동 트랜잭션 관리)
#     - 성공 시: 자동 커밋
#     - 실패 시: 자동 롤백
#     """
#     db: Session | None = None
#     try:
#         db = pg_client.get_session()
#         request.app.state.db_down = False # 연결이 되었으면 db_down 상태를 False 로
#         yield db
#         # 모든 작업이 성공하면 자동 커밋
#         db.commit()
#     except OperationalError:
#         request.app.state.db_down = True # 연결이 되었으면 db_down 상태를 True 로
#         if db is not None:
#             db.rollback()
#         raise HTTPException(status_code=503, detail="Database unavailable")
#     except Exception:
#         # 다른 예외가 발생하면 롤백
#         if db is not None:
#             db.rollback()
#         raise
#     finally:
#         if db is not None:
#             db.close()