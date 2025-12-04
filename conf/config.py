"""
@경로 : conf/config.py
"""

import os
from utils.log.logging import logger


base_dir = os.path.dirname(os.path.abspath(__file__)) # 현재 파일 위치 (conf/)
project_root = os.path.dirname(base_dir)         # 상위 폴더 (Project Root)


# 환경변수
class Env():
    
    USE_AZURE = os.getenv("USE_AZURE")
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    PROJECT_ROOT = project_root

    @staticmethod
    def load_env_file(env_file_path: str) -> None:
        """
        .env 파일에서 환경변수를 로드하는 함수
        
        Args:
            env_file_path (str): .env 파일의 경로
        """
        try:
            if os.path.exists(env_file_path):
                with open(env_file_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        # 빈 줄이나 주석 무시
                        if not line or line.startswith('#'):
                            continue
                        # KEY=VALUE 형태 파싱
                        if '=' in line:
                            key, value = line.split('=', 1)
                            os.environ[key.strip()] = value.strip()
                            logger.info(f"환경변수 로드됨: env_file_path: {env_file_path}, {key.strip()}")
            else:
                logger.info(f"환경변수 파일을 찾을 수 없습니다: {env_file_path}")
        except Exception as e:
            logger.error(f"환경변수 파일 로드 중 오류: {e}")

    @staticmethod
    def setup_environment():
        """환경변수 파일 로드 및 설정"""
        current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        env_file_path = os.path.join(current_dir, "key", "gemini.env")
        logger.info(f"환경변수 파일 로드 중: {env_file_path}")
        Env.load_env_file(env_file_path)
        logger.info("환경변수 파일 로드 완료")

    @staticmethod
    def check_google_api_key():
        """Google API Key 확인 및 안내"""
        if "GOOGLE_API_KEY" not in os.environ:
            print("[GUIDE] 실행을 위해 Google AI Studio 무료 키가 필요합니다.")
            # os.environ["GOOGLE_API_KEY"] = input("Google API Key 입력: ") # Colab 등에서 사용 시
            return False
        return True
    

