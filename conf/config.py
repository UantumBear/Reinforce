"""
@경로 : conf/config.py
"""

import os
from utils.log.logging import logger


base_dir = os.path.dirname(os.path.abspath(__file__)) # 현재 파일 위치 (conf/)
project_root = os.path.dirname(base_dir)         # 상위 폴더 (Project Root)


# 환경변수
class Env():
    """ 초기값 선언, 실제 값은 setup_environment() 실행 후 갱신 """
    # AZURE
    USE_AZURE = None
    AZURE_OPENAI_API_KEY = None
    AZURE_OPENAI_ENDPOINT = None
    AZURE_OPENAI_API_VERSION = None
    AZURE_GPT5_CHAT_DEPLOYMENT = None
    AZURE_GPT4O_MINI_DEPLOYMENT = None
    # GOOGLE
    GOOGLE_API_KEY = None

    # DIR
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
        
        # 먼저 루트의 .env 파일에서 USE_AZURE 설정 로드
        main_env_path = os.path.join(current_dir, ".env")
        if os.path.exists(main_env_path):
            logger.info(f"메인 환경변수 파일 로드: {main_env_path}")
            Env.load_env_file(main_env_path)
            
        # USE_AZURE 값에 따라 적절한 환경변수 파일 로드
        use_azure_raw = os.getenv("USE_AZURE", "False")
        use_azure = use_azure_raw.lower() in ("true", "1", "yes", "on")
        
        if use_azure:
            logger.info("Azure 모드로 실행 - azure.env 로드")
            azure_env_path = os.path.join(current_dir, "key", "azure.env")
            if os.path.exists(azure_env_path):
                Env.load_env_file(azure_env_path)
                logger.info(f"Azure 환경변수 파일 로드: {azure_env_path}")
                # AZURE
                Env.USE_AZURE = os.getenv("USE_AZURE", "False").lower() in ("true", "1", "yes", "on")
                Env.AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
                Env.AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
                Env.AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION")
                Env.AZURE_GPT5_CHAT_DEPLOYMENT = os.getenv("AZURE_GPT5_CHAT_DEPLOYMENT")
                Env.AZURE_GPT4O_MINI_DEPLOYMENT = os.getenv("AZURE_GPT4O_MINI_DEPLOYMENT")
               
            else:
                logger.error(f"Azure 환경변수 파일을 찾을 수 없습니다: {azure_env_path}")
        else:
            logger.info("Gemini 모드로 실행 - gemini.env 로드")
            gemini_env_path = os.path.join(current_dir, "key", "gemini.env")
            if os.path.exists(gemini_env_path):
                Env.load_env_file(gemini_env_path)
                logger.info(f"Gemini 환경변수 파일 로드: {gemini_env_path}")
                 # GOOGLE
                Env.GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
            else:
                logger.error(f"Gemini 환경변수 파일을 찾을 수 없습니다: {gemini_env_path}")
            
        logger.info(f"최종 USE_AZURE 설정: {use_azure} (원본값: {use_azure_raw})")
        logger.info("환경변수 파일 로드 완료")

    @staticmethod
    def check_google_api_key():
        """Google API Key 확인 및 안내"""
        if "GOOGLE_API_KEY" not in os.environ:
            print("[GUIDE] 실행을 위해 Google AI Studio 무료 키가 필요합니다.")
            # os.environ["GOOGLE_API_KEY"] = input("Google API Key 입력: ") # Colab 등에서 사용 시
            return False
        return True
    
    @staticmethod
    def check_azure_api_key():
        """Azure API Key 확인"""
        # Azure 사용 시 필요한 필수 키들이 있는지 체크
        required_keys = ["AZURE_OPENAI_API_KEY", "AZURE_OPENAI_ENDPOINT"]
        missing_keys = [key for key in required_keys if key not in os.environ]
        
        if missing_keys:
            logger.error(f"[Env] Azure 관련 필수 키가 누락되었습니다: {missing_keys}")
            print("[GUIDE] key/azure.env 파일에 다음 키들이 정의되어 있어야 합니다:")
            print(" - AZURE_OPENAI_API_KEY")
            print(" - AZURE_OPENAI_ENDPOINT")
            print(" - AZURE_OPENAI_API_VERSION (Optional)")
            return False
        return True
    

