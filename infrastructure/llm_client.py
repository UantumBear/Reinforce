"""
@경로: infrastructure/llm_client.py
@설명: DSPy LLM 초기화 및 환경설정 (Azure OpenAI 연결)
"""

import dspy
import sys
from conf.config import Settings

def setup_lms(verbose=True):
    """
    DSPy LLM 환경을 설정하고 전역 configure를 수행
    
    @ verbose (bool): 설정 정보 출력 여부
    
    @ Return:
        dspy.LM: 설정된 LLM 객체
    """
    Settings.setup()
    
    # Azure 설정 확인 (AppConfig에 값이 있는지 체크)
    if not Settings.AZURE_OPENAI_API_KEY:
        print("[FAIL] Azure API Key가 설정되지 않았습니다. .env 파일을 확인해주세요.")
        sys.exit(1)
    
    if verbose:
        print("[CHECK] LLM 설정 확인:")
        print(f"   - Model: {Settings.GENERATOR_MODEL}")
        print(f"   - Endpoint: {Settings.API_BASE}")
        print()
    
    # LLM 객체 생성
    lm = dspy.LM(
        model=f"azure/{Settings.GENERATOR_MODEL}" if Settings.USE_AZURE else f"google/{Settings.GENERATOR_MODEL}",
        api_key=Settings.API_KEY,
        api_base=Settings.API_BASE,
        api_version=Settings.API_VERSION
    )
    
    # DSPy 전역 설정
    # "앞으로 별말 없으면 무조건 이 Azure 모델 써!" 라고 전역 변수로 박아두는 것
    dspy.configure(lm=lm)
    
    return lm

def get_configured_llm():
    """이미 configure된 LLM을 가져오기"""
    return dspy.settings.lm