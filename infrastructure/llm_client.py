"""
@경로: infrastructure/llm_client.py
@설명: DSPy LLM 초기화 및 환경설정 (Azure OpenAI 연결)
"""

import dspy
import sys
from conf.config import Settings

def setup_lms(verbose=True):
    """
    테스터 LLM 환경을 설정하고 전역 configure를 수행
    (Optimizer가 만든 프롬프트를 테스트하는 깨끗한 LLM)
    
    @ verbose (bool): 설정 정보 출력 여부
    
    @ Return:
        dspy.LM: 설정된 테스터 LLM 객체
    """
    Settings.setup()
    
    # Azure 설정 확인 (AppConfig에 값이 있는지 체크)
    if not Settings.AZURE_OPENAI_API_KEY:
        print("[FAIL] Azure API Key가 설정되지 않았습니다. .env 파일을 확인해주세요.")
        sys.exit(1)
    
    if verbose:
        print("[CHECK] Tester LLM 설정 확인:")
        print(f"   - Model: {Settings.TESTER_MODEL}")
        print(f"   - Endpoint: {Settings.API_BASE}")
        print()
    
    # Tester LLM 객체 생성 (깨끗한 LLM - 프롬프트 테스트용)
    lm = dspy.LM(
        model=f"azure/{Settings.TESTER_MODEL}" if Settings.USE_AZURE else f"google/{Settings.TESTER_MODEL}",
        api_key=Settings.API_KEY,
        api_base=Settings.API_BASE,
        api_version=Settings.API_VERSION
    )
    
    # DSPy 전역 설정
    # "앞으로 별말 없으면 무조건 이 Azure 모델 써!" 라고 전역 변수로 박아두는 것
    # 즉, 별도 설정 없으면 dspy 프레임워크는 자동으로 아래 lm 객체를 모델로 사용하게 된다. 
    dspy.configure(lm=lm)
    
    return lm

def get_configured_llm():
    """이미 configure된 Tester LLM을 가져오기"""
    return dspy.settings.lm

def get_optimizer_llm():
    """
    Optimizer 전용 LLM 인스턴스 생성 및 반환
    (Agent에서 프롬프트 최적화 시 사용)
    """
    Settings.setup()
    
    optimizer_lm = dspy.LM(
        model=f"azure/{Settings.OPTIMIZER_MODEL}",
        api_key=Settings.API_KEY,
        api_base=Settings.API_BASE,
        api_version=Settings.API_VERSION
    )
    
    return optimizer_lm