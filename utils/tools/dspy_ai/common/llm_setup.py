"""
DSPy 공통 LLM 설정 모듈

@경로: utils/tools/dspy_ai/common/llm_setup.py
@설명: DSPy LLM 초기화 및 환경설정 공통 함수 제공
"""

import dspy
from conf.config import Env


def setup_dspy_llm(verbose=True):
    """
    DSPy LLM 환경을 설정하고 전역 configure를 수행
    
    Args:
        verbose (bool): 설정 정보 출력 여부
    
    Returns:
        dspy.LM: 설정된 LLM 객체
    
    Raises:
        SystemExit: Azure 환경변수가 올바르게 설정되지 않은 경우
    """
    # 환경변수 로드
    Env.setup_environment()
    
    # Azure 설정 확인
    if not Env.check_azure_api_key():
        print("[FAIL] Azure 환경변수가 제대로 설정되지 않았습니다. 프로그램을 종료합니다.")
        exit(1)
    
    if verbose:
        print("[CHECK] Azure 설정 확인:")
        print(f"   - API Key: {'설정됨' if Env.AZURE_OPENAI_API_KEY else '없음'}")
        print(f"   - Endpoint: {Env.AZURE_OPENAI_ENDPOINT}")  
        print(f"   - Deployment: {Env.AZURE_GPT4DOT1_DEPLOYMENT}")
        print()
    
    # LLM 객체 생성
    lm = dspy.LM(
        model=f"azure/{Env.AZURE_GPT4DOT1_DEPLOYMENT}", # AZURE_GPT4O_MINI_DEPLOYMENT
        api_key=Env.AZURE_OPENAI_API_KEY,
        api_base=Env.AZURE_OPENAI_ENDPOINT,
        api_version=Env.AZURE_OPENAI_API_VERSION
    )
    
    # DSPy 전역 설정
    # 엔트리포인트에서 한 번만 호출하는 게 정석
    # 코드 곳곳에 API 설정 흩뿌리지 말자는 철학
    # 연구나 논문 작업, 혹은 혼자 돌리는 스크립트에서는 한 번에 하나의 LLM만 쓰는 경우가 많기 때문
    # "앞으로 별말 없으면 무조건 이 Azure 모델 써!" 라고 전역 변수로 박아두는 것
    dspy.configure(lm=lm)
    
    if verbose:
        print("[SUCCESS] DSPy LLM 설정 완료!")
        print()
    
    return lm


def get_configured_llm():
    """
    이미 configure된 LLM을 가져오기
    
    Returns:
        dspy.LM: 현재 설정된 LLM 객체
    """
    return dspy.settings.lm