"""
@경로: infrastructure/llm_client.py
@설명: DSPy/RAGAS 등 에서 사용할 LLM 및 임베딩 모델 초기화 및 환경설정
"""

import dspy
import sys
from conf.config import Settings

from infrastructure.embed_client import get_embedding_client

def setup_lms(verbose=True):
    """
    테스터 LLM 환경을 설정하고 전역 configure를 수행
    (또한 RAGAS 모델도 초기화)
    
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
        
        # # RAGAS 설정 상태 표시
        # if LANGCHAIN_AVAILABLE:
        #     print("[CHECK] RAGAS 지원 준비 완료")
        #     print(f"   - RAGAS Chat Model: {Settings.TESTER_MODEL}")
        #     print(f"   - RAGAS Embedding: text-embedding-ada-002")
        #     print()
        # else:
        #     print("[WARNING] RAGAS 지원 불가 - langchain-openai 설치 필요")
        #     print()
    
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


def get_ragas_model():
    """
    RAGAS 평가용 모델 반환
    """
    try:
        # 1. Chat Model (평가자)
        # RAGAS 평가자는 보통 성능 좋은 GPT-4급을 권장하지만, 설정에 따름
        from langchain_openai import AzureChatOpenAI
        
        chat_model = AzureChatOpenAI(
            azure_deployment=Settings.TESTER_MODEL,
            api_version=Settings.API_VERSION,
            azure_endpoint=Settings.API_BASE,
            api_key=Settings.API_KEY,
            temperature=0.0
        )
        
        # 2. Embedding Model (유사도 계산용)
        # [핵심 변경] embed_client에게 LangChain 객체 생성을 위임!
        # config.py의 USE_AZURE_EMBEDDING 설정에 따라 알아서 가져옴
        use_azure = getattr(Settings, "USE_AZURE_EMBEDDING", True) # 설정이 없으면 기본 True
        
        # 여기서 embed_client의 싱글톤 인스턴스를 가져오고 -> LangChain 객체를 달라고 함
        embed_client = get_embedding_client(use_azure=use_azure)
        embedding_model = embed_client.get_langchain_instance()
        
        return chat_model, embedding_model
        
    except Exception as e:
        print(f"[Error] RAGAS 모델 초기화 실패: {e}")
        return None, None