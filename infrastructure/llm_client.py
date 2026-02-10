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
    embed_client가 설정에 따라 알아서 선택한 모델을 사용
    """
    try:
        # 1. o4-mini 호환 커스텀 RAGAS 래퍼 클래스
        from ragas.llms.base import LangchainLLMWrapper
        from langchain_openai import AzureChatOpenAI
        
        class O4MiniCompatibleRagasWrapper(LangchainLLMWrapper):
            """o4-mini 호환 RAGAS 래퍼"""
            
            def get_temperature(self, n: int) -> float:
                """o4-mini 호환 temperature 반환 (항상 1.0)"""
                return 1.0  # o4-mini는 temperature=1만 지원
        
        # 기본 AzureChatOpenAI 모델 생성
        base_chat_model = AzureChatOpenAI(
            openai_api_version=Settings.API_VERSION,
            azure_endpoint=Settings.API_BASE,
            azure_deployment=Settings.RAGAS_CHAT_MODEL,
            model=Settings.RAGAS_CHAT_MODEL,
            validate_base_url=False,
            max_retries=0,
            temperature=1.0,  # o4-mini 지원값으로 명시적 설정
        )
        
        # o4-mini 호환 RAGAS 래퍼로 감싸기
        chat_model = O4MiniCompatibleRagasWrapper(
            langchain_llm=base_chat_model,
            run_config=None  # 기본 RunConfig 사용
        )
        
        # 2. Embedding Model (유사도 계산용)
        embed_client = get_embedding_client()  # 기본값으로 알아서 결정
        embedding_model = embed_client.get_langchain_instance()
        
        print(f"[SUCCESS] RAGAS 모델 초기화 완료")
        print(f"   - Chat Model: {Settings.RAGAS_CHAT_MODEL} (o4-mini 호환 래퍼)")
        print(f"   - Embedding Model: {type(embedding_model)}")
        
        return chat_model, embedding_model
        
    except Exception as e:
        print(f"[Error] RAGAS 모델 초기화 실패: {e}")
        return None, None