"""
@경로: utils/tools/ragas/config.py
@설명: RAGAS 평가 설정 관리 (llm_client 통합)
"""

from dataclasses import dataclass
from typing import List, Optional, Any

@dataclass
class RagasConfig:
    """RAGAS 평가 설정 클래스"""
    
    # 매니저 설정 (이제 llm_client에서 초기화)
    use_llm_client: bool = True  # infrastructure/llm_client.py 사용 여부
    
    # 평가 메트릭 설정
    use_faithfulness: bool = True
    use_answer_relevancy: bool = True
    use_context_precision: bool = False
    use_context_recall: bool = False
    
    # 평가 임계값
    faithfulness_threshold: float = 0.7
    answer_relevancy_threshold: float = 0.7
    
    # 배치 설정
    batch_size: int = 1  # 한 번에 평가할 데이터 개수
    
    # RAGAS 내부 설정
    max_retries: int = 3
    timeout: int = 30
    
    # 취소된 설정 (llm_client에서 처리)
    # llm_model: str = "gpt-3.5-turbo"  
    # embedding_model: str = "text-embedding-ada-002"
    
    @classmethod
    def default(cls) -> "RagasConfig":
        """기본 설정 반환 (llm_client 사용)"""
        return cls(use_llm_client=True)
    
    @classmethod  
    def with_custom_models(cls, **kwargs) -> "RagasConfig":
        """커스텀 모델 설정 (llm_client 비사용)"""
        return cls(use_llm_client=False, **kwargs)