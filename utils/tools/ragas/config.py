"""
@경로: utils/tools/ragas/config.py
@설명: RAGAS 평가 설정 관리 (llm_client 통합)
"""

from dataclasses import dataclass
@dataclass
class RagasConfig:
    """RAGAS 평가 설정 클래스"""
    
    # [평가 메트릭 활성화 여부]
    use_faithfulness: bool = True
    use_answer_relevancy: bool = True
    use_context_precision: bool = False
    use_context_recall: bool = False
    
    # [Pass/Fail 판정 임계값]
    faithfulness_threshold: float = 0.7
    answer_relevancy_threshold: float = 0.7
    
    @classmethod
    def default(cls) -> "RagasConfig":
        """기본 설정 반환"""
        return cls()