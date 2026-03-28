"""
@경로: schemas/rl_optimization_log.py
@설명: RL 최적화 로그 생성을 위한 Pydantic Schema
- 타입 안전성과 IDE 자동완성 지원
- 팩토리 메서드를 통한 로그 생성 패턴 제공

- 2026.03.20 초기 작성, 현재 미사용 중
"""

import math
from datetime import datetime
from typing import Optional
from pydantic import BaseModel, Field


class RlOptimizationLogSchema(BaseModel):
    """RL 최적화 로그 스키마"""
    
    # 기본 메타데이터
    experiment_id: str
    episode: int
    
    # 프롬프트 및 입출력
    instruction: Optional[str] = None
    question: Optional[str] = None
    context: Optional[str] = None
    model_answer: Optional[str] = None
    gold_answer: Optional[str] = None
    
    # 점수
    total_score: Optional[float] = None
    raw_similarity: Optional[float] = None
    
    # 실험 메타데이터
    dataset_size: Optional[int] = None
    avg_total_score: Optional[float] = None
    
    # 모델 정보
    optimizer_model_nm: str
    optimizer_model_provider: Optional[str] = "azure"
    tester_model_nm: str
    tester_model_provider: Optional[str] = "azure"
    
    # RAGAS 평가자 모델 정보
    ragas_chat_model_nm: Optional[str] = None
    ragas_chat_model_provider: Optional[str] = None
    embedding_model_nm: Optional[str] = None
    
    # 평가 결과
    is_faithful: Optional[str] = None
    is_style_match: Optional[str] = None
    
    # 헌법(Constitution) 위반 여부
    constitution_status: Optional[str] = "Pass"
    constitution_violation_reason: Optional[str] = None
    
    # RAGAS 점수들
    ragas_faithfulness_score: Optional[float] = None
    ragas_answer_relevancy_score: Optional[float] = None
    ragas_context_precision_score: Optional[float] = None
    ragas_context_recall_score: Optional[float] = None
    
    # Accuracy 점수 (정확도, 0.0 ~ 1.0 범위)
    accuracy: Optional[float] = None
    
    # 분석 및 피드백
    critical_review: Optional[str] = None  # 프롬프트 최적화 관점 피드백
    full_analysis: Optional[str] = None    # 샘플 답안 관점 비판
    
    # Optimizer LLM 관련
    optimizer_system_prompt: Optional[str] = None
    optimizer_total_input: Optional[str] = None
    
    # 실행 상태
    is_success: bool = True
    error_log: Optional[str] = None
    
    created_at: datetime = Field(default_factory=datetime.now)
    
    class Config:
        """Pydantic 설정"""
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }
    
    # ============================================================
    # 팩토리 메서드들
    # ============================================================
    
   