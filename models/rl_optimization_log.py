"""
Docstring for models.rl_optimization_log

@경로: models/rl_optimization_log.py
@설명:
- 강화학습 최적화 과정에서 생성되는 로그를 저장하기 위한 데이터베이스 모델
"""


from sqlalchemy import Column, Integer, String, Float, Text, Boolean, DateTime
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import declarative_base
from datetime import datetime

Base = declarative_base()

class RlOptimizationLog(Base):
    __tablename__ = 'rl_optimization_logs'

    id = Column(Integer, primary_key=True, autoincrement=True)
    experiment_id = Column(String(50), nullable=False, index=True) # 실험 그룹 ID
    episode = Column(Integer, nullable=False)
    
    instruction = Column(Text, nullable=True)
    question = Column(Text, nullable=True)
    context = Column(Text, nullable=True)
    
    model_answer = Column(Text, nullable=True)
    gold_answer = Column(Text, nullable=True)
    
    total_score = Column(Float, nullable=True)
    raw_similarity = Column(Float, nullable=True)
    
    # 실험 메타데이터
    dataset_size = Column(Integer, nullable=True)  # 실험에 사용된 데이터 개수
    avg_total_score = Column(Float, nullable=True)  # 해당 에피소드의 평균 점수
    dataset_nm = Column(String(100), nullable=True)  # 데이터셋 이름 (예: openai/gsm8k, nasa/cmapss-fd001)

    optimizer_model_nm = Column(String(100), nullable=True)  # 최적화 담당 모델명 (예: gpt-5-nano)
    optimizer_model_provider = Column(String(50), nullable=True)  # 최적화 모델 제공사 (azure)

    tester_model_nm = Column(String(100), nullable=True)  # 테스팅 담당 모델명 (예: gpt-5-mini)
    tester_model_provider = Column(String(50), nullable=True)  # 테스팅 모델 제공사 (azure)

    # RAGAS 평가자 모델 정보 추가
    ragas_chat_model_nm = Column(String(100), nullable=True)  # RAGAS 평가자 모델명
    ragas_chat_model_provider = Column(String(50), nullable=True)  # RAGAS 평가자 모델 제공사

    # 임베딩 모델 정보 추가 (RAGAS 평가 시 사용)
    embedding_model_nm = Column(String(100), nullable=True)  # 임베딩 모델
    
    # "Pass", "Fail" 문자열 저장을 위해 String 사용
    is_faithful = Column(String(20), nullable=True)
    is_style_match = Column(String(20), nullable=True)
    
    # 헌법(Constitution) 위반 여부
    constitution_status = Column(String(20), nullable=True, default='Pass')
    constitution_violation_reason = Column(Text, nullable=True)
    
    # RAGAS 평가 점수들 (0.0 ~ 1.0 범위)
    ragas_faithfulness_score = Column(Float, nullable=True)
    ragas_answer_relevancy_score = Column(Float, nullable=True) 
    ragas_context_precision_score = Column(Float, nullable=True)
    ragas_context_recall_score = Column(Float, nullable=True)
    
    # Accuracy 점수 (정확도, 0.0 ~ 1.0 범위)
    accuracy = Column(Float, nullable=True) # Train 샘플의 개별 정답 여부 (0 or 1)
    
    # Validation 관련 컬럼
    validation_info = Column(JSONB, nullable=True)  # Validation 샘플들의 상세 정보 (JSON 형태)
    validation_accuracy = Column(Float, nullable=True)  # Validation 데이터셋 전체의 평균 accuracy
    validation_dataset_size = Column(Integer, nullable=True)  # Validation 데이터셋 샘플 개수
    
    # LLM 호출 카운트 (각 row 생성 시 실제 LLM 호출 횟수 추적)
    forward_tester_llm_call_cnt = Column(Integer, default=0)  # Forward Model(답변 생성) LLM 호출 횟수
    backward_judge_llm_call_cnt = Column(Integer, default=0)  # Backward Judge(평가) LLM 호출 횟수
    backward_optimizer_llm_call_cnt = Column(Integer, default=0)  # Backward Optimizer(프롬프트 개선) LLM 호출 횟수
    
    # RAGAS 종합 평가 결과 (컬럼 생성하지 않았음)
    # ragas_is_faithful = Column(Boolean, nullable=True)
    # ragas_is_relevant = Column(Boolean, nullable=True)
    
    critical_review = Column(Text, nullable=True)
    full_analysis = Column(Text, nullable=True)
    
    # OptimizerLLM 관련 정보
    optimizer_system_prompt = Column(Text, nullable=True)  # OptimizerLLM이 사용한 시스템 프롬프트
    optimizer_total_input = Column(Text, nullable=True)    # OptimizerLLM에게 전달된 전체 입력 내용
    
    is_success = Column(Boolean, default=True)
    error_log = Column(Text, nullable=True)
    
    created_at = Column(DateTime, default=datetime.now)

    def __repr__(self):
        return f"<Log(exp={self.experiment_id}, ep={self.episode}, score={self.total_score})>"