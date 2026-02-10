"""
Docstring for models.rl_optimization_log

@경로: models/rl_optimization_log.py
@설명:
- 강화학습 최적화 과정에서 생성되는 로그를 저장하기 위한 데이터베이스 모델
"""


from sqlalchemy import Column, Integer, String, Float, Text, Boolean, DateTime
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
    optimizer_model_nm = Column(String(100), nullable=True)  # 최적화 담당 모델명 (예: gpt-5-nano)
    optimizer_model_provider = Column(String(50), nullable=True)  # 최적화 모델 제공사 (azure)
    tester_model_nm = Column(String(100), nullable=True)  # 테스팅 담당 모델명 (예: gpt-5-mini)
    tester_model_provider = Column(String(50), nullable=True)  # 테스팅 모델 제공사 (azure)
    
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
    
    # RAGAS 종합 평가 결과 (컬럼 생성하지 않았음)
    # ragas_is_faithful = Column(Boolean, nullable=True)
    # ragas_is_relevant = Column(Boolean, nullable=True)
    
    critical_review = Column(Text, nullable=True)
    full_analysis = Column(Text, nullable=True)
    
    is_success = Column(Boolean, default=True)
    error_log = Column(Text, nullable=True)
    
    created_at = Column(DateTime, default=datetime.now)

    def __repr__(self):
        return f"<Log(exp={self.experiment_id}, ep={self.episode}, score={self.total_score})>"