"""
@경로: utils/tools/ragas/ragas_evaluator.py
@설명: RAGAS 기반 RAG 평가 메인 클래스 (llm_client 통합)
"""

import warnings
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from .config import RagasConfig

# RAGAS 관련 경고 억제
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# For RAGAS
try:
    from infrastructure.llm_client import get_ragas_model
    LLM_CLIENT_AVAILABLE = True
except ImportError:
    print("[Warning] llm_client not available")
    LLM_CLIENT_AVAILABLE = False

try:
    from ragas import evaluate
    from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
    from datasets import Dataset
    RAGAS_AVAILABLE = True
except ImportError as e:
    print(f"[Warning] RAGAS not installed: {e}")
    print("Install with: pip install ragas")
    RAGAS_AVAILABLE = False
    
    # Fallback 클래스들 (RAGAS 없을 때 에러 방지용)
    class Dataset:
        @classmethod
        def from_dict(cls, data):
            return None    
    def evaluate(*args, **kwargs):
        return {"faithfulness": 0.0, "answer_relevancy": 0.0}    
    faithfulness = None
    answer_relevancy = None

@dataclass
class RagasResult:
    """RAGAS 평가 결과 클래스"""
    faithfulness_score: float = 0.0
    answer_relevancy_score: float = 0.0
    context_precision_score: float = 0.0
    context_recall_score: float = 0.0
    
    is_faithful: bool = False
    is_relevant: bool = False
    
    raw_results: Dict[str, Any] = None
    error_message: str = ""

class RagasEvaluator:
    """RAGAS 기반 RAG 평가기 """
    
    def __init__(self, config: Optional[RagasConfig] = None):
        self.config = config or RagasConfig.default()
        
        if not RAGAS_AVAILABLE:
            print("[Warning] RagasEvaluator initialized without RAGAS library")
            return
        
        # 모델 초기화
        self.chat_model = None
        self.embedding_model = None
        self._init_models()
        
        # 사용할 메트릭 설정
        self.metrics = []
        if self.config.use_faithfulness and faithfulness:
            self.metrics.append(faithfulness)
        if self.config.use_answer_relevancy and answer_relevancy:
            self.metrics.append(answer_relevancy)
        if self.config.use_context_precision and context_precision:
            self.metrics.append(context_precision)
        if self.config.use_context_recall and context_recall:
            self.metrics.append(context_recall)
            
        print(f"[RAGAS] Initialized with {len(self.metrics)} metrics")
    
    def _init_models(self):
        """모델 초기화 (llm_client 연동)"""
        # 설정에 use_llm_client 옵션이 없으면 기본값 True로 가정하거나 config에 추가 필요
        # 여기서는 LLM_CLIENT_AVAILABLE이면 무조건 시도하는 것으로 작성
        if LLM_CLIENT_AVAILABLE:
            try:
                print("[RAGAS] Fetching models from llm_client...")
                # llm_client.py의 get_ragas_model() 호출
                chat_model, embedding_model = get_ragas_model()
                
                if chat_model and embedding_model:
                    self.chat_model = chat_model
                    self.embedding_model = embedding_model
                    print(f"[RAGAS] Models loaded successfully via llm_client")
                else:
                    print("[RAGAS] llm_client returned None for models. Using RAGAS defaults.")
            except Exception as e:
                print(f"[Warning] Failed to load models from llm_client: {e}")
                print("[RAGAS] Using RAGAS default models (OpenAI default)")
        else:
            print("[RAGAS] llm_client not available. Using RAGAS default models.")
    
    def evaluate_single(
        self,
        question: str,
        answer: str,
        contexts: List[str],
        ground_truth: Optional[str] = None
    ) -> RagasResult:
        """단일 QA 쌍 평가"""
        
        if not RAGAS_AVAILABLE:
            print("[Error] RAGAS not available, returning default result")
            return RagasResult(error_message="RAGAS not installed")
        
        try:
            # RAGAS 데이터 형식으로 변환
            data = {
                'question': [question],
                'answer': [answer],
                'contexts': [contexts],  # contexts는 List[str] 형태여야 함
            }
            
            if ground_truth:
                data['ground_truths'] = [ground_truth] # rAGAS 에서 사용하는 필드명
            
            # Dataset 생성
            dataset = Dataset.from_dict(data)
            
            # 모델 사용 여부에 따라 평가 방식 달리 사용
            if self.chat_model and self.embedding_model:
                # llm_client 모델 사용
                print("[RAGAS] Using llm_client models for evaluation")
                # RAGAS에 모델 전달 (방법은 RAGAS 버전에 따라 다름)
                results = evaluate(
                    dataset, 
                    metrics=self.metrics,
                    llm=self.chat_model,
                    embeddings=self.embedding_model
                )
            else:
                # 기본 RAGAS 모델 사용
                print("[RAGAS] Using default models for evaluation") 
                results = evaluate(dataset, metrics=self.metrics)
            
            # 결과 파싱
            result = RagasResult()
            result.raw_results = results
            
            if 'faithfulness' in results:
                result.faithfulness_score = float(results['faithfulness'])
                result.is_faithful = result.faithfulness_score >= self.config.faithfulness_threshold
            
            if 'answer_relevancy' in results:
                result.answer_relevancy_score = float(results['answer_relevancy']) 
                result.is_relevant = result.answer_relevancy_score >= self.config.answer_relevancy_threshold
            
            if 'context_precision' in results:
                result.context_precision_score = float(results['context_precision'])
                
            if 'context_recall' in results:
                result.context_recall_score = float(results['context_recall'])
            
            return result
            
        except Exception as e:
            print(f"[Error] RAGAS evaluation failed: {e}")
            return RagasResult(error_message=str(e))
    
