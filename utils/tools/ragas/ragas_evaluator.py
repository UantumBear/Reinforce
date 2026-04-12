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
        
        # 사용할 메트릭 설정 (RAGAS v0.1+ 방식에 맞춤)
        self.metrics = []
        
        # 메트릭에 모델 설정 시도 (가능한 경우에만)
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
                # RAGAS v0.1+에서는 reference 필드를 사용
                data['reference'] = [ground_truth]  # context_precision/recall용
                data['ground_truths'] = [ground_truth]  # 기존 메트릭 호환성용
            
            # Dataset 생성
            dataset = Dataset.from_dict(data)
            
            # RAGAS 평가 실행 (v0.1+ 방식)
            if self.chat_model and self.embedding_model:
                # 커스텀 모델 사용
                print("[RAGAS] Using llm_client models for evaluation")
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
            
            # 결과 파싱 - EvaluationResult 타입 지원
            result = RagasResult()
            result.raw_results = results
            
            # 디버깅: 실제 결과 구조 확인
            print(f"[RAGAS DEBUG] Results type: {type(results)}")
            print(f"[RAGAS DEBUG] Results content: {results}")
            
            # EvaluationResult 타입 처리
            if hasattr(results, '_repr_dict'):
                # EvaluationResult 객체인 경우 _repr_dict 사용
                results_dict = results._repr_dict
                print(f"[RAGAS DEBUG] Using EvaluationResult._repr_dict: {results_dict}")
            elif hasattr(results, 'iloc'):
                # DataFrame인 경우 첫 번째 행 추출
                row_data = results.iloc[0].to_dict() if len(results) > 0 else {}
                print(f"[RAGAS DEBUG] Extracted DataFrame row: {row_data}")
                results_dict = row_data
            elif isinstance(results, dict):
                # 딕셔너리인 경우 그대로 사용
                results_dict = results
                print(f"[RAGAS DEBUG] Using dict result: {results_dict}")
            else:
                # 기타 형태는 빈 딕셔너리
                results_dict = {}
                print(f"[RAGAS DEBUG] Unknown results type, using empty dict")
            
            # 안전한 결과 파싱
            result.faithfulness_score = float(results_dict.get('faithfulness', 0.0))
            result.is_faithful = result.faithfulness_score >= self.config.faithfulness_threshold
            
            result.answer_relevancy_score = float(results_dict.get('answer_relevancy', 0.0))
            result.is_relevant = result.answer_relevancy_score >= self.config.answer_relevancy_threshold
            
            result.context_precision_score = float(results_dict.get('context_precision', 0.0))
            result.context_recall_score = float(results_dict.get('context_recall', 0.0))
            
            print(f"[RAGAS DEBUG] Final parsed scores: faithfulness={result.faithfulness_score:.4f}, answer_relevancy={result.answer_relevancy_score:.4f}")
            
            return result
            
        except Exception as e:
            print(f"[Error] RAGAS evaluation failed: {e}")
            return RagasResult(error_message=str(e))
    
