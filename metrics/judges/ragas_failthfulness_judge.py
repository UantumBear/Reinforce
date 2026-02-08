
"""
@경로: metrics/judges/ragas_failthfulness_judge.py
@설명: RAGAS 기반 신뢰도 평가기 래퍼
- 이 답변이 주어진 문서 밖의 말을 하지 않았는가? 를 평가하기 위한 평가기 이다.
- 논문의 주제가, 도메인 헌법이 중요한 RAG 시스템에서의 프롬프트 최적화이므로,
  단순히 LLM에게 (dspy.pred 기반) "이 답변이 맥락에 충실한가?" 라고 묻는 방식으로는 신뢰할 수 없어서
  RAGAS 기반의 평가기를 별도로 구현하여 사용한다. 

- 문서 안에서 답안을 찾았는 지가 중요하다. (할루시네이션 방지 용도)
"""

from typing import Optional, List, Union
from utils.tools.ragas.ragas_evaluator import RagasEvaluator, RagasResult
from utils.tools.ragas.config import RagasConfig

class RagasFaithfulnessJudge:
    """RAGAS Faithfulness를 사용한 신뢰도 평가기"""
    
    def __init__(self, config: Optional[RagasConfig] = None):
        self.evaluator = RagasEvaluator(config)
        self.config = config or RagasConfig.default()
        
    def evaluate(
        self, 
        question: str,
        answer: str, 
        context: Union[str, List[str]],
        gold_answer: Optional[str] = None
    ) -> dict:
        """
        Faithfulness 평가 수행
        
        Args:
            question: 질문
            answer: 모델이 생성한 답변
            context: 참조 맥락 (문자열 또는 문자열 리스트)
            gold_answer: 정답 (선택사항)
            
        Returns:
            dict: 평가 결과 {'is_faithful': bool, 'score': float, 'reason': str}
        """
        
        # context를 List[str] 형태로 변환
        if isinstance(context, str):
            contexts = [context] if context.strip() else []
        else:
            contexts = context if context else []
            
        # RAGAS 평가 실행
        result: RagasResult = self.evaluator.evaluate_single(
            question=question,
            answer=answer, 
            contexts=contexts,
            ground_truth=gold_answer
        )
        
        # 결과 변환
        if result.error_message:
            return {
                'is_faithful': False,
                'score': 0.0,
                'reason': f"Evaluation error: {result.error_message}",
                'raw_score': None
            }
        
        # 신뢰도 판단 로직
        is_faithful = result.is_faithful
        score = result.faithfulness_score
        
        # 이유 생성
        if is_faithful:
            reason = f"답변이 제공된 맥락에 충실함 (faithfulness: {score:.3f})"
        else:
            reason = f"답변이 맥락에 충실하지 않음 (faithfulness: {score:.3f}, 임계값: {self.config.faithfulness_threshold})"
            
        # 추가 정보가 있다면 포함
        if result.answer_relevancy_score > 0:
            reason += f", 질문 관련성: {result.answer_relevancy_score:.3f}"
        
        return {
            'is_faithful': is_faithful,
            'score': score,
            'reason': reason,
            'raw_score': result.faithfulness_score,
            'relevancy_score': result.answer_relevancy_score,
            'ragas_result': result
        }
    
    def __call__(self, question: str, answer: str, context: Union[str, List[str]], gold_answer: Optional[str] = None) -> dict:
        """호출 가능 인터페이스"""
        return self.evaluate(question, answer, context, gold_answer)