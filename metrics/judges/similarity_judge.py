"""
@경로: metrics/judges/similarity_judge.py
@설명: 임베딩 기반 의미적 유사도 계산기
- 단순 유사도가 아닌 의미적 유사도인 근거는?
- 
"""

from infrastructure.embed_client import get_embedding_client

class SimilarityJudge:
    """
    [역할]
    실제 임베딩 기반 의미적 유사도 계산기
    LLM 기반 예측 대신 벡터 임베딩을 사용한 정확한 코사인 유사도 계산
    """
    
    def __init__(self, use_azure=False):
        self.embed_client = get_embedding_client(use_azure=use_azure)
    
    def __call__(self, gold_answer, predicted_answer):
        """
        두 답변 간의 의미적 유사도를 계산
        
        @param gold_answer: 정답
        @param predicted_answer: 예측 답변  
        @return: 0.0~1.0 사이의 유사도 점수
        """
        try:
            similarity = self.embed_client.calculate_similarity(gold_answer, predicted_answer)
            # 0~1 범위 보장
            return max(0.0, min(1.0, similarity))
        except Exception as e:
            print(f"[WARNING] 임베딩 유사도 계산 실패: {e}")
            return 0.0