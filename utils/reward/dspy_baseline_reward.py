"""
## 현재 실험에서 쓰지 않는 함수,
## 기존 dspy 잔재 ... reinforce -> dspy -> Textgrad

@경로: utils/reward/dspy_baseline_reward.py
@설명: [Paper Mode / Baseline] 단순 의미적 유사도(Semantic Similarity) 기반 보상
       - 규제(Constitution), 환각(Faithfulness) 여부는 무시하고
       - 오직 '정답(Golden Answer)'과 의미가 통하는지만 판단 (Scalar Score)
"""

import dspy

# ------------------------------------------------------------------------------
# 1. [Judge Signature] 단순 정답 비교 심판관
# ------------------------------------------------------------------------------

class SemanticSimilarityJudge(dspy.Signature):
    """
    [역할]
    당신은 채점관입니다.
    [Predicted Answer]가 [Golden Answer]의 핵심 의미를 정확히 담고 있는지 판단하세요.
    
    주의사항:
    2. '팩트(Fact)'가 일치하는지 봅니다.
    3. 정답의 핵심 키워드나 숫자가 포함되어 있다면 관대하게 점수를 주세요.
    
    출력 형식:
    - 0.0 에서 1.0 사이의 실수 (Float)
    - 1.0: 완벽히 같은 의미
    - 0.0: 완전히 틀린 내용
    """
    gold_answer = dspy.InputField(desc="The correct answer (Ground Truth)")
    predicted_answer = dspy.InputField(desc="The model's generated answer")
    score = dspy.OutputField(desc="Float score between 0.0 and 1.0")


# ------------------------------------------------------------------------------
# 2. [Metric Function] DSPy에 꽂아서 쓸 단순 스칼라 함수
# ------------------------------------------------------------------------------

def baseline_similarity_metric(gold, pred, trace=None):
    """
    [Paper Mode용 스칼라 메트릭]
    LLM을 통해 정답 유사도를 0.0~1.0 사이로 반환합니다.
    """
    
    # 1. 데이터 추출
    gold_answer = gold.answer
    pred_answer = pred.answer

    # 2. LLM Judge 호출 (단순 비교)
    judge = dspy.Predict(SemanticSimilarityJudge)
    
    try:
        response = judge(gold_answer=gold_answer, predicted_answer=pred_answer)
        
        # 문자열을 float으로 변환
        score_str = response.score
        
        # 가끔 LLM이 "Score: 0.8" 처럼 줄 수 있으므로 숫자만 파싱하는 안전장치
        import re
        match = re.search(r"0\.\d+|1\.0|0|1", str(score_str))
        if match:
            final_score = float(match.group())
        else:
            # 파싱 실패 시, 내용이 비슷하면 0.5라도 줌 (최적화 멈춤 방지)
            final_score = 0.0
            
    except Exception as e:
        print(f"[Baseline Metric Error] {e}")
        final_score = 0.0

    # 3. 결과 반환 (로그 출력 없이 조용히 점수만 리턴)
    # Baseline은 보통 내부 로직을 보여주지 않고 점수만 뱉는 것이 특징입니다.
    return final_score