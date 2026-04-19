def get_gsm8k_experiment_context() -> str:
    """
    GSM8K 실험의 컨텍스트 정보를 반환.
    JudgeLLM과 OptimizerLLM이 실험 목표를 이해하도록 돕는다.
    system_prompt의 role_description에 포함시켜 사용.
    """
    return """
[Experiment Context for GSM8K Optimization]
This is a prompt optimization experiment:
- TesterLLM: Solves math problems (unaware of the experiment)
- JudgeLLM: Evaluates TesterLLM's answers and provides feedback
- OptimizerLLM: Improves TesterLLM's system prompt based on feedback

Optimization Goals:
1. Maximize accuracy (get the correct answer)
2. Improve similarity to Gold Answer style

Feedback Priority:
1. Correctness is critical - wrong answers are critical errors
2. Only when correct, improve style similarity to Gold Answer

Guidelines:
- Avoid redundant or unnecessary content in feedback
- Focus on actionable improvements to the system prompt
"""


def get_gsm8k_ex_objective_function(ground_truth: str) -> str:
    return f"""
    You are a critical and rigorous evaluator for RAG systems. 
    Your task is to examine the predicted answer step-by-step and identify potential flaws.

    **Reference Answer:** {ground_truth}

    **Evaluation Criteria:**
    1. Does the prediction fully address the question based on the given context?
    2. Are there any factual inaccuracies or hallucinations?
    3. Is the reasoning clear and logically sound?
    4. What specific improvements would make this answer better?

    Provide concise, actionable feedback focused on how to improve the answer generation prompt.
    """

def get_gsm8k_baseline_objective_function(ground_truth: str) -> str:
    """
    GSM8K 벤치마크용 기본 Objective Function 프롬프트 생성 함수.
    실제 논문에서는 아예 GSM8k 에서는 목적 함수를 사용하지 않는다. 
    
    Args:
        ground_truth (str): 문제에 대한 정답 (Reference Answer)
        
    Returns:
        str: 최적화 프롬프트로 사용할 문자열
    """
    return f"""
    """


def get_gsm8k_improve_objective_function(
        ground_truth: str,
        similarity_score: float | None = None) -> str:
    """
    GSM8K 벤치마크용 개선된 Objective Function 프롬프트 생성 함수.
    Improve 모드에서만 similarity_score를 참고 지표로 전달한다.
    """
    similarity_score_text = "[N/A]"
    if similarity_score is not None:
        similarity_score_text = str(similarity_score)

    return f"""
    <Environment 설명>
        여기는 '프롬프트 최적화 실험' 환경입니다. 
        - 문제를 푸는 TesterLLM
        - TesterLLM이 생성한 응답과 모범답안을 비교하여 어떤 개선이 필요한지 피드백하는 JudgeLLM
        - JudgeLLM이 생성한 피드백을 바탕으로 TesterLLM 이 사용할 프롬프트를 개선하는 OptimizerLLM이 있습니다.
        - JudgeLLM과 OptimizerLLM은 해당 실험을 이해하고 있지만, TesterLLM은 실험에 대한 정보를 갖지 않습니다. 그저 문제를 풀 뿐입니다.

        <지표 설명>
        - Accuracy : 정답 일치 여부, 맞으면 1 틀리면 0. 
        - Similarity Score : TesterLLM이 생성한 답변과 모범 답안의 스타일 유사도 (0.0 ~ 1.0), 임배딩 모델이 계산한 실제 유사도 지표.
        </지표 설명>

        당신은 이 중 JudgeLLM 입니다. 아래의 규칙을 참고하여, 적절한 피드백을 생성하세요.
        
        <규칙>
            프롬프트 최적화 실험 목표
            1. Accuracy 1 (정답 맞추기)
            2. Similarity Score 1 (TesterLLM 의 응답이 모범 답안과 유사한 형태, 응답 스타일을 갖도록 개선)

            피드백 작성 우선 순위
            1. Accuracy > Similarity Score
            --> 즉, (2번) 답안 스타일이 유사하다 해도, (1번) 정답이 틀릴 경우 critical error 입니다.
            --> (1번) 정답이 맞을 경우에는, (2번) 답안 스타일을 유사하게 만드는 방향으로 
                OptimizerLLM 이 상세한 프롬프트를 작성할 수 있도록 피드백을 작성하세요.

            피드백 작성 양식
            - 최종 작성 전, 피드백에 중복되는 내용 혹은 불필요한 내용이 없는지 점검하세요.
        </규칙>

        <실험 데이터>
            모범 답안 (Gold Answer):
            {ground_truth}

            모범 답안과 Forward Engine 답안 유사도:
            {similarity_score_text}
        </실험 데이터>
    </Environment 설명>
    """

    # 참고로, Forward Engine 이 생성한 답변 (prediction) 은 loss 를 통해 자동으로 전달 됨.


