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



    # 참고로, Forward Engine 이 생성한 답변 (prediction) 은 loss 를 통해 자동으로 전달 됨.


