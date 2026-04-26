def get_improve_objective_function(
        ground_truth: str,
    similarity_score: float | None = None,
    accuracy_score: float | None = None) -> str:
    """
    GSM8K 벤치마크용 개선된 Objective Function 프롬프트 생성 함수.
    Improve 모드에서만 similarity_score를 참고 지표로 전달한다.
    """
    similarity_score_text = "[N/A]"
    if similarity_score is not None:
        similarity_score_text = str(similarity_score)

    accuracy_text = "[N/A]"
    if accuracy_score is not None:
        accuracy_text = "1" if float(accuracy_score) >= 1.0 else "0"

    return f"""
    <Environment 설명>
        여기는 '프롬프트 최적화 실험' 환경입니다. 
        - 문제를 푸는 TesterLLM
        - TesterLLM이 생성한 응답과 모범답안을 비교하여 어떤 개선이 필요한지 피드백하는 JudgeLLM
        - JudgeLLM이 생성한 피드백을 바탕으로 TesterLLM 이 사용할 프롬프트를 개선하는 OptimizerLLM이 있습니다.
        - JudgeLLM과 OptimizerLLM은 해당 실험을 이해하고 있지만, TesterLLM은 실험에 대한 정보를 갖지 않습니다. 그저 문제를 풀 뿐입니다.

        <지표 설명>
        - accuracy : 정답 일치 여부, 맞으면 1 틀리면 0.
        - similarity_score : TesterLLM이 생성한 답변과 모범 답안의 스타일 유사도 (0.0 ~ 1.0), 임배딩 모델이 계산한 실제 유사도 지표.
        </지표 설명>

        당신은 이 중 JudgeLLM 입니다. 
        아래의 규칙을 참고하여, <출력형식> 내에 존재하는 각 판단 영역별 <review> 태그 내에 적절한 피드백을 생성하세요.
        
        <규칙>
            [프롬프트 최적화 실험 목표]
            1. accuracy 1 (정답 맞추기)
            2. similarity_score 1 (TesterLLM 의 응답이 모범 답안과 유사한 형태, 응답 스타일을 갖도록 개선)

            [우선 순위]
            accuracy > similarity_score
            즉, 답안 스타일이 유사하다 해도, 정답이 틀릴 경우 최적화에 실패한 프롬프트입니다.

            [피드백 작성 양식]
            - 최종 작성 전, 피드백에 중복되는 내용 혹은 불필요한 내용이 없는지 점검하세요.
            - 피드백은 반드시 <출력형식> 섹션에 명시된 html 태그 구조로 작성하세요.
        </규칙>

        <실험 데이터>
            모범 답안 (Gold Answer):
            {ground_truth}

            정답 일치 여부 (Accuracy):
            {accuracy_text}
            위 값은, <출력형식> 섹션의 <accuracy> 태그에 들어갈 0 또는 1 값입니다.

            모범 답안과 Forward Engine 답안 유사도:
            {similarity_score_text}
            위 값은, <출력형식> 섹션의 <similarity_score> 태그에 들어갈 값입니다.
        </실험 데이터>

        <출력형식>
            <accuracy_analysis>
                <accuracy> {accuracy_text} </accuracy>
                <review> [이곳에 어떻게 하면 정답을 향상시킬 수 있을지 제안하는 피드백을 작성하세요] </review>
            </accuracy_analysis>
            <style_format_analysis>
                <similarity_score> {similarity_score_text} </similarity_score>
                <review> [이곳에 어떻게 하면 모범답안과 유사한 답변을 생성시킬 수 있을지 제안하는 피드백을 작성하세요] </review>
            </style_format_analysis>

            <total_analysis>
                <review> [이곳에 총 평을 작성하세요.] </review>
            </total_analysis>

        </출력형식>
    </Environment 설명>
    """

    # 참고로, Forward Engine 이 생성한 답변 (prediction) 은 loss 를 통해 자동으로 전달 됨.


