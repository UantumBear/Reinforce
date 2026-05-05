def get_improve_objective_function(
        ground_truth: str,
    similarity_score: float | None = None,
    accuracy_score: float | None = None,
    student_raw_trajectory: str | None = None,
    previous_rejection_context: str = "") -> str:
    """
    GSM8K 벤치마크용 개선된 Objective Function 프롬프트 생성 함수.
    Improve 모드에서만 similarity_score를 참고 지표로 전달한다.
    
    @Param:
        ground_truth: 모범 답안
        similarity_score: TesterLLM 답변과 모범 답안의 유사도 (0.0 ~ 1.0)
        accuracy_score: 정답 일치 여부 (0 또는 1)
        student_raw_trajectory: TesterLLM이 생성한 전체 사고 과정 (Chain of Thought 텍스트)
    """
    similarity_score_text = "[N/A]"
    if similarity_score is not None:
        similarity_score_text = str(similarity_score)

    accuracy_text = "[N/A]"
    if accuracy_score is not None:
        accuracy_text = "1" if float(accuracy_score) >= 1.0 else "0"
    
    # ##### 차별점 #####
    # [Improve] TesterLLM의 전체 Chain of Thought 텍스트를 Judge 입력으로 제공
    # - student_raw_trajectory가 None이면 "[미제공]" 표시
    # - 제공되면 전체 사고 과정을 <실험 데이터>에 포함
    student_trajectory_text = "[미제공]"
    if student_raw_trajectory is not None:
        student_trajectory_text = student_raw_trajectory
    ###################

    return f"""
    <Role>
        당신은 Environment(프롬프트 최적화 실험)에서 JudgeLLM 역할을 수행합니다.

        당신의 목적은 TesterLLM의 답변을 평가하고,
        OptimizerLLM이 다음 iteration의 TesterLLM 시스템 프롬프트를 개선할 수 있도록
        구조화된 언어 피드백을 생성하는 것입니다.

        단, TesterLLM은 실제 추론 시 문제만 입력받습니다.
        TesterLLM은 실험 정보, 데이터셋 정보, Gold Answer, 평가 점수, JudgeLLM 피드백을 알 수 없습니다.

        **아래의 정보를 참고하여
        <output_format> 내에 존재하는 각 판단 영역별 <review> 태그 내에 적절한 피드백을 생성하세요.
        <output_format> 내에 존재하는 <review> 태그 외의 텍스트는 절대 변경하지 마세요. 실제 실험 데이터이기 때문입니다.**

    </Role>

    <Environment 설명>
        여기는 '프롬프트 최적화 실험' 환경입니다.
        - 실험 단계 및 주요 LLM 역할을 소개합니다.

        <실험 단계>
            - 1. '테스트 평가 단계'
            테스트 평가 단계에서는 TesterLLM이 초기 기본 시스템 프롬프트를 가지고 문제에 대한 답변을 생성합니다.
            해당 답변을 정답과 비교하여 성능을 평가합니다. 

            - 2. 'Train 단계'
            Train 단계에서는 TesterLLM 이 현 시스템 프롬프트를 가지고 문제에 대한 답변을 생성합니다.
            Train 단계에서 TesterLLM은 응답을 
              <CoT>해당 응답을 생성한 생각의 과정 작성</CoT>
              <Response>실제 답변 작성</Response> 과정으로 작성하여, 
              JudegeLLM이 답변의 전체 사고 과정을 알 수 있도록 함.
              (단, TetserLLM 은 실험에 대해 알지 못하며, 단지 응답의 형태만 지정된 양식으로 생성할 뿐임.)

            JudgeLLM 에게 정보를 제공할 다양한 JudegeTools 가 사용됨.
            예를 들면 embedding model 은 TesterLLM이 생성한 답변의 <Response> 태그 내 텍스트와 모범답안의 유사도를 평가.

            JudgeLLM 이 TesterLLM의 답변을 평가하고 피드백을 작성합니다.
            OptimierLLM이 JudgeLLM의 피드백을 바탕으로 TesterLLM이 다음 iteration에서 사용할 시스템 프롬프트를 작성합니다.
        
            - 3.'Validation 단계'
            작성된 프롬프트와 기존 프롬프트의 성능을 비교 평가합니다.
            더 나은 프롬프트가 생성되면 해당 프롬프트를 채택합니다.
            오차 범위 내에서 성능이 떨어진 경우에는 JudgeLLM 이 왜 성능이 떨어졌는지 분석한 추가 피드백을 작성합니다. 

            - 4. 'Iteration' 
            위의 2. Train 단계와 3. Validation 단계를 반복하여 프롬프트를 개선해 나갑니다.

            - 5. '최종 테스트 평가 단계'
            최종적으로 개선된 프롬프트를 가지고 TesterLLM이 문제에 대한 답변을 생성합니다.
            Train 단계에서는 <CoT>와 <Response> 태그 양식으로 답변을 생성했지만,
            최종 테스트 평가 단계에서는 TesterLLM 은 오로지 OptimizerLLM 이 생성한 프롬프트를 이용해 답변을 생성합니다.
            최종 프롬프트 외의 어떠한 제약도 없습니다.
            실제 상용 서비스에서 Service LLM이 최종 프롬프트를 가지고 사용자 질문에 답변을 생성하는 상황과 유사합니다.

        </실험 단계>
        <실험 내 LLM 역할>
            - TesterLLM : 문제에 대한 답변을 생성하는 역할, 실험에 대한 어떠한 정보도 가져서는 안됨.
            - JudgeLLM : TesterLLM이 생성한 응답과 모범답안을 비교하여 어떤 개선이 필요한지 피드백하는 역할.
            - OptimizerLLM : JudgeLLM이 생성한 피드백을 바탕으로 TesterLLM이 사용할 프롬프트를 개선하는 역할.
        
            JudgeLLM과 OptimizerLLM은 본 실험 환경을 명확히 이해하고, 본인에게 주어진 역할 만을 이행해야 함.
        </실험 내 LLM 역할>

        <지표 설명>
            - accuracy : 정답 일치 여부, 맞으면 1 틀리면 0.
            - similarity_score : TesterLLM이 생성한 답변과 모범 답안의 스타일 유사도 (0.0 ~ 1.0), 임배딩 모델이 계산한 실제 유사도 지표.
        </지표 설명>

        <실험 목표>
            1. accuracy 1 (정답 맞추기)
            2. similarity_score 1 (TesterLLM 의 응답이 모범 답안과 유사한 형태, 응답 스타일을 갖도록 개선)
        </실험 목표>
        
        <review_작성규칙>
            

            [우선 순위]
            accuracy > similarity_score
            즉, 답안 스타일이 유사하다 해도, 정답이 틀릴 경우 최적화에 실패한 프롬프트입니다.]

            [피드백 작성 방향 가이드라인]
            특정 문제의 숫자, 특정 Gold Answer, 특정 데이터셋 샘플에 과적합된 지침을 작성하지 마세요.
            죽, 피드백은 가능한 한 일반적인 문제 해결 전략으로 변환하세요.
            만일 개선을 위해 데이터를 이용하고자 한다면, Few-shot 예시 형태로 사용하도록 접근하세요.

            [피드백 작성 양식]
            - 최종 작성 전, 피드백에 중복되는 내용 혹은 불필요한 내용이 없는지 점검하세요.
            - 피드백은 반드시 <output_format> 섹션에 명시된 html 태그 구조로 작성하세요.
            - <output_format> 태그 설명
              meta_for_optimizer
                → OptimizerLLM이 TesterLLM 에게 전달할 프롬프트를 작성하는 과정에서,
                  Environment(실험 정보) 를 작성하지 않도록 금지하는 제약 정보 작성. (그 외 개선을 위한 정보 작성 금지)

              accuracy_analysis
                → 정답률 관점의 분석

              style_format_analysis
                → 형식/스타일 관점의 분석

              total_analysis
                → 실제 TesterLLM 프롬프트에 반영 가능한 일반 지침을 포함한, 종합 판단
            - <output_format> 내 <review> 태그 내의 내용을 작성할 때, 해당 내용 또한 구조화된 태그 형태로 작성하세요.
              예를 들면, 데이터를 이용한 개선에서 <few_shot_example> 태그를 review 태그 내에 활용하는 형태는 허용됩니다.
        </review_작성규칙>

        <실험 데이터>
            모범 답안 (Gold Answer):
            {ground_truth}

            정답 일치 여부 (Accuracy):
            {accuracy_text}
            위 값은, <output_format> 섹션의 <accuracy> 태그에 들어갈 0 또는 1 값입니다.

            모범 답안과 Forward Engine 답안 유사도:
            {similarity_score_text}
            위 값은, <output_format> 섹션의 <similarity_score> 태그에 들어갈 값입니다.

            TesterLLM의 전체 풀이 과정 (Chain of Thought):
            {student_trajectory_text}
            
            {f'''[Iteration 거절 컨텍스트]
            이전 iteration에서 후보 프롬프트가 거절되었습니다.
            두 프롬프트를 비교하여 무엇이 문제인지 분석하고,
            현재 iteration의 비평에서 해당 문제점을 고려하세요.

            {previous_rejection_context}''' if previous_rejection_context else ''}
            
        </실험 데이터>

        <output_format>
            <meta_for_optimizer>
                OptimizerLLM은 아래의 제약사항을 따릅니다.
                <environment_exposure_ban>
                    <review> [이곳에 OptimizerLLM 이 지켜야할 절대 규칙을 작성하세요.] </review>
                </environment_exposure_ban>
            </meta_for_optimizer>
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

        </output_format>
    </Environment 설명>
    """

    # 참고로, Forward Engine 이 생성한 답변 (prediction) 은 loss 를 통해 자동으로 전달 됨.


