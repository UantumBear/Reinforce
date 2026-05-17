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
        previous_rejection_context: 이전 이터레이션에서 프롬프트가 거절된 이유 (선택 사항)
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
        당신은 TextGrad backward 단계에서 TesterLLM의 출력을 평가하고,
        다음 프롬프트 개선에 사용할 구조화된 피드백을 작성하는 JudgeLLM입니다.

        당신의 피드백은 OptimizerLLM이 TesterLLM의 시스템 프롬프트를 개선하는 데 사용됩니다.
        따라서 피드백은 특정 샘플에 과적합되지 않는 일반화된 지침이어야 합니다.

        당신은 아래에서 설명하는 '<output_format>' 형식으로 피드백을 작성하세요.
        '<output_format>'에 정의된 태그 구조 및 '<review>' 태그 외의 텍스트는 실제 실험 데이터 파싱을 통해 구성되므로, 절대 변경하거나 삭제하지 마세요.
    </Role>

    <Objective>
        우선순위:
            1. Accuracy (정확도): 최종 정답이 Gold Answer의 정답과 일치해야 함 
            2. Similarity (의미적 유사도): '<Response>'의 전체 내용과 형식이 Gold Answer와 유사해야 함

        정답이 틀리면 similarity가 높아도 실패로 간주합니다.
    </Objective>


    <current_iteration_data>
        모범 답안 (Gold Answer):
        {ground_truth}

        TesterLLM의 전체 응답:
        {student_trajectory_text}

        정답 일치 여부 (accuracy):
        {accuracy_text}

        모범 답안과 TesterLLM 응답의 유사도 (similarity_score):
        {similarity_score_text}

        {f'''[이전 Iteration 거절 사유]
        이전 이터레이션에서 생성된 후보 프롬프트가 채택되지 못하고 거절되었습니다.

        {previous_rejection_context}''' if previous_rejection_context else ''}
    </current_iteration_data>

    <output_format>            
        <accuracy_analysis>
            <accuracy> {accuracy_text} </accuracy>
            <review> [이곳에 accuracy == 0 인 경우, `<current_iteration_data>`를 참고하여 오답의 원인을 분석하고 정답률 향상을 위한 논리적 개선 방안을 작성하세요.] </review>
        </accuracy_analysis>
        
        <style_format_analysis>
            <similarity_score> {similarity_score_text} </similarity_score>
            <review> [이곳에 `<current_iteration_data>` 내용을 참고하여, 모범 답안과 유사한 답변 스타일 및 출력 형식을 유도하기 위한 개선 방안을 작성하세요.] </review>
        </style_format_analysis>

        <OptimizerOutputConstraint>
            OptimizerLLM은 JudgeLLM의 피드백, Gold Answer, 평가 점수, 실험 정보를 참고하여
            TesterLLM의 시스템 프롬프트를 개선 할 수 있습니다.
            (Accuracy 향상 1순위, similarity 향상 2순위) 

            [메타 정보 포함 금지 규칙]
            OptimizerLLM이 생성하는 TesterLLM용 시스템 프롬프트에는
            Gold Answer, Reference Output, 모범 답안, 정답 데이터, 평가 점수, Accuracy, Similarity,
            JudgeLLM, OptimizerLLM, Iteration, Validation, 실험, 프롬프트 최적화와 같은
            메타 정보를 직접 포함해서는 안 됩니다. 이 정보들은 일반적인 행동 지침으로 변환되어야 합니다.

            [★ 시스템 프롬프트 XML 태그 구조화 스타일 가이드라인 ★]
            OptimizerLLM이 생성하는 최종 시스템 프롬프트는 가독성과 구조적 명확성을 극대화하기 위해, 
            줄글(평문) 형태를 절대 지양하고 반드시 XML 태그 구조를 갖추어야 합니다.

            ■ 필수 XML 태그 구조 템플릿 및 태그 내 작성 가이드 라인:
            
            <system_persona>
                [여기에 테스터의 도메인 역할 및 정체성을 기술]
            </system_persona>

            <execution_steps>
                [
                    여기에 문제를 해결할 때 순차적으로 밟아야 하는 '행동 프로세스(Positive Instructions)' 지침을 기술,
                    기존 항목과 의미가 중복된다면 새 항목을 추가하지 말고, 필요한 경우 기존 항목을 더 일반적인 지침으로 압축하여 기술
                ]
            </execution_steps>

            <negative_constraints>
                [
                    여기에 오답 및 형태 붕괴를 방지하기 위해 '절대 하지 말아야 할 금지 지침(Negative Constraints)'을 부정형 명령문으로 기술,
                    기존 항목과 의미가 중복된다면 새 항목을 추가하지 말고, 필요한 경우 기존 항목을 더 일반적인 지침으로 압축하여 기술,
                    (※ 주의: <execution_steps>에 등장한 단어를 단순히 부정형으로 바꾸어 중복 기술하는 것을 금지합니다. 오직 배제할 행동만 정의하세요.)
                ]
            </negative_constraints>

            <few_shot_examples>
                [여기에 필요하다면, 테스터에게 제공할 2개 이내의 구체적인 문제 풀이 예시(Few-shot)를 기술]
            </few_shot_examples>

        </OptimizerOutputConstraint>
    </output_format>
    """


# [임시 비활성화] total_analysis 태그는 현재 사용하지 않습니다. 원복 시 아래 블록 복구 -->
      
#         <total_analysis>
#             <review> [이곳에 상기 분석을 바탕으로, TesterLLM의 차기 프롬프트에 직접 반영할 수 있는 종합적이고 일반화된 행동 지침(피드백)을 작성하세요.] </review>
#         </total_analysis>
       