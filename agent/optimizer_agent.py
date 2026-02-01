"""
@경로: agent/optimizer_agent.py
@설명: 
- 프롬프트 최적화 에이전트 (Prompt Optimization Agent)
- [Input]: Env에서 관측된 State(실패 로그, 현재 지시문)를 받는다.
- [Process]: 이전의 실수(Verbal Feedback)를 분석하여 개선 방향을 추론한다.
- [Output]: 새로운 프롬프트(Action)을 생성하여 반환(Return)한다.
- (※ 반환된 Action은 Main Loop가 받아 Env에 적용한다.)
"""
import dspy
from agent.prompts.optimizer_llm import OPTIMIZER_SYSTEM_PROMPT
from infrastructure.llm_client import get_optimizer_llm

"""
Optimizer LLM 역할 정의
- DSPy의 Signature 클래스에 있는 Docstring(주석)은 단순한 설명이 아니다.
- DSPy는 이 Docstring을 분석하여 LLM에게 보낼 '구조화된 프롬프트'를 자동으로 조립한다.
"""
class OptimizerLLM(dspy.Signature):
    attempted_instruction = dspy.InputField(desc="직전 단계 프롬프트: OptimizerLLM이 만들어서 '깨끗한 LLM'이 사용했던 시스템 프롬프트")
    total_score = dspy.InputField(desc="직전 단계 프롬프트로 인한 총점수 (0.0 ~ 1.0)")
    similarity_score = dspy.InputField(desc="직전단계 프롬프트로 인한 평균 semantic 유사도 점수 (0.0 ~ 1.0)")
    # 'verbal_feedback'은 전체적인 총평(Analysis)을 넣으면 좋다고 한다. 고민해보자.
    verbal_feedback = dspy.InputField(desc="언어 피드백")
    fail_case_feedback = dspy.InputField(desc="실패 사례 피드백 (오답노트)")
    new_instruction = dspy.OutputField(desc="개선된 시스템 프롬프트 (System Instruction Only)")

# 클래스 정의가 끝난 뒤, 강제로 Docstring을 주입한다. 
# OptimizerLLM 클래스 밑에 """ """ 형태로 doctring을 작성할 수도 있지만, 가독성이 떨어지기에 별도로 분리하였다.
# DSPy는 실행 시점에 이 __doc__을 읽어서 프롬프트를 구성하므로 완벽하게 작동한다.
OptimizerLLM.__doc__ = OPTIMIZER_SYSTEM_PROMPT

"""
PromptOptimizerAgent 클래스 정의
"""
class PromptOptimizerAgent:
    def __init__(self):
        # [중앙화된 LLM 관리]
        # infrastructure/llm_client.py에서 관리되는 Optimizer LLM 가져오기
        self.optimizer_lm = get_optimizer_llm()
        
        # [PromptOptimizerAgent Module 초기화]
        # 기본 모듈만 생성 (LLM은 실행 시 컨텍스트로 지정)
        self.llm_module = dspy.Predict(OptimizerLLM)
        # self.llm_module2 = textgrad.function(OptimizerLLM)
        """
        @ dspy.Predict(Signature):
        - Signature의 Docstring, InputField, OutputField를 분석한다. 
        - 이를 바탕으로 LLM에게 보낼 '구조화된 프롬프트'를 자동으로 조립하는 역할한다. (dspy의 핵심 기능)
        - 또한, LLM의 응답(String)에서 필요한 OutputField만 추출(Parsing)하는 기능도 포함한다. 

        @ self.llm_module:
        - 최적화를 수행할 함수형 객체(Callable Module) 이다. 
        - [상태]: init 시점에는 대기 상태(Idle)이며, API 호출은 발생하지 않는다. 
        - [실행]: act 에서 호출되어, 새로운 프롬프트를 생성한다.
        - response = self.llm_module(current_instruction=...) 형태로 사용된다.
        - 위와 같이 호출 될 때 실제 LLM API 호출이 발생한다. 
        """

    def act(self, state):
        """
        @ act():
        - Action 은 Agent 의 존재 가치이다.
        - Agent가 Env과 상호작용하는 유일한 방법이다.
        - Env로부터 State 를 받아서 보고, Action 을 결정한다.

        - [Input]: state (Dict)

        [state 구성 요소]
        - state['current_instruction']: (Env 입장에서는 현재였지만, Agent 입장에서는 과거)
        - similarity_score_str = str(state.get("current_similarity_score", 0.0))
        - verbal_feedback = state.get("verbal_feedback", "")
        - fail_case_feedback = state.get("fail_case_feedback", "")
        """

        # 1. State 파싱
        attempted_instruction = state.get("current_instruction", "")
        total_score_str = str(state.get("total_score", 0.0))
        similarity_score_str = str(state.get("current_similarity_score", 0.0))
        fail_case_feedback = state.get("fail_case_feedback", "") 
        verbal_feedback = state.get("verbal_feedback", "No analysis.")

        
        # 피드백이 없으면(성공했으면) 행동하지 않음
        if not fail_case_feedback or fail_case_feedback == "None":
            return None

        # 3. Optimizer LLM 컨텍스트에서 실행
        with dspy.context(lm=self.optimizer_lm):
            response = self.llm_module(
                attempted_instruction=attempted_instruction,  # Signature의 attempted_instruction에 매핑
                total_score=total_score_str,            # Signature의 total_score에 매핑
                similarity_score=similarity_score_str,  # Signature의 similarity_score에 매핑
                verbal_feedback=verbal_feedback,
                fail_case_feedback=fail_case_feedback
            )

        return response.new_instruction