"""
@경로 : utils/agents/agent.py
@설명: 프롬프트 최적화 Agent class
"""
import time
from typing import Dict
# from langchain_core.prompts import PromptTemplate # 단순한 문자열 프롬프트 템플릿
from langchain_core.prompts import ChatPromptTemplate # 대화형 프롬프트 템플릿
from langchain_core.output_parsers import StrOutputParser
from conf.config import Env
from utils.models.model import ModelFactory


class OptimizationAgent:
    """
        프롬프트 최적화를 위한 에이전트 클래스 (Optimizer LLM)
        강화학습의 Policy 역할을 담당한다.

        목표: LLM을 사용한, LLM의 프롬프트 최적화
        - 에이전트가 환경과 상호작용하며 시행착오를 통해 최적의 정책을 학습한다. 

    @역할 :  
        - 이 에이전트는 '프롬프트 엔지니어'이다. 
        - 환경(Env)에서 피드백(점수, 오답노트)을 받아서, Target LLM이 사용할 '새로운 시스템 프롬프트'를 작성한다.

    @목표 :
        질문에 대해, 모범답안(reference_answer)과 유사한 답변이 나오도록 시스템 프롬프트를 수정하는 것.    
    @State (상태) :
        에이전트가 다음 행동을 결정하는 데 필요한 모든 정보
        - previous_prompt : 이전에 사용된 프롬프트 == 에이전트가 직전에 생성한 행동
        - previous_reward : 이전 프롬프트에 대한 보상 (점수) == 직전 행동에 대한 보상
        - question : 현재 질문 == 최적화 대상인 목표 질문.
        - reference_answer : 참조 답안 == Target LLM이 도달해야 하는 목표 답변(정답)
        - worst_case : (MDP-v3용) 가장 점수가 낮았던 오답 사례 {question, reference, prediction}

    @ACtion (행동) :

    @Reward (보상) :
        - Target LLM이 새로운 프롬프트를 적용하여 생성한 답변과 모범 답안(reference_answer) 간의 코사인 유사도


    """
    
    def __init__(self, version: str = "MDP-v3"):
        """에이전트 초기화
        
        @param
            use_azure (bool): Azure 사용 여부
        """

        self.model_factory = ModelFactory()
        
        self.optimizer_llm = self.model_factory.get_llm(model_type="optimizer") # 프롬프트를 생성하는 LLM
        self.use_azure = Env.USE_AZURE
        self.version = version
        
        # 에이전트(Optimizer LLM)에게 어떤 행동을 할지 지시하는 프롬프트, 버전에 따라 내부에서 다르게 생성됨.
        self.policy_prompt = self.get_policy_prompt()


    def act(self, state: Dict) -> str:
        """ 상태(State)를 보고 행동(Action: 새 프롬프트 생성)을 결정
        @param
            state (Dict): 현재 상태
        @return
            str: 새로운 프롬프트
        """
        if not self.use_azure: 
            # 제미나이는 무료티어에서 분당 요청 수 제한이 있으므로 대기
            time.sleep(2)
            
        # 체인 = PromptTemplate | LLM | OutputParser
        # 체인 구성: Prompt(설정+피드백) -> LLM(생성) -> String(텍스트추출)
        chain = self.policy_prompt | self.optimizer_llm | StrOutputParser()

        # 1. State에서 필요한 정보 추출 및 기본값 설정        
        prev_reward = state.get("previous_reward", 0.0)
        prev_reward_t2 = state.get("previous_reward_t2", 0.0)  
        worst_case = state.get("worst_case", {}) or {}      
        worst_prompt = worst_case.get("prompt", state.get("previous_prompt", "N/A"))

        # 2. 프롬프트 템플릿에 넣을 변수 딕셔너리 생성
        input_variables = {
            # [History] 과거 기록
            "previous_prompt_t2": state.get("previous_prompt_t2", ""), # main_logging.py 에서 넘김
            "previous_reward_t2": prev_reward_t2,
            "previous_prompt": state.get("previous_prompt", ""), 
            "previous_reward": prev_reward,
            
            # [Reflexion] 오답 노트 (최악의 사례)
            "worst_question": worst_case.get("question", "N/A"),
            "worst_reference": worst_case.get("reference", "N/A")[:200],
            "worst_prediction": worst_case.get("prediction", "N/A")[:200],
            "worst_prompt": worst_prompt[:200],
            
            # [Goal] 현재 목표
            "current_question": state.get("question", ""),
            "reference_answer": state.get("reference_answer", "")[:100],
            
            # [Analysis] 분석 보조 지표
            "score_trend": f"2단계 전: {prev_reward_t2:.4f} → 직전: {prev_reward:.4f}",
            "improvement_needed": "높음" if prev_reward < 0.7 else ("보통" if prev_reward < 0.85 else "낮음")
        }

        # CoT Reasoning: 패턴 분석과 논리적 추론을 유도
        # Reflexion: 과거 실패 사례를 통한 자기 성찰 및 개선
        # 3. 체인 실행 (Action 생성)
        return chain.invoke(input_variables).strip()
            




    def get_policy_prompt(self) -> ChatPromptTemplate:
        """현재 정책 프롬프트 반환
        
        @Return
            PromptTemplate: 정책 프롬프트

        @설명
            이 프롬프트는 이전 점수(previous_reward)라는 '보상' 정보를 포함하고 있어 LLM이 이 보상을 바탕으로 다음 '행동(새 프롬프트)'을 계획하도록 유도한다.
            이는, 직전 단계 정보만 사용하는 방식 (MDP 기본)에 해당한다.
            원리) 마르코프 속성 - 현재 상태가 주어지면, 이전 상태들의 정보와 무관하게 다음 상태를 결정한다.

        Q. 이렇게 하면, 직전 단계 정보만으로는 과거의 성공적인 패턴(step2, step3)의 공통점을 취하고,
            실패한 패턴(step1)을 버리는 방식으로 조절하는 것이 불가능하지 않는가..?

        A. LLM의 내재적 능력에 기대어 보았다.
        프롬프트 내의 (이전 프롬프트, 점수) 이전 정보를 기반으로, 

        - Chain-of-Thought (CoT) Reasoning
        - Reflexion
        
        """
        policy_prompt = ChatPromptTemplate.from_template("")

        # if self.version == "MDP-v1":
        #     # MDP-v1 프롬프트: 직전 단계 정보만 사용
        #     policy_prompt = PromptTemplate.from_template("""
        #     [System] 당신은 프롬프트 최적화 전문가입니다.
        #     [Goal] 질문 '{question}'에 대해, 모범답안 '{reference_summary}'와 유사한 답변이 나오도록 시스템 프롬프트를 수정하세요.
        #     [History] 이전 프롬프트: "{previous_prompt}" -> 점수: {previous_reward:.4f}
        #     [Action] 새로운 프롬프트만 출력:
        #     """)
        # elif self.version == "MDP-v2":
        #     # MDP-v2 프롬프트: 2단계 전 정보도 활용
        #     policy_prompt = PromptTemplate.from_template("""
        #     [System] 당신은 프롬프트 최적화 전문가입니다.
        #     [Goal] 질문 '{question}'에 대해, 모범답안 '{reference_summary}'와 유사한 답변이 나오도록 시스템 프롬프트를 수정하세요.
        #     [History] 2단계 전 프롬프트: "{previous_prompt_t2}" -> 점수: {previous_reward_t2:.4f}
        #     [History] 이전 프롬프트: "{previous_prompt}" -> 점수: {previous_reward:.4f}
        #     [Action] 새로운 프롬프트만 출력:
        #     """)
        # elif self.version == "MDP-v3":
        #     # MDP-v3: 2단계 전 + 직전 정보 + 최악의 오답 사례(worst_case) 활용
        #     return PromptTemplate.from_template(
        #         """
        #         [System] 당신은 한국어 LLM 프롬프트 최적화 전문가입니다.
        #         [Goal] 여러 질문들에 대해 평균적으로 높은 코사인 유사도 점수를 얻을 수 있도록
        #             "시스템 프롬프트"를 개선해야 합니다.

        #         [History]
        #         - 2단계 전 시스템 프롬프트:
        #         "{previous_prompt_t2}"
        #         → 평균 점수: {previous_reward_t2:.4f}

        #         - 직전 시스템 프롬프트:
        #         "{previous_prompt}"
        #         → 평균 점수: {previous_reward:.4f}

        #         [Feedback]
        #         아래는 직전 에피소드에서 점수가 가장 낮았던 "최악의 오답 사례"입니다.
        #         이 사례를 분석하여,
        #         전체적인 평균 성능을 높이면서도 동일한 실수를 피할 수 있도록
        #         새로운 시스템 프롬프트를 설계하세요.

        #         - 질문: {worst_question}
        #         - 모범답안(정답): {worst_reference}
        #         - 모델의 응답: {worst_prediction}
        #         - 그때 사용된 시스템 프롬프트: {worst_prompt}

        #         [Action]
        #         - 위 히스토리와 피드백을 모두 반영한
        #         새로운 "시스템 프롬프트"만 출력하세요.
        #         - 불필요한 설명, 따옴표, 마크다운, 코멘트는 포함하지 마세요.
        #         """.strip()
        #     )
        if self.version == "MDP-v3":
            # CoT Reasoning과 Reflexion을 활용한 개선된 프롬프트
            return ChatPromptTemplate.from_messages([
            # ============================================================
            # [1. System Role] 변하지 않는 규칙, 목표, 제약사항 (헌법)
            # ============================================================
            ("system", """
당신은 한국어 LLM 프롬프트 최적화 전문가입니다. Chain-of-Thought 추론과 Reflexion 기법을 활용하여 체계적으로 프롬프트를 개선합니다.

[최우선 목표]
질의응답 챗봇 LLM의 성능을 최적화하는 '시스템 프롬프트'를 설계하는 것입니다.
챗봇이 질문에 대해 모범답안의 **논리적 구조, 서술 스타일, 핵심 정보**를 정확히 반영하도록 유도해야 합니다.

[중요: 챗봇과 당신의 차이점 인식]
1. **당신(Optimizer)**: 질문과 '모범답안(정답)'을 모두 볼 수 있습니다. (전지적 시점)
2. **챗봇(Target)**: 오직 **'질문'**과 **'검색된 문서(Context)'**만 볼 수 있습니다. **'모범답안'은 절대 볼 수 없습니다.**

[절대 금지 사항 (Constraints)]
1. 생성하는 프롬프트에 **"모범답안을 참고해라", "제공된 정답과 일치시켜라"** 같은 표현을 **절대 포함하지 마십시오.** (챗봇은 정답을 볼 수 없기 때문입니다.)
2. 대신, 당신이 모범답안을 분석하여 **"답변의 길이, 말투, 두괄식 여부, 필수 포함 항목"** 등을 구체적으로 지시하십시오.
3. 불필요한 서론("분석 결과...", "프롬프트입니다...") 없이 **오직 시스템 프롬프트 내용만** 출력하십시오.
             
[추론 과정]
1. **성과 패턴**: 어떤 프롬프트 요소가 높은/낮은 유사도를 만들었는가?
2. **실패 원인**: 왜 챗봇이 모범답안과 다른 방향으로 답변했는가?
3. **성공 요소**: 좋은 점수를 받은 프롬프트의 공통점은 무엇인가?
4. **개선 방향**: 챗봇이 모범답안과 유사하게 답변하도록 어떻게 유도할 것인가?

[평가 구조 이해]
1. **당신(프롬프트 전문가)**이 시스템 프롬프트를 생성
2. **질의응답 챗봇 LLM**이 그 프롬프트를 사용해 질문에 답변
3. **평가**: 질의응답 챗봇의 답변과 모범답안 간의 **코사인 유사도**로 성능 측정
   - 유사도 높음(0.8+) = 성공적인 프롬프트 (좋은 상태)
   - 유사도 낮음(0.6-) = 실패한 프롬프트 (개선 필요)
             
[최종 액션]
위의 분석을 바탕으로 **질의응답 챗봇이 사용할 새로운 시스템 프롬프트**만 출력하세요.
- 설명이나 주석 없이 프롬프트 내용만
- 챗봇이 모범답안과 유사한 답변을 생성하도록 명확히 유도
- 실패 패턴을 방지하고 성공 패턴을 강화하는 방향으로
- 한국어로 자연스럽게 작성
"""),
                # 2. 사용자 요청 (CoT + Reflexion 구조)
                ("user", """
[현재 상황 분석]
1. 성과 추이: {score_trend}
2. 현재 목표 질문: "{current_question}"
3. 참고할 모범답안(당신만 볼 수 있음): "{reference_answer}"

[Reflexion - 과거 시도 및 반성]
**2단계 전 시스템 프롬프트:**
"{previous_prompt_t2}"
→ 챗봇 평균 유사도: {previous_reward_t2:.4f}

**직전 시스템 프롬프트:**
"{previous_prompt}"  
→ 챗봇 평균 유사도: {previous_reward:.4f}

**최악의 실패 사례 (Reflexion 대상):**
- 질문: {worst_question}
- 모범답안: {worst_reference}
- 챗봇이 실제로 생성한 답변(오답): {worst_prediction}
- 그때 사용된 시스템 프롬프트: {worst_prompt}
→ [분석 요청] 이 경우의 유사도가 가장 낮았습니다. 챗봇이 왜 모범답안 스타일대로 답변하지 못했는지 원인을 분석하세요.

[최종 액션]
위의 분석을 바탕으로, 챗봇이 모범답안과 유사한 답변을 생성하도록 유도하는 **새로운 시스템 프롬프트**를 작성하세요.
(설명 없이 프롬프트 본문만 출력)
""")
        ])
        else:
            policy_prompt = ChatPromptTemplate.from_template("Error: Invalid Version")
         

        return policy_prompt