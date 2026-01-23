"""
| 역할 | 도구 | 설명 |
|---|---|---|
| 감독관 (Optimizer) | DSPy | 프롬프트를 깎고, 점수를 매기고, 수정합니다. (지능 담당) |
| 수험생 (Clean LLM) | LangChain | 일반적인 서비스 환경과 동일한 체인을 돌립니다. (실행 담당) |
| 시험지 (Prompt) | String | DSPy가 만든 텍스트가 LangChain의 SystemMessage로 주입됩니다. |

"""
#####################################################################
################## models/rag_module.py 대신 사용 ##################
import dspy
from langchain.agents import initialize_agent, Tool
from langchain.llms import OpenAI # 또는 AzureChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

class LangChainStudent(dspy.Module):
    def __init__(self, tools):
        super().__init__()
        # 1. LangChain 컴포넌트 설정
        # (실제로는 infrastructure.llm_client 설정을 가져와야 함)
        self.llm = OpenAI(temperature=0) 
        self.tools = tools
        
        # 2. Agent 초기화 (여기서는 Zero-shot React 예시)
        # 주의: 실제 실행 시점(forward)에 프롬프트를 바꿀 수 있도록 구조를 잡아야 함
        self.agent_executor = initialize_agent(
            tools=tools, 
            llm=self.llm, 
            agent="zero-shot-react-description", 
            verbose=True
        )
        
        # 3. 최적화 대상인 '지시문'을 저장할 변수
        self.current_instruction = "당신은 도움이 되는 AI 비서입니다."

    def update_instruction(self, new_instruction):
        """
        [핵심] RL Agent가 만든 '새로운 프롬프트'를 LangChain에 주입하는 함수
        LangChain Agent의 System Prompt를 교체합니다.
        """
        self.current_instruction = new_instruction
        
        # LangChain Agent 종류에 따라 프롬프트 교체 방식이 다름
        # 예: agent.agent.llm_chain.prompt.template 수정 등
        # 여기서는 개념적으로 설명: 실행 시 prefix로 붙여버림
        
    def forward(self, question, context=None):
        """
        DSPy Env가 호출하는 실행 함수
        """
        # 1. 최적화된 지시문 + 질문 결합
        final_prompt = f"{self.current_instruction}\n\n질문: {question}"
        if context:
            final_prompt += f"\n참고 정보: {context}"
            
        # 2. LangChain Agent 실행 (도구 사용 가능!)
        try:
            response = self.agent_executor.run(final_prompt)
        except Exception as e:
            response = f"Error: {str(e)}"
            
        # 3. DSPy 형식(Prediction)으로 반환
        return dspy.Prediction(answer=response)


#####################################################################
############# environment/dspy_rag_env.py 수정 예시 #################
def step(self, action):
        # Action(새 프롬프트)이 들어오면 학생에게 주입
        if action:
            # 기존: self.student_module.prog.signature.__doc__ = action
            # 변경: LangChain Wrapper의 메서드 호출
            if hasattr(self.student_module, 'update_instruction'):
                self.student_module.update_instruction(action)
            else:
                # 기존 DSPy 모듈일 경우 처리
                pass 
        
        # ... (평가 로직 동일) ...

#####################################################################
############# langchain 의 agent #################
"""

LangChain의 Agent는 "네가 가진 도구(Tools)를 이용해서, 
이 문제를 해결할 방법을 스스로 생각해서 실행해"라고 자율성을 주는 것이다.
# LangChain Agent 핵심 구성 요소 정리

| 구성 요소 (Component) | 역할 (Role/Analogy) | 상세 설명 (Description) |
| :--- | :--- | :--- |
| **1. tools** | **작업 도구함**<br>*(쥐여주는 무기)* | • **미리 만들어둔 함수(Function)들의 리스트**입니다.<br>• Agent는 도구의 **"이름"**과 **"설명(Description)"**만 보고, 현재 상황에 필요한 도구를 스스로 판단합니다.<br>• *예시: `GoogleSearchTool`(검색), `CalculatorTool`(계산)* |
| **2. llm** | **두뇌**<br>*(생각 담당 엔진)* | • 도구를 언제, 어떻게 사용할지 결정하는 **판단력**을 가진 모델입니다. (예: GPT-4, Claude 3.5)<br>• 개발자가 짠 정적 로직(`if-else`) 대신, **실시간으로 판단하여 실행 경로를 작성**합니다. |
| **3. agent**<br>`"zero-shot-react-description"` | **행동 지침**<br>*(일하는 프로토콜)* | **이름에 담긴 3가지 핵심 의미:**<br>1. **Zero-shot**: 예제(Few-shot) 없이도 알아서 수행함.<br>2. **ReAct (Reason+Act)**: **"생각(Reason)하고 행동(Act)"**함.<br>&nbsp;&nbsp;→ 그냥 답을 뱉지 않고 *"혼잣말(Thought) → 행동(Action) → 관찰(Observation)"* 과정을 거침.<br>3. **Description**: 도구를 선택할 때 **도구의 설명서(docstring)**를 읽고 결정함. |


+ 그리고 AI 툴로만든 모든 파일은 워터마크가 있어야 함.
"""

