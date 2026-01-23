"""
기능 테스트 용도의 단독 실행 파일 (프로젝트와 무관하게 단일 파일로 작동한다.)

 (LangChain 1.2.4+ / LangGraph 호환)

@경로: utils.colab.services.test.tool_run.py
@실행명령어: 
# 로컬
python services/test/langc/tool_run.py
# 코랩 셀에서 실행할 때
!uv run python utils/colab/services/test/langc/tool_run.py
"""

"""
공부
프롬프트의 종류
## SystemMessage (시스템 메시지)
역할: [감독의 지시사항]
AI의 성격, 말투, 절대 어기면 안 되는 규칙을 설정합니다. 대화 내내 AI의 머릿속에 박혀있는 "헌법" 같은 존재

## HumanMessage (휴먼 메시지)
역할: [사용자의 질문]
실제 채팅창에서 사용자가 입력하는 질문입니다. 우리가 query 변수에 담는 내용이 바로 이것

## AIMessage (AI 메시지)
역할: [AI의 대답]
설명: AI가 내뱉은 답변입니다. 보통은 결과물로 받지만, "가짜 기억(Few-shot Learning)"을 주입할 때 개발자가 임의로 만들어서 찔러 넣기도 한다.
단순 텍스트뿐만 아니라 "도구 호출 요청(tool_calls)" 정보도 여기에 담긴다.

## ToolMessage (툴 메시지)
역할: [도구의 실행 결과]
이게 바로 LangGraph가 돌아가는 핵심
- AI가 도구를 쓰겠다고 함 (AIMessage에 tool_calls 포함)
- LangGraph가 실제로 파이썬 함수를 실행함
- 그 실행 결과("성공했습니다" or "에러")를 이 ToolMessage에 담아서 다시 AI에게 보여줌
ex)
ToolMessage(
    tool_call_id="call_AbCd123", # 어떤 요청에 대한 답인지 ID로 매칭
    content="성공: 파일이 'test.txt'에 저장되었습니다."
)

(특수) MessagesPlaceholder (메시지 플레이스홀더)
역할: [대화 기억 저장소]
이건 직접적인 메시지는 아니지만, "이전 대화 내용(History)" 이 몽땅 쏟아져 들어갈 빈 자리를 예약해두는 것이다.

"""


import os
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.prebuilt import create_react_agent

# 1. 환경 변수 로드 (.env가 있으면 로드)
load_dotenv()

def main():
    print(">>> [System] Tool Test Start...")

    # ------------------------------------------------------------------
    # [핵심 수정] .strip()을 붙여서 숨어있는 엔터(\r, \n)나 공백을 제거합니다.
    # ------------------------------------------------------------------
    def get_clean_env(key):
        value = os.getenv(key)
        if value:
            return value.strip() # 문자열 앞뒤의 공백/줄바꿈 제거
        return None

    api_key = get_clean_env("AZURE_OPENAI_API_KEY")
    azure_endpoint = get_clean_env("AZURE_OPENAI_ENDPOINT")
    api_version = get_clean_env("AZURE_OPENAI_API_VERSION")
    deployment_name = get_clean_env("AZURE_GPT4DOT1_DEPLOYMEN")

    # 값 확인
    if not api_key or not azure_endpoint:
        print("[Error] API Key 또는 Endpoint가 없습니다.")
        return

    # LLM 초기화
    llm = AzureChatOpenAI(
        deployment_name=deployment_name,
        api_key=api_key,
        azure_endpoint=azure_endpoint,
        api_version=api_version,
        temperature=0
    )

    # 3. 도구 정의 (AI가 이 도구를 써야 파일이 진짜로 생깁니다)
    네, 눈썰미가 대단하십니다! 👍 아까는 여러 기능을 보여드리느라 save_meeting_log를 좀 더 일반적인 save_file로 바꿨었는데, 원하시는 대로 **"회의록 전용 도구"**를 다시 살려서 넣어드릴게요.

이번에는 [시간 확인] + [계산] + [회의록 작성 및 저장] + [파일 확인] 이 4가지가 모두 들어간 종합 선물 세트입니다.

아래 코드를 덮어쓰기 하시면 됩니다.

Python

"""
종합 AI 에이전트 (회의록 도구 복구 버전)
기능: 회의록 저장, 파일 목록 확인, 시간 확인, 계산기

@경로: utils/colab/services/test/langc/tool_run.py
"""

import os
import math
from datetime import datetime
from dotenv import load_dotenv

# LangChain 관련 임포트
from langchain_openai import AzureChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.prebuilt import create_react_agent

# 환경 변수 로드
load_dotenv()

def main():
    print(">>> [System] Meeting Log Agent Start...")

    # 1. 환경 변수 처리 (Colab / Local 호환 및 공백 제거)
    def get_clean_env(key):
        value = os.getenv(key)
        if value: return value.strip()
        return None

    api_key = get_clean_env("AZURE_OPENAI_API_KEY")
    azure_endpoint = get_clean_env("AZURE_OPENAI_ENDPOINT")
    api_version = get_clean_env("AZURE_OPENAI_API_VERSION")
    deployment_name = get_clean_env("AZURE_GPT4DOT1_DEPLOYMEN")

    # 2. LLM 초기화
    llm = AzureChatOpenAI(
        deployment_name=deployment_name,
        api_key=api_key,
        azure_endpoint=azure_endpoint,
        api_version=api_version,
        temperature=0
    )

    # ---------------------------------------------------------
    # 3. 도구(Tools) 정의
    # ---------------------------------------------------------
    
    # [도구 1] 회의록 저장 (요청하신 도구 복구!)
    @tool
    def save_meeting_log(content: str, filename: str) -> str:
        """
        [기능] 작성된 회의록 내용을 파일로 저장합니다.
        [인자] content(회의록 내용), filename(저장할 파일명)
        사용자가 '회의록'을 저장해달라고 하면 반드시 이 도구를 사용하세요.
        """
        try:
            # 절대 경로 문제 방지
            filepath = os.path.join(os.getcwd(), filename)
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(content)
            return f"성공: 회의록이 '{filename}'에 저장되었습니다."
        except Exception as e:
            return f"에러: {e}"

    # [도구 2] 파일 목록 확인
    @tool
    def list_files() -> str:
        """
        [기능] 현재 폴더에 있는 파일들의 목록을 보여줍니다.
        파일이 잘 생성되었는지 검증할 때 사용합니다.
        """
        try:
            files = os.listdir('.')
            return f"현재 파일 목록: {files}"
        except Exception as e:
            return f"목록 조회 실패: {e}"

    # [도구 3] 현재 시간 확인
    @tool
    def get_current_time() -> str:
        """
        [기능] 현재 날짜와 시간을 알려줍니다.
        회의 일시를 기록해야 할 때 사용하세요.
        """
        now = datetime.now().strftime("%Y년 %m월 %d일 %H시 %M분")
        return f"현재 시간: {now}"

    # [도구 4] 계산기
    @tool
    def calculate(expression: str) -> str:
        """
        [기능] 수학 수식을 계산해줍니다.
        예산이나 인원수 계산 등이 필요할 때 사용하세요.
        """
        try:
            result = eval(expression, {"__builtins__": None}, {"math": math})
            return f"계산 결과: {result}"
        except Exception as e:
            return f"계산 오류: {e}"

    # 도구 리스트 취합
    my_tools = [save_meeting_log, list_files, get_current_time, calculate]

    # ---------------------------------------------------------
    # 4. 에이전트 생성
    # ---------------------------------------------------------
    agent_executor = create_react_agent(llm, my_tools)
    #  langraph의 핵심 기능이다.
    # 파이썬 함수를 "LLM이 읽을 수 있는 매뉴얼(JSON)"로 번역해서 LLM에게 미리 찔러넣어준다.


    # ---------------------------------------------------------
    # 5. 복합 미션 부여
    # ---------------------------------------------------------
    # 상황: 회의록을 쓰는데 날짜도 모르고, 예산 계산도 해야 함.
    query = """
    다음 순서대로 업무를 처리해주세요.

    1. [시간 확인] 현재 시간이 언제인지 확인하세요.
    2. [계산] 이번 프로젝트 예산은 '3000달러 * 1350원' 입니다. 얼마인지 계산하세요.
    3. [회의록 작성] 위 시간과 예산 정보를 포함해서 간단한 '프로젝트 킥오프 회의록'을 작성하세요.
    4. [저장] 작성된 내용을 'kickoff_meeting.txt' 파일로 저장하세요. (save_meeting_log 도구 사용)
    5. [확인] 마지막으로 현재 폴더 파일 목록을 조회해서 파일이 잘 생겼는지 확인해주세요.
    """
    
    print(f"\n>>> [User Query]\n{query.strip()}\n")
    print(">>> [AI Thinking] 에이전트가 도구를 선택하고 실행합니다...\n")

    # 실행
    result = agent_executor.invoke({
        "messages": [
            SystemMessage(content="당신은 꼼꼼한 AI 비서입니다. 각 단계를 빠짐없이 수행하고 결과를 보고하세요."),
            HumanMessage(content=query)
        ]
    })
    
    # ---------------------------------------------------------
    # 6. 결과 출력
    # ---------------------------------------------------------
    print(f"\n>>> [Final Result]\n{result['messages'][-1].content}")

if __name__ == "__main__":
    main()