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
    @tool
    def save_meeting_log(content: str, filename: str) -> str:
        """
        [매우 중요] 회의록 내용을 파일로 저장하는 도구입니다.
        사용자가 저장을 요청하면 묻지 말고 즉시 이 도구를 사용하세요.
        """
        try:
            # 절대 경로 문제 방지를 위해 현재 위치에 저장
            current_path = os.getcwd()
            full_path = os.path.join(current_path, filename)
            
            with open(full_path, "w", encoding="utf-8") as f:
                f.write(content)
            return f"성공: 파일이 '{full_path}'에 저장되었습니다."
        except Exception as e:
            return f"실패: {e}"

    # 4. 에이전트 생성
    agent_executor = create_react_agent(llm, [save_meeting_log])

    # 5. [핵심] 딴소리 못하게 강력한 명령 내리기
    query = """
    오늘 날짜가 포함된 간단한 테스트 회의록을 작성하세요.
    그리고 작성된 내용을 'test_log.txt' 라는 파일명으로 **즉시 저장하세요**.
    
    [주의사항]
    - 사용자에게 파일명을 묻지 마세요.
    - "저장할까요?"라고 묻지 마세요.
    - save_meeting_log 도구를 반드시 실행하세요.
    """
    
    print(f"\n>>> [User Query] {query.strip()}\n")
    
    # 6. 실행
    # SystemMessage를 추가하여 AI의 페르소나를 고정합니다.
    result = agent_executor.invoke({
        "messages": [
            SystemMessage(content="당신은 요청받은 즉시 행동하는 AI입니다. 질문보다 행동을 우선하세요."),
            HumanMessage(content=query)
        ]
    })
    
    print(f"\n>>> [AI Final Response] {result['messages'][-1].content}")

    # 7. 진짜 파일이 생겼는지 팩트 체크 (Python 코드로 확인)
    print("\n---------------------------------------------------")
    target_file = "test_log.txt"
    if os.path.exists(target_file):
        print(f"✅ [성공] 진짜로 '{target_file}' 파일이 생성되었습니다!")
        print("   (왼쪽 파일 탐색기에서 새로고침 하시면 보입니다)")
        
        # 파일 내용 살짝 보여주기
        with open(target_file, "r", encoding="utf-8") as f:
            print(f"\n[파일 내용 미리보기]:\n{f.read()[:100]}...")
    else:
        print(f"❌ [실패] AI가 또 말을 안 들었습니다. 파일이 없습니다.")

if __name__ == "__main__":
    main()