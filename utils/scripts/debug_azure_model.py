# python scripts/debug_azure_model.py
import os
import sys
from pathlib import Path

# 프로젝트 루트를 파이썬 패스에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from openai import AzureOpenAI
from conf.config import Settings

# 1. 설정 로드
Settings.setup()

# 2. Azure 클라이언트 생성 (LangChain 아님, 순수 클라이언트)
client = AzureOpenAI(
    api_key=Settings.AZURE_OPENAI_API_KEY,
    api_version=Settings.AZURE_OPENAI_API_VERSION,
    azure_endpoint=Settings.AZURE_OPENAI_ENDPOINT
)

deployment_name = Settings.TESTER_MODEL  # gpt-5-mini

print(f"🔍 배포 이름(Deployment Name): {deployment_name}")
print("-" * 50)

try:
    # 3. o1 계열 확인 사살을 위해 temperature=1로 요청
    response = client.chat.completions.create(
        model=deployment_name,
        messages=[{"role": "user", "content": "Hello!"}],
        temperature=1.0 
    )

    # 4. 정체 공개
    print(f"✅ 실제 모델 ID (Real Model ID): {response.model}")
    
    # 5. o1 모델만의 특징(Reasoning Token) 확인
    if hasattr(response.usage, 'completion_tokens_details'):
        details = response.usage.completion_tokens_details
        if hasattr(details, 'reasoning_tokens'):
             print(f"🧠 Reasoning Tokens 사용량: {details.reasoning_tokens}")
             print("   (이 값이 0 이상이면 빼박 o1 계열 모델입니다!)")

except Exception as e:
    print(f"❌ 에러 발생: {e}")
    if "unsupported_value" in str(e) and "temperature" in str(e):
        print("👉 증거 확보: temperature=0 에러가 난다면 100% o1 모델입니다.")