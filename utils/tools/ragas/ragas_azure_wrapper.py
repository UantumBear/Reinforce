"""
@경로: utils.tools.ragas.ragas_azure_wrapper
@설명: RAGAS 평가용 Azure OpenAI 래퍼 클래스
- RAGAS가 강제로 주입하는 temperature 파라미터를 무시하기 위해 생성하였다.
- TODO 26.02.09 아직 문제를 해결하지 못하였다. 
"""


from langchain_openai import AzureChatOpenAI
from typing import Any, List, Optional

# ----------------------------------------------------------------
# RAGAS 호환용 래퍼 클래스
# RAGAS가 강제로 주입하는 temperature 파라미터를 "먹어버리는" 클래스
# ----------------------------------------------------------------

class RagasAzureOpenAIWrapper(AzureChatOpenAI):
    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        # RAGAS가 내부적으로 temperature=1e-8 등을 보내면 여기서 삭제합니다.
        if "temperature" in kwargs:
            del kwargs["temperature"]
            print("[RAGAS Wrapper] Removed temperature parameter for RAGAS compatibility.")
            
        # 부모 클래스(AzureChatOpenAI) 호출 시에는 temperature 없이 호출됩니다.
        # 따라서 모델은 자신의 기본값(대부분 1.0)으로 동작합니다.
        return super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs)