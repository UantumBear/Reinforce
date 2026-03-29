"""
@경로: utils.tools.ragas.ragas_azure_wrapper
@설명: RAGAS 평가용 Azure OpenAI 래퍼 클래스
- RAGAS가 강제로 주입하는 temperature 파라미터를 o4-mini 호환 값으로 변경
- o4-mini는 temperature=1만 지원하므로 모든 temperature 값을 1로 고정
"""

from langchain_openai import AzureChatOpenAI
from langchain_core.messages import AIMessage, BaseMessage
from typing import Any, List, Optional, Dict, Union

# ----------------------------------------------------------------
# RAGAS 호환용 래퍼 클래스
# RAGAS가 강제로 주입하는 temperature 파라미터를 o4-mini 호환값(1)으로 변경
# ----------------------------------------------------------------

class RagasAzureOpenAIWrapper(AzureChatOpenAI):
    """OpenAI o4-mini와 RAGAS 호환용 래퍼 클래스"""
    
    def _fix_temperature(self, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """temperature 파라미터를 o4-mini 호환값으로 수정"""
        if "temperature" in kwargs:
            # o4-mini는 temperature=1만 지원하므로 강제로 1로 설정
            kwargs["temperature"] = 1.0
            print(f"[RAGAS Wrapper] Fixed temperature to 1.0 for o4-mini compatibility")
        return kwargs
    
    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        """동기 생성 메소드 - temperature 수정"""
        kwargs = self._fix_temperature(kwargs)
        return super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs)
    
    async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs):
        """비동기 생성 메소드 - temperature 수정"""
        kwargs = self._fix_temperature(kwargs)
        return await super()._agenerate(messages, stop=stop, run_manager=run_manager, **kwargs)
    
    def invoke(self, input, config=None, **kwargs):
        """invoke 메소드 - temperature 수정"""
        kwargs = self._fix_temperature(kwargs)
        return super().invoke(input, config=config, **kwargs)
    
    async def ainvoke(self, input, config=None, **kwargs):
        """비동기 invoke 메소드 - temperature 수정"""
        kwargs = self._fix_temperature(kwargs)
        return await super().ainvoke(input, config=config, **kwargs)
    
    def _call(self, messages, stop=None, run_manager=None, **kwargs):
        """_call 메소드 - temperature 수정"""
        kwargs = self._fix_temperature(kwargs)
        return super()._call(messages, stop=stop, run_manager=run_manager, **kwargs)
    
    def set_run_config(self, config):
        """RAGAS 호환용 - run config 설정"""
        # RAGAS에서 요구하는 메소드, 부모 클래스에 있으면 호출하고 없으면 무시
        if hasattr(super(), 'set_run_config'):
            return super().set_run_config(config)
        else:
            # config를 내부적으로 저장하거나 무시
            self._run_config = config
            return self
    
    def with_config(self, config=None, **kwargs):
        """LangChain Runnable 인터페이스 호환"""
        if hasattr(super(), 'with_config'):
            return super().with_config(config=config, **kwargs)
        else:
            # Fallback: 새 인스턴스 리턴하거나 self 리턴
            return self
    
    def bind(self, **kwargs):
        """바인딩 메소드 - temperature 수정"""
        kwargs = self._fix_temperature(kwargs)
        return super().bind(**kwargs)
    
    def __getattr__(self, name):
        """누락된 속성들에 대한 fallback"""
        # 부모 클래스에서 속성을 찾아보고, 없으면 기본 동작
        try:
            return getattr(super(), name)
        except AttributeError:
            # RAGAS가 요구하는 일반적인 메소드들에 대한 기본 구현
            if name in ['get_num_tokens', 'get_token_ids', 'get_sub_prompts']:
                return lambda *args, **kwargs: None
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")