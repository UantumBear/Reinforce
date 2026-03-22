"""
@경로 utils/llm_patches/textgrad_patches.py
@설명 TextGrad 라이브러리의 OpenAI API 호환성 문제를 해결하는 몽키 패치
- textgrad==0.1.8 기준으로 작성 됨.
"""
import inspect
import textgrad as tg
from textgrad.optimizer.optimizer import TextualGradientDescentwithMomentum, construct_tgd_prompt


def patch_textgrad_openai_compatibility():
    """
    TextGrad 라이브러리의 OpenAI API 호환성 문제를 해결하는 몽키 패치.
    
    @문제상황:
    - TextGrad는 구버전 OpenAI API를 기준으로 작성됨
    - 최신 모델(o-series, gpt-5)은 다른 API 파라미터를 요구:
      1) max_tokens 대신 max_completion_tokens 사용
      2) temperature 조정 불가 (무조건 1로 설정되어야 함)
    
    @해결방법:
    - TextGrad.ChatOpenAI 클래스의 _generate_from_single_prompt 메서드를 런타임에 덮어씀
    - o-series/gpt-5 모델은 새로운 API 스펙으로 호출
    - 다른 모델(gpt-4 등)은 원래 구현 사용
    
    @몽키패치란?:
    - 런타임에 외부 라이브러리의 코드를 동적으로 변경하는 기법
    - 라이브러리 소스코드를 직접 수정하지 않고도 동작을 바꿀 수 있음
    - 임시 해결책이며, TextGrad가 업데이트되면 제거 가능
    """
    from textgrad.engine.openai import ChatOpenAI

    # 이미 패치가 적용되었는지 확인 (중복 패치 방지)
    if hasattr(ChatOpenAI, "_original_generate_from_single_prompt"):
        return

    # 원본 메서드를 백업 (다른 모델은 이 원본을 계속 사용)
    ChatOpenAI._original_generate_from_single_prompt = ChatOpenAI._generate_from_single_prompt

    def _patched_generate_from_single_prompt(self, prompt: str, system_prompt: str = None, temperature=0, max_tokens=10000, top_p=0.99):
        """
        수정된 프롬프트 생성 메서드.
        모델명에 따라 다른 API 호출 방식을 사용한다.
        
        [개선] max_tokens을 role에 따라 동적으로 조정:
        - OptimizerLLM (backward): 긴 프롬프트 생성 필요 → 10000 tokens
        - ForwardLLM (tester): 답변 생성 → 2000 tokens (기본값)
        """
        import os
        
        model_name = (self.model_string or "").lower()
        
        # [추가] System prompt에 "optimizer"가 포함되어 있으면 OptimizerLLM으로 판단
        sys_prompt_arg = system_prompt if system_prompt else self.system_prompt
        is_optimizer = "optimizer" in (sys_prompt_arg or "").lower() or "optimization" in (sys_prompt_arg or "").lower()
        
        # OptimizerLLM은 더 많은 토큰 필요 (환경변수로 오버라이드 가능)
        if is_optimizer:
            max_tokens = int(os.getenv("TEXTGRAD_OPTIMIZER_MAX_TOKENS", "20000"))
        else:
            max_tokens = int(os.getenv("TEXTGRAD_FORWARD_MAX_TOKENS", str(max_tokens)))
        
        # o-series (o1, o3 등) 또는 gpt-5 모델인 경우
        if model_name.startswith("o") or model_name.startswith("gpt-5"):
            # 캐시 확인 (이전에 동일한 프롬프트를 호출했다면 재사용)
            cache_or_none = self._check_cache(sys_prompt_arg + prompt)
            if cache_or_none is not None:
                return cache_or_none

            # OpenAI API 호출 (최신 API 스펙 사용)
            response = self.client.chat.completions.create(
                model=self.model_string,
                messages=[
                    {"role": "system", "content": sys_prompt_arg},
                    {"role": "user", "content": prompt},
                ],
                max_completion_tokens=max_tokens,  # [중요] max_tokens 대신 max_completion_tokens
                temperature=1,  # [중요] o-series는 temperature=1 고정
            )

            response = response.choices[0].message.content
            self._save_cache(sys_prompt_arg + prompt, response)
            return response

        # 다른 모델(gpt-4, gpt-4o 등)은 원래 구현 사용
        return ChatOpenAI._original_generate_from_single_prompt(self, prompt, system_prompt, temperature, max_tokens, top_p)

    # TextGrad의 메서드를 패치된 버전으로 교체
    ChatOpenAI._generate_from_single_prompt = _patched_generate_from_single_prompt

def patch_textgrad_momentum_compatibility():
    """textgrad==0.1.8 momentum optimizer의 _update_prompt 반환 누락을 보완한다."""
    source = inspect.getsource(TextualGradientDescentwithMomentum._update_prompt)
    if "return prompt" in source:
        return

    def _patched_update_prompt(self, variable: tg.Variable, momentum_storage_idx: int):
        past_values = ""
        past_n_steps = self.momentum_storage[momentum_storage_idx]
        for i, step_info in enumerate(past_n_steps):
            past_values += f"\n{variable.get_role_description()} at Step {i + 1}: {step_info['value']}.\n"

        optimizer_information = {
            "variable_desc": variable.get_role_description(),
            "variable_value": variable.value,
            "variable_grad": variable.get_gradient_text(),
            "variable_short": variable.get_short_value(),
            "constraint_text": self.constraint_text,
            "past_values": past_values,
            "new_variable_start_tag": self.new_variable_tags[0],
            "new_variable_end_tag": self.new_variable_tags[1],
            "in_context_examples": "\n".join(self.in_context_examples),
        }

        prompt = construct_tgd_prompt(
            do_momentum=(self.do_momentum and (past_values != "")),
            do_constrained=self.do_constrained,
            do_in_context_examples=(self.do_in_context_examples and (len(self.in_context_examples) > 0)),
            **optimizer_information,
        )
        return prompt

    TextualGradientDescentwithMomentum._update_prompt = _patched_update_prompt