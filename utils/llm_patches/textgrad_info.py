"""
TEXTGRAD 내부 에서 가지고 있는 정보를 추출하기 위한 함수 모음.

"""
from utils.log.console import print_step
import textgrad as tg
from textgrad.optimizer.optimizer import TextualGradientDescentwithMomentum

# 1. TGD optimizer 인스턴스에서 시스템 프롬프트 추출
def get_tgd_optimizer_system_prompt(optimizer) -> str:
    """TGD optimizer 인스턴스가 실제 사용하는 시스템 프롬프트를 추출한다."""
    for attr_name in ("optimizer_system_prompt", "system_prompt"):
        prompt_value = getattr(optimizer, attr_name, None)
        if isinstance(prompt_value, str) and prompt_value.strip():
            # print_step("[TGD Optimizer의 SYSTEM_PROMPT]")
            # print(prompt_value)
            """
            'You are part of an optimization system that improves text (i.e., variable). 
            You will be asked to creatively and critically improve 
            prompts, solutions to problems, code, or any other text-based variable. 
            You will receive some feedback, and use the feedback to improve the variable. 
            The feedback may be noisy, identify what is important and what is correct. 
            Pay attention to the role description of the variable, 
            and the context in which it is used. 
            This is very important: 
            You MUST give your response by sending the improved variable 
            between <IMPROVED_VARIABLE> {improved variable} </IMPROVED_VARIABLE> tags. 
            The text you send between the tags will directly replace the variable.\n\n\n
            ### Glossary of tags that will be sent to you:\n
            # # - <LM_SYSTEM_PROMPT>: The system prompt for the language model.\n
            # # - <LM_INPUT>: The input to the language model.\n
            # # - <LM_OUTPUT>: The output of the language model.\n
            # # - <FEEDBACK>: The feedback to the variable.\n
            # # - <CONVERSATION>: The conversation history.\n
            # # - <FOCUS>: The focus of the optimization.\n
            # # - <ROLE>: The role description of the variable.'
            """
            return prompt_value
    return "[N/A] Unable to capture TGD optimizer system prompt from optimizer instance."

# 2. TGD update 프롬프트(str/list)를 로그 저장용 문자열로 변환
def stringify_tgd_update_prompt(prompt_value) -> str:
    """TGD update prompt(str/list)를 로그 저장용 문자열로 변환한다."""
    if prompt_value is None:
        return "[N/A] TGD update prompt is unavailable."
    if isinstance(prompt_value, str):
        return prompt_value
    if isinstance(prompt_value, list):
        return "\n\n".join(str(item) for item in prompt_value)
    return str(prompt_value)

# 3. optimizer 타입별 _update_prompt 시그니처를 맞춰 로그 문자열로 변환
def capture_optimizer_update_prompt(optimizer, variable: tg.Variable, momentum_storage_idx: int = 0) -> str:
    """optimizer 타입별 _update_prompt 시그니처를 맞춰 로그 문자열로 변환한다."""
    if isinstance(optimizer, TextualGradientDescentwithMomentum):
        prompt_value = optimizer._update_prompt(variable, momentum_storage_idx=momentum_storage_idx)
    else:
        prompt_value = optimizer._update_prompt(variable)
    return stringify_tgd_update_prompt(prompt_value)