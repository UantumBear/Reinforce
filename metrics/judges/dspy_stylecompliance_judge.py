import dspy

# 심판관 프롬프트
from metrics.prompts.dspy_judges_prompts import STYLE_PROMPT


class StyleComplianceJudge(dspy.Signature):
    __doc__ = STYLE_PROMPT

    target_reference = dspy.InputField(desc="따라야 할 스타일이 담긴 모범 답안")
    model_answer = dspy.InputField(desc="모델이 생성한 답변")
    
    is_compliant = dspy.OutputField(desc="True if style matches, False otherwise")
    feedback = dspy.OutputField(desc="스타일이 어떻게 다른지 구체적인 피드백 (한국어)")