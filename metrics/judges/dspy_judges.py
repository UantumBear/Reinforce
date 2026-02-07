import dspy

# 심판관 프롬프트
from metrics.prompts.dspy_judges_prompts import CONSTITUTION_PROMPT, FAITHFULNESS_PROMPT, STYLE_PROMPT


class ConstitutionJudge(dspy.Signature):
    __doc__ = CONSTITUTION_PROMPT
    
    answer = dspy.InputField()
    violation_detected = dspy.OutputField(desc="True or False")
    reason = dspy.OutputField(desc="Reason for violation if any")

class FaithfulnessJudge(dspy.Signature):
    __doc__ = FAITHFULNESS_PROMPT

    context = dspy.InputField()
    answer = dspy.InputField()
    is_faithful = dspy.OutputField(desc="True or False")
    reason = dspy.OutputField(desc="Explanation")

class StyleComplianceJudge(dspy.Signature):
    __doc__ = STYLE_PROMPT

    target_reference = dspy.InputField(desc="따라야 할 스타일이 담긴 모범 답안")
    model_answer = dspy.InputField(desc="모델이 생성한 답변")
    
    is_compliant = dspy.OutputField(desc="True if style matches, False otherwise")
    feedback = dspy.OutputField(desc="스타일이 어떻게 다른지 구체적인 피드백 (한국어)")