

import dspy
# 심판관 프롬프트
from metrics.prompts.dspy_judges_prompts import CONSTITUTION_PROMPT


class ConstitutionJudge(dspy.Signature):
    __doc__ = CONSTITUTION_PROMPT
    
    answer = dspy.InputField()
    violation_detected = dspy.OutputField(desc="True or False")
    reason = dspy.OutputField(desc="Reason for violation if any")