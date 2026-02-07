import dspy

# 심판관 프롬프트
from metrics.prompts.dspy_judges_prompts import FAITHFULNESS_PROMPT

class FaithfulnessJudge(dspy.Signature):
    __doc__ = FAITHFULNESS_PROMPT

    context = dspy.InputField()
    answer = dspy.InputField()
    is_faithful = dspy.OutputField(desc="True or False")
    reason = dspy.OutputField(desc="Explanation")