"""
@경로: metrics.judges.dspy_faithfulness_judge
- 26.02.08 현재 미사용.
- 단순히 LLM 에게 Failthfullness 를 "평가해줘" 라는 식으로 평가하는 것은 신뢰할 수 없음.
- 따라서 RAGAS 기반의 Faithfulness Judge 를 별도로 구현하여 사용해보려고 함.

"""

# 소스를 다시 사용하고자 한다면 아래 부분 주석 해제
"""
import dspy

# 심판관 프롬프트
from metrics.prompts.dspy_judges_prompts import FAITHFULNESS_PROMPT

class FaithfulnessJudge(dspy.Signature):
    __doc__ = FAITHFULNESS_PROMPT

    context = dspy.InputField()
    answer = dspy.InputField()
    is_faithful = dspy.OutputField(desc="True or False")
    reason = dspy.OutputField(desc="Explanation")

"""