

import dspy
# 심판관 프롬프트
from metrics.prompts.dspy_judges_prompts import CONSTITUTION_PROMPT


class ConstitutionJudge(dspy.Signature):
    __doc__ = CONSTITUTION_PROMPT
    
    answer = dspy.InputField()
    violation_detected = dspy.OutputField(desc="True or False")
    reason = dspy.OutputField(desc="Reason for violation if any")


# 헌법 위반 사유 추출 헬퍼 함수
# 아래 함수는, 굳이 ContitutionJudge (헌법 심판관) 객체를 생성하지 않아도,
# 특정 데이터 행의 'critical_review' 필드에서 헌법 위반 사유를 추출할 수 있도록 돕습니다.
# 물론, [CRITICAL: Constitution Violation] 이와 같은 형태는 헌법 심판관에 의해 생성된다는 점을 전제로 합니다.

def _extract_violation_reason(critical_review: str) -> str:
    """
    critical_review에서 헌법 위반 사유를 추출하는 헬퍼 함수
    
    @param critical_review: 비판적 검토 텍스트
    @return: 헌법 위반 사유 또는 빈 문자열
    """
    if not critical_review or "[CRITICAL: Constitution Violation]" not in critical_review:
        return ""
    
    # "[CRITICAL: Constitution Violation]" 다음 부분을 추출
    try:
        parts = critical_review.split("[CRITICAL: Constitution Violation]", 1)
        if len(parts) > 1:
            return parts[1].strip()
    except Exception:
        pass
    
    return ""