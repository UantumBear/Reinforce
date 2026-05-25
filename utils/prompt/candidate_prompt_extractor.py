from dataclasses import dataclass
import re


@dataclass
class CandidatePromptExtractionResult:
    """
    후보 프롬프트 추출 결과를 담는 데이터 클래스
    """
    candidate_text: str
    matched_pattern: str | None
    success: bool
    pattern_match_logs: list[str]


class CandidatePromptExtractor:
    """
    후보 프롬프트를 추출하는 클래스
    """
    def __init__(self):
        self.patterns = [
            r"<new_variable>(.*?)</new_variable>",
            r"<IMPROVED_VARIABLE>(.*?)</IMPROVED_VARIABLE>",
            r"<refined_template>(.*?)</refined_template>",
            r"<OPTIMIZER_WRITING_TEXT_START>(.*?)<OPTIMIZER_WRITING_TEXT_END>",
            r"```(.*?)```",
        ]

    def extract(
        self,
        optimizer_response_text: str,
        fallback_prompt: str,
    ) -> CandidatePromptExtractionResult:
        actual_candidate_text = None
        matched_pattern = None
        pattern_match_logs = []

        # STEP 1. 정의된 패턴을 순서대로 돌며 후보 프롬프트 추출을 시도한다.
        for pattern in self.patterns:
            match = re.search(pattern, optimizer_response_text, re.DOTALL | re.IGNORECASE)

            if not match:
                pattern_match_logs.append(f"❌ {pattern}: 매칭 실패")
                continue

            candidate = match.group(1).strip()
            rejected_reason = []

            # STEP 2. 추출된 텍스트가 실제 후보로 쓸 수 있는지 최소 조건을 검사한다.
            if not candidate:
                rejected_reason.append("빈 문자열")
            if "{" in candidate:
                rejected_reason.append("중괄호 포함")
            if "the improved variable" in candidate.lower():
                rejected_reason.append("placeholder 텍스트")

            if rejected_reason:
                pattern_match_logs.append(
                    f"⚠️ {pattern}: 매칭되었으나 거부됨 ({', '.join(rejected_reason)})"
                )
                continue

            actual_candidate_text = candidate
            matched_pattern = pattern
            pattern_match_logs.append(f"✅ {pattern}: 매칭 성공 & 사용됨")
            break

        # STEP 3. 끝까지 유효한 후보를 못 찾으면 현재 프롬프트를 fallback 으로 사용한다.
        success = actual_candidate_text is not None

        if not success:
            actual_candidate_text = fallback_prompt

        # STEP 4. 메인 로직이 그대로 사용할 수 있도록 추출 결과를 구조화해 반환한다.
        return CandidatePromptExtractionResult(
            candidate_text=actual_candidate_text,
            matched_pattern=matched_pattern,
            success=success,
            pattern_match_logs=pattern_match_logs,
        )