import re
from utils.text.normalization import normalize_text_field

# 함수 1. 콘텐트 필터링 여부 판별 함수
def is_azure_content_filter_error(error_message: str) -> bool:
    """Azure OpenAI 콘텐츠 필터 차단 메시지 여부를 판별한다."""
    normalized = (error_message or "").lower()
    filter_signatures = [
        "content_filter",
        "responsibleaipolicyviolation",
        "safety system",
        "jailbreak",
        "violence",
        "self_harm",
        "sexual",
        "hate",
    ]
    return any(signature in normalized for signature in filter_signatures)

# 함수 2. 프롬프트 인젝션/탈옥 패턴 감지 함수
def has_jailbreak_like_pattern(text: str) -> bool:
    """프롬프트 인젝션/탈옥으로 오인될 수 있는 패턴을 감지한다."""
    normalized = (text or "").lower()
    patterns = [
        "ignore previous instructions",
        "disregard all",
        "you are now",
        "developer mode",
        "prompt injection",
        "시스템 프롬프트를 무시",
        "이전 지시를 무시",
        "규칙을 무시",
    ]
    return any(pattern in normalized for pattern in patterns)

# 함수 3. Azure 필터 회피를 위한 프롬프트 완화 함수
def sanitize_for_azure_filter(value, max_chars: int) -> str:
    """Azure 필터 민감도를 낮추기 위해 과도한 길이/명시적 프롬프트 인젝션 문구를 완화한다."""
    text = normalize_text_field(value)
    text = re.sub(r"ignore\s+previous\s+instructions", "[redacted-instruction]", text, flags=re.IGNORECASE)
    text = re.sub(r"system\s+prompt", "system-guidance", text, flags=re.IGNORECASE)
    text = re.sub(r"(이전\s*지시\s*무시|시스템\s*프롬프트)", "[완화됨]", text)
    if len(text) > max_chars:
        return text[:max_chars] + " ...[truncated]"
    return text