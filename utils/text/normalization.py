def normalize_text_field(value) -> str:
    """context/question/answer 필드를 문자열로 안전하게 정규화한다."""
    if value is None:
        return ""
    if isinstance(value, list):
        text = " ".join(str(item) for item in value)
    else:
        text = str(value)
    return " ".join(text.split())