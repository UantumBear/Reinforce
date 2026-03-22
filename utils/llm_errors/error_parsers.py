def extract_root_error_message(error: Exception) -> str:
    """
    @역할: tenacity.RetryError 내부의 실제 예외(BadRequestError 등)를 추출하는 함수.
    @설명: LLM API 호출 시 tenacity 라이브러리로 자동 재시도를 구현하면, 
          최종 실패 시 tenacity.RetryError가 발생하는데, 이 예외 객체 내부에 실제 원인 예외가 포함되어 있다.
    """
    try:
        # tenacity RetryError 케이스: last_attempt.exception()에 실제 원인이 있음
        if hasattr(error, "last_attempt"):
            inner_exc = error.last_attempt.exception()
            if inner_exc is not None:
                return str(inner_exc)
    except Exception:
        pass

    return str(error)