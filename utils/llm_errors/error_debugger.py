"""
@경로: utils/llm_errors/error_debugger.py
@설명: TextGrad 최적화 중 Content Filter 디버깅을 위한 유틸리티 함수들
"""


def debug_individual_backward_samples(
    losses: list,
    episode: int,
    iteration: int,
    optimizer,
    extract_root_error_message_fn,
) -> bool:
    """
    개별 샘플별 Content Filter 테스트를 수행합니다.
    
    @목적:
    - 배치 내 어떤 샘플이 Azure Content Filter를 유발하는지 식별
    - 각 loss에 대해 개별적으로 backward()를 호출하여 필터 통과 여부 확인
    - 문제가 되는 샘플의 Loss Feedback과 Gradient Text 출력
    
    @param losses: TextGrad loss 객체 리스트
    @param episode: 현재 episode 번호
    @param iteration: 현재 iteration 번호
    @param optimizer: TextGrad optimizer 객체 (zero_grad() 호출용)
    @param extract_root_error_message_fn: 에러 메시지 추출 함수
    
    @return: True면 디버깅 모드로 iteration 스킵 필요, False면 정상 진행
    
    @사용예:
    ```python
    from utils.llm_errors.error_debuger import debug_individual_backward_samples
    from utils.llm_errors.error_parsers import extract_root_error_message
    
    if DEBUG_INDIVIDUAL_BACKWARD:
        should_skip = debug_individual_backward_samples(
            losses, episode, iteration, optimizer, extract_root_error_message
        )
        if should_skip:
            continue
    ```
    """
    print(f"\n{'='*80}")
    print(f"[디버깅 모드] Episode {episode}, Iteration {iteration}: 개별 샘플 필터 테스트")
    print(f"{'='*80}")
    
    for i, loss_item in enumerate(losses):
        try:
            # 개별 loss에 대해 backward() 호출하여 필터 통과 여부 확인
            loss_item.backward()
            print(f"  ✅ [Sample {i+1}/{len(losses)}] 필터 통과")
            optimizer.zero_grad()  # 테스트 후 gradient 초기화 (다음 샘플 테스트를 위해)
            
        except Exception as e:
            root_error = extract_root_error_message_fn(e)
            
            # Content Filter 에러 감지
            if "content filter" in root_error.lower() or "content management policy" in root_error.lower():
                print(f"\n❌ ❌ ❌ [범인 검거!] Sample {i+1}/{len(losses)}번이 Content Filter를 유발했습니다.")
                print(f"\n--- 문제의 Loss Feedback 내용 ---")
                print(loss_item.value)
                print(f"--- (길이: {len(loss_item.value)} chars) ---\n")
                
                # Gradient Text가 있으면 출력
                if hasattr(loss_item, 'get_gradient_text'):
                    grad_text = loss_item.get_gradient_text()
                    if grad_text:
                        print(f"--- Gradient Text ---")
                        print(grad_text[:500])  # 처음 500자만
                        print(f"...\n")
            else:
                # 기타 에러
                print(f"  ⚠️ [Sample {i+1}/{len(losses)}] 기타 에러: {root_error[:100]}")
            
            # 에러 발생해도 계속 테스트 진행
            optimizer.zero_grad()
    
    print(f"{'='*80}")
    print(f"[디버깅 완료] 이번 iteration은 디버깅만 수행하고 실제 업데이트는 건너뜁니다.\n")
    optimizer.zero_grad()
    
    # True를 반환하여 caller에게 iteration 스킵하도록 알림
    return True
