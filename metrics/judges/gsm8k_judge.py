"""
@경로: metrics/judges/gsm8k_judge.py
@설명: GSM8k 데이터셋 전용 정확도 평가 Judge
@용도: 수학 문제 답변의 최종 숫자 추출 및 정확도 계산

[배경]
- GSM8k (Grade School Math 8K): 초등학교 수준의 수학 문제 데이터셋
- TextGrad 논문에 따르면 string-based exact match로 정확도 계산
- 모델 응답의 최종 숫자(final number)와 정답(ground-truth)을 비교

[답변 포맷]
GSM8k 데이터셋의 답변은 일반적으로 아래 형식을 따릅니다:
- "The answer is #### 1234" 
- "#### 42"
- 또는 마지막 줄에 숫자만 표시

[사용 예시]
    from metrics.judges.gsm8k_judge import compute_gsm8k_accuracy
    
    model_answer = "Let me solve this step by step... The answer is #### 1234"
    ground_truth = "1234"
    
    accuracy = compute_gsm8k_accuracy(model_answer, ground_truth)
    # Returns: 1.0 (정확), 0.0 (오답)
"""

import textgrad as tg


# ------------------------------------------------------------------------------------------
# 아래 함수는 텍스트에서 최종 숫자를 추출하기 위해 직접 구현한 함수이다 ------------------------------
# 
# import re
# from typing import Optional
# def extract_final_number(text: str) -> Optional[str]:
#     """
#     답변 텍스트에서 최종 숫자를 추출합니다.
    
#     @Args:
#         text: 모델이 생성한 답변 또는 정답 텍스트
    
#     @Return:
#         추출된 숫자 문자열 (쉼표/공백 제거됨). 숫자가 없으면 None
    
#     @추출 전략 (TextGrad 논문 기준):
#         1. "#### 숫자" 패턴 우선 검색 (GSM8k 표준 형식)
#         2. 텍스트 마지막 줄의 순수 숫자
#         3. 텍스트 전체에서 **마지막** 숫자 찾기 (논문 명시: "the last numerical value")
    
#     @정규화:
#         - 쉼표(,)와 공백 제거: "1,234" -> "1234"
#         - 소수점은 유지: "3.14" -> "3.14"
#         - 음수 기호 유지: "-42" -> "-42"
    
#     @Example:
#         "The answer is #### 1,234" -> "1234"
#         "So the total is 42 dollars." -> "42"
#         "#### -3.14" -> "-3.14"
    
#     @논문 출처:
#         TextGrad 논문 Evaluation 섹션:
#         "For GSM8k and Object Counting, we extract the last numerical value 
#         provided in the answer and compare it to the ground-truth answer."
#     """
#     if not text:
#         return None
    
#     # 전략 1: "#### 숫자" 형식 찾기 (GSM8k 표준)
#     # 패턴: #### 뒤에 나오는 숫자 (음수, 소수점 포함)
#     pattern_answer_marker = r'####\s*(-?\d[\d,\s]*\.?\d*)'
#     match = re.search(pattern_answer_marker, text)
#     if match:
#         number_str = match.group(1)
#         # 쉼표와 공백 제거
#         cleaned = number_str.replace(',', '').replace(' ', '')
#         return cleaned
    
#     # 전략 2: 텍스트 마지막 줄에서 숫자 찾기
#     # 예: "The final answer is:\n42" 같은 경우
#     lines = text.strip().split('\n')
#     if lines:
#         last_line = lines[-1].strip()
#         # 마지막 줄에서 숫자만 추출 (음수, 소수점 포함)
#         pattern_pure_number = r'^(-?\d[\d,\s]*\.?\d*)$'
#         match = re.match(pattern_pure_number, last_line)
#         if match:
#             number_str = match.group(1)
#             cleaned = number_str.replace(',', '').replace(' ', '')
#             return cleaned
    
#     # 전략 3: 텍스트 전체에서 마지막으로 나타난 숫자 찾기 (논문 방식)
#     # 예: "... and the total is 1,234 dollars"
#     # 모든 숫자를 찾아서 **마지막 것** 선택
#     pattern_all_numbers = r'-?\d[\d,\s]*\.?\d*'
#     all_matches = re.findall(pattern_all_numbers, text)
    
#     if all_matches:
#         # 마지막 숫자를 선택 (논문 명시: "the last numerical value")
#         last_number = all_matches[-1]
#         cleaned = last_number.replace(',', '').replace(' ', '')
#         return cleaned
    
#     return None


# def compute_gsm8k_accuracy(model_answer: str, ground_truth: str) -> float:
#     """
#     GSM8k 정확도를 계산합니다.
    
#     @Args:
#         model_answer: 모델이 생성한 답변
#         ground_truth: 실제 정답
    
#     @Return:
#         1.0: 정답과 일치
#         0.0: 정답과 불일치 또는 숫자 추출 실패
    
#     @계산 방식:
#         - TextGrad 논문: string-based exact match
#         - 양쪽 텍스트에서 최종 숫자 추출
#         - 추출된 숫자를 float로 변환하여 비교
#         - 완전 일치 시 1.0, 그 외 0.0
    
#     @Example:
#         compute_gsm8k_accuracy("#### 1234", "1234") -> 1.0
#         compute_gsm8k_accuracy("The answer is 42", "#### 42") -> 1.0
#         compute_gsm8k_accuracy("1,234", "1234") -> 1.0
#         compute_gsm8k_accuracy("42", "43") -> 0.0
#     """
#     # 양쪽에서 최종 숫자 추출
#     predicted_number = extract_final_number(model_answer)
#     correct_number = extract_final_number(ground_truth)
    
#     # 둘 중 하나라도 숫자 추출 실패 시 오답 처리
#     if predicted_number is None or correct_number is None:
#         return 0.0
    
#     # String-based exact match
#     # float 변환 후 비교 (부동소수점 오차 방지)
#     try:
#         pred_float = float(predicted_number)
#         correct_float = float(correct_number)
        
#         # 완전 일치 확인
#         # 부동소수점 비교이므로 아주 작은 오차 허용
#         if abs(pred_float - correct_float) < 1e-6:
#             return 1.0
#         else:
#             return 0.0
#     except (ValueError, OverflowError):
#         # 숫자 변환 실패 시 문자열 직접 비교
#         if predicted_number == correct_number:
#             return 1.0
#         else:
#             return 0.0


# def create_gsm8k_judge():
#     """
#     GSM8k Judge 인스턴스를 생성합니다.
    
#     @Note:
#         - GSM8k는 LLM 없이 순수 함수 기반으로 평가
#         - 따라서 별도의 Judge 객체가 필요 없으며 함수만 제공
#         - 일관성을 위해 create_*_judge() 패턴은 유지
    
#     @Return:
#         compute_gsm8k_accuracy 함수 자체를 반환
#     """
#     return compute_gsm8k_accuracy
# ------------------------------------------------------------------------------------------


# ------------------------------------------------------------------------------------------
# 아래는 실제 TextGrad 논문에서 GSM8k 평가에 사용된 정확도 계산 함수이다. 
# TODO 출처 찾기 
# ============================================================================
# [TextGrad 논문 재현] StringBasedFunction을 위한 GSM8k 평가 함수들
# ============================================================================
# 전체 파이프라인:
#   1. Model Forward: model(query) → prediction      [LLM 호출 O] 답변 생성
#   2. Eval Forward: eval_fn(pred, gt) → 0 or 1     [LLM 호출 X] 정답 체크 (Python)
#   3. Eval Backward: loss.backward() → gradient     [LLM 호출 O] 피드백 생성
#
# 비용 비교:
#   - StringBasedFunction: 2번 LLM 호출 (Model Forward + Eval Backward)
#   - TextLoss: 3번 LLM 호출 (Model Forward + Eval Forward + Eval Backward)
# 
# ============================================================================

def parse_integer_answer(answer: str, only_first_line: bool=False):
    """답변에서 숫자를 추출하는 함수 (TextGrad 논문 방식)"""
    try:
        if only_first_line:
            answer = answer.strip().split('\n')[0]
        answer = answer.strip()
        # 숫자가 포함된 마지막 토큰 찾기
        answer = [token for token in answer.split() if any(c.isdigit() for c in token)][-1]
        answer = answer.split('.')[0]
        answer = ''.join([c for c in answer if c.isdigit()])
        answer = int(answer)
    except (ValueError, IndexError):
        answer = 0
    return answer

def string_based_equality_fn(prediction: tg.Variable, ground_truth_answer: tg.Variable):
    """TextGrad 논문의 StringBasedFunction용 비교 함수
    
    Returns:
        int: 1 (정답) 또는 0 (오답)
    """
    pred_num = parse_integer_answer(str(prediction.value))
    gt_num = parse_integer_answer(str(ground_truth_answer.value))
    return int(pred_num == gt_num)