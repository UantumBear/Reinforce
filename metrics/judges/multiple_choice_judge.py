"""
@경로: metrics/judges/multiple_choice_judge.py
@설명: Multiple-choice 질문에 대한 정확도 평가 유틸리티
@용도: GPQA, MMLU, HQH 등 객관식 데이터셋 평가

일단 textgrad 논문에서 GPQA/MMLU/HQH 데이터셋에 대해 Test-time updates를 3번으로 설정하여,
생성된 답변들에 대해 다수결 투표(Majority Voting)를 하는 방식을 사용한다고 명시되어 있어서,
해당 evaluation 방식을 지원하기 위해, 
추후 Judge Class 를 만들 수도 있지만 일단은
함수들을 해당 위치에 모아둔다.

[주요 기능]
- 모델 답변에서 선택지(A, B, C, D) 추출
- Test-time updates 결과에 대한 다수결(majority voting) 투표
- Exact match 방식의 정확도(accuracy) 계산

[사용 예시]
    from metrics.judges.multiple_choice_judge import (
        extract_choice_from_answer,
        majority_vote,
        compute_accuracy
    )
    
    # 3번 답변 생성 (test-time updates)
    predictions = ["The answer is A", "I think B", "Answer: A"]
    choices = [extract_choice_from_answer(p) for p in predictions]
    
    # Majority voting
    final_choice = majority_vote(choices)  # "A"
    
    # Accuracy 계산
    accuracy = compute_accuracy(final_choice, "A")  # 1.0
"""

import re
from collections import Counter
from typing import List


def extract_choice_from_answer(answer: str) -> str:
    """
    모델 답변에서 선택지(A, B, C, D)를 추출합니다.
    
    @Args:
        answer: 모델이 생성한 답변 텍스트
    
    @Return:
        추출된 선택지 (A, B, C, D 중 하나) 또는 빈 문자열
    
    @Example:
        "The answer is B." -> "B"
        "I think the correct choice is (C)" -> "C"
        "A is the best option" -> "A"
    """
    if not answer:
        return ""
    
    answer_upper = answer.upper()
    
    # 패턴 1: "answer is X" 형태
    match = re.search(r'ANSWER\s+IS\s+([A-D])', answer_upper)
    if match:
        return match.group(1)
    
    # 패턴 2: "(X)" 괄호 형태
    match = re.search(r'\(([A-D])\)', answer_upper)
    if match:
        return match.group(1)
    
    # 패턴 3: "X." 또는 "X)" 형태
    match = re.search(r'\b([A-D])[.):]', answer_upper)
    if match:
        return match.group(1)
    
    # 패턴 4: 단순히 A, B, C, D만 있는 경우 (문장 시작 또는 독립적으로)
    match = re.search(r'\b([A-D])\b', answer_upper)
    if match:
        return match.group(1)
    
    return ""


def majority_vote(choices: List[str]) -> str:
    """
    선택지 리스트에서 다수결 투표를 수행합니다.
    
    @Args:
        choices: 선택지 리스트 (예: ["A", "B", "A"])
    
    @Return:
        가장 많이 나온 선택지. 동점이면 첫 번째 최다 선택지 반환
    
    @Example:
        ["A", "B", "A"] -> "A"
        ["A", "B", "C"] -> "A" (모두 동점이면 첫 번째)
    """
    if not choices:
        return ""
    
    # 빈 문자열 제거
    valid_choices = [c for c in choices if c]
    if not valid_choices:
        return ""
    
    # 가장 많이 나온 선택지 찾기
    counter = Counter(valid_choices)
    most_common = counter.most_common(1)[0][0]
    return most_common


def compute_accuracy(predicted_choice: str, correct_choice: str) -> float:
    """
    예측 선택지와 정답 선택지를 비교하여 정확도를 계산합니다.
    
    @Args:
        predicted_choice: 모델이 예측한 선택지 (A, B, C, D)
        correct_choice: 실제 정답 선택지 (A, B, C, D)
    
    @Return:
        정확하면 1.0, 틀리면 0.0
    
    @Example:
        compute_accuracy("A", "A") -> 1.0
        compute_accuracy("B", "A") -> 0.0
    """
    if not predicted_choice or not correct_choice:
        return 0.0
    
    return 1.0 if predicted_choice.upper() == correct_choice.upper() else 0.0
