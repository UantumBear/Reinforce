"""
@경로: main_textgrad_baseline.py
@설명: TextGrad를 활용한 프롬프트 최적화 및 DB 저장
- TextGrad 의 기본 논문 을 재현한다.
- 추후 내가 설계한 다양한 Judge 모델과, 보상함수와도 연동할 수 있도록 구조화한다.
- TextGrad의 'Step'을 'Episode' 컬럼으로 매핑하여, 각 최적화 단계별로 상세 로그를 DB에 저장한다.
- 일단 내 연구의 최종 프레임워크가 어떻게 될 지는 모르겠으나,
  TextGrad baseline 과, TextGrad + Judges+Reward 의 차이를 비교하고,
  또 다른 OPRO 논문과의 비교도 추가하여,
  각 연구논문 들 baseline 과, 그 상태에서 계층적 피드백을 주었을때의 차이를 비교하면 좋을 것 같다.

- 2026.03.07 데이터 로그 적재 부 추가
NULL(값 없음):
    샘플 스킵(인젝션 패턴) 시 점수 필드
    Judge 초기화 자체 실패로 해당 점수를 계산할 수 없는 경우
NaN(계산 망가짐):
    샘플 처리 예외로 평가가 깨진 경우
    RAGAS 평가가 실행됐지만 내부 Evaluation error/예외가 난 경우
    (유사도 호출에서 예외 발생 시도 NaN)

[중요] context vs <CONTEXT> 태그의 차이
====================================
이 코드에서 "context"라는 용어가 두 가지 다른 의미로 사용됩니다:

1. **데이터의 context** (RAG 문서 자료)
   - 변수명: context, val_context
   - 의미: RAG 챗봇에서 LLM에게 제공하는 문서 자료, 배경 정보
   - 예시: NASA 센서 로그, KLUE 검색 결과, 뉴스 기사 등
   - GSM8k 같은 수학 문제는 context가 없음 (빈 문자열)
   - 사용처: Forward Model 입력 (답변 생성 LLM)
   
2. **TextGrad optimizer의 <CONTEXT> 태그** (이전 최적화 피드백)
   - 의미: Optimizer LLM에게 주는 "이전 최적화 시도의 피드백 이력"
   - 내용: "이전에 이런 프롬프트로 이런 문제를 풀었더니 이런 오답과 피드백이 나왔다"
   - TextGrad 라이브러리가 자동으로 관리 (backward()로 생성된 gradient를 채움)
   - 사용처: Optimizer LLM이 프롬프트를 개선할 때 참고
   - 이 코드에서는 직접 조작하지 않음 (라이브러리 내부 처리)

두 개념은 완전히 다르며, 혼동하지 않도록 주의가 필요합니다.
====================================
"""
DEBUG_INDIVIDUAL_BACKWARD = False # 디버그 모드

import os
import re
import math

from datetime import datetime
import textgrad as tg
from textgrad.optimizer.optimizer import TextualGradientDescentwithMomentum
from utils.environment.experiment import TextGradExperiment
from utils.environment.textgrad_log_builder import (
    create_base_log,
    create_skip_log,
    create_success_log,
    create_error_log
)
from utils.log.console import print_step

from datafile.data_loader import load_dataset
from infrastructure.llm_client import get_textgrad_backward_engine, get_textgrad_forward_engine

from metrics.judges.ragas_failthfulness_judge import RagasFaithfulnessJudge
from conf.config import Settings
# 로그 저장을 위한 import (main_train.py 방식)
from models.rl_optimization_log import RlOptimizationLog
from db.connection.pg_client import pg_client

# 공통으로 사용 가능 한 utils 함수들
from utils.llm_errors.error_parsers import extract_root_error_message
from utils.llm_errors.error_debugger import debug_individual_backward_samples
from utils.llm_safety.azure_prompt_filters import has_jailbreak_like_pattern
from utils.text.normalization import normalize_text_field
from utils.llm_patches.textgrad_patches import patch_textgrad_openai_compatibility, patch_textgrad_momentum_compatibility
from utils.llm_patches.textgrad_info import get_tgd_optimizer_system_prompt, stringify_tgd_update_prompt


# 기타 LLM get 함수들
from metrics.judges.similarity_judge import create_similarity_judge 
from metrics.judges.ragas_failthfulness_judge import create_ragas_judge
from metrics.prompts.textgrad_baseline_prompts import build_azure_safe_optimizer_system_prompt

# Multiple-choice 평가 유틸리티 (GPQA/MMLU/HQH용)
from metrics.judges.multiple_choice_judge import (
    extract_choice_from_answer,
    majority_vote,
    compute_accuracy
)

# GSM8k 평가 유틸리티 (수학 문제 데이터셋용)
from metrics.judges.gsm8k_judge import compute_gsm8k_accuracy


def extract_momentum_history(optimizer, variable: tg.Variable, momentum_storage_idx: int = 0) -> str:
    """
    Momentum optimizer의 과거 gradient 누적 이력을 로그용 문자열로 추출한다.
    
    @목적: 
    - momentum_storage에 저장된 과거 step들의 value와 gradient를 명시적으로 가시화
    - optimizer_total_input에 포함시켜 분석 시 momentum 효과를 추적 가능하게 함
    
    @Return:
        str: 과거 step별 프롬프트 값과 gradient를 구조화한 문자열
    """
    if not isinstance(optimizer, TextualGradientDescentwithMomentum):
        return "[N/A] This optimizer does not use momentum."
    
    if not hasattr(optimizer, 'momentum_storage'):
        return "[N/A] momentum_storage attribute not found in optimizer."
    
    try:
        past_n_steps = optimizer.momentum_storage[momentum_storage_idx]
    except (IndexError, KeyError, TypeError):
        return "[N/A] momentum_storage is empty or index out of range."
    
    if not past_n_steps:
        return "[Empty] No momentum history accumulated yet (first episode)."
    
    history_lines = []
    history_lines.append(f"=== MOMENTUM HISTORY (window size: {len(past_n_steps)}) ===")
    history_lines.append("아래는 과거 episode들에서 누적된 프롬프트 변화와 gradient 피드백입니다.")
    history_lines.append("Optimizer는 이 이력을 참고하여 다음 프롬프트를 생성합니다.\n")
    
    for i, step_info in enumerate(past_n_steps):
        step_value = step_info.get('value', '[N/A]')
        step_gradient = step_info.get('gradient', '[N/A]')
        
        history_lines.append(f"--- Past Step {i+1} ---")
        history_lines.append(f"Prompt Value at Step {i+1}:")
        history_lines.append(f"{step_value}")
        history_lines.append(f"\nGradient Feedback at Step {i+1}:")
        history_lines.append(f"{step_gradient}")
        history_lines.append("")  # 빈 줄로 구분
    
    return "\n".join(history_lines)


def build_tgd_optimizer_total_input(
    optimizer_system_prompt: str,
    instruction: str,
    question: str,
    context: str,  # ← RAG 문서 자료 (데이터의 context, TextGrad <CONTEXT> 태그 아님)
    gold_answer: str,
    model_answer: str | None,
    evaluation_instruction: str | None,
    answer_feedback: str | None,
) -> str:
    """
    [에러/스킵 케이스 전용] TGD optimizer 입력을 추정 형식으로 재구성한다.
    실제 optimizer 가 정상 실행되었을 때는 거기서 뽑아오면 되지만, 중간에 에러가 난 경우에는,
    어디까지 프롬프트가 구성되어있는지 확인 하기 어렵기 때문에,
    이 함수를 구현한 것이다. 

    26.03.29
    현재 baseline에서는 정의만 되어있고 실제로 사용하지 않는다. 
    improve 버전에서 에러/스킵 케이스의 로깅을 강화할 때 사용하고 있다. 
    
    @주의:
    - 이 함수는 optimizer가 실제로 실행되지 않은 에러/스킵 케이스에서만 사용됩니다.
    - 정상 처리 시에는 capture_optimizer_update_prompt()로 실제 optimizer 프롬프트를 저장합니다.
    - 출력 문자열 첫 줄에 "[TextGrad TGD Optimizer Input - Estimated Format]"을 명시하여
      이것이 실제 optimizer 입력이 아닌 추정값임을 표시합니다.
    
    @파라미터 주의사항:
    - context: RAG 문서 자료 (데이터의 context)
      ※ TextGrad optimizer의 <CONTEXT> 태그(이전 최적화 피드백)와는 다릅니다.
      ※ <CONTEXT> 태그는 TextGrad 라이브러리가 자동으로 관리하며, 이 함수와 무관합니다.
    
    @용도:
    - 샘플 스킵 (jailbreak 패턴 감지)
    - 샘플 처리 예외 (Azure content filter, 평가 실패 등)
    - DB 분석 시 "만약 정상 처리됐다면 이런 입력이 들어갔을 것"이라는 컨텍스트 제공
    """
    safe_model_answer = model_answer if model_answer is not None else "[N/A]"
    safe_eval_instruction = evaluation_instruction if evaluation_instruction is not None else "[N/A]"
    safe_answer_feedback = answer_feedback if answer_feedback is not None else "[N/A]"

    # [로깅용 입력 구성]
    # context: RAG 문서 자료 (데이터의 context)
    # - GSM8k처럼 context가 없는 데이터셋은 생략
    # - NASA/KLUE처럼 context가 있는 데이터셋은 포함
    # ※주의: 이것은 로깅용이며, TextGrad optimizer의 <CONTEXT> 태그와는 무관합니다.
    if context.strip():
        sample_input = f"question: {question}\ncontext: {context}\ngold_answer: {gold_answer}\n\n"
    else:
        sample_input = f"question: {question}\ngold_answer: {gold_answer}\n\n"

    return (
        "=" * 80 + "\n"
        "[중요] 이 내용은 실제 optimizer에게 전달된 문자열이 아닙니다.\n"
        "샘플 처리 실패(에러/스킵)로 인해 optimizer가 실행되지 않아,\n"
        "실제 optimizer 입력을 추출할 수 없는 상황에서 사용자 정의 함수로 추정한 형식입니다.\n"
        "정상 처리 시에는 optimizer._update_prompt() + stringify_tgd_update_prompt()로 실제 입력을 기록합니다.\n"
        "=" * 80 + "\n\n"
        "[TextGrad TGD Optimizer Input - Estimated Format]\n\n"
        "=== OPTIMIZER SYSTEM PROMPT ===\n"
        f"{optimizer_system_prompt}\n\n"
        "=== OPTIMIZATION TARGET (instruction) ===\n"
        f"{instruction}\n\n"
        "=== SAMPLE INPUT ===\n"
        f"{sample_input}"
        "=== FORWARD OUTPUT ===\n"
        f"model_answer: {safe_model_answer}\n\n"
        "=== LOSS EVALUATION INSTRUCTION ===\n"
        f"{safe_eval_instruction}\n\n"
        "=== ANSWER-LEVEL FEEDBACK (TextLoss output) ===\n"
        f"{safe_answer_feedback}\n"
    )


def main():
    print_step("0. [Settings] TextGrad 실험 환경 설정")
    # 패치를 명시적으로 먼저 적용
    TextGradExperiment.apply_patches()  # ← 여기서만 실행
    EXPERIMENT_INS = TextGradExperiment(mode='baseline')
    
    print_step("1. [Settings] 기본 백엔드 설정 초기화")
    Settings.setup()
    

    print_step("2. 데이터 로드 및 Train/Validation 분할")
    dataset, train_pool, validation_dataset = EXPERIMENT_INS.load_and_split_data()
    
    # [TextGrad 논문] Test-time updates 설정 (데이터셋별 자동 최적화)
    # - GPQA/MMLU/HQH: 3번 답변 생성 + Majority Voting (multiple-choice)
    # - 그 외 데이터셋: 1번 생성 (일반 RAG/생성 태스크)
    test_time_updates = EXPERIMENT_INS.get_test_time_updates()
    print(f"[✓] Test-time updates: {test_time_updates}번 (데이터셋: {EXPERIMENT_INS.default_dataset_name})")
    
    print_step("3. TextGrad 환경 설정 및 엔진 초기화")
    # TextGrad baseline용 experiment_id 생성
    current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
    experiment_id = f"textgrad_baseline_{current_time}"
    
    # TextGrad 엔진 초기화: Forward(전방 생성) / Backward(역전파 피드백) 분리
    # get_textgrad_*_engine()은 (engine, model_name) 튜플을 반환:
    #   - engine: 실제 LLM API를 호출하는 실행 객체 (예: ChatOpenAI 인스턴스)
    #   - model_name: 사용된 모델명 문자열 (예: "gpt-5.4-mini") - DB 로그 기록용
    forward_engine, textgrad_forward_model_nm = get_textgrad_forward_engine()
    backward_engine, textgrad_backward_model_nm = get_textgrad_backward_engine()
    
    # forward_engine: 답변 생성용 LLM 엔진 (tg.BlackboxLLM에서 사용)
    # backward_engine: 피드백 생성용 LLM 엔진 (TextGrad의 "비평가/교사" 역할)
    #   - loss.backward() 시 생성된 피드백(gradient text)을 바탕으로 
    #   - optimizer가 프롬프트를 어떻게 개선할지 판단할 때 사용
    tg.set_backward_engine(backward_engine)
    
    print_step("4. TextGrad 최적화 실행")
    similarity_judge = create_similarity_judge()
    ragas_judge = create_ragas_judge()
    
    # 데이터셋 타입 감지 (accuracy 계산용)
    dataset_name_lower = EXPERIMENT_INS.default_dataset_name.lower()
    is_multiple_choice = any(keyword in dataset_name_lower for keyword in ['gpqa', 'mmlu', 'hqh'])
    is_gsm8k = 'gsm8k' in dataset_name_lower
    
    print(f"[✓] 데이터셋 타입: multiple_choice={is_multiple_choice}, gsm8k={is_gsm8k}")
    
    # [연구 로드맵] 현재는 TextGrad Baseline 재현 단계
    # 향후 발전 방향: tg.TextLoss(평가 지시문 문자열) 대신
    # → CaseAwareJudgeLoss() 클래스로 교체 (8가지 기업용 RAG 지표 평가)
    #   - Faithfulness, Relevancy, Completeness, Conciseness 등
    #   - 각 Judge는 구조화된 JSON 형태로 평가 결과 반환
    # judge_loss_fn = CaseAwareJudgeLoss()  # TODO: 다음 단계 구현

    # 3. 최적화 대상 정의
    # role_description은 [지금 고쳐야 할 대상(변수)의 정체]를 말한다.
    # initial_prompt는 데이터셋에 따라 TextGradExperiment에서 자동 설정
    initial_prompt = EXPERIMENT_INS.get_initial_prompt()
    print(f"[✓] 초기 프롬프트: {initial_prompt}..." if len(initial_prompt) > 100 else f"[✓] 초기 프롬프트: {initial_prompt}")
    # 아래 system_prompt 는 Forward Engeind (=ServiceLLM) 에게 전달되는 시스템 프롬프트.
    system_prompt = tg.Variable(
        initial_prompt, 
        requires_grad=True, 
        role_description="system prompt to the language model"
    )

    # [중요] Prompt Optimization 경로
    # engine.generate()를 직접 호출하면 system_prompt와 계산 그래프가 연결되지 않아
    # get_gradient_text()가 비어 있을 수 있다.
    # 따라서 TextGrad 권장 방식인 BlackboxLLM(system_prompt=...)을 사용한다.
    model = tg.BlackboxLLM(engine=forward_engine, system_prompt=system_prompt)

    momentum_window = int(
        os.getenv("TEXTGRAD_MOMENTUM_WINDOW", os.getenv("TEXTGRAD_MOMENTUM_GRADIENT_MEMORY", "3"))
    )

    # 일반 옵티마이저 (비교/회귀 확인용)
    # optimizer = tg.TGD(
    #     parameters=list(model.parameters()),
    #     engine=backward_engine,
    #     gradient_memory=momentum_window,
    # )

    # 모멘텀 적용 옵티마이저 (논문 재현 경로)
    # [Azure Content Filter 회피 전략]
    # optimizer_system_prompt를 커스터마이징하여 건전한 최적화 가이드라인 제공

    # TODO 아래 custom 프롬프트는 improve 실험 모드에서 사용할 것
    # custom_optimizer_system_prompt = build_azure_safe_optimizer_system_prompt()
    
    optimizer = TextualGradientDescentwithMomentum(
        parameters=list(model.parameters()),
        engine=backward_engine,
        momentum_window=momentum_window,
        # optimizer_system_prompt=custom_optimizer_system_prompt,
    )
    # optimizer(TGD): TextGrad의 텍스트 경사하강 업데이트기.
    # backward에서 나온 피드백을 입력으로 받아, 최적화 대상 변수(system_prompt.value)를 한 step씩 실제로 갱신한다.
    optimizer_system_prompt = get_tgd_optimizer_system_prompt(optimizer)
    # 위 시스템 프롬프트 기본 버전에는 (라이브러리)
    #  <IMPROVED_VARIABLE> 이 태그 안에 응답을 생성해서 넣으라고 되어있음.

    # 4. 최적화 루프 - TextGrad 논문 설정
    # [TextGrad 논문 재현 설정]
    # TextGradExperiment에서 데이터셋별로 자동 설정된 값 사용:
    # - GSM8k: Train 200 / Val 300, Iterations 12, Batch 3
    # - Object Counting & Word Sorting: Train 50 / Val 100, Iterations 12, Batch 3
    # - 기타: 기본값 또는 환경변수
    
    episodes = EXPERIMENT_INS.default_episodes
    iterations_per_episode = EXPERIMENT_INS.default_iterations
    batch_size = EXPERIMENT_INS.default_batch_size
    
    optimization_logs = []  # DB 저장용 로그 버퍼
    # [TODO] 새로운 컬럼 필요 시 아래 주석 참고:
    # - iteration (int): Episode 내 iteration 번호 (1~12)
    # - validation_score_current (float): 현재 프롬프트의 validation 점수
    # - validation_score_candidate (float): 후보 프롬프트의 validation 점수
    # - prompt_accepted (boolean): 프롬프트 업데이트 여부
    # - sample_type (string): 'train' or 'validation'

    print(f"--- TextGrad Baseline Optimization (논문 재현) 시작 ---")
    print(f"Episodes: {episodes}, Iterations/Episode: {iterations_per_episode}, Batch size: {batch_size}")


    # ========== [TextGrad 논문 재현 루프 시작] ==========
    for episode in range(1, episodes + 1):
        print(f"\\n{'='*80}")
        print(f"Episode {episode}/{episodes} 시작")
        print(f"{'='*80}")
        
        # Episode 공통 필드 생성 (모든 샘플에 재사용)
        base_log = create_base_log(experiment_id, episode, textgrad_backward_model_nm, textgrad_forward_model_nm)
        
        episode_log_start_idx = len(optimization_logs)
        
        for iteration in range(1, iterations_per_episode + 1):
            print(f"\\n--- Iteration {iteration}/{iterations_per_episode} ---")
            
            # 1) Train: 배치 크기만큼 무작위 복원 추출
            import random
            batch = random.choices(train_pool, k=batch_size)  # 복원 추출
            
            losses = []
            iteration_log_start_idx = len(optimization_logs)
            
            # Train 샘플 처리
            for data in batch:
                # [데이터 추출] 아래 변수들은 RAG 챗봇용 입력 데이터입니다
                # - context: RAG 문서 자료 (GSM8k는 빈 문자열, NASA/KLUE는 실제 문서)
                # - question: 사용자 질문
                # - ground_truth: 정답
                # ※주의: 여기의 'context'는 TextGrad optimizer의 <CONTEXT> 태그와 무관합니다.
                #   <CONTEXT> 태그는 "이전 최적화 피드백"을 담으며, TextGrad 라이브러리가 자동 관리합니다.
                context = normalize_text_field(data.get('context', ''))
                question = normalize_text_field(data.get('question', ''))
                ground_truth = normalize_text_field(data.get('answer', ''))
                
                if has_jailbreak_like_pattern(context) or has_jailbreak_like_pattern(question) or has_jailbreak_like_pattern(ground_truth):
                    # Jailbreak 패턴 감지 - 스킵
                    optimization_logs.append(create_skip_log(
                        base_log, system_prompt.value, question, context, ground_truth
                    ))
                    continue
                
                try:
                    # [TextGrad 논문] Test-time updates: 데이터셋별 최적화
                    # GPQA/MMLU/HQH: 3번 생성 + Majority Voting
                    # 그 외 데이터셋: 1번 생성 (일반 RAG)
                    
                    # [Forward Model 입력 구성]
                    # context: RAG 문서 자료 (데이터셋에 따라 있거나 없음)
                    # - GSM8k: context 없음 (수학 문제만 있음) → "Question: ..."
                    # - NASA/KLUE: context 있음 (문서 자료) → "Context: ...\nQuestion: ..."
                    # ※주의: 이것은 Forward Model(답변 생성 LLM)에게 주는 입력입니다.
                    #   TextGrad Optimizer의 <CONTEXT> 태그(이전 피드백)와는 다릅니다.
                    if context.strip():
                        inputs = f"Context: {context}\nQuestion: {question}"
                    else:
                        inputs = f"Question: {question}"
                    
                    query_var = tg.Variable(inputs, role_description="RAG 입력", requires_grad=False)
                    
                    # test_time_updates 횟수만큼 답변 생성
                    test_time_predictions = []
                    test_time_choices = []
                    first_prediction_var = None
                    
                    for update_idx in range(test_time_updates):
                        pred_var = model(query_var)
                        pred_text = pred_var.value
                        test_time_predictions.append(pred_text)
                        
                        # 첫 번째 예측 변수를 gradient 계산용으로 보관
                        if update_idx == 0:
                            first_prediction_var = pred_var
                        
                        # 선택지 추출 (A, B, C, D) - multiple-choice 데이터셋용
                        choice = extract_choice_from_answer(pred_text)
                        test_time_choices.append(choice)
                    
                    # 첫 번째 예측을 대표 예측으로 사용
                    prediction_var = first_prediction_var
                    prediction = test_time_predictions[0]
                    
                    # Accuracy 계산 (데이터셋별 전략)
                    # 1. Multiple-choice (GPQA/MMLU/HQH): test_time_updates=3, majority voting
                    # 2. GSM8k (수학): test_time_updates=1, 최종 숫자 추출 비교
                    # 3. 그 외: accuracy 계산 안 함
                    accuracy = None
                    
                    if is_multiple_choice and test_time_updates > 1:
                        # Multiple-choice: Majority voting으로 최종 답 선택
                        final_choice = majority_vote(test_time_choices)
                        
                        # 정답에서도 선택지 추출
                        correct_choice = extract_choice_from_answer(ground_truth)
                        
                        # Exact match accuracy 계산
                        accuracy = compute_accuracy(final_choice, correct_choice)
                        
                        print(f"[Accuracy - Multiple Choice] Predictions: {test_time_choices} -> Final: {final_choice}, Correct: {correct_choice}, Acc: {accuracy}")
                    
                    elif is_gsm8k:
                        # GSM8k: 최종 숫자 추출하여 비교
                        accuracy = compute_gsm8k_accuracy(prediction, ground_truth)
                        
                        print(f"[Accuracy - GSM8k] Model: {prediction[:100]}..., Ground Truth: {ground_truth[:100]}..., Acc: {accuracy}")
                    
                    # Gradient 생성 (TextGrad 논문 방식: 엄격한 페르소나 부여)
                    # 논문의 Solution Refinement 평가 프롬프트를 RAG 도메인에 맞게 수정
                    # [정의] evaluation_instruction: 비평가 TeacherLLM(백워드엔진) 에게 건네는 채점 가이드 라인이다.
                    # INFO:  비판 가이드라인 설정
                    # TODO: 이건 되는데..
                    evaluation_instruction = (
                        "You are a critical and rigorous evaluator for RAG systems. "
                        "Your task is to examine the predicted answer step-by-step and identify potential flaws.\n\n"
                        f"**Reference Answer:** {ground_truth}\n\n"
                        "**Evaluation Criteria:**\n"
                        "1. Does the prediction fully address the question based on the given context?\n"
                        "2. Are there any factual inaccuracies or hallucinations?\n"
                        "3. Is the reasoning clear and logically sound?\n"
                        "4. What specific improvements would make this answer better?\n\n"
                        "Provide concise, actionable feedback focused on how to improve the answer generation prompt."
                    )
                    loss = tg.TextLoss(evaluation_instruction) # (Teacher)에게 "이런 기준으로 채점해!"라고 지시.
                    computed_loss = loss(prediction_var) # 학생(Tester)이 응답을 낸다. 
                    losses.append(computed_loss)
                    
                    # 점수 계산
                    raw_similarity = None
                    if similarity_judge is not None:
                        try:
                            raw_similarity = similarity_judge(ground_truth, prediction)
                            print(f"[Debug] Raw similarity score: {raw_similarity}")
                        except Exception:
                            raw_similarity = math.nan
                    
                    ragas_faithfulness_score = None
                    ragas_answer_relevancy_score = None
                    if ragas_judge is not None:
                        try:
                            ragas_result = ragas_judge.evaluate(question=question, answer=prediction, context=context, gold_answer=ground_truth)
                            if not str(ragas_result.get('reason', '')).startswith("Evaluation error:"):
                                ragas_faithfulness_score = ragas_result.get('score')
                                ragas_answer_relevancy_score = ragas_result.get('relevancy_score')
                            else:
                                ragas_faithfulness_score = math.nan
                                ragas_answer_relevancy_score = math.nan
                        except Exception:
                            ragas_faithfulness_score = math.nan
                            ragas_answer_relevancy_score = math.nan
                    
                    # Train 샘플 로그 저장
                    optimization_logs.append(create_success_log(
                        base_log, system_prompt.value, question, context, ground_truth,
                        prediction, computed_loss.value, raw_similarity,
                        ragas_faithfulness_score, ragas_answer_relevancy_score,
                        optimizer_system_prompt, accuracy
                    ))
                    
                except Exception as sample_error:
                    root_error = extract_root_error_message(sample_error)
                    # 샘플 처리 예외 - 에러 로그 기록
                    optimization_logs.append(create_error_log(
                        base_log, system_prompt.value, question, context, ground_truth, root_error
                    ))
                    continue
            
            # 2) Gradient 계산 (TextGrad 논문 방식)
            if not losses:
                print(f"[Warning] Iteration {iteration}: 유효한 loss 없음, 업데이트 건너뜀")
                continue
            
            # ====================================== 디버깅 ====================================== #
            # [디버깅] 개별 샘플별 Content Filter 테스트
            if DEBUG_INDIVIDUAL_BACKWARD:
                should_skip = debug_individual_backward_samples(
                    losses=losses,
                    episode=episode,
                    iteration=iteration,
                    optimizer=optimizer,
                    extract_root_error_message_fn=extract_root_error_message
                )
                if should_skip:
                    continue
            # ====================================== 디버깅 ====================================== #
            
            # (Code Snippet 2) 논문 본문에 수록된 기본 프롬프트 최적화 코드
            # 논문의 핵심: tg.sum()으로 배치 손실 병합 후 backward
            # 이 과정에서 backward_engine(Teacher LLM)이 "어떻게 고쳐야 하는지"에 대한
            # 텍스트 기울기(Textual Gradient)를 생성합니다.
            total_loss = tg.sum(losses)
            # INFO [비평][STEP2] 비판 실행
            # [중요] TeacherLLM이 답과 모범답안을 비교해 비평(Gradient)을 작성
            # ※주의: 여기서 생성되는 gradient는 나중에 TextGrad optimizer의 <CONTEXT> 태그에 들어갑니다.
            #   <CONTEXT> 태그: "이전 최적화 시도의 피드백" (optimizer LLM이 프롬프트 개선할 때 참고)
            #   데이터의 context (RAG 문서 자료)와는 다른 개념입니다.
            total_loss.backward()
            
            # 3) 후보 프롬프트 생성 (step() 전에!)
            # 현재 gradient를 기반으로 optimizer 입력문을 만들고,
            # backward_engine(optimizer LLM)을 실제로 한 번 호출해서
            # <new_variable>...</new_variable> 형태의 후보 프롬프트를 생성한다.

            # [중요] zero_grad() 전에 gradient 텍스트를 먼저 백업
            # INFO [비평][STEP3] 비판 텍스트 추출
            prompt_feedback_text = system_prompt.get_gradient_text().strip() or "[N/A]"

            # [TextGrad Optimizer 입력문 생성]
            # optimizer._update_prompt()는 다음과 같은 구조의 프롬프트를 생성합니다:
            # - <ROLE>: 최적화 대상의 역할 설명
            # - <VARIABLE>: 현재 버전의 프롬프트 (system_prompt.value)
            # - <CONTEXT>: 이전 최적화 시도의 피드백 (gradient) ← **여기가 핵심!**
            # - <PAST_ITERATIONS>: 과거 시도 이력 (momentum 사용 시)
            #
            # ※주의: <CONTEXT> 태그는 "이전 최적화 피드백"을 담습니다.
            #   데이터의 context (RAG 문서 자료)와는 완전히 다른 개념입니다.
            #   TextGrad 라이브러리가 자동으로 gradient를 <CONTEXT> 태그에 채웁니다.
            if isinstance(optimizer, TextualGradientDescentwithMomentum):
                update_prompt_value = optimizer._update_prompt(
                    system_prompt,
                    momentum_storage_idx=0,
                )
            else:
                update_prompt_value = optimizer._update_prompt(system_prompt)

            optimizer_update_input = stringify_tgd_update_prompt(update_prompt_value)
            momentum_history = extract_momentum_history(
                optimizer,
                system_prompt,
                momentum_storage_idx=0,
            )

            # 실제 optimizer LLM 호출
            try:
                optimizer_response = backward_engine(
                    optimizer_update_input,
                    system_prompt=optimizer_system_prompt,
                )
            except TypeError:
                # 일부 엔진은 system_prompt 인자를 받지 않을 수 있으므로 fallback
                merged_optimizer_input = (
                    f"{optimizer_system_prompt}\n\n"
                    f"{optimizer_update_input}"
                )
                optimizer_response = backward_engine(merged_optimizer_input)

            optimizer_response_text = str(optimizer_response).strip()

            # optimizer_total_input 로그용: 실제 입력 + 모멘텀 이력
            optimizer_total_input_with_momentum = (
                f"{optimizer_update_input}\n\n"
                f"{'='*80}\n"
                f"{momentum_history}"
            )

            # -----------------------------------------------------------------
            # [디버깅] Optimizer 입력 / 응답 출력
            print(f"\n{'='*80}")
            print("[DEBUG] Optimizer 입력 프롬프트:")
            print(f"{'='*80}")
            print(optimizer_update_input)

            print(f"\n{'='*80}")
            print("[DEBUG] Optimizer가 생성한 전체 응답:")
            print(f"{'='*80}")
            print(optimizer_response_text)
            print(f"\n{'='*80}\n")

            # 여러 종류의 태그를 다 잡을 수 있게 정규표현식 보강
            patterns = [
                r"<new_variable>(.*?)</new_variable>",
                r"<IMPROVED_VARIABLE>(.*?)</IMPROVED_VARIABLE>",
                r"<refined_template>(.*?)</refined_template>",
                r"<OPTIMIZER_WRITING_TEXT_START>(.*?)<OPTIMIZER_WRITING_TEXT_END>",
                r"```(.*?)```",
            ]

            actual_candidate_text = None
            matched_pattern = None
            pattern_results = []  # 각 패턴별 시도 결과 저장

            for pattern in patterns:
                match = re.search(pattern, optimizer_response_text, re.DOTALL | re.IGNORECASE)
                if match:
                    candidate = match.group(1).strip()

                    rejected_reason = []
                    if not candidate:
                        rejected_reason.append("빈 문자열")
                    if "{" in candidate:
                        rejected_reason.append("중괄호 포함")
                    if "the improved variable" in candidate.lower():
                        rejected_reason.append("placeholder 텍스트")

                    if not rejected_reason:
                        actual_candidate_text = candidate
                        matched_pattern = pattern
                        pattern_results.append(f"✅ {pattern}: 매칭 성공 & 사용됨")
                        print(f"✅ 후보 프롬프트 추출 성공! (패턴: {pattern})")
                        break
                    else:
                        pattern_results.append(
                            f"⚠️ {pattern}: 매칭되었으나 거부됨 ({', '.join(rejected_reason)})"
                        )
                else:
                    pattern_results.append(f"❌ {pattern}: 매칭 실패")

            if not actual_candidate_text:
                print(f"\n{'!'*80}")
                print("⚠️ [경고] 후보 프롬프트 추출 실패!")
                print(f"{'!'*80}")
                print("\n[패턴별 매칭 시도 결과]")
                for result in pattern_results:
                    print(f"  {result}")
                print("\n[원인 분석]")
                print("  1. Optimizer LLM이 요구된 태그를 사용하지 않았거나")
                print("  2. 매칭되었지만 placeholder 텍스트를 반환했습니다.")
                print("\n[Optimizer 응답 (디버깅용)]")
                print("-" * 80)
                print(optimizer_response_text)
                print("-" * 80)
                print("\n→ 이번 iteration은 현재 프롬프트를 후보로 간주하고 비교를 계속합니다.\n")
                actual_candidate_text = system_prompt.value

            print(f"\n[추출된 후보 프롬프트]")
            print(f"매칭 패턴: {matched_pattern or '[N/A]'}")
            print(f"길이: {len(actual_candidate_text)} chars")
            print(f"내용: {actual_candidate_text}...")
            print(f"{'='*80}\n")
            # -----------------------------------------------------------------

            # 4) Validation 평가
            val_score_current = 0.0
            val_score_candidate = 0.0
            val_count = 0
            
            print(f"Validation 평가 중 (현재 vs 후보 대결)...")
            
            # 현재 프롬프트 원본 백업
            original_prompt_value = system_prompt.value # Forward 엔진의 시스템 프롬프트 값

            for val_data in validation_dataset:
                # [Validation 데이터 추출]
                # val_context: RAG 문서 자료 (GSM8k는 빈 문자열, NASA/KLUE는 실제 문서)
                # ※주의: TextGrad optimizer의 <CONTEXT> 태그(이전 피드백)와는 무관
                val_context = normalize_text_field(val_data.get('context', ''))
                val_question = normalize_text_field(val_data.get('question', ''))
                val_gt = normalize_text_field(val_data.get('answer', ''))
                
                # [Forward Model 입력 구성 - Validation]
                # context 유무에 따라 입력 형식 분기
                # - GSM8k: "Question: [수학문제]"
                # - NASA/KLUE: "Context: [문서]\nQuestion: [질문]"
                if val_context.strip():
                    val_inputs = f"Context: {val_context}\nQuestion: {val_question}"
                else:
                    val_inputs = f"Question: {val_question}"
                
                try:
                    # --- A. 현재(Current) 프롬프트 성능 측정 ---
                    system_prompt.value = original_prompt_value
                    # [수정] role_description을 반드시 포함해야 함
                    val_var_curr = tg.Variable(val_inputs, role_description="Validation input", requires_grad=False)
                    pred_curr = model(val_var_curr).value
                    
                    score_curr = similarity_judge(val_gt, pred_curr) if similarity_judge else 0
                    val_score_current += score_curr

                    # --- B. 후보(Candidate) 프롬프트 성능 측정 ---
                    system_prompt.value = actual_candidate_text
                    # [수정] role_description을 반드시 포함해야 함
                    val_var_cand = tg.Variable(val_inputs, role_description="Validation input", requires_grad=False)
                    pred_cand = model(val_var_cand).value
                    
                    score_cand = similarity_judge(val_gt, pred_cand) if similarity_judge else 0
                    val_score_candidate += score_cand
                    
                    val_count += 1
                except Exception as e:
                    print(f"Validation 샘플 에러: {e}")
                    continue
            
            # [중요] 평가가 끝나면 반드시 시스템 프롬프트를 원래대로 복구
            system_prompt.value = original_prompt_value

            if val_count > 0:
                val_score_current /= val_count
                val_score_candidate /= val_count
            
            # 5) 프롬프트 선택 및 업데이트
            # TODO Textgrad 논문의 논리가 잘 반영된게 맞는지 확인해야 함..
            # textgrad의 batch의 의도는 하나의 iteration 내에서 돌린 batch 중 best 를 뽑자는게 아니라! (중요)
            # 여러개의 batch 를 검토하여 얻은 피드백을 모두 반영한 프롬프트를 만들고자 하는 것이다. 
            # TextGrad 논문의 '배치 최적화(Batch Optimization)' 섹션
            # 논문에서는 배치 내의 여러 데이터 인스턴스에서 전파된 기울기(피드백)들을 
            # tg.sum()을 통해 하나로 이어 붙인(concatenated) 뒤 옵티마이저에 전달한다고 설명.
            # 즉, 빔 서치처럼 여러 개의 프롬프트 후보군을 만들어 경쟁시키는 것이 아니라, 
            # 옵티마이저가 여러 출처(다양한 문제)에서 온 다각적인 피드백을 한 번에 모두 확인하고 
            # 단 1개의 종합적인 개선안을 만들도록 하는 것이 배치의 진짜 목적이다. 
            # 그리고 아래 소스는, 아래와 같은 의도이다.
            # 합쳐서 만든 새 프롬프트가 진짜 좋은지 한 번 더 검증해보고, 점수가 높을 때만 바꾸자!" (방어적/신중함)
            # 논문의 실험 세팅을 보면, 매 반복(iteration)이 끝난 후 검증 데이터셋(validation set)을 돌려보고 
            # 이전 프롬프트보다 성능이 향상되었을 때만 프롬프트를 업데이트(덮어쓰기) 한다고 나와있다. 
            # 5) 프롬프트 선택 및 업데이트
            if val_score_candidate > val_score_current:
                if hasattr(optimizer, "_update_momentum_storage"):
                    optimizer._update_momentum_storage(system_prompt, momentum_storage_idx=0)

                system_prompt.set_value(actual_candidate_text)
                prompt_accepted = True
                print(f"✅ Prompt accepted & Updated (val: {val_score_current:.3f} -> {val_score_candidate:.3f})")
            else:
                system_prompt.set_value(original_prompt_value)
                prompt_accepted = False
                print(f"❌ Prompt rejected (val: {val_score_current:.3f} vs {val_score_candidate:.3f})")

            # 6) Iteration 로그 업데이트
            for idx in range(iteration_log_start_idx, len(optimization_logs)):
                optimization_logs[idx]['prompt_feedback'] = prompt_feedback_text
                optimization_logs[idx]['optimizer_total_input'] = optimizer_total_input_with_momentum
                # optimization_logs[idx]['validation_score_current'] = val_score_current
                # optimization_logs[idx]['validation_score_candidate'] = val_score_candidate
                # optimization_logs[idx]['prompt_accepted'] = prompt_accepted

            # 마지막에 gradient 비우기
            optimizer.zero_grad()
                    
        # Episode 종료 후 평균 점수 계산
        successful_scores = []
        for log in optimization_logs[episode_log_start_idx:]:
            if log.get('is_success') and log.get('total_score') is not None:
                score = log.get('total_score')
                if not (isinstance(score, float) and math.isnan(score)):
                    successful_scores.append(score)
        
        episode_avg_score = sum(successful_scores) / len(successful_scores) if successful_scores else None
        
        for idx in range(episode_log_start_idx, len(optimization_logs)):
            optimization_logs[idx]['dataset_size'] = iterations_per_episode * batch_size
            optimization_logs[idx]['avg_total_score'] = episode_avg_score
        
        print(f"\nEpisode {episode} 완료: 평균 점수 = {episode_avg_score}")
        print(f"현재 프롬프트: {system_prompt.value}")
    
    # ========== [TextGrad 논문 재현 루프 끝] ==========

    # 5. DB 저장 (main_train.py 방식 참고)
    print_step("5. DB 로그 저장")
    session = None
    try:
        session = pg_client.get_session()
        for log_data in optimization_logs:
            record = RlOptimizationLog(
                experiment_id=log_data['experiment_id'],
                episode=log_data['episode'],
                instruction=log_data['instruction'],
                question=log_data['question'],
                context=log_data['context'],
                model_answer=log_data['model_answer'],
                gold_answer=log_data['gold_answer'],
                total_score=log_data.get('total_score'),
                raw_similarity=log_data.get('raw_similarity'),
                ragas_faithfulness_score=log_data.get('ragas_faithfulness_score'),
                ragas_answer_relevancy_score=log_data.get('ragas_answer_relevancy_score'),
                accuracy=log_data.get('accuracy'),
                dataset_size=log_data.get('dataset_size'),
                avg_total_score=log_data.get('avg_total_score'),
                optimizer_model_nm=log_data['optimizer_model_nm'],
                optimizer_model_provider="azure",
                tester_model_nm=log_data['tester_model_nm'],
                tester_model_provider="azure",
                optimizer_system_prompt=log_data.get('optimizer_system_prompt'),
                optimizer_total_input=log_data.get('optimizer_total_input'),
                # critical_review: 프롬프트 최적화 관점의 TextGrad feedback
                critical_review=log_data['prompt_feedback'],
                # full_analysis: 샘플 답안(예측 vs 정답) 관점의 비판 텍스트
                full_analysis=log_data['answer_feedback'],
                is_success=log_data['is_success'],
                error_log=log_data['error_log'],
                created_at=log_data['created_at']
            )
            session.add(record)
        
        session.commit()
        print(f"[✓] DB 저장 완료: {len(optimization_logs)}건")
        
    except Exception as e:
        print(f"[!] DB 저장 실패: {str(e)}")
        if session is not None:
            session.rollback()
    finally:
        if session is not None:
            session.close()

    print_step("6. 최적화 완료")

    print("\n--- 최적화 완료 ---")
    print(f"Final optimized prompt: {system_prompt.value}")

if __name__ == "__main__":
    print_step("=== TextGrad Baseline 프롬프트 최적화 시작 ===")
    main()