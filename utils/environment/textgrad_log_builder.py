"""
@경로: utils/environment/textgrad_log_builder.py
@설명: TextGrad 최적화 실험용 DB 로그 레코드 빌더 함수들
@용도: RlOptimizationLog 모델에 저장할 딕셔너리 생성

[배경]
- TextGrad baseline/improve 실험에서 공통으로 사용하는 로그 생성 함수
- Episode별 샘플 처리 결과를 DB에 저장하기 위한 구조화된 딕셔너리 생성
- 스킵/성공/에러 케이스별로 적절한 필드 값 설정

[사용 예시]
    from utils.environment.textgrad_log_builder import (
        create_base_log,
        create_skip_log,
        create_success_log,
        create_error_log
    )
    
    # Episode 공통 필드 생성
    base_log = create_base_log(experiment_id, episode, backward_model, forward_model)
    
    # 케이스별 로그 생성
    skip_log = create_skip_log(base_log, system_prompt, question, context, truth)
    success_log = create_success_log(base_log, ..., accuracy=0.9)
    error_log = create_error_log(base_log, ..., error_msg)

[로그 케이스별 필드 전략]
1. NULL (값 없음):
   - 샘플 스킵(인젝션 패턴) 시 점수 필드
   - Judge 초기화 자체 실패로 해당 점수를 계산할 수 없는 경우

2. NaN (계산 망가짐):
   - 샘플 처리 예외로 평가가 깨진 경우
   - RAGAS 평가가 실행됐지만 내부 Evaluation error/예외가 난 경우
   - 유사도 호출에서 예외 발생 시도 NaN

[중요] context 파라미터에 대한 주의사항
=========================================
이 모듈의 함수들에서 'context' 파라미터는 **데이터의 context** (RAG 문서 자료)를 의미합니다.
- NASA 센서 로그, KLUE 검색 결과 등
- GSM8k는 context가 없음 (빈 문자열)

이것은 **TextGrad optimizer의 <CONTEXT> 태그**와는 다릅니다.
- <CONTEXT> 태그: "이전 최적화 시도의 피드백" (optimizer LLM이 프롬프트 개선 시 참고)
- TextGrad 라이브러리가 자동으로 관리하며, 이 모듈에서는 직접 다루지 않습니다.
=========================================
"""

import math
from datetime import datetime
from textgrad.optimizer.optimizer import TextualGradientDescentwithMomentum


def create_base_log(experiment_id: str, episode: int, backward_model: str, forward_model: str, embedding_model: str | None = None, dataset_nm: str | None = None) -> dict:
    """
    Episode 내 모든 샘플에 공통인 기본 필드를 반환한다.
    
    @목적: 반복되는 필드 재사용으로 코드 중복 제거 및 유지보수성 향상
    @Return: 공통 필드만 포함된 딕셔너리 (각 샘플은 copy()해서 사용)
    """
    return {
        'experiment_id': experiment_id,
        'episode': episode,
        'optimizer_model_nm': backward_model,
        'tester_model_nm': forward_model,
        'embedding_model_nm': embedding_model,
        'dataset_nm': dataset_nm,
        'created_at': datetime.now(),
    }


def create_skip_log(
    base_log: dict,
    system_prompt_value: str,
    question: str,
    context: str,  # ← RAG 문서 자료 (데이터의 context, TextGrad <CONTEXT> 태그 아님)
    ground_truth: str,
) -> dict:
    """
    Jailbreak 패턴 감지로 스킵된 샘플의 로그를 생성한다.
    
    @파라미터 주의사항:
    - context: RAG 문서 자료 (데이터의 context)
      ※ TextGrad optimizer의 <CONTEXT> 태그 (이전 최적화 피드백)와는 다릅니다.
    """
    log = base_log.copy()
    log.update({
        'instruction': system_prompt_value,
        'question': question,
        'context': context,
        'model_answer': None,
        'gold_answer': ground_truth,
        'total_score': None,
        'raw_similarity': None,
        'ragas_faithfulness_score': None,
        'ragas_answer_relevancy_score': None,
        'forward_tester_llm_call_cnt': 0,
        'backward_judge_llm_call_cnt': 0,
        'backward_optimizer_llm_call_cnt': 0,
        'answer_feedback': "[N/A] 잠재적 인젝션 패턴 포함 샘플",
        'prompt_feedback': "[N/A] 샘플 스킵",
        'validation_info': None,  # iteration 종료 후 채움
        'validation_accuracy': None,  # iteration 종료 후 채움
        'validation_dataset_size': None,  # iteration 종료 후 채움
        'is_success': False,
        'error_log': "[Skipped] potential jailbreak-like pattern",
    })
    return log


def create_success_log(
    base_log: dict,
    system_prompt_value: str,
    question: str,
    context: str,  # ← RAG 문서 자료 (데이터의 context, TextGrad <CONTEXT> 태그 아님)
    ground_truth: str,
    prediction: str,
    computed_loss_value: str,
    raw_similarity: float | None,
    ragas_faithfulness_score: float | None,
    ragas_answer_relevancy_score: float | None,
    optimizer_system_prompt: str,
    accuracy: float | None = None,
    forward_tester_llm_call_cnt: int = 0,
    backward_judge_llm_call_cnt: int = 0,
    backward_optimizer_llm_call_cnt: int = 0,
    evaluation_instruction: str | None = None,
    backward_judge_total_input: str | None = None,
) -> dict:
    """
    정상 처리된 Train 샘플의 로그를 생성한다.
    
    @파라미터 주의사항:
    - context: RAG 문서 자료 (데이터의 context)
      ※ TextGrad optimizer의 <CONTEXT> 태그 (이전 최적화 피드백)와는 다릅니다.
    - forward_tester_llm_call_cnt: Forward Model(답변 생성) LLM 호출 횟수
    - backward_judge_llm_call_cnt: Backward Judge(평가) LLM 호출 횟수
    - backward_optimizer_llm_call_cnt: Backward Optimizer(프롬프트 개선) LLM 호출 횟수
    - evaluation_instruction: 비평가 엔진에게 전달하는 평가 기준 (우선순위, 고려사항 등)
    - backward_judge_total_input: 비평가 엔진에게 전달되는 전체 입력 내용
    """
    log = base_log.copy()
    log.update({
        'instruction': system_prompt_value,
        'question': question,
        'context': context,
        'model_answer': prediction,
        'gold_answer': ground_truth,
        'total_score': raw_similarity,
        'raw_similarity': raw_similarity,
        'ragas_faithfulness_score': ragas_faithfulness_score,
        'ragas_answer_relevancy_score': ragas_answer_relevancy_score,
        'accuracy': accuracy,
        'forward_tester_llm_call_cnt': forward_tester_llm_call_cnt,
        'backward_judge_llm_call_cnt': backward_judge_llm_call_cnt,
        'backward_optimizer_llm_call_cnt': backward_optimizer_llm_call_cnt,
        'answer_feedback': computed_loss_value,
        'prompt_feedback': None,  # iteration 종료 후 채움
        'validation_info': None,  # iteration 종료 후 채움
        'validation_accuracy': None,  # iteration 종료 후 채움
        'validation_dataset_size': None,  # iteration 종료 후 채움
        'is_success': True,
        'error_log': None,
        'optimizer_system_prompt': optimizer_system_prompt,
        'optimizer_total_input': None,  # iteration 종료 후 채움
        'evaluation_instruction': evaluation_instruction,
        'backward_judge_total_input': backward_judge_total_input,
    })
    return log


def create_error_log(
    base_log: dict,
    system_prompt_value: str,
    question: str,
    context: str,  # ← RAG 문서 자료 (데이터의 context, TextGrad <CONTEXT> 태그 아님)
    ground_truth: str,
    error_message: str,
) -> dict:
    """
    샘플 처리 중 예외가 발생한 경우의 로그를 생성한다.
    
    @파라미터 주의사항:
    - context: RAG 문서 자료 (데이터의 context)
      ※ TextGrad optimizer의 <CONTEXT> 태그 (이전 최적화 피드백)와는 다릅니다.
    """
    log = base_log.copy()
    log.update({
        'instruction': system_prompt_value,
        'question': question,
        'context': context,
        'model_answer': None,
        'gold_answer': ground_truth,
        'total_score': math.nan,
        'raw_similarity': math.nan,
        'ragas_faithfulness_score': math.nan,
        'forward_tester_llm_call_cnt': 0,
        'backward_judge_llm_call_cnt': 0,
        'backward_optimizer_llm_call_cnt': 0,
        'ragas_answer_relevancy_score': math.nan,
        'answer_feedback': "[N/A] 답변 평가 실패",
        'prompt_feedback': "[N/A] 샘플 처리 실패",
        'validation_info': None,  # iteration 종료 후 채움
        'validation_accuracy': None,  # iteration 종료 후 채움
        'validation_dataset_size': None,  # iteration 종료 후 채움
        'is_success': False,
        'error_log': f"[Sample Error] {error_message}",
    })
    return log

def extract_momentum_history(optimizer, momentum_storage_idx: int = 0) -> str:
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
    history_lines.append(f"=== ! 여기부터는 실제 input과 이어지는 내용이 아님. 분석 용도의 추가 로그 ! ===")
    history_lines.append("아래는 과거 episode들에서 누적된 프롬프트 변화와 gradient 피드백입니다.")
    history_lines.append("Optimizer는 이 이력을 참고하여 다음 프롬프트를 생성합니다.\n")
    
    for i, step_info in enumerate(past_n_steps):
        step_value = step_info.get('value', '[N/A]')
        step_gradient = step_info.get('gradients', '[N/A]')  # ← 'gradient' → 'gradients' (TextGrad 라이브러리 키명과 일치)
        
        history_lines.append(f"--- Past Step {i+1} ---")
        history_lines.append(f"Prompt Value at Step {i+1}:")
        history_lines.append(f"{step_value}")
        history_lines.append(f"\nGradient Feedback at Step {i+1}:")
        history_lines.append(f"{step_gradient}")
        history_lines.append("")  # 빈 줄로 구분
    
    return "\n".join(history_lines)


def build_backward_judge_total_input(
    evaluation_instruction: str | None,
    prediction: str,
) -> str:
    """
    Backward Judge (비평가 엔진)에게 전달되는 입력을 추정하여 재구성한다.
    
    TextLoss가 backward_engine을 호출할 때 사용하는 입력 형식을 추정합니다.
    실제 TextGrad 라이브러리 내부 구현에 따라 다를 수 있으나,
    일반적으로 evaluation_instruction + prediction을 조합한 형태입니다.
    
    @파라미터:
    - evaluation_instruction: 비평가 엔진에게 전달하는 평가 기준 프롬프트
    - prediction: Forward 엔진이 생성한 답변
    
    @Return:
        str: Backward Judge 입력 추정 문자열
    """
    safe_eval_instruction = evaluation_instruction if evaluation_instruction else "[N/A]"
    separator = "=" * 80
    
    return (
        "※ 해당 데이터는 실제 전달되는 문자열을 추출한 것이 아닌, "
        "evaluation_instruction과 prediction을 조합하여 가공한 추정 데이터입니다.\n\n"
        f"{separator}\n"
        "[Backward Judge Total Input - Estimated Format]\n"
        "TextLoss가 backward_engine(비평가 LLM)에게 전달한 것으로 추정되는 입력입니다.\n"
        f"{separator}\n\n"
        "=== EVALUATION INSTRUCTION (평가 기준) ===\n"
        f"{safe_eval_instruction}\n\n"
        "=== PREDICTION (평가 대상 답변) ===\n"
        f"{prediction}\n"
    )


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