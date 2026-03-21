"""
@경로: main_textgrad_improve.py
@설명: TextGrad + 계층적 피드백 구조를 활용한 프롬프트 최적화 및 DB 저장
- TextGrad Baseline 논문과 차별화된, 계층적(Hierarchical) 피드백 구조 적용
- 답변 평가(TextLoss)와 프롬프트 피드백을 3계층 rubric으로 구조화
- Layer 1: Fact Alignment (사실 정확도)
- Layer 2: Context Grounding (문맥 근거)
- Layer 3: Expression Quality (표현 품질)
- 이를 통해 더 구조화되고 체계적인 프롬프트 최적화를 수행한다.

[Baseline 대비 차별점]
1. build_hierarchical_evaluation_instruction(): 계층형 평가 지시문 생성
2. build_hierarchical_prompt_feedback(): gradient + 샘플 비평을 계층화하여 피드백 구성

- 2026.03.07 데이터 로그 적재 부 추가
NULL(값 없음):
    샘플 스킵(인젝션 패턴) 시 점수 필드
    Judge 초기화 자체 실패로 해당 점수를 계산할 수 없는 경우
NaN(계산 망가짐):
    샘플 처리 예외로 평가가 깨진 경우
    RAGAS 평가가 실행됐지만 내부 Evaluation error/예외가 난 경우
    (유사도 호출에서 예외 발생 시도 NaN)
"""

import os

import math

from datetime import datetime
import textgrad as tg
from textgrad.optimizer.optimizer import TextualGradientDescentwithMomentum
from utils.log.console import print_step

from datafile.data_loader import load_dataset
from infrastructure.llm_client import get_textgrad_backward_engine, get_textgrad_forward_engine

from conf.config import Settings
# 로그 저장을 위한 import (main_train.py 방식)
from models.rl_optimization_log import RlOptimizationLog
from db.connection.pg_client import pg_client

# 공통으로 사용 가능 한 utils 함수들
from utils.llm_errors.error_parsers import extract_root_error_message
from utils.llm_safety.azure_prompt_filters import has_jailbreak_like_pattern, sanitize_for_azure_filter
from utils.text.normalization import normalize_text_field
from utils.llm_patches.textgrad_patches import patch_textgrad_openai_compatibility, patch_textgrad_momentum_compatibility

# 기타 LLM get 함수들
from metrics.judges.similarity_judge import create_similarity_judge
from metrics.judges.ragas_failthfulness_judge import create_ragas_judge


def compact_textgrad_gradients(variable: tg.Variable, max_chars_per_gradient: int = 700) -> None:
    """TextGrad 업데이트 프롬프트 길이를 줄이기 위해 gradient context를 제거/압축한다."""
    for grad in list(variable.gradients):
        compacted = sanitize_for_azure_filter(grad.value, max_chars=max_chars_per_gradient)
        if has_jailbreak_like_pattern(compacted):
            compacted = "답변의 정확성, 근거성, 간결성을 높이도록 프롬프트를 개선하세요."
        grad.set_value(compacted)
        variable.gradients_context[grad] = None


def get_tgd_optimizer_system_prompt(optimizer) -> str:
    """TGD optimizer 인스턴스가 실제 사용하는 시스템 프롬프트를 추출한다."""
    for attr_name in ("optimizer_system_prompt", "system_prompt"):
        prompt_value = getattr(optimizer, attr_name, None)
        if isinstance(prompt_value, str) and prompt_value.strip():
            return prompt_value
    return "[N/A] Unable to capture TGD optimizer system prompt from optimizer instance."


def stringify_tgd_update_prompt(prompt_value) -> str:
    """TGD update prompt(str/list)를 로그 저장용 문자열로 변환한다."""
    if prompt_value is None:
        return "[N/A] TGD update prompt is unavailable."
    if isinstance(prompt_value, str):
        return prompt_value
    if isinstance(prompt_value, list):
        return "\n\n".join(str(item) for item in prompt_value)
    return str(prompt_value)


def capture_optimizer_update_prompt(optimizer, variable: tg.Variable, momentum_storage_idx: int = 0) -> str:
    """optimizer 타입별 _update_prompt 시그니처를 맞춰 로그 문자열로 변환한다."""
    if isinstance(optimizer, TextualGradientDescentwithMomentum):
        prompt_value = optimizer._update_prompt(variable, momentum_storage_idx=momentum_storage_idx)
    else:
        prompt_value = optimizer._update_prompt(variable)
    return stringify_tgd_update_prompt(prompt_value)


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
    context: str,
    gold_answer: str,
    model_answer: str | None,
    evaluation_instruction: str | None,
    answer_feedback: str | None,
) -> str:
    """
    [에러/스킵 케이스 전용] TGD optimizer 입력을 추정 형식으로 재구성한다.
    
    @주의:
    - 이 함수는 optimizer가 실제로 실행되지 않은 에러/스킵 케이스에서만 사용됩니다.
    - 정상 처리 시에는 capture_optimizer_update_prompt()로 실제 optimizer 프롬프트를 저장합니다.
    - 출력 문자열 첫 줄에 "[TextGrad TGD Optimizer Input - Estimated Format]"을 명시하여
      이것이 실제 optimizer 입력이 아닌 추정값임을 표시합니다.
    
    @용도:
    - 샘플 스킵 (jailbreak 패턴 감지)
    - 샘플 처리 예외 (Azure content filter, 평가 실패 등)
    - DB 분석 시 "만약 정상 처리됐다면 이런 입력이 들어갔을 것"이라는 컨텍스트 제공
    """
    safe_model_answer = model_answer if model_answer is not None else "[N/A]"
    safe_eval_instruction = evaluation_instruction if evaluation_instruction is not None else "[N/A]"
    safe_answer_feedback = answer_feedback if answer_feedback is not None else "[N/A]"

    return (
        "[TextGrad TGD Optimizer Input - Estimated Format]\n\n"
        "=== OPTIMIZER SYSTEM PROMPT ===\n"
        f"{optimizer_system_prompt}\n\n"
        "=== OPTIMIZATION TARGET (instruction) ===\n"
        f"{instruction}\n\n"
        "=== SAMPLE INPUT ===\n"
        f"question: {question}\n"
        f"context: {context}\n"
        f"gold_answer: {gold_answer}\n\n"
        "=== FORWARD OUTPUT ===\n"
        f"model_answer: {safe_model_answer}\n\n"
        "=== LOSS EVALUATION INSTRUCTION ===\n"
        f"{safe_eval_instruction}\n\n"
        "=== ANSWER-LEVEL FEEDBACK (TextLoss output) ===\n"
        f"{safe_answer_feedback}\n"
    )


################### 차별점 ############################################################################
## TextGrad Baseline 논문과 차별화된, 내가 설계한 계층적 피드백 구조로 프롬프트 개선 지시문을 생성하도록 추가한다. 

def build_hierarchical_evaluation_instruction(ground_truth: str) -> str:
    """답변 비평(TextLoss)을 계층형 rubric으로 구성한다."""
    return (
        f"[Ground Truth]\n{ground_truth}\n\n"
        "[Layer 1: Fact Alignment]\n"
        "- 정답 대비 사실 오류, 누락, 환각 가능성을 먼저 지적하세요.\n"
        "[Layer 2: Context Grounding]\n"
        "- 답변의 핵심 주장별로 문맥 근거 유무를 짚어주세요.\n"
        "[Layer 3: Expression Quality]\n"
        "- 간결성, 명확성, 논리 흐름 개선점을 제안하세요.\n"
        "출력 형식: (1) 치명 오류 3개 이내 (2) 즉시 적용 가능한 개선 지시 3개"
    )


def build_hierarchical_prompt_feedback(gradient_text: str, episode_logs: list[dict]) -> str:
    """프롬프트 업데이트용 피드백을 gradient + 샘플 비평으로 계층화한다."""
    sample_feedbacks: list[str] = []
    for row in episode_logs:
        raw_feedback = row.get('answer_feedback')
        if raw_feedback is None:
            continue
        normalized_feedback = normalize_text_field(raw_feedback).strip()
        if normalized_feedback:
            sample_feedbacks.append(normalized_feedback)
        if len(sample_feedbacks) >= 5:
            break

    if not sample_feedbacks:
        sample_feedback_block = "- [N/A] 유효한 샘플 비평이 없어 gradient 중심으로 업데이트"
    else:
        sample_feedback_block = "\n".join(f"- {item}" for item in sample_feedbacks)

    cleaned_gradient = normalize_text_field(gradient_text).strip() or "[N/A] TextGrad prompt feedback is empty."
    return (
        "[Layer A: TextGrad Gradient]\n"
        f"{cleaned_gradient}\n\n"
        "[Layer B: Sample Critiques]\n"
        f"{sample_feedback_block}\n\n"
        "[Layer C: Prompt Rewrite Directives]\n"
        "- 사실 정확도와 정답 일치도를 최우선으로 유지\n"
        "- 문맥에 없는 추론은 금지하고 근거 부족 시 명시\n"
        "- 불필요한 장황함을 줄이고 핵심 답변을 우선 제시"
    )

################################################################################################################


def create_base_log(experiment_id: str, episode: int, backward_model: str, forward_model: str) -> dict:
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
        'created_at': datetime.now(),
    }


def create_skip_log(
    base_log: dict,
    system_prompt_value: str,
    question: str,
    context: str,
    ground_truth: str,
) -> dict:
    """Jailbreak 패턴 감지로 스킵된 샘플의 로그를 생성한다."""
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
        'answer_feedback': "[N/A] 잠재적 인젝션 패턴 포함 샘플",
        'prompt_feedback': "[N/A] 샘플 스킵",
        'is_success': False,
        'error_log': "[Skipped] potential jailbreak-like pattern",
    })
    return log


def create_success_log(
    base_log: dict,
    system_prompt_value: str,
    question: str,
    context: str,
    ground_truth: str,
    prediction: str,
    computed_loss_value: str,
    raw_similarity: float | None,
    ragas_faithfulness_score: float | None,
    ragas_answer_relevancy_score: float | None,
    optimizer_system_prompt: str,
) -> dict:
    """정상 처리된 Train 샘플의 로그를 생성한다."""
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
        'answer_feedback': computed_loss_value,
        'prompt_feedback': None,  # iteration 종료 후 채움
        'is_success': True,
        'error_log': None,
        'optimizer_system_prompt': optimizer_system_prompt,
        'optimizer_total_input': None,  # iteration 종료 후 채움
    })
    return log


def create_error_log(
    base_log: dict,
    system_prompt_value: str,
    question: str,
    context: str,
    ground_truth: str,
    error_message: str,
) -> dict:
    """샘플 처리 중 예외가 발생한 경우의 로그를 생성한다."""
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
        'ragas_answer_relevancy_score': math.nan,
        'answer_feedback': "[N/A] 답변 평가 실패",
        'prompt_feedback': "[N/A] 샘플 처리 실패",
        'is_success': False,
        'error_log': f"[Sample Error] {error_message}",
    })
    return log


def main():
    patch_textgrad_openai_compatibility()
    patch_textgrad_momentum_compatibility()

    print_step("0. [Settings] 설정 초기화")
    Settings.setup()
    
    print_step("1. [Infrastructure] TextGrad LLM 연결 설정")
    
    print_step("2. 데이터 로드 및 Train/Validation 분할")
    dataset_name = "didi0di/klue-mrc-ko-rag-cot"
    
    # TextGrad 논문: 총 데이터 로드 후 train/validation 분할
    # - Train: 무작위 복원 추출용 풀 (충분히 큰 데이터셋)
    # - Validation: 프롬프트 선택용 고정 세트
    total_sample_size = int(os.getenv("TEXTGRAD_TOTAL_SAMPLES", "50"))  # Train pool용
    validation_size = int(os.getenv("TEXTGRAD_VALIDATION_SIZE", "10"))
    
    full_dataset = load_dataset(dataset_name=dataset_name, sample_size=total_sample_size + validation_size)
    
    if not full_dataset:
        print("[FAIL] 데이터 로드 실패")
        return
    
    # Train/Validation 분할
    import random
    random.seed(42)  # 재현성
    shuffled = full_dataset.copy()
    random.shuffle(shuffled)
    
    train_pool = shuffled[:total_sample_size]  # 복원 추출용 풀
    validation_dataset = shuffled[total_sample_size:total_sample_size + validation_size]
    
    print(f"[✓] Train pool: {len(train_pool)}개, Validation: {len(validation_dataset)}개")
    
    print_step("3. TextGrad 환경 설정 및 엔진 초기화")
    # TextGrad improve용 experiment_id 생성
    current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
    experiment_id = f"textgrad_improve_{current_time}"
    
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
    
    # [연구 로드맵] 현재는 TextGrad Baseline 재현 단계
    # 향후 발전 방향: tg.TextLoss(평가 지시문 문자열) 대신
    # → CaseAwareJudgeLoss() 클래스로 교체 (8가지 기업용 RAG 지표 평가)
    #   - Faithfulness, Relevancy, Completeness, Conciseness 등
    #   - 각 Judge는 구조화된 JSON 형태로 평가 결과 반환
    # judge_loss_fn = CaseAwareJudgeLoss()  # TODO: 다음 단계 구현

    # 3. 최적화 대상 정의
    # role_descriptio은  [지금 고쳐야 할 대상(변수)의 정체] 를 말한다.
    initial_prompt = "주어진 문맥을 바탕으로 질문에 친절하게 답해주세요."
    system_prompt = tg.Variable(
        initial_prompt, 
        requires_grad=True, 
        role_description="RAG 시스템의 답변 생성을 위한 시스템 프롬프트"
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
    optimizer = TextualGradientDescentwithMomentum(
        parameters=list(model.parameters()),
        engine=backward_engine,
        momentum_window=momentum_window,
    )
    # optimizer(TGD): TextGrad의 텍스트 경사하강 업데이트기.
    # backward에서 나온 피드백을 입력으로 받아, 최적화 대상 변수(system_prompt.value)를 한 step씩 실제로 갱신한다.
    optimizer_system_prompt = get_tgd_optimizer_system_prompt(optimizer)

    # 4. 최적화 루프 - TextGrad 논문 설정
    # [TextGrad 논문 재현 설정]
    # - 배치 크기 (Batch size): 3
    # - 반복 횟수 (Iterations per episode): 12
    # - 총 훈련 데이터 수: 36개 (3 × 12, 복원 추출)
    # - 매 iteration마다 validation으로 평가, 성능 향상 시에만 업데이트
    
    episodes = int(os.getenv("TEXTGRAD_EPISODES", "3"))  # 논문에서는 보통 3~5 episodes # TODO 나중에 3으로 고칠 것 (논문재현)
    iterations_per_episode = int(os.getenv("TEXTGRAD_ITERATIONS_PER_EPISODE", "2"))  # TODO 원본 12
    batch_size = int(os.getenv("TEXTGRAD_BATCH_SIZE", "3")) # TODO 원본 
    
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
                context = sanitize_for_azure_filter(data.get('context', ''), max_chars=900)
                question = sanitize_for_azure_filter(data.get('question', ''), max_chars=500)
                ground_truth = sanitize_for_azure_filter(data.get('answer', ''), max_chars=800)
                
                if has_jailbreak_like_pattern(context) or has_jailbreak_like_pattern(question) or has_jailbreak_like_pattern(ground_truth):
                    # Jailbreak 패턴 감지 - 스킵
                    optimization_logs.append(create_skip_log(
                        base_log, system_prompt.value, question, context, ground_truth
                    ))
                    continue
                
                try:
                    # 답변 생성
                    inputs = f"Context: {context}\nQuestion: {question}"
                    query_var = tg.Variable(inputs, role_description="RAG 입력", requires_grad=False)
                    prediction_var = model(query_var)
                    prediction = prediction_var.value
                    
                    ############################### 차별점 ###############################################
                    # TextGrad의 평가 (Loss) - 계층형 rubric 적용
                    evaluation_instruction = build_hierarchical_evaluation_instruction(ground_truth)
                    ######################################################################################
                    loss = tg.TextLoss(evaluation_instruction)
                    computed_loss = loss(prediction_var)
                    losses.append(computed_loss)
                    
                    # 점수 계산
                    raw_similarity = None
                    if similarity_judge is not None:
                        try:
                            raw_similarity = similarity_judge(ground_truth, prediction)
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
                        optimizer_system_prompt
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
            
            
            # (Code Snippet 2) 논문 본문에 수록된 기본 프롬프트 최적화 코드
            # 논문의 핵심: tg.sum()으로 배치 손실 병합 후 backward
            # 이 과정에서 backward_engine(Teacher LLM)이 "어떻게 고쳐야 하는지"에 대한
            # 텍스트 기울기(Textual Gradient)를 생성합니다.
            total_loss = tg.sum(losses)
            total_loss.backward()
            
            # [개선] Azure Content Filter 회피를 위한 gradient 압축
            compact_textgrad_gradients(system_prompt)
            
            # 3) 후보 프롬프트 생성 (step() 전에!)
            # optimizer._update_prompt()는 후보 프롬프트를 생성만 하고 적용하지 않음
            tgd_update_prompt = capture_optimizer_update_prompt(optimizer, system_prompt, momentum_storage_idx=0)
            momentum_history = extract_momentum_history(optimizer, system_prompt, momentum_storage_idx=0)
            
            # 후보 프롬프트 추출 (TODO: 이 부분은 TextGrad 내부 구조에 따라 조정 필요)
            # 현재는 optimizer._update_prompt()의 결과를 파싱해서 <new_variable> 태그 추출
            candidate_prompt = system_prompt.value  # 임시: 실제로는 파싱 필요
            # [TODO] candidate_prompt 추출 로직 구현 필요
            # TextGrad optimizer는 <new_variable>...</new_variable> 형태로 반환
            # 이를 파싱하여 실제 후보 프롬프트 텍스트를 추출해야 함
            
            # 4) Validation 평가
            val_score_current = 0.0
            val_score_candidate = 0.0
            val_count = 0
            
            print(f"Validation 평가 중...")
            for val_data in validation_dataset:
                val_context = sanitize_for_azure_filter(val_data.get('context', ''), max_chars=900)
                val_question = sanitize_for_azure_filter(val_data.get('question', ''), max_chars=500)
                val_ground_truth = sanitize_for_azure_filter(val_data.get('answer', ''), max_chars=800)
                
                try:
                    val_inputs = f"Context: {val_context}\nQuestion: {val_question}"
                    
                    # 현재 프롬프트로 평가
                    val_query_current = tg.Variable(val_inputs, role_description="Validation 입력", requires_grad=False)
                    val_pred_current_var = model(val_query_current)
                    val_pred_current = val_pred_current_var.value
                    
                    if similarity_judge is not None:
                        score_current = similarity_judge(val_ground_truth, val_pred_current)
                        val_score_current += score_current
                    
                    # [TODO] 후보 프롬프트로 평가
                    # 실제 구현에서는 일시적으로 system_prompt.value를 candidate_prompt로 변경하고
                    # 평가 후 다시 복원해야 함
                    # 지금은 placeholder로 현재와 동일 점수 사용
                    val_score_candidate += score_current  # [TODO] 실제 후보 평가로 교체
                    
                    val_count += 1
                except Exception:
                    continue
            
            if val_count > 0:
                val_score_current /= val_count
                val_score_candidate /= val_count
            
            # 5) 프롬프트 선택 및 업데이트
            prompt_accepted = False
            if val_score_candidate > val_score_current:
                optimizer.step()  # 후보 채택
                prompt_accepted = True
                print(f"✅ Prompt accepted (val: {val_score_current:.3f} -> {val_score_candidate:.3f})")
            else:
                print(f"❌ Prompt rejected (val: {val_score_current:.3f} vs {val_score_candidate:.3f})")
            
            optimizer.zero_grad()
            
            # 6) Iteration 로그 업데이트
            ##################################### 차별점 #############################################
            # TextGrad textual feedback: 최적화 대상(system_prompt)에 대한 gradient 텍스트
            # -> iteration 단위 피드백이므로, 이번 iteration의 모든 row에 동일하게 기록
            prompt_feedback_text = build_hierarchical_prompt_feedback(
                gradient_text=system_prompt.get_gradient_text(),
                episode_logs=optimization_logs[iteration_log_start_idx:],
            )
            ##########################################################################################
            optimizer_total_input_with_momentum = f"{tgd_update_prompt}\n\n{'='*80}\n{momentum_history}"
            
            for idx in range(iteration_log_start_idx, len(optimization_logs)):
                optimization_logs[idx]['prompt_feedback'] = prompt_feedback_text
                optimization_logs[idx]['optimizer_total_input'] = optimizer_total_input_with_momentum
                # optimization_logs[idx]['validation_score_current'] = val_score_current  # [TODO] 새 컬럼
                # optimization_logs[idx]['validation_score_candidate'] = val_score_candidate  # [TODO] 새 컬럼
                # optimization_logs[idx]['prompt_accepted'] = prompt_accepted  # [TODO] 새 컬럼
        
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