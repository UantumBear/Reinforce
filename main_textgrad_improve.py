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
"""

import os
import re
import math
from datetime import datetime
import textgrad as tg
from utils.log.console import print_step

from datafile.data_loader import load_dataset
from infrastructure.llm_client import get_textgrad_backward_engine, get_textgrad_forward_engine
from metrics.judges.similarity_judge import SimilarityJudge
from metrics.judges.ragas_failthfulness_judge import RagasFaithfulnessJudge
from conf.config import Settings
# 로그 저장을 위한 import (main_train.py 방식)
from models.rl_optimization_log import RlOptimizationLog
from db.connection.pg_client import pg_client


def patch_textgrad_openai_compatibility():
    from textgrad.engine.openai import ChatOpenAI

    if hasattr(ChatOpenAI, "_original_generate_from_single_prompt"):
        return

    ChatOpenAI._original_generate_from_single_prompt = ChatOpenAI._generate_from_single_prompt

    def _patched_generate_from_single_prompt(self, prompt: str, system_prompt: str = None, temperature=0, max_tokens=2000, top_p=0.99):
        model_name = (self.model_string or "").lower()
        if model_name.startswith("o") or model_name.startswith("gpt-5"):
            sys_prompt_arg = system_prompt if system_prompt else self.system_prompt

            cache_or_none = self._check_cache(sys_prompt_arg + prompt)
            if cache_or_none is not None:
                return cache_or_none

            response = self.client.chat.completions.create(
                model=self.model_string,
                messages=[
                    {"role": "system", "content": sys_prompt_arg},
                    {"role": "user", "content": prompt},
                ],
                max_completion_tokens=max_tokens,
                temperature=1,
            )

            response = response.choices[0].message.content
            self._save_cache(sys_prompt_arg + prompt, response)
            return response

        return ChatOpenAI._original_generate_from_single_prompt(self, prompt, system_prompt, temperature, max_tokens, top_p)

    ChatOpenAI._generate_from_single_prompt = _patched_generate_from_single_prompt


def extract_root_error_message(error: Exception) -> str:
    """tenacity.RetryError 내부의 실제 예외(BadRequestError 등)를 추출한다."""
    try:
        # tenacity RetryError 케이스: last_attempt.exception()에 실제 원인이 있음
        if hasattr(error, "last_attempt"):
            inner_exc = error.last_attempt.exception()
            if inner_exc is not None:
                return str(inner_exc)
    except Exception:
        pass

    return str(error)


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


def normalize_text_field(value) -> str:
    """context/question/answer 필드를 문자열로 안전하게 정규화한다."""
    if value is None:
        return ""
    if isinstance(value, list):
        text = " ".join(str(item) for item in value)
    else:
        text = str(value)
    return " ".join(text.split())


def sanitize_for_azure_filter(value, max_chars: int) -> str:
    """Azure 필터 민감도를 낮추기 위해 과도한 길이/명시적 프롬프트 인젝션 문구를 완화한다."""
    text = normalize_text_field(value)
    text = re.sub(r"ignore\s+previous\s+instructions", "[redacted-instruction]", text, flags=re.IGNORECASE)
    text = re.sub(r"system\s+prompt", "system-guidance", text, flags=re.IGNORECASE)
    text = re.sub(r"(이전\s*지시\s*무시|시스템\s*프롬프트)", "[완화됨]", text)
    if len(text) > max_chars:
        return text[:max_chars] + " ...[truncated]"
    return text


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
    """TGD optimizer 입력을 분석용 문자열로 재구성한다."""
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


def create_similarity_judge() -> SimilarityJudge | None:
    """main_train.py와 동일한 임베딩 기반 유사도 평가기를 초기화한다."""
    try:
        return SimilarityJudge(use_azure=False)
    except SystemExit:
        print("[Warning] SimilarityJudge 초기화 실패(SystemExit). 유사도 점수는 None(NULL)으로 기록됩니다.")
        return None
    except Exception as e:
        print(f"[Warning] SimilarityJudge 초기화 실패: {e}. 유사도 점수는 None(NULL)으로 기록됩니다.")
        return None


def create_ragas_judge() -> RagasFaithfulnessJudge | None:
    """RAGAS 기반 Faithfulness/Relevancy 평가기를 초기화한다(평가 로그 전용)."""
    try:
        return RagasFaithfulnessJudge()
    except SystemExit:
        print("[Warning] RAGAS Judge 초기화 실패(SystemExit). RAGAS 점수는 None으로 기록됩니다.")
        return None
    except Exception as e:
        print(f"[Warning] RAGAS Judge 초기화 실패: {e}. RAGAS 점수는 None으로 기록됩니다.")
        return None

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

def main():
    patch_textgrad_openai_compatibility()

    print_step("0. [Settings] 설정 초기화")
    Settings.setup()
    
    print_step("1. [Infrastructure] TextGrad LLM 연결 설정")
    
    print_step("2. 데이터 로드")
    dataset_name = "didi0di/klue-mrc-ko-rag-cot"
    train_dataset = load_dataset(dataset_name=dataset_name, sample_size=3)
    
    if not train_dataset:
        print("[팅FAIL] 데이터 로드 실패")
        return
    
    print_step("3. TextGrad 환경 설정 및 엔진 초기화")
    # TextGrad improve용 experiment_id 생성
    current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
    experiment_id = f"textgrad_improve_{current_time}"
    
    # TextGrad 엔진 초기화: Forward(전방 생성) / Backward(역전파 피드백) 분리
    textgrad_forward_model = (
        os.getenv("TEXTGRAD_FORWARD_ENGINE_MODEL")
        or os.getenv("TEXTGRAD_TESTER_MODEL")
        or Settings.TESTER_MODEL
    )
    textgrad_backward_model = (
        os.getenv("TEXTGRAD_BACKWARD_ENGINE_MODEL")
        or os.getenv("TEXTGRAD_TEACHER_MODEL")
        or Settings.OPTIMIZER_MODEL
    )

    forward_engine = get_textgrad_forward_engine()
    backward_engine = get_textgrad_backward_engine()
    tg.set_backward_engine(backward_engine)
    
    print_step("4. TextGrad 최적화 실행")
    similarity_judge = create_similarity_judge()
    ragas_judge = create_ragas_judge()

    # 3. 최적화 대상 정의
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

    # TODO(reproduction-momentum): 현재는 vanilla TGD(기본값)만 사용 중.
    # 논문 재현 모드에서는 TextGrad의 momentum 확장 사용 여부를 실험 플래그로 분리해 적용 필요.
    # - 변경 포인트 1: 아래 optimizer 초기화에서 momentum 관련 인자를 명시적으로 전달
    # - 변경 포인트 2: update 프롬프트에 과거 iteration의 변수/gradient 히스토리 컨텍스트를 포함
    # - 변경 포인트 3: vanilla vs momentum 결과를 같은 seed/스케줄로 나란히 로깅
    # 참고: 실제 인자명/지원 옵션은 textgrad 버전에 따라 다를 수 있으므로 도입 시점에 API 확인 필요.
    # optimizer = tg.TGD(parameters=list(model.parameters()), engine=backward_engine)
    # NOTE: textgrad==0.1.8 에서는 momentum 유사 확장을 gradient_memory로 사용한다.
    # 0이면 기존 vanilla TGD 동작, 1 이상이면 과거 gradient 피드백을 update prompt에 포함한다.
    momentum_gradient_memory = int(os.getenv("TEXTGRAD_MOMENTUM_GRADIENT_MEMORY", "3"))
    optimizer = tg.TGD(
        parameters=list(model.parameters()),
        engine=backward_engine,
        gradient_memory=momentum_gradient_memory,
    )
    # optimizer(TGD): TextGrad의 텍스트 경사하강 업데이트기.
    # backward에서 나온 피드백을 입력으로 받아, 최적화 대상 변수(system_prompt.value)를 한 step씩 실제로 갱신한다.
    optimizer_system_prompt = get_tgd_optimizer_system_prompt(optimizer)
    # 4. 최적화 루프
    # [설계 의도]
    # - main_train.py와 용어를 맞추기 위해, 여기서는 "1 episode = train_dataset 전체 평가 + 1회 업데이트"로 정의한다.
    # - 즉, TextGrad의 mini-batch 개념을 사용하지 않고 episode 단위 업데이트를 사용한다.
    episodes = 3
    batch_size = len(train_dataset)  # episode 기반 설계에서는 전체 데이터셋을 한 번에 묶는다.
    optimization_logs = []  # DB 저장용 로그 버퍼

    print(f"--- TextGrad Improve Optimization 및 DB 저장 시작 ---")

    for episode in range(1, episodes + 1):
        # episode 기반 설계: batch는 항상 전체 train_dataset
        batch = train_dataset[:batch_size]
        losses = []
        episode_log_start_idx = len(optimization_logs)

        for data in batch:
            context = sanitize_for_azure_filter(data.get('context', ''), max_chars=900)
            question = sanitize_for_azure_filter(data.get('question', ''), max_chars=500)
            ground_truth = sanitize_for_azure_filter(data.get('answer', ''), max_chars=800)

            if has_jailbreak_like_pattern(context) or has_jailbreak_like_pattern(question) or has_jailbreak_like_pattern(ground_truth):
                optimizer_total_input = build_tgd_optimizer_total_input(
                    optimizer_system_prompt=optimizer_system_prompt,
                    instruction=system_prompt.value,
                    question=question,
                    context=context,
                    gold_answer=ground_truth,
                    model_answer=None,
                    evaluation_instruction=None,
                    answer_feedback="[N/A] 잠재적 인젝션 패턴 포함 샘플",
                )
                optimization_logs.append({
                    'experiment_id': experiment_id,
                    'episode': episode,
                    'instruction': system_prompt.value,
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
                    'error_log': "[Skipped] potential jailbreak-like pattern in dataset sample",
                    'optimizer_model_nm': textgrad_backward_model,
                    'tester_model_nm': textgrad_forward_model,
                    'optimizer_system_prompt': optimizer_system_prompt,
                    'optimizer_total_input': optimizer_total_input,
                    'created_at': datetime.now()
                })
                print(f"[Warning] Episode {episode} 샘플 스킵 - jailbreak 유사 패턴 감지")
                continue

            try:
                # 답변 생성 및 평가
                inputs = f"Context: {context}\nQuestion: {question}"
                # prediction = engine.geneate() 와 같은 형태로 쓸 경우, system_prompt 와 계산 그래프가 끊긴다고 한다. TODO gpt의 의견의로 실제 확인 필요...
                # prediction = engine.generate(f"{system_prompt.value}\n{inputs}")
                query_var = tg.Variable(
                    inputs,
                    role_description="RAG 입력(context + question)",
                    requires_grad=False
                )
                prediction_var = model(query_var)
                prediction = prediction_var.value

                ############################### 차별점 ###############################################
                # TextGrad의 평가 (Loss)
                evaluation_instruction = build_hierarchical_evaluation_instruction(ground_truth)
                ######################################################################################
                loss = tg.TextLoss(evaluation_instruction)
                computed_loss = loss(prediction_var)
                losses.append(computed_loss)
                optimizer_total_input = build_tgd_optimizer_total_input(
                    optimizer_system_prompt=optimizer_system_prompt,
                    instruction=system_prompt.value,
                    question=question,
                    context=context,
                    gold_answer=ground_truth,
                    model_answer=prediction,
                    evaluation_instruction=evaluation_instruction,
                    answer_feedback=computed_loss.value,
                )
                raw_similarity = None
                if similarity_judge is not None:
                    try:
                        raw_similarity = similarity_judge(ground_truth, prediction)
                    except Exception as similarity_error:
                        raw_similarity = math.nan
                        print(f"[Warning] Episode {episode} Similarity 계산 실패: {similarity_error}")

                ragas_faithfulness_score = None
                ragas_answer_relevancy_score = None
                if ragas_judge is not None:
                    try:
                        ragas_result = ragas_judge.evaluate(
                            question=question,
                            answer=prediction,
                            context=context,
                            gold_answer=ground_truth,
                        )
                        ragas_reason = str(ragas_result.get('reason', ''))
                        if ragas_reason.startswith("Evaluation error:"):
                            ragas_faithfulness_score = math.nan
                            ragas_answer_relevancy_score = math.nan
                        else:
                            ragas_faithfulness_score = ragas_result.get('score')
                            ragas_answer_relevancy_score = ragas_result.get('relevancy_score')
                    except Exception as ragas_error:
                        ragas_faithfulness_score = math.nan
                        ragas_answer_relevancy_score = math.nan
                        print(f"[Warning] Episode {episode} RAGAS 평가 실패: {ragas_error}")

                # DB 저장용 로그: episode 번호를 main_train.py 기준으로 기록
                optimization_logs.append({
                    'experiment_id': experiment_id,
                    'episode': episode,
                    'instruction': system_prompt.value,
                    'question': question,
                    'context': context,
                    'model_answer': prediction,
                    'gold_answer': ground_truth,
                    'total_score': raw_similarity,
                    'raw_similarity': raw_similarity,
                    'ragas_faithfulness_score': ragas_faithfulness_score,
                    'ragas_answer_relevancy_score': ragas_answer_relevancy_score,
                    'answer_feedback': computed_loss.value,
                    'prompt_feedback': None,
                    'is_success': True,
                    'error_log': None,
                    'optimizer_model_nm': textgrad_backward_model,
                    'tester_model_nm': textgrad_forward_model,
                    'optimizer_system_prompt': optimizer_system_prompt,
                    'optimizer_total_input': optimizer_total_input,
                    'created_at': datetime.now()
                })
            except Exception as sample_error:
                root_error = extract_root_error_message(sample_error)
                if is_azure_content_filter_error(root_error):
                    error_note = f"[Azure Content Filter] {root_error}"
                else:
                    error_note = f"[Sample Evaluation Error] {root_error}"

                optimizer_total_input = build_tgd_optimizer_total_input(
                    optimizer_system_prompt=optimizer_system_prompt,
                    instruction=system_prompt.value,
                    question=question,
                    context=context,
                    gold_answer=ground_truth,
                    model_answer=None,
                    evaluation_instruction=None,
                    answer_feedback="[N/A] 답변 평가 실패",
                )
                optimization_logs.append({
                    'experiment_id': experiment_id,
                    'episode': episode,
                    'instruction': system_prompt.value,
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
                    'error_log': error_note,
                    'optimizer_model_nm': textgrad_backward_model,
                    'tester_model_nm': textgrad_forward_model,
                    'optimizer_system_prompt': optimizer_system_prompt,
                    'optimizer_total_input': optimizer_total_input,
                    'created_at': datetime.now()
                })
                print(f"[Warning] Episode {episode} 샘플 처리 실패 - {error_note}")
                continue

        successful_scores = []
        has_nan_score = False
        for log in optimization_logs[episode_log_start_idx:]:
            if not log.get('is_success'):
                continue
            score = log.get('total_score')
            if score is None:
                continue
            if isinstance(score, float) and math.isnan(score):
                has_nan_score = True
                continue
            successful_scores.append(score)

        if has_nan_score:
            episode_avg_score = math.nan
        elif successful_scores:
            episode_avg_score = sum(successful_scores) / len(successful_scores)
        else:
            episode_avg_score = None
        for idx in range(episode_log_start_idx, len(optimization_logs)):
            optimization_logs[idx]['dataset_size'] = len(batch)
            optimization_logs[idx]['avg_total_score'] = episode_avg_score

        # episode 종료 시점에 1회만 업데이트 (main_train.py의 episode 흐름과 유사)
        if not losses:
            for idx in range(episode_log_start_idx, len(optimization_logs)):
                if optimization_logs[idx].get('prompt_feedback') is None:
                    optimization_logs[idx]['prompt_feedback'] = "[N/A] 모든 샘플이 실패하여 업데이트를 건너뜀"
            print(f"[Warning] Episode {episode}/{episodes} 유효한 loss가 없어 업데이트를 건너뜁니다.")
            continue

        total_loss = losses[0]
        for loss_item in losses[1:]:
            total_loss = total_loss + loss_item
        try:
            total_loss.backward()

            compact_textgrad_gradients(system_prompt)

            ##################################### 차별점 #############################################
            # TextGrad textual feedback: 최적화 대상(system_prompt)에 대한 gradient 텍스트
            # -> episode 단위 피드백이므로, 이번 episode의 모든 row에 동일하게 기록
            prompt_feedback_text = build_hierarchical_prompt_feedback(
                gradient_text=system_prompt.get_gradient_text(),
                episode_logs=optimization_logs[episode_log_start_idx:],
            )
            ##########################################################################################

            for idx in range(episode_log_start_idx, len(optimization_logs)):
                optimization_logs[idx]['prompt_feedback'] = prompt_feedback_text

            optimizer.step()
            optimizer.zero_grad()

            print(f"Episode {episode}/{episodes} 완료 (dataset_size={len(batch)}, update=1회)")
            print(f"Backward engine model: {textgrad_backward_model} | Forward engine model: {textgrad_forward_model}")
            print(f"Current prompt: {system_prompt.value}")

        except Exception as e:
            root_error = extract_root_error_message(e)
            if is_azure_content_filter_error(root_error):
                fallback_prompt = (
                    "주어진 문맥에서 확인 가능한 정보만 사용해 간결하고 사실적으로 답변하세요. "
                    "근거가 부족하면 추측하지 말고 정보 부족을 명시하세요."
                )
                system_prompt.set_value(fallback_prompt)

                for idx in range(episode_log_start_idx, len(optimization_logs)):
                    optimization_logs[idx]['prompt_feedback'] = (
                        "[Fallback Applied] Azure content filter로 optimizer.step 차단되어 "
                        "안전한 기본 프롬프트로 대체함"
                    )

                optimizer.zero_grad()
                print(f"[Warning] Episode {episode}/{episodes} 업데이트 필터 차단")
                print("         - 안전한 fallback 프롬프트로 대체 후 다음 episode 진행")
                continue

            episode_error = f"[TextGrad Backward/Update Error] {root_error}"
            for idx in range(episode_log_start_idx, len(optimization_logs)):
                optimization_logs[idx]['prompt_feedback'] = "[N/A] Prompt feedback unavailable due to backward/update failure."
                optimization_logs[idx]['is_success'] = False
                optimization_logs[idx]['error_log'] = episode_error

            optimizer.zero_grad()
            print(f"[Warning] Episode {episode}/{episodes} 업데이트 실패")
            print(f"         - Root Cause: {root_error}")
            continue

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
    print_step("=== TextGrad Improve 프롬프트 최적화 시작 ===")
    main()