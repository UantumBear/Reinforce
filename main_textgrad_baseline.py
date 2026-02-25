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
"""

import os
import re
from datetime import datetime
import textgrad as tg
from utils.log.console import print_step

from datafile.data_loader import load_dataset
from infrastructure.llm_client import setup_lms
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


def main():
    patch_textgrad_openai_compatibility()

    print_step("0. [Settings] 설정 초기화")
    Settings.setup()
    
    print_step("1. [Infrastructure] LLM 연결 설정")
    setup_lms()
    
    print_step("2. 데이터 로드")
    dataset_name = "didi0di/klue-mrc-ko-rag-cot"
    train_dataset = load_dataset(dataset_name=dataset_name, sample_size=3)
    
    if not train_dataset:
        print("[팅FAIL] 데이터 로드 실패")
        return
    
    print_step("3. TextGrad 환경 설정 및 엔진 초기화")
    # TextGrad baseline용 experiment_id 생성
    current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
    experiment_id = f"textgrad_baseline_{current_time}"
    
    # TextGrad 엔진 초기화 (현재 textgrad 버전 호환)
    textgrad_model = Settings.AZURE_GPTO4_MINI_DEPLOYMENT or Settings.TESTER_MODEL
    os.environ["AZURE_OPENAI_API_KEY"] = Settings.AZURE_OPENAI_API_KEY or ""
    os.environ["AZURE_OPENAI_API_BASE"] = Settings.AZURE_OPENAI_ENDPOINT or ""
    os.environ["AZURE_OPENAI_API_VERSION"] = Settings.AZURE_OPENAI_API_VERSION or ""
    os.environ["OPENAI_API_KEY"] = Settings.AZURE_OPENAI_API_KEY or ""
    engine = tg.get_engine(f"azure-{textgrad_model}")
    tg.set_backward_engine(engine)
    
    print_step("4. TextGrad 최적화 실행")
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
    model = tg.BlackboxLLM(engine=engine, system_prompt=system_prompt)

    optimizer = tg.TGD(parameters=list(model.parameters()), engine=engine)

    # 4. 최적화 루프
    # [설계 의도]
    # - main_train.py와 용어를 맞추기 위해, 여기서는 "1 episode = train_dataset 전체 평가 + 1회 업데이트"로 정의한다.
    # - 즉, TextGrad의 mini-batch 개념을 사용하지 않고 episode 단위 업데이트를 사용한다.
    episodes = 3
    batch_size = len(train_dataset)  # episode 기반 설계에서는 전체 데이터셋을 한 번에 묶는다.
    optimization_logs = []  # DB 저장용 로그 버퍼

    print(f"--- TextGrad Baseline Optimization 및 DB 저장 시작 ---")

    for episode in range(1, episodes + 1):
        # episode 기반 설계: batch는 항상 전체 train_dataset
        batch = train_dataset[:batch_size]
        losses = []
        episode_log_start_idx = len(optimization_logs)

        for data in batch:
            context = sanitize_for_azure_filter(data.get('context', ''), max_chars=2500)
            question = sanitize_for_azure_filter(data.get('question', ''), max_chars=500)
            ground_truth = sanitize_for_azure_filter(data.get('answer', ''), max_chars=800)

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

                # TextGrad의 평가 (Loss)
                evaluation_instruction = (
                    f"정답(Ground Truth): {ground_truth}\n"
                    "모델 답변과의 차이점을 간결하게 비교하고, 개선점을 제안해주세요."
                )
                loss = tg.TextLoss(evaluation_instruction)
                computed_loss = loss(prediction_var)
                losses.append(computed_loss)

                # DB 저장용 로그: episode 번호를 main_train.py 기준으로 기록
                optimization_logs.append({
                    'experiment_id': experiment_id,
                    'episode': episode,
                    'instruction': system_prompt.value,
                    'question': question,
                    'context': context,
                    'model_answer': prediction,
                    'gold_answer': ground_truth,
                    'answer_feedback': computed_loss.value,
                    'prompt_feedback': None,
                    'is_success': True,
                    'error_log': None,
                    'optimizer_model_nm': textgrad_model,
                    'tester_model_nm': textgrad_model,
                    'created_at': datetime.now()
                })
            except Exception as sample_error:
                root_error = extract_root_error_message(sample_error)
                if is_azure_content_filter_error(root_error):
                    error_note = f"[Azure Content Filter] {root_error}"
                else:
                    error_note = f"[Sample Evaluation Error] {root_error}"

                optimization_logs.append({
                    'experiment_id': experiment_id,
                    'episode': episode,
                    'instruction': system_prompt.value,
                    'question': question,
                    'context': context,
                    'model_answer': None,
                    'gold_answer': ground_truth,
                    'answer_feedback': "[N/A] 답변 평가 실패",
                    'prompt_feedback': "[N/A] 샘플 처리 실패",
                    'is_success': False,
                    'error_log': error_note,
                    'optimizer_model_nm': textgrad_model,
                    'tester_model_nm': textgrad_model,
                    'created_at': datetime.now()
                })
                print(f"[Warning] Episode {episode} 샘플 처리 실패 - {error_note}")
                continue

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

            # TextGrad textual feedback: 최적화 대상(system_prompt)에 대한 gradient 텍스트
            # -> episode 단위 피드백이므로, 이번 episode의 모든 row에 동일하게 기록
            prompt_feedback_text = system_prompt.get_gradient_text().strip()
            if not prompt_feedback_text:
                prompt_feedback_text = "[N/A] TextGrad prompt feedback is empty."

            for idx in range(episode_log_start_idx, len(optimization_logs)):
                optimization_logs[idx]['prompt_feedback'] = prompt_feedback_text

            optimizer.step()
            optimizer.zero_grad()

            print(f"Episode {episode}/{episodes} 완료 (dataset_size={len(batch)}, update=1회)")
            print(f"Current prompt: {system_prompt.value}")

        except Exception as e:
            root_error = extract_root_error_message(e)
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
                optimizer_model_nm=log_data['optimizer_model_nm'],
                optimizer_model_provider="azure",
                tester_model_nm=log_data['tester_model_nm'],
                tester_model_provider="azure",
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