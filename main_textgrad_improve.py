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
- 2026.03.30 논문 방식 Validation 캐싱 및 Forward 분기 구현
    [변경 1] Validation 루프 - 현재 프롬프트 캐싱 방식 적용 (논문 근거)
        이전: 매 iteration마다 현재 프롬프트 + 후보 프롬프트 둘 다 평가 (val_count × 2번 LLM 호출)
        이후: Episode 시작 시 현재 프롬프트를 1회 평가해 캐시(cached_val_score_current),
              매 iteration은 후보 프롬프트만 평가 (val_count × 1번),
              프롬프트 채택 시 캐시 점수도 갱신하여 다음 iteration의 기준값으로 사용
        근거: "기존 프롬프트의 점수는 이미 알고 있거나 캐싱되어 있으므로
               새로운 후보 프롬프트에 대해서만 Validation 데이터셋 크기만큼 LLM을 호출" (논문)
    [변경 2] Train Forward 분기 명확화
        이전: for 루프로 test_time_updates번 무조건 반복 후 결과 합산
        이후: if/else 분기로 완전히 분리
              - GPQA/MMLU/HQH (is_multiple_choice and test_time_updates > 1):
                  3회 생성 + Majority Voting → Accuracy (현재 실험에서는 실행되지 않는 분기, 구조만 유지)
              - GSM8k / 그 외 (else):
                  model(query_var) 단 1회 호출 (논문 기준 프롬프트 최적화 루프)
                  GSM8k면 compute_gsm8k_accuracy, 그 외면 accuracy=None

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

# ============================================================================
# [성능 프로파일링] 파일 실행 시작 시점 기록
# ============================================================================
import time
_SCRIPT_START_TIME = time.time()

def _print_elapsed(label):
    """파일 시작부터 현재까지 경과 시간 출력"""
    elapsed = time.time() - _SCRIPT_START_TIME
    print(f"[⏱️  {elapsed:.2f}s] {label}")


import textgrad as tg
_print_elapsed(f"textgrad import 완료")


_print_elapsed("기본 라이브러리 import 시작")
import os
import re
import math
import random
import atexit
import traceback
from datetime import datetime
_print_elapsed("기본 라이브러리 완료")

_print_elapsed("textgrad 내부 모듈 import 시작")
from textgrad.autograd.string_based_ops import StringBasedFunction
from textgrad.optimizer.optimizer import TextualGradientDescentwithMomentum
_print_elapsed("textgrad 내부 모듈 완료")

_print_elapsed("로컬 utils 모듈 import 시작")
from utils.environment.experiment import TextGradExperiment
from utils.environment.textgrad_log_builder import (
    create_base_log,
    create_skip_log,
    create_success_log,
    create_error_log,
    extract_momentum_history,
    build_tgd_optimizer_total_input,
    build_backward_judge_total_input,
)
from utils.log.console import print_step
_print_elapsed("utils 모듈 완료")

_print_elapsed("datafile, infrastructure import 시작")
# from datafile.data_loader import load_dataset
from infrastructure.llm_client import get_textgrad_backward_engine, get_textgrad_forward_engine
_print_elapsed("datafile, infrastructure 완료")

_print_elapsed("metrics.judges import 시작")
from metrics.judges.ragas_failthfulness_judge import RagasFaithfulnessJudge
_print_elapsed("RagasFaithfulnessJudge 완료")

_print_elapsed("conf, models, db import 시작")
from conf.config import Settings
# 로그 저장을 위한 import (main_train.py 방식)
from models.rl_optimization_log import RlOptimizationLog
from db.connection.pg_client import pg_client
_print_elapsed("conf, models, db 완료")

# 공통으로 사용 가능 한 utils 함수들
_print_elapsed("나머지 utils 함수들 import 시작")
from utils.llm_errors.error_parsers import extract_root_error_message
from utils.llm_errors.error_debugger import debug_individual_backward_samples
from utils.llm_safety.azure_prompt_filters import has_jailbreak_like_pattern
from utils.text.normalization import normalize_text_field
from utils.llm_patches.textgrad_patches import patch_textgrad_openai_compatibility, patch_textgrad_momentum_compatibility
from utils.llm_patches.textgrad_info import get_tgd_optimizer_system_prompt, stringify_tgd_update_prompt
_print_elapsed("나머지 utils 완료")

# GSM8k 평가 함수
_print_elapsed("Judge 함수들 import 시작")
from metrics.judges.gsm8k_judge import parse_integer_answer

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

_print_elapsed("Judge 함수들 완료")
# GSM8k 평가 유틸리티 (수학 문제 데이터셋용)
from metrics.judges.gsm8k_judge import string_based_equality_fn

# Import 완료 시점 출력
_print_elapsed("모든 라이브러리 Import 완료")


def main():
    _print_elapsed("main() 함수 진입")
    
    print_step("0. [Settings] TextGrad 실험 환경 설정")
    # 패치를 명시적으로 먼저 적용
    TextGradExperiment.apply_patches()  # ← 여기서만 실행
    EXPERIMENT_INS = TextGradExperiment(mode='improve')
    _print_elapsed("실험 환경 설정 완료")
    
    print_step("1. [Settings] 기본 백엔드 설정 초기화")
    Settings.setup()
    _print_elapsed("백엔드 설정 초기화 완료")
    

    print_step("2. 데이터 로드 및 Train/Validation 분할")
    dataset, train_pool, validation_dataset = EXPERIMENT_INS.load_and_split_data()
    _print_elapsed("데이터 로드 완료")
    
    # [TextGrad 논문] Test-time updates 설정 (데이터셋별 자동 최적화)
    # - GPQA/MMLU/HQH: 3번 답변 생성 + Majority Voting (multiple-choice)
    # - 그 외 데이터셋: 1번 생성 (일반 RAG/생성 태스크)
    test_time_updates = EXPERIMENT_INS.get_test_time_updates()
    print(f"[✓] Test-time updates: {test_time_updates}번 (데이터셋: {EXPERIMENT_INS.default_dataset_name})")
    
    print_step("3. TextGrad 환경 설정 및 엔진 초기화")
    # TextGrad experiment_id 생성
    current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
    experiment_id = f"textgrad_{EXPERIMENT_INS.mode}_{current_time}"
    
    # ============================================================================
    # TextGrad 엔진 초기화: 2가지 역할로 나뉜 LLM
    # ============================================================================
    # get_textgrad_*_engine()은 (engine, model_name) 튜플을 반환:
    #   - engine: 실제 LLM API를 호출하는 실행 객체 (예: ChatOpenAI 인스턴스)
    #   - model_name: 사용된 모델명 문자열 (예: "gpt-4o-mini") - DB 로그 기록용
    
    # 1️⃣ forward_engine (답변 생성자 LLM)
    #    ┌─────────────────────────────────────────────────┐
    #    │ 역할: 문제를 풀고 답변 생성 (단 1가지!)        │
    #    │ 예시: "5 + 3 = 8개입니다"                      │
    #    │ 사용처: model(query) 호출 시만                 │
    #    └─────────────────────────────────────────────────┘
    forward_engine, textgrad_forward_model_nm = get_textgrad_forward_engine()
    
    # 2️⃣ backward_engine (선생님 LLM - 평가/피드백/최적화 모두 담당!)
    #    ┌─────────────────────────────────────────────────────────────────┐
    #    │ 역할 A: 평가 (Evaluation Forward)                             │
    #    │   - TextLoss 사용 시: loss(prediction) 호출하면 이 엔진 사용  │
    #    │   - 예시: "이 답변은 정확합니다. 10/10점"                      │
    #    │   ★ 주의: forward_engine 아님!                                │
    #    ├─────────────────────────────────────────────────────────────────┤
    #    │ 역할 B: 피드백 생성 (Evaluation Backward)                     │
    #    │   - loss.backward() 호출하면 이 엔진 사용                     │
    #    │   - 예시: "프롬프트에 '단계별로 풀이하라' 추가 필요"          │
    #    ├─────────────────────────────────────────────────────────────────┤
    #    │ 역할 C: 프롬프트 개선 (Optimizer)                             │
    #    │   - optimizer.step() 호출하면 이 엔진 사용                    │
    #    │   - 예시: "문제를 풀어라" → "문제를 단계별로 풀어라"          │
    #    └─────────────────────────────────────────────────────────────────┘
    #    ★ 핵심: backward_engine 하나가 A, B, C 역할 모두 담당!
    backward_engine, textgrad_backward_model_nm = get_textgrad_backward_engine()
    
    # TextGrad 라이브러리에 backward_engine 전역 설정
    # 이후 모든 평가/피드백/최적화는 이 backward_engine 사용!
    tg.set_backward_engine(backward_engine)
    _print_elapsed("TextGrad 엔진 초기화 완료")
    
    print_step("4. TextGrad 최적화 실행")
    similarity_judge = create_similarity_judge()
    embedding_model_nm = similarity_judge.embedding_model_nm if similarity_judge else None
    print(f"[DEBUG] Similarity Judge 초기화: {similarity_judge is not None}")
    print(f"[DEBUG] Embedding Model Name: {embedding_model_nm}")
    ragas_judge = create_ragas_judge()
    _print_elapsed("Judge 모델 초기화 완료")
    
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

    # 3. 최적화 대상 정의 (system_prompt)
    # ★ 이 system_prompt가 forward_engine(답변 생성자 LLM)에게 전달됨
    # ★ TextGrad의 최종 목표: 이 프롬프트를 개선해서 답변 품질을 높이기!
    # 
    # initial_prompt는 데이터셋에 따라 TextGradExperiment에서 자동 설정
    # 예: "Solve the following math problem step by step."
    initial_prompt = EXPERIMENT_INS.get_initial_prompt()
    print(f"[✓] 초기 프롬프트: {initial_prompt}..." if len(initial_prompt) > 100 else f"[✓] 초기 프롬프트: {initial_prompt}")
    
    # requires_grad=True: optimizer가 이 프롬프트를 개선할 수 있도록 설정
    # role_description: backward_engine(평가자 LLM)이 피드백 생성 시 참고
    
    ##### 차별점 #####
    # [Baseline] experiment_context = "" (실험 컨텍스트 미제공, 논문 재현)
    # [Improve] experiment_context = get_gsm8k_experiment_context() (실험 목표 전달)
    experiment_context = EXPERIMENT_INS.get_experiment_context()
    ###################
    
    role_desc = f"system prompt to the language model\n{experiment_context}"
    
    system_prompt = tg.Variable(
        initial_prompt, 
        requires_grad=True, 
        role_description=role_desc
    )

    # [중요] BlackboxLLM: forward_engine(답변 생성자 LLM)을 감싼 wrapper
    # - engine.generate()를 직접 호출하면 system_prompt와 계산 그래프가 연결되지 않음
    # - BlackboxLLM을 사용하면 system_prompt가 최적화 대상으로 등록됨
    # - model(query) 호출 시 내부적으로 forward_engine이 답변 생성
    # ★ model ≠ forward_engine (model은 forward_engine을 사용하는 wrapper)
    model = tg.BlackboxLLM(engine=forward_engine, system_prompt=system_prompt)

    momentum_window = int(
        os.getenv("TEXTGRAD_MOMENTUM_WINDOW", os.getenv("TEXTGRAD_MOMENTUM_GRADIENT_MEMORY", "3"))
    )

    # ============================================================================
    # 4. Optimizer 생성 - backward_engine(평가자 LLM)을 사용하여 프롬프트 개선
    # ============================================================================
    # TextualGradientDescentwithMomentum:
    # - parameters: 개선할 대상 (system_prompt)
    # - engine: backward_engine(평가자 LLM) 사용
    # - optimizer.step() 호출 시:
    #   1. system_prompt.gradients에서 피드백(gradient) 가져오기
    #   2. backward_engine에게 "이 피드백을 바탕으로 새 프롬프트 만들어줘" 요청
    #   3. backward_engine이 개선된 프롬프트 생성
    #   4. system_prompt 업데이트
    # 
    # 예: 피드백 "단계별 설명 필요" → backward_engine: "문제를 단계별로 풀어라"
    
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
        engine=backward_engine,  # ← backward_engine(평가자 LLM)이 프롬프트 개선!
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
    #
    # [설계 방침] episode 개념 없음
    # 논문에는 episode 개념이 없습니다. 반복 단위는 iteration 하나입니다.
    # DB의 episode 컬럼에는 iteration 번호와 동일한 값을 저장합니다. (episode == iteration)

    total_iterations = EXPERIMENT_INS.default_iterations
    batch_size = EXPERIMENT_INS.default_batch_size

    optimization_logs = []  # DB 저장용 로그 버퍼

    # ======================================================================
    # [비상 저장] 이터레이션 예외 / Ctrl+C / 프로세스 종료 시 로그 유실 방지
    # ======================================================================
    #
    # ── 왜 매개변수 없이 optimization_logs를 쓸 수 있는가? ──────────────────
    #
    #   _do_db_save / _emergency_save 는 main() 함수 '안에서' 정의된 함수입니다.
    #   Python은 내부 함수가 자신을 감싼 바깥 스코프의 변수를 자동으로
    #   '참조(capture)' 하도록 만들어졌습니다. 이를 클로저(Closure)라 합니다.
    #
    #   핵심: 값을 복사하는 게 아니라 같은 메모리 주소를 공유합니다.
    #
    #   예시:
    #       optimization_logs = []          # main() 스코프, 주소 0x1234
    #       def _do_db_save():
    #           for log in optimization_logs  # 동일한 주소 0x1234 참조
    #
    #   따라서 루프에서 optimization_logs.append(...) 로 항목이 쌓이면,
    #   나중에 _do_db_save() 가 호출될 때도 쌓인 항목이 그대로 보입니다.
    #
    # ── _save_done 을 왜 [False] 리스트로 만들었는가? ──────────────────────
    #
    #   _save_done = False 처럼 단순 변수로 두면 문제가 생깁니다.
    #   내부 함수에서 _save_done = True 로 '재할당'하는 순간,
    #   Python은 바깥 변수를 바꾸는 게 아니라 내부 함수 로컬에
    #   새 변수를 만들어버립니다. (바깥 _save_done 은 여전히 False)
    #
    #   해결책 A: _save_done = [False] → 재할당 대신 _save_done[0] = True 로
    #             리스트 내용(내부값)만 바꾸면, 재할당이 없으므로 바깥과
    #             동일한 리스트 객체를 그대로 공유합니다.
    #
    #   해결책 B: nonlocal _save_done 선언 후 _save_done = True 도 동작하지만,
    #             [False] 패턴이 더 짧아서 이 방식을 채택했습니다.
    #
    # ── atexit 동작 원리 ────────────────────────────────────────────────────
    #
    #   atexit.register(fn) 은 Python 인터프리터가 종료되기 직전에 fn을
    #   자동 호출하도록 예약합니다.
    #
    #   적용되는 경우:
    #     - 이터레이션 레벨 예외 (backward/optimizer 등) → main() 밖으로 전파 → 종료
    #     - Ctrl+C (KeyboardInterrupt)                  → 전파 → 종료
    #     - 정상 완료 (→ 정상 저장 후 _save_done[0]=True → 이미 저장됐으면 스킵)
    #
    #   적용 안 되는 경우:
    #     - 작업 관리자 등으로 프로세스 강제 종료 (SIGKILL) → atexit 실행 불가
    #
    # ======================================================================
    _save_done = [False]  # 정상 저장 완료 여부. 리스트인 이유: 위 주석 참조.

    def _do_db_save():
        """DB 저장 - 정상 종료 및 비상 종료 시 공통으로 사용"""
        if not optimization_logs:
            print("[!] 저장할 로그가 없습니다.")
            return
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
                    forward_tester_llm_call_cnt=log_data.get('forward_tester_llm_call_cnt', 0),
                    backward_judge_llm_call_cnt=log_data.get('backward_judge_llm_call_cnt', 0),
                    backward_optimizer_llm_call_cnt=log_data.get('backward_optimizer_llm_call_cnt', 0),
                    validation_info=log_data.get('validation_info'),
                    validation_accuracy=log_data.get('validation_accuracy'),
                    validation_dataset_size=log_data.get('validation_dataset_size'),
                    dataset_size=log_data.get('dataset_size'),
                    avg_total_score=log_data.get('avg_total_score'),
                    dataset_nm=log_data.get('dataset_nm'),
                    optimizer_model_nm=log_data['optimizer_model_nm'],
                    optimizer_model_provider="azure",
                    tester_model_nm=log_data['tester_model_nm'],
                    tester_model_provider="azure",
                    embedding_model_nm=log_data.get('embedding_model_nm'),
                    optimizer_system_prompt=log_data.get('optimizer_system_prompt'),
                    optimizer_total_input=log_data.get('optimizer_total_input'),
                    evaluation_instruction=log_data.get('evaluation_instruction'),
                    backward_judge_total_input=log_data.get('backward_judge_total_input'),
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

    def _emergency_save():
        """비정상 종료 / Ctrl+C 시 atexit에서 자동 호출되는 비상 저장 함수"""
        if _save_done[0]:
            return  # 이미 정상 저장된 경우 스킵
        print("\n[!] 비정상 종료 또는 인터럽트 감지 - 현재까지의 로그를 저장합니다...")
        _do_db_save()

    atexit.register(_emergency_save)

    print(f"--- TextGrad Baseline Optimization (논문 재현) 시작 ---")
    print(f"Total Iterations: {total_iterations}, Batch size: {batch_size}")

    # -----------------------------------------------------------------------
    # [논문 구현] 현재 프롬프트의 Validation 점수 초기 캐싱 (루프 진입 전 1회)
    # "기존 프롬프트의 점수는 이미 알고 있거나 캐싱되어 있으므로
    #  새로운 후보 프롬프트에 대해서만 Validation 데이터셋 크기만큼 LLM을 호출합니다." (논문)
    # - reject 시: 캐시 그대로 유지 (재평가 없음)
    # - accept 시: cached_val_score_current = val_score_candidate 로 갱신
    # -----------------------------------------------------------------------
    print(f"\n[초기 캐싱] 초기 프롬프트 Validation 점수 평가...")
    cached_val_score_current = 0.0
    cached_val_count = 0
    initial_validation_info = {}  # 초기 프롬프트의 validation 샘플 정보

    if is_gsm8k:
        for val_idx, val_data in enumerate(validation_dataset, 1):
            # ※주의: val_context는 RAG 문서 자료이며, TextGrad의 <CONTEXT> 태그(이전 최적화 피드백)와 무관합니다.
            val_context = normalize_text_field(val_data.get('context', ''))
            val_question = normalize_text_field(val_data.get('question', ''))
            val_gt = normalize_text_field(val_data.get('answer', ''))
            val_inputs = f"Question: {val_question}" if not val_context.strip() else f"Context: {val_context}\nQuestion: {val_question}"
            try:
                val_var = tg.Variable(val_inputs, role_description="Validation input", requires_grad=False)
                pred = model(val_var).value
                # GSM8k 정확도 계산: 숫자 추출 후 비교
                pred_num = parse_integer_answer(pred)
                gt_num = parse_integer_answer(val_gt)
                # 파싱 실패는 무조건 오답(0점) 처리
                if pred_num is None or gt_num is None:
                    score = 0.0
                else:
                    score = 1.0 if pred_num == gt_num else 0.0
                
                # Validation 샘플 정보 수집
                initial_validation_info[str(val_idx - 1)] = {
                    "Q": val_question,
                    "A": pred,
                    "GA": val_gt,
                    "score": score
                }
                
                cached_val_score_current += score
                cached_val_count += 1
            except Exception as e:
                print(f"  ⚠️ 초기 캐시 평가 샘플 [{val_idx}] 에러:")
                print(f"     에러 타입: {type(e).__name__}")
                print(f"     에러 메시지: {e}")
                print(f"     질문: {val_question[:100]}...")
                print(f"\n  [상세 스택 트레이스]")
                traceback.print_exc()
                print()
                continue
    elif is_multiple_choice:
        # GPQA 등 multiple-choice (현재 실험에서 실행되지 않는 분기, 구조만 유지)
        for val_idx, val_data in enumerate(validation_dataset, 1):
            # ※주의: val_context는 RAG 문서 자료이며, TextGrad의 <CONTEXT> 태그(이전 최적화 피드백)와 무관합니다.
            val_context = normalize_text_field(val_data.get('context', ''))
            val_question = normalize_text_field(val_data.get('question', ''))
            val_gt = normalize_text_field(val_data.get('answer', ''))
            val_inputs = f"Question: {val_question}" if not val_context.strip() else f"Context: {val_context}\nQuestion: {val_question}"
            try:
                val_var = tg.Variable(val_inputs, role_description="Validation input", requires_grad=False)
                pred = model(val_var).value
                score = similarity_judge(val_gt, pred) if similarity_judge else 0.0
                
                # Validation 샘플 정보 수집
                initial_validation_info[str(val_idx - 1)] = {
                    "Q": val_question,
                    "A": pred,
                    "GA": val_gt,
                    "score": score
                }
                
                cached_val_score_current += score
                cached_val_count += 1
            except Exception as e:
                print(f"  ⚠️ 초기 캐시 평가 샘플 [{val_idx}] 에러:")
                print(f"     에러 타입: {type(e).__name__}")
                print(f"     에러 메시지: {e}")
                print(f"     질문: {val_question[:100]}...")
                print(f"\n  [상세 스택 트레이스]")
                traceback.print_exc()
                print()
                continue
    else:
        for val_idx, val_data in enumerate(validation_dataset, 1):
            # ※주의: val_context는 RAG 문서 자료이며, TextGrad의 <CONTEXT> 태그(이전 최적화 피드백)와 무관합니다.
            val_context = normalize_text_field(val_data.get('context', ''))
            val_question = normalize_text_field(val_data.get('question', ''))
            val_gt = normalize_text_field(val_data.get('answer', ''))
            val_inputs = f"Question: {val_question}" if not val_context.strip() else f"Context: {val_context}\nQuestion: {val_question}"
            try:
                val_var = tg.Variable(val_inputs, role_description="Validation input", requires_grad=False)
                pred = model(val_var).value
                score = similarity_judge(val_gt, pred) if similarity_judge else 0.0
                
                # Validation 샘플 정보 수집
                initial_validation_info[str(val_idx - 1)] = {
                    "Q": val_question,
                    "A": pred,
                    "GA": val_gt,
                    "score": score
                }
                
                cached_val_score_current += score
                cached_val_count += 1
            except Exception as e:
                print(f"  ⚠️ 초기 캐시 평가 샘플 [{val_idx}] 에러:")
                print(f"     에러 타입: {type(e).__name__}")
                print(f"     에러 메시지: {e}")
                print(f"     질문: {val_question[:100]}...")
                print(f"\n  [상세 스택 트레이스]")
                traceback.print_exc()
                print()
                continue

    if cached_val_count > 0:
        cached_val_score_current /= cached_val_count
    print(f"[초기 캐싱] 완료: {cached_val_score_current:.4f} ({cached_val_count}개 평가)")
    print(f"[초기 캐싱] Validation 샘플 정보: {len(initial_validation_info)}개 수집")

    # ========== [TextGrad 논문 재현 루프 시작] ==========
    for iteration in range(1, total_iterations + 1):
        print(f"\n{'='*80}")
        print(f"Iteration {iteration}/{total_iterations} 시작")
        print(f"{'='*80}")

        # DB 로그용 공통 필드 생성
        # episode 컬럼 = iteration 번호 (논문에 episode 개념 없음, DB 컬럼 재사용)
        base_log = create_base_log(
            experiment_id, 
            iteration, 
            textgrad_backward_model_nm, 
            textgrad_forward_model_nm, 
            embedding_model_nm,
            dataset_nm=EXPERIMENT_INS.default_dataset_name
        )

        iteration_log_start_idx = len(optimization_logs)

        # 1) Train: 배치 크기만큼 무작위 복원 추출
        batch = random.choices(train_pool, k=batch_size)  # 복원 추출

        losses = []
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
                # [Forward Model 입력 구성]
                # - GSM8k: context 없음 → "Question: ..."
                # - NASA/KLUE: context 있음 → "Context: ...\nQuestion: ..."
                # ※주의: 이것은 Forward Model에게 주는 입력입니다.
                #   TextGrad Optimizer의 <CONTEXT> 태그(이전 피드백)와는 다릅니다.
                if context.strip():
                    inputs = f"Context: {context}\nQuestion: {question}"
                else:
                    inputs = f"Question: {question}"

                query_var = tg.Variable(inputs, role_description="RAG 입력", requires_grad=False)

                # [TextGrad 논문] 데이터셋별 Forward 전략 분기
                # - GSM8k (프롬프트 최적화): test_time_updates=1, 단일 생성
                # - GPQA/MMLU/HQH (솔루션 최적화): test_time_updates=3, 다중 생성 + Majority Voting
                #   ※ GPQA는 현재 실험에서 사용하지 않으나, 분기 구조는 유지합니다.

                if is_multiple_choice and test_time_updates > 1:
                    # [GPQA/MMLU/HQH 경로] - 솔루션 최적화 루프
                    # test_time_updates(=3)번 forward_engine으로 답변 생성 후 Majority Voting
                    # ※ 현재 실험에서는 실행되지 않는 분기입니다.
                    test_time_predictions = []
                    test_time_choices = []
                    first_prediction_var = None

                    for update_idx in range(test_time_updates):
                        pred_var = model(query_var)  # ← forward_engine(답변 생성자 LLM) 호출
                        pred_text = pred_var.value
                        test_time_predictions.append(pred_text)

                        if update_idx == 0:
                            first_prediction_var = pred_var

                        choice = extract_choice_from_answer(pred_text)
                        test_time_choices.append(choice)

                    prediction_var = first_prediction_var
                    prediction = test_time_predictions[0]

                    # Majority voting으로 최종 답 선택
                    final_choice = majority_vote(test_time_choices)
                    correct_choice = extract_choice_from_answer(ground_truth)
                    accuracy = compute_accuracy(final_choice, correct_choice)
                    print(f"[Accuracy - Multiple Choice] Predictions: {test_time_choices} -> Final: {final_choice}, Correct: {correct_choice}, Acc: {accuracy}")

                else:
                    # [GSM8k / 그 외 경로] - 프롬프트 최적화 루프
                    # test_time_updates=1: Model Forward 1회만 호출 (논문 기준)
                    
                    # ============================================================================
                    # ★ Step 1: Model Forward - 답변 생성자 LLM(forward_engine)이 답변 생성
                    # ============================================================================
                    # model(query_var) 호출 시:
                    # 1. BlackboxLLM이 내부적으로 forward_engine 호출
                    # 2. forward_engine에게 system_prompt + query 전달
                    # 3. forward_engine(답변 생성자 LLM)이 답변 생성
                    # 
                    # 예: "5 + 3은 몇 개? 단계별로 풀어라" → forward_engine: "5 + 3 = 8개"
                    prediction_var = model(query_var)  # ← forward_engine(답변 생성자 LLM) 호출!
                    prediction = prediction_var.value
                    
                    # ★ accuracy는 StringBasedFunction 결과를 변환하여 사용 (아래에서 설정)
                    accuracy = None
                
                # ★ Step 2: Evaluation Function 생성 (평가 방식 선택)
                # [정의] evaluation_instruction: 평가자 LLM(backward_engine)에게 건네는 채점 가이드라인
                # 데이터셋(gsm8k 등)과 모드(baseline/improve)에 따라 다른 평가 방식 사용
                
                # 점수 계산 (비평가 LLM에게 참고 지표로도 전달)
                raw_similarity = None
                if similarity_judge is not None:
                    try:
                        raw_similarity = similarity_judge(ground_truth, prediction)
                        print(f"[Debug] Raw similarity score: {raw_similarity}")
                    except Exception:
                        raw_similarity = math.nan
                
                # [조건부 처리] GSM8k baseline: StringBasedFunction (논문 방식) / 그 외: TextLoss (기존 방식)
                # evaluation_instruction 초기화 (GSM8k StringBasedFunction 케이스에서는 None)
                evaluation_instruction = None
                
                if is_gsm8k and EXPERIMENT_INS.mode == 'baseline':
                    # [논문 재현] StringBasedFunction 사용:
                    # - Evaluation Forward: Python 함수로 0/1 계산 (backward_engine 호출 X, 비용 절감!)
                    # - Evaluation Backward: backward_engine(평가자 LLM)으로 피드백 생성
                    # 
                    # 흐름: forward_engine이 답변 생성 → Python이 정답 체크 → backward_engine이 피드백
                    ground_truth_var = tg.Variable(
                        ground_truth, 
                        role_description="the correct answer for the math problem",
                        requires_grad=False
                    )
                    eval_fn = StringBasedFunction(
                        string_based_equality_fn,
                        function_purpose="Checks if the prediction is correct by comparing the numerical answer"
                    )
                    computed_loss = eval_fn(
                        inputs=dict(prediction=prediction_var, ground_truth_answer=ground_truth_var),
                        response_role_description="Whether the prediction is correct (1) or not (0)"
                    )
                    losses.append(computed_loss)
                    print(f"[StringBasedFunction] Result: {computed_loss.value} (0=wrong, 1=correct)")
                    
                    # ★ Accuracy 변환: StringBasedFunction 결과를 재사용 (중복 계산 제거!)
                    # computed_loss.value는 "0" 또는 "1" (문자열 또는 int)
                    # → float로 변환하여 DB 저장용 accuracy로 사용
                    try:
                        accuracy = float(computed_loss.value)
                        print(f"[Accuracy - GSM8k] From StringBasedFunction: {accuracy}")
                    except (ValueError, TypeError):
                        accuracy = None
                        print(f"[Warning] Failed to convert loss to accuracy: {computed_loss.value}")
                        
                else:
                    # [기존 방식] TextLoss 사용:
                    # - Evaluation Forward: backward_engine(평가자 LLM)이 답변 평가하고 점수 생성
                    # - Evaluation Backward: backward_engine(평가자 LLM)이 피드백 생성
                    # 
                    # 흐름: forward_engine이 답변 생성 → backward_engine이 평가 → backward_engine이 피드백
                    # (총 LLM 호출 3번 vs StringBasedFunction은 2번 - 비용 1/3 절감!)
                    
                    ############################### 차별점 ###############################################
                    # [Improve] similarity_score를 참고 지표로 함께 전달하여 critic LLM이 보조 판단에 활용
                    evaluation_instruction = EXPERIMENT_INS.get_objective_function(
                        ground_truth,
                        similarity_score=raw_similarity
                    )
                    ######################################################################################
                    
                    loss = tg.TextLoss(evaluation_instruction)
                    computed_loss = loss(prediction_var)  # ← backward_engine(평가자 LLM)이 평가
                    losses.append(computed_loss)
                
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
                # backward_judge_total_input 생성 (evaluation_instruction이 있을 때만)
                backward_judge_total_input = None
                if evaluation_instruction is not None:
                    backward_judge_total_input = build_backward_judge_total_input(
                        evaluation_instruction=evaluation_instruction,
                        prediction=prediction
                    )
                
                optimization_logs.append(create_success_log(
                    base_log, system_prompt.value, question, context, ground_truth,
                    prediction, computed_loss.value, raw_similarity,
                    ragas_faithfulness_score, ragas_answer_relevancy_score,
                    optimizer_system_prompt, accuracy,
                    evaluation_instruction=evaluation_instruction,
                    backward_judge_total_input=backward_judge_total_input
                ))
                
            except Exception as sample_error:
                root_error = extract_root_error_message(sample_error)
                # 샘플 처리 예외 - 에러 로그 기록
                optimization_logs.append(create_error_log(
                    base_log, system_prompt.value, question, context, ground_truth, root_error
                ))
                
                # [치명적 에러 처리] 배치 일관성을 위해 즉시 중단
                print(f"\n{'='*80}")
                print(f"[!] 치명적 에러 발생 - 실험을 중단합니다")
                print(f"{'='*80}")
                print(f"Iteration: {iteration}/{total_iterations}")
                print(f"에러 타입: {type(sample_error).__name__}")
                print(f"에러 메시지: {root_error}")
                print(f"질문: {question[:200]}...")
                print(f"\n[상세 스택 트레이스]")
                traceback.print_exc()
                print(f"\n실험을 중단합니다. (로그는 자동 저장됩니다)")
                
                # 실험 중단 (atexit가 자동으로 로그 저장)
                raise RuntimeError(
                    f"Sample processing failed at iteration {iteration}. "
                    f"Error: {root_error}"
                ) from sample_error
            
        # 2) Gradient 계산 (TextGrad 논문 방식)
        if not losses:
            print(f"[Warning] Iteration {iteration}: 유효한 loss 없음, 업데이트 건너뜀")
            continue

        # ====================================== 디버깅 ====================================== #
        if DEBUG_INDIVIDUAL_BACKWARD:
            should_skip = debug_individual_backward_samples(
                losses=losses,
                episode=iteration,
                iteration=iteration,
                optimizer=optimizer,
                extract_root_error_message_fn=extract_root_error_message
            )
            if should_skip:
                continue
        # ====================================== 디버깅 ====================================== #

        # ============================================================================
        # Backward Pass: 평가자 LLM(backward_engine)이 피드백(gradient) 생성
        # ============================================================================
        # 1. tg.sum()으로 배치 내 모든 loss 병합
        # 2. backward() 호출 시:
        #    - StringBasedFunction: backward_engine이 피드백 생성 (LLM 호출 1회)
        #    - TextLoss: backward_engine이 피드백 생성 (LLM 호출 1회)
        # 3. 생성된 피드백(gradient)은 system_prompt.gradients에 저장됨
        # 
        # 예시 피드백: "프롬프트에 '단계별로 풀이하라'를 추가하세요"
        total_loss = tg.sum(losses)
        total_loss.backward()  # ← backward_engine(평가자 LLM)이 피드백 생성!

        # 3) 후보 프롬프트 생성 (optimizer.step() 전에 gradient 텍스트 백업)
        # system_prompt.get_gradient_text(): backward()에서 생성된 피드백 텍스트
        
        ##################################### 차별점 #############################################
        # [Baseline] 단순 gradient 텍스트만 사용 (샘플 비평 미사용)
        # [Improve] 샘플 비평을 3계층 구조화 (optimization_logs 실제 전달)
    
        prompt_feedback_text = EXPERIMENT_INS.extract_feedback_str(
            system_prompt=system_prompt,
            optimization_logs=optimization_logs,
            iteration_log_start_idx=iteration_log_start_idx
        )
        
        # [Improve 모드 핵심] 계층형 피드백을 Optimizer에게 실제로 전달
        # ★ TextGrad의 optimizer._update_prompt()는 variable.get_gradient_text()를 사용하는데,
        #    이것은 variable.gradients의 value들을 읽어옵니다.
        # ★ 따라서 gradient Variable들의 value를 계층형 피드백으로 교체해야 합니다!
        if EXPERIMENT_INS.mode == 'improve':
            for grad_var in system_prompt.gradients:
                grad_var.value = prompt_feedback_text
            print(f"[Improve] 계층형 피드백 적용 완료: {len(system_prompt.gradients)}개 gradient 교체")
        ##########################################################################################

        # ============================================================================
        # Optimizer Step: 평가자 LLM(backward_engine)이 새 프롬프트 생성
        # ============================================================================
        # optimizer._update_prompt() 내부에서 backward_engine이 호출됨
        # 입력: 현재 프롬프트 + 피드백(gradient) + 과거 이력(momentum)
        # 출력: 개선된 새 프롬프트
        # 예: "문제를 풀어라" → "문제를 단계별로 풀고 답을 명확히 제시하라"
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
            momentum_storage_idx=0,
        )

        try:
            optimizer_response = backward_engine(
                optimizer_update_input,
                system_prompt=optimizer_system_prompt,
            )
        except TypeError:
            merged_optimizer_input = (
                f"{optimizer_system_prompt}\n\n"
                f"{optimizer_update_input}"
            )
            optimizer_response = backward_engine(merged_optimizer_input)

        optimizer_response_text = str(optimizer_response).strip()
        optimizer_total_input_with_momentum = (
            f"{optimizer_update_input}\n\n"
            f"{'='*80}\n"
            f"{momentum_history}"
        )

        print(f"\n{'='*80}")
        print("[DEBUG] Optimizer 입력 프롬프트:")
        print(f"{'='*80}")
        print(optimizer_update_input)
        print(f"\n{'='*80}")
        print("[DEBUG] Optimizer가 생성한 전체 응답:")
        print(f"{'='*80}")
        print(optimizer_response_text)
        print(f"\n{'='*80}\n")

        patterns = [
            r"<new_variable>(.*?)</new_variable>",
            r"<IMPROVED_VARIABLE>(.*?)</IMPROVED_VARIABLE>",
            r"<refined_template>(.*?)</refined_template>",
            r"<OPTIMIZER_WRITING_TEXT_START>(.*?)<OPTIMIZER_WRITING_TEXT_END>",
            r"```(.*?)```",
        ]

        actual_candidate_text = None
        matched_pattern = None
        pattern_results = []

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

        # 4) Validation 평가 - 후보 프롬프트만 평가 (논문 방식)
        # "기존 프롬프트의 점수는 캐싱되어 있으므로 후보 프롬프트에 대해서만 LLM을 호출합니다."
        val_score_candidate = 0.0
        val_count = 0
        validation_info = {}  # 현재 iteration의 validation 샘플 정보

        print(f"Validation 평가 중 (후보 프롬프트만 평가)... 총 {len(validation_dataset)}개 샘플")

        original_prompt_value = system_prompt.value
        system_prompt.value = actual_candidate_text

        for val_idx, val_data in enumerate(validation_dataset, 1):
            if val_idx % 5 == 0 or val_idx == 1:
                print(f"  [{val_idx}/{len(validation_dataset)}] 후보 프롬프트 평가 중...")
            # [데이터 추출] RAG 챗봇용 입력 데이터
            # - val_context: RAG 문서 자료 (GSM8k는 빈 문자열, NASA/KLUE는 실제 문서)
            # ※주의: 여기의 'val_context'는 TextGrad optimizer의 <CONTEXT> 태그와 무관합니다.
            #   <CONTEXT> 태그는 "이전 최적화 피드백"을 담으며, TextGrad 라이브러리가 자동 관리합니다.
            val_context = normalize_text_field(val_data.get('context', ''))
            val_question = normalize_text_field(val_data.get('question', ''))
            val_gt = normalize_text_field(val_data.get('answer', ''))

            # - GSM8k: context 없음 → "Question: ..."
            # - NASA/KLUE: context 있음 → "Context: ...\nQuestion: ..."
            # ※주의: 이것은 Forward Model에게 주는 입력입니다.
            #   TextGrad Optimizer의 <CONTEXT> 태그(이전 피드백)와는 다릅니다.
            if val_context.strip():
                val_inputs = f"Context: {val_context}\nQuestion: {val_question}"
            else:
                val_inputs = f"Question: {val_question}"

            try:
                val_var_cand = tg.Variable(val_inputs, role_description="Validation input", requires_grad=False)
                pred_cand = model(val_var_cand).value

                if is_gsm8k:
                    # GSM8k 정확도 계산: 숫자 추출 후 비교
                    pred_num = parse_integer_answer(pred_cand)
                    gt_num = parse_integer_answer(val_gt)
                    # 파싱 실패는 무조건 오답(0점) 처리
                    if pred_num is None or gt_num is None:
                        score_cand = 0.0
                    else:
                        score_cand = 1.0 if pred_num == gt_num else 0.0
                elif is_multiple_choice:
                    score_cand = similarity_judge(val_gt, pred_cand) if similarity_judge else 0.0
                else:
                    score_cand = similarity_judge(val_gt, pred_cand) if similarity_judge else 0.0

                # Validation 샘플 정보 수집
                validation_info[str(val_idx - 1)] = {
                    "Q": val_question,
                    "A": pred_cand,
                    "GA": val_gt,
                    "score": score_cand
                }

                val_score_candidate += score_cand
                val_count += 1
            except Exception as e:
                print(f"⚠️ Validation 샘플 [{val_idx}] 에러: {e}")
                continue

        system_prompt.value = original_prompt_value

        if val_count > 0:
            val_score_candidate /= val_count

        val_score_current = cached_val_score_current

        # 5) 프롬프트 선택 및 업데이트
        # [중요] 성능 개선 여부와 상관없이, 이번 이터레이션의 Gradient와 Value를 기록합니다.
        # 이렇게 해야 다음 이터레이션에서 LLM이 "방금 했던 실패"를 보고 배울 수 있습니다.
        if hasattr(optimizer, "_update_momentum_storage"):
                optimizer._update_momentum_storage(system_prompt, momentum_storage_idx=0)

        # 후보 프롬프트가 현재보다 성능이 높을 때만 업데이트합니다.
        if val_score_candidate > val_score_current:  
            system_prompt.set_value(actual_candidate_text)
            # [캐시 갱신] 채택된 후보 프롬프트의 점수를 다음 iteration의 현재 점수로 사용
            cached_val_score_current = val_score_candidate
            print(f"✅ Prompt accepted & Updated (val: {val_score_current:.3f} -> {val_score_candidate:.3f})")
        else:
            system_prompt.set_value(original_prompt_value)
            print(f"❌ Prompt rejected (val: {val_score_current:.3f} vs {val_score_candidate:.3f})")

        # 6) Iteration 로그 업데이트
        for idx in range(iteration_log_start_idx, len(optimization_logs)):
            optimization_logs[idx]['prompt_feedback'] = prompt_feedback_text
            optimization_logs[idx]['optimizer_total_input'] = optimizer_total_input_with_momentum

        # Iteration 종료 후 평균 점수 계산
        successful_scores = []
        for log in optimization_logs[iteration_log_start_idx:]:
            if log.get('is_success') and log.get('total_score') is not None:
                score = log.get('total_score')
                if not (isinstance(score, float) and math.isnan(score)):
                    successful_scores.append(score)

        iteration_avg_score = sum(successful_scores) / len(successful_scores) if successful_scores else None

        # Validation 정보를 해당 iteration의 모든 로그에 추가
        for idx in range(iteration_log_start_idx, len(optimization_logs)):
            optimization_logs[idx]['dataset_size'] = batch_size
            optimization_logs[idx]['avg_total_score'] = iteration_avg_score
            optimization_logs[idx]['validation_info'] = validation_info
            optimization_logs[idx]['validation_accuracy'] = val_score_candidate
            optimization_logs[idx]['validation_dataset_size'] = len(validation_dataset)

        # 마지막에 gradient 비우기
        optimizer.zero_grad()

        print(f"\nIteration {iteration} 완료: 평균 점수 = {iteration_avg_score}")
        print(f"현재 프롬프트: {system_prompt.value}")

    # ========== [TextGrad 논문 재현 루프 끝] ==========

    # 5. DB 저장
    print_step("5. DB 로그 저장")
    _do_db_save()
    _save_done[0] = True  # atexit 비상 저장 비활성화 (정상 저장 완료)

    print_step("6. 최적화 완료")

    print("\n--- 최적화 완료 ---")
    print(f"Final optimized prompt: {system_prompt.value}")

if __name__ == "__main__":
    print_step("=== TextGrad 프롬프트 최적화 시작 ===")
    main()