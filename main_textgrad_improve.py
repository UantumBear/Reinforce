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
from concurrent.futures import ThreadPoolExecutor, as_completed # 병렬 루프
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
from reward.hierarchical_evaluator import HierarchicalEvaluator

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


# ============================================================================
# [CoT 추출] 모델별 Chain of Thought 처리 함수
# ============================================================================


def extract_reasoning_from_gpt5_response(
    raw_response_text: str,
    forward_engine=None,
    query_text: str | None = None,
    system_prompt_text: str | None = None
) -> str:
    """
    GPT-5 모델의 응답에서 Chain of Thought (reasoning)을 추출한다.
    
    GPT-5는 OpenAI의 최신 모델로, 다음과 같은 기능을 지원할 수 있습니다:
    - response_metadata에 reasoning 정보 포함
    - 별도의 API 옵션으로 extended thinking 활용 가능
    
    @Param:
        raw_response_text: 현재까지 수집한 응답 텍스트
        forward_engine: 필요시 추가 API 호출용 LLM 객체
        query_text: 필요시 재질의용 쿼리 텍스트
        system_prompt_text: 필요시 재질의용 시스템 프롬프트
    
    @Return:
        최종 응답 텍스트 (reasoning 정보 포함)
    
    @주요 기능 (향후 확장):
        1. OpenAI API response_metadata에서 reasoning 필드 추출
        2. extended thinking 모드 활용 시 thinking 콘텐츠 추출
        3. 명시적 프롬프트로 chain-of-thought 생성 유도
    """
    
    print(f"[CoT 추출 - GPT-5] Chain of Thought 추출 시작")
    print(f"  - 응답 길이: {len(raw_response_text)} 글자")
    
    # [현재 구현]
    # - API 레이어에서 raw_response_text로 전체 응답 받음
    # - 내부적으로 reasoning 포함될 가능성 있음
    # 
    # [향후 구현 계획]
    # 1. forward_engine.generate() 호출 시 extended_thinking=True 옵션 추가
    # 2. response.response_metadata['reasoning'] 필드 추출
    # 3. 또는 forward_engine에서 직접 reasoning 텍스트 접근
    
    # TODO: backward_engine (평가자 LLM)을 활용하여 reasoning 재구성 가능
    # 예: "다음 응답의 사고 과정을 재구성하세요: {raw_response_text}"
    
    final_response = raw_response_text
    print(f"[CoT 추출 - GPT-5] 최종 응답 길이: {len(final_response)} 글자")
    
    return final_response


def extract_reasoning_from_gpt4_response(
    raw_response_text: str,
    forward_engine=None,
    query_text: str | None = None,
    system_prompt_text: str | None = None
) -> str:
    """
    GPT-4 모델의 응답에서 Chain of Thought를 추출한다.
    
    GPT-4는 GPT-5에 비해 reasoning 기능이 제한적이므로,
    다음 전략을 사용합니다:
    - 명시적 프롬프트 지시 (앞서 baseline_prompt.py에서 추가함)
    - 또는 backward_engine을 활용한 사후 reasoning 재구성
    
    @Param:
        raw_response_text: 현재까지 수집한 응답 텍스트
        forward_engine: 필요시 추가 API 호출용 LLM 객체
        query_text: 필요시 재질의용 쿼리 텍스트
        system_prompt_text: 필요시 재질의용 시스템 프롬프트
    
    @Return:
        최종 응답 텍스트 (reasoning 정보 포함)
    
    @주요 특징:
        1. baseline_prompt의 "Think step by step" 명령 사용
        2. 초기 프롬프트에서 이미 reasoning 생성 유도
        3. 추가 API 호출 최소화 (비용 효율)
    """
    
    print(f"[CoT 추출 - GPT-4] Chain of Thought 추출 시작")
    print(f"  - 응답 길이: {len(raw_response_text)} 글자")
    
    # [현재 구현]
    # - baseline_prompt에서 "Think step by step" 이미 요청함
    # - LLM이 자동으로 단계별 사고 과정 포함
    # - raw_response_text에 이미 reasoning 포함될 가능성 높음
    #
    # [향후 개선]
    # 1. 초기 프롬프트에 더 강력한 CoT 요청 추가
    #    (예: "각 단계를 명확히 구분하여 작성하세요")
    # 2. response 내에서 step marker (Step 1:, Step 2:, 등) 추출
    # 3. 필요시 backward_engine으로 reasoning 재구성
    
    final_response = raw_response_text
    print(f"[CoT 추출 - GPT-4] 최종 응답 길이: {len(final_response)} 글자")
    
    return final_response


def extract_tester_response_with_cot(
    forward_model_name: str,
    raw_response_text: str,
    forward_engine=None,
    query_text: str | None = None,
    system_prompt_text: str | None = None
) -> str:
    """
    TesterLLM의 응답에서 Chain of Thought 텍스트를 추출한다.
    모델별로 다른 처리 전략을 적용한다.
    
    @Param:
        forward_model_name: forward_engine의 모델명 (e.g., "gpt-4o-mini", "gpt-3.5-turbo", "gpt-5" 등)
        raw_response_text: 현재까지 수집한 응답 텍스트
        forward_engine: 필요시 추가 API 호출용 LLM 객체
        query_text: 필요시 재질의용 쿼리 텍스트
        system_prompt_text: 필요시 재질의용 시스템 프롬프트
    
    @Return:
        최종 응답 텍스트 (CoT 정보 포함 여부는 모델에 따라 다름)
    
    @모델별 분기 기준:
        1. GPT-5: reasoning 필드 활용 (최신 기능) ← extract_reasoning_from_gpt5_response()
        2. GPT-4: 명시적 프롬프트 + 단계별 사고 유도 ← extract_reasoning_from_gpt4_response()
        3. GPT-3.5: 기본 응답 (향후 <thinking> 태그 지원)
        4. Claude: 자체 reasoning 메커니즘 (향후 지원)
    """
    
    model_lower = forward_model_name.lower()
    print(f"[CoT 추출 DISPATCH] Forward Model: {forward_model_name}")
    
    # ===== GPT-5 감지 (최신 모델) =====
    if 'gpt-5' in model_lower:
        print(f"[CoT 추출] GPT-5 감지 ({forward_model_name}): GPT-5 전용 reasoning 추출")
        return extract_reasoning_from_gpt5_response(
            raw_response_text=raw_response_text,
            forward_engine=forward_engine,
            query_text=query_text,
            system_prompt_text=system_prompt_text
        )
    
    # ===== GPT-4 감지 (gpt-4, gpt-4o, gpt-4-turbo 등) =====
    elif 'gpt-4' in model_lower:
        print(f"[CoT 추출] GPT-4 감지 ({forward_model_name}): GPT-4 전용 reasoning 추출")
        return extract_reasoning_from_gpt4_response(
            raw_response_text=raw_response_text,
            forward_engine=forward_engine,
            query_text=query_text,
            system_prompt_text=system_prompt_text
        )
    
    # ===== GPT-3.5 감지 =====
    elif 'gpt-3.5' in model_lower or 'gpt-35' in model_lower:
        # ##### 차별점 #####
        # [gpt-3.5-turbo] 현재: 아무 처리 없음
        # [TODO] 향후: <thinking> 태그 프롬프트 추가
        #   - 초기 프롬프트에 "반드시 <thinking>...</thinking> 태그로 생각 과정을 감싸서 작성하세요" 추가
        #   - 그러면 LLM이 자동으로 thinking 블록을 생성하도록 유도 가능
        ###################
        print(f"[CoT 추출] GPT-3.5 감지 ({forward_model_name}): 기본 응답 사용 (향후 <thinking> 태그 처리 예정)")
        return raw_response_text
    
    # ===== Claude 감지 (향후 확장) =====
    elif 'claude' in model_lower:
        print(f"[CoT 추출] Claude 감지 ({forward_model_name}): Claude 전용 처리 준비 중...")
        # TODO: Claude의 thinking_blocks 또는 자체 reasoning 메커니즘 활용
        return raw_response_text
    
    else:
        # 기타 모델: 기본 응답 사용
        print(f"[CoT 추출] ⚠️ 알 수 없는 모델 ({forward_model_name}): 기본 응답 사용")
        return raw_response_text


def build_train_output_format_instruction() -> str:
    """Train 단계에서만 적용할 TesterLLM 출력 포맷 강제 지시문을 생성한다."""
    return (
        "\n\n[중요 출력 형식]\n"
        "반드시 아래 XML 태그 형식을 정확히 지켜서 답변하세요.\n"
        "1) 풀이/사고 과정은 <CoT>...</CoT> 태그 안에 작성\n"
        "2) 최종 답변만 <Response>...</Response> 태그 안에 작성\n"
        "3) <Response>에는 최종 답만 간결하게 작성\n"
        "4) 반드시 <CoT>와 <Response> 둘 다 포함\n"
    )


def force_train_input_with_cot_response_tags(forward_input: str) -> str:
    """기존 Forward 입력에 CoT/Response 출력 형식 요구사항을 덧붙인다."""
    return f"{forward_input}{build_train_output_format_instruction()}"


def extract_response_text_from_tester_output(raw_output_text: str) -> tuple[str, bool]:
    """TesterLLM 원문에서 <Response> 태그 값을 추출하고, 없으면 원문을 그대로 반환한다."""
    match = re.search(r"<Response>(.*?)</Response>", raw_output_text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip(), True
    return raw_output_text.strip(), False


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

    # 병렬 루프를 위한 설정
    test_eval_max_workers = 4
    try:
        test_eval_max_workers = max(1, int(os.getenv("TEXTGRAD_TEST_MAX_WORKERS", "4")))
    except (TypeError, ValueError):
        test_eval_max_workers = 4
    print(f"[✓] Test 평가 병렬 워커 수: {test_eval_max_workers}")
    # 병렬 루프를 위한 설정

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
    # ----------------------------------------- 차별점 -----------------------------------------
    # improve 모드 전용: backward Judge 피드백 출력 태그 검사기
    feedback_structure_validator = HierarchicalEvaluator(judge_llm=None)
    _print_elapsed("Judge 모델 초기화 완료")
    # ----------------------------------------- 차별점 -----------------------------------------
    
    
    # 데이터셋 타입 감지 (accuracy 계산용)
    dataset_name_lower = EXPERIMENT_INS.default_dataset_name.lower()
    is_multiple_choice = any(keyword in dataset_name_lower for keyword in ['gpqa', 'mmlu', 'hqh'])
    is_gsm8k = 'gsm8k' in dataset_name_lower
    is_object_counting = 'object_counting' in dataset_name_lower
    is_numeric_exact_match_dataset = is_gsm8k or is_object_counting
    
    print(
        f"[✓] 데이터셋 타입: multiple_choice={is_multiple_choice}, "
        f"gsm8k={is_gsm8k}, object_counting={is_object_counting}, "
        f"numeric_exact_match={is_numeric_exact_match_dataset}"
    )
    
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
    _saved_count = [0]  # 이미 DB에 저장된 로그 수. 이터레이션마다 누적 저장 시 중복 방지용.

    def _do_db_save():
        """DB 저장 - 신규 로그(_saved_count 이후)만 저장. 이터레이션 완료 시 및 종료 시 공통 사용."""
        logs_to_save = optimization_logs[_saved_count[0]:]
        if not logs_to_save:
            print("[!] 저장할 신규 로그가 없습니다.")
            return
        session = None
        try:
            session = pg_client.get_session()
            for log_data in logs_to_save:
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
                    test_info=log_data.get('test_info'),
                    test_accuracy=log_data.get('test_accuracy'),
                    test_dataset_size=log_data.get('test_dataset_size'),
                    dataset_size=log_data.get('dataset_size'),
                    train_batch_size=log_data.get('train_batch_size'),
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
                    # critical_review: backward Judge가 생성한 샘플별 비평 원문
                    critical_review=log_data['answer_feedback'],
                    # full_analysis: optimizer에 투입된 프롬프트 최적화용 계층 피드백
                    full_analysis=log_data['prompt_feedback'],
                    is_success=log_data['is_success'],
                    error_log=log_data['error_log'],
                    created_at=log_data['created_at']
                )
                session.add(record)
            session.commit()
            _saved_count[0] = len(optimization_logs)  # 저장 완료 후 카운터 갱신
            print(f"[✓] DB 저장 완료: {len(logs_to_save)}건 (누계: {_saved_count[0]}건)")
        except Exception as e:
            print(f"[!] DB 저장 실패: {str(e)}")
            if session is not None:
                session.rollback()
        finally:
            if session is not None:
                session.close()

    def _update_test_summary_row(episode: int, test_info: dict, test_accuracy: float | None, test_dataset_size: int | None):
        """이미 삽입된 test 요약 row(episode 단일 row)를 주기적으로 업데이트한다."""
        session = None
        try:
            session = pg_client.get_session()
            record = (
                session.query(RlOptimizationLog)
                .filter(
                    RlOptimizationLog.experiment_id == experiment_id,
                    RlOptimizationLog.episode == episode,
                )
                .order_by(RlOptimizationLog.id.desc())
                .first()
            )
            if record is None:
                print(f"[!] Test 요약 row 업데이트 실패: episode={episode} row를 찾지 못했습니다.")
                return

            record.test_info = test_info
            record.test_accuracy = test_accuracy
            record.test_dataset_size = test_dataset_size
            record.accuracy = None  # accuracy는 train 샘플 컬럼이므로 test 요약에서는 미사용
            record.validation_info = None
            record.validation_accuracy = None
            record.validation_dataset_size = None
            record.dataset_size = len(train_pool)
            record.train_batch_size = batch_size
            record.avg_total_score = None
            session.commit()
        except Exception as e:
            print(f"[!] Test 요약 row 업데이트 실패: {str(e)}")
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

    def _evaluate_single_validation_sample(sample_idx: int, sample_data: dict):
        """Validation 샘플 1건 평가 (병렬 워커용).
        주의: 호출 시점의 system_prompt.value 를 forward_engine 이 사용함.
              초기 캐싱은 현재 프롬프트, 후보 평가는 사전에 swap된 candidate 프롬프트 상태에서 호출.
        반환: (sample_idx, score, success, sample_info_or_None, error_or_None)
        """
        sample_context = normalize_text_field(sample_data.get('context', ''))
        sample_question = normalize_text_field(sample_data.get('question', ''))
        sample_gt = normalize_text_field(sample_data.get('answer', ''))
        sample_persona = sample_data.get('system_persona', '')
        sample_inputs = EXPERIMENT_INS.build_forward_input(sample_question, sample_context, sample_persona)
        try:
            sample_var = tg.Variable(sample_inputs, role_description="Validation input", requires_grad=False)
            sample_pred = model(sample_var).value
            sample_pred_for_eval, _ = extract_response_text_from_tester_output(sample_pred)

            if is_numeric_exact_match_dataset:
                pred_num = parse_integer_answer(sample_pred_for_eval)
                gt_num = parse_integer_answer(sample_gt)
                # 파싱 실패는 무조건 오답(0점) 처리
                if pred_num is None or gt_num is None:
                    sample_score = 0.0
                else:
                    sample_score = 1.0 if pred_num == gt_num else 0.0
            elif is_multiple_choice:
                sample_score = similarity_judge(sample_gt, sample_pred_for_eval) if similarity_judge else 0.0
            else:
                sample_score = similarity_judge(sample_gt, sample_pred_for_eval) if similarity_judge else 0.0

            sample_info = {
                "Q": sample_question,
                "A": sample_pred,
                "GA": sample_gt,
                "score": sample_score,
            }
            return sample_idx, sample_score, True, sample_info, None
        except Exception as eval_error:
            return sample_idx, None, False, None, eval_error

    # 병렬 평가 (test 평가와 동일한 워커 풀 크기 사용)
    val_workers = max(1, min(test_eval_max_workers, len(validation_dataset))) if validation_dataset else 1
    with ThreadPoolExecutor(max_workers=val_workers) as val_executor:
        val_futures = [
            val_executor.submit(_evaluate_single_validation_sample, vi, vdata)
            for vi, vdata in enumerate(validation_dataset)
        ]
        for completed_idx, future in enumerate(as_completed(val_futures), 1):
            sample_idx, sample_score, success, sample_info, eval_error = future.result()
            if success and sample_score is not None:
                # 0-based 문자열 키 (기존 동작과 동일)
                initial_validation_info[str(sample_idx)] = sample_info
                cached_val_score_current += sample_score
                cached_val_count += 1
            else:
                print(f"  ⚠️ 초기 캐시 평가 샘플 [{sample_idx + 1}] 에러:")
                if eval_error is not None:
                    print(f"     에러 타입: {type(eval_error).__name__}")
                    print(f"     에러 메시지: {eval_error}")
            if completed_idx % 5 == 0 or completed_idx == 1:
                print(f"  [{completed_idx}/{len(validation_dataset)}] 초기 캐싱 평가 진행 중...")

    if cached_val_count > 0:
        cached_val_score_current /= cached_val_count
    print(f"[초기 캐싱] 완료: {cached_val_score_current:.4f} ({cached_val_count}개 평가)")
    print(f"[초기 캐싱] Validation 샘플 정보: {len(initial_validation_info)}개 수집")

    ##### 차별점 #####
    # [Improve] 최종 Test 평가용 global best prompt 추적
    # - 학습 중 채택(허용 오차 포함) 로직은 유지
    # - 최종 Test에서는 validation 기준 최고 성능 프롬프트를 사용
    global_best_prompt_value = system_prompt.value
    global_best_val_score = cached_val_score_current
    global_best_iteration = 0
    ###################

    # -----------------------------------------------------------------------
    # [episode=0] 초기 프롬프트를 전체 Test Set으로 평가 (논문 Apple-to-Apple 비교용)
    # 논문 기준: 동일한 Test Set(GSM8k 1,319개)으로 최적화 전/후 성능을 비교합니다.
    # 최적화 루프(ep1~12)의 Validation Set(300개)과는 별개 평가입니다.
    # EXPERIMENT_INS.enable_test_evaluation == False 이면 이 블록 전체를 건너뜁니다.
    # -----------------------------------------------------------------------
    if not EXPERIMENT_INS.enable_test_evaluation:
        print(f"\n[episode=0] Test 평가 비활성화 (enable_test_evaluation=False), 건너뜁니다.")
        test_dataset = []
    else:
        print(f"\n[episode=0] 초기 프롬프트 Test Set 전체 평가 시작...")
        test_dataset = EXPERIMENT_INS.load_test_data()  # type: ignore[assignment]

    def _evaluate_single_test_sample(sample_idx: int, sample_data: dict, role_description: str):
        sample_context = normalize_text_field(sample_data.get('context', ''))
        sample_question = normalize_text_field(sample_data.get('question', ''))
        sample_gt = normalize_text_field(sample_data.get('answer', ''))
        sample_persona = sample_data.get('system_persona', '')
        sample_inputs = EXPERIMENT_INS.build_forward_input(sample_question, sample_context, sample_persona)

        try:
            sample_var = tg.Variable(sample_inputs, role_description=role_description, requires_grad=False)
            sample_pred = model(sample_var).value
            sample_pred_for_eval, _ = extract_response_text_from_tester_output(sample_pred)

            if is_numeric_exact_match_dataset:
                pred_num = parse_integer_answer(sample_pred_for_eval)
                gt_num = parse_integer_answer(sample_gt)
                sample_score = 1.0 if (pred_num is not None and gt_num is not None and pred_num == gt_num) else 0.0
            elif is_multiple_choice:
                sample_score = similarity_judge(sample_gt, sample_pred_for_eval) if similarity_judge else 0.0
            else:
                sample_score = similarity_judge(sample_gt, sample_pred_for_eval) if similarity_judge else 0.0

            sample_info = {
                "Q": sample_question,
                "A": sample_pred,
                "GA": sample_gt,
                "score": sample_score,
            }
            return sample_idx, sample_score, True, sample_info
        except Exception as error:
            root_error = extract_root_error_message(error)
            sample_info = {
                "Q": sample_question,
                "A": "[ERROR]",
                "GA": sample_gt,
                "score": None,
                "error": root_error,
            }
            return sample_idx, None, False, sample_info

    if test_dataset:
        base_log_ep0 = create_base_log(
            experiment_id, 0,
            textgrad_backward_model_nm,
            textgrad_forward_model_nm,
            embedding_model_nm,
            dataset_nm=EXPERIMENT_INS.default_dataset_name,
        )
        ep0_score = 0.0
        ep0_count = 0
        ep0_test_info = {}

        ep0_summary_log = create_success_log(
            base_log_ep0,
            system_prompt.value,
            question="[Test Summary] episode=0",
            context="",
            ground_truth="[N/A]",
            prediction="[N/A]",
            computed_loss_value="[N/A] 초기 Test 평가 요약 row (backward 없음)",
            raw_similarity=None,
            ragas_faithfulness_score=None,
            ragas_answer_relevancy_score=None,
            optimizer_system_prompt=optimizer_system_prompt,
            accuracy=None,
        )
        ep0_summary_log['test_info'] = {}
        ep0_summary_log['test_accuracy'] = None
        ep0_summary_log['test_dataset_size'] = len(test_dataset)
        ep0_summary_log['validation_info'] = None
        ep0_summary_log['validation_accuracy'] = None
        ep0_summary_log['validation_dataset_size'] = None
        ep0_summary_log['dataset_size'] = len(train_pool)
        ep0_summary_log['train_batch_size'] = batch_size
        ep0_summary_log['avg_total_score'] = None
        optimization_logs.append(ep0_summary_log)
        _do_db_save()  # episode=0 summary row 최초 insert

        ep0_workers = min(test_eval_max_workers, len(test_dataset))
        with ThreadPoolExecutor(max_workers=ep0_workers) as executor:
            ep0_futures = [
                executor.submit(_evaluate_single_test_sample, ep0_idx, ep0_data, "Test input")
                for ep0_idx, ep0_data in enumerate(test_dataset)
            ]

            for completed_idx, future in enumerate(as_completed(ep0_futures), 1):
                sample_idx, ep0_score_sample, success, sample_info = future.result()
                ep0_test_info[str(sample_idx)] = sample_info

                if success and ep0_score_sample is not None:
                    ep0_score += ep0_score_sample
                    ep0_count += 1

                if completed_idx % 50 == 0 or completed_idx == 1:
                    print(f"  [episode=0] [{completed_idx}/{len(test_dataset)}] 초기 프롬프트 Test 평가 중...")

                if completed_idx % 100 == 0:
                    running_acc = ep0_score / ep0_count if ep0_count > 0 else 0.0
                    _update_test_summary_row(
                        episode=0,
                        test_info=ep0_test_info,
                        test_accuracy=running_acc,
                        test_dataset_size=len(test_dataset),
                    )
                    print(f"  [episode=0] 중간 저장 완료: {completed_idx}/{len(test_dataset)} (acc={running_acc:.4f})")

        ep0_accuracy = ep0_score / ep0_count if ep0_count > 0 else 0.0
        print(f"[episode=0] 완료: Test Set 정확도 = {ep0_accuracy:.4f} ({ep0_count}/{len(test_dataset)}개 평가)")
        _update_test_summary_row(
            episode=0,
            test_info=ep0_test_info,
            test_accuracy=ep0_accuracy,
            test_dataset_size=len(test_dataset),
        )
    else:  # test_dataset 없음 (enable_test_evaluation=True 인데 데이터 없는 경우)
        print(f"[episode=0] Test 데이터셋 없음, 건너뜁니다.")

    # ========== [TextGrad 논문 재현 루프 시작] ==========
    # ##### 차별점 #####
    # [Improve] 이전 iteration에서 후보 프롬프트가 거절된 경우, 해당 정보를
    #           다음 iteration의 JudgeLLM evaluation instruction에 주입한다.
    #           거절 → XML 문자열 저장, 채택 → 빈 문자열 리셋
    previous_rejection_info_str = ""  # iteration 간 유지되는 거절 컨텍스트
    ###################

    random.seed(42)  # train batch 재현성 보장 (실험 간 동일한 batch 순서)
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

        # [논문 정렬] Iteration 시작 시 gradient 초기화
        optimizer.zero_grad()

        # 1) Train: 배치 크기만큼 무작위 복원 추출
        batch = random.choices(train_pool, k=batch_size)  # 복원 추출

        losses = []

        # =========================================================================
        # [병렬화] 배치 내 각 샘플의 forward + Judge 평가(loss(prediction_var))만
        # ThreadPoolExecutor 워커에서 동시 실행. 사이드이펙트(태그 보정,
        # ragas, optimization_logs append, 에러 raise)는 모두 메인 스레드에서
        # 직렬로 처리하여 실험 결과 동등성을 보장합니다.
        # =========================================================================
        def _process_one_train_sample(idx_local, data_local):
            """순수 LLM I/O + 계산만 수행. 공유 상태(optimization_logs 등) 수정 없음."""
            ctx = normalize_text_field(data_local.get('context', ''))
            q = normalize_text_field(data_local.get('question', ''))
            gt = normalize_text_field(data_local.get('answer', ''))
            persona = data_local.get('system_persona', '')

            if has_jailbreak_like_pattern(ctx) or has_jailbreak_like_pattern(q) or has_jailbreak_like_pattern(gt):
                return {
                    "idx": idx_local, "status": "skip",
                    "question": q, "context": ctx, "ground_truth": gt,
                }
            try:
                # [Forward Model 입력 구성]
                raw_inputs_local = EXPERIMENT_INS.build_forward_input(q, ctx, persona)
                ##### 차별점 #####
                # [Improve] Train 단계에서 CoT/Response 태그 형식을 강제
                inputs_local = force_train_input_with_cot_response_tags(raw_inputs_local)
                ###################
                query_var_local = tg.Variable(inputs_local, role_description="RAG 입력", requires_grad=False)

                if is_multiple_choice and test_time_updates > 1:
                    # [GPQA/MMLU/HQH 경로] - Majority Voting (현재 실험에서 실행되지 않는 분기)
                    test_time_predictions_l = []
                    test_time_choices_l = []
                    first_prediction_var_l = None
                    for upd_idx in range(test_time_updates):
                        pv = model(query_var_local)  # forward_engine 호출
                        pt = pv.value
                        test_time_predictions_l.append(pt)
                        if upd_idx == 0:
                            first_prediction_var_l = pv
                        pte, _ = extract_response_text_from_tester_output(pt)
                        ch = extract_choice_from_answer(pte)
                        test_time_choices_l.append(ch)
                    prediction_var_local = first_prediction_var_l
                    prediction_local = test_time_predictions_l[0]
                    prediction_for_eval_local, _ = extract_response_text_from_tester_output(prediction_local)
                    final_choice_l = majority_vote(test_time_choices_l)
                    correct_choice_l = extract_choice_from_answer(gt)
                    acc = compute_accuracy(final_choice_l, correct_choice_l)
                    tester_full_raw = prediction_local
                else:
                    # [GSM8k / 그 외 경로] - 단일 forward
                    prediction_var_local = model(query_var_local)  # forward_engine 호출
                    prediction_local = prediction_var_local.value
                    ##### 차별점 #####
                    tester_full_raw = extract_tester_response_with_cot(
                        forward_model_name=textgrad_forward_model_nm,
                        raw_response_text=prediction_local,
                        forward_engine=forward_engine,
                        query_text=query_var_local.value,
                        system_prompt_text=system_prompt.value,
                    )
                    prediction_for_eval_local, _ = extract_response_text_from_tester_output(tester_full_raw)
                    ###################
                    acc = None

                # 유사도 점수 (참고 지표)
                raw_sim = None
                if similarity_judge is not None:
                    try:
                        raw_sim = similarity_judge(gt, prediction_for_eval_local)
                    except Exception:
                        raw_sim = math.nan

                # 평가/손실 계산
                eval_inst = None
                cl = None
                if is_numeric_exact_match_dataset and EXPERIMENT_INS.mode == 'baseline':
                    # [논문 재현] StringBasedFunction (LLM 호출 없음)
                    gt_var = tg.Variable(
                        gt,
                        role_description="the correct answer for the math problem",
                        requires_grad=False,
                    )
                    eval_fn = StringBasedFunction(
                        string_based_equality_fn,
                        function_purpose="Checks if the prediction is correct by comparing the numerical answer",
                    )
                    cl = eval_fn(
                        inputs=dict(prediction=prediction_var_local, ground_truth_answer=gt_var),
                        response_role_description="Whether the prediction is correct (1) or not (0)",
                    )
                    try:
                        acc = float(cl.value)
                    except (ValueError, TypeError):
                        acc = None
                else:
                    # [TextLoss 경로] - JudgeLLM(backward_engine)이 평가 수행
                    ############################### 차별점 ###############################################
                    objective_acc = None
                    if EXPERIMENT_INS.mode == 'improve' and is_numeric_exact_match_dataset:
                        pn = parse_integer_answer(prediction_for_eval_local)
                        gn = parse_integer_answer(gt)
                        if pn is not None and gn is not None:
                            objective_acc = 1.0 if pn == gn else 0.0
                    acc = objective_acc

                    eval_inst = EXPERIMENT_INS.get_objective_function(
                        gt,
                        similarity_score=raw_sim,
                        accuracy_score=objective_acc,
                        prediction=tester_full_raw,
                        previous_rejection_context=previous_rejection_info_str,
                    )
                    ######################################################################################
                    loss_local = tg.TextLoss(eval_inst)
                    cl = loss_local(prediction_var_local)  # ← JudgeLLM 호출 (병렬화의 핵심 수익 지점)

                return {
                    "idx": idx_local, "status": "success",
                    "question": q, "context": ctx, "ground_truth": gt,
                    "prediction": prediction_local,
                    "computed_loss": cl,
                    "evaluation_instruction": eval_inst,
                    "raw_similarity": raw_sim,
                    "accuracy": acc,
                }
            except Exception as e:
                return {
                    "idx": idx_local, "status": "error",
                    "question": q, "context": ctx, "ground_truth": gt,
                    "error": e,
                }

        # 워커 실행 + 인덱스 보존 (직렬과 동일한 후처리 순서 보장)
        batch_results = [None] * len(batch)
        batch_workers = max(1, min(test_eval_max_workers, len(batch)))
        with ThreadPoolExecutor(max_workers=batch_workers) as batch_ex:
            batch_futures = {
                batch_ex.submit(_process_one_train_sample, i, d): i
                for i, d in enumerate(batch)
            }
            for bf in as_completed(batch_futures):
                bi = batch_futures[bf]
                batch_results[bi] = bf.result()

        # =========================================================================
        # [메인 스레드 직렬 후처리] 사이드이펙트는 모두 여기서만 발생
        # - computed_loss.value 태그 보정 (improve 모드)
        # - ragas_judge.evaluate 호출
        # - optimization_logs.append (순서 보존)
        # - 치명적 에러 발생 시 raise → 배치 일관성 유지
        # =========================================================================
        for r in batch_results:
            if r is None:
                continue

            if r["status"] == "skip":
                # Jailbreak 패턴 감지 - 스킵 로그
                optimization_logs.append(create_skip_log(
                    base_log, system_prompt.value, r["question"], r["context"], r["ground_truth"]
                ))
                continue

            if r["status"] == "error":
                sample_error = r["error"]
                root_error = extract_root_error_message(sample_error)
                optimization_logs.append(create_error_log(
                    base_log, system_prompt.value, r["question"], r["context"], r["ground_truth"], root_error
                ))
                # [치명적 에러 처리] 배치 일관성을 위해 즉시 중단 (기존 동작 동일)
                print(f"\n{'='*80}")
                print(f"[!] 치명적 에러 발생 - 실험을 중단합니다")
                print(f"{'='*80}")
                print(f"Iteration: {iteration}/{total_iterations}")
                print(f"에러 타입: {type(sample_error).__name__}")
                print(f"에러 메시지: {root_error}")
                print(f"질문: {r['question'][:200]}...")
                print(f"\n[상세 스택 트레이스]")
                # 워커에서 잡힌 예외이므로 __traceback__ 사용
                traceback.print_exception(type(sample_error), sample_error, sample_error.__traceback__)
                print(f"\n실험을 중단합니다. (로그는 자동 저장됩니다)")
                raise RuntimeError(
                    f"Sample processing failed at iteration {iteration}. "
                    f"Error: {root_error}"
                ) from sample_error

            # success 케이스
            computed_loss = r["computed_loss"]
            evaluation_instruction = r["evaluation_instruction"]
            prediction = r["prediction"]
            question = r["question"]
            context = r["context"]
            ground_truth = r["ground_truth"]
            raw_similarity = r["raw_similarity"]
            accuracy = r["accuracy"]

            # 진단 로그 (직렬 버전과 의미 동등)
            if evaluation_instruction is None:
                # StringBasedFunction 경로
                print(f"[StringBasedFunction] Result: {computed_loss.value} (0=wrong, 1=correct)")
                if accuracy is not None:
                    print(f"[Accuracy - Numeric Exact Match] From StringBasedFunction: {accuracy}")
            if raw_similarity is not None:
                print(f"[Debug] Raw similarity score: {raw_similarity}")

            # improve 모드: backward Judge 출력 태그 구조 검증/보정
            feedback_tag_validation = None
            if EXPERIMENT_INS.mode == 'improve' and evaluation_instruction is not None:
                feedback_tag_validation = feedback_structure_validator.validate_feedback_output_tags(
                    computed_loss.value
                )
                if not feedback_tag_validation.get("is_valid", False):
                    print("⚠️ [Feedback Tag Validation] backward Judge 출력 형식이 요구 태그를 충족하지 않았습니다.")
                    print(f"   - missing_open_tags: {feedback_tag_validation.get('missing_open_tags')}")
                    print(f"   - missing_close_tags: {feedback_tag_validation.get('missing_close_tags')}")
                    print(f"   - nested_errors: {feedback_tag_validation.get('nested_errors')}")

                    fixed_feedback = feedback_structure_validator.fix_feedback_output_tags(
                        computed_loss.value
                    )
                    if hasattr(computed_loss, "set_value"):
                        computed_loss.set_value(fixed_feedback)
                    else:
                        computed_loss.value = fixed_feedback

                    feedback_tag_validation = feedback_structure_validator.validate_feedback_output_tags(
                        computed_loss.value
                    )
                    print(
                        f"✅ [Feedback Tag Validation] 자동 보정 완료. "
                        f"is_valid={feedback_tag_validation.get('is_valid')}"
                    )

            # losses는 직렬 인덱스 순서대로 추가 (batch_results는 이미 인덱스 순서)
            losses.append(computed_loss)

            # ragas (메인 스레드에서 직렬 호출)
            ragas_faithfulness_score = None
            ragas_answer_relevancy_score = None
            if ragas_judge is not None:
                try:
                    ragas_result = ragas_judge.evaluate(
                        question=question, answer=prediction,
                        context=context, gold_answer=ground_truth,
                    )
                    if not str(ragas_result.get('reason', '')).startswith("Evaluation error:"):
                        ragas_faithfulness_score = ragas_result.get('score')
                        ragas_answer_relevancy_score = ragas_result.get('relevancy_score')
                    else:
                        ragas_faithfulness_score = math.nan
                        ragas_answer_relevancy_score = math.nan
                except Exception:
                    ragas_faithfulness_score = math.nan
                    ragas_answer_relevancy_score = math.nan

            # 로그 저장
            backward_judge_total_input = None
            if evaluation_instruction is not None:
                backward_judge_total_input = build_backward_judge_total_input(
                    evaluation_instruction=evaluation_instruction,
                    prediction=prediction,
                )
                if feedback_tag_validation is not None:
                    backward_judge_total_input += (
                        "\n\n=== FEEDBACK TAG VALIDATION (improve mode only) ===\n"
                        f"is_valid: {feedback_tag_validation.get('is_valid')}\n"
                        f"missing_open_tags: {feedback_tag_validation.get('missing_open_tags')}\n"
                        f"missing_close_tags: {feedback_tag_validation.get('missing_close_tags')}\n"
                        f"missing_pairs: {feedback_tag_validation.get('missing_pairs')}\n"
                        f"nested_errors: {feedback_tag_validation.get('nested_errors')}\n"
                        f"has_prefix_before_iteration: {feedback_tag_validation.get('has_prefix_before_iteration')}\n"
                    )

            optimization_logs.append(create_success_log(
                base_log, system_prompt.value, question, context, ground_truth,
                prediction, computed_loss.value, raw_similarity,
                ragas_faithfulness_score, ragas_answer_relevancy_score,
                optimizer_system_prompt, accuracy,
                evaluation_instruction=evaluation_instruction,
                backward_judge_total_input=backward_judge_total_input,
            ))

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

        # [중요] system_prompt.value 스왑은 병렬 실행 외부에서 수행 → 모든 워커가 동일한 candidate 상태에서 평가
        original_prompt_value = system_prompt.value
        system_prompt.value = actual_candidate_text

        try:
            cand_workers = max(1, min(test_eval_max_workers, len(validation_dataset))) if validation_dataset else 1
            with ThreadPoolExecutor(max_workers=cand_workers) as cand_executor:
                cand_futures = [
                    cand_executor.submit(_evaluate_single_validation_sample, vi, vdata)
                    for vi, vdata in enumerate(validation_dataset)
                ]
                for completed_idx, future in enumerate(as_completed(cand_futures), 1):
                    sample_idx, score_cand, success, sample_info, eval_error = future.result()
                    if success and score_cand is not None:
                        # 0-based 문자열 키 (기존 동작과 동일)
                        validation_info[str(sample_idx)] = sample_info
                        val_score_candidate += score_cand
                        val_count += 1
                    else:
                        print(f"⚠️ Validation 샘플 [{sample_idx + 1}] 에러: {eval_error}")
                    if completed_idx % 5 == 0 or completed_idx == 1:
                        print(f"  [{completed_idx}/{len(validation_dataset)}] 후보 프롬프트 평가 진행 중...")
        finally:
            # 예외 발생 시에도 원본 프롬프트 복원
            system_prompt.value = original_prompt_value

        if val_count > 0:
            val_score_candidate /= val_count

        val_score_current = cached_val_score_current

        # 5) 프롬프트 선택 및 업데이트
        # [중요] 성능 개선 여부와 상관없이, 이번 이터레이션의 Gradient와 Value를 기록합니다.
        # 이렇게 해야 다음 이터레이션에서 LLM이 "방금 했던 실패"를 보고 배울 수 있습니다.
        if hasattr(optimizer, "_update_momentum_storage"):
                optimizer._update_momentum_storage(system_prompt, momentum_storage_idx=0)

        # [진단] 후보 프롬프트가 실제로 새로운 텍스트인지 확인
        is_new_candidate = actual_candidate_text != original_prompt_value
        if not is_new_candidate:
            print(f"⚠️  [진단] Optimizer가 현재 프롬프트와 동일한 텍스트를 반환함 (regex 추출 실패 또는 LLM이 변경 없이 반환)")
            print(f"       → validation 생략, 현재 프롬프트 유지")

        # 후보 프롬프트가 현재보다 성능이 높거나 같을 때 업데이트 (>= 사용: validation_size가 작으면 동점도 허용)
        # [주의] 실제로 새 텍스트가 추출된 경우에만 비교 (동일 텍스트면 비교 의미 없음)

        ##### 차별점 #####
        # [Baseline] val_score_candidate >= val_score_current (동점 이상만 채택)
        # [Improve]  val_score_candidate >= val_score_current - ACCEPTANCE_TOLERANCE (노이즈 허용)
        #            → 데이터셋별로 validation_size에 맞게 experiment.py에서 설정
        ACCEPTANCE_TOLERANCE = EXPERIMENT_INS.acceptance_tolerance

        # 후보 프롬프트 비교 정보 - 다음 iteration의 backward JudgeLLM에게 컨텍스트로 전달
        # STEP_BEFORE: 이번 iteration 시작 시점의 현재 프롬프트 (채택 전)
        # STEP_NOW: optimizer가 생성한 후보 프롬프트
        # STEP_BETWEEN_DIFF: validation 점수 차이 (양수=개선, 음수=하락)
        score_diff = val_score_candidate - val_score_current
        CANDIDATE_NON_EXCEPT_INFO = {
            "STEP_BEFORE_SYSTEM_PROMPT": original_prompt_value,
            "STEP_NOW_SYSTEM_PROMPT": actual_candidate_text,
            "STEP_BETWEEN_DIFF": f"{score_diff:+.4f} (before={val_score_current:.4f}, after={val_score_candidate:.4f})",
            "MESSAGE": "후보 프롬프트의 성능이 허용 범위 오차보다 더 떨어집니다. 해당 두 프롬프트를 비교하여 JudgeLLM의 비평 작성이 필요합니다."
        }

        ##### 차별점 #####
        # [Improve] global best prompt 갱신 (채택 여부와 독립적으로 추적)
        # - 후보가 현재 프롬프트와 다른 텍스트이고
        # - validation 평가가 1개 이상 성공했으며
        # - 역대 최고 validation 점수를 갱신한 경우
        if is_new_candidate and val_count > 0 and val_score_candidate > global_best_val_score:
            global_best_val_score = val_score_candidate
            global_best_prompt_value = actual_candidate_text
            global_best_iteration = iteration
            print(
                f"🏆 [Global Best] iteration={iteration} "
                f"score={global_best_val_score:.4f} 로 갱신"
            )
        ###################

        if is_new_candidate and score_diff >= -ACCEPTANCE_TOLERANCE:
            system_prompt.set_value(actual_candidate_text)
            # [캐시 갱신] 채택된 후보 프롬프트의 점수를 다음 iteration의 현재 점수로 사용
            cached_val_score_current = val_score_candidate
            ##### 차별점 #####
            # [Improve] 채택 시: 이전 거절 컨텍스트 리셋 (JudgeLLM에게 더 이상 전달 불필요)
            previous_rejection_info_str = ""
            ###################
            if score_diff >= 0:
                print(f"✅ Prompt accepted & Updated (val: {val_score_current:.3f} -> {val_score_candidate:.3f})")
            else:
                print(f"✅ Prompt accepted within tolerance (val: {val_score_current:.3f} -> {val_score_candidate:.3f}, diff: {val_score_candidate - val_score_current:.3f})")
        else:
            system_prompt.set_value(original_prompt_value)
            if is_new_candidate:
                print(f"❌ Prompt rejected (val: {val_score_current:.3f} vs {val_score_candidate:.3f}, diff: {val_score_candidate - val_score_current:.3f})")
                ##### 차별점 #####
                # [Improve] 거절 시: CANDIDATE_NON_EXCEPT_INFO를 XML 태그 문자열로 포맷하여 저장
                #           → 다음 iteration의 JudgeLLM evaluation instruction에 주입됨
                previous_rejection_info_str = (
                    "<CANDIDATE_NON_EXCEPT_INFO>\n"
                    f"  <STEP_BEFORE_SYSTEM_PROMPT>\n{CANDIDATE_NON_EXCEPT_INFO['STEP_BEFORE_SYSTEM_PROMPT']}\n  </STEP_BEFORE_SYSTEM_PROMPT>\n"
                    f"  <STEP_NOW_SYSTEM_PROMPT>\n{CANDIDATE_NON_EXCEPT_INFO['STEP_NOW_SYSTEM_PROMPT']}\n  </STEP_NOW_SYSTEM_PROMPT>\n"
                    f"  <STEP_BETWEEN_DIFF>{CANDIDATE_NON_EXCEPT_INFO['STEP_BETWEEN_DIFF']}</STEP_BETWEEN_DIFF>\n"
                    f"  <MESSAGE>{CANDIDATE_NON_EXCEPT_INFO['MESSAGE']}</MESSAGE>\n"
                    "</CANDIDATE_NON_EXCEPT_INFO>"
                )
                ###################

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
            optimization_logs[idx]['dataset_size'] = len(train_pool)
            optimization_logs[idx]['train_batch_size'] = batch_size
            optimization_logs[idx]['avg_total_score'] = iteration_avg_score
            optimization_logs[idx]['validation_info'] = validation_info
            optimization_logs[idx]['validation_accuracy'] = val_score_candidate
            optimization_logs[idx]['validation_dataset_size'] = len(validation_dataset)

        print(f"\nIteration {iteration} 완료: 평균 점수 = {iteration_avg_score}")
        print(f"현재 프롬프트: {system_prompt.value}")

        # [즉시 저장] 이터레이션 1개 완료 즉시 DB에 저장 (중간 확인 / Ctrl+C 없이 보존)
        _do_db_save()

    # ========== [TextGrad 논문 재현 루프 끝] ==========

    # -----------------------------------------------------------------------
    # [episode=total_iterations+1] 최종 프롬프트를 전체 Test Set으로 평가
    # - 학습/최적화 단계가 아니므로 validation_* 필드는 사용하지 않습니다.
    # - 논문과 동일한 Apple-to-Apple 비교를 위한 최종 성능 측정 단계입니다.
    # - EXPERIMENT_INS.enable_test_evaluation == False 이면 건너뜁니다.
    # -----------------------------------------------------------------------
    final_episode = total_iterations + 1
    if not EXPERIMENT_INS.enable_test_evaluation:
        print(f"\n[episode={final_episode}] Test 평가 비활성화 (enable_test_evaluation=False), 건너뜁니다.")
        final_test_dataset = []
    else:
        print(f"\n[episode={final_episode}] 최종 프롬프트 Test Set 전체 평가 시성...")
        final_test_dataset = EXPERIMENT_INS.load_test_data()  # type: ignore[assignment]

    ##### 차별점 #####
    # [Improve] 최종 Test 직전 global best prompt로 스왑
    # - 학습 종료 시점의 마지막 프롬프트는 유지하되
    # - Test 평가는 validation 기준 최고 프롬프트로 수행
    last_prompt_value = system_prompt.value
    if system_prompt.value != global_best_prompt_value:
        print(
            "[Final Test Prompt] 마지막 프롬프트 대신 global best 프롬프트를 사용합니다.\n"
            f"  - last_prompt_score(reference): {cached_val_score_current:.4f}\n"
            f"  - best_prompt_score: {global_best_val_score:.4f} (iteration={global_best_iteration})"
        )
    else:
        print(
            "[Final Test Prompt] 마지막 프롬프트와 global best 프롬프트가 동일합니다.\n"
            f"  - best_prompt_score: {global_best_val_score:.4f} (iteration={global_best_iteration})"
        )
    system_prompt.set_value(global_best_prompt_value)
    ###################

    if final_test_dataset:
        base_log_final = create_base_log(
            experiment_id,
            final_episode,
            textgrad_backward_model_nm,
            textgrad_forward_model_nm,
            embedding_model_nm,
            dataset_nm=EXPERIMENT_INS.default_dataset_name,
        )
        final_test_score = 0.0
        final_test_count = 0
        final_test_info = {}

        final_summary_log = create_success_log(
            base_log_final,
            system_prompt.value,
            question=f"[Test Summary] episode={final_episode}",
            context="",
            ground_truth="[N/A]",
            prediction="[N/A]",
            computed_loss_value="[N/A] 최종 Test 평가 요약 row (backward 없음)",
            raw_similarity=None,
            ragas_faithfulness_score=None,
            ragas_answer_relevancy_score=None,
            optimizer_system_prompt=optimizer_system_prompt,
            accuracy=None,
        )
        final_summary_log['test_info'] = {}
        final_summary_log['test_accuracy'] = None
        final_summary_log['test_dataset_size'] = len(final_test_dataset)
        final_summary_log['validation_info'] = None
        final_summary_log['validation_accuracy'] = None
        final_summary_log['validation_dataset_size'] = None
        final_summary_log['dataset_size'] = len(train_pool)
        final_summary_log['train_batch_size'] = batch_size
        final_summary_log['avg_total_score'] = None
        optimization_logs.append(final_summary_log)
        _do_db_save()  # final summary row 최초 insert

        final_workers = min(test_eval_max_workers, len(final_test_dataset))
        with ThreadPoolExecutor(max_workers=final_workers) as executor:
            final_futures = [
                executor.submit(_evaluate_single_test_sample, final_idx, final_data, "Final test input")
                for final_idx, final_data in enumerate(final_test_dataset)
            ]

            for completed_idx, future in enumerate(as_completed(final_futures), 1):
                sample_idx, final_score_sample, success, sample_info = future.result()
                final_test_info[str(sample_idx)] = sample_info

                if success and final_score_sample is not None:
                    final_test_score += final_score_sample
                    final_test_count += 1

                if completed_idx % 50 == 0 or completed_idx == 1:
                    print(f"  [episode={final_episode}] [{completed_idx}/{len(final_test_dataset)}] 최종 프롬프트 Test 평가 중...")

                if completed_idx % 100 == 0:
                    running_acc = final_test_score / final_test_count if final_test_count > 0 else 0.0
                    _update_test_summary_row(
                        episode=final_episode,
                        test_info=final_test_info,
                        test_accuracy=running_acc,
                        test_dataset_size=len(final_test_dataset),
                    )
                    print(
                        f"  [episode={final_episode}] 중간 저장 완료: "
                        f"{completed_idx}/{len(final_test_dataset)} (acc={running_acc:.4f})"
                    )

        final_test_accuracy = final_test_score / final_test_count if final_test_count > 0 else 0.0
        print(
            f"[episode={final_episode}] 완료: 최종 Test Set 정확도 = "
            f"{final_test_accuracy:.4f} ({final_test_count}/{len(final_test_dataset)}개 평가)"
        )
        _update_test_summary_row(
            episode=final_episode,
            test_info=final_test_info,
            test_accuracy=final_test_accuracy,
            test_dataset_size=len(final_test_dataset),
        )
    else:  # final_test_dataset 없음 (enable_test_evaluation=True 인데 데이터 없는 경우)
        if EXPERIMENT_INS.enable_test_evaluation:
            print(f"[episode={final_episode}] Test 데이터셋 없음, 건너뜁니다.")
        # enable_test_evaluation=False 인 경우: 이미 위에서 skip 메시지 출력함

    # 5. DB 저장 (루프 정상 완료 후 - 마지막 이터레이션 이후 잔여 로그 방어적 저장)
    print_step("5. DB 로그 저장")
    _do_db_save()
    _save_done[0] = True  # atexit 비상 저장 비활성화 (정상 저장 완료)

    print_step("6. 최적화 완료")

    print("\n--- 최적화 완료 ---")
    print(f"Last accepted prompt: {last_prompt_value}")
    print(f"Best validation prompt (used for final test): {global_best_prompt_value}")
    print(f"Final optimized prompt: {system_prompt.value}")

if __name__ == "__main__":
    print_step("=== TextGrad 프롬프트 최적화 시작 ===")
    main()