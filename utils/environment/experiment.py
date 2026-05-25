"""
textgrad_baseline.py 와 textgrad_improve.py 에서
차별점을 두는 부분을 관리하는 부분.
가능한 main 이 아닌 해당 위치에서만 차이점을 변수 및 함수로 선언해 둔 후 관리 한다.
한눈에, 두 실험의 다른 부분을 파악하기 위한 용도이다.

[클래스 기반 설계]
- TextGradExperiment: 실험 모드(baseline/improve)별 차별화 로직을 캡슐화
- 실험 설정값, 프롬프트, 평가 지시문 등을 self 속성으로 관리
- main 함수에서는 experiment = TextGradExperiment('baseline') 형태로 사용
"""
import os
from typing import Literal, Tuple, List

from utils.llm_patches.textgrad_patches import patch_textgrad_openai_compatibility, patch_textgrad_momentum_compatibility
from datafile.data_loader import load_dataset
from datafile.gsm8k_data_preprocessor import load_gsm8k_test_dataset
from agent.prompts.baseline_prompt import (
    GSM8K_INIT_PROMPT,
    GSM8K_INIT_PROMPT_IMPROVE,
    GPQA_INIT_PROMPT,
    MMLU_INIT_PROMPT,
    DEFAULT_INIT_PROMPT
)
from reward.gsm8k_objective_func import (
    get_gsm8k_baseline_objective_function,
    get_gsm8k_experiment_context,
)

from reward.improve_objective_func import get_improve_objective_function
from textgrad.optimizer.optimizer import get_gradient_and_context_text



class TextGradExperiment:
    """
    TextGrad 실험의 baseline/improve 모드별 차별화 로직을 관리하는 클래스.
    
    @사용법:
        experiment = TextGradExperiment(mode='baseline')
        
        # 데이터셋별 최적화 설정
        test_time_updates = experiment.get_test_time_updates()  # GPQA/MMLU/HQH: 3, 그 외: 1
        
        # 데이터 로드
        dataset, train_pool, validation_dataset = experiment.load_and_split_data()
    
    @차별점 관리:
        1. optimizer_system_prompt - baseline: None (라이브러리 기본값), improve: 커스텀
        2. test_time_updates - GPQA/MMLU/HQH: 3번, 그 외: 1번 (데이터셋 자동 감지)
        3. dataset_name - 실험에 사용할 데이터셋 이름
    """

    _patches_applied = False  # 클래스 변수
    
    @classmethod
    def apply_patches(cls):
        """
        TextGrad 라이브러리 패치를 적용한다. (한 번만 실행됨)
        textgrad 0.1.8 ver 의 버그를 패치하는 용도로, 최상위에서 실행해야 한다.
        (라이브러리 자체를 갈아끼우는 방식)
        """
        if cls._patches_applied:
            return
        
        patch_textgrad_openai_compatibility()
        patch_textgrad_momentum_compatibility()
        cls._patches_applied = True
        print("[✓] TextGrad 패치 적용 완료")

    
    def __init__(self, mode: Literal['baseline', 'improve'] = 'baseline'):
        """
        실험 모드 초기화.
        
        @Args:
            mode: 'baseline' 또는 'improve'
        """
        if mode not in ['baseline', 'improve']:
            raise ValueError(f"mode는 'baseline' 또는 'improve'이어야 합니다. 입력값: {mode}")
        
        self.mode = mode
        self.experiment_id_prefix = f"textgrad_{mode}"
        
        # [중앙 집중식] 데이터셋 시드 - 모든 로드/분할/배치에서 사용
        self.random_seed = 52  # 변경하면 모든 곳에 자동 반영됨 ( 42)
        
        # 실험 설정값 (환경변수 기본값)
        self._load_experiment_config()
    
    def _load_experiment_config(self):
        """
        환경변수 기본값을 로드한다.
        데이터셋별로 TextGrad 논문의 설정을 적용한다.
        
        [TextGrad 논문 재현 설정]
        시스템 프롬프트 최적화(Prompt Optimization) 세팅:
        
        1. GSM8k:
           - Train: 200개 / Validation: 300개
           - Iterations: 12회 / Batch Size: 3
        
        2. Object Counting & Word Sorting (BBH):
           - Train: 50-51개 / Validation: 100개
           - Iterations: 12회 / Batch Size: 3
        
        3. 기타 데이터셋 (GPQA, MMLU, NASA 등):
           - 논문에 명시되지 않은 경우 기본값 사용
        """
        # 데이터셋 이름 설정
        # self.default_dataset_name = "nasa/cmapss-fd001"  # NASA dataset (기존)
        # self.default_dataset_name = "Idavidrein/gpqa-diamond"  # GPQA Diamond - 가장 높은 품질의 문제 (448개)

        # FIX-1 : 논문재현 및 비교 1
        # self.default_dataset_name = "openai/gsm8k"  # GSM8k - Grade School Math 8K (논문 재현)
        
        # FIX-2 : 논문재현 및 비교 2
        self.default_dataset_name = "lukaemon/bbh/object_counting"  # BBH Object Counting - BBH의 객체 수 세기 태스크 (논문 재현 및 개선 모두에서 사용)
        # FIX-3 : 신규 데이터로 실험 
        # self.default_dataset_name = "telagentbench"  # SKT TelAgentBench - 통신/에이전트 벤치마크 데이터셋

        # [설계 방침] episode 개념 없음
        # TextGrad 논문에는 episode 개념이 없고, iteration(step) 단위로만 진행됩니다.
        # DB의 episode 컬럼에는 iteration 번호와 동일한 값이 저장됩니다. (episode == iteration)
        # 데이터셋별 설정 분기
        # [기본값] RAGAS 평가는 필요 데이터셋에서만 켜고, 기본은 활성화로 둔다.
        self.ragas_judge = True
        dataset_name_lower = self.default_dataset_name.lower()
        
        if 'gsm8k' in dataset_name_lower:
            # GSM8k: Grade School Math 8K
            self.default_iterations = 12  # 논문 기준 총 iteration 횟수
            self.default_batch_size = 3
            self.default_total_sample_size = 200  # Train
            self.default_validation_size = 50    # 논문: 300, 비용 절감을 위해 축소
            self.ragas_judge = False  # GSM8k은 RAG 없이 순수 생성 태스크이므로 ragas=False로 설정
            # -- improve --
            self.acceptance_tolerance = 0.09  # 50*0.1 = 5 샘플 노이즈 허용
            self.acceptance_tolerance_gap = 0.005
            

        elif 'lukaemon/bbh' in dataset_name_lower or 'object_counting' in dataset_name_lower or 'word_sorting' in dataset_name_lower:
            # BBH Object Counting / Word Sorting
            # 논문 세팅: Train 50-51, Validation 100, Iterations 12, Batch 3
            self.default_iterations = 12
            self.default_batch_size = 3
            self.default_total_sample_size = 51 # 절대 바꾸지 말 것 (논문 재현용)
            self.default_validation_size = 50  # 논문: 100, 비용 절감을 위해 축소
            self.ragas_judge = False  # BBH는 RAG 없이 순수 생성 태스크이므로 ragas=False로 설정
            # -- improve --
            self.acceptance_tolerance = 0.09  # 50*0.1 = 5 샘플 노이즈 허용
            self.acceptance_tolerance_gap = 0.005
            

        elif 'telagentbench' in dataset_name_lower:
            # SKT TelAgentBench: 통신/에이전트 벤치마크 데이터셋
            # [주의] 논문 고정 세팅이 없으므로 초기 실험용 보수적 기본값 사용
            self.default_iterations = 12
            self.default_batch_size = 3
            self.default_total_sample_size = 200
            self.default_validation_size = 3
            # validation_size=3 → 1샘플=0.33, 샘플 노이즈 크므로 동점만 허용
            self.acceptance_tolerance = 0.0
            
        else:
            # 기타 데이터셋 (GPQA, MMLU, NASA 등) - 논문에 명시되지 않은 경우
            # 환경변수 우선, 없으면 보수적인 기본값 사용
            self.default_iterations = int(os.getenv("TEXTGRAD_ITERATIONS_PER_EPISODE", "2"))
            self.default_batch_size = int(os.getenv("TEXTGRAD_BATCH_SIZE", "1"))
            self.default_total_sample_size = 20   # Train
            self.default_validation_size = 3      # Validation
            self.acceptance_tolerance = 0.0
        
        self.default_initial_prompt = ""

        # [Test 평가 제어]
        # True: episode=0(초기) 및 최종 Test Set 전체 평가 실행 (1,319개 × 2회)
        # False: Test 평가 생략 → 개발/디버깅 시 시간 절약용
        # ★ baseline / improve 공통 플래그
        self.enable_test_evaluation = True  # 1000개 이상의 테스트셋 테스트 건너뛰고 싶은 경우 False 로 설정
        
        print(f"[✓] 데이터셋별 설정 적용: {self.default_dataset_name}")
        print(f"    - Total Iterations: {self.default_iterations}")
        print(f"    - Batch Size: {self.default_batch_size}")
        print(f"    - Train: {self.default_total_sample_size}개, Validation: {self.default_validation_size}개")
        print(f"    - Acceptance Tolerance: {self.acceptance_tolerance} (1샘플={1/self.default_validation_size:.4f})")

    def load_and_split_data(self) -> Tuple[List, List, List]:
        """
        데이터셋을 로드하고 Train/Validation으로 분할한다.
    
        @Return:
            (dataset, train_pool, validation_dataset) 튜플
            - dataset: 전체 로드된 데이터셋
            - train_pool: 복원 추출용 Train 데이터 풀
            - validation_dataset: Validation용 데이터 (빠른 평가용)
        
        @Raises:
            ValueError: 데이터 로드 실패 시
        """
        # 데이터 로드
        dataset = load_dataset(
            dataset_name=self.default_dataset_name,
            sample_size=self.default_total_sample_size + self.default_validation_size,
            random_seed=self.random_seed  # [중앙 집중식] 시드 사용
        )
        if not dataset:
            raise ValueError(f"데이터 로드 실패: {self.default_dataset_name}")

        # Train/Validation 분할 (data_loader가 반환한 순서를 기준으로 슬라이싱)
        train_pool = dataset[:self.default_total_sample_size]
        validation_dataset = dataset[self.default_total_sample_size:self.default_total_sample_size + self.default_validation_size]

        if not train_pool or not validation_dataset:
            raise ValueError(
                f"데이터 로드/분할 실패: {self.default_dataset_name} "
                f"(train={len(train_pool)}, validation={len(validation_dataset)})"
            )
        
        print(f"[✓] Train pool: {len(train_pool)}개, Validation: {len(validation_dataset)}개")
        
        return dataset, train_pool, validation_dataset  

    def load_test_data(self) -> List:
        """
        전체 Test 데이터셋을 로드한다. (초기/최종 성능 측정용 - Apple-to-Apple 비교)

        @논문 근거:
            TextGrad 논문에서는 최적화 전/후 비교를 동일한 Test Set(GSM8k 1,319개)으로 측정합니다.
            현재 최적화 루프는 Validation Set(300개)을 기준으로 프롬프트 채택/거절을 결정하므로,
            이 함수로 로드한 Test Set은 그 평가와는 별개로 논문 기준 성능 비교에 사용됩니다.

        @Return:
            Test 데이터셋 리스트 (전체, 셔플 없음)

        @지원 데이터셋:
            - GSM8k: test.csv (1,319개)
            - 기타: 미지원 (빈 리스트 반환 후 경고)
        """
        dataset_name_lower = self.default_dataset_name.lower()

        if 'gsm8k' in dataset_name_lower:
            # 빠른 테스트를 위해서는 일단은 sample_size = 200 정도로 줄여서 실행 (원본:None)
            test_dataset = load_gsm8k_test_dataset(sample_size=None, random_seed=self.random_seed)  # [중앙 집중식] 시드 사용
            print(f"[✓] Test 데이터셋 로드 완료: {len(test_dataset)}개 (GSM8k test split)")
            return test_dataset
        else:
            print(f"[!] load_test_data: '{self.default_dataset_name}'는 Test split 로드를 지원하지 않습니다. 빈 리스트 반환.")
            return []

    def get_test_time_updates(self) -> int:
        """
        데이터셋에 따른 test-time updates 횟수를 반환한다.
        
        @논문 근거:
            TextGrad 논문에서는 해결하려는 최적화 태스크의 성격에 따라 
            test-time updates(솔루션 생성 반복) 횟수를 다르게 설정합니다.
            
        @Return:
            - GPQA, MMLU, HQH: 3번 (multiple-choice 질문 + majority voting)
            - 그 외: 1번 (기본값, 일반 RAG/생성 태스크)
        
        @참고:
            - GPQA: Graduate-level science questions
            - MMLU: Massive Multitask Language Understanding
            - HQH: High-Quality Hallucination detection
        """
        dataset_name_lower = self.default_dataset_name.lower()
        
        # GPQA, MMLU, HQH 데이터셋은 3번의 test-time updates 적용
        # (multiple-choice 답변 + majority voting 전략)
        if any(keyword in dataset_name_lower for keyword in ['gpqa', 'mmlu', 'hqh']):
            return 3
        
        # 그 외 데이터셋은 1번 (기본 생성)
        return 1
    
    def get_initial_prompt(self) -> str:
        """
        데이터셋과 모드에 맞는 초기 프롬프트를 반환한다.
        
        @논문 근거:
            TextGrad 논문에서는 각 태스크별로 task-specific initial prompt를 설정하여
            최적화의 출발점을 제공합니다. 빈 프롬프트로 시작하면 optimizer가 참고할 
            "현재 버전"이 없어 최적화가 비효율적일 수 있습니다.
        
        @차별점:
            - baseline: 논문 재현을 위해 형식이 포함된 초기 프롬프트 사용
            - improve: OptimizerLLM의 자유도를 높이기 위해 단순화된 초기 프롬프트 사용
        
        @Return:
            데이터셋과 모드에 적합한 초기 프롬프트 문자열
            - GSM8k (baseline): 수학 문제 풀이용 step-by-step 프롬프트 (형식 포함)
            - GSM8k (improve): 단순화된 프롬프트 (OptimizerLLM이 형식 학습)
            - GPQA/MMLU: 객관식 문제용 프롬프트
            - 그 외: 일반 RAG용 기본 프롬프트
        """
        dataset_name_lower = self.default_dataset_name.lower()
        
        # GSM8k: 수학 문제
        if 'gsm8k' in dataset_name_lower:
            if self.mode == 'baseline':
                return GSM8K_INIT_PROMPT  # 형식 포함 (논문 재현)
            elif self.mode == 'improve':
                return GSM8K_INIT_PROMPT_IMPROVE  # 단순화 (OptimizerLLM 학습)
        
        # GPQA: Graduate-level science questions
        elif 'gpqa' in dataset_name_lower:
            return GPQA_INIT_PROMPT
        
        # MMLU: Massive Multitask Language Understanding
        elif 'mmlu' in dataset_name_lower:
            return MMLU_INIT_PROMPT
        
        # 기본값: 일반 RAG (NASA 등)
        else:
            return DEFAULT_INIT_PROMPT
    
    def _build_hierarchical_evaluation_instruction(self, ground_truth: str, similarity_score: float | None = None) -> str:
        """Improve 모드: 계층형 rubric (3-Layer 구조)로 답변 비평"""
        similarity_score_text = "[N/A]"
        if similarity_score is not None:
            similarity_score_text = str(similarity_score)

        return (
            f"[Ground Truth]\n{ground_truth}\n\n"
            "[Reference Similarity Score]\n"
            f"- Gold answer와 현재 답변 간 semantic similarity 참고값: {similarity_score_text}\n"
            "- 단, 이 값은 보조 지표이며 정답 일치 여부보다 우선하지 마세요.\n"
            "[Layer 1: Fact Alignment]\n"
            "- 정답 대비 사실 오류, 누락, 환각 가능성을 먼저 지적하세요.\n"
            "[Layer 2: Context Grounding]\n"
            "- 답변의 핵심 주장별로 문맥 근거 유무를 짚어주세요.\n"
            "[Layer 3: Expression Quality]\n"
            "- 간결성, 명확성, 논리 흐름 개선점을 제안하세요.\n"
            "출력 형식: (1) 치명 오류 3개 이내 (2) 즉시 적용 가능한 개선 지시 3개"
        )
    
    def _build_baseline_evaluation_instruction(self, ground_truth: str) -> str:
        """Baseline 모드: 단순 4가지 기준 평가 (TextGrad 논문 방식)"""
        return (
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
    
    def get_objective_function(
        self,
        ground_truth: str,
        similarity_score: float | None = None,
        accuracy_score: float | None = None,
        prediction: str | None = None,
        previous_rejection_context: str = "",
    ) -> str:
        """
        데이터셋과 실험 모드에 맞는 Objective Function(평가 지시문)을 반환한다.
        
        @논문 근거:
            TextGrad 논문에서는 데이터셋별로 다른 평가 전략을 사용합니다:
            - GSM8k baseline: StringBasedFunction 사용 (이 함수 호출 안 됨)
            - GSM8k improve: 커스텀 평가 지시문 사용
            - 기타 데이터셋: baseline(단순 평가) vs improve(계층형 평가)
        
        @Args:
            ground_truth: 정답 (Reference Answer)
            similarity_score: gold answer와 현재 예측 간 semantic similarity 참고값
            accuracy_score: 정답 일치 여부(0.0/1.0) 참고값
            prediction: TesterLLM이 생성한 전체 사고 과정 (Chain of Thought 텍스트)
        
        @Return:
            Objective Function 문자열 (TextLoss에 전달할 평가 지시문)
        """
        dataset_name_lower = self.default_dataset_name.lower()

        if self.mode == 'improve':
            # Improve 모드에서만 similarity_score와 prediction(Chain of Thought)을 함께 전달
            # 미사용:  self._build_hierarchical_evaluation_instruction(ground_truth, similarity_score=similarity_score)
            print("get_improve_objective_function 를 목적 함수로 사용합니다.")
            return get_improve_objective_function(
                ground_truth,
                similarity_score=similarity_score,
                accuracy_score=accuracy_score,
                student_raw_trajectory=prediction,
                previous_rejection_context=previous_rejection_context,
            )
        else:
            # GSM8k 데이터셋
            if 'gsm8k' in dataset_name_lower:
                if self.mode == 'baseline':
                    print("gsm8k 은 목적함수를 두지 않습니다.")
                    return None
            elif 'object_counting' in dataset_name_lower:
                if self.mode == 'baseline':
                    print("object_counting 은 목적함수를 두지 않습니다.")
                    return None
                
            # 기타 데이터셋 (GPQA, MMLU, NASA 등)
            else:
                return self._build_baseline_evaluation_instruction(ground_truth)
    
    def get_experiment_context(self) -> str:
        """
        실험 컨텍스트 정보를 반환한다.
        system_prompt의 role_description에 포함시켜 비평가와 optimizer가 실험 목표를 이해하도록 돕는다.
        
        @차별점:
            - baseline: 빈 문자열 반환 (TextGrad 논문 재현, 실험 컨텍스트 미제공)
            - improve: 데이터셋별 실험 컨텍스트 반환 (비평가/optimizer가 실험 목표 이해)
        
        @Return:
            모드별, 데이터셋별 실험 컨텍스트 문자열
        """
        # Baseline 모드: 실험 컨텍스트를 제공하지 않음 (논문 재현)
        if self.mode == 'baseline':
            return ""
        
        # Improve 모드: 데이터셋별 실험 컨텍스트 제공
        dataset_name_lower = self.default_dataset_name.lower()
        
        if 'gsm8k' in dataset_name_lower:
            return get_gsm8k_experiment_context()
        else:
            # 기타 데이터셋은 기본 컨텍스트 반환
            return ""

    def build_forward_input(self, question: str, context: str, system_persona: str = "") -> str:
        """
        Forward Model(답변 생성자)에게 전달할 입력 문자열을 구성합니다.

        데이터셋별로 입력 포맷이 다릅니다:
          - TelAgentBench: [Persona & Rules] + [Available Tools] + [User Utterance] 구조
          - GSM8k / 기타: 기존 "Context: ...\nQuestion: ..." 또는 "Question: ..." 형식 (변경 없음)

        @param question: 사용자 발화 (TelAgentBench: conversation에서 추출한 user 메시지)
        @param context: 배경 자료 (TelAgentBench: functions+metadata, GSM8k: 빈 문자열)
        @param system_persona: 고객 페르소나 + 응답 규칙 (TelAgentBench 전용, 그 외 빈 문자열)
        @return: Forward Model에 전달할 최종 입력 문자열
        """
        dataset_name_lower = self.default_dataset_name.lower()

        if 'telagentbench' in dataset_name_lower and system_persona:
            # TelAgentBench: 페르소나/규칙 + 함수 목록 + 사용자 발화를 명확히 구분
            parts = [f"[Persona & Rules]\n{system_persona}"]
            if context.strip():
                parts.append(f"[Available Tools]\n{context}")
            parts.append(f"[User Utterance]\n{question}")
            return "\n\n".join(parts)
        else:
            # GSM8k / 기타: 기존 방식 그대로 (영향 없음)
            if context.strip():
                return f"Context: {context}\nQuestion: {question}"
            else:
                return f"Question: {question}"

    def extract_feedback_str(self, system_prompt, optimization_logs: list = None, iteration_log_start_idx: int = None) -> str:
        """
        backward() 실행 후 생성된 프롬프트 피드백을 추출한다.
        
        @Args:
            system_prompt: TextGrad Variable 객체 (system_prompt)
            optimization_logs: DB 저장용 로그 버퍼 (improve 모드에서 샘플 비평 추출용)
            iteration_log_start_idx: 현재 iteration 로그 시작 인덱스 (improve 모드용)
        
        @Return:
            - baseline: 단순 gradient 텍스트
            - improve: gradient + 샘플 비평을 3계층 구조화
        """
        if self.mode == 'baseline':
            # Baseline: gradient + <CONVERSATION> 컨텍스트 포함 텍스트 반환
            return str(get_gradient_and_context_text(system_prompt)).strip() or "[N/A]"
        
        elif self.mode == 'improve':
            # Improve: 계층형 피드백 구조
            gradient_text = str(get_gradient_and_context_text(system_prompt))
            
            # 샘플 비평 수집
            sample_feedbacks: list[str] = []
            if optimization_logs and iteration_log_start_idx is not None:
                for row in optimization_logs[iteration_log_start_idx:]:
                    raw_feedback = row.get('answer_feedback')
                    if raw_feedback is None:
                        continue
                    # normalize_text_field 대신 간단한 정규화 (import 없이)
                    normalized_feedback = str(raw_feedback).strip()
                    if normalized_feedback:
                        sample_feedbacks.append(normalized_feedback)
                    if len(sample_feedbacks) >= 5:
                        break
            
            if not sample_feedbacks:
                sample_feedback_block = "- [N/A] 유효한 샘플 비평이 없어 gradient 중심으로 업데이트"
            else:
                # 차별점 #############################################################################################
                # 각 샘플에 번호를 붙여서 구조화
                formatted_samples = []
                for idx, feedback in enumerate(sample_feedbacks, 1):
                    formatted_samples.append(
                        f"<{idx}번째 테스트>\n{feedback}\n</{idx}번째 테스트>"
                    )
                sample_feedback_block = "\n\n".join(formatted_samples)
            
            cleaned_gradient = gradient_text.strip() or "[N/A] TextGrad prompt feedback is empty."
            
            return (
                "<프롬프트 개선 방향>\n"
                f"{cleaned_gradient}\n\n"
                "</프롬프트 개선 방향>\n\n"
                "<각 Train Sample 피드백>\n"
                f"{sample_feedback_block}\n\n"
                "</각 Train Sample 피드백>\n"
            )
        
        else:
            # fallback: baseline 방식
            return system_prompt.get_gradient_text().strip() or "[N/A]"
        
    
    # ------------------------------- 실험 Config 관리 -------------------------------
    @property
    def is_numeric_exact_match_dataset(self) -> bool:
        dataset_name_lower = self.default_dataset_name.lower()

        is_gsm8k = 'gsm8k' in dataset_name_lower
        is_object_counting = 'object_counting' in dataset_name_lower

        is_numeric_exact_match_dataset = is_gsm8k or is_object_counting

        print(
            f"[✓] 데이터셋 타입: "
            f"gsm8k={is_gsm8k}, object_counting={is_object_counting}, "
            f"numeric_exact_match={is_numeric_exact_match_dataset}"
        )

        return is_numeric_exact_match_dataset
    
   