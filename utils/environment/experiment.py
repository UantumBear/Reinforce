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

from metrics.prompts.textgrad_improve_prompts import TEXTGRAD_IMPROVE_PROMPT_V001
from utils.llm_patches.textgrad_patches import patch_textgrad_openai_compatibility, patch_textgrad_momentum_compatibility
from datafile.data_loader import load_dataset
from agent.prompts.baseline_prompt import (
    GSM8K_INIT_PROMPT,
    GPQA_INIT_PROMPT,
    MMLU_INIT_PROMPT,
    DEFAULT_INIT_PROMPT
)
from reward.gsm8k_objective_func import (
    get_gsm8k_baseline_objective_function,
    get_gsm8k_improve_objective_function
)



class TextGradExperiment:
    """
    TextGrad 실험의 baseline/improve 모드별 차별화 로직을 관리하는 클래스.
    
    @사용법:
        experiment = TextGradExperiment(mode='baseline')
        
        # 데이터셋별 최적화 설정
        test_time_updates = experiment.get_test_time_updates()  # GPQA/MMLU/HQH: 3, 그 외: 1
        optimizer_prompt = experiment.get_optimizer_system_prompt()
        
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
        self.default_dataset_name = "openai/gsm8k"  # GSM8k - Grade School Math 8K (논문 재현)

        # [설계 방침] episode 개념 없음
        # TextGrad 논문에는 episode 개념이 없고, iteration(step) 단위로만 진행됩니다.
        # DB의 episode 컬럼에는 iteration 번호와 동일한 값이 저장됩니다. (episode == iteration)
        # 데이터셋별 설정 분기
        dataset_name_lower = self.default_dataset_name.lower()
        
        if 'gsm8k' in dataset_name_lower:
            # GSM8k: Grade School Math 8K
            self.default_iterations = 12  # 논문 기준 총 iteration 횟수
            self.default_batch_size = 3
            self.default_total_sample_size = 200  # Train
            self.default_validation_size = 20     # Validation (원래 300 → 속도 개선을 위해 20으로 축소)
            
        elif any(keyword in dataset_name_lower for keyword in ['object_counting', 'word_sorting', 'bbh']):
            # Object Counting & Word Sorting (Big-Bench Hard)
            self.default_iterations = 12  # 논문 기준 총 iteration 횟수
            self.default_batch_size = 3
            self.default_total_sample_size = 50   # Train (50-51개)
            self.default_validation_size = 100    # Validation
            
        else:
            # 기타 데이터셋 (GPQA, MMLU, NASA 등) - 논문에 명시되지 않은 경우
            # 환경변수 우선, 없으면 보수적인 기본값 사용
            self.default_iterations = int(os.getenv("TEXTGRAD_ITERATIONS_PER_EPISODE", "2"))
            self.default_batch_size = int(os.getenv("TEXTGRAD_BATCH_SIZE", "1"))
            self.default_total_sample_size = 20   # Train
            self.default_validation_size = 5      # Validation
        
        self.default_initial_prompt = ""
        
        print(f"[✓] 데이터셋별 설정 적용: {self.default_dataset_name}")
        print(f"    - Total Iterations: {self.default_iterations}")
        print(f"    - Batch Size: {self.default_batch_size}")
        print(f"    - Train: {self.default_total_sample_size}개, Validation: {self.default_validation_size}개")

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
            random_seed=42
        ) # random_seed로 셔플된 데이터 반환
        
        if not dataset:
            raise ValueError(f"데이터 로드 실패: {self.default_dataset_name}")
        
        # Train/Validation 분할 (이미 셔플되어 있으므로 바로 슬라이싱)
        train_pool = dataset[:self.default_total_sample_size]
        validation_dataset = dataset[self.default_total_sample_size:self.default_total_sample_size + self.default_validation_size]
        
        print(f"[✓] Train pool: {len(train_pool)}개, Validation: {len(validation_dataset)}개")
        
        return dataset, train_pool, validation_dataset  
        
    
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
    
    def get_optimizer_system_prompt(self) -> str | None:
        """
        OptimizerLLM 시스템 프롬프트를 반환한다.
        
        @Return:
            - baseline: None (라이브러리 기본 시스템 프롬프트 사용)
            - improve: 구조적 언어 피드백 적용 커스텀 프롬프트
        """
        if self.mode == 'baseline':
            return None  # 라이브러리 기본값 사용
        elif self.mode == 'improve':
            return TEXTGRAD_IMPROVE_PROMPT_V001
    
    def get_initial_prompt(self) -> str:
        """
        데이터셋에 맞는 초기 프롬프트를 반환한다.
        
        @논문 근거:
            TextGrad 논문에서는 각 태스크별로 task-specific initial prompt를 설정하여
            최적화의 출발점을 제공합니다. 빈 프롬프트로 시작하면 optimizer가 참고할 
            "현재 버전"이 없어 최적화가 비효율적일 수 있습니다.
        
        @Return:
            데이터셋에 적합한 초기 프롬프트 문자열
            - GSM8k: 수학 문제 풀이용 step-by-step 프롬프트
            - GPQA/MMLU: 객관식 문제용 프롬프트
            - 그 외: 일반 RAG용 기본 프롬프트
        """
        dataset_name_lower = self.default_dataset_name.lower()
        
        # GSM8k: 수학 문제
        if 'gsm8k' in dataset_name_lower:
            return GSM8K_INIT_PROMPT
        
        # GPQA: Graduate-level science questions
        elif 'gpqa' in dataset_name_lower:
            return GPQA_INIT_PROMPT
        
        # MMLU: Massive Multitask Language Understanding
        elif 'mmlu' in dataset_name_lower:
            return MMLU_INIT_PROMPT
        
        # 기본값: 일반 RAG (NASA 등)
        else:
            return DEFAULT_INIT_PROMPT
    
    def get_objective_function(self, ground_truth: str) -> str:
        """
        데이터셋과 실험 모드에 맞는 Objective Function(평가 지시문)을 반환한다.
        
        @논문 근거:
            TextGrad 논문에서는 데이터셋별로 다른 평가 전략을 사용합니다:
            - GSM8k baseline: 실제 논문에서는 목적 함수를 사용하지 않음 (빈 문자열 반환)
            - GSM8k improve: 커스텀 평가 지시문 사용
            - 기타 데이터셋: 일반 RAG 평가 지시문
        
        @Args:
            ground_truth: 정답 (Reference Answer)
        
        @Return:
            Objective Function 문자열 (TextLoss에 전달할 평가 지시문)
        """
        dataset_name_lower = self.default_dataset_name.lower()
        
        # GSM8k 데이터셋
        if 'gsm8k' in dataset_name_lower:
            if self.mode == 'baseline':
                return get_gsm8k_baseline_objective_function(ground_truth)
            elif self.mode == 'improve':
                return get_gsm8k_improve_objective_function(ground_truth)
        
        # 기타 데이터셋 (GPQA, MMLU, NASA 등) - 추후 확장
        # TODO: 각 데이터셋별 objective function 구현
        else:
            # 임시: 일반 RAG 평가 지시문 (기본값)
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
    
    
    # def build_evaluation_instruction(self, ground_truth: str) -> str:
    #     """
    #     답변 평가용 지시문을 생성한다 (TextLoss에 전달).
        
    #     @Args:
    #         ground_truth: 모범답안 (정답)
        
    #     @Return:
    #         - baseline: 단순 4가지 기준 평가
    #         - improve: 계층형 3-Layer rubric 평가
    #     """
    #     if self.mode == 'baseline':
    #         return self._build_baseline_evaluation_instruction(ground_truth)
    #     elif self.mode == 'improve':
    #         return self._build_hierarchical_evaluation_instruction(ground_truth)
    
    # def _build_baseline_evaluation_instruction(self, ground_truth: str) -> str:
    #     """Baseline: 단순 평가 기준 (TextGrad 논문 방식)"""
    #     return (
    #         "You are a critical and rigorous evaluator for RAG systems. "
    #         "Your task is to examine the predicted answer step-by-step and identify potential flaws.\n\n"
    #         f"**Reference Answer:** {ground_truth}\n\n"
    #         "**Evaluation Criteria:**\n"
    #         "1. Does the prediction fully address the question based on the given context?\n"
    #         "2. Are there any factual inaccuracies or hallucinations?\n"
    #         "3. Is the reasoning clear and logically sound?\n"
    #         "4. What specific improvements would make this answer better?\n\n"
    #         "Provide concise, actionable feedback focused on how to improve the answer generation prompt."
    #     )
    
    # def _build_hierarchical_evaluation_instruction(self, ground_truth: str) -> str:
    #     """Improve: 계층형 rubric (3-Layer 구조)"""
    #     return (
    #         f"[Ground Truth]\n{ground_truth}\n\n"
    #         "[Layer 1: Fact Alignment]\n"
    #         "- 정답 대비 사실 오류, 누락, 환각 가능성을 먼저 지적하세요.\n"
    #         "[Layer 2: Context Grounding]\n"
    #         "- 답변의 핵심 주장별로 문맥 근거 유무를 짚어주세요.\n"
    #         "[Layer 3: Expression Quality]\n"
    #         "- 간결성, 명확성, 논리 흐름 개선점을 제안하세요.\n"
    #         "출력 형식: (1) 치명 오류 3개 이내 (2) 즉시 적용 가능한 개선 지시 3개"
    #     )
    
    # def should_compact_gradients(self) -> bool:
    #     """
    #     Gradient 압축 활성화 여부를 반환한다.
        
    #     @Return:
    #         - baseline: False (압축 안 함)
    #         - improve: True (구조적 언어 피드백 적용을 위해 압축)
    #     """
    #     return self.mode == 'improve'
    
    # def should_use_hierarchical_feedback(self) -> bool:
    #     """
    #     계층적 피드백 구조 사용 여부를 반환한다.
        
    #     @Return:
    #         - baseline: False (단순 gradient 텍스트 사용)
    #         - improve: True (build_hierarchical_prompt_feedback 사용)
    #     """
    #     return self.mode == 'improve'
    
    # def get_episodes(self) -> int:
    #     """환경변수 또는 기본값에서 Episodes 수를 가져온다."""
    #     return int(os.getenv("TEXTGRAD_EPISODES", str(self.default_episodes)))
    
    # def get_iterations_per_episode(self) -> int:
    #     """환경변수 또는 기본값에서 Iterations/Episode 수를 가져온다."""
    #     return int(os.getenv("TEXTGRAD_ITERATIONS_PER_EPISODE", str(self.default_iterations)))
    
    # def get_batch_size(self) -> int:
    #     """환경변수 또는 기본값에서 Batch size를 가져온다."""
    #     return int(os.getenv("TEXTGRAD_BATCH_SIZE", str(self.default_batch_size)))
    
    # def get_dataset_name(self) -> str:
    #     """실험 모드별 기본 데이터셋 이름을 반환한다."""
    #     return os.getenv("TEXTGRAD_DATASET", self.default_dataset)
    
    # def get_total_sample_size(self) -> int:
    #     """환경변수 또는 기본값에서 Train pool 크기를 가져온다."""
    #     return int(os.getenv("TEXTGRAD_TOTAL_SAMPLES", str(self.default_total_samples)))
    
    # def get_validation_size(self) -> int:
    #     """환경변수 또는 기본값에서 Validation set 크기를 가져온다."""
    #     return int(os.getenv("TEXTGRAD_VALIDATION_SIZE", str(self.default_validation_size)))
    
    # def get_initial_prompt(self) -> str:
    #     """실험 모드별 초기 시스템 프롬프트를 반환한다."""
    #     return self.default_initial_prompt
    
    # def get_experiment_id(self, timestamp: str) -> str:
    #     """
    #     실험 ID를 생성한다.
        
    #     @Args:
    #         timestamp: datetime.now().strftime('%Y%m%d_%H%M%S')
        
    #     @Return:
    #         예: "textgrad_baseline_20260324_143022"
    #     """
    #     return f"{self.experiment_id_prefix}_{timestamp}"
    
    # def __repr__(self) -> str:
    #     return (
    #         f"TextGradExperiment(mode='{self.mode}', "
    #         f"episodes={self.get_episodes()}, "
    #         f"batch_size={self.get_batch_size()}, "
    #         f"dataset='{self.get_dataset_name()}')"
    #     )