"""
@경로: main_train.py
@실행명령어: python main_train.py
@설명: 
- 메인 학습 스크립트 (The Conductor)
- DSPy 기반 RAG 모델의 '강화학습 기반 프롬프트 최적화'를 실행합니다.
- OpenAI Gym(Gymnasium) 스타일의 인터페이스를 사용하여 Agent와 Env를 상호작용시킵니다.
- 시간을 흐르게 하며(Loop), Agent의 Action을 Env에 전달하고 결과를 관측합니다.

- 메인 학습 스크립트 (The Conductor) + [데이터 로깅 기능 추가]
- 학습 과정(Episode) 별 상세 결과(질문, 답변, 점수, 피드백 등)를 기록하여 
- 학습 종료 후 'optimization_log.csv' 파일로 저장합니다.

- Azure Content Filter 발생 시 해당 턴을 무효화하고 재시도하는 로직 추가
"""
# ----------------------------------------------------------------
# 기본 라이브러리 import 
import warnings
from utils.log.console import print_step

# Pydantic 관련 경고(UserWarning) 무시하기
warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")

# 1. [Module Imports]
from infrastructure.llm_client import setup_lms
from model.rag_module import RAG_CoT
from agent.optimizer_agent import PromptOptimizerAgent


# 보상
from reward.dspy_hierarchical import hierarchical_feedback_metric # (이전 대화에서 만든 보상함수)
# 학습 용 데이터 셋
from datafile.data_loader import load_dataset
# 최적화기
from optimizers.verbal_reinforce_optimizer import VerbalReinforceOptimizer

# DB 저장을 위한 추가 import
from conf.config import Settings

# ----------------------------------------------------------------
# [Main Execution]
# ----------------------------------------------------------------
if __name__ == "__main__":
    print_step("=== DSPy RAG 모델 강화학습 기반 프롬프트 최적화 시작 ===")

    print_step("0. [Settings] 설정 초기화")
    Settings.setup()
    
    # [Infrastructure] LLM (Azure/OpenAI) 연결 설정
    # infrastructure/llm_client.py 에 설정된 내용을 불러온다.
    print_step("1. [Infrastructure] LLM 연결 설정")
    setup_lms()
    
    print_step("2. 데이터 및 모델 준비")
    # 데이터 셋
    trainset = load_dataset(sample_size=3, random_seed=42)
    if not trainset: exit()  # 데이터 로드 실패 시 종료
    # CleanLLM (TesterLLM) 모델 - RAG 기반 학생 모델 (에피소드 마다 매번 리셋되는 존재)
    student = RAG_CoT()
    
    # TODO 이제 Agent 는 무슨 역할이지..?
    agent = PromptOptimizerAgent()


    print_step("3. Optimizer 설정") 
    optimizer = VerbalReinforceOptimizer(
        metric=hierarchical_feedback_metric,
        agent=agent,
        max_episodes=5,
        log_dir="datafile/results"
    )

    print_step("4. [Execution] 최적화 실행")
    # compile() 한 줄로 모든 학습 과정이 진행된다.
    # compile() 내부에서 Env와 Agent가 상호작용하며 학습이 진행된다.
    # 학습 루프 (The Training Loop) == Main Run 역할을 한다.
    optimized_student = optimizer.compile(student, trainset=trainset)


