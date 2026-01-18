"""
@경로: main_train.py
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

import dspy
import os
import warnings
import pandas as pd  # 데이터 저장을 위해 추가
from datetime import datetime
from pathlib import Path
import time # 재시도 시 잠시 대기를 위해 추가

# Pydantic 관련 경고(UserWarning) 무시하기
warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")

# 1. [Module Imports]
from infrastructure.llm_client import setup_lms
from models.rag_module import RAG_CoT, RAGSignature
from agent.optimizer_agent import PromptOptimizerAgent
from environment.dspy_rag_env import DSPyRAGEnv
from reward.dspy_hierarchical import hierarchical_feedback_metric # (이전 대화에서 만든 보상함수)
from datasets.data_loader import load_dataset
# ----------------------------------------------------------------
# [Global Logger] 전역 로깅 변수 및 래퍼 함수
# ----------------------------------------------------------------
HISTORY_DATA = []      # 모든 로그가 여기에 쌓입니다.
CURRENT_EPISODE = 0    # 현재 에피소드 번호를 추적합니다.
CURRENT_INSTRUCTION = "" # 현재 적용된 지시문을 추적합니다.

def wrapped_metric(gold, pred, trace=None):
    """
    [Metric Wrapper]
    기존 보상 함수(hierarchical_feedback_metric)를 실행하되,
    그 결과를 가로채서 HISTORY_DATA에 저장하는 역할을 합니다.
    """
    # 1. 원래 보상 함수 실행 (점수 계산 및 pred에 로그 부착)
    score = hierarchical_feedback_metric(gold, pred, trace)
    
    # 2. pred 객체에 부착된 상세 로그 가져오기
    log = getattr(pred, 'feedback_log', {})
    
    # 3. 데이터 추출 (안전하게 get 사용)
    question = log.get("Goal", {}).get("current_question", "")
    gold_answer = log.get("Goal", {}).get("reference_answer", "")
    model_answer = pred.answer
    
    # Context 추출 (gold.context가 리스트일 수도, 문자열일 수도 있음)
    context_raw = getattr(gold, 'context', "")
    context_str = " ".join(context_raw) if isinstance(context_raw, list) else str(context_raw)

    score_card = log.get("ScoreCard", {})
    analysis = log.get("Analysis", "")
    
    # 4. 한 행(Row) 데이터 만들기
    row = {
        "Episode": CURRENT_EPISODE,
        "Instruction": CURRENT_INSTRUCTION, # 당시의 프롬프트
        "Question": question,
        "Context": context_str,
        "Model_Answer": model_answer,
        "Gold_Answer": gold_answer,
        "Total_Score": score,
        "Raw_Similarity": score_card.get("raw_similarity", 0.0),
        "Is_Faithful": score_card.get("faithfulness", "Unknown"),
        "Is_Style_Match": score_card.get("format", "Unknown"),
        "Critical_Review": score_card.get("critical_review", "None"),
        "Full_Analysis": analysis,
        # 학습 루프 자체의 성공 유무
        "Is_Success": True,  # 성공함
        "Error_Log": ""      # 에러 없음
    }
    
    # 5. 전역 리스트에 추가
    HISTORY_DATA.append(row)
    
    return score

# ----------------------------------------------------------------
# [Data Loader] 실험용 데이터셋 준비 (실제 데이터로 교체 필요)
# ----------------------------------------------------------------


# ----------------------------------------------------------------
# [Main Execution]
# ----------------------------------------------------------------
if __name__ == "__main__":
    
    # 1. [Infrastructure] LLM (Azure/OpenAI) 연결 설정
    # infrastructure/llm_client.py 에 설정된 내용을 불러옵니다.
    setup_lms()
    
    # 2. [Initialization] 환경 세팅
    print("\n[System] Initializing RL Components...")
    
    # (A) 데이터셋 로드
    trainset = load_dataset(sample_size=5)
    if not trainset: exit()  # 데이터 로드 실패 시 종료
    
    # (B) 학생 모델 생성 (테스터, 매번 리셋되는 존재)
    student = RAG_CoT() 
    
    # (C) 감독관 생성 (Env) - 학생과 데이터, 채점기준을 가지고 있음    
    env = DSPyRAGEnv(
        student_module=student, 
        trainset=trainset, 
        metric_fn=wrapped_metric
    )
    
    # (D) 코치 생성 (Agent) - 프롬프트를 최적화하는 두뇌
    agent = PromptOptimizerAgent()

    # 3. [Baseline Evaluation] 초기 상태 평가
    # 학습 시작 전, 기본 프롬프트(또는 빈 프롬프트) 상태에서의 점수를 측정합니다.
    print(f"\n>>> [Episode 0] Baseline Evaluation (Before Optimization)")
    
    CURRENT_EPISODE = 0  # 전역 변수 업데이트
    # 초기 프롬프트 가져오기 (docstring)
    # CURRENT_INSTRUCTION = student.prog.signature.__doc__
    # student.prog.signature.__doc__ 은 아래와 같은 에러가 발생했다. 
    # # AttributeError: 'ChainOfThought' object has no attribute 'signature'
    CURRENT_INSTRUCTION = RAGSignature.__doc__      # signature의 docstring(=instructions)

    # 베이스라인도 필터에 걸릴 수 있으므로 재시도 로직 적용
    while True:
        try:
            # action=None을 주면, Env는 프롬프트 변경 없이 평가만 수행하고 State(피드백)를 반환합니다.
            state, reward, done = env.step(action=None)
            print(f"    - Baseline Score: {reward:.2f}")
            break # 성공하면 탈출
        except Exception as e:
            print(f"\n[Warning] Baseline 평가 중 Azure Filter 발동. 3초 후 재시도합니다... ({e})")
            time.sleep(3)

    
    
    print(f"    - Baseline Score: {reward:.2f}")
    print(f"    - Initial Instruction: {state['current_instruction'][:50]}...")

    # 4. [Optimization Loop] 강화학습 루프 시작 (Action -> Step -> Observe)
    MAX_EPISODES = 5
    successful_episodes = 0 # 성공한 횟수 카운터
    
    # [핵심] 성공 횟수가 목표치에 도달할 때까지 무한 반복
    # Azure API 자체 필터링으로 인해 에피소드가 중간에 실패할 수 있으므로, 일정 횟수를 보장하기 위함.
    while successful_episodes < MAX_EPISODES:

        # 현재 몇 번째 성공을 목표로 하는지 (실패해도 이 번호로 기록됨)
        ep_num = successful_episodes + 1
        print(f"\n>>> [Episode {ep_num}] Optimization Step (Attempting...)")

        try:
        
            # (Step 1) Agent: State(오답노트, 점수 등)를 보고 Action(새 지시문) 결정
            # Agent는 내부적으로 LLM을 사용하여 이전의 실패 원인을 분석합니다.
            # Agent가 프롬프트를 만드는 과정에서도 필터가 걸릴 수 있음
            new_instruction = agent.act(state)
            
            # (Stop Condition) Agent가 더 이상 고칠 게 없다고 판단하면 종료
            if new_instruction is None:
                print("    [Stop] Agent decided to stop optimization (No critical feedback).")
                break
                
            print(f"    [Agent] Generated New Instruction (Action)")

            # (Step 2) Env: Action 적용 -> 재시험 -> Next State 반환
            # 감독관이 새 지시문을 학생에게 주입하고 다시 시험을 치르게 합니다.
            # Env Step (이때 내부적으로 evaluate -> wrapped_metric 호출됨)
            CURRENT_EPISODE = ep_num          # 전역 변수 업데이트
            CURRENT_INSTRUCTION = new_instruction # 전역 변수 업데이트

            # 여기서 에러나면 -> HISTORY_DATA에 추가 안 됨 -> except로 이동
            next_state, reward, done = env.step(action=new_instruction)
            # [설명] state를 굳이 next_state로 받는 이유 (보존의 법칙)
            # state는 Agent가 다음 행동을 결정하기 위한 입력값이다.
            # env.step()이 실행되다가 에러가 터졌을때, state 는 이전 상태로 남아있어야 한다.
            # reward, done은 그냥 써도 되는 이유 (일회용 성적표)
            # reward와 done은 다음 행동을 위한 입력값이 아니라, 방금 행동에 대한 결과이다. 
            # 즉 env.step() 이 실행되다가 에러가 나면 reward와 done은 어차피 무효화되므로 보존할 필요가 없다.
            
            print(f"    [Env] Evaluation Complete. Score: {reward:.2f}")
            
            # --- [여기까지 오면 성공!] ---
            # 성공했으므로 상태 업데이트 및 카운트 증가
            state = next_state
            successful_episodes += 1

            # (Step 3) Check Done
            if done: # 만점(1.0) 도달 시
                print("\n    [INFO] Perfect Score Achieved! Optimization Complete.")
                break
        except Exception as e:
            # --- [실패 시 처리] ---
            print(f"\n[Blocked] Azure Content Filter 발동!")
            print(f"     [Reason] {e}")
            
            # 1. 실패 로그를 수동으로 생성하여 기록합니다.
            fail_row = {
                "Episode": ep_num,
                "Instruction": CURRENT_INSTRUCTION, # 실패한 원흉 프롬프트
                "Question": "Blocked by Filter",    # 질문 단계에서 막혔을 수 있음
                "Context": "",
                "Model_Answer": "",
                "Gold_Answer": "",
                "Total_Score": 0.0,                 # 0점 처리
                "Raw_Similarity": 0.0,
                "Is_Faithful": "Fail",
                "Is_Style_Match": "Fail",
                "Critical_Review": "Azure Content Policy Violation",
                "Full_Analysis": f"Azure Filter Triggered: {str(e)}",
                
                # 실패 표시
                "Is_Success": False, 
                "Error_Log": str(e)
            }
            
            # 로그 리스트에 추가
            HISTORY_DATA.append(fail_row)
            print("     실패 이력을 로그에 기록했습니다.")

            # 2. Agent에게 실패 원인 주입 (학습용)
            state["fail_case_feedback"] = (
                f"[SYSTEM ERROR] Azure Content Filter(안전 정책)에 의해 프롬프트가 차단되었습니다.\n"
                f"상세 에러: {str(e)}\n"
                f"Action Required: 위 에러를 피할 수 있도록 지시문을 수정하여 다시 작성하세요."
            )
            
            print("     [RETRY] 실패 피드백을 Agent에게 전달하고 다시 시도합니다...")
            time.sleep(2)
            continue # 재시도 (successful_episodes는 증가 안 함)

    # 5. [Final Result] 최종 결과 저장 및 출력
    print("\n=============================================")
    print("           OPTIMIZATION FINISHED             ")
    print("=============================================")
    print(f"Final Score: {state.get('current_similarity_score', 0.0):.2f}")
    print("\n[Best Optimized Instruction]:")
    print(state['current_instruction'])
    
    # (Optional) 최적화된 프롬프트가 적용된 모델 저장
    # student.save("optimized_rag_student.json")
    # print("\n[System] Optimized model saved to 'optimized_rag_student.json'")

    # 5. [Save Logs] 결과 저장
    print("\n=============================================")
    print("           SAVING RESULTS TO CSV             ")
    print("=============================================")
    
    # DataFrame 변환
    df = pd.DataFrame(HISTORY_DATA)

    # 프로젝트 루트 기준 상대 경로
    save_dir = Path("datasets/results")

    # 폴더가 없으면 생성 (부모 폴더까지 포함해서 생성)
    save_dir.mkdir(parents=True, exist_ok=True)

    
    # 파일명 생성 (시간 포함)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"optimization_log_{timestamp}.csv"

    # 전체 경로 결합
    full_path = save_dir / filename
    
    # CSV 저장 (한글 깨짐 방지 utf-8-sig)
    df.to_csv(full_path, index=False, encoding='utf-8-sig')
    
    print(f"[Complete] Log saved to: {full_path.resolve()}")
    print(f"Total Rows: {len(df)}")
    
    # 엑셀로도 저장하고 싶으면 아래 주석 해제
    # df.to_excel(f"optimization_log_{timestamp}.xlsx", index=False)