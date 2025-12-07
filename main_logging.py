"""
@경로 : main_logging.py
@명령어: python main_logging.py
@설명 : 강화학습 루프를 실행하고 결과(CSV, Excel, Graph)를 저장하는 메인 스크립트
"""

import os
import csv
import time
from datetime import datetime
from typing import Dict, Any
import matplotlib.pyplot as plt # 그래프 그리는 용도
import pandas as pd 

# 프로젝트 구조에 맞게 임포트
from utils.environment.env import PromptOptimizationEnv
from utils.agents.agent import OptimizationAgent
from utils.models.model import ModelFactory
from conf.config import Env
from utils.log.logging import logger

# 환경변수 설정
Env.setup_environment()
if not Env.USE_AZURE:
    Env.check_google_api_key()
elif Env.USE_AZURE:
    Env.check_azure_api_key()

# ==========================================
# [설정] CSV 저장 파일명 및 학습 파라미터
# ==========================================
NUM_EPISODES = 10

def run_training_loop():
    # 1. 환경 및 에이전트 초기화
    
    env = PromptOptimizationEnv()
    agent = OptimizationAgent(version="MDP-v3")  # 기본 정책 버전


    timestamp = datetime.now().strftime("%y%m%d-%H%M%S")
    dataset_name = getattr(env, "dataset_name", "UnknownDataset") # env에서 데이터셋 이름 가져오기
    # 파일명에 데이터셋 이름 포함 (예: rl_KorQuADDataset_episode_log_251204...)
    CSV_EPISODE = f"rl_{dataset_name}_episode_log_{timestamp}.csv"
    CSV_SAMPLES = f"rl_{dataset_name}_samples_log_{timestamp}.csv"

    print("Setting:", "Azure OpenAI" if Env.USE_AZURE else "Google Gemini (Free)")
    print("=" * 60)
    print(f"[DEBUG] USE_AZURE environment variable: {os.getenv('USE_AZURE')}")
    print(f"[DEBUG] USE_AZURE parsed value: {Env.USE_AZURE}")
    print(f"[INFO] 데이터셋: {dataset_name}")
    print(f"[INFO] 프롬프트 최적화 강화학습 시작! (총 {NUM_EPISODES} 에피소드)")
    print(f"[INFO] 로그 파일: {CSV_EPISODE}, {CSV_SAMPLES}")
    print("=" * 60)

    # 2. 히스토리 변수 초기화
    # t-1 (직전 단계): 초기값은 기본 시스템 프롬프트
    initial_prompt = "You are a helpful assistant."
    prompt_t_minus_1 = initial_prompt
    reward_t_minus_1 = 0.0

    # t-2 (2단계 전): 초기값은 없음
    prompt_t_minus_2 = ""
    reward_t_minus_2 = 0.0

    # MDP-v3용: 최악의 오답 사례도 상태로 유지
    worst_case_state: Dict[str, Any] = {}

    best_reward = -1.0
    best_prompt = ""

    # 3. CSV 파일 준비 및 헤더 작성
    episode_fieldnames = [
        "episode",
        "agent_version",
        "prompt_t-2",
        "reward_t-2",
        "prompt_t-1",
        "reward_t-1",
        "action_prompt",
        "response_sample",
        "avg_reward",
        "worst_question",   # 최악 케이스가 뽑혔을 당시의 질문
        "worst_reference",  # 최악 케이스가 뽑혔을 당시의 모범답안
        "worst_prediction", # 최악 케이스가 뽑혔을 당시의 예측값
        "worst_score",
    ]
    sample_fieldnames = [
        "episode",
        "sample_index",
        "question",     # 사용자의 질문
        "action_prompt",# 모델이 사용했던 프롬프트 == agent가 생성한 프롬프트
        "reference",    # 모델이 생성해야 했던 모범답안
        "prediction",   # 모델이 생성한 답변
        "score",        # 당시의 점수
        "context"       # 당시의 문맥 (있다면)
    ]

    # utf-8-sig는 엑셀에서 한글 깨짐 방지용
    with open(CSV_EPISODE, "w", newline="", encoding="utf-8-sig") as ep_file, \
     open(CSV_SAMPLES, "w", newline="", encoding="utf-8-sig") as sm_file:

        ep_writer = csv.DictWriter(ep_file, fieldnames=episode_fieldnames)
        sm_writer = csv.DictWriter(sm_file, fieldnames=sample_fieldnames)

        ep_writer.writeheader()
        sm_writer.writeheader()

        # 4. 학습 루프 시작
        for episode in range(1, NUM_EPISODES + 1):
            state: Dict[str, Any] = {
                "previous_prompt": prompt_t_minus_1,
                "previous_reward": reward_t_minus_1,
                "previous_prompt_t2": prompt_t_minus_2,
                "previous_reward_t2": reward_t_minus_2,
                "worst_case": worst_case_state,
            }

            new_prompt = agent.act(state)

            prediction, avg_reward, info = env.step(new_prompt)
            worst_case_state = info.get("worst_case", {}) or {}
            batch_details = info.get("batch_details", []) 

            worst_q = worst_case_state.get("question", "")
            worst_ref = worst_case_state.get("reference", "")
            worst_pred = worst_case_state.get("prediction", "")
            worst_score = worst_case_state.get("score", 0.0)

            # (1) 에피소드 단위 로그
            ep_writer.writerow(
                {
                    "episode": episode,
                    "agent_version": getattr(agent, "version", "unknown"),
                    "prompt_t-2": prompt_t_minus_2,
                    "reward_t-2": reward_t_minus_2,
                    "prompt_t-1": prompt_t_minus_1,
                    "reward_t-1": reward_t_minus_1,
                    "action_prompt": new_prompt,
                    "response_sample": prediction,
                    "avg_reward": avg_reward,
                    "worst_question": worst_q,
                    "worst_reference": worst_ref,
                    "worst_prediction": worst_pred,
                    "worst_score": worst_score,
                }
            )
            ep_file.flush()

            # (2) 샘플 단위 로그 (batch_size=3개 전부)
            for idx, sample in enumerate(batch_details, start=1):
                sm_writer.writerow(
                    {
                        "episode": episode,
                        "sample_index": idx,
                        "question": sample.get("question", ""),
                        "action_prompt": new_prompt,
                        "reference": sample.get("reference", ""),
                        "prediction": sample.get("prediction", ""),
                        "score": sample.get("score", 0.0),
                        "context": sample.get("context", ""),
                    }
                )
            sm_file.flush()

            # (5) 콘솔 출력
            print(f"\n[INFO] Episode {episode}/{NUM_EPISODES}")
            print(f"[INFO] Prompt: {new_prompt[:80]}...")
            print(f"[INFO] Sample Response: {prediction[:80]}...")
            print(f"[INFO] Avg Reward (batch cosine sim): {avg_reward:.4f}")
            if worst_q:
                print(f"[INFO] Worst Case Score: {worst_score:.4f}")
                print(f"[INFO] Worst Q: {worst_q[:60]}...")
            # 최고 기록 갱신 확인
            if avg_reward > best_reward:
                best_reward = avg_reward
                best_prompt = new_prompt
                print("[INFO] New Best Score!")

            # (6) 히스토리 변수 업데이트 (Shift)
            prompt_t_minus_2 = prompt_t_minus_1
            reward_t_minus_2 = reward_t_minus_1

            prompt_t_minus_1 = new_prompt
            reward_t_minus_1 = avg_reward


    # ==========================================
    # CSV -> 보기 좋은 엑셀로 변환 및 저장
    # ==========================================
    print("\n[INFO] CSV 로그를 엑셀(.xlsx)로 변환 중...")
    
    # 엑셀 파일명 생성
    EXCEL_SAMPLES = CSV_SAMPLES.replace(".csv", ".xlsx")
    
    try:
        # 1. CSV 읽기
        df = pd.read_csv(CSV_SAMPLES)
        
        # 2. 엑셀로 저장 (xlsxwriter 엔진 사용)
        with pd.ExcelWriter(EXCEL_SAMPLES, engine='xlsxwriter') as writer:
            df.to_excel(writer, index=False, sheet_name='Samples')
            
            # 3. 워크시트 객체 가져오기
            workbook  = writer.book
            worksheet = writer.sheets['Samples']
            
            # 4. 열 너비 지정
            worksheet.set_column('A:B', 10)  # episode, index
            worksheet.set_column('C:C', 40)  # question
            worksheet.set_column('D:D', 50)  # action_prompt
            worksheet.set_column('E:E', 50)  # reference
            worksheet.set_column('F:F', 50)  # prediction
            worksheet.set_column('G:G', 10)  # score
            
            # 텍스트 줄바꿈 설정
            text_wrap_format = workbook.add_format({'text_wrap': True, 'valign': 'top'})
            worksheet.set_column('C:C', 50, text_wrap_format)
            worksheet.set_column('D:D', 60, text_wrap_format)
            worksheet.set_column('F:F', 60, text_wrap_format)

        print(f"[INFO] 엑셀 변환 완료! 파일: {os.path.abspath(EXCEL_SAMPLES)}")
        
    except Exception as e:
        print(f"[Warning] 엑셀 변환 실패 (pandas/xlsxwriter 설치 필요): {e}")


    # ==========================================
    # [수정] 보고서용 학습 그래프 그리기 & 저장 (하나의 차트에 겹쳐 그리기)
    # ==========================================
    print(f"\n[INFO] 학습 결과 그래프 생성 중...")
    GRAPH_FILE = CSV_EPISODE.replace(".csv", ".png")

    try:
        # 1. 데이터 로드
        df = pd.read_csv(CSV_EPISODE)

        # 2. 그래프 그리기 (하나의 차트에 겹쳐서)
        plt.figure(figsize=(10, 6))

        # (1) 평균 점수 (Blue)
        plt.plot(df['episode'], df['avg_reward'], marker='o', color='b', label='Avg Reward (Mean)')
        
        # (2) 최악 점수 (Red)
        plt.plot(df['episode'], df['worst_score'], marker='x', color='r', label='Worst Score (Robustness)')

        plt.title(f'Learning Curve: Avg vs Worst ({dataset_name})')
        plt.xlabel('Episode')
        plt.ylabel('Score (Cosine Similarity)')
        plt.ylim(0.0, 1.05) # 점수 범위 고정 (보기 좋게)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend(loc='lower right') # 범례 위치 지정

        # 3. 저장하기
        plt.tight_layout()
        plt.savefig(GRAPH_FILE)
        print(f"[INFO] 그래프 저장 완료! 파일: {os.path.abspath(GRAPH_FILE)}")

    except Exception as e:
        print(f"[Warning] 그래프 생성 실패 (matplotlib 설치 필요): {e}")

    # 5. 최종 결과 출력
    print("\n" + "=" * 60)
    print("[INFO] 학습 완료! 최종 결과")
    print(f"[INFO] 에피소드 로그 파일: {os.path.abspath(CSV_EPISODE)}")
    print(f"[INFO] 샘플 상세 로그 파일: {os.path.abspath(CSV_SAMPLES)}")
    print(f"[INFO] Best Avg Reward: {best_reward:.4f}")
    print(f"[INFO] Best Prompt Summary: {best_prompt[:200]}...")
    print("=" * 60)

if __name__ == "__main__":
    # @명령어: python main_logging.py
    run_training_loop()