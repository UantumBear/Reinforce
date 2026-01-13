"""
Docstring for utils.tools.dspy_ai.run_optimization

@명령어 예시:
1. 논문 재현 모드 (기본값):
   python utils/tools/dspy_ai/run_optimization.py --mode paper

2. 내 연구 모드 (커스텀 메트릭):
   python utils/tools/dspy_ai/run_optimization.py --mode research
"""
import csv
import os
import json
import pandas as pd
import argparse
import dspy
import random
from datetime import datetime
from dspy.teleprompt import COPRO, BootstrapFewShot
from dspy.evaluate import Evaluate

# 사용자 정의 모듈 Import
from utils.tools.dspy_ai.common.llm_setup import setup_dspy_llm
from utils.tools.dspy_ai.common.load_dspy_data import load_data_for_dspy
# 사용자 정의 보상 설계 함수 Import
from utils.reward.dspy_hierarchical_reward import hierarchical_feedback_metric




# --- [설정] 인자 파싱 (Argument Parsing) ---
parser = argparse.ArgumentParser(description="DSPy Optimization Runner")
parser.add_argument('--mode', type=str, default='paper', choices=['paper', 'research'], 
                    help="실행 모드 선택: 'paper'(논문재현) 또는 'research'(내연구)")
args = parser.parse_args()

# --- [0] 환경 설정 ---
lm = setup_dspy_llm()

# --- [1] 데이터셋 준비 ---
print(f"[INFO] 현재 모드: {args.mode.upper()}")
print("[PROCESS] 데이터셋 로드 중...")
all_data, dataset_name = load_data_for_dspy()
print(f"[INFO] 사용 데이터셋: {dataset_name}")

random.seed(42)
random.shuffle(all_data)

trainset = all_data[:5]    # 0번 ~ 4번 (딱 5개!) -> 학습에 사용
devset = all_data[5:10]    # 5번 ~ 9번 (딱 5개!) -> 평가에 사용

# --- [2] 모듈 & 시그니처 정의 ---
# RAG 데이터셋은 '질문'과 '문서(Context)'를 보고 답해야 한다.
# 놀랍게도, RAGSignature 클래스 안에 들어가는 주석의 내용이 프롬프트가 된다..!
# 가장 큰 방향성(페르소나, 말투, 금기사항)은 이 클래스 주석(Docstring)에 적어야 효과가 제일 좋다고 한다..
# 또한 desc 파라미터도 프롬프트에 반영되므로, 적절히 활용하면 좋다. 
# 난 일단 테스트를 위해 중립적으로 적어두었다. 
# answer의 desc를 "단답형 답변" 으로 적으면, fewshot 데이터가 장문이더라도 단답형으로 유도된다.
class RAGSignature(dspy.Signature):
    """Answer the question based on the provided context."""
    context = dspy.InputField(desc="참고할 문서 내용")
    question = dspy.InputField(desc="질문")
    answer = dspy.OutputField(desc="질문에 대한 답변")

class RAG_CoT(dspy.Module):
    """
    RAG_CoT()란?
    사용자가 정의한 모듈(Module) 클래스. 질문과 문서를 받아서 답변을 생성하는 역할을 한다.
    """
    def __init__(self):
        super().__init__()
        self.prog = dspy.ChainOfThought(RAGSignature)
    def forward(self, question, context):
        """
        Docstring for forward
        
        forward 함수란?
        LLM에게 실제로 프롬프트를 보내고, 응답을 받는 역할을 한다.
        여기서는 질문과 문서를 받아서 답변을 생성한다.
        """
        # 1. LLM에게 예측 요청
        prediction = self.prog(question=question, context=context)
        
        # 2. 추가 정보 삽입
        # 새로운 보상 함수가 'Faithfulness(환각 여부)'를 검사하려면 
        # "이 모델이 어떤 Context를 보고 답했는지" 정보가 prediction 안에 있어야 한다.
        # 따라서 수동으로 context 정보를 prediction 객체에 붙여줍니다.
        # prediction['context'] = context # ... 최종적으로제거 !!
        
        return prediction

# --- [3] 메트릭 선택 로직 (핵심!) ---
# if args.mode == 'paper':
#     # 논문 재현용: 정확히 일치해야 정답 (Hard Metric)
#     selected_metric = dspy.evaluate.answer_exact_match
#     print("[INFO] 메트릭 설정: Exact Match (논문 재현용)")
    
# elif args.mode == 'research':
#     # 내 연구용: 계층적 보상 함수 (Soft Metric + Feedback)
#     selected_metric = hierarchical_feedback_metric
#     print("[INFO] 메트릭 설정: Hierarchical Feedback Metric (연구용 - Constitutional AI)")
class MetricWrapper:
    def __init__(self, metric_fn, log_file):
        self.metric_fn = metric_fn
        self.log_file = log_file
        self.step_count = 0
        
        # 파일이 없으면 헤더(제목)를 먼저 씁니다.
        if not os.path.exists(self.log_file):
            with open(self.log_file, "w", encoding="utf-8-sig", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["Step", "Question", "Model_Answer", "Score", "True_Answer"])

    def __call__(self, example, pred, trace=None):
        # 1. 원래 DSPy 메트릭 함수로 채점
        score = self.metric_fn(example, pred, trace)
        
        # 2. 결과 기록 준비
        self.step_count += 1
        question = getattr(example, 'question', 'N/A')
        true_answer = getattr(example, 'answer', 'N/A')
        pred_answer = getattr(pred, 'answer', 'N/A')
        
        # 3. CSV 파일에 한 줄 추가 (실시간 저장)
        with open(self.log_file, "a", encoding="utf-8-sig", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([self.step_count, question, pred_answer, score, true_answer])
            
        return score

# --- [3] 메트릭 선택 로직 (수정됨: 래퍼 적용) ---
# 로그 파일명 자동 생성 (score_log_데이터셋_모드_시간.csv)
current_time = datetime.now().strftime("%H%M%S")
score_log_file = f"datasets/practice/score_log_{dataset_name}_{args.mode}_{current_time}.csv"

print(f"[INFO] 실시간 점수 로그가 '{score_log_file}' 에 저장됩니다.")

if args.mode == 'paper':
    # [논문 모드] 원래 메트릭을 base_metric에 담습니다.
    base_metric = dspy.evaluate.answer_exact_match
    print("[INFO] 기본 메트릭: Exact Match")
    
elif args.mode == 'research':
    # [연구 모드] 원래 메트릭을 base_metric에 담습니다.
    base_metric = hierarchical_feedback_metric
    print("[INFO] 기본 메트릭: Hierarchical Feedback Metric")

# ★ [핵심] 여기서 원래 메트릭에 '기록 기능(Wrapper)'을 씌웁니다!
# 이제 DSPy는 'selected_metric'을 호출할 때마다 자동으로 엑셀에 기록합니다.
selected_metric = MetricWrapper(base_metric, score_log_file)

# --- [4] 평가 도구 설정 ---
# display_progress=True :  터미널에 진행바 표시
evaluator = Evaluate(devset=devset, metric=selected_metric, num_threads=1, display_progress=True)

# --- [5] 최적화 전 평가 ---
print("\n========== [1. 최적화 전 (Zero-shot)] ==========")
uncompiled_rag = RAG_CoT()

# 연구 모드일 때는 터미널에 JSON 피드백이 주루룩 뜨는 게 정상
baseline_score = evaluator(uncompiled_rag)
print(f"[INFO] 점수: {baseline_score}")

# --- [6] 최적화 수행 (Hybrid Strategy: COPRO + FewShot) ---
print("\n========== [2. 최적화 수행 (Hybrid: Instruction + Few-Shot)] ==========")
# [6-1] 지시문(Instruction) 깎기 (by COPRO)
# 선택된 메트릭(selected_metric)을 사용하여 최적화를 진행한다. 
# 목적: "어떤 말투로 지시해야 내 보상 함수 점수가 잘 나올까?"를 고민한다.
print(f"[Step 1] 지시문(Instruction) 최적화 중... (COPRO)")

# breadth=5 : 5가지 다른 지시문을 써보고 제일 좋은 걸 고름
copro_optimizer = COPRO(
    metric=selected_metric,
    breadth=5, 
    track_to_use='large' # medium or large
)
# 기본 깡통 로봇(RAG_CoT)을 넣어서 -> 지시문이 개선된 로봇을 받음

# 의미: "COPRO야, 네가 만든 지시문을 테스트할 때 '4개 쓰레드'로 '진행상황' 보여주면서 채점해라."
eval_kwargs = dict(num_threads=4, display_progress=True)
instruction_optimized_rag = copro_optimizer.compile(
    RAG_CoT(), 
    trainset=trainset, 
    eval_kwargs=eval_kwargs 
)

print("[Step 1 완료] 더 나은 지시문을 찾았습니다!")

print("[Step 1 완료] 더 나은 지시문을 찾았습니다!")

# [6-2] 예시(Few-Shot) 박아넣기 (by BootstrapFewShot)

# max_bootstrapped_demos=4 : 메트릭에서 높은 점수를 받은 '모범 답안' 4개를 골라낸다.
""" 아래는 기본 체이닝 없이 그냥 Few-Shot 예시를 추가하는 방식
    teleprompter = BootstrapFewShot(metric=selected_metric, max_bootstrapped_demos=4)
    compiled_rag = teleprompter.compile(RAG_CoT(), trainset=trainset)
    즉 RAG_CoT 깡통 CoT 대신 instruction_optimized_rag 를 넣어야 한다.
"""
print(f"[Step 2] 예시(Few-Shot) 선별 중... (BootstrapFewShot)")
fewshot_optimizer = BootstrapFewShot(
    metric=selected_metric, 
    max_bootstrapped_demos=4
)

# 1단계 결과물(instruction_optimized_rag)을 넣어서 2단계를 진행
compiled_rag = fewshot_optimizer.compile(instruction_optimized_rag, trainset=trainset)

print("[Step 2 완료] 최종 최적화가 끝났습니다!")
"""
# comile() 이란?
    딥러닝 학습(가중치 조정, Weight Update) 이 아닌, In-Context Learning 이다. 
    LLM에게 보여줄 "컨닝 페이퍼(프롬프트)"를 최적화하는 과정이다.
    즉 내가 직접 State-action-reward 를 하드코딩 하던 것 대신, 
    DSPy 프레임워크를 사용하여 자동으로 최적의 프롬프트를 찾도록 한 것이다.
# compiled_rag 란?
    compile()가 반환하는 객체. 
    RAG_CoT 모듈에 "최적화된 프롬프트"가 삽입된 형태이다.
    따라서 compiled_rag(question, context) 처럼 호출하면, 
    최적화된 프롬프트로 LLM에게 질의하여 답변을 생성한다.
"""

# --- [7] 최적화 후 평가 ---
print("\n========== [3. 최적화 후 (Compiled)] ==========")
final_score = evaluator(compiled_rag)
print(f"[INFO] 점수: {final_score}")

print(f"\n[결론] {args.mode} 모드 성능 변화: {baseline_score} -> {final_score}")

# --- [7] 평가 후 프롬프트 확인해보기 ---
print("\n========== [4. 최적화된 프롬프트 뜯어보기] ==========")
# GPT가 실제로 받은 최종 프롬프트(시스템 프롬프트 + Few-Shot 예제들)를 출력합니다.
_ = compiled_rag(
    question="테스트 질문",
    context="테스트 컨텍스트"
)

lm.inspect_history(n=1)

# =========================================================
# --- [8] 최적화된 프롬프트(프로그램) 저장하기 --
# =========================================================
timestamp = datetime.now().strftime("%y%m%d-%H%M%S")
save_path = f"datasets/practice/optimized_rag_{dataset_name}_{args.mode}_{timestamp}.json"
compiled_rag.save(save_path) # 이 결과는 JSON은 “부품(시그니처/데모)”

print(f"\n[저장 완료] 최적화된 프롬프트가 '{save_path}' 파일로 저장되었습니다.")



# =========================================================
# 실험 과정 전체를 CSV로 저장하기
# =========================================================
print(f"\n[PROCESS] 전체 대화 내역(History)을 CSV로 변환 중...")

history_data = []

# lm.history에는 {messages: [...], response: ...} 형태의 딕셔너리가 리스트로 쌓여있습니다.
for idx, item in enumerate(lm.history):
    
    # 1. 입력 데이터 (Prompt) 추출
    system_instruction = ""
    user_full_prompt = "" # 여기에 [데모 + 컨텍스트 + 질문]이 다 들어있습니다.
    
    for msg in item['messages']:
        role = msg['role']
        content = msg['content']
        
        if role == 'system':
            system_instruction = content
        elif role == 'user':
            user_full_prompt += content

    # 2. 출력 데이터 (Answer) 추출
    try:
        # LLM이 뱉은 답변 (Reasoning + Answer)
        model_output = item['response']['choices'][0]['message']['content']
    except Exception:
        model_output = "Error or Empty Response"

    # 3. 데이터 구조화
    history_data.append({
        "Turn": idx + 1,
        "Type": "Optimization_Log", # 나중에 구분하기 위해
        "System_Instruction": system_instruction, # 지시문 (COPRO가 깎은 것)
        "User_Input_Full": user_full_prompt,      # 입력 전체 (여기에 Context, Question이 포함됨)
        "Model_Output": model_output              # AI의 답변
    })

# 4. 판다스로 변환 및 CSV 저장
df_log = pd.DataFrame(history_data)

# 파일명 생성 (시간 포함)
csv_filename = f"datasets/practice/log_{dataset_name}_{args.mode}_{timestamp}.csv"

# 한글 깨짐 방지를 위해 encoding='utf-8-sig' 사용
df_log.to_csv(csv_filename, index=False, encoding='utf-8-sig')

print(f"[로그 저장] 실험 내역이 '{csv_filename}' 파일로 저장되었습니다.")
print("엑셀에서 열어서 [User_Input_Full] 열을 보시면 질문과 컨텍스트를 확인하실 수 있습니다!")