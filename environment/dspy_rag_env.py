"""
@경로: environment/dspy_rag_env.py
@설명: 
- 강화학습의 '환경(Environment)' 역할을 수행하는 클래스입니다.
- Agent로부터 'Action(새 지시문)'을 받아서 Model(Student)에 적용합니다.
- Model을 통해 시험(Dataset)을 치르고, 채점(Metric)을 수행합니다.
- 그 결과인 'Reward(점수)'와 'State(피드백 로그)'를 Agent에게 반환합니다.
"""

import dspy
import copy

# ------------------------------------------------------------------
# [Helper Functions] DSPy 모듈의 지시문을 안전하게 교체하는 도구들
# ------------------------------------------------------------------
def get_signature(predictor):
    """Predictor 객체 내부 깊숙이 숨어있는 Signature를 찾아냅니다."""
    if hasattr(predictor, "extended_signature"): return predictor.extended_signature
    if hasattr(predictor, "signature"): return predictor.signature
    if hasattr(predictor, "predictor"): return get_signature(predictor.predictor)
    return None

def update_instruction(student_module, new_instruction):
    """
    [Action Apply]
    Agent가 만든 새 지시문을 Student 모델의 뇌(Signature)에 이식합니다.
    """
    # 모듈 내부의 모든 Predictor를 순회하며 지시문 업데이트
    # (보통 RAG_CoT에는 1개의 Predictor가 있지만, 확장성을 위해 루프 사용)
    for predictor in student_module.predictors():
        target_sig = get_signature(predictor)
        
        # 원본을 훼손하지 않기 위해 extended_signature가 없으면 복제 생성
        if not hasattr(predictor, "extended_signature"):
            try:
                # 시그니처 복제 시도 (일부 DSPy 버전 호환성)
                predictor.extended_signature = copy.deepcopy(target_sig)
            except:
                predictor.extended_signature = target_sig
        
        # 지시문 덮어쓰기 (수술 집도)
        if hasattr(predictor, "extended_signature"):
            predictor.extended_signature.instructions = new_instruction

def get_current_instruction_text(student_module):
    """현재 모델에 적용되어 있는 지시문을 문자열로 가져옵니다."""
    try:
        predictor = student_module.predictors()[0]
        sig = get_signature(predictor)
        return sig.instructions
    except:
        return "No instruction found."

# ------------------------------------------------------------------
# [Environment Class] 실제 강화학습 루프가 돌아가는 경기장
# ------------------------------------------------------------------

class DSPyRAGEnv:
    def __init__(self, student_module, trainset, metric_fn):
        """
        @param student_module: 최적화 대상 모델 (RAG_CoT) -> '모델(Student)'
        @param trainset: 학습용 문제집 -> '데이터셋'
        @param metric_fn: 채점 및 피드백 생성 함수 -> '리워드 함수'
        """
        self.student = student_module 
        self.trainset = trainset      
        self.metric = metric_fn       
        
        # 초기 상태 기록
        self.last_score = 0.0

    def reset(self):
        """
        [Reset]
        환경을 초기화하고, 아무것도 하지 않은 상태의 State를 반환합니다.
        보통 Main Loop의 첫 번째 단계에서 호출됩니다.
        """
        return {
            "current_instruction": get_current_instruction_text(self.student),
            "current_similarity_score": 0.0,
            "verbal_feedback": "Initial State. No evaluation performed yet.",
            "fail_case_feedback": "" # 빈 문자열이면 Agent가 첫 턴에 수정을 안 할 수 있으므로, main에서 step(None)으로 초기 평가를 돌리는 것을 권장
        }

    def step(self, action):
        """
        [Step]
        1. Agent의 Action(새 지시문)을 받아서 모델에 적용합니다.
        2. 모델이 문제를 풉니다 (Evaluation).
        3. 채점하고 언어적 피드백을 수집합니다.
        4. Next State, Reward, Done 여부를 반환합니다.
        
        @param action: Agent가 생성한 new_instruction (String)
        """
        
        # 1. [Action Apply] 지시문 업데이트
        # action이 None이면 업데이트 없이 현재 상태로 평가만 수행 (초기 평가용)
        if action:
            update_instruction(self.student, action)
            # 디버깅용 출력
            # print(f"[Env] Instruction Updated: {action[:50]}...")

        # 2. [Simulation] 평가 및 피드백 수집
        total_score = 0
        failed_examples = []
        analyses = []
        
        # 데이터셋 순회 (Batch Evaluation)
        for example in self.trainset:
            # 예측 생성
            pred = self.student(question=example.question, context=example.context)
            
            # 채점 (Metric 함수 내부에서 pred.feedback_log에 피드백을 심어줌)
            score = self.metric(example, pred)
            total_score += score
            
            # [State Construction] 실패 사례 및 Verbal Feedback 수집
            if score < 1.0:
                # Reward 함수가 심어놓은 로그 추출
                feedback_data = getattr(pred, "feedback_log", {})
                
                # 1) 전체적인 분석 멘트 수집 (Analysis)
                analysis_text = feedback_data.get("Analysis", "")
                if analysis_text and analysis_text not in analyses:
                    analyses.append(analysis_text)
                
                # 2) 구체적인 실패 원인 (Critical Review)
                crit_review = feedback_data.get("ScoreCard", {}).get("critical_review", "")
                format_issue = feedback_data.get("ScoreCard", {}).get("format", "")
                
                # 실패 로그 포맷팅 (Agent가 읽기 좋게)
                log_entry = (
                    f"Question: {example.question}\n"
                    f"Model Answer: {pred.answer}\n"
                    f"Gold Answer: {example.answer}\n"
                    f"Issue: {crit_review} (Format: {format_issue})\n"
                    f"Score: {score:.2f}\n"
                    f"--------------------------------"
                )
                failed_examples.append(log_entry)

        # 3. [Result Aggregation]
        avg_score = total_score / len(self.trainset)
        self.last_score = avg_score
        
        # Verbal Feedback 요약 (너무 길면 앞부분만 사용하거나 연결)
        # 예: "헌법 위반 감지. 스타일 불일치." 등등
        final_verbal_feedback = " ".join(analyses[:3]) if analyses else "No specific verbal feedback."

        # 실패 로그 합치기 (Agent에게 보여줄 오답노트)
        # 너무 많으면 토큰 터지니까 3~5개만 자름
        final_fail_log = "\n".join(failed_examples[:5])

        # 4. [Next State Construction] Agent의 Input 스펙에 맞춤
        next_state = {
            "current_instruction": get_current_instruction_text(self.student), # 시도했던(Attempted) 프롬프트
            "current_similarity_score": avg_score,                             # 점수
            "verbal_feedback": final_verbal_feedback,                          # 전체 총평
            "fail_case_feedback": final_fail_log                               # 오답 노트
        }

        # 5. [Done Condition]
        # 점수가 1.0(만점)이거나, 더 이상 개선이 불가능하다고 판단될 때 (여기선 만점 기준)
        done = (avg_score == 1.0)

        return next_state, avg_score, done