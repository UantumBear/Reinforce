"""
@경로: optimizers/verbal_reinforce_optimizer.py
@설명: 강화학습 기반 프롬프트 최적화용 Verbal Reinforce Optimizer 클래스 구현
- DSPy의 Teleprompter를 상속하여 구현
- Agent가 생성한 지시문을 모델에 적용하고 평가를 반복

- 26.02.07 main_trian.py의 평가 루프와 dspy_rag_env.py의 step() 메서드 로직을 통합하여 작성

"""


import dspy
from dspy.teleprompt import Teleprompter
import copy
import time
from datetime import datetime
import pandas as pd 
from pathlib import Path
from utils.log.console import print_step
from db.models.rl_optimization_log import RlOptimizationLog
from db.connection.pg_client import pg_client
from conf.config import Settings

class VerbalReinforceOptimizer(Teleprompter):
    def __init__(self, metric, agent, max_episodes=5, log_dir="datasets/results", log_callback=None, experiment_id=None):
        """
        @param metric: 평가 함수 (wrapped_metric)
        @param agent: 프롬프트를 수정하는 Agent (PromptOptimizerAgent)
        @param max_episodes: 최대 최적화 시도 횟수
        @param log_callback: 평가 결과를 저장/출력할 외부 콜백 함수 (선택)
        """

        self.metric = metric
        self.agent = agent
        self.max_episodes = max_episodes
        self.log_dir = log_dir
        self.log_callback = log_callback
        
        # 최적화 과정의 히스토리를 저장할 리스트
        self.history = []
        self.experiment_id = experiment_id or f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    def compile(self, student, trainset):
        """
        [핵심] DSPy의 표준 학습 메서드이다.
        여기서 'Loop'와 'Environment' 역할을 모두 수행한다.

        @param student: 테스트할 DSPy 모듈 (RAG_CoT)
        @param trainset: 평가에 사용할 데이터셋 (List of Examples)
        """
        print_step("[VerbalReinforceOptimizer] compile() 시작 ... ")
        try:
            print_step("[Optimizer] Clean Model 복제 및 초기화")
            # (학생 모델=최적화된 프롬프트를 받아서 테스트하기 위해 사용하는 모델) 
            current_student = student.deepcopy()
            best_student = student.deepcopy()
            best_score = 0.0
            
            print_step("[Episode 0] Baseline 평가")
            # 초기 지시문 가져오기
            current_instruction = self._get_instruction(current_student)  
            print(f"[DEBUG] Initial instruction: {current_instruction[:200]}...")
            state = {
                "current_instruction": current_instruction,
                "current_similarity_score": 0.0,
                "total_score": 0.0,
                "verbal_feedback": "[N/A] Initial State: 초기 상태이므로 Verbal Feedback 없음.",
                "fail_case_feedback": "[N/A] Initial State: 초기 상태이므로 fail case Feedback 없음."
            }


            print_step("[Optimization Loop] 0번(Baseline) ~ Max번(Optimization) 통합 루프 START ... ")
            target_success_count = 1 + self.max_episodes 
            successful_episodes = 0
            # (부연설명)
            # 0회차는 baseline 실험으로 최적화 없이 깨끗한 LLM 상태에서 평가를 수행하는 것이다.
            # 1~5회차 까지 최적화를 진행한다.
            # 즉 총 6번의 '깨끗한LLM' 호출이 발생한다. dataset 크기 n에 대해 (1 + 5) * n 번의 LLM 호출이 발생한다.
            # Optimizer LLM은 반면, dataset 을 모은 전체 평가에 대한 피드백을 들고 한 번만 호출된다.
            # 즉 Optimizer LLM은 최대 5번 호출된다.
            
        
            while successful_episodes < target_success_count:
                # 성공 횟수가 목표치에 도달할 때까지 무한 반복 --> TODO 이부분은 무한 반복 대신 뭔가 다른거 생각해보기. 
                # Azure API 자체 필터링으로 인해 에피소드가 중간에 실패할 수 있으므로, 일정 횟수를 보장하기 위함.

                # 햔재 에피소드 번호 (0부터 시작)
                ep_num = successful_episodes
                episode_start_time = datetime.now() # 에피소드 시작 시간 기록

                print(f"\n>>> [Episode {ep_num}] Optimization Step (Attempting...)")
                try:
                    if ep_num == 0:
                        # [Case 0] Baseline: 아무것도 안 하고 기존 지시문 그대로 사용
                        print("    [Info] Baseline Evaluation (Skip Agent)")
                        instruction_to_test = current_instruction
                        print(f"    [DEBUG] Episode {ep_num} - Using baseline instruction: {instruction_to_test[:100]}...")
                    else: 
                        print_step("ACTION!")
                        # [Action] 현재 상황(State)을 보고, 새로운 작전(Instruction)을 짠다.
                        # 
                        # state에는 이전 턴의 평가 결과(유사도 점수 + 각종 Judges 판단 결과 + 언어 피드백)가 들어있다.
                        # Agent는 State(오답노트, 점수 등)를 보고 Action(새 지시문)을 결정한다.
                        print(f"    [DEBUG] Previous instruction: {current_instruction[:100]}...")
                        print(f"    [DEBUG] State for agent: {state}")
                        # 즉, 여기서 OptimizerLLM 은 새로운 프롬프트를 생성한다!
                        new_instruction = self.agent.act(state)
                        print(f"    [DEBUG] Agent generated new instruction: {new_instruction[:100] if new_instruction else 'None'}...")
                    
                        # Stop Condition
                        if new_instruction is None: 
                            # 알 수 없는 사유에 의해 new_instruction 이 생성되지 않았거나,
                            # 안전 필터(Safety Filter) 등으로 인해 생성이 실패한 경우 등 ... 
                            # TODO 이 부분은 실험을 해보며 보완해야 할 것 같다. 
                            print("    [Stop] Agent decided to stop optimization.")
                            break
                    
                        # CleanLLM 에게 새 지시문을 전달한다. 
                        print(f"    [DEBUG] Updating instruction from: {self._get_instruction(current_student)[:100]}...")
                        self._update_instruction(current_student, new_instruction)
                        print(f"    [DEBUG] Updated instruction to: {self._get_instruction(current_student)[:100]}...")
                        # DSPy의 current_student 객체는 모델 그 자체이다. 이 안에 프롬프트가 들어있다. 
                        instruction_to_test = new_instruction
                        # current_instruction도 업데이트 (다음 에피소드 디버깅용)
                        current_instruction = new_instruction
                        print(f"    [DEBUG] Episode {ep_num} - Using new instruction: {instruction_to_test[:100]}...")
                    
                    print_step("EVALUATE!")
                    # [Reward + State -> Evaluate -> Reward + Next State] 새 지시문을 가지고 다시 평가를 수행한다.
                    # 여기서 각종 Judges들이 동원되어 평가가 수행되어 점수가 계산되고,
                    # 언어 피드백(next_state) 도 함께 반환된다. 
                    score, next_state = self._evaluate(
                        current_student, 
                        trainset, 
                        episode=ep_num, 
                        instruction=instruction_to_test,
                        ep_start_time=episode_start_time,
                        agent=self.agent  # OptimizerLLM 정보 추가
                    )
                    
                    print(f"    [Result] Score: {score:.2f}")

                    # --- [성공 시 처리] ---
                    # 4. 상태 업데이트
                    print_step("STATE UPDATE!")
                    # 이번 턴의 결과(next_state)가 다음 턴의 입력(state)이 된다.
                    state = next_state
                    successful_episodes += 1
                    
                    # Best Score 갱신 시 Model 저장
                    # best_score 는 각 데이터셋이 아닌, Episode 단위의 평균 점수를 말한다.
                    if score > best_score:
                        best_score = score
                        best_student = current_student.deepcopy()
                        print(f"    [Best] New Best Model Found! (Score: {best_score:.2f})")
                    
                    if score >= 1.0:
                        print("    [Info] Perfect Score Achieved.")
                        break

                except Exception as e:
                    print(f"\n[Error] Optimization Loop Error: {e}")
                    # 에러 발생 시 Agent에게 피드백을 주고 재시도하는 로직을 추가할 수 있습니다.
                    # 여기서는 간단히 로그만 남기고 넘어갑니다.
                    # 로그 기록
                    inst_log = instruction_to_test if 'instruction_to_test' in locals() else "Unknown"
                    self._log_failure(ep_num, inst_log, str(e), ep_start_time=episode_start_time)
                    
                    # Agent에게 에러 피드백 주입 (다음 턴에 반영하도록)
                    # 구조화된 피드백으로 에러 정보 전달
                    state["verbal_feedback"] = f"""
[System] RAG 시스템 프롬프트 최적화 - Episode {ep_num} 에러 발생

[Error Information]
- 에러 발생 Episode: {ep_num}
- 에러 메시지: {str(e)}
- 실패한 프롬프트: {inst_log[:200] if 'inst_log' in locals() else 'Unknown'}...

[Action Required]
위 에러를 피할 수 있도록 안전한 지시문을 작성하여 다시 시도하세요.
"""
                    # 실패 케이스 요약도 함께 업데이트
                    state["fail_case_feedback"] = f"[SYSTEM ERROR] Episode {ep_num}: {str(e)}"
                    
                    time.sleep(2)
                    continue # successful_episodes 증가 안 하고 다시 루프


            print(f"\n[Done] Optimization Finished. Best Score: {best_score:.2f}")
            
            # 전체 최적화 완료 후 한 번만 저장
            print_step("[Final] 최종 결과 저장 시작")
            self.save_logs()
            
            return best_student
        except Exception as e:
            print(f"[Fatal Error] VerbalReinforceOptimizer compile 중 Error: {e}")
            # 치명적 에러 발생 시에도 수집된 데이터는 저장
            if self.history:
                print_step("[Emergency] 에러 발생으로 인한 긴급 데이터 저장")
                self.save_logs()
            return student  # 실패 시 원본 학생 모델 반환

    def _evaluate(self, student, trainset, episode, instruction, ep_start_time, agent=None):
        """
        평가 수행 및 로그 기록 (self.history에 저장)
        """
        scores = []
        feedback_logs = []

        # [NEW] 메타데이터 준비 (설정 파일에서 가져오기)
        opt_model = Settings.OPTIMIZER_MODEL
        test_model = Settings.TESTER_MODEL
        
        # OptimizerLLM 정보 추출
        optimizer_system_prompt = agent.last_system_prompt if agent else None
        optimizer_total_input = agent.last_total_input if agent else None
        
        # 데이터셋 순회하며 평가
        for example in trainset:
            # 예측 수행
            pred = student(question=example.question, context=example.context)
            
            # 채점 (metric 함수 내부에서 로깅 등 부가 작업 수행 가능)
            score = self.metric(example, pred)
            scores.append(score)
            
            # Agent에게 줄 피드백 데이터 수집 
            # (hierarchical_feedback_metric은 pred.feedback_log에 정보를 담아둡니다)
            log_data = getattr(pred, 'feedback_log', {})
            feedback_logs.append(log_data)

            # 데이터 추출 (안전하게 get 사용)
            score_card = log_data.get("ScoreCard", {})
            # answer_sheet = log_data.get("AnswerSheet", {}) # == example.answer 와 동일
            
            # Context가 리스트일 수도, 문자열일 수도 있음
            context_raw = getattr(example, 'context', "")
            context_str = " ".join(context_raw) if isinstance(context_raw, list) else str(context_raw)

            # 각 Episode 내에서도 Test(dataset) 별로 기록되는 로그
            row = {
                "Episode": episode,
                "Instruction": instruction, 
                "Question": example.question,
                "Context": context_str,
                "Model_Answer": pred.answer,
                "Gold_Answer": example.answer, 
                
                # 점수 및 평가 상세
                "Total_Score": score, # 해당 1 Episode, 1 Dataset Test 에 대한 점수
                "Raw_Similarity": score_card.get("raw_similarity", 0.0),
                "Is_Faithful": score_card.get("faithfulness", "Unknown"),
                "Is_Style_Match": score_card.get("format", "Unknown"),
                "Constitution_Status": score_card.get("constitution", "Pass"),
                "Constitution_Violation_Reason": score_card.get("critical_review", "") if score_card.get("constitution") != "Pass" else "",
                "Critical_Review": score_card.get("critical_review", "None"),
                "Full_Analysis": log_data.get("Analysis", ""),
                
                # RAGAS 평가 점수들
                "Ragas_Faithfulness_Score": score_card.get("ragas_faithfulness_score", None),
                "Ragas_Answer_Relevancy_Score": score_card.get("ragas_answer_relevancy_score", None),
                "Ragas_Context_Precision_Score": score_card.get("ragas_context_precision_score", None),
                "Ragas_Context_Recall_Score": score_card.get("ragas_context_recall_score", None),
                # "Ragas_Is_Faithful": score_card.get("ragas_is_faithful", None),
                # "Ragas_Is_Relevant": score_card.get("ragas_is_relevant", None),
                
                # 시스템 메타데이터
                "Is_Success": True,
                "Error_Log": "",
                "Episode_Start_Time": ep_start_time, # TODO 아직 DB에 컬럼을 만들지 않음, created_at 에 적재 중.
                # "Episode_End_Time": ep_end_time, # TODO 아직 DB에 컬럼을 만들지 않음
                
                "Avg_Total_Score": -1, # TODO 나중에 n dataset 전체 ( one Episode) 에 대한 평균 점수로 채워넣을 예정
                # 모델 정보 (설정에서 가져옴)
                "Optimizer_Model_Name": opt_model,
                "Optimizer_Model_Provider": "azure",
                "Tester_Model_Name": test_model,
                "Tester_Model_Provider": "azure",
                
                # OptimizerLLM 관련 정보
                "Optimizer_System_Prompt": optimizer_system_prompt,
                "Optimizer_Total_Input": optimizer_total_input
            }
            
            # 리스트에 추가 (이게 없어서 저장이 안 되었던 겁니다!)
            self.history.append(row)

        avg_score = sum(scores) / len(scores) if scores else 0.0
        
        # 다음 턴을 위한 State 구성 (MDP-v3 스타일 구조화된 피드백)
        structured_feedback = self._aggregate_feedback(feedback_logs, episode, instruction, avg_score)
        fail_cases_only = self._extract_fail_cases_from_feedback(feedback_logs, instruction, avg_score) # 실패 케이스만 추출 (프롬프트와 평균 점수 포함)
        
        next_state = {
            "current_instruction": instruction,
            "current_similarity_score": avg_score, # 해당 에피소드에서의 평균 유사도 점수
            "total_score": avg_score,  # total_score도 추가
            # 구조화된 피드백 (History + Feedback + Action 통합)
            "verbal_feedback": structured_feedback,
            # 기존 agent/optimizer_agent.py가 사용하는 fail_case_feedback (실패 케이스만)
            "fail_case_feedback": fail_cases_only
        }
        
        return avg_score, next_state

    # --- Helper Methods (기존 dspy_rag_env.py에서 가져옴) ---

    def _get_instruction(self, student_module):
        # DSPy 모듈 구조에 따라 지시문을 추출하는 로직
        try:
            # 보통 첫 번째 Predictor의 서명을 가져옴
            predictor = student_module.predictors()[0]

            print(f"predictor: {predictor}")

            if hasattr(predictor, "extended_signature"):
                return predictor.extended_signature.instructions
            return predictor.signature.instructions
        
            
        except:
            return "No instruction found."

    def _update_instruction(self, student_module, new_instruction):
        # DSPy 모듈의 지시문을 업데이트하는 로직
        for predictor in student_module.predictors():
            # 기존 Signature를 복사해서 extended_signature로 만듦 (안전장치)
            if not hasattr(predictor, "extended_signature"):
                target_sig = predictor.signature
                predictor.extended_signature = target_sig
            
            # 지시문 교체
            predictor.extended_signature.instructions = new_instruction

    def _aggregate_feedback(self, logs, episode, instruction, avg_score):
        """
        MDP-v3 스타일의 구조화된 피드백 생성
        [History] + [Feedback] + [Action] 섹션으로 체계적으로 구성
        """
        
        # 1. 현재 에피소드 평가 정보 수집
        evaluation_summary = []
        failed_cases = []
        
        for log in logs:
            score_card = log.get("ScoreCard", {})
            analysis = log.get("Analysis", "")
            answer_sheet = log.get("AnswerSheet", {})
            
            # 평가 요약 수집
            if analysis:
                evaluation_summary.append(analysis)
            
            # 실패 사례 수집 (점수 낮은 것들)
            score = score_card.get("final_total_score", 0.0)
            if score < 1.0:
                question = answer_sheet.get("current_question", "")
                gold_answer = answer_sheet.get("reference_answer", "")
                critical_review = score_card.get("critical_review", "")
                
                failed_cases.append({
                    "question": question[:200],
                    "gold_answer": gold_answer[:200], 
                    "issue": critical_review[:300],
                    "score": score
                })
        
        # 2. 구조화된 피드백 생성
        structured_feedback = f"""
[System] RAG 시스템 프롬프트 최적화 - Episode {episode} 평가 결과

[Current Performance]
- 현재 Episode: {episode}
- CleanLLM이 사용한 시스템 프롬프트:
"{instruction}" 
- 현재 Episode 평균 성능 점수: {avg_score:.4f}

[Detailed Evaluation]
"""
        
        # 평가 상세 요약 추가 (최대 2개)
        if evaluation_summary:
            for i, summary in enumerate(evaluation_summary[:2], 1):
                structured_feedback += f"• 평가 {i}: {summary[:400]}\n"
        
        structured_feedback += "\n[Failed Cases Analysis]"
        
        # 실패 사례 분석 (최악 1-2개)
        if failed_cases:
            # 점수가 가장 낮은 사례들 선택
            worst_cases = sorted(failed_cases, key=lambda x: x['score'])[:2]
            
            for i, case in enumerate(worst_cases, 1):
                structured_feedback += f"""
실패 사례 {i}:
- 질문: {case['question']}
- 모범답안: {case['gold_answer']}
- 문제점: {case['issue']}
- 점수: {case['score']:.3f}
"""
        else:
            structured_feedback += "\n• 모든 테스트 케이스 통과 (성공)"
            
        structured_feedback += f"""

[Improvement Direction]
위 평가 결과를 바탕으로 다음 에피소드에서는:
1. 실패한 사례들의 공통 문제점을 해결하도록 프롬프트를 개선하세요
2. 평균 점수 {avg_score:.4f}에서 더 높은 성능을 목표로 하세요
3. 성공한 요소들은 유지하면서 문제 영역을 보완하세요
"""
        
        return structured_feedback.strip()
    
    def _extract_fail_cases_from_feedback(self, logs, instruction, avg_score):
        """
        [사용자 의도 반영] 오답노트용 실패 케이스 추출:
        1 에피소드에서 테스트한 모든 샘플 중 가장 낮은 점수를 받은 1개 케이스만 반환
        해당 케이스의 프롬프트, 질문, CleanLLM 답변, 점수를 오답노트로 활용
        
        Args:
            logs: 평가 결과 로그들
            instruction: 현재 에피소드에서 사용된 프롬프트 
            avg_score: 해당 에피소드의 평균 점수
        """
        if not logs:
            return "[Initial State] No evaluation data available."
            
        # 가장 낮은 점수의 케이스 찾기
        worst_case = None
        lowest_score = float('inf')
        
        for log in logs:
            score_card = log.get("ScoreCard", {})
            score = score_card.get("final_total_score", 0.0)
            
            # 가장 낮은 점수 업데이트
            if score < lowest_score:
                lowest_score = score
                worst_case = log
        
        # 실패 케이스가 없으면 (모든 케이스가 완벽한 경우)
        if worst_case is None:
            return "[Perfect Performance] All cases achieved perfect scores."
            
        # 가장 낮은 점수 케이스의 정보 추출
        answer_sheet = worst_case.get("AnswerSheet", {})
        score_card = worst_case.get("ScoreCard", {})
        
        question = answer_sheet.get("current_question", "")[:200]
        gold_answer = answer_sheet.get("reference_answer", "")[:200]
        model_answer = answer_sheet.get("model_answer", "")[:200]  # CleanLLM 답변
        critical_review = score_card.get("critical_review", "")[:300]
        
        # 오답노트 형태로 포맷팅 (프롬프트, 평균 점수, 가장 성능이 떨어진 1개 케이스)
        fail_case_note = (
            f"=== EPISODE PERFORMANCE ANALYSIS ===\n"
            f"Average Score: {avg_score:.3f}\n"
            f"Used Prompt: {instruction[:300]}{'...' if len(instruction) > 300 else ''}\n\n"
            f"=== WORST CASE (Score: {lowest_score:.3f}) ===\n"
            f"Question: {question}\n\n"
            f"Gold Answer: {gold_answer}\n\n"
            f"Model Answer: {model_answer}\n\n"
            f"Issue Analysis: {critical_review}\n"
            f"================================"
        )
        
        return fail_case_note

    def _get_fail_cases(self, logs):
        """
        [Original Logic from dspy_rag_env.py]
        점수가 1.0(만점) 미만인 케이스들을 모아서
        Agent가 참고할 수 있는 '오답 노트(Fail Case Feedback)'를 생성합니다.
        --> 변경
        TODO 한 Episode 내에서 시도했던 dataset 중, 가장 점수가 낮은 경우의 오답 노트를 생성합니다. 
        """
        failed_examples = []
        
        for log in logs:
            # 1. 점수 확인 (ScoreCard 내부의 final_total_score 확인)
            score_card = log.get("ScoreCard", {})
            score = score_card.get("final_total_score", 0.0)
            
            # 2. 만점이 아닌 경우(실패 사례)만 오답노트에 추가
            if score < 1.0:
                # 데이터 추출 (AnswerSheet 및 ScoreCard 활용)
                answer_sheet = log.get("AnswerSheet", {})
                question = answer_sheet.get("current_question", "")
                gold_answer = answer_sheet.get("reference_answer", "")
                
                # 비평(Critical Review) 및 포맷 이슈 추출
                crit_review = score_card.get("critical_review", "")
                format_issue = score_card.get("format", "")
                
                # 3. 로그 포맷팅 (Agent가 읽기 편한 텍스트로 변환)
                # (주의: Model Answer는 로그에 포함되어 있지 않을 수 있어 제외하거나 필요한 경우 추가 로직 필요)
                log_entry = (
                    f"Question: {question}\n"
                    f"Gold Answer: {gold_answer}\n"
                    f"Issue: {crit_review} (Format: {format_issue})\n"
                    f"Score: {score:.2f}\n"
                    f"--------------------------------"
                )
                failed_examples.append(log_entry)
        
        # 실패 사례가 없으면 빈 문자열 반환 (Agent가 Stop 할 수 있게)
        if not failed_examples:
            return ""

        # 토큰 제한을 위해 최대 5개까지만 합쳐서 반환
        return "\n".join(failed_examples[:5])
    

    def save_logs(self):
        """CSV 및 DB 저장 (기존 save_results_to_storage 역할)"""
        if not self.history:
            print("[Warning] 저장할 데이터가 없습니다.")
            return

        df = pd.DataFrame(self.history) # # DataFrame 으로 변환
        
        print_step("[Log] CSV 저장 시작 ... ")
        save_path = Path(self.log_dir)
        save_path.mkdir(parents=True, exist_ok=True) # 폴더가 없으면 생성
        filename = f"optimization_log_{self.experiment_id}.csv"
        df.to_csv(save_path / filename, index=False, encoding='utf-8-sig')
        print(f"\n[Log] CSV 저장 완료 ... ! {filename}")

        print_step("[Log] DB 저장 시작 ... ")
        try:
            session = pg_client.get_session() # PgClient 사용 (이미 모든 DB 설정이 구현되어 있음)
            db_records = []
            
            # [최적화 1] 에피소드별 평균 점수 미리 계산 (GroupBy)
            ep_avg_scores = df[df['Is_Success']==True].groupby('Episode')['Total_Score'].mean().to_dict()

            # [최적화 2] 에피소드별 데이터 개수(=Dataset Size) 미리 계산 (GroupBy)  <-- [NEW]
            ep_sizes = df.groupby('Episode').size().to_dict()
            
            for _, row in df.iterrows():
                episode_num = row['Episode']

                record = RlOptimizationLog(
                    experiment_id=self.experiment_id,
                    episode=row['Episode'],
                    instruction=row['Instruction'],
                    question=row['Question'],
                    context=row['Context'],
                    model_answer=row['Model_Answer'],
                    gold_answer=row['Gold_Answer'],
                    total_score=row['Total_Score'],  # 개별 데이터 점수
                    raw_similarity=row['Raw_Similarity'],
                    dataset_size=int(ep_sizes.get(episode_num, 0)),  # 실험에 사용된 데이터 개수
                    avg_total_score=ep_avg_scores.get(episode_num, 0.0),  # 해당 에피소드의 평균 점수
                    optimizer_model_nm=row['Optimizer_Model_Name'],  # 최적화 모델명
                    optimizer_model_provider=row['Optimizer_Model_Provider'],  # 최적화 모델 제공사
                    tester_model_nm=row['Tester_Model_Name'],  # 테스팅 모델명
                    tester_model_provider=row['Tester_Model_Provider'],  # 테스팅 모델 제공사
                    is_faithful=row['Is_Faithful'],
                    is_style_match=row['Is_Style_Match'],
                    constitution_status=row['Constitution_Status'],
                    constitution_violation_reason=row['Constitution_Violation_Reason'],
                    critical_review=row['Critical_Review'],
                    full_analysis=row['Full_Analysis'],
                    # RAGAS 평가 점수들
                    ragas_faithfulness_score=row.get('Ragas_Faithfulness_Score'),
                    ragas_answer_relevancy_score=row.get('Ragas_Answer_Relevancy_Score'),
                    ragas_context_precision_score=row.get('Ragas_Context_Precision_Score'),
                    ragas_context_recall_score=row.get('Ragas_Context_Recall_Score'),
                    # ragas_is_faithful=row.get('Ragas_Is_Faithful'),
                    # ragas_is_relevant=row.get('Ragas_Is_Relevant'),
                    # OptimizerLLM 관련 정보
                    optimizer_system_prompt=row.get('Optimizer_System_Prompt'),
                    optimizer_total_input=row.get('Optimizer_Total_Input'),
                    is_success=row['Is_Success'],
                    error_log=row['Error_Log'],
                    created_at=row['Episode_Start_Time'] if pd.notnull(row['Episode_Start_Time']) else datetime.now()  # 실제 에피소드 시작 시간 사용
                )
                db_records.append(record)
            
            session.add_all(db_records)
            session.commit()
            print(f"[Log] DB Saved: {len(db_records)} records.")
            print(f"Experiment ID: {self.experiment_id}")
            session.close()

            print_step("[Log] DB 저장 완료 ... !")
        except Exception as e:
            print(f"[Warning] DB Save Failed: {e}")

        # 엑셀로도 저장하고 싶으면 아래 주석 해제
        # excel_filename = f"optimization_log_{self.experiment_id}.xlsx"
        # df.to_excel(save_path / excel_filename, index=False)

    def _log_failure(self, episode, instruction, error_msg, ep_start_time=datetime.now()):
        """
        실패 시 로그를 기록하는 헬퍼 함수
        (원본 main_train.py의 fail_row 생성 로직을 여기로 이동)
        """
        fail_row = {
            "Episode": episode,                 # ep_num
            "Instruction": instruction,         # 실패한 원흉 프롬프트
            "Question": "",    
            "Context": "",
            "Model_Answer": "",
            "Gold_Answer": "",
            "Total_Score": 0.0,                 # 0점 처리 ... 근데 이거 0점 처리 하는게 맞나? 실험에서 빼야하나? 생각해보기
            "Raw_Similarity": 0.0,
            # 실패 원인 명시
            # [수정] 의미 없는 평가 지표는 'Fail' 대신 'Error'나 'N/A'로 처리
            "Is_Faithful": "Error",      # 답변이 없으니 신뢰도 평가 불가 -> Error
            "Is_Style_Match": "Error",   # 답변이 없으니 스타일 평가 불가 -> Error
            "Constitution_Status": "Error",
            "Constitution_Violation_Reason": "Error",
            "Critical_Review": "Azure Content Policy Violation",
            "Full_Analysis": f"[System Exception] {error_msg}", # 실험 자체가 실패했음을 명시
            
            # RAGAS 평가 점수들 (실패 시 기본값)
            "Ragas_Faithfulness_Score": None,
            "Ragas_Answer_Relevancy_Score": None, 
            "Ragas_Context_Precision_Score": None,
            "Ragas_Context_Recall_Score": None,
            # "Ragas_Is_Faithful": None,
            # "Ragas_Is_Relevant": None,
            
            # 시스템 로그
            "Is_Success": False,
            "Error_Log": error_msg,
            "Episode_Start_Time": ep_start_time,
            "Optimizer_Model_Name": Settings.OPTIMIZER_MODEL,      # 최적화 모델명
            "Optimizer_Model_Provider": "azure",  # 최적화 모델 제공사
            # "Experiment_ID": self.experiment_id
            "Tester_Model_Name": Settings.TESTER_MODEL,      # 테스팅 모델명
            "Tester_Model_Provider": "azure"  # 테스팅 모델 제공사
            
            # 모델 정보 등은 self.config 등이 있다면 추가 가능
        }


        
        self.history.append(fail_row)
        print("     실패 이력을 로그에 기록했습니다.")

        # TODO 아래 부분은 기존 소스에 남아있던 부분인데 필요할지 안할지 고민해보기
        # 2. Agent에게 실패 원인 주입 (학습용)
        # state["fail_case_feedback"] = (
        #     f"[SYSTEM ERROR] Azure Content Filter(안전 정책)에 의해 프롬프트가 차단되었습니다.\n"
        #     f"상세 에러: {str(e)}\n"
        #     f"Action Required: 위 에러를 피할 수 있도록 지시문을 수정하여 다시 작성하세요."
        # )
        
        # print("     [RETRY] 실패 피드백을 Agent에게 전달하고 다시 시도합니다...")

        # # TODO 실행이 중단되더라도 무조건 데이터 저장 (CSV + DB)