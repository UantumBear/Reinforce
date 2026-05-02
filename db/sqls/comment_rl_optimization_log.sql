
-- 컬럼별 설명 추가 (COMMENT)
COMMENT ON COLUMN rl_optimization_logs.accuracy IS 'Train 단계에서 처리된 각 샘플의 정답 여부 (0: 오답, 1: 정답). GSM8k의 경우 숫자 추출 후 비교하여 계산. Validation 평균 accuracy와는 다름.';
COMMENT ON COLUMN rl_optimization_logs.avg_total_score IS 'Train 단계에서 해당 iteration의 batch 샘플들의 평균 점수 (raw_similarity 기준)';
COMMENT ON COLUMN rl_optimization_logs.dataset_size IS 'Train 단계에서 해당 iteration의 배치 샘플 개수 (batch_size). 실제로 프롬프트 최적화에 사용된 샘플 수.';
COMMENT ON COLUMN rl_optimization_logs.dataset_nm IS '실험에 사용된 데이터셋 이름 (예: openai/gsm8k, nasa/cmapss-fd001, Idavidrein/gpqa-diamond)';
COMMENT ON COLUMN rl_optimization_logs.validation_info IS 'Validation 단계에서 평가된 각 샘플의 상세 정보. JSON 형식: {"0": {"Q": "질문", "A": "모델응답", "GA": "모범답안", "score": 1.0}, "1": {...}, ...}';
COMMENT ON COLUMN rl_optimization_logs.validation_accuracy IS 'Validation 데이터셋 전체에 대한 평균 accuracy (0.0 ~ 1.0). 프롬프트 성능 평가 지표로 사용.';
COMMENT ON COLUMN rl_optimization_logs.validation_dataset_size IS 'Validation 데이터셋 샘플 개수. 프롬프트 성능 평가에 사용된 검증 데이터 크기.';
COMMENT ON COLUMN rl_optimization_logs.forward_tester_llm_call_cnt IS 'Forward Model(답변 생성자 LLM) 호출 횟수';
COMMENT ON COLUMN rl_optimization_logs.backward_judge_llm_call_cnt IS 'Backward Judge(평가자 LLM) 호출 횟수';
COMMENT ON COLUMN rl_optimization_logs.backward_optimizer_llm_call_cnt IS 'Backward Optimizer(프롬프트 개선 LLM) 호출 횟수';
COMMENT ON COLUMN rl_optimization_logs.full_analysis IS 'system_prompt.gradients - backward() 실행 후 JudgeLLM이 생성한 gradient 피드백 텍스트. baseline: gradient 원문, improve: gradient + 해당 iteration batch 샘플 비평 묶음(계층형 구조).';
