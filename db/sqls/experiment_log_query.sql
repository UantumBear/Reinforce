-- ===================================================================
-- 실험 로그 조회 쿼리 (유사도 포함 + 에피소드 정렬)
-- ===================================================================

-- 1. 기본 조회 (에피소드 오름차순, raw_similarity 포함)
SELECT 
    id,
    experiment_id as "실험 ID",
    episode as "에피소드", 
    total_score as "점수",
    raw_similarity as "유사도",  -- 추가된 컬럼
    is_faithful as "충실도",
    is_style_match as "스타일",
    is_success as "성공",
    created_at as "시간"
FROM rl_optimization_logs 
WHERE experiment_id = 'exp_20260131_223111'  -- 원하는 실험 ID로 변경
ORDER BY episode ASC, created_at ASC;  -- 에피소드 0부터 정렬


-- 2. 에피소드별 평균 포함 조회
SELECT 
    id,
    experiment_id as "실험 ID",
    episode as "에피소드", 
    total_score as "점수",
    raw_similarity as "유사도",
    avg_total_score as "에피소드평균",
    is_faithful as "충실도",
    is_style_match as "스타일", 
    optimizer_model_nm as "최적화모델",
    tester_model_nm as "테스터모델",
    is_success as "성공",
    created_at as "시간"
FROM rl_optimization_logs 
WHERE experiment_id = 'exp_20260131_223111'  -- 원하는 실험 ID로 변경
ORDER BY episode ASC, created_at ASC;


-- 3. 최신 실험만 조회
SELECT 
    id,
    experiment_id as "실험 ID",
    episode as "에피소드", 
    total_score as "점수",
    raw_similarity as "유사도",
    is_faithful as "충실도",
    is_style_match as "스타일",
    is_success as "성공",
    created_at as "시간"
FROM rl_optimization_logs 
WHERE experiment_id = (
    SELECT experiment_id 
    FROM rl_optimization_logs 
    ORDER BY created_at DESC 
    LIMIT 1
)
ORDER BY episode ASC, created_at ASC;  -- 에피소드 0부터 정렬


-- 4. 컬럼 순서를 사용자 요청에 맞게 조정 (점수 - 유사도 - 충실도)
SELECT 
    id,
    experiment_id as "실험 ID",
    episode as "에피소드", 
    total_score as "점수",
    raw_similarity as "유사도",    -- 점수 다음에 위치
    is_faithful as "충실도",       -- 유사도 다음에 위치  
    is_style_match as "스타일",
    is_success as "성공",
    created_at as "시간"
FROM rl_optimization_logs 
WHERE experiment_id = 'exp_20260131_223111'  -- 실험 ID 변경 필요
ORDER BY episode ASC, id ASC;  -- 에피소드 0부터, 같은 에피소드는 ID 순으로


-- ===================================================================
-- 참고: 실험 ID 목록 조회
-- ===================================================================
SELECT DISTINCT experiment_id, MIN(created_at) as 시작시간, COUNT(*) as 데이터수
FROM rl_optimization_logs 
GROUP BY experiment_id 
ORDER BY MIN(created_at) DESC;