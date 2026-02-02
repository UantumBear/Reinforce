CREATE TABLE rl_optimization_logs (
    id SERIAL PRIMARY KEY,
    experiment_id VARCHAR(50) NOT NULL,
    episode INTEGER NOT NULL,
    
    instruction TEXT,
    question TEXT,
    context TEXT,
    
    model_answer TEXT,
    gold_answer TEXT,
    
    total_score REAL,
    raw_similarity REAL,
    
    is_faithful VARCHAR(20),
    is_style_match VARCHAR(20),
    
    critical_review TEXT,
    full_analysis TEXT,
    
    is_success BOOLEAN DEFAULT TRUE,
    error_log TEXT,
    
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 성능 최적화를 위한 인덱스 생성
CREATE INDEX idx_rl_optimization_logs_experiment_id ON rl_optimization_logs(experiment_id);
CREATE INDEX idx_rl_optimization_logs_episode ON rl_optimization_logs(episode);
CREATE INDEX idx_rl_optimization_logs_created_at ON rl_optimization_logs(created_at);
CREATE INDEX idx_rl_optimization_logs_total_score ON rl_optimization_logs(total_score);

-- 컬럼 추가: 데이터셋 크기와 평균 점수
ALTER TABLE rl_optimization_logs ADD COLUMN dataset_size INTEGER;
ALTER TABLE rl_optimization_logs ADD COLUMN avg_total_score REAL;

-- 컬럼 추가: 모델 정보
ALTER TABLE rl_optimization_logs ADD COLUMN optimizer_model_nm VARCHAR(100);
ALTER TABLE rl_optimization_logs ADD COLUMN optimizer_model_provider VARCHAR(50);
ALTER TABLE rl_optimization_logs ADD COLUMN tester_model_nm VARCHAR(100);
ALTER TABLE rl_optimization_logs ADD COLUMN tester_model_provider VARCHAR(50);

-- 컬럼 추가: 헌법(Constitution) 위반 여부
ALTER TABLE rl_optimization_logs ADD COLUMN constitution_status VARCHAR(20) DEFAULT 'Pass';
ALTER TABLE rl_optimization_logs ADD COLUMN constitution_violation_reason TEXT;

-- REAL 이란 PostgresSQL 의 부동소수점 타입 중 하나이다.
