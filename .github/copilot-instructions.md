# GitHub Copilot 개발 가이드

이 문서는 GitHub Copilot이 코드 작성 시 참고하는 프로젝트 규칙입니다.

## 🎯 핵심 설계 원칙

### 1. 중앙 집중식 실험 관리
- **모든 실험 설정은 `TextGradExperiment` 클래스에서 관리**
  - 위치: `utils/environment/experiment.py`
  - Baseline vs Improve 차이는 `mode` 변수 하나로 제어
  - main 파일은 `EXPERIMENT_INS` 인스턴스를 통해서만 설정 접근

```python
# ✅ 올바른 예: EXPERIMENT_INS를 통한 중앙 관리
EXPERIMENT_INS = TextGradExperiment(mode='baseline')  # or mode='improve'
evaluation_fn = EXPERIMENT_INS.get_objective_function(ground_truth)
feedback_text = EXPERIMENT_INS.extract_feedback_str(system_prompt, logs, idx)

# ❌ 잘못된 예: main 파일에 직접 분기 로직
if mode == 'baseline':
    evaluation_fn = build_simple_eval(ground_truth)
else:
    evaluation_fn = build_complex_eval(ground_truth)
```

### 2. Diff 친화적 코드 작성
- **Baseline과 Improve 파일은 최대한 동일한 구조 유지**
- 차이점은 `##### 차별점 #####` 주석 블록으로 명확히 표시
- 함수 호출 형식은 통일, 인자 값만 다르게 전달
  - Baseline: `function(param=None)` (사용 안 함을 명시)
  - Improve: `function(param=actual_value)` (실제 값 전달)

```python
##### 차별점 #####
# [Baseline] optimization_logs=None  → 샘플 비평 미사용
# [Improve] optimization_logs=logs   → 샘플 비평 활용하여 계층형 피드백 생성
feedback = EXPERIMENT_INS.extract_feedback_str(
    system_prompt=system_prompt,
    optimization_logs=optimization_logs,  # ← 이 부분만 다름
    iteration_log_start_idx=iteration_log_start_idx
)
###################
```

### 3. 모드 기반 분기 처리 규칙
```python
# TextGradExperiment 클래스 내부에서 mode 기반 분기
def get_objective_function(self, ground_truth: str) -> str:
    if self.mode == 'baseline':
        return self._build_baseline_evaluation_instruction(ground_truth)
    elif self.mode == 'improve':
        return self._build_hierarchical_evaluation_instruction(ground_truth)

def extract_feedback_str(self, system_prompt, optimization_logs=None, iteration_log_start_idx=None) -> str:
    if self.mode == 'baseline':
        # 단순 gradient만 반환
        return system_prompt.get_gradient_text().strip() or "[N/A]"
    elif self.mode == 'improve':
        # gradient + 샘플 비평을 3계층 구조화
        ...
```

---

## 📂 프로젝트 구조

### 🔬 실험 Main 파일
- `main_textgrad_baseline.py` - TextGrad 논문 재현 (`mode='baseline'`)
- `main_textgrad_improve.py` - 계층형 피드백 적용 (`mode='improve'`)
- `main_textgrad_baseline_copy.py` - Baseline + Improve 차별점 통합 버전 (비교용)

### 🏗️ 핵심 클래스 및 함수

#### `TextGradExperiment` (`utils/environment/experiment.py`)
프로젝트의 **중앙 제어 클래스**입니다. 모든 실험 설정과 차별점은 여기서 관리됩니다.

**주요 메서드:**
- `get_objective_function(ground_truth)` - 평가 지시문 생성 (Evaluation Instruction)
  - Baseline: 4가지 기준 단순 평가
  - Improve: 3-Layer 계층형 rubric 평가
  
- `extract_feedback_str(system_prompt, optimization_logs, iteration_log_start_idx)` - 피드백 텍스트 추출
  - Baseline: 단순 gradient 텍스트
  - Improve: gradient + 샘플 비평을 3계층 구조화
  
- `load_and_split_data()` - 데이터 로드 및 Train/Validation 분할
- `get_initial_prompt()` - 초기 시스템 프롬프트 반환
- `get_test_time_updates()` - 데이터셋별 답변 생성 횟수 (GPQA=3, GSM8k=1)

#### 데이터 로드 (`datafile/data_loader.py`)
- `load_dataset(dataset_name, sample_size)` - 데이터셋 로드

#### 로그 빌더 (`utils/environment/textgrad_log_builder.py`)
DB 저장용 로그 생성 헬퍼 함수들:
- `create_base_log()` - 공통 필드 생성
- `create_success_log()` - 성공 케이스 로그
- `create_error_log()` - 에러 케이스 로그
- `create_skip_log()` - 스킵 케이스 로그

---

## 🔧 코드 수정 시 체크리스트

### ✅ 새 기능 추가 시
1. [ ] Baseline과 Improve의 **차이점**인가요?
   - **Yes**: `TextGradExperiment` 클래스에 메서드 추가하고 `mode`로 분기
   - **No**: 공통 utils 함수로 구현

2. [ ] main 파일 수정이 필요한가요?
   - [ ] `main_textgrad_baseline.py` 수정
   - [ ] `main_textgrad_improve.py` 수정
   - [ ] 두 파일의 **구조는 동일**하게 유지 (인자 값만 다르게)

3. [ ] 차별점 주석 추가했나요?
   ```python
   ##### 차별점 #####
   # [Baseline] 설명
   # [Improve] 설명
   code_here()
   ###################
   ```

### ✅ 리팩토링 시
1. [ ] 기존 차별점을 `TextGradExperiment`로 통합할 수 있나요?
2. [ ] Baseline/Improve 파일 간 diff가 최소화되나요?
3. [ ] 주석이 명확한가요? (특히 `##### 차별점 #####` 블록)

---

## 📊 TextGrad 워크플로우 이해

### 1️⃣ Forward Pass (답변 생성)
```python
# forward_engine(답변 생성자 LLM)이 답변 생성
prediction_var = model(query_var)  # BlackboxLLM이 forward_engine 호출
```

### 2️⃣ Evaluation (평가)
```python
# ##### 차별점 #####
# [Baseline] StringBasedFunction (Python 함수로 0/1 계산)
# [Improve] TextLoss (backward_engine이 평가 + 피드백 생성)

evaluation_instruction = EXPERIMENT_INS.get_objective_function(ground_truth)
loss = tg.TextLoss(evaluation_instruction)
computed_loss = loss(prediction_var)
```

### 3️⃣ Backward Pass (피드백 생성)
```python
# backward_engine(평가자 LLM)이 gradient(피드백) 생성
total_loss.backward()
```

### 4️⃣ Feedback 추출
```python
# ##### 차별점 #####
# [Baseline] system_prompt.get_gradient_text() → 단순 gradient
# [Improve] gradient + 샘플 비평 → 3계층 구조화

feedback = EXPERIMENT_INS.extract_feedback_str(
    system_prompt=system_prompt,
    optimization_logs=optimization_logs,  # Baseline: None, Improve: actual_logs
    iteration_log_start_idx=iteration_log_start_idx
)
```

### 5️⃣ Optimizer Step (프롬프트 개선)
```python
# backward_engine이 개선된 프롬프트 생성
optimizer.step()
```

---

## 🎨 코딩 스타일 가이드

### 주석 작성 규칙
```python
# [중요] 핵심 개념 설명 시
# [주의] 주의사항, 함정 경고 시
# [TextGrad 논문] 논문 기반 구현 시
# [차별점] Baseline vs Improve 차이 시 → ##### 차별점 ##### 블록 사용

##### 차별점 #####
# [Baseline] 기능 A 사용
# [Improve] 기능 B 사용
###################
```

### 변수명 규칙
- `context` - RAG 문서 자료 (데이터의 context)
  - ⚠️ TextGrad의 `<CONTEXT>` 태그(이전 최적화 피드백)와는 다름!
- `system_prompt` - 최적화 대상 프롬프트 (TextGrad Variable 객체)
- `forward_engine` - 답변 생성자 LLM
- `backward_engine` - 평가/피드백/최적화 담당 LLM (3가지 역할 모두!)

### 에러 처리
```python
try:
    # 샘플 처리 로직
    ...
except Exception as sample_error:
    root_error = extract_root_error_message(sample_error)
    optimization_logs.append(create_error_log(
        base_log, system_prompt.value, question, context, ground_truth, root_error
    ))
    # [치명적 에러] 배치 일관성을 위해 즉시 중단
    raise RuntimeError(f"Sample processing failed...") from sample_error
```

---

## 🔍 차별점 요약표

| 구분 | Baseline | Improve |
|------|----------|---------|
| **실험 모드** | `TextGradExperiment(mode='baseline')` | `TextGradExperiment(mode='improve')` |
| **평가 방식** | 4가지 기준 단순 평가 | 3-Layer 계층형 rubric |
| **피드백 구조** | 단순 gradient 텍스트 | gradient + 샘플 비평 3계층 구조 |
| **평가 함수** | `StringBasedFunction` (GSM8k) or `TextLoss` | `TextLoss` (계층형 평가 지시문) |
| **optimization_logs 사용** | `None` (미사용) | 실제 logs 전달 (샘플 비평 수집) |

---

## 📖 참고 문서
- `docs/project_design.md` - 전체 프로젝트 설계
- `docs/project_structure.md` - 디렉토리 구조
- `docs/textgrad_baseline_reproduction_gap.md` - 논문 재현 이슈

---

## 🚀 빠른 시작 예제

### 새 실험 모드 추가하기
```python
# 1. utils/environment/experiment.py에 메서드 추가
class TextGradExperiment:
    def get_new_feature(self):
        if self.mode == 'baseline':
            return baseline_version()
        elif self.mode == 'improve':
            return improve_version()
        elif self.mode == 'my_new_mode':  # ← 새 모드 추가
            return my_new_version()

# 2. main 파일에서 호출
result = EXPERIMENT_INS.get_new_feature()

# 3. 차별점 주석 추가
##### 차별점 #####
# [Baseline] baseline_version() 사용
# [Improve] improve_version() 사용
# [My New Mode] my_new_version() 사용
###################
```

---

**마지막 업데이트**: 2026-04-18  
**작성자**: 프로젝트 팀  
**문의**: 코드 리뷰 시 이 가이드를 참고해주세요!
