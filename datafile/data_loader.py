"""
@경로: datafile/data_loader.py
@설명: CSV 파일 등을 읽어 dspy.Example 리스트로 변환하는 모듈
"""
import random
import pandas as pd
import dspy
from pathlib import Path
import os
import json

# 프로젝트 루트 경로 (상대 경로 계산용)
# 이 파일(data_loader.py)의 부모(datafile)의 부모(ProjectRoot)
BASE_DIR = Path(__file__).resolve().parent.parent

def load_dataset(dataset_name=None, sample_size=None, random_seed=42, file_path=None):
    """
    데이터셋 이름을 기준으로 적절한 로더를 호출하여 DSPy Example 리스트로 반환합니다.
    
    @param dataset_name: 데이터셋 이름 (필수, None이거나 지원하지 않는 값이면 에러 발생)
        - "HJUNN/Finance-Law-merge-rag-dataset"
        - "didi0di/klue-mrc-ko-rag-cot"
        - "nasa/cmapss-fd001" ~ "nasa/cmapss-fd004"
        - "openai/gsm8k"
    @param sample_size: 샘플링할 데이터 개수 (None이면 전체 사용)
    @param random_seed: 랜덤시드 (재현 가능한 실험을 위해 기본값 42)
    @param file_path: 커스텀 파일 경로 (dataset_name이 None일 때만 사용)
    
    @raises ValueError: dataset_name이 None이거나 지원하지 않는 값인 경우
    """
    
    # 1. dataset_name 기준으로 적절한 로더 호출
    if dataset_name == "HJUNN/Finance-Law-merge-rag-dataset":
        return load_finance_law_dataset(sample_size=sample_size, random_seed=random_seed)
    elif dataset_name == "didi0di/klue-mrc-ko-rag-cot":
        return load_klue_rag_dataset(sample_size=sample_size, random_seed=random_seed)
    elif dataset_name and dataset_name.startswith("nasa/cmapss"):
        # "nasa/cmapss-fd001", "nasa/cmapss-fd002", "nasa/cmapss-fd003", "nasa/cmapss-fd004"
        return load_nasa_cmapss_dataset(dataset_name, sample_size=sample_size, random_seed=random_seed)
    elif dataset_name and dataset_name.startswith("openai/gsm8k"):
        # "openai/gsm8k-main" (TextGrad 논문 재현용)
        return load_gsm8k_dataset(sample_size=sample_size, random_seed=random_seed)
    
    # 2. dataset_name이 없거나 매칭되지 않으면 에러 발생 (실험 재현성을 위해 명시적 지정 필수)
    if dataset_name is None:
        if file_path is None:
            raise ValueError(
                "dataset_name 또는 file_path를 지정해야 합니다. "
                "사용 가능한 dataset_name: 'HJUNN/Finance-Law-merge-rag-dataset', 'didi0di/klue-mrc-ko-rag-cot', 'nasa/cmapss-fd001~fd004', 'openai/gsm8k'"
            )
    else:
        # dataset_name이 있지만 위에서 매칭되지 않은 경우
        raise ValueError(
            f"지원하지 않는 dataset_name: '{dataset_name}'. "
            "사용 가능한 dataset_name: 'HJUNN/Finance-Law-merge-rag-dataset', 'didi0di/klue-mrc-ko-rag-cot', 'nasa/cmapss-fd001~fd004', 'openai/gsm8k'"
        )
    
    # 3. file_path 기반 처리 (dataset_name이 None이고 file_path가 있는 경우만)
    if file_path is not None:
        # 경로가 문자열이면 Path 객체로 변환
        if isinstance(file_path, str):
            file_path = Path(file_path)
        
        # 파일 확장자에 따라 처리 방법 결정
        file_extension = file_path.suffix.lower()
        
        if file_extension == '.json':
            return _load_json_dataset(file_path, sample_size, random_seed)
        elif file_extension == '.csv':
            return _load_csv_dataset(file_path, sample_size, random_seed)
        else:
            print(f"[Error] 지원하지 않는 파일 형식: {file_extension}")
            return _get_hardcoded_examples()


def load_finance_law_dataset(sample_size=None, random_seed=42):
    """
    Finance Law JSON 데이터셋을 로드합니다.
    search_result 배열을 context로, answer를 gold_answer로 매핑합니다.
    """
    file_path = BASE_DIR / "datafile/original/HJUNN/Finance_Law_merge_rag_dataset_full.json"
    
    # 파일 존재 확인
    if not os.path.exists(file_path):
        print(f"[Warning] Finance Law 데이터셋을 찾을 수 없습니다: {file_path}")
        return _get_hardcoded_examples()
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        print(f"[Data Loader] Finance Law 데이터셋 로드 완료. 총 {len(data)}개 항목.")

    except Exception as e:
        print(f"[Error] Finance Law 데이터셋 로드 실패: {e}")
        return _get_hardcoded_examples()

    # 데이터 샘플링
    if sample_size:
        random.seed(random_seed)
        data = random.sample(data, min(sample_size, len(data)))
        print(f"[Data Loader] 랜덤시드 {random_seed}로 {len(data)}개 샘플 추출")

    # DSPy Example 변환 (Finance Law 전용 로직)
    dataset = []
    
    for idx, item in enumerate(data):
        # 필수 데이터 검증
        if not item.get('question') or not item.get('answer'):
            continue
            
        # search_result 배열을 context 문자열로 변환
        search_results = item.get('search_result', [])
        if isinstance(search_results, list):
            context_str = '\n\n'.join(search_results) if search_results else ""
        else:
            context_str = str(search_results)

        # Finance Law 데이터셋 전용 매핑 (기존 코드 호환성을 위해 answer 필드 사용)
        example = dspy.Example(
            question=item['question'],
            context=context_str,
            answer=item['answer']  # gold_answer 대신 answer 필드 사용 (호환성)
        ).with_inputs("question", "context")
        
        dataset.append(example)

    print(f"[Data Loader] Finance Law 데이터셋 변환 완료: {len(dataset)}개 예제")
    
    # 재현 가능한 셔플 (Train/Validation 분할을 위해)
    random.seed(random_seed)
    random.shuffle(dataset)
    print(f"[Data Loader] 랜덤시드 {random_seed}로 데이터셋 셔플 완료")
    
    return dataset


def load_gsm8k_dataset(sample_size=None, random_seed=42):
    """
    GSM8k (Grade School Math 8K) 데이터셋을 로드합니다.
    TextGrad 논문 재현용.
    
    @데이터셋 구조:
        - question: 수학 문제 (영어)
        - answer: 정답 ("#### 숫자" 형식 포함)
        - context 컬럼 없음 (GSM8k는 문제 자체가 전부)
    
    @매핑:
        - question: 그대로 사용
        - context: 빈 문자열 (GSM8k는 context가 없음)
        - answer: 정답
    
    @참고:
        - train.csv: 학습용 데이터 (프롬프트 최적화에 사용)
        - test.csv: 평가용 데이터 (최종 성능 측정에 사용)
        - 이 함수는 train.csv를 로드하여 experiment.py에서 train/validation 분리
    """
    file_path = BASE_DIR / "datafile/original/openai/gsm8k/main/train.csv"
    
    # 파일 존재 확인
    if not os.path.exists(file_path):
        print(f"[Warning] GSM8k 데이터셋을 찾을 수 없습니다: {file_path}")
        return _get_hardcoded_examples()
    
    try:
        df = pd.read_csv(file_path)
        print(f"[Data Loader] GSM8k 데이터셋 로드 완료. 총 {len(df)}개 행.")
        print(f"[CHECK] CSV 컬럼 목록: {df.columns.tolist()}")

    except Exception as e:
        print(f"[Error] GSM8k 데이터셋 로드 실패: {e}")
        return _get_hardcoded_examples()

    # 데이터 샘플링
    if sample_size:
        df = df.sample(n=min(sample_size, len(df)), random_state=random_seed)
        df = df.reset_index(drop=True)
        print(f"[Data Loader] 랜덤시드 {random_seed}로 {len(df)}개 샘플 추출")

    # DSPy Example 변환 (GSM8k 전용 로직)
    dataset = []
    
    for idx, row in df.iterrows():
        # 필수 데이터 검증
        if pd.isna(row.get('question')) or pd.isna(row.get('answer')):
            continue

        # GSM8k는 context가 없음 (수학 문제 자체가 전부)
        example = dspy.Example(
            question=row['question'],
            context='',  # GSM8k는 context 없음
            answer=row['answer']
        ).with_inputs("question", "context")
        
        dataset.append(example)

    print(f"[Data Loader] GSM8k 데이터셋 변환 완료: {len(dataset)}개 예제")
    
    # 재현 가능한 셔플 (Train/Validation 분할을 위해)
    random.seed(random_seed)
    random.shuffle(dataset)
    print(f"[Data Loader] 랜덤시드 {random_seed}로 데이터셋 셔플 완료")
    
    return dataset


def load_klue_rag_dataset(sample_size=None, random_seed=42):
    """
    KLUE RAG CSV 데이터셋을 로드합니다.
    기존 필드 매핑을 그대로 유지합니다.
    """
    file_path = BASE_DIR / "datafile/original/didi0di/klue-mrc-ko-rag-cot/search_result_3.csv"
    
    # 파일 존재 확인
    if not os.path.exists(file_path):
        print(f"[Warning] KLUE RAG 데이터셋을 찾을 수 없습니다: {file_path}")
        return _get_hardcoded_examples()
    
    try:
        df = pd.read_csv(file_path)
        print(f"[Data Loader] KLUE RAG 데이터셋 로드 완료. 총 {len(df)}개 행.")
        print(f"[CHECK] CSV 컬럼 목록: {df.columns.tolist()}")

    except Exception as e:
        print(f"[Error] KLUE RAG 데이터셋 로드 실패: {e}")
        return _get_hardcoded_examples()

    # 데이터 샘플링
    if sample_size:
        df = df.sample(n=min(sample_size, len(df)), random_state=random_seed)
        df = df.reset_index(drop=True)
        print(f"[Data Loader] 랜덤시드 {random_seed}로 {len(df)}개 샘플 추출")

    # DSPy Example 변환 (KLUE RAG 전용 로직)
    dataset = []
    
    for idx, row in df.iterrows():
        # 필수 데이터 검증
        if pd.isna(row.get('question')) or pd.isna(row.get('answer')):
            continue

        # KLUE RAG 데이터셋 전용 매핑 (기존 유지)
        example = dspy.Example(
            question=row['question'],
            context=row['search_result'],
            answer=row['answer']  # 기존대로 answer 필드 사용
        ).with_inputs("question", "context")
        
        dataset.append(example)

    print(f"[Data Loader] KLUE RAG 데이터셋 변환 완료: {len(dataset)}개 예제")
    
    # 재현 가능한 셔플 (Train/Validation 분할을 위해)
    random.seed(random_seed)
    random.shuffle(dataset)
    print(f"[Data Loader] 랜덤시드 {random_seed}로 데이터셋 셔플 완료")
    
    return dataset


def load_nasa_cmapss_dataset(dataset_name="nasa/cmapss-fd001", sample_size=None, random_seed=42):
    """
    NASA C-MAPSS 데이터셋을 로드합니다.
    'Code-based Facts' + 'LLM Narrative' 구조로 생성된 gold answer 포맷을 지원합니다.
    
    @param dataset_name: "nasa/cmapss-fd001" ~ "nasa/cmapss-fd004" 형식
    @param sample_size: 샘플링할 데이터 개수 (None이면 전체 사용)
    @param random_seed: 랜덤시드
    @return: dspy.Example 리스트
    
    데이터 구조:
    - id: 엔진 ID
    - input_log: 최근 30 사이클의 센서 로그 (텍스트)
    - gold_facts: 코드 기반 객관적 지표 (dict)
      - recent_avg: 최근 평균값
      - change_from_initial: 초기 대비 변화량
      - volatility: 변동성
      - trend_slope: 추세 기울기
      - actual_rul: 실제 RUL 값
      - status_label: 상태 등급
    - gold_standard_report: LLM이 작성한 모범 보고서 (텍스트)
    
    DSPy 매핑:
    - question: "다음 센서 로그를 분석하고 엔진 상태 보고서를 작성하시오."
    - context: input_log (센서 로그)
    - answer: gold_standard_report (모범 답안 보고서)
    - gold_facts: 검증용 객관적 지표 (추가 필드)
    - actual_rul: 정답 RUL 값 (추가 필드)
    """
    
    # 데이터셋 이름에서 FD 번호 추출 (예: "nasa/cmapss-fd001" -> "FD001")
    fd_number = None
    if "fd001" in dataset_name.lower():
        fd_number = "FD001"
    elif "fd002" in dataset_name.lower():
        fd_number = "FD002"
    elif "fd003" in dataset_name.lower():
        fd_number = "FD003"
    elif "fd004" in dataset_name.lower():
        fd_number = "FD004"
    else:
        # FD 번호가 없으면 기본 gold_standard_dataset_ver_gemini.json 사용
        fd_number = None
    
    # 파일 경로 결정
    if fd_number:
        file_path = BASE_DIR / f"datafile/preprocess/nasa/gold_standard_dataset_ver_gemini_{fd_number}_cnt_30.json"
    else:
        file_path = BASE_DIR / "datafile/preprocess/nasa/gold_standard_dataset_ver_gemini.json"
    
    # 파일 존재 확인
    if not os.path.exists(file_path):
        print(f"[Warning] NASA C-MAPSS 데이터셋을 찾을 수 없습니다: {file_path}")
        print(f"[Warning] 사용 가능한 데이터셋: nasa/cmapss-fd001 ~ nasa/cmapss-fd004")
        return _get_hardcoded_examples()
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        print(f"[Data Loader] NASA C-MAPSS {fd_number or ''} 데이터셋 로드 완료. 총 {len(data)}개 엔진.")

    except Exception as e:
        print(f"[Error] NASA C-MAPSS 데이터셋 로드 실패: {e}")
        return _get_hardcoded_examples()

    # 데이터 샘플링
    if sample_size:
        random.seed(random_seed)
        data = random.sample(data, min(sample_size, len(data)))
        print(f"[Data Loader] 랜덤시드 {random_seed}로 {len(data)}개 엔진 샘플 추출")

    # DSPy Example 변환 (NASA C-MAPSS 전용 로직)
    dataset = []
    
    for idx, item in enumerate(data):
        # 필수 데이터 검증
        if not item.get('input_log') or not item.get('gold_standard_report'):
            print(f"[Warning] 항목 {idx} 스킵: input_log 또는 gold_standard_report 누락")
            continue
        
        # NASA C-MAPSS는 질문이 고정적임
        question = "다음 센서 로그를 분석하고 엔진 상태 보고서를 작성하시오."
        
        # NASA 데이터셋 전용 매핑
        example = dspy.Example(
            question=question,                                      # 고정 질문
            context=item['input_log'],                             # 센서 로그
            answer=item['gold_standard_report'],                   # 모범 보고서
            gold_facts=item.get('gold_facts', {}),                 # 객관적 지표 (검증용)
            actual_rul=item.get('gold_facts', {}).get('actual_rul', None),  # RUL 값
            status_label=item.get('gold_facts', {}).get('status_label', None),  # 상태 등급
            engine_id=item.get('id', idx)                          # 엔진 ID
        ).with_inputs("question", "context")
        
        dataset.append(example)

    print(f"[Data Loader] NASA C-MAPSS 데이터셋 변환 완료: {len(dataset)}개 예제")
    print(f"[Data Loader] 각 예제는 gold_facts(객관 지표), actual_rul(RUL), status_label(상태)를 포함합니다.")
    
    # 재현 가능한 셔플 (Train/Validation 분할을 위해)
    random.seed(random_seed)
    random.shuffle(dataset)
    print(f"[Data Loader] 랜덤시드 {random_seed}로 데이터셋 셔플 완료")
    
    return dataset


def _load_json_dataset(file_path, sample_size=None, random_seed=42):
    """
    범용 JSON 파일 로더 (기존 호환성을 위해 유지)
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        print(f"[Data Loader] '{file_path.name}' 로드 완료. 총 {len(data)}개 항목.")

    except Exception as e:
        print(f"[Error] JSON 로드 실패: {e}")
        return _get_hardcoded_examples()

    # 데이터 샘플링
    if sample_size:
        random.seed(random_seed)
        data = random.sample(data, min(sample_size, len(data)))
        print(f"[Data Loader] 랜덤시드 {random_seed}로 {len(data)}개 샘플 추출")

    # DSPy Example 변환 (범용 JSON 로직)
    dataset = []
    
    for idx, item in enumerate(data):
        if not item.get('question') or not item.get('answer'):
            continue
            
        search_results = item.get('search_result', [])
        if isinstance(search_results, list):
            context_str = '\n\n'.join(search_results) if search_results else ""
        else:
            context_str = str(search_results)

        example = dspy.Example(
            question=item['question'],
            context=context_str,
            answer=item['answer']  # 호환성을 위해 answer 필드 사용
        ).with_inputs("question", "context")
        
        dataset.append(example)

    print(f"[Data Loader] JSON 데이터셋 변환 완료: {len(dataset)}개 예제")
    
    # 재현 가능한 셔플 (Train/Validation 분할을 위해)
    random.seed(random_seed)
    random.shuffle(dataset)
    print(f"[Data Loader] 랜덤시드 {random_seed}로 데이터셋 셔플 완료")
    
    return dataset


def _load_csv_dataset(file_path, sample_size=None, random_seed=42):
    """
    범용 CSV 파일 로더 (기존 호환성을 위해 유지)
    """
    try:
        df = pd.read_csv(file_path)
        print(f"[Data Loader] '{file_path.name}' 로드 완료. 총 {len(df)}개 행.")
        print(f"[CHECK] CSV 컬럼 목록: {df.columns.tolist()}")

    except Exception as e:
        print(f"[Error] CSV 로드 실패: {e}")
        return _get_hardcoded_examples()

    # 데이터 샘플링
    if sample_size:
        df = df.sample(n=min(sample_size, len(df)), random_state=random_seed)
        df = df.reset_index(drop=True)
        print(f"[Data Loader] 랜덤시드 {random_seed}로 {len(df)}개 샘플 추출")

    # DSPy Example 변환 (범용 CSV 로직)
    dataset = []
    
    for idx, row in df.iterrows():
        if pd.isna(row.get('question')) or pd.isna(row.get('answer')):
            continue

        # search_result 컬럼이 없을 수 있으므로 안전하게 처리
        context = row.get('search_result', row.get('context', ''))
        if pd.isna(context):
            context = ''

        example = dspy.Example(
            question=row['question'],
            context=context,
            answer=row['answer']
        ).with_inputs("question", "context")
        
        dataset.append(example)

    print(f"[Data Loader] CSV 데이터셋 변환 완료: {len(dataset)}개 예제")
    
    # 재현 가능한 셔플 (Train/Validation 분할을 위해)
    
    random.seed(random_seed)
    random.shuffle(dataset)
    print(f"[Data Loader] 랜덤시드 {random_seed}로 데이터셋 셔플 완료")
    
    return dataset

def _get_hardcoded_examples(): # TODO 나중에 함수명 바꾸기
    """파일이 없을 때 로그를 출력하고 아무것도 하지 않음. """
    raise NotImplementedError("데이터가 정의되지 않았습니다. 파일 경로를 확인하세요.")