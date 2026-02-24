"""
@경로: datafile/data_loader.py
@설명: CSV 파일 등을 읽어 dspy.Example 리스트로 변환하는 모듈
"""

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
    
    @param dataset_name: 데이터셋 이름 ("HJUNN/Finance-Law-merge-rag-dataset" 또는 "didi0di/klue-mrc-ko-rag-cot")
    @param sample_size: 샘플링할 데이터 개수 (None이면 전체 사용)
    @param random_seed: 랜덤시드 (재현 가능한 실험을 위해 기본값 42)
    @param file_path: 커스텀 파일 경로 (dataset_name보다 우선순위 낮음)
    """
    
    # 1. dataset_name 기준으로 적절한 로더 호출
    if dataset_name == "HJUNN/Finance-Law-merge-rag-dataset":
        return load_finance_law_dataset(sample_size=sample_size, random_seed=random_seed)
    elif dataset_name == "didi0di/klue-mrc-ko-rag-cot":
        return load_klue_rag_dataset(sample_size=sample_size, random_seed=random_seed)
    
    # 2. dataset_name이 없으면 file_path 기반으로 처리 (기존 호환성)
    if file_path is None:
        # 기본값: KLUE RAG 데이터셋
        return load_klue_rag_dataset(sample_size=sample_size, random_seed=random_seed)
    else:
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
        import random
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
        import random
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

        example = dspy.Example(
            question=row['question'],
            context=row['search_result'],
            answer=row['answer']
        ).with_inputs("question", "context")
        
        dataset.append(example)

    print(f"[Data Loader] CSV 데이터셋 변환 완료: {len(dataset)}개 예제")
    return dataset

def _get_hardcoded_examples(): # TODO 나중에 함수명 바꾸기
    """파일이 없을 때 로그를 출력하고 아무것도 하지 않음. """
    raise NotImplementedError("데이터가 정의되지 않았습니다. 파일 경로를 확인하세요.")