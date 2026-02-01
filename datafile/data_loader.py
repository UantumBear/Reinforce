"""
@경로: datafile/data_loader.py
@설명: CSV 파일 등을 읽어 dspy.Example 리스트로 변환하는 모듈
"""

import pandas as pd
import dspy
from pathlib import Path
import os

# 프로젝트 루트 경로 (상대 경로 계산용)
# 이 파일(data_loader.py)의 부모(datafile)의 부모(ProjectRoot)
BASE_DIR = Path(__file__).resolve().parent.parent

def load_dataset(file_path=None, sample_size=None):
    """
    CSV 파일을 로드하여 DSPy Example 리스트로 반환합니다.
    file_path가 없으면 기본 경로의 파일을 찾습니다.
    """
    
    # 1. 파일 경로 설정 (이미지에 있는 경로 반영)
    if file_path is None:
        # 기본적으로 읽어올 파일 경로 설정
        file_path = BASE_DIR / "datafile/original/didi0di/klue-mrc-ko-rag-cot/search_result_3.csv"
    
    # 파일 존재 확인
    if not os.path.exists(file_path):
        print(f"[Warning] 파일을 찾을 수 없습니다: {file_path}")
        print(">> 하드코딩된 예시 데이터를 반환합니다.")
        return _get_hardcoded_examples()

    # 2. CSV 로드
    try:
        df = pd.read_csv(file_path)
        print(f"[Data Loader] '{file_path.name}' 로드 완료. 총 {len(df)}개 행.")

        print(f"[CHECK] CSV 컬럼 목록: {df.columns.tolist()}")

    except Exception as e:
        print(f"[Error] CSV 로드 실패: {e}")
        return _get_hardcoded_examples()

    # 3. 데이터 샘플링 (테스트용으로 개수 제한 가능)
    if sample_size:
        df = df.head(sample_size)

    # 4. DSPy Example 변환
    dataset = []
    
    # [중요] CSV의 컬럼명을 확인해야 합니다! 
    # 일단 일반적인 이름(question, context, answer)으로 가정하고 작성합니다.
    # 만약 CSV 컬럼명이 'query', 'ground_truth' 등으로 되어있다면 아래를 수정해야 합니다.
    
    for idx, row in df.iterrows():
        # 필수 데이터가 비어있는 경우 건너뜀
        if pd.isna(row.get('question')) or pd.isna(row.get('answer')):
            continue

        # [수정 포인트] CSV 컬럼명 매핑
        # context -> search_result 로 변경되었습니다.
        example = dspy.Example(
            question=row['question'],      # CSV 컬럼명: 'question' (일치)
            context=row['search_result'],  # CSV 컬럼명: 'search_result' (수정됨!)
            answer=row['answer']           # CSV 컬럼명: 'answer' (일치)
        ).with_inputs("question", "context")
        
        dataset.append(example)

    print(f"[Data Loader] DSPy 데이터셋 변환 완료: {len(dataset)}개 예제")
    return dataset

def _get_hardcoded_examples():
    """파일이 없을 때 사용할 비상용 하드코딩 데이터"""
    return [
        dspy.Example(
            question="이순신 장군이 전사한 해전은?",
            context="이순신은 1598년 노량 해전에서 전사했다. 그의 마지막 말은 '나의 죽음을 적에게 알리지 말라'였다.",
            answer="노량 해전"
        ).with_inputs("question", "context"),
        dspy.Example(
            question="반도체 공정 중 식각 공정이란?",
            context="식각(Etching)은 웨이퍼 표면의 불필요한 부분을 깎아내는 공정이다. 이를 통해 회로 패턴을 형성한다.",
            answer="웨이퍼 표면의 불필요한 부분을 깎아내어 회로 패턴을 형성하는 공정"
        ).with_inputs("question", "context")
    ]