"""
Docstring for utils.tools.dspy_ai.common.load_dspy_data

@경로: utils/tools/dspy_ai/common/load_dspy_data.py
@설명: DSPy 프레임워크에서 사용할 수 있도록 데이터셋을 로드하고 변환하는 소스 코드
@호출 방법: from utils.tools.dspy_ai.common.load_dspy_data import load_data_for_dspy 
@명령어: python utils/tools/dspy_ai/common/load_dspy_data.py
"""

import pandas as pd
import dspy
import ast  # 문자열을 진짜 리스트로 바꿔주는 도구
from pathlib import Path
import os

def load_data_for_dspy():
    # 1. 파일 경로 (본인 경로에 맞게 수정)
    data_path = "datafile/original/didi0di/klue-mrc-ko-rag-cot/search_result_3.csv"
    dataset_name = "klue-mrc-ko-rag-cot"  # 데이터셋 식별용
    
    if not os.path.exists(data_path):
        print(f"[ERROR] 파일을 찾을 수 없습니다: {data_path}")
        return [], ""

    print(f"[INFO] 데이터 로딩 중...: {data_path}")
    df = pd.read_csv(data_path)

    dataset = []
    
    for index, row in df.iterrows():
        # [수정된 부분] 문자열을 진짜 리스트로 변환!
        # 예: "['문서1', '문서2']" -> ['문서1', '문서2']
        context_list = ast.literal_eval(row['search_result'])
        
        # 리스트 안의 문서들을 하나의 긴 글(Context)로 합치기
        # (DSPy는 보통 하나의 긴 텍스트를 좋아합니다)
        context_str = "\n\n".join(context_list)
        
        example = dspy.Example(
            question=row['question'],
            context=context_str,       # 깨끗하게 합쳐진 문서 내용
            answer=str(row['answer'])
        )
        
        # 입력으로 사용할 필드 지정
        example = example.with_inputs('question', 'context')
        
        dataset.append(example)

    print(f"[SUCCESS] 총 {len(dataset)}개의 데이터가 변환되었습니다.")
    return dataset, dataset_name

if __name__ == "__main__":
    data, dataset_name = load_data_for_dspy()
    if data:
        print(f"\n[INFO] 데이터셋: {dataset_name}")
        print("\n[INFO] 변환된 데이터 확인 (Context 일부):")
        # 첫 번째 데이터의 문맥 앞부분만 출력해서 잘 합쳐졌는지 확인
        print(data[0].context[:200])