"""
@경로: datafile/gsm8k_data_preprocessor.py
@설명: GSM8k 데이터셋 전용 로더
        data_loader.py 에서 위임받아 호출됩니다.

@데이터셋: openai/gsm8k (main config)
@저장 경로: datafile/original/openai/gsm8k/main/train.csv
@용도: TextGrad 프롬프트 최적화 논문 재현
"""
import random
import pandas as pd
import dspy
from pathlib import Path

# 프로젝트 루트 경로
# 이 파일(datafile/gsm8k_data_preprocessor.py)의 부모(datafile)의 부모(ProjectRoot)
BASE_DIR = Path(__file__).resolve().parent.parent


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
    file_path = BASE_DIR / "datafile" / "original" / "openai" / "gsm8k" / "main" / "train.csv"

    if not file_path.exists():
        raise FileNotFoundError(
            f"GSM8k 데이터셋을 찾을 수 없습니다: {file_path}\n"
            f"먼저 utils/datasets/baseline/gsm8k_download_datasets.py 를 실행하세요."
        )

    try:
        df = pd.read_csv(file_path)
        print(f"[GSM8k Loader] 로드 완료. 총 {len(df)}개 행.")
        print(f"[GSM8k Loader] 컬럼: {df.columns.tolist()}")
    except Exception as e:
        raise RuntimeError(f"GSM8k 데이터셋 읽기 실패: {e}") from e

    if sample_size:
        df = df.sample(n=min(sample_size, len(df)), random_state=random_seed)
        df = df.reset_index(drop=True)
        print(f"[GSM8k Loader] {random_seed} 시드로 {len(df)}개 샘플 추출")

    dataset = []
    for _, row in df.iterrows():
        if pd.isna(row.get("question")) or pd.isna(row.get("answer")):
            continue
        example = dspy.Example(
            question=row["question"],
            context="",  # GSM8k는 context 없음
            answer=row["answer"],
        ).with_inputs("question", "context")
        dataset.append(example)

    print(f"[GSM8k Loader] 변환 완료: {len(dataset)}개 예제")

    random.seed(random_seed)
    random.shuffle(dataset)
    print(f"[GSM8k Loader] {random_seed} 시드로 셔플 완료")

    return dataset
