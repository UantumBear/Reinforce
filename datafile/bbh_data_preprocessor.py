"""
@경로: datafile/bbh_data_preprocessor.py
@설명: BBH (Big Bench Hard) 데이터셋 전용 로더
        data_loader.py 에서 위임받아 호출됩니다.

@데이터셋: lukaemon/bbh
@저장 경로: datafile/original/lukaemon/bbh/<task>/
@지원 태스크 (예시):
    - object_counting
    - word_sorting
    - 기타 lukaemon/bbh 서브태스크 (다운로드된 것만 사용 가능)
@파일 종류:
    - train_51.csv  : 최적화용 서브셋 (51개)
    - validation_100.csv : 검증용 서브셋 (100개)
    - test.csv      : 원본 전체 테스트셋
"""
import random
import pandas as pd
import dspy
from pathlib import Path

# 프로젝트 루트 경로
# 이 파일(datafile/bbh_data_preprocessor.py)의 부모(datafile)의 부모(ProjectRoot)
BASE_DIR = Path(__file__).resolve().parent.parent


def load_bbh_dataset(dataset_name: str, sample_size=None, random_seed=42):
    """
    BBH (Big Bench Hard) 데이터셋을 로드합니다.
    TextGrad 논문 재현용.

    @param dataset_name: 아래 형식 중 하나
        - "lukaemon/bbh/<task>"          → train_51.csv (기본)
        - "lukaemon/bbh/<task>/train"    → train_51.csv
        - "lukaemon/bbh/<task>/valid"    → validation_100.csv
        - "lukaemon/bbh/<task>/test"     → test.csv
        예) "lukaemon/bbh/object_counting"
            "lukaemon/bbh/object_counting/train"

    @데이터셋 컬럼:
        - input  → question 으로 매핑
        - target → answer 로 매핑
        - context: 빈 문자열 (BBH는 context 없음)

    @return: dspy.Example 리스트
    """
    # dataset_name 파싱: "lukaemon/bbh/<task>[/<split>]"
    parts = dataset_name.strip("/").split("/")
    # parts 예: ["lukaemon", "bbh", "object_counting"] 또는
    #           ["lukaemon", "bbh", "object_counting", "train"]
    if len(parts) < 3 or parts[0] != "lukaemon" or parts[1] != "bbh":
        raise ValueError(
            f"BBH dataset_name 형식이 올바르지 않습니다: '{dataset_name}'\n"
            f"올바른 형식: 'lukaemon/bbh/<task>' 또는 'lukaemon/bbh/<task>/train|valid|test'"
        )

    task = parts[2]
    split = parts[3].lower() if len(parts) >= 4 else "train"

    split_to_file = {
        "train": "train_51.csv",
        "valid": "validation_100.csv",
        "validation": "validation_100.csv",
        "test": "test.csv",
    }
    if split not in split_to_file:
        raise ValueError(
            f"지원하지 않는 BBH split: '{split}'. "
            f"사용 가능: train, valid, validation, test"
        )

    file_name = split_to_file[split]
    file_path = BASE_DIR / "datafile" / "original" / "lukaemon" / "bbh" / task / file_name

    if not file_path.exists():
        raise FileNotFoundError(
            f"BBH 데이터셋을 찾을 수 없습니다: {file_path}\n"
            f"먼저 utils/datasets/baseline/bbh_download_datasets.py 를 실행하세요.\n"
            f"  예) python utils/datasets/baseline/bbh_download_datasets.py --task {task}"
        )

    try:
        df = pd.read_csv(file_path)
        print(f"[BBH Loader] ({task}/{file_name}) 로드 완료. 총 {len(df)}개 행.")
        print(f"[BBH Loader] 컬럼: {df.columns.tolist()}")
    except Exception as e:
        raise RuntimeError(f"BBH 데이터셋 읽기 실패: {e}") from e

    if sample_size:
        df = df.sample(n=min(sample_size, len(df)), random_state=random_seed)
        df = df.reset_index(drop=True)
        print(f"[BBH Loader] {random_seed} 시드로 {len(df)}개 샘플 추출")

    dataset = []
    for _, row in df.iterrows():
        if pd.isna(row.get("input")) or pd.isna(row.get("target")):
            continue
        example = dspy.Example(
            question=str(row["input"]),
            context="",  # BBH는 context 없음
            answer=str(row["target"]),
        ).with_inputs("question", "context")
        dataset.append(example)

    print(f"[BBH Loader] ({task}) 변환 완료: {len(dataset)}개 예제")

    random.seed(random_seed)
    random.shuffle(dataset)
    print(f"[BBH Loader] {random_seed} 시드로 셔플 완료")

    return dataset
