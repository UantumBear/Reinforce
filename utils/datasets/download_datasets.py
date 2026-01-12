"""
Docstring for utils.datasets.download_datasets

@경로: utils/datasets/download_datasets.py
@설명: 지정된 데이터셋을 다운로드하여 프로젝트의 datasets/original 폴더에 저장하는 소스 코드
@명령어: python utils/datasets/download_datasets.py
"""


import os
from pathlib import Path
from datasets import load_dataset
import pandas as pd

def download_dataset():
    # dataset_id = "didi0di/klue-mrc-ko-rag-cot" # Git에 올리지 않으므로 다운받기 1
    dataset_id = "iamjoon/klue-mrc-ko-rag-dataset"  # Git에 올리지 않으므로 다운받기 2
    print(f"[INFO] 데이터셋 다운로드 시작: {dataset_id} ...")

    try:
        # ---------------------------------------------------------
        # [1] 경로 설정
        # ---------------------------------------------------------
        # 현재 실행 중인 파일의 위치: .../Reinforce/utils/datasets/download_datasets.py
        current_file = Path(__file__).resolve()
        
        # 프로젝트 루트 찾기 (3단계 위로: 파일 -> datasets폴더 -> utils폴더 -> Reinforce루트)
        project_root = current_file.parents[2]
        
        # 목표 저장 경로: .../Reinforce/datasets/original/{dataset_id}
        # (dataset_id에 포함된 '/' 덕분에 자동으로 하위 폴더가 생성된다)
        save_dir = project_root / "datasets" / "original" / dataset_id
        
        # 폴더가 없으면 자동으로 생성 (parents=True: 상위 폴더까지 생성, exist_ok=True: 이미 있어도 에러 안 냄)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"[INFO] 저장될 폴더 경로:\n   {save_dir}")

        # ---------------------------------------------------------
        # [2] 데이터 다운로드 및 저장
        # ---------------------------------------------------------
        dataset = load_dataset(dataset_id)
        
        # [2] 데이터 다운로드
        dataset = load_dataset(dataset_id)
        
        print(f"\n[INFO] 다운로드된 데이터 목록(Keys): {list(dataset.keys())}")

        # [3] 모든 데이터 저장 (train이 없어도 동작하도록 수정)
        # 데이터셋에 있는 모든 키(split)를 돌면서 다 저장합니다.
        for split_name in dataset.keys():
            print(f"\nrunning... [{split_name}] 저장 중...")
            
            output_path = save_dir / f"{split_name}.csv"
            
            # pandas로 변환 및 저장
            df = pd.DataFrame(dataset[split_name])
            df.to_csv(output_path, index=False, encoding='utf-8-sig')
            
            print(f"  -> 저장 완료: {output_path.name}")

        print(f"\n[SUCCESS] 모든 작업이 완료되었습니다!")

    except Exception as e:
        print(f"\n[ERROR] 오류 발생:\n{e}")

if __name__ == "__main__":
    download_dataset()