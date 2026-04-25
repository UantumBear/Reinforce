"""
Docstring for utils.datasets.baseline.sktTelagentbench_download_datasets

@경로: utils/datasets/baseline/sktTelagentbench_download_datasets.py
@설명: SKT TelAgentBench 원본 데이터를 다운로드하고 안전하게 파싱하는 소스 코드
@데이터셋: skt/TelAgentBench (Gated Dataset)
@용도: 제약 조건 및 답변 포맷(Form) 강제를 위한 프롬프트 최적화 성능 평가용

@특이사항: 
- Hugging Face datasets 라이브러리의 PyArrow 스키마 에러(ArrowInvalid)를 우회하기 위해,
  snapshot_download로 원본 JSON을 직접 다운로드한 후 Pandas로 안전하게 파싱합니다.

@명령어: python utils/datasets/baseline/sktTelagentbench_download_datasets.py
"""

import os
import json
from pathlib import Path
import pandas as pd
from dotenv import load_dotenv
from huggingface_hub import snapshot_download, login, HfApi


def download_and_parse_telagentbench():
    dataset_id = "skt/TelAgentBench"
    
    print(f"[INFO] TelAgentBench 데이터셋 직접 다운로드(Snapshot) 시작: {dataset_id}")
    
    try:
        # ---------------------------------------------------------
        # [1] 경로 설정
        # ---------------------------------------------------------
        current_file = Path(__file__).resolve()
        project_root = current_file.parents[3]
        
        # 저장 경로 세분화 (raw 데이터와 변환된 csv 분리)
        save_base_dir = project_root / "datafile" / "original" / "skt" / "telagentbench"
        raw_dir = save_base_dir / "raw"
        csv_dir = save_base_dir / "csv"
        
        raw_dir.mkdir(parents=True, exist_ok=True)
        csv_dir.mkdir(parents=True, exist_ok=True)
        
        # ---------------------------------------------------------
        # [2] Hugging Face 토큰 로드 및 인증
        # ---------------------------------------------------------
        env_path = project_root / "key" / "huggingface.env"
        hf_token = None
        
        if env_path.exists():
            load_dotenv(env_path)
            hf_token = os.getenv("HF_TOKEN")
            
            if not hf_token or hf_token == "your_token_here":
                raise ValueError("유효한 HF_TOKEN이 없습니다. .env 파일을 확인해주세요.")
                
            login(token=hf_token, add_to_git_credential=False)
            api = HfApi()
            user_info = api.whoami(token=hf_token)
            print(f"[INFO] 로그인 사용자: {user_info['name']}\n")
        else:
            raise FileNotFoundError(f"토큰 환경변수 파일을 찾을 수 없습니다: {env_path}")
            
        # ---------------------------------------------------------
        # [3] 레포지토리 전체 스냅샷 다운로드 (PyArrow 우회)
        # ---------------------------------------------------------
        print(f"[INFO] 원본 JSON 파일 다운로드 중... (기존 캐시가 있다면 빠르게 넘어갑니다)")
        snapshot_dir = snapshot_download(
            repo_id=dataset_id,
            repo_type="dataset",
            token=hf_token,
            local_dir=str(raw_dir),
            ignore_patterns=["*.git*", "*.md"] # 불필요한 파일 제외
        )
        print(f"[INFO] 원본 다운로드 완료: {raw_dir}\n")
        
        # ---------------------------------------------------------
        # [4] JSON -> CSV 안전한 변환 (Pandas 활용)
        # ---------------------------------------------------------
        print(f"{'='*70}")
        print(f"  JSON 데이터 CSV 파싱 및 변환 시작")
        print(f"{'='*70}")
        
        # 다운로드된 폴더 순회
        for root, dirs, files in os.walk(snapshot_dir):
            for file in files:
                if file.endswith(".json"):
                    file_path = Path(root) / file
                    # 상위 폴더명을 카테고리로 사용 (예: TelAgent_Plan, TelAgent_IF 등)
                    category = Path(root).name 
                    
                    # 카테고리가 없는 최상위 경로의 JSON이면 default로 지정
                    if category == raw_dir.name:
                        category = "default"
                        
                    print(f"\n  [{category}] {file} 파싱 중...")
                    
                    try:
                        # JSON 파일을 안전하게 로드
                        with open(file_path, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                            
                        # 리스트 형태의 데이터인지 확인 후 DataFrame 생성
                        if isinstance(data, list):
                            df = pd.DataFrame(data)
                        elif isinstance(data, dict):
                            # 만약 딕셔너리로 감싸져 있다면, 내부 리스트를 찾음
                            for key, val in data.items():
                                if isinstance(val, list):
                                    df = pd.DataFrame(val)
                                    break
                            else:
                                df = pd.DataFrame([data]) # 리스트가 없으면 단일 객체로 처리
                        else:
                            print(f"  [WARNING] 지원하지 않는 JSON 구조입니다: {file}")
                            continue
                            
                        # CSV 저장 경로 구성 (카테고리별 폴더 생성)
                        out_folder = csv_dir / category
                        out_folder.mkdir(parents=True, exist_ok=True)
                        
                        out_path = out_folder / file.replace(".json", ".csv")
                        
                        # 인덱스 없이 utf-8-sig(BOM 포함)로 저장하여 엑셀 등에서 한글 깨짐 방지
                        df.to_csv(out_path, index=False, encoding='utf-8-sig')
                        
                        print(f"  ✓ 저장 완료: {out_path.relative_to(project_root)}")
                        print(f"    - 데이터 형태: {len(df)}행 x {len(df.columns)}열")
                        
                    except Exception as e:
                        print(f"  [ERROR] {file} 변환 실패: {str(e)}")
                        
        print(f"\n{'='*70}")
        print(f"  전체 데이터 변환 완료!")
        print(f"{'='*70}")
        print(f"[INFO] 최종 CSV 저장 경로: {csv_dir.relative_to(project_root)}")
        print(f"[INFO] 이제 TextGrad에서 위 경로의 CSV 파일들을 안전하게 불러와 평가에 활용할 수 있습니다.")

    except Exception as e:
        print(f"\n[ERROR] 스크립트 실행 중 오류가 발생했습니다: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("="*70)
    print("  TelAgentBench Dataset Direct Downloader (Bypass PyArrow)")
    print("  - 용도: 제약조건 준수 및 포맷팅(Form) 성능 최적화")
    print("="*70)
    print()
    
    download_and_parse_telagentbench()
    
    print("\n[완료] 스크립트 실행이 종료되었습니다.")