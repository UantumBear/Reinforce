"""
Docstring for utils.datasets.baseline.gsm8k_download_datasets

@경로: utils/datasets/baseline/gsm8k_download_datasets.py
@설명: GSM8k 데이터셋을 다운로드하여 프로젝트의 datafile/original 폴더에 저장하는 전용 소스 코드
@데이터셋: openai/gsm8k (public dataset)
@Config: main, socratic
@용도: 프롬프트 최적화 (Prompt Optimization) - TextGrad 논문

@명령어: python utils/datasets/baseline/gsm8k_download_datasets.py

@참고:
- GSM8k: Grade School Math 8K
- 초등학교 수준의 수학 문제 풀이 데이터셋
- train: 7473개, test: 1319개
"""

import os
from pathlib import Path
from datasets import load_dataset
import pandas as pd
from dotenv import load_dotenv
from huggingface_hub import HfApi, login


def download_gsm8k_dataset():
    """
    GSM8k 데이터셋을 다운로드하고 CSV 형식으로 저장합니다.
    
    GSM8k는 2개의 config를 제공합니다:
    - main: 기본 데이터셋 (일반적으로 사용)
    - socratic: 소크라테스식 질문-답변 형식
    """
    
    dataset_id = "openai/gsm8k"
    
    # 다운로드할 config 리스트
    configs = [
        'main',         # 일반적으로 사용되는 기본 데이터셋
        # 'socratic',   # 필요시 주석 해제
    ]
    
    print(f"[INFO] GSM8k 데이터셋 다운로드 시작: {dataset_id}")
    print(f"[INFO] 다운로드할 Config: {', '.join(configs)}\n")
    
    try:
        # ---------------------------------------------------------
        # [1] 경로 설정
        # ---------------------------------------------------------
        # 현재 파일 위치: .../Reinforce/utils/datasets/baseline/gsm8k_download_datasets.py
        current_file = Path(__file__).resolve()
        
        # 프로젝트 루트 찾기 (4단계 위로: 파일 -> baseline폴더 -> datasets폴더 -> utils폴더 -> Reinforce루트)
        project_root = current_file.parents[3]
        
        # 목표 저장 경로: .../Reinforce/datafile/original/openai/gsm8k
        save_base_dir = project_root / "datafile" / "original" / "openai" / "gsm8k"
        save_base_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"[INFO] 저장 기본 경로: {save_base_dir}\n")
        
        # ---------------------------------------------------------
        # [2] Hugging Face 토큰 로드 (선택적)
        # ---------------------------------------------------------
        # GSM8k는 public dataset이므로 토큰 없이도 다운로드 가능하지만,
        # rate limit 방지를 위해 토큰 사용 권장
        env_path = project_root / "key" / "huggingface.env"
        hf_token = None
        
        if env_path.exists():
            load_dotenv(env_path)
            hf_token = os.getenv("HF_TOKEN")
            
            if hf_token and hf_token != "your_token_here":
                print("[INFO] Hugging Face 토큰 로드 완료 ✓")
                try:
                    login(token=hf_token, add_to_git_credential=False)
                    api = HfApi()
                    user_info = api.whoami(token=hf_token)
                    print(f"[INFO] 로그인 사용자: {user_info['name']}\n")
                except Exception as e:
                    print(f"[WARNING] 토큰 인증 실패 (토큰 없이 다운로드 시도): {e}\n")
                    hf_token = None
            else:
                print("[INFO] 토큰 없이 다운로드 시도 (public dataset)\n")
        else:
            print("[INFO] 토큰 없이 다운로드 시도 (public dataset)\n")
        
        # ---------------------------------------------------------
        # [3] 각 Config별 데이터 다운로드 및 저장
        # ---------------------------------------------------------
        for config_name in configs:
            print(f"{'='*70}")
            print(f"  Config: {config_name}")
            print(f"{'='*70}")
            
            try:
                # 데이터셋 다운로드 (캐시 활용)
                print(f"[INFO] 다운로드 중... (시간이 걸릴 수 있습니다)")
                
                if hf_token:
                    dataset = load_dataset(dataset_id, config_name, token=hf_token)
                else:
                    dataset = load_dataset(dataset_id, config_name)
                
                if not dataset or len(dataset.keys()) == 0:
                    print(f"[ERROR] 데이터셋이 비어있습니다!")
                    continue
                
                print(f"[INFO] ✓ 다운로드 성공!")
                print(f"[INFO] Split 목록: {list(dataset.keys())}")
                
                # Config별 저장 폴더
                config_save_dir = save_base_dir / config_name
                config_save_dir.mkdir(parents=True, exist_ok=True)
                
                # 각 split별로 저장 (train, test)
                for split_name in dataset.keys():
                    split_data = dataset[split_name]
                    data_count = len(split_data)
                    
                    print(f"\n  [{split_name}] 저장 중... (총 {data_count}개)")
                    
                    # CSV 파일로 저장
                    output_path = config_save_dir / f"{split_name}.csv"
                    df = pd.DataFrame(split_data)
                    df.to_csv(output_path, index=False, encoding='utf-8-sig')
                    
                    print(f"  ✓ 저장 완료: {output_path.relative_to(project_root)}")
                    
                    # 데이터 샘플 정보 출력
                    if len(df) > 0:
                        print(f"    - 컬럼: {list(df.columns)}")
                        print(f"    - 샘플 수: {len(df)}개")
                        
                        # 첫 번째 샘플 미리보기
                        if len(df) > 0:
                            print(f"\n    [샘플 미리보기]")
                            first_row = df.iloc[0]
                            for col in df.columns:
                                value = str(first_row[col])[:100]  # 100자까지만
                                print(f"      {col}: {value}...")
                
                print(f"\n[SUCCESS] '{config_name}' 다운로드 완료!\n")
                
            except Exception as e:
                print(f"\n[ERROR] '{config_name}' 다운로드 실패!")
                print(f"[ERROR] 에러 상세: {str(e)}")
                print(f"\n[HINT] 가능한 원인:")
                print(f"  1. 인터넷 연결을 확인하세요")
                print(f"  2. Hugging Face 서버 상태를 확인하세요")
                print(f"  3. 데이터셋 URL: https://huggingface.co/datasets/{dataset_id}")
                continue
        
        print(f"\n{'='*70}")
        print(f"  전체 다운로드 완료!")
        print(f"{'='*70}")
        print(f"[INFO] 저장 경로: {save_base_dir.relative_to(project_root)}")
        print(f"[INFO] TextGrad 프롬프트 최적화 실험에 사용할 수 있습니다.")
        
    except KeyboardInterrupt:
        print("\n[INFO] 사용자가 다운로드를 중단했습니다.")
    except Exception as e:
        print(f"\n[ERROR] 예상치 못한 오류 발생: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    print("="*70)
    print("  GSM8k Dataset Downloader")
    print("  - 용도: 프롬프트 최적화 (Prompt Optimization)")
    print("  - TextGrad 논문 재현용")
    print("="*70)
    print()
    
    download_gsm8k_dataset()
    
    print("\n[완료] 스크립트 실행이 종료되었습니다.")
