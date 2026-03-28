"""
Docstring for utils.datasets.baseline.gpqa_download_datasets

@경로: utils/datasets/baseline/gpqa_download_datasets.py
@설명: GPQA 데이터셋을 다운로드하여 프로젝트의 datasets/original 폴더에 저장하는 전용 소스 코드
@데이터셋: Idavidrein/gpqa (gated dataset)
@Config: gpqa_diamond, gpqa_main, gpqa_extended, gpqa_experts

@명령어: python utils/datasets/baseline/gpqa_download_datasets.py

"""

import os
from pathlib import Path
from datasets import load_dataset
import pandas as pd
from dotenv import load_dotenv
from huggingface_hub import HfApi, login


def download_gpqa_dataset():
    """
    GPQA 데이터셋을 다운로드하고 CSV 형식으로 저장합니다.
    
    GPQA는 4개의 config를 제공합니다:
    - gpqa_diamond: 가장 높은 품질의 문제 (448개)
    - gpqa_main: 메인 데이터셋
    - gpqa_extended: 확장 버전
    - gpqa_experts: 전문가 검증 버전
    """
    
    dataset_id = "Idavidrein/gpqa"
    
    # 다운로드할 config 리스트 (필요한 것만 선택 가능)
    configs = [
        'gpqa_diamond',
        'gpqa_main', 
        'gpqa_extended',
        # 'gpqa_experts'  # 필요시 주석 해제
    ]
    
    print(f"[INFO] GPQA 데이터셋 다운로드 시작: {dataset_id}")
    print(f"[INFO] 다운로드할 Config: {', '.join(configs)}\n")
    
    try:
        # ---------------------------------------------------------
        # [1] 경로 설정
        # ---------------------------------------------------------
        # 현재 파일 위치: .../Reinforce/utils/datasets/baseline/gpqa_download_datasets.py
        current_file = Path(__file__).resolve()
        
        # 프로젝트 루트 찾기 (4단계 위로: 파일 -> baseline폴더 -> datasets폴더 -> utils폴더 -> Reinforce루트)
        project_root = current_file.parents[3]
        
        # 목표 저장 경로: .../Reinforce/datafile/original/Idavidrein/gpqa
        save_base_dir = project_root / "datafile" / "original" / "Idavidrein" / "gpqa"
        save_base_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"[INFO] 저장 기본 경로: {save_base_dir}\n")
        
        # ---------------------------------------------------------
        # [2] Hugging Face 토큰 로드 및 인증
        # ---------------------------------------------------------
        env_path = project_root / "key" / "huggingface.env"
        load_dotenv(env_path)
        hf_token = os.getenv("HF_TOKEN")
        
        if not hf_token or hf_token == "your_token_here":
            print("[ERROR] Hugging Face 토큰이 설정되지 않았습니다!")
            print(f"[INFO] {env_path} 파일에 실제 토큰을 입력하세요.")
            print("[INFO] 토큰 발급: https://huggingface.co/settings/tokens")
            print("[INFO] GPQA 접근 권한 요청: https://huggingface.co/datasets/Idavidrein/gpqa")
            return
        
        print("[INFO] Hugging Face 토큰 로드 완료 ✓")
        
        # Hugging Face에 로그인 시도
        try:
            print("[INFO] Hugging Face 로그인 중...")
            login(token=hf_token, add_to_git_credential=False)
            print("[INFO] Hugging Face 로그인 성공 ✓")
            
            # 토큰 유효성 및 데이터셋 접근 권한 확인
            api = HfApi()
            user_info = api.whoami(token=hf_token)
            print(f"[INFO] 로그인 사용자: {user_info['name']}\n")
            
        except Exception as e:
            print(f"[ERROR] Hugging Face 로그인 실패: {e}")
            print("[INFO] 토큰이 유효한지 확인하세요.")
            print("[INFO] GPQA 데이터셋 접근 권한을 요청했는지 확인하세요:")
            print("       https://huggingface.co/datasets/Idavidrein/gpqa")
            return
        
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
                dataset = load_dataset(
                    dataset_id, 
                    config_name, 
                    token=hf_token,
                    trust_remote_code=True  # 필요한 경우 원격 코드 신뢰
                )
                
                if not dataset or len(dataset.keys()) == 0:
                    print(f"[ERROR] 데이터셋이 비어있습니다!")
                    continue
                
                print(f"[INFO] ✓ 다운로드 성공!")
                print(f"[INFO] Split 목록: {list(dataset.keys())}")
                
                # Config별 저장 폴더
                config_save_dir = save_base_dir / config_name
                config_save_dir.mkdir(parents=True, exist_ok=True)
                
                # 각 split별로 저장 (train, validation, test 등)
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
                
                print(f"\n[SUCCESS] '{config_name}' 다운로드 완료!\n")
                
            except Exception as e:
                print(f"\n[ERROR] '{config_name}' 다운로드 실패!")
                print(f"[ERROR] 에러 상세: {str(e)}")
                print(f"\n[HINT] 가능한 원인:")
                print(f"  1. GPQA 데이터셋 접근 권한이 승인되지 않았을 수 있습니다")
                print(f"     -> https://huggingface.co/datasets/Idavidrein/gpqa 에서 'Access repository' 확인")
                print(f"  2. 토큰 권한이 'Read' 권한을 포함하는지 확인")
                print(f"  3. 인터넷 연결 확인")
                print(f"  4. 방화벽/프록시 설정 확인\n")
                
                import traceback
                traceback.print_exc()
                continue
        
        print(f"\n{'='*70}")
        print(f"[SUCCESS] 모든 GPQA 데이터셋 다운로드가 완료되었습니다!")
        print(f"[INFO] 저장 위치: {save_base_dir.relative_to(project_root)}")
        print(f"{'='*70}")
        
    except Exception as e:
        print(f"\n[ERROR] 전체 프로세스 오류 발생:\n{e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    download_gpqa_dataset()
