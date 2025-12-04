"""
@경로: model/download_model.py
@설명: Hugging Face에서 한국어 문장 임베딩 모델 다운로드 및 저장
@명령어: python model/download_model.py

- 문장의 의미를 768차원 벡터로 변환하는 임베딩 모델
- 그 벡터를 기반으로 분류·검색·유사도 등 여러 작업을 할 수 있는 모델
"""

import os
from sentence_transformers import SentenceTransformer

class ModelLoader:
    """
    Hugging Face 모델을 로컬 특정 경로에 다운로드 및 저장하는 클래스
    """
    
    def __init__(self):
        # 다운로드할 모델 ID (Hugging Face)
        self.model_id = "jhgan/ko-sroberta-multitask"
        
        # 저장할 로컬 경로 (프로젝트 루트 기준)
        # model/embedding/ko-sroberta-multitask/   
        base_dir = os.path.dirname(os.path.abspath(__file__)) # 현재 파일 위치 (model/)
        project_root = os.path.dirname(base_dir)         # 상위 폴더 (Project Root)     
        
        self.save_path = os.path.join(
            project_root, 
            "model", 
            "embedding", 
            "ko-sroberta-multitask"
        )
        print(f"모델 저장 경로 설정: {self.save_path}")

    def model_download(self):
        """
        모델을 Hugging Face에서 다운로드하여 지정된 경로에 저장합니다.
        이미 존재하면 다운로드를 건너뛸 수 있습니다.
        """
        print(f"다운로드 시작: {self.model_id}")
        print(f"저장 경로: {self.save_path}")

        # 1. 저장 경로 폴더 생성 (없으면 생성)
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path, exist_ok=True)
            print("폴더 생성 완료")
        else:
            # (선택) 이미 폴더가 있고 모델 파일이 있다면 스킵할 수도 있음
            # 여기서는 덮어쓰기 방식으로 진행합니다.
            print("ℹ기존 폴더가 존재합니다. 모델을 확인합니다.")

        try:
            # 2. 모델 로드 (Hugging Face Hub에서 캐시로 다운로드됨)
            print("모델 다운로드 중... (시간이 조금 걸릴 수 있습니다)")
            model = SentenceTransformer(self.model_id)
            
            # 3. 지정된 경로에 저장
            model.save(self.save_path)
            
            print(f"다운로드 및 저장 완료!")
            print(f"사용 시 경로: {self.save_path}")
            return self.save_path

        except Exception as e:
            print(f"다운로드 중 오류 발생: {e}")
            return None

# 실행 예시
if __name__ == "__main__":
    loader = ModelLoader()
    loader.model_download()