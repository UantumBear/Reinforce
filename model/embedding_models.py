"""
@경로: model/embedding_models.py
@설명: 임베딩 모델 설정 및 다운로드 관리
- 사용 가능한 임베딩 모델들의 메타데이터와 다운로드 기능을 제공합니다.
- 단독 실행 용 파일 입니다.
"""

import os
from dataclasses import dataclass
from typing import Dict, Optional
from sentence_transformers import SentenceTransformer

@dataclass
class EmbeddingModelInfo:
    """임베딩 모델 정보를 담는 데이터 클래스"""
    model_id: str           # Hugging Face 모델 ID
    name: str              # 사용자 친화적 이름
    language: str          # 지원 언어
    dimensions: int        # 임베딩 차원
    description: str       # 모델 설명
    local_path: str        # 로컬 저장 경로

class EmbeddingModelManager:
    """임베딩 모델 관리 클래스"""
    
    def __init__(self):
        # 프로젝트 루트 경로 설정
        base_dir = os.path.dirname(os.path.abspath(__file__))  # model/
        project_root = os.path.dirname(base_dir)               # 프로젝트 루트
        self.embedding_base_path = os.path.join(project_root, "model", "embedding")
        
        # 사용 가능한 모델들 정의
        self.models: Dict[str, EmbeddingModelInfo] = {
            # 한국어 특화 모델
            "korean": EmbeddingModelInfo(
                model_id="jhgan/ko-sroberta-multitask",
                name="Korean SRoBERTa MultiTask",
                language="Korean",
                dimensions=768,
                description="한국어 특화 고성능 임베딩 모델",
                local_path=os.path.join(self.embedding_base_path, "ko-sroberta-multitask")
            ),
            
            # 다국어 모델 
            "multilingual": EmbeddingModelInfo(
                model_id="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
                name="Multilingual MiniLM",
                language="Multilingual (Korean supported)",
                dimensions=384,
                description="다국어 지원 경량 모델 (한국어 포함)",
                local_path=os.path.join(self.embedding_base_path, "paraphrase-multilingual-MiniLM-L12-v2")
            ),
            
            # 영어 특화 고성능 모델
            "english": EmbeddingModelInfo(
                model_id="sentence-transformers/all-MiniLM-L6-v2",
                name="English MiniLM",
                language="English",
                dimensions=384,
                description="영어 특화 경량 고속 모델",
                local_path=os.path.join(self.embedding_base_path, "all-MiniLM-L6-v2")
            ),
            
            # 고성능 다국어 모델
            "multilingual_large": EmbeddingModelInfo(
                model_id="sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
                name="Multilingual MPNet",
                language="Multilingual (Korean supported)",
                dimensions=768,
                description="고성능 다국어 모델 (한국어 포함, 큰 용량)",
                local_path=os.path.join(self.embedding_base_path, "paraphrase-multilingual-mpnet-base-v2")
            )
        }
        
        # 기본 모델 설정 (한국어 프로젝트이므로 다국어 모델 사용)
        self.default_model = "multilingual"
    
    def list_models(self):
        """사용 가능한 모델 목록 출력"""
        print("\n=== 사용 가능한 임베딩 모델 목록 ===")
        for key, model_info in self.models.items():
            print(f"\n[{key}] {model_info.name}")
            print(f"  - 언어: {model_info.language}")
            print(f"  - 차원: {model_info.dimensions}")
            print(f"  - 설명: {model_info.description}")
            print(f"  - 모델ID: {model_info.model_id}")
        print("\n" + "="*50)
    
    def download_model(self, model_key: str = None) -> Optional[str]:
        """
        지정된 모델을 다운로드합니다.
        
        Args:
            model_key: 다운로드할 모델 키 (없으면 기본 모델)
            
        Returns:
            모델 로컬 경로 (실패시 None)
        """
        if model_key is None:
            model_key = self.default_model
            
        if model_key not in self.models:
            print(f"[FAIL] 모델 '{model_key}'를 찾을 수 없습니다.")
            self.list_models()
            return None
            
        model_info = self.models[model_key]
        
        # 기존 모델 존재 여부 확인
        if os.path.exists(model_info.local_path) and os.listdir(model_info.local_path):
            print(f"\n[SUCCESS] 이미 모델이 존재합니다: {model_info.name}")
            print(f"   경로: {model_info.local_path}")
            print("   다운로드를 건너뜁니다.")
            return model_info.local_path
        
        print(f"\n[INFO] 모델 다운로드 시작: {model_info.name}")
        print(f"   모델 ID: {model_info.model_id}")
        print(f"   저장 경로: {model_info.local_path}")
        
        # 저장 경로 생성
        os.makedirs(model_info.local_path, exist_ok=True)
        
        try:
            # 모델 다운로드 (Hugging Face Hub에서)
            print("   다운로드 중... (시간이 소요될 수 있습니다)")
            model = SentenceTransformer(model_info.model_id)
            
            # 로컬 경로에 저장
            model.save(model_info.local_path)
            
            print(f"[SUCCESS] 다운로드 완료!")
            print(f"   사용법: get_model('{model_key}') 또는 load_model_from_path('{model_info.local_path}')")
            
            return model_info.local_path
            
        except Exception as e:
            print(f"[FAIL] 다운로드 실패: {e}")
            return None
    
    def get_model_info(self, model_key: str) -> Optional[EmbeddingModelInfo]:
        """모델 정보 반환"""
        return self.models.get(model_key)
    
    def get_model(self, model_key: str = None) -> Optional[SentenceTransformer]:
        """
        모델을 로드해서 반환합니다 (로컬에 없으면 다운로드)
        
        Args:
            model_key: 로드할 모델 키 (없으면 기본 모델)
            
        Returns:
            SentenceTransformer 모델 객체
        """
        if model_key is None:
            model_key = self.default_model
            
        model_info = self.get_model_info(model_key)
        if not model_info:
            print(f"[FAIL] 모델 '{model_key}'를 찾을 수 없습니다.")
            return None
        
        # 로컬에 모델이 있는지 확인
        if os.path.exists(model_info.local_path) and os.listdir(model_info.local_path):
            print(f"[INFO] 로컬 모델 로드: {model_info.name}")
            return SentenceTransformer(model_info.local_path)
        else:
            print(f"[INFO] 모델이 로컬에 없습니다. 다운로드합니다...")
            path = self.download_model(model_key)
            if path:
                return SentenceTransformer(path)
            return None

def load_model_from_path(model_path: str) -> Optional[SentenceTransformer]:
    """지정된 경로에서 모델을 직접 로드"""
    try:
        return SentenceTransformer(model_path)
    except Exception as e:
        print(f"[FAIL] 모델 로드 실패 ({model_path}): {e}")
        return None

# 전역 모델 매니저 인스턴스
model_manager = EmbeddingModelManager()

# 편의 함수들
def list_available_models():
    """사용 가능한 모델 목록 출력"""
    model_manager.list_models()

def download_embedding_model(model_key: str = None) -> Optional[str]:
    """임베딩 모델 다운로드"""
    return model_manager.download_model(model_key)

def get_embedding_model(model_key: str = None) -> Optional[SentenceTransformer]:
    """임베딩 모델 가져오기 (없으면 다운로드)"""
    return model_manager.get_model(model_key)

if __name__ == "__main__":
    # 명령행에서 실행시 기본 동작
    import sys
    
    if len(sys.argv) > 1:
        model_key = sys.argv[1]
        print(f"모델 '{model_key}' 다운로드를 시작합니다...")
        download_embedding_model(model_key)
    else:
        list_available_models()
        print("\n사용법:")
        print("  python model/embedding_models.py                    # 모델 목록 확인")
        print("  python model/embedding_models.py multilingual       # 특정 모델 다운로드")
        print("  python model/embedding_models.py korean            # 한국어 모델 다운로드")