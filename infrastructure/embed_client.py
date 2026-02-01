"""
@경로: infrastructure/embed_client.py
@설명: 임베딩 모델 관리 (로컬 모델 + Azure OpenAI 임베딩)
"""

import sys
from pathlib import Path
from conf.config import Settings

class EmbedClient:
    """
    임베딩 클라이언트 통합 관리 클래스
    로컬 모델과 Azure 모델을 통합 관리하고 모델 재사용을 통한 성능 최적화
    """
    
    def __init__(self, use_azure=False, auto_load=True):
        """
        초기화
        
        Args:
            use_azure (bool): True면 Azure, False면 로컬 모델 사용
            auto_load (bool): 초기화 시 자동으로 모델 로드 여부
        """
        self.use_azure = use_azure
        self._local_model = None
        self._azure_client = None
        
        if auto_load:
            self.get_model()  # 미리 로드
    
    def get_model(self):
        """
        설정에 따라 적절한 모델 반환 (캐싱된 모델 재사용)
        
        Returns:
            임베딩 모델 또는 클라이언트
        """
        if self.use_azure:
            return self._get_azure_client()
        else:
            return self._get_local_model()
    
    def _get_local_model(self):
        """로컬 모델 로드 및 캐싱"""
        if self._local_model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError:
                print("[ERROR] sentence-transformers 패키지가 설치되지 않았습니다.")
                print("설치 명령: pip install sentence-transformers")
                sys.exit(1)
            
            # 로컬 모델 경로 설정
            model_path = Path("model/embedding/ko-sroberta-multitask")
            
            # 절대 경로로 변환
            if not model_path.is_absolute():
                model_path = Path(__file__).parent.parent / model_path
            
            if not model_path.exists():
                print(f"[ERROR] 임베딩 모델을 찾을 수 없습니다: {model_path}")
                print("모델 다운로드가 필요할 수 있습니다.")
                sys.exit(1)
            
            try:
                # 로컬 모델 로드 및 캐싱
                self._local_model = SentenceTransformer(str(model_path))
                print(f"[SUCCESS] 로컬 임베딩 모델 로드 완료: {model_path.name}")
            except Exception as e:
                print(f"[ERROR] 로컬 임베딩 모델 로드 실패: {e}")
                sys.exit(1)
        
        return self._local_model
    
    def _get_azure_client(self):
        """Azure 클라이언트 로드 및 캐싱"""
        if self._azure_client is None:
            try:
                from openai import AzureOpenAI
            except ImportError:
                print("[ERROR] openai 패키지가 설치되지 않았습니다.")
                print("설치 명령: pip install openai")
                sys.exit(1)
            
            Settings.setup()
            
            if not Settings.AZURE_OPENAI_API_KEY:
                print("[ERROR] Azure OpenAI API Key가 설정되지 않았습니다.")
                sys.exit(1)
            
            try:
                self._azure_client = AzureOpenAI(
                    api_key=Settings.AZURE_OPENAI_API_KEY,
                    api_version=Settings.AZURE_OPENAI_API_VERSION,
                    azure_endpoint=Settings.AZURE_OPENAI_ENDPOINT
                )
                print("[SUCCESS] Azure 임베딩 클라이언트 로드 완료")
            except Exception as e:
                print(f"[ERROR] Azure 임베딩 클라이언트 생성 실패: {e}")
                sys.exit(1)
        
        return self._azure_client
    
    def calculate_similarity(self, text1, text2):
        """
        두 텍스트 간의 임베딩 기반 코사인 유사도 계산
        
        Args:
            text1 (str): 첫 번째 텍스트
            text2 (str): 두 번째 텍스트
            
        Returns:
            float: 0.0~1.0 사이의 유사도 점수
        """
        if self.use_azure:
            return self._calculate_azure_similarity(text1, text2)
        else:
            return self._calculate_local_similarity(text1, text2)
    
    def _calculate_local_similarity(self, text1, text2):
        """로컬 모델을 사용한 유사도 계산"""
        try:
            from sklearn.metrics.pairwise import cosine_similarity
            import numpy as np
        except ImportError:
            print("[ERROR] scikit-learn 패키지가 설치되지 않았습니다.")
            print("설치 명령: pip install scikit-learn")
            return 0.0
        
        try:
            model = self._get_local_model()
            
            # 두 텍스트를 벡터로 변환
            embeddings = model.encode([text1, text2])
            
            # 코사인 유사도 계산
            embedding1 = np.array(embeddings[0]).reshape(1, -1)
            embedding2 = np.array(embeddings[1]).reshape(1, -1)
            
            similarity = cosine_similarity(embedding1, embedding2)[0][0]
            
            # 음수 값을 0으로 클리핑 (0~1 범위 보장)
            return max(0.0, float(similarity))
            
        except Exception as e:
            print(f"[WARNING] 로컬 임베딩 유사도 계산 실패: {e}")
            return 0.0
    
    def _calculate_azure_similarity(self, text1, text2):
        """Azure 모델을 사용한 유사도 계산"""
        try:
            import numpy as np
            from sklearn.metrics.pairwise import cosine_similarity
        except ImportError:
            print("[ERROR] scikit-learn, numpy 패키지가 설치되지 않았습니다.")
            return 0.0
        
        try:
            client = self._get_azure_client()
            
            # Azure 임베딩 모델명 (설정에서 가져오거나 기본값)
            embedding_model = getattr(Settings, 'AZURE_EMBEDDING3_SMALL_DEPLOYMENT', 'text-embedding-ada-002')
            
            # 두 텍스트의 임베딩 벡터 생성
            response1 = client.embeddings.create(input=[text1], model=embedding_model)
            response2 = client.embeddings.create(input=[text2], model=embedding_model)
            
            # 임베딩 벡터 추출
            embedding1 = np.array(response1.data[0].embedding).reshape(1, -1)
            embedding2 = np.array(response2.data[0].embedding).reshape(1, -1)
            
            # 코사인 유사도 계산
            similarity = cosine_similarity(embedding1, embedding2)[0][0]
            
            return max(0.0, float(similarity))
            
        except Exception as e:
            print(f"[WARNING] Azure 임베딩 유사도 계산 실패: {e}")
            return 0.0
    
    def encode(self, texts):
        """
        텍스트를 임베딩 벡터로 변환
        
        Args:
            texts (str or list): 변환할 텍스트(들)
            
        Returns:
            numpy.ndarray: 임베딩 벡터(들)
        """
        if self.use_azure:
            return self._encode_azure(texts)
        else:
            return self._encode_local(texts)
    
    def _encode_local(self, texts):
        """로컬 모델로 임베딩 생성"""
        model = self._get_local_model()
        return model.encode(texts)
    
    def _encode_azure(self, texts):
        """Azure 모델로 임베딩 생성"""
        try:
            import numpy as np
        except ImportError:
            print("[ERROR] numpy 패키지가 설치되지 않았습니다.")
            return None
        
        client = self._get_azure_client()
        embedding_model = getattr(Settings, 'AZURE_EMBEDDING3_SMALL_DEPLOYMENT')
        
        if isinstance(texts, str):
            texts = [texts]
        
        try:
            response = client.embeddings.create(input=texts, model=embedding_model)
            embeddings = [np.array(data.embedding) for data in response.data]
            return np.array(embeddings)
        except Exception as e:
            print(f"[ERROR] Azure 임베딩 생성 실패: {e}")
            return None


# ================================================================
# 편의를 위한 글로벌 함수들 (기존 코드 호환성)  
# ================================================================

# 전역 인스턴스 (싱글톤 패턴)
_global_local_client = None
_global_azure_client = None

def get_embedding_client(use_azure=False):
    """
    글로벌 임베딩 클라이언트 반환 (싱글톤)
    
    Args:
        use_azure (bool): True면 Azure, False면 로컬
        
    Returns:
        EmbedClient: 임베딩 클라이언트 인스턴스
    """
    global _global_local_client, _global_azure_client
    
    if use_azure:
        if _global_azure_client is None:
            _global_azure_client = EmbedClient(use_azure=True)
        return _global_azure_client
    else:
        if _global_local_client is None:
            _global_local_client = EmbedClient(use_azure=False)
        return _global_local_client

def calculate_embedding_similarity(text1, text2, use_azure=False):
    """
    편의 함수: 두 텍스트 간 유사도 계산
    
    Args:
        text1, text2 (str): 비교할 텍스트들
        use_azure (bool): Azure 사용 여부
        
    Returns:
        float: 유사도 점수 (0.0~1.0)
    """
    client = get_embedding_client(use_azure)
    return client.calculate_similarity(text1, text2)