"""
@경로: utils/models/ko_sroberta_multitask.py
@설명: 한국어 SRoBERTa multitask 모델을 위한 임베딩 클래스
@사용법: 로컬 ko-sroberta-multitask 모델을 로드하여 임베딩 생성
"""

import os
from typing import List
from pathlib import Path
from utils.log.logging import logger

class KoSRoBERTaMultitaskEmbeddings:
    """한국어 SRoBERTa multitask 임베딩 클래스"""
    
    def __init__(self, model_path: str = None):
        """
        Ko-SRoBERTa multitask 모델 초기화
        
        Args:
            model_path (str): 모델 경로. None인 경우 기본 경로 사용
        """
        try:
            from sentence_transformers import SentenceTransformer
            
            if model_path is None:
                # 기본 모델 경로 설정
                current_dir = Path(__file__).parent.parent.parent
                model_path = current_dir / "model" / "embedding" / "ko-sroberta-multitask"
            
            self.model_path = str(model_path)
            
            # 모델 로드
            logger.info(f"Loading Ko-SRoBERTa multitask model from: {self.model_path}")
            self.model = SentenceTransformer(self.model_path)
            logger.info("Ko-SRoBERTa multitask model loaded successfully")
            
        except ImportError as e:
            logger.error("sentence-transformers library not found. Please install it: pip install sentence-transformers")
            raise ImportError("sentence-transformers library is required for Ko-SRoBERTa embeddings") from e
        except Exception as e:
            logger.error(f"Failed to load Ko-SRoBERTa multitask model: {str(e)}")
            raise RuntimeError(f"Failed to load Ko-SRoBERTa multitask model from {self.model_path}") from e
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        문서들에 대한 임베딩 생성
        
        Args:
            texts (List[str]): 임베딩할 텍스트 리스트
            
        Returns:
            List[List[float]]: 각 텍스트에 대한 임베딩 벡터 리스트
        """
        try:
            logger.debug(f"Generating embeddings for {len(texts)} documents")
            embeddings = self.model.encode(texts, convert_to_tensor=False, show_progress_bar=False)
            return embeddings.tolist()
        except Exception as e:
            logger.error(f"Failed to generate embeddings for documents: {str(e)}")
            raise RuntimeError("Failed to generate document embeddings") from e
    
    def embed_query(self, text: str) -> List[float]:
        """
        단일 쿼리에 대한 임베딩 생성
        
        Args:
            text (str): 임베딩할 텍스트
            
        Returns:
            List[float]: 임베딩 벡터
        """
        try:
            logger.debug("Generating embedding for query")
            embedding = self.model.encode([text], convert_to_tensor=False, show_progress_bar=False)
            return embedding[0].tolist()
        except Exception as e:
            logger.error(f"Failed to generate embedding for query: {str(e)}")
            raise RuntimeError("Failed to generate query embedding") from e
    
    def get_embedding_dimension(self) -> int:
        """
        임베딩 차원 수 반환
        
        Returns:
            int: 임베딩 벡터의 차원 수
        """
        return self.model.get_sentence_embedding_dimension()
    
    def encode(self, texts: List[str] | str, **kwargs) -> List[List[float]] | List[float]:
        """
        텍스트(들)을 임베딩으로 변환 (sentence-transformers 스타일 API)
        
        Args:
            texts: 단일 텍스트 또는 텍스트 리스트
            **kwargs: 추가 인자들
            
        Returns:
            임베딩 벡터 또는 임베딩 벡터 리스트
        """
        if isinstance(texts, str):
            return self.embed_query(texts)
        else:
            return self.embed_documents(texts)

    def similarity(self, embeddings1: List[List[float]], embeddings2: List[List[float]]) -> List[List[float]]:
        """
        두 임베딩 집합 간의 유사도 계산
        
        Args:
            embeddings1: 첫 번째 임베딩 집합
            embeddings2: 두 번째 임베딩 집합
            
        Returns:
            유사도 행렬
        """
        try:
            import numpy as np
            from sklearn.metrics.pairwise import cosine_similarity
            
            similarity_matrix = cosine_similarity(embeddings1, embeddings2)
            return similarity_matrix.tolist()
        except ImportError as e:
            logger.error("numpy and scikit-learn are required for similarity calculation")
            raise ImportError("numpy and scikit-learn are required for similarity calculation") from e
        except Exception as e:
            logger.error(f"Failed to calculate similarity: {str(e)}")
            raise RuntimeError("Failed to calculate similarity") from e


class LangChainKoSRoBERTaEmbeddings:
    """LangChain 호환 Ko-SRoBERTa 임베딩 래퍼 클래스"""
    
    def __init__(self, model_path: str = None):
        """
        LangChain 호환 Ko-SRoBERTa 임베딩 초기화
        
        Args:
            model_path (str): 모델 경로. None인 경우 기본 경로 사용
        """
        self.embeddings = KoSRoBERTaMultitaskEmbeddings(model_path)
        logger.info("LangChain compatible Ko-SRoBERTa embeddings initialized")
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """LangChain BaseEmbeddings 인터페이스 구현 - 문서 임베딩"""
        return self.embeddings.embed_documents(texts)
    
    def embed_query(self, text: str) -> List[float]:
        """LangChain BaseEmbeddings 인터페이스 구현 - 쿼리 임베딩"""
        return self.embeddings.embed_query(text)
    
    async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
        """LangChain BaseEmbeddings 인터페이스 구현 - 비동기 문서 임베딩"""
        return self.embed_documents(texts)
    
    async def aembed_query(self, text: str) -> List[float]:
        """LangChain BaseEmbeddings 인터페이스 구현 - 비동기 쿼리 임베딩"""
        return self.embed_query(text)
