"""
@경로: utils/models/model.py
@설명: LLM 및 임베딩 모델 팩토리 클래스
@사용법: azure 를 사용할경우 user_auzre = True 로 설정, 그 외에는 gemini 모델 사용
"""
import os
import torch  # GPU 감지용
from langchain_huggingface import HuggingFaceEmbeddings 
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from conf.config import Env
from utils.log.logging import logger

class ModelFactory:
    """LLM 및 임베딩 모델을 생성하는 팩토리 클래스"""
    
    def __init__(self):
        """모델 팩토리 초기화
        
        Args:
            use_azure (bool): Azure 사용 여부
        """
        self.use_azure = Env.USE_AZURE
        self.azure_config = {
            "deployment_name": "gpt-4",
            "embedding_name": "text-embedding-3-small"
        }
    
    def get_llm(self, model_type="optimizer"):
        """설정에 따라 Azure 또는 Gemini 모델을 반환
        
        Args:
            model_type (str): "optimizer" 또는 "target"
            
        Returns:
            LLM instance
        """
        if self.use_azure:
            # Azure용 임포트
            logger.info("Using Azure OpenAI models")
            # from langchain_openai import AzureChatOpenAI
            
            # temp = 1.0 if model_type == "optimizer" else 0.7
            # return AzureChatOpenAI(
            #     azure_deployment=self.azure_config["deployment_name"],
            #     temperature=temp
            # )
        else:
            # Gemini용 임포트
            from langchain_google_genai import ChatGoogleGenerativeAI
            
            # gemini-1.5-flash 대신 gemini-2.5-flash를 기본 모델로 사용
            # Google Generative AI API
            
            # 모델별 Temperature 설정 (optimizer는 더 탐색적, target은 더 안정적)
            temp = 1.0 if model_type == "optimizer" else 0.7
            
            # 주요 모델: gemini-2.5-flash 시도
            primary_model_name = "gemini-2.5-flash"
            # 대체 모델: gemini-1.0-pro 시도 (무료 티어)
            fallback_model_name = "gemini-1.0-pro"
            
            try:
                logger.info(f"Trying primary Gemini model: {primary_model_name}")
                return ChatGoogleGenerativeAI(
                    model=primary_model_name,
                    temperature=temp
                )
            except Exception as e:
                logger.warning(f"{primary_model_name} failed ({e.__class__.__name__}). Trying fallback model: {fallback_model_name}")
                # 대체 모델 시도
                try:
                    return ChatGoogleGenerativeAI(
                        model=fallback_model_name,
                        temperature=temp
                    )
                except Exception as e_fallback:
                    logger.error(f"Both {primary_model_name} and {fallback_model_name} failed. Check your API key and model availability. Error: {e_fallback}")
                    # 모델 인스턴스 반환 대신 None을 반환하거나 예외를 다시 발생시킬 수 있다.
                    # 여기서는 에러를 출력하고 None을 반환한다.
                    return None 

    def get_embedding_model(self):
        """설정에 따라 임베딩 모델 반환
        
        Returns:
            Embedding instance
        """
        if self.use_azure:
            logger.info("Using Azure OpenAI models")
            # from langchain_openai import AzureOpenAIEmbeddings
            # return AzureOpenAIEmbeddings(azure_deployment=self.azure_config["embedding_name"])
        else:
            # 로컬 Ko-SRoBERTa multitask 모델 사용 -- 이건 직접 구현한 래퍼 클래스
            logger.info("Using local Ko-SRoBERTa multitask embedding model")
            from utils.models.ko_sroberta_multitask import LangChainKoSRoBERTaEmbeddings
            return LangChainKoSRoBERTaEmbeddings()

    def get_langchain_embedding_model(self):
        """
        [새로운 함수] GPU 가속을 지원하는 LangChain 표준 임베딩 모델 반환
        - torch를 사용하여 GPU(CUDA/MPS)를 자동 감지합니다.
        - HuggingFaceEmbeddings 표준 라이브러리를 사용합니다.
        """
        if self.use_azure:
            logger.info("Using Azure OpenAI models")
            return None
        else:
            # 1. 모델 경로 설정
            # 로컬 경로 우선 확인, 없으면 허깅페이스 Hub 사용
            local_model_path = os.path.join(os.getcwd(), "model", "embedding", "ko-sroberta-multitask")
            
            if os.path.exists(local_model_path) and os.listdir(local_model_path):
                model_name_or_path = local_model_path
                logger.info(f"[INFO] Found local model at: {local_model_path}")
            else:
                model_name_or_path = "jhgan/ko-sroberta-multitask"
                logger.info(f"[INFO] Local model not found. Using HuggingFace Hub: {model_name_or_path}")

            # 2. 하드웨어 가속 자동 감지 (분기 처리)
            if torch.cuda.is_available():
                device = "cuda"
                logger.info(f"[INFO] GPU(CUDA) detected! Embedding will be FAST.")
            elif torch.backends.mps.is_available():
                device = "mps"
                logger.info(f"[INFO] Apple Silicon(MPS) detected! Embedding will be FAST.")
            else:
                device = "cpu"
                logger.info(f"[INFO] No GPU detected. Embedding will run on CPU.")

            # 3. LangChain 표준 모델 반환
            return HuggingFaceEmbeddings(
                model_name=model_name_or_path,
                model_kwargs={'device': device},       # GPU 사용
                encode_kwargs={'normalize_embeddings': True}
            )