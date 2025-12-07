"""
@경로: utils/models/model.py
@설명: LLM 및 임베딩 모델 팩토리 클래스
@사용법: azure 를 사용할경우 user_auzre = True 로 설정, 그 외에는 gemini 모델 사용
"""
import os
import time  # 재시도 대기용
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
            "deployment_name": Env.AZURE_GPT5_CHAT_DEPLOYMENT
            # "embedding_name": "text-embedding-3-small"
        }
    
    
    def get_llm(self, model_type="optimizer"):
        logger.info(f"[CHECK] self.use_azure in ModelFactory: {self.use_azure}")
        """설정에 따라 Azure 또는 Gemini 모델을 반환
        
        Args:
            model_type (str): "optimizer" 또는 "target"
            
        Returns:
            LLM instance
        """
        if self.use_azure:
            # Azure OpenAI 사용
            logger.info("Using Azure OpenAI models")
            try:
                from langchain_openai import AzureChatOpenAI
                
                # 모델별 Temperature 설정
                temp = 1.0 if model_type == "optimizer" else 0.7
                
                return AzureChatOpenAI(
                    azure_deployment=self.azure_config["deployment_name"],
                    temperature=temp,
                    api_version="2024-02-15-preview"  # 최신 API 버전
                )
            except ImportError:
                logger.error("langchain_openai not installed. Install with: pip install langchain-openai")
                return None
            except Exception as e:
                logger.error(f"Azure OpenAI initialization failed: {e}")
                return None
        else:
            # Gemini용 임포트
            from langchain_google_genai import ChatGoogleGenerativeAI
            
            # gemini-1.5-flash 대신 gemini-2.5-flash를 기본 모델로 사용
            # Google Generative AI API
            
            # 모델별 Temperature 설정 (optimizer는 더 탐색적, target은 더 안정적)
            temp = 1.0 if model_type == "optimizer" else 0.7
            # gemini-1.5-flash X
            # gemini-1.5-flash-8b X 
            # 최신 무료 Gemini 모델 사용 (2024년 12월 기준)
            # 주요 모델: gemini-2.0-flash-exp (최신 실험 버전)
            primary_model_name = "gemini-2.5-flash-lite"
            
            try:
                logger.info(f"Trying primary Gemini model: {primary_model_name}")
                llm = ChatGoogleGenerativeAI(
                    model=primary_model_name,
                    temperature=temp
                )
                return llm.with_retry(stop_after_attempt=10, wait_exponential_jitter=True)
            except Exception as e:
                logger.error(f"Failed to load primary Gemini model: {e}")

    # def get_embedding_model(self):
    #     """설정에 따라 임베딩 모델 반환
        
    #     Returns:
    #         Embedding instance
    #     """
    #     if self.use_azure:
    #         # Azure OpenAI Embeddings 사용  
    #         logger.info("Using Azure OpenAI Embeddings")
    #         try:
    #             from langchain_openai import AzureOpenAIEmbeddings
    #             return AzureOpenAIEmbeddings(
    #                 azure_deployment=self.azure_config["embedding_name"],
    #                 api_version="2024-02-15-preview"
    #             )
    #         except ImportError:
    #             logger.error("langchain_openai not installed. Install with: pip install langchain-openai")
    #             return None
    #         except Exception as e:
    #             logger.error(f"Azure OpenAI Embeddings initialization failed: {e}")
    #             return None
    #     else:
    #         # 로컬 Ko-SRoBERTa multitask 모델 사용 -- 직접 구현한 래퍼 클래스
    #         logger.info("Using local Ko-SRoBERTa multitask embedding model")
    #         from utils.models.ko_sroberta_multitask import LangChainKoSRoBERTaEmbeddings
    #         return LangChainKoSRoBERTaEmbeddings()

    def get_langchain_embedding_model(self):
        """
        @역할: GPU 가속을 지원하는 LangChain 표준 임베딩 모델 반환
        - torch를 사용하여 GPU(CUDA/MPS)를 자동 감지한다.
        - HuggingFaceEmbeddings 표준 라이브러리를 사용한다.
        """
        
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
            encode_kwargs={'normalize_embeddings': True} # 임베딩 정규화 - 벡터의 길이를 1로 정규화한다. (L2 Normalization)
        )
    
        """
        @Gemini

        임베딩, 코사인 유사도와 관련해서 궁금했던, 제미나이의 답변
        Q. cosine similarity 계산 시, 결과 범위는 어떻게 되는가? ... 와 관련하여 이어진 이야기
        A.
        이해하시기 힘들었던 부분이 정확히 어디인지 알겠습니다. 
        "수학 공식과 현실 데이터의 차이" 때문에 생기는 괴리감입니다.
        수학적으로는 -1까지 갈 수 있는 게 맞지만, 
        우리가 다루는 '글자(Text)'의 세계에서는 -1이 나올 일이 거의 없는 이유를 아주 쉽게, 시계바늘에 비유해서 설명해 드릴게요.
        
        1. 수학적인 정의 (이론): 

        시계바늘의 각도코사인 유사도는 두 벡터(화살표) 사이의 각도를 재는 것입니다.
        1.0 (완전 일치): 시계바늘 두 개가 12시를 가리키며 겹쳐 있음. 
        0.0 (관계 없음): 하나는 12시, 하나는 3시를 가리킴. 
        -1.0 (완전 반대): 하나는 12시, 하나는 6시를 가리킴. 
        수학 공식상으로는 화살표가 정반대 방향으로 갈 수 있으니 -1.0이 나오는 게 가능합니다.
        
        2. 텍스트 임베딩의 현실 (실전): 
        "언어의 세계에는 '반대편'이 없다"
        그런데 우리가 쓰는 **임베딩 모델(HuggingFace 등)**이 만드는 벡터의 세계에서는 
        화살표들이 **모두 한쪽 방향(12시~3시 사이)**으로만 몰려 있습니다.
        왜 그럴까요?
        이유 : "반대말"은 "수학적 반대"가 아니다
        우리는 "사랑"의 반대말이 "미움"이라고 생각하지만, 
        AI가 보기에 이 두 단어는 **"사람의 감정"**이라는 공통된 주제를 가지고 있습니다.
        사랑 vs 미움: 둘 다 감정이고, 사람에게 쓰는 말입니다. 
        유사도가 0.6 정도로 꽤 높게 나옵니다. (안 비슷하지 않습니다!)
        사랑 vs 냉장고: 이건 주제가 아예 다릅니다. 유사도가 0.1 ~ 0.2 정도로 낮게 나옵니다.
        
        AI 입장에서 **-1.0(정반대)**이 나오려면, **"모든 차원에서 의미가 완벽하게 뒤집힌 무언가"**여야 하는데, 
        인간의 언어에는 그런 단어가 존재하지 않습니다. 
        ("사랑"의 수학적 정반대 물질 같은 건 없으니까요.)
        """

