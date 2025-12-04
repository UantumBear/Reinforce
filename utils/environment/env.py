"""
@경로 : utils/environment/env.py
@설명: 프롬프트 최적화 Environment class
"""
import time
from typing import Tuple, Dict
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
# from langchain_core.prompts import PromptTemplate
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from conf.config import Env
from utils.models.model import ModelFactory # LLM 모델과 임베딩 모델 관리
from utils.datasets.klue_mrc import KlueMrcDataset # 데이터셋 관리
from utils.datasets.aihub_llm_development_qa import AihubLlmDevelopmentQaDataset
from utils.datasets.korquad import KorQuADDataset


class PromptOptimizationEnv:
    """
        환경(Environment) 클래스
        
        [역할]
        1. Agent가 만든 프롬프트를 받는다.
        2. 그 프롬프트를 Target LLM(챗봇)에게 '시스템 프롬프트'로 장착시킨다. 
        3. Target LLM에게 문제를 풀게 한다.
        4. 정답과 비교하여 점수(Reward)를 매겨 Agent에게 돌려준다.
    """
    
    def __init__(self):
        """환경 초기화
        @설명 :
            - LLM 모델 (Target LLM) 과 임베딩 모델을 초기화
            - 데이터셋을 로드
        """

        self.use_azure = Env.USE_AZURE # LLM 모델, 임베딩 모델에서 azure 를 사용할 지 여부 (... 과제 제출용으로 사용하지 않음.)
        # azure 는 개인적으로 공부할 때 사용하고 있었는데요, 과제 제출 시에는 실행 방법을 명시하라고 적혀있어서,
        # USE_AZURE 변수는 False로 설정하고, 구글 무료티어로 바로 사용가능 한 모델을 사용하도록 하였습니다.
        # embedding 모델도 마찬가지로, api 가 아닌, 로컬에서 실행가능한 모델로 설정하였습니다. 

        # ModelFactory 인스턴스 생성
        self.model_factory = ModelFactory()
        # self.dataset = KlueMrcDataset(split="train") # KLUE MRC 데이터셋 로드 (학습용)
        # self.dataset = AihubLlmDevelopmentQaDataset(split="train") # AIHub LLM 개발용 QA 데이터셋 로드 (학습용)
        self.dataset = KorQuADDataset(split="train") # KorQuAD 데이터셋 로드 (학습용)

        self.target_llm = self.model_factory.get_llm(model_type="target") # LLM 모델 (실제, 응답-답변 생성에 쓰이는 테스트 모델)
        # self.embedding_model = self.model_factory.get_embedding_model()   # embdding 모델 # 직접 선언한 래퍼클래스 Ko-SRoBERTa multitask 모델
        self.embedding_model = self.model_factory.get_langchain_embedding_model() # langchain 임베딩 모델 (로컬 Ko-SRoBERTa multitask 모델)

    def step(self, action_prompt: str) -> Tuple[str, float, Dict]:
        """
        환경에서 한 스텝 실행
        
        Args:
            action_prompt (str): 실행할 프롬프트
            
        Returns:
            Tuple[str, float, Dict]: 예측값, 보상, 추가 정보
        """
        if not self.use_azure: 
            time.sleep(1)  # 무료 버전 rate limit 방지

        total_reward = 0
        batch_size = 3 # 데이터셋에서 3개만 랜덤으로 뽑아서 테스트 (Mini-batch)        
        batch_samples = self.dataset.get_random_samples(batch_size)

        worst_case = {}
        min_score = 2.0 # 코사인 유사도 최대값은 1.0이므로 그보다 큰 값으로 초기화
        last_prediction = "" # 로깅용 대표 예측값
        batch_details = []  # 배치별 상세 정보 저장용

        #  배치 평가 루프

        for q, a, c in batch_samples:
            
            # Target LLM 프롬프트 구성 변경 (RAG 시뮬레이션)
            # System: Agent가 만든 최적화된 프롬프트 (말투, 형식 지시 등)
            # User: "지문(Context)"을 주고, 이걸 보고 답하라고 지시
            target_chain_prompt = ChatPromptTemplate.from_messages([
                ("system", action_prompt),
                ("user", """
다음 [참고 문서]의 내용을 바탕으로 질문에 답변하세요.

[참고 문서]
{context}

[질문]
{question}
""")
            ])

            # 체인 연결
            chain = target_chain_prompt | self.target_llm | StrOutputParser()
            
            try:
                # [수정 5] invoke 할 때 context 변수(c)도 같이 넘겨줌
                pred = chain.invoke({"question": q, "context": c})
            except Exception as e:
                print(f"[Error] Target LLM fail: {e}")
                pred = "Error generating response"

            # --- 이 아래는 점수 계산 로직 (기존과 동일) ---
            
            try:
                embeddings = self.embedding_model.embed_documents([a, pred])
                score = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
            except Exception:
                score = 0.0

            total_reward += score
            last_prediction = pred

            sample_info = {
                "question": q,
                "reference": a,
                "prediction": pred,
                "score": score,
            }
            batch_details.append(sample_info)

            if score < min_score:
                min_score = score
                worst_case = sample_info.copy()
                worst_case["prompt"] = action_prompt 

        avg_reward = total_reward / max(1, len(batch_samples))

        return last_prediction, avg_reward, {"worst_case": worst_case, "batch_details": batch_details}
        # for q, a in batch_samples:
        #     # Target LLM에게 "너는 이런 역할을 해"라고 지시 주입
        #     # action_prompt(Agent가 만든 것) -> System Role
        #     # q(실제 질문) -> User Role
        #     target_chain_prompt = ChatPromptTemplate.from_messages([
        #         ("system", action_prompt), # 여기가 핵심! Agent의 결과물이 곧 환경의 설정이 된다.
        #         ("user", "{question}")
        #     ])

        #     # 체인 연결: 프롬프트 -> Target LLM -> 문자열파서
        #     chain = target_chain_prompt | self.target_llm | StrOutputParser()
            
        #     try:
        #         # Target LLM 실행 (답변 생성)
        #         pred = chain.invoke({"question": q})
        #     except Exception as e:
        #         print(f"[Error] Target LLM fail: {e}")
        #         pred = "Error generating response"

        #     # 점수 계산 (임베딩 & 코사인 유사도)
        #     # 정답(a)과 예측(pred)을 임베딩
        #     try:
        #         embeddings = self.embedding_model.embed_documents([a, pred])
        #         # embeddings[0]: 정답 벡터, embeddings[1]: 예측 벡터
        #         score = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
        #     except Exception:
        #         score = 0.0

        #     total_reward += score
        #     last_prediction = pred

        #     # 로깅용 상세 정보
        #     sample_info = {
        #         "question": q,
        #         "reference": a,
        #         "prediction": pred,
        #         "score": score,
        #     }
        #     batch_details.append(sample_info)

        #     # 오답 노트 갱신 (가장 점수가 낮은 케이스)
        #     if score < min_score:
        #         min_score = score
        #         worst_case = sample_info.copy()
        #         worst_case["prompt"] = action_prompt # 당시 썼던 프롬프트 저장

        # # 평균 점수
        # avg_reward = total_reward / max(1, len(batch_samples))

        # # prediction -> 배치의 마지막 예측값 (또는 worst_case['prediction'])
        # # reward -> 평균 점수 (avg_reward)
        # # info -> worst_case 정보 (Agent가 참고할 수 있게)
        # return last_prediction, avg_reward, {"worst_case": worst_case, "batch_details": batch_details}
    


  