"""
@경로 : utils/environment/env.py
@설명: 프롬프트 최적화 Environment class
"""
import time
from typing import Tuple, Dict
import numpy as np
import torch # GPU 메모리 확인용
from sklearn.metrics.pairwise import cosine_similarity
# from langchain_core.prompts import PromptTemplate
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from conf.config import Env
from utils.models.model import ModelFactory # LLM 모델과 임베딩 모델 관리
from utils.datasets.klue_mrc import KlueMrcDataset # 데이터셋 관리
from utils.datasets.aihub_llm_development_qa import AihubLlmDevelopmentQaDataset
from utils.datasets.korquad import KorQuADDataset
from utils.datasets.klue_rag import KlueMrcKoRagDataset # KLUE RAG 데이터셋


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
        # self.dataset = KorQuADDataset(split="train") # KorQuAD 데이터셋 로드 (학습용)
        self.dataset = KlueMrcKoRagDataset(split="train") # KLUE RAG 데이터셋 로드 (학습용)

        """
        KorQuAD 데이터셋을 사용한 이유:
        - 질문에 대한 정답이 단답형으로 명확하게 주어져 있고, 질문-정답 뿐 아니라, 참조할 문서 또한 주어져 있어,
          RAG 챗봇의 성능 평가에 적합하다고 판단.
        - AIHub LLM Development QA 데이터셋은 질문-정답 만 존재, 문서 정보가 없으며,
          응답에 '문서가 없습니다.'와 같은 샘플 응답이 포함되어 있어 부적합.
        - 하지만, 그렇기에, 프롬프트를 결정하기 쉬울 수 밖에 없다는 판단도 들어서,
          주관식으로 음답하는 datasets 을 찾아서 테스트 해볼 필요가 있다고 생각한다. 

        앞으로 사용해볼 데이터셋 목록 찾아보기
        - KoCoQa 
        """

        self.target_llm = self.model_factory.get_llm(model_type="target") # 재시도 기능 포함
        # self.embedding_model = self.model_factory.get_embedding_model()   # embdding 모델 # 직접 선언한 래퍼클래스 Ko-SRoBERTa multitask 모델
        self.embedding_model = self.model_factory.get_langchain_embedding_model() # langchain 임베딩 모델 (로컬 Ko-SRoBERTa multitask 모델)

        self.dataset_name = self.dataset.__class__.__name__ # 데이터셋 이름 저장 (로깅파일 저장 용)

    def step(self, action_prompt: str) -> Tuple[str, float, Dict]:
        """
        환경에서 한 스텝 실행
        @Param
            action_prompt (str): 에이전트가 생성한 새로운 프롬프트
            
        @Return
            Tuple[str, float, Dict]: 예측값, 보상, 추가 정보
        """
        if not self.use_azure: 
            # Target LLM 호출 전 대기
            time.sleep(1)  # 무료 버전 rate limit 방지


        total_reward = 0
        valid_sample_count = 0 # 정상적으로 채점된 샘플 수
        # 호출 중 에러가 날 경우, 일단 점수를 0.0으로 처리하고, 다음 샘플로 넘겼는데,
        # 해당 점수는 평균 산출 시 제외해야 하므로, valid_sample_count 변수를 추가. 

        batch_size = 5 # 데이터셋에서 n개만 랜덤으로 뽑아서 테스트 (Mini-batch)        
        batch_samples = self.dataset.get_random_samples(batch_size)

        worst_case = {}
        min_score = 2.0 # 코사인 유사도 최대값은 1.0이므로 그보다 큰 값으로 초기화
        last_prediction = "" # 로깅용 대표 예측값
        batch_details = []  # 배치별 상세 정보 저장용

        #  배치 평가 루프

        for q, a, c in batch_samples:
            # 이 샘플이 정상적으로 처리되었는지 확인하는 플래그
            # 초기값은 False(에러 없음)로 시작
            is_error = False


            # 프롬프트를 사용할 질의응답LLM에게 건네줄 프롬프트 구성 (RAG 시뮬레이션)
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
            
            # Target LLM 실행 (답변 생성)
            try:
                # invoke 할 때 context 변수(c)도 같이 넘겨줌
                pred = chain.invoke({"question": q, "context": c})
            except Exception as e:
                print(f"[Error] Target LLM fail: {e}")
                pred = "응답 생성 중 오류가 발생했습니다."
                is_error = True # [2-2 수정 포인트] 에러 발생 시 플래그 표시

            
            # 점수 계산 (임베딩 & 코사인 유사도)
            if torch.cuda.is_available():
                # 현재 할당된 GPU 메모리 (MB 단위 변환)
                mem_alloc = torch.cuda.memory_allocated() / 1024 / 1024
                # print(f"   [GPU Active] Mem: {mem_alloc:.1f}MB used | Device: {self.embedding_model.client.device}") 
                # LangChain 버전에 따라 client 접근이 다를 수 있으니 안전하게 아래처럼:
                print(f"   ► [GPU 연산 시작] 현재 GPU 메모리 사용량: {mem_alloc:.1f}MB")
            else:
                print("   ► [CPU 연산 시작] GPU가 감지되지 않음")

            # 정답(a)과 예측(pred)을 임베딩
            try:
                embeddings = self.embedding_model.embed_documents([a, pred]) # 여기서 GPU 사용
                score = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
            except Exception:
                score = 0.0
                is_error = True # [2-2 수정 포인트] 임베딩 에러 시에도 플래그 표시

            print(f"   └ [Sample] Score: {score:.4f} | Q: {q[:20]}... | Pred: {pred[:30]}...")

            # 에러가 아닐 때만 마지막 예측값 저장
            if not is_error:
                total_reward += score
                valid_sample_count += 1
                last_prediction = pred  # 챗봇(Target LLM)이 내놓은 답변 중 하나를 저장 (로깅 csv 용도)

            sample_info = {
                "question": q,
                "reference": a,
                "prediction": pred, 
                "score": score,
            }
            batch_details.append(sample_info)


            # 오답 노트(Worst Case) 갱신 로직
            # 에러가 아니고(not is_error) AND 점수가 낮을 때만 갱신
            # 이유: 시스템 에러로 0점 나온 걸 Agent에게 보여주면 "프롬프트 탓"이라고 오해함.
            if not is_error and score < min_score:
                min_score = score
                worst_case = sample_info.copy()
                worst_case["prompt"] = action_prompt 

        # 정상적으로 채점된 개수(valid_sample_count)로만 나눔
        # dataset 의 일부 샘플이, azure 정책에 걸려서 에러가 나는 경우가 많음.
        if valid_sample_count > 0:
            avg_reward = total_reward / valid_sample_count
        else:
            avg_reward = 0.0 # 전부 다 에러난 경우

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
    


  