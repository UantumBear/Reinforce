"""
@경로: models/rag_module.py
@설명: 
- RAG(Retrieval-Augmented Generation) 시스템의 논리 구조를 정의합니다.
- DSPy의 ChainOfThought를 사용하여 '생각(Reasoning)' 능력을 부여합니다.
- 이 모듈이 바로 Optimizer Agent가 최적화할 '대상(Student)'입니다.
- 단, 학습 종료 후 Student 는 남아있지 않습니다.
- 환경은 매 에피소드 마다 새로운 Student 모듈을 생성하여 사용합니다.
- 그 이유는 결국 '최적화된 프롬프트'를 적용하고자 하는 모델은 아무것도 모르는 '깨끗한 LLM' 이기 때문입니다.
- 즉, 이 모듈은 '프롬프트를 테스트 해보기 위한 테스터' 역할 일 뿐입니다.
- Optimizer LLM 이 만든 프롬프트를 받아 Response 를 생성하는 역할만 수행합니다.
- 따라서, 이 모듈 내부에 어떤 학습 파라미터도 존재하지 않습니다.
"""

import dspy

# ----------------------------------------------------------------
# 1. 시그니처 (Signature): 입출력 인터페이스 정의
# RAGSignature 는 Optimizer LLM이 만든 프롬프트 만을 적용하여 테스트 하므로
# Signature를 최소화 하여 작성한다. 
# ----------------------------------------------------------------
class RAGSignature(dspy.Signature):
    # 입력 필드 (Input)
    context = dspy.InputField(desc="context")
    question = dspy.InputField(desc="question")
    
    # 출력 필드 (Output)
    answer = dspy.OutputField(desc="answer")

# ----------------------------------------------------------------
# 2. 모듈 (Module): 실제 동작 로직 정의
# ----------------------------------------------------------------
class RAG_CoT(dspy.Module):
    def __init__(self):
        super().__init__()
        
        # [핵심] ChainOfThought를 사용하여 추론 능력 활성화
        # 이 부분이 Agent가 최적화할 때 건드리는 핵심 부품입니다.
        # 초기에는 RAGSignature의 Docstring이 기본 프롬프트(Instruction)로 사용됩니다.
        self.prog = dspy.ChainOfThought(RAGSignature)
    
    def forward(self, question, context):
        """
        [실행 단계]
        1. 메인 루프나 Env에서 전달받은 question과 context를 입력으로 받습니다.
        2. self.prog(CoT)를 실행하여 답변을 생성합니다.
        """
        
        # (옵션) 만약 context가 리스트로 들어오면 하나의 문자열로 합침
        if isinstance(context, list):
            context = "\n\n".join(context)
            
        # DSPy 모듈 실행
        # 여기서 Azure OpenAI로 요청이 날아갑니다.
        prediction = self.prog(context=context, question=question)
        
        return prediction