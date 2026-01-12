"""

DSPy가 미리 만들어둔 함수


dspy.evaluate.answer_exact_match: 정답과 토씨 하나 안 틀리고 똑같은지. --> 채점기LLM 용도
dspy.evaluate.answer_passage_match: 검색된 문서 안에 정답 단어가 포함되어 있는지.


# 커스텀 채점 로직 사용 방식

import dspy
from dspy.teleprompt import BootstrapFewShot

# 1.채점 로직 
def custom_validate_answer(example, prediction, trace=None):
    if len(prediction.answer) > 100: return False
    if example.keyword not in prediction.answer: return False
    return True

# 2. 최적화 도구(Optimizer) 설정
# "나는 BootstrapFewShot이라는 방식을 쓸 건데, 채점 기준(metric)은 'validate_answer'를 써라"
teleprompter = BootstrapFewShot(metric=custom_validate_answer) 

# 3. 컴파일 (최적화 시작)
# "내 프로그램(my_rag)을 훈련 데이터(trainset)로 최적화해라"
compiled_program = teleprompter.compile(student=my_rag, trainset=trainset)


여기서 BootstrapFewShot 란?
내가 푼 문제 중에 정답 맞힌 것만 골라서, 다음 시험 볼 때 컨닝 페이퍼(예시)로 쓰겠다. 라는
되게 간단한 옵티마이저

DSPy에서는 옵티마이저를 Teleprompter라고 부름

## DSPy 주요 옵티마이저 소스 코드 매핑

| 옵티마이저 이름 | 실제 소스 파일 위치 (대략적인 경로) | 클래스명 | 설명 |
| :--- | :--- | :--- | :--- |
| **LabeledFewShot** | `dspy/teleprompt/vanilla.py` | `LabeledFewShot` | 사용자가 입력한 예제 중 k개를 무작위 선택 (가장 기본) |
| **BootstrapFewShot** | `dspy/teleprompt/bootstrap.py` | `BootstrapFewShot` | 모델이 스스로 생성하고 검증한 고품질 예제를 사용 (Teacher-Student 방식) |
| **Bootstrap...RandomSearch** | `dspy/teleprompt/bootstrap.py` | `BootstrapFewShotWithRandomSearch` | BootstrapFewShot을 여러 번 수행하여 최적의 예제 조합을 탐색 |
| **KNNFewShot** | `dspy/teleprompt/knn_fewshot.py` | `KNNFewShot` | 입력 질문과 가장 유사한(KNN) 예제를 동적으로 선택 |
| **COPRO** | `dspy/teleprompt/copro_optimizer.py` | `COPRO` | 프롬프트의 지시문(Signature Instruction) 자체를 개선 및 최적화 |
| **MIPROv2** | `dspy/teleprompt/mipro_optimizer.py` | `MIPROv2` (또는 `MIPRO`) | 베이지안 최적화를 통해 지시문과 예제(Few-shot)를 동시에 최적화 |

"""

