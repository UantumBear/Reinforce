GEPA (Genetic-Pareto Prompt Optimization)는 최근 주목받고 있는 프롬프트 자동 최적화 도구입니다. 

LLM API(OpenAI 등)가 가장 필수적이며, 허깅페이스 모델 다운로드는 선택 사항입니다.

```powerShell
pip install gepa
```

### GEPA를 사용하기 위해 준비해야 할 것 (필수)

GEPA는 모델 자체가 아니라, 기존 모델의 프롬프트를 깎아주는 '도구(Optimizer)'입니다. 따라서 다음 3가지가 반드시 필요합니다.

#### ① 고성능 LLM의 API Key (가장 중요)
GEPA는 '유전 알고리즘'과 '자연어 성찰(Reflection)'을 사용하여 프롬프트를 개선합니다.  
이때 "프롬프트를 어떻게 고칠지 고민하는 역할(Optimizer Model)"을 수행할 똑똑한 LLM이 필요합니다.

추천: GPT-4o, Claude 3.5 Sonnet 등 (성능이 좋을수록 최적화 결과가 좋습니다)

준비물: OPENAI_API_KEY 또는 ANTHROPIC_API_KEY 등

#### ② 최적화할 대상 (Task Model & System)
어떤 시스템의 프롬프트를 고칠 것인지 코드로 정의되어 있어야 합니다.

GEPA는 DSPy 프레임워크와 연동이 잘 됩니다 (dspy.GEPA).   
기존에 작성하신 랭체인이나 파이썬 코드라면 GEPA가 요구하는 형태(GEPAAdapter)로 감싸주어야 할 수 있습니다.

#### ③ 평가 데이터셋 (Dataset & Metric)
GEPA가 "이 프롬프트가 더 낫다"라고 판단하려면 채점 기준이 필요합니다.  

데이터: 질문(Input)과 모범 답안(Gold Output)이 포함된 데이터셋 (예: 50~100개 정도의 훈련 데이터)  

평가 지표 (Metric): LLM이 내놓은 답이 정답과 얼마나 유사한지 점수를 매기는 함수 (예: 정확도, 코사인 유사도, 혹은 LLM-as-a-Judge)  