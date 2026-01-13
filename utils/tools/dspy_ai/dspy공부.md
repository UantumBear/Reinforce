LangChain 객체 -> DSPy 래퍼 -> DSPy 옵티마이저

```PowerShell
pip install dspy
```

# DsPy 란?

DSPy의 별명이 **"프롬프트를 위한 PyTorch"**이다.  
딥러닝이 "가중치(Weight)"를 학습하듯이, DSPy는 "프롬프트(Prompt)"를 학습(최적화)한다.

##### 어떻게 최적화하는가? (강화학습이랑 똑같다!)
강화학습(RL)의 루틴  
[시도 -> 보상(Reward) 확인 -> 수정]  
DSPy의 teleprompter.compile이 정확히 이 일을 한다.  
```powershell
# [dspy\teleprompt\bootstrap.py]

# 시도 (Try): 
LLM한테 문제를 풀게 시켜본다. (처음엔 빈약한 프롬프트로)

def _bootstrap_one_example(self, example, round_idx=0):
    # 온도=1.0으로 새로운 시도 (다양성 확보)
    lm = lm.copy(rollout_id=round_idx, temperature=1.0) if round_idx > 0 else lm
    # 실제 예측 수행
    prediction = teacher(**example.inputs())

# 보상 확인 (Metric):
내가 정해준 채점 기준(answer_exact_match)으로 채점한다.
"너 맞았어? 틀렸어?"
if self.metric:
    metric_val = self.metric(example, prediction, trace)
    if self.metric_threshold:
        success = metric_val >= self.metric_threshold  # 임계값 비교
    else:
        success = metric_val  # Boolean 성공/실패

# 수정 (Update Prompt):

성공하면(Reward): "오, 방금 네가 했던 생각(Reasoning) 아주 좋았어. 이거 '모범 답안(Few-shot Example)'으로 박제해서 프롬프트에 추가하자!"
if success:  # 성공하면
    for step in trace:
        # 성공한 reasoning을 demo로 박제
        demo = dspy.Example(augmented=True, **inputs, **outputs)
        # 프롬프트에 Few-shot Example으로 추가
        self.name2traces[predictor_name].append(demo)

실패하면(Reward): "방금 그 논리는 쓰레기네. 버려."

# 결과:
처음에는 텅 비어있던 프롬프트가, 학습(Compile)이 끝나고 나면 "성공했던 경험들(Best Practices)"로 가득 채워진 최강의 프롬프트로 진화한다. 

# 최종 프롬프트 업데이트
def _train(self):
    # 성공했던 예시들(augmented_demos)과 기본 예시들(raw_demos)을 합쳐서
    # 최강의 프롬프트로 조합
    predictor.demos = augmented_demos + raw_demos
```

!! 그러면 일일히 기말 강화학습 과제를 진행했듯이 일일히 Action-State-Reward 구조를 하드코딩 할 필요가 없다..!!

보상함수를 얼마나 기가 막히게 설계 하느냐에 달린 것 같다..!  


기본 compile 함수를 돌리자, RAG 문서를 예시 자체를 넣어 최적화 하는 Fewshot 기법이 적용되어 있었다.  
내가 생각했던 건, (몰라서 거기까지 밖에 생각을 못했지만)  
'지시문(Instruction)'을 고치고 싶다" (Optimizer 변경)  관점 이었는데,  

Before: "질문에 답해."

After: "너는 보험 전문가야. 답변할 때는 약관 제3조를 우선하고, 절대로 고객 정보를 유출하지 마. 문서는 사실 관계만 참고해."  

DSPy에서 이 '지시문(Instruction)' 자체를 다시 써주는(Re-writing) 도구도 있다고 한다.  

**MIPROv2**나 **COPRO**라는 Optimizer