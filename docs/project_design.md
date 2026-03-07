해당 프로젝트에는 run (실행) 을 할 py 파일이 여러개가 있다.  (혹은 개발할 예정)
직접 실행하는 파일들은 main_{특징}.py 형태를 띈다.

1. main_train.py
2. main_textgrad_baseline.py
3. main_textgrad_heierarchical.py

# main_train.py
main_train.py 는 해당 프로젝트의 가장 기본이 되는 이론을 구현한 실험을 실행하는 파일이다.  
Reinforce 프로젝트는 Action, State, Reward 를 모티브로 해서 
- Action 은 OptimizerLLM 이 프롬프트를 만들고, 해당 프롬프트를 이용해서 CleanLLM(TesterLLM) 이 테스트를 실행.
- State 는 TesterLLM 이 실행한 결과를 분석해서, 점수표와 verbal feedback 을 생성해서 다음 step(Episode) 에 전달.
- Rewards 는 CleanLLM 이 만든 답안과 모범 답안과의 semantic simialrity 에 특정 패널티를 반영한 Score 점수
라는 개념을 가지고, 직전 Episode 에서 생성된 언어피드백을 기반으로 다시 OptimizerLLM 이 프롬프트를 생성해서, 원하는 목표 달성을 위해 프롬프트를 개선해 나간다는 목표를 가진다.

# main_textgrad_baseline.py
main_textgrad_baseline.py 는 해당 프로젝트의 비교 연구가 될 baseline 실험을 실행하는 파일이다.
논문 TEXTGRAD 를 구현하는 것을 목표로 하되, 각 main_{주제}.py 의 실험파일들이 쉽게 부품을 교체하고 갈아끼울 수 있는 형태로 소스코드를 구현하는 것을 목표로 한다.

# main_textgrad_heirarchical.py
main_textgrad_heierarchical.py 는 textgrad_baseline 실험에, 딱 "계층적인 언어 피드백" 만 추가한 형태로 실험을 실행하는 파일이다.



