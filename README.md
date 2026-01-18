# Prompt Optimization (프롬프트 최적화)

이 프로젝트는 LLM을 사용하여 RAG 시스템의 프롬프트를 자동으로 최적화하는 강화학습 에이전트입니다.  
##### Release Tag v1.0
State-Action-Reward 강화학습을 모티브로 설계한 프롬프트 최적화 프레임워크,  Google generative ai (gemini) 를 사용하였습니다.  

##### 진행 중
최근에 나온 다양한 프롬프트 최적화 도구를 사용해, 디벨롭해보고자 합니다.    
In-Context RL (맥락 기반 강화학습)   


---
```mermaid
flowchart TD
    %% 1. 상단: 시작 및 초기화
    Start([시작: Main Execution]) --> Init[초기화: Setup LLM, Load Data]
    Init --> Baseline[Episode 0: Baseline 평가]
    
    %% 2. 중단: 메인 루프
    subgraph LoopGraph ["Optimization Loop (While)"]
        direction TB
        %% Baseline에서 루프 진입 연결
        Baseline --> CheckCond{성공 횟수 < 5?}
        
        CheckCond -- No (목표 달성) --> Finish([종료: 결과 저장])
        CheckCond -- Yes --> AgentAct[Agent.act: 새 프롬프트 생성]
        
        AgentAct -- New Instruction --> EnvStep[Env.step: 평가 및 채점]
        
        EnvStep -- Error (Azure Filter) --> Retry[Log 삭제 & 대기]
        Retry --> CheckCond
        
        EnvStep -- Success --> UpdateState[State 갱신 & 성공 카운트 +1]
        UpdateState --> CheckCond
    end
    
    %% 3. 하단: 상세 로직 (Env.step의 내부)
    subgraph DetailGraph ["Detail: Env.step (세부 로직)"]
        direction TB
        RunRAG[RAG 검색 & 답변 생성] --> Metric[채점: 헌법/신뢰도/스타일]
        Metric --> MakeLog[피드백 생성]
    end

    %% [핵심] 투명 연결선 (~~~)으로 세로 배치 강제
    %% Finish(위쪽 덩어리 끝) 바로 아래에 RunRAG(아래쪽 덩어리 시작)가 오도록 설정
    Finish ~~~ RunRAG
```



---

# Reinforce Project

## Dependencies & Acknowledgments

This project uses code from the following open source projects:

### DSPy
- **Source**: [Stanford DSPy](https://github.com/stanfordnlp/dspy)  
- **License**: MIT License  
- **Usage**: Bootstrap algorithm implementation study (`utils/tools/dspy_ai/study/`)
- **Copyright**: Copyright (c) 2023 Stanford Future Data Systems

---

## 환경 설정 가이드 (Installation)

자세한 설치 및 환경설정 방법은 [Setup Guide](docs/setup_guide.md)를 참고하세요.
