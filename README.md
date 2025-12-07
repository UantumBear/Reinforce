# 프롬프트 최적화 강화학습 (Prompt Optimization RL)

이 프로젝트는 LLM을 사용하여 RAG 시스템의 프롬프트를 자동으로 최적화하는 강화학습 에이전트입니다.

## 환경 설정 가이드 (Installation)

이 프로젝트는 **GPU 가속(NVIDIA CUDA / Apple MPS)**을 자동으로 감지하여 지원합니다.
GPU가 없어도 CPU 모드로 실행 가능합니다.

### 1. 필수 라이브러리 설치
```bash
# 라이브러리 설치 
pip install -r requirements.txt
# pytorch 설치 - 본인 환경에 맞게 설치
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url [https://download.pytorch.org/whl/cu121](https://download.pytorch.org/whl/cu121)
```
## 가상환경 생성 및 활성화
```bash
C:\Users\litl\AppData\Local\Programs\Python\Python312\python.exe -m venv venv
.\venv\Scripts\activate
```

## 프로젝트 실행
(venv) python main_logging.py
