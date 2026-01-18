
이 프로젝트는 **GPU 가속(NVIDIA CUDA / Apple MPS)**을 자동으로 감지하여 지원합니다.
GPU가 없어도 CPU 모드로 실행 가능합니다.

### 1. 필수 라이브러리 설치
### 셋팅 방식 1 ... (최신)
```powershell
uv sync
```

### 셋팅 방식 2 ... (old, 호환 안 될 수 있음. 기록을 위해 작성해 둠)
```powershell
# 가상환경 생성
C:\Users\litl\AppData\Local\Programs\Python\Python312\python.exe -m venv venv
# 가상환경 활성화
.\venv\Scripts\activate
# 라이브러리 설치 
pip install -r requirements.txt
# pytorch 설치 - 본인 환경에 맞게 설치
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url [https://download.pytorch.org/whl/cu121](https://download.pytorch.org/whl/cu121)

```

## 프로젝트 실행
(.venv) python main_logging.py

## 디버깅 모드 실행
.vscode/launch.json 에 python 에 가상환경 패스 추가  
Ctrl + Shift + P 하고 Python: Select Interpreter 로 가상환경 패스 선택


---

# VSCode Extension 추천

#### csv 뷰어
##### ReprEng (Identifier=repreng.csv, Version=1.2.2)  
##### Markdown Preview Mermaid Support 
md 파일에서 다이어그램을 그려주는 도구.