
# uv 설치 (최초 수행)

#### uv 설치 명령어
```powershell
# (VSCode Terminal 에서 수행)
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```
```powershell
# >> 결과
Downloading uv 0.9.24 (x86_64-pc-windows-msvc)
Installing to C:\Users\litl\.local\bin
  uv.exe
  uvx.exe
  uvw.exe
everything's installed!

To add C:\Users\litl\.local\bin to your PATH, either restart your shell or run:

    set Path=C:\Users\litl\.local\bin;%Path%   (cmd)
    $env:Path = "C:\Users\litl\.local\bin;$env:Path"   (powershell)
```

#### uv 설치 확인
```powershell
# VSCode 를 껐다 킨 후 수행합니다.
Get-Command uv | Select-Object Source ; uv --version
```
```powershell
# >> 결과
uv 0.9.24 (0fda1525e 2026-01-09)
Source
------
C:\Users\litl\.local\bin\uv.exe
```

#### uv 프로젝트 셋팅
```powershell
uv init --python 3.12
```
```powershell
# >> 결과
Initialized project `reinforce`
```
```powershell
# 설치된 파이썬 버전 확인하기
uv run python --version
```
```powershell
# >> 결과
Using CPython 3.12.4 interpreter at: C:\Users\litl\anaconda3\python.exe
Creating virtual environment at: .venv
Python 3.12.4     
```

#### uv 라이브러리 설치하기
```powershell
# 기존에 관리하던 requirements.txt 를 이식하기 위함
uv add -r requirements.txt

# 위 명령어로 일단 전체 라이브러리를 == 버전으로 추가해준 후, >= 로 일괄 변경해주었다.
```
```powershell
# torch 는 따로 add
# 보안 경고 무시하고 --index-strategy unsafe-best-match
# pipy나 pytorch,org 에서 찾아와라!
uv add torch==2.5.1+cu121 torchvision==0.20.1+cu121 torchaudio==2.5.1+cu121 --index https://download.pytorch.org/whl/cu121 --index-strategy unsafe-best-match
```

```powershell
# 해당 호환성에 맞추어, 전체 라이브러리 업그레이드
uv sync --upgrade
```