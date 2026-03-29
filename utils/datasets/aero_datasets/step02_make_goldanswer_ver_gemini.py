
"""
@경로: utils/datasets/aero_datasets/step02_make_goldanswer_ver_gemini.py
@설명: C-MAPSS 데이터를 활용하여 'Code-based Facts' + 'LLM Narrative' 구조의 Gold Answer 생성
- 피드백 반영: 수치적 근거(Facts)를 코드로 먼저 계산하고, 이를 기반으로 LLM이 보고서를 작성하게 함.
- 데이터셋: NASA C-MAPSS FD001 (test_FD001.txt, RUL_FD001.txt)

- textgrad 에서 사용할 Gold Answer 의 적절한 데이터셋을 만들기 위해 작성된 스크립트이다.
- Nasa 에서 공식 터보 팬 엔진 데이터셋인 C-MAPSS의 FD001 버전을 다운받았다. 
  해당 데이터를 기반으로 보고서 작성을 요청하여 Gold Answer를 생성한다.
- textgrad 의 최적화 과정에서는 기본적인 QA 챗봇 데이터를 넣을 경우, 
  기본 LLM 성능이 뛰어나서, 전혀 개선의 여지가 없게 된다. 때문에 
  객관적인 수치를 기반으로 한 분석 보고서를 작성하여 Gold Answer 를 만든 후
  TextGrad에서는 clean LLM 이 Data 와 Gold Answer 만 보고, 해당 보고서와 같은 형식의 답변을 내놓도록 최적화하는 형태로
  실험을 진행하기 위해 작성하였다. 

@실행방법:
uv run python utils/datasets/aero_datasets/step02_make_goldanswer_ver_gemini.py

"""

import json
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm

# 프로젝트 루트 설정 및 모듈 임포트
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from infrastructure.llm_client import get_azure_openai_client
from conf.config import Settings

# 1. 설정
Settings.setup()
client = get_azure_openai_client()
MODEL = Settings.DATASET_GENERATOR_MODEL # 최적화처럼 많이 도는 모델이 아니므로 가장 최신 모델로 설정

# C-MAPSS 컬럼 정의
COLUMNS = ['unit', 'cycle', 'os1', 'os2', 'os3'] + [f's{i}' for i in range(1, 22)]

def calculate_gold_facts(group, ground_truth_rul):
    """
    코드를 통해 객관적인 센서 데이터 분석 지표(Facts)를 생성합니다.
    """
    # 최근 30사이클 데이터 추출
    recent_30 = group.tail(30)
    first_cycle_data = group.iloc[0]
    
    # 분석 대상 핵심 센서 (보통 FD001에서 유의미한 변화를 보이는 센서들 선정)
    key_sensors = ['s2', 's3', 's4', 's7', 's11', 's12', 's15']
    
    # 실제 데이터 길이 사용 (30개 미만일 수 있음)
    n_cycles = len(recent_30)
    
    facts = {
        "recent_avg": recent_30[key_sensors].mean().to_dict(),
        "change_from_initial": (recent_30[key_sensors].mean() - first_cycle_data[key_sensors]).to_dict(),
        "volatility": recent_30[key_sensors].std().to_dict(),
        "trend_slope": {s: float(np.polyfit(range(n_cycles), recent_30[s], 1)[0]) for s in key_sensors},
        "actual_rul": int(ground_truth_rul),
        "status_label": "위험" if ground_truth_rul < 30 else "주의" if ground_truth_rul < 70 else "정상"
    }
    return facts

TEACHER_SYSTEM_PROMPT = """
당신은 항공 엔진 데이터 분석 전문가입니다. 
제공되는 분석 사실(Facts)을 바탕으로 정비사를 위한 **가독성이 뛰어난 마크다운(Markdown) 보고서**를 작성하세요.

### [작성 규칙]
1. 모든 섹션 제목은 `###` 헤더를 사용하세요.
2. 각 섹션 사이에는 반드시 **빈 줄(Double Newline)**을 추가하여 가독성을 높이세요.
3. 수치나 상태 정보는 `**`를 사용하여 **굵게** 표시하세요.
4. 모든 리스트 항목은 `-` 불렛 포인트를 사용하세요.

---

### ** [Unit {id}] 엔진 상태 분석 보고서 **

### [상태 등급]
- **{status_label}** (정상/주의/위험 중 하나)

### [핵심 센서 요약]
- **s4**: 최근 평균 **{avg}**, 초기 대비 **{diff}**, 추세 기울기 **{slope}** 등 요약 (3개 센서)

### [이상 징후]
- 발견된 이상 현상을 구체적으로 서술하세요. (섹션 간 줄바꿈 필수)

### [점검 권고]
- **RUL {actual_rul}** 기반의 구체적인 권고안 2개

### [근거 문장]
- 수치 데이터가 포함된 근거 문장 2개
"""

TEACHER_SYSTEM_PROMPT_V2 = """
당신은 항공 엔진 데이터 분석 전문가입니다. 
제공되는 분석 사실(Facts)을 바탕으로 정비사를 위한 **가독성이 극대화된 마크다운 보고서**를 작성하세요.

### [가독성 필수 규칙 - 미준수 시 정비 오류 발생 위험]
1. **섹션 구분**: 각 대주제(###) 사이에는 반드시 **빈 줄을 2줄(Double Newline)** 삽입하세요.
2. **리스트**: 모든 리스트 항목(-) 앞뒤에는 줄바꿈을 넣어 항목이 섞이지 않게 하세요.
3. **강조**: 핵심 수치와 상태 등급은 반드시 `**텍스트**` 형식을 사용하여 굵게 표시하세요.
4. **절대 금기**: 모든 내용을 한 줄로 이어 쓰지 마세요. 가독성이 낮으면 즉시 탈락입니다.

---

### [보고서 출력 템플릿 (이 양식을 복사해서 채우세요)]

## ** [Unit {id}] 엔진 상태 분석 보고서 **

### [상태 등급]
- **{status_label}**

### [핵심 센서 요약]
- **s4**: 최근 평균 **{avg_s4}**, 초기 대비 **{diff_s4}**, 추세 기울기 **{slope_s4}**
- **s11**: 최근 평균 **{avg_s11}**, 초기 대비 **{diff_s11}**, 추세 기울기 **{slope_s11}**
- **s12**: 최근 평균 **{avg_s12}**, 초기 대비 **{diff_s12}**, 추세 기울기 **{slope_s12}**

### [이상 징후]
- (첫 번째 이상 현상 상세 설명)
- (두 번째 이상 현상 상세 설명)

### [점검 권고]
- **RUL {actual_rul}** 기반 권고안 1
- 권고안 2

### [근거 문장]
- (수치가 포함된 근거 문장 1)
- (수치가 포함된 근거 문장 2)
"""


def generate_gold_standards(test_data_path, rul_data_path, output_json_path, count=10):
    # 1. 데이터 로드
    print("[INFO] C-MAPSS 데이터를 로드하고 분석 중...")
    test_df = pd.read_csv(test_data_path, sep=r'\s+', header=None, names=COLUMNS)
    rul_df = pd.read_csv(rul_data_path, sep=r'\s+', header=None, names=['rul'])
    
    gold_standard_results = []
    unique_units = test_df['unit'].unique()[:count]

    # 분석 대상 핵심 센서 정의 (calculate_gold_facts와 동일하게 유지)
    key_sensors = ['s2', 's3', 's4', 's7', 's11', 's12', 's15']
    
    for i, unit_id in enumerate(tqdm(unique_units, desc="Gold Answer 생성 중")):
        unit_data = test_df[test_df['unit'] == unit_id]
        gt_rul = rul_df.iloc[i].values[0]
        
        # 데이터가 충분한지 확인
        if len(unit_data) < 10:
            print(f"[WARNING] Unit {unit_id}의 데이터가 부족합니다 ({len(unit_data)} cycles). 건너뜁니다.")
            continue
        
        # 단계 1: 코드 기반 Facts 계산 (이게 핵심!)
        facts = calculate_gold_facts(unit_data, gt_rul)
        
        # 입력용 원본 로그 (최근 30사이클, gold_facts와 동일한 센서 사용)
        input_log_str = unit_data.tail(30)[['cycle'] + key_sensors].to_string(index=False)

        try:
            # 프롬프트의 {id} 부분을 실제 unit_id로 치환
            system_prompt = TEACHER_SYSTEM_PROMPT_V2.replace("{id}", str(unit_id))

            # 단계 2: LLM을 활용한 Narrative 생성
            prompt_content = f"분석용 Facts 데이터: {json.dumps(facts, indent=2)}\n\n입력 로그 샘플:\n{input_log_str}"
            
            response = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt_content}
                ]
            )

            gold_narrative = response.choices[0].message.content

            # 최종 데이터 구조화
            gold_standard_results.append({
                "id": int(unit_id),
                "input_log": input_log_str, # LLM 모델이 보게 될 입력
                "gold_facts": facts,        # TextGrad가 내부적으로 참조하거나 평가할 객관적 지표
                "gold_standard_report": gold_narrative # LLM이 따라야 할 모범 답안 서술
            })

        except Exception as e:
            print(f"[ERROR] Unit {unit_id} 생성 중 오류 발생: {e}")

    # 결과 저장
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(gold_standard_results, f, indent=4, ensure_ascii=False)
    
    print(f"\n[SUCCESS] 생성 완료! 총 {len(gold_standard_results)}개 생성됨")
    print(f"[SUCCESS] 저장 위치: {output_json_path}")

if __name__ == "__main__":
    DATA_PK = 'FD001'  # FD001, FD002, FD003, FD004 중 선택 가능 (현재는 FD001로 고정)
    COUNT_NUM = 30
    # NASA 데이터셋 경로
    TEST_DATA_PATH = PROJECT_ROOT / 'datafile' / 'raw' / 'nasa' / 'CMAPSSData' / f'test_{DATA_PK}.txt'
    RUL_DATA_PATH = PROJECT_ROOT / 'datafile' / 'raw' / 'nasa' / 'CMAPSSData' / f'RUL_{DATA_PK}.txt'
    OUTPUT_PATH = PROJECT_ROOT / 'datafile' / 'preprocess' / 'nasa' / f'gold_standard_dataset_ver_gemini_{DATA_PK}_cnt_{COUNT_NUM}_v2.json'
    
    # 폴더가 없으면 생성
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    generate_gold_standards(
        str(TEST_DATA_PATH), 
        str(RUL_DATA_PATH), 
        str(OUTPUT_PATH), 
        count=COUNT_NUM # 테스트용으로 5개만 생성
    )

    # uv run python utils/datasets/aero_datasets/step02_make_goldanswer_ver_gemini.py