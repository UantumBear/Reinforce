"""
@경로: datafile/skt_data_preprocessor.py
@설명: SKT TelAgentBench 데이터셋 전용 로더
        data_loader.py 에서 위임받아 호출됩니다.

@데이터셋: skt/TelAgentBench
@저장 경로: datafile/original/skt/telagentbench/csv/
@지원 스키마:
    - action_qa (기본): TelAgent_Action + possible_answer 병합
    - plan:             TelAgent_Plan
    - if:               TelAgent_IF
"""
import ast
import random
import pandas as pd
import dspy
from pathlib import Path

# 프로젝트 루트 경로
# 이 파일(datafile/skt_data_preprocessor.py)의 부모(datafile)의 부모(ProjectRoot)
BASE_DIR = Path(__file__).resolve().parent.parent


def _parse_telagent_conversation(raw_question: str):
    """
    TelAgentBench action_qa 의 question 필드(대화 리스트 문자열)를 파싱하여
    (system_persona, user_utterance) 튜플을 반환합니다.

    @param raw_question: CSV에서 읽은 question 열의 원시 문자열
        예) "[[{'role': 'system', 'content': '...'}, {'role': 'user', 'content': '...'}]]"
    @return: (system_persona, user_utterance)
        - 파싱 성공: system content와 user content 분리
        - 파싱 실패: ("", raw_question) 반환 (기존 포맷 유지)
    """
    try:
        parsed = ast.literal_eval(raw_question)
        if isinstance(parsed, list) and len(parsed) > 0:
            messages = parsed[0] if isinstance(parsed[0], list) else parsed
            system_persona = ""
            user_utterance = raw_question  # fallback
            for msg in messages:
                if isinstance(msg, dict):
                    role = msg.get("role", "")
                    content = msg.get("content", "")
                    if role == "system":
                        system_persona = content
                    elif role == "user":
                        user_utterance = content
            return system_persona, user_utterance
    except Exception:
        pass
    return "", raw_question


def load_telagentbench_dataset(dataset_name="telagentbench", sample_size=None, random_seed=42):
    """
    SKT TelAgentBench CSV 데이터셋을 로드합니다.

    @지원 스키마:
        - TelAgent_Action + possible_answer: question/function/metadata + ground_truth 병합
        - TelAgent_Plan: query / reference_information
        - TelAgent_IF: input / expected_output / metadata

    @주의:
        dataset_name에 접미어를 붙여 로더를 선택할 수 있습니다.
        - "telagentbench"(기본): Action + possible_answer 병합
        - "telagentbench_if": TelAgent_IF
        - "telagentbench_plan": TelAgent_Plan
    """
    dataset_name_lower = dataset_name.lower()
    telagent_root = BASE_DIR / "datafile" / "original" / "skt" / "telagentbench" / "csv"

    try:
        if "_if" in dataset_name_lower or dataset_name_lower.endswith("/if"):
            dataset_kind = "if"
            file_path = telagent_root / "TelAgent_IF" / "telif_general_ko.csv"
            if not file_path.exists():
                raise FileNotFoundError(f"TelAgentBench IF 데이터셋을 찾을 수 없습니다: {file_path}")
            df = pd.read_csv(file_path)

        elif "_plan" in dataset_name_lower or dataset_name_lower.endswith("/plan"):
            dataset_kind = "plan"
            file_path = telagent_root / "TelAgent_Plan" / "validation_dataset_1111.csv"
            if not file_path.exists():
                raise FileNotFoundError(f"TelAgentBench Plan 데이터셋을 찾을 수 없습니다: {file_path}")
            df = pd.read_csv(file_path)

        else:
            # 기본값: Action 질문과 possible_answer 정답을 id 기준으로 병합
            dataset_kind = "action_qa"
            action_dir = telagent_root / "TelAgent_Action"
            possible_dir = telagent_root / "possible_answer"

            if not action_dir.exists() or not possible_dir.exists():
                raise FileNotFoundError(
                    f"TelAgentBench 병합용 폴더를 찾을 수 없습니다:\n"
                    f"  Action:   {action_dir}\n"
                    f"  Possible: {possible_dir}"
                )

            merged_frames = []
            for action_file in sorted(action_dir.glob("*.csv")):
                possible_file = possible_dir / action_file.name
                if not possible_file.exists():
                    continue
                action_df = pd.read_csv(action_file)
                possible_df = pd.read_csv(possible_file)
                if "id" not in action_df.columns or "id" not in possible_df.columns:
                    continue
                gt_col = "ground_truth" if "ground_truth" in possible_df.columns else None
                if gt_col is None:
                    continue
                merged_df = action_df.merge(
                    possible_df[["id", gt_col]], on="id", how="inner"
                )
                merged_df["__source_file"] = action_file.name
                merged_frames.append(merged_df)

            if not merged_frames:
                raise RuntimeError("TelAgentBench Action+possible_answer 병합 결과가 비어 있습니다.")

            df = pd.concat(merged_frames, ignore_index=True)

        print(f"[TelAgentBench Loader] ({dataset_kind}) 로드 완료. 총 {len(df)}개 행.")
        print(f"[TelAgentBench Loader] 컬럼: {df.columns.tolist()}")

    except Exception as e:
        raise RuntimeError(f"TelAgentBench 데이터셋 로드 실패: {e}") from e

    if sample_size:
        df = df.sample(n=min(sample_size, len(df)), random_state=random_seed)
        df = df.reset_index(drop=True)
        print(f"[TelAgentBench Loader] {random_seed} 시드로 {len(df)}개 샘플 추출")

    dataset = []
    empty_answer_count = 0

    for _, row in df.iterrows():
        if dataset_kind == "action_qa":
            question_raw = str(row.get("question", ""))
            function_desc = row.get("function", "")
            metadata = row.get("metadata", "")
            context = f"functions: {function_desc}\nmetadata: {metadata}"
            answer = row.get("ground_truth", "")
            system_persona, question = _parse_telagent_conversation(question_raw)
        elif dataset_kind == "plan":
            question = row.get("query", "")
            context = row.get("reference_information", "")
            answer = row.get("answer", "") if "answer" in df.columns else ""
            system_persona = ""
        else:  # if
            question = row.get("input", "")
            context = row.get("metadata", "")
            answer = row.get("expected_output", "")
            system_persona = ""

        if pd.isna(question) or str(question).strip() == "":
            continue
        if pd.isna(context):
            context = ""
        if pd.isna(answer):
            answer = ""
        if str(answer).strip() == "":
            empty_answer_count += 1

        example = dspy.Example(
            question=str(question),
            context=str(context),
            answer=str(answer),
            system_persona=system_persona,
            dataset_kind=dataset_kind,
        ).with_inputs("question", "context")
        dataset.append(example)

    print(f"[TelAgentBench Loader] ({dataset_kind}) 변환 완료: {len(dataset)}개 예제")
    if empty_answer_count:
        print(f"[TelAgentBench Loader] answer 비어 있는 샘플: {empty_answer_count}개")

    random.seed(random_seed)
    random.shuffle(dataset)
    print(f"[TelAgentBench Loader] {random_seed} 시드로 셔플 완료")

    return dataset
