"""
Analyze SKT TelAgentBench CSV artifacts.

Path:
    utils/datasets/baseline/sktTelagentbench_analyze.py

Usage:
    python utils/datasets/baseline/sktTelagentbench_analyze.py
    python utils/datasets/baseline/sktTelagentbench_analyze.py --detail
    python utils/datasets/baseline/sktTelagentbench_analyze.py --train-size 200

What it reports:
- CSV file counts and row counts by folder
- Per-file row counts (optional, --detail)
- Action + possible_answer merged row counts by id (the same basis used by the
  default telagentbench loader in data_loader.py)
- Validation-size suggestions given a train-size target
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd


@dataclass
class FolderSummary:
    name: str
    file_count: int
    total_rows: int
    rows_by_file: dict[str, int]


def _count_rows(csv_path: Path) -> int:
    df = pd.read_csv(csv_path)
    return len(df)


def _summarize_folder(folder_path: Path, display_name: str) -> FolderSummary:
    rows_by_file: dict[str, int] = {}
    for csv_file in sorted(folder_path.glob("*.csv")):
        try:
            rows_by_file[csv_file.name] = _count_rows(csv_file)
        except Exception:
            rows_by_file[csv_file.name] = -1

    ok_counts = [v for v in rows_by_file.values() if v >= 0]
    return FolderSummary(
        name=display_name,
        file_count=len(rows_by_file),
        total_rows=sum(ok_counts),
        rows_by_file=rows_by_file,
    )


def _merge_count_by_id(action_file: Path, possible_file: Path) -> tuple[int, int, int, int]:
    """
    Returns:
        (action_rows, possible_rows, merged_rows, unique_merged_ids)
    """
    action_df = pd.read_csv(action_file)
    possible_df = pd.read_csv(possible_file)

    if "id" not in action_df.columns or "id" not in possible_df.columns:
        raise ValueError("Both files must contain 'id' column")

    if "ground_truth" not in possible_df.columns:
        raise ValueError("possible_answer file must contain 'ground_truth' column")

    merged_df = action_df.merge(possible_df[["id", "ground_truth"]], on="id", how="inner")
    unique_merged_ids = merged_df["id"].nunique()
    return len(action_df), len(possible_df), len(merged_df), int(unique_merged_ids)


def _iter_matching_pairs(action_dir: Path, possible_dir: Path) -> Iterable[tuple[Path, Path]]:
    for action_file in sorted(action_dir.glob("*.csv")):
        possible_file = possible_dir / action_file.name
        if possible_file.exists():
            yield action_file, possible_file


def analyze_telagentbench(detail: bool, train_size: int) -> None:
    current_file = Path(__file__).resolve()
    project_root = current_file.parents[3]

    csv_root = project_root / "datafile" / "original" / "skt" / "telagentbench" / "csv"
    action_dir = csv_root / "TelAgent_Action"
    possible_dir = csv_root / "possible_answer"
    if_dir = csv_root / "TelAgent_IF"
    plan_dir = csv_root / "TelAgent_Plan"

    print("=" * 80)
    print("SKT TelAgentBench Analyzer")
    print("=" * 80)
    print(f"[Path] {csv_root}")

    missing_dirs = [
        p for p in [action_dir, possible_dir, if_dir, plan_dir] if not p.exists()
    ]
    if missing_dirs:
        print("[ERROR] Missing required folders:")
        for p in missing_dirs:
            print(f"  - {p}")
        return

    summaries = [
        _summarize_folder(action_dir, "TelAgent_Action"),
        _summarize_folder(possible_dir, "possible_answer"),
        _summarize_folder(if_dir, "TelAgent_IF"),
        _summarize_folder(plan_dir, "TelAgent_Plan"),
    ]

    print("\n[1] Folder summaries")
    for s in summaries:
        print(f"- {s.name}: files={s.file_count}, rows(total)={s.total_rows}")

    if detail:
        print("\n[1-Detail] Per-file row counts")
        for s in summaries:
            print(f"\n{s.name}")
            for fname, cnt in s.rows_by_file.items():
                suffix = " (read error)" if cnt < 0 else ""
                print(f"  - {fname}: {cnt}{suffix}")

    print("\n[2] Default loader basis: Action + possible_answer merged by id")
    total_action_rows = 0
    total_possible_rows = 0
    total_merged_rows = 0
    total_unique_merged_ids = 0
    pair_count = 0

    for action_file, possible_file in _iter_matching_pairs(action_dir, possible_dir):
        pair_count += 1
        try:
            action_rows, possible_rows, merged_rows, unique_ids = _merge_count_by_id(
                action_file, possible_file
            )
            total_action_rows += action_rows
            total_possible_rows += possible_rows
            total_merged_rows += merged_rows
            total_unique_merged_ids += unique_ids

            if detail:
                print(
                    f"- {action_file.name}: action={action_rows}, "
                    f"possible={possible_rows}, merged={merged_rows}, unique_ids={unique_ids}"
                )
        except Exception as e:
            print(f"- {action_file.name}: merge error ({e})")

    print(f"- Matching file pairs: {pair_count}")
    print(f"- Action rows (sum): {total_action_rows}")
    print(f"- possible_answer rows (sum): {total_possible_rows}")
    print(f"- Merged rows by id (sum): {total_merged_rows}")
    print(f"- Unique merged ids (sum by file): {total_unique_merged_ids}")

    print("\n[3] Validation-size guidance")
    print(f"- Current train-size target: {train_size}")
    print(f"- Max validation if train is fixed to {train_size}: {max(total_merged_rows - train_size, 0)}")
    print(f"- Absolute upper bound (train >= 1 assumed): {max(total_merged_rows - 1, 0)}")

    print("\n[4] Mode references")
    print("- telagentbench       -> Action + possible_answer merged")
    print("- telagentbench_if    -> TelAgent_IF/telif_general_ko.csv")
    print("- telagentbench_plan  -> TelAgent_Plan/validation_dataset_1111.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze SKT TelAgentBench dataset artifacts")
    parser.add_argument(
        "--detail",
        action="store_true",
        help="Print per-file row and merge counts",
    )
    parser.add_argument(
        "--train-size",
        type=int,
        default=200,
        help="Train size target used for validation guidance (default: 200)",
    )

    args = parser.parse_args()
    analyze_telagentbench(detail=args.detail, train_size=args.train_size)
