"""
Download Big Bench Hard (BBH) datasets for baseline reproduction.

Path: utils/datasets/baseline/bbh_download_datasets.py
Usage:
  python utils/datasets/baseline/bbh_download_datasets.py
  python utils/datasets/baseline/bbh_download_datasets.py --task object_counting
  python utils/datasets/baseline/bbh_download_datasets.py --task word_sorting

Save path: datafile/original/lukaemon/bbh/<task>/

What this script does:
1. Downloads BBH task data from Hugging Face.
2. Saves raw splits as CSV.
3. Creates deterministic reproduction subsets:
   - train_51.csv
   - validation_100.csv

Default dataset id: lukaemon/bbh
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import pandas as pd
from datasets import load_dataset
from dotenv import load_dotenv


def _load_hf_token(project_root: Path) -> str | None:
    """Load HF token from key/huggingface.env if present."""
    env_path = project_root / "key" / "huggingface.env"
    if not env_path.exists():
        return None

    load_dotenv(env_path)
    token = os.getenv("HF_TOKEN")
    if token and token != "your_token_here":
        return token
    return None


def _save_split_csv(df: pd.DataFrame, output_path: Path, project_root: Path) -> None:
    """Save dataframe to UTF-8 CSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"  saved: {output_path.relative_to(project_root)} ({len(df)} rows)")


def _make_repro_subsets(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build deterministic subsets for prompt optimization reproduction.

    TextGrad paper setup often uses about 51 train and 100 validation examples
    for BBH Object Counting / Word Sorting tasks.
    """
    train_n = min(51, len(df))
    val_n = min(100, max(0, len(df) - train_n))

    train_df = df.iloc[:train_n].reset_index(drop=True)
    val_df = df.iloc[train_n : train_n + val_n].reset_index(drop=True)
    return train_df, val_df


def download_bbh_dataset(task: str = "object_counting", dataset_id: str = "lukaemon/bbh") -> None:
    """Download a BBH task and export it as CSV files."""
    current_file = Path(__file__).resolve()
    # .../Reinforce/utils/datasets/baseline/bbh_download_datasets.py -> project root
    # parents[0]=baseline, [1]=datasets, [2]=utils, [3]=Reinforce(project root)
    project_root = current_file.parents[3]

    save_base_dir = project_root / "datafile" / "original" / "lukaemon" / "bbh" / task
    save_base_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print(f"BBH downloader start | dataset={dataset_id} | task={task}")
    print(f"output dir: {save_base_dir}")
    print("=" * 72)

    hf_token = _load_hf_token(project_root)
    if hf_token:
        print("HF token loaded from key/huggingface.env")
    else:
        print("HF token not found, using anonymous access")

    try:
        if hf_token:
            dataset = load_dataset(dataset_id, task, token=hf_token)
        else:
            dataset = load_dataset(dataset_id, task)
    except Exception as exc:
        print(f"[ERROR] failed to download dataset: {exc}")
        print("Hint: check internet, dataset id/task name, and HF token permissions.")
        return

    split_names = list(dataset.keys())
    print(f"splits: {split_names}")

    # Save all raw splits first.
    for split_name in split_names:
        split_df = pd.DataFrame(dataset[split_name])
        raw_path = save_base_dir / f"{split_name}.csv"
        _save_split_csv(split_df, raw_path, project_root)

    # Create reproduction subsets from test split if available.
    source_split = "test" if "test" in dataset else split_names[0]
    source_df = pd.DataFrame(dataset[source_split])
    train_df, val_df = _make_repro_subsets(source_df)

    _save_split_csv(train_df, save_base_dir / "train_51.csv", project_root)
    _save_split_csv(val_df, save_base_dir / "validation_100.csv", project_root)

    if len(train_df) < 51 or len(val_df) < 100:
        print("[WARN] source split is smaller than 151 rows; subset files were truncated.")

    print("\nDone. You can now use these files for BBH baseline experiments.")


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Download BBH datasets for baseline experiments")
    parser.add_argument(
        "--task",
        type=str,
        default="object_counting",
        choices=["object_counting", "word_sorting"],
        help="BBH task name to download",
    )
    parser.add_argument(
        "--dataset-id",
        type=str,
        default="lukaemon/bbh",
        help="Hugging Face dataset id",
    )
    return parser


def main() -> None:
    parser = _build_arg_parser()
    args = parser.parse_args()
    download_bbh_dataset(task=args.task, dataset_id=args.dataset_id)


if __name__ == "__main__":
    main()
