"""
Script to extract hallucination prompts from benchmark result files
and split into Train and Test according to a configurable ratio
"""
import argparse
import json
import random
import sys
from pathlib import Path
from typing import Dict, List, Tuple
from typing import List, Tuple
# Add repo root to sys.path before package imports
def find_repo_root() -> Path:
    cur = Path(__file__).resolve().parent
    for parent in [cur] + list(cur.parents):
        if (parent / "pkg_halluc").is_dir() and (parent / "data").is_dir():
            return parent
    return Path(__file__).resolve().parents[1]

REPO_ROOT = find_repo_root()
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pandas as pd
from pkg_halluc.package_loader.unlearn_loader import PackageUnlearningDataset
from pkg_halluc.package_loader.utils import parse_package_list

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


def row_has_hallucination(row: pd.Series) -> bool:
    """Check if a CSV row contains any hallucinated package in any mode or pip"""
    hall_1 = parse_package_list(row.get("hallucinated_1", row.get("hallucination_1", [])))
    hall_2 = parse_package_list(row.get("hallucinated_2", row.get("hallucination_2", [])))
    hall_pip = parse_package_list(row.get("pip_hallucinated", [])) if "pip_hallucinated" in row else []
    return len(hall_1) > 0 or len(hall_2) > 0 or len(hall_pip) > 0


def process_file(
    csv_path: Path,
    target_count: int = 100,
    train_ratio: float = 0.9,
    seed: int = 42,
) -> Tuple[List[str], List[str], pd.DataFrame]:
    """
    Extract target_count unique hallucination prompts from a CSV and split into train and test.
    Returns: (train_prompts, test_prompts, filtered_train_df)
    """
    df = pd.read_csv(csv_path)
    prompt_col = "Prompts" if "Prompts" in df.columns else ("Questions" if "Questions" in df.columns else df.columns[0])

    # Find rows with hallucination
    has_hall = df.apply(row_has_hallucination, axis=1)
    hall_df = df[has_hall]

    # Collect unique prompts that produce hallucinations
    unique_hall_prompts = []
    seen = set()
    for p in hall_df[prompt_col].dropna().astype(str):
        clean_p = p.strip()
        if clean_p and clean_p not in seen:
            seen.add(clean_p)
            unique_hall_prompts.append(clean_p)

    total_avail = len(unique_hall_prompts)
    if total_avail < target_count:
        raise ValueError(
            f"File {csv_path.name} only has {total_avail} unique hallucination prompts, "
            f"which is less than target {target_count}."
        )

    # Deterministic sampling using seeded random
    rng = random.Random(seed)
    selected_prompts = rng.sample(unique_hall_prompts, target_count)

    n_train = int(target_count * train_ratio)
    n_test = target_count - n_train

    train_prompts = selected_prompts[:n_train]
    test_prompts = selected_prompts[n_train:]

    assert len(train_prompts) == n_train, f"Expected {n_train} train prompts, got {len(train_prompts)}"
    assert len(test_prompts) == n_test, f"Expected {n_test} test prompts, got {len(test_prompts)}"
    assert set(train_prompts).isdisjoint(set(test_prompts)), "Train and Test prompts overlap!"

    # Filter original df for train rows
    train_set_prompts = set(train_prompts)
    train_df = df[df[prompt_col].astype(str).str.strip().isin(train_set_prompts)].copy()

    return train_prompts, test_prompts, train_df


def main():
    parser = argparse.ArgumentParser(description="Split hallucination prompts per file into Train and Test sets")
    parser.add_argument(
        "--model",
        default="llama3.2-3b",
        help="Model name or family (e.g. llama3.2-3b, qwen2.5-coder-1.5b) to auto-locate data directory",
    )
    parser.add_argument(
        "--data_dir",
        default=None,
        help="Directory containing the 4 benchmark result CSV files (default: auto-detected based on --model)",
    )
    parser.add_argument(
        "--out_dir",
        default=None,
        help="Output directory for train/test splits and master dataset",
    )
    parser.add_argument(
        "--n_per_file",
        type=int,
        default=100,
        help="Number of hallucination prompts to extract per file",
    )
    parser.add_argument(
        "--train_ratio",
        type=float,
        default=0.9,
        help="Train split ratio (default: 0.9)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic sampling",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)

    print(f"Using data directory: {data_dir}")
    out_dir = Path(args.out_dir) if args.out_dir else (data_dir / "train_test_split")
    train_csv_dir = out_dir / "train_csvs"
    out_dir.mkdir(parents=True, exist_ok=True)
    train_csv_dir.mkdir(parents=True, exist_ok=True)

    csv_files = [
        "LLM_AT_results.csv",
        "LLM_LY_results.csv",
        "SO_AT_results.csv",
        "SO_LY_results.csv",
    ]

    all_train_prompts: List[str] = []
    all_test_prompts: List[str] = []
    train_meta: Dict[str, List[str]] = {}
    test_meta: Dict[str, List[str]] = {}
    saved_train_csv_paths: List[Path] = []
    
    for idx, fname in enumerate(csv_files):
        csv_path = data_dir / fname
        if not csv_path.is_file():
            raise FileNotFoundError(f"Result file not found: {csv_path}")

        file_seed = args.seed + idx * 1000
        train_p, test_p, train_df = process_file(
            csv_path,
            target_count=args.n_per_file,
            train_ratio=args.train_ratio,
            seed=file_seed,
        )

        all_train_prompts.extend(train_p)
        all_test_prompts.extend(test_p)
        train_meta[fname] = train_p
        test_meta[fname] = test_p

        # Save filtered train CSV
        stem = csv_path.stem
        train_csv_path = train_csv_dir / f"{stem}_train.csv"
        train_df.to_csv(train_csv_path, index=False, encoding="utf-8")
        saved_train_csv_paths.append(train_csv_path)

    print("-" * 70)
    # Verification of zero leakage
    train_set = set(all_train_prompts)
    test_set = set(all_test_prompts)
    overlap = train_set.intersection(test_set)

    print(f"Total Train: {len(all_train_prompts)} (Unique: {len(train_set)})")
    print(f"Total Test: {len(all_test_prompts)} (Unique: {len(test_set)})")
    print(f"Train/Test Overlap: {len(overlap)} (Zero leakage: {len(overlap) == 0})")
    assert len(overlap) == 0, f"Critical: Found {len(overlap)} overlapping prompts between Train and Test!"

    test_prompts_file = out_dir / "test_prompts.jsonl"
    with test_prompts_file.open("w", encoding="utf-8") as f:
        for p in all_test_prompts:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    train_prompts_file = out_dir / "train_prompts.jsonl"
    with train_prompts_file.open("w", encoding="utf-8") as f:
        for p in all_train_prompts:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    with (out_dir / "split_metadata.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "seed": args.seed,
                "n_per_file": args.n_per_file,
                "train_ratio": args.train_ratio,
                "total_train": len(all_train_prompts),
                "total_test": len(all_test_prompts),
                "train_by_file": {k: len(v) for k, v in train_meta.items()},
                "test_by_file": {k: len(v) for k, v in test_meta.items()},
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    train_csv_strs = [str(p) for p in saved_train_csv_paths]
    dataset = PackageUnlearningDataset(
        data_source=train_csv_strs,
        split_type="all",
        query_modes=[1, 2],
        auto_save=False,
    )
    master_train_file = out_dir / "master_train.json"
    dataset.save_to_file(master_train_file)

    # Summary of records
    forget_records = [r for r in dataset.records if r.split_type == "forget"]
    retain_records = [r for r in dataset.records if r.split_type == "retain"]
    print(f"- Forget records: {len(forget_records)}")
    print(f"- Retain records: {len(retain_records)}")


if __name__ == "__main__":
    main()

