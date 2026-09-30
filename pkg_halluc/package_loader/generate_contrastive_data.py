"""
Generate Contrastive Data for CAA Package Hallucination Steering.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple, Union

if sys.stdout.encoding != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pkg_halluc.package_loader.utils import parse_package_list, split_records_by_prompt


def _normalize_record(r: Any) -> Dict[str, Any]:
    """Convert an unlearning record or dict-like object to a dictionary."""
    if isinstance(r, dict):
        return r
    if hasattr(r, "to_dict"):
        return r.to_dict()
    if hasattr(r, "__dict__"):
        return vars(r)
    return dict(r)  
    
def locate_packages(
    packages: List[str],
    hallucinated_set: Set[str],
) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Determine the character spans (char_start, char_end) for each package in the completion:
    - First package (idx = 0): starts at index 0.
    - Subsequent packages (idx > 0): start immediately after comma-space separator (', ').
    """
    packages_info: List[Dict[str, Any]] = []
    current_char = 0

    lower_hall_set = {p.lower() for p in hallucinated_set}

    for idx, pkg in enumerate(packages):
        separator = ", " if idx > 0 else ""
        current_char += len(separator)

        char_start = current_char
        char_end = char_start + len(pkg)
        current_char = char_end

        is_hall = (pkg in hallucinated_set) or (pkg.lower() in lower_hall_set)

        packages_info.append(
            {
                "package_name": pkg,
                "is_hallucinated": is_hall,
                "position_idx": idx,
                "is_first": (idx == 0),
                "preceding_separator": separator,
                "char_start": char_start,
                "char_end": char_end,
            }
        )

    completion_str = ", ".join(packages)
    return completion_str, packages_info


def save_json(data: Any, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def process_samples_with_augmentations(
    base_samples: List[Dict[str, Any]],
    num_shuffles_per_sample: int,
    rng: random.Random,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    For each base sample:
    1. Generates the original record (preserving original package order).
    2. Generates up to num_shuffles_per_sample distinct shuffled permutations,
       diversifying token positions and preceding context for each package.
    """
    original_records: List[Dict[str, Any]] = []
    shuffled_only_records: List[Dict[str, Any]] = []

    for item in base_samples:
        item = _normalize_record(item)
        valid_pkgs = parse_package_list(item.get("valid_packages", []))
        hall_pkgs = parse_package_list(item.get("hallucinated_packages", []))
        hall_set = set(hall_pkgs)

        rejected_str = item.get("rejected") or item.get("completion", "")
        if isinstance(rejected_str, list):
            pkgs_with_hall = [str(p).strip() for p in rejected_str if str(p).strip()]
        elif isinstance(rejected_str, str):
            pkgs_with_hall = [p.strip() for p in rejected_str.split(",") if p.strip()]
        else:
            pkgs_with_hall = []

        if not pkgs_with_hall:
            pkgs_with_hall = list(valid_pkgs) + list(hall_pkgs)

        sys_prompt = item.get("system_prompt", "").strip() if item.get("system_prompt") else ""
        usr_prompt = item.get("user_prompt", "").strip() if item.get("user_prompt") else ""
        question = f"{sys_prompt}\n\n{usr_prompt}" if sys_prompt else usr_prompt

        # Original Sample
        orig_comp, orig_pkgs_info = locate_packages(pkgs_with_hall, hall_set)
        orig_record = {
            "sample_id": item.get("sample_id", ""),
            "base_sample_id": item.get("sample_id", ""),
            "is_shuffled_variation": False,
            "variation_idx": 0,
            "question": question,
            "answer_matching_behavior": ", ".join(valid_pkgs) if valid_pkgs else "None",
            "answer_not_matching_behavior": ", ".join(hall_pkgs),
            "completion": orig_comp,
            "packages_info": orig_pkgs_info,
            "valid_packages": valid_pkgs,
            "hallucinated_packages": hall_pkgs,
            "system_prompt": sys_prompt,
            "user_prompt": usr_prompt,
            "split_type": item.get("split_type", "forget"),
            "mode": item.get("mode", 1),
        }
        original_records.append(orig_record)

        # Shuffled Variations
        can_shuffle = (
            len(pkgs_with_hall) > 1
            or len(valid_pkgs) > 1
            or len(hall_pkgs) > 1
        )
        if not can_shuffle or num_shuffles_per_sample <= 0:
            continue

        seen_completion_perms = {tuple(pkgs_with_hall)}
        seen_valid_perms = {tuple(valid_pkgs)}
        seen_hall_perms = {tuple(hall_pkgs)}

        shuffles_added = 0
        for attempt in range(num_shuffles_per_sample * 5):
            if shuffles_added >= num_shuffles_per_sample:
                break

            # Permute completion packages
            shuff_comp_pkgs = list(pkgs_with_hall)
            if len(shuff_comp_pkgs) > 1:
                rng.shuffle(shuff_comp_pkgs)

            # Permute valid and hallucinated subsets
            shuff_valid_pkgs = list(valid_pkgs)
            if len(shuff_valid_pkgs) > 1:
                rng.shuffle(shuff_valid_pkgs)

            shuff_hall_pkgs = list(hall_pkgs)
            if len(shuff_hall_pkgs) > 1:
                rng.shuffle(shuff_hall_pkgs)

            comp_tuple = tuple(shuff_comp_pkgs)
            valid_tuple = tuple(shuff_valid_pkgs)
            hall_tuple = tuple(shuff_hall_pkgs)

            # Skip if this permutation combination has already been added
            is_new = (
                comp_tuple not in seen_completion_perms
                or valid_tuple not in seen_valid_perms
                or hall_tuple not in seen_hall_perms
            )
            if not is_new and attempt < (num_shuffles_per_sample * 3):
                continue

            seen_completion_perms.add(comp_tuple)
            seen_valid_perms.add(valid_tuple)
            seen_hall_perms.add(hall_tuple)
            shuffles_added += 1

            shuff_comp, shuff_pkgs_info = locate_packages(shuff_comp_pkgs, hall_set)
            shuff_record = {
                "sample_id": f"{item.get('sample_id', '')}_shuff_{shuffles_added}",
                "base_sample_id": item.get("sample_id", ""),
                "is_shuffled_variation": True,
                "variation_idx": shuffles_added,
                "question": question,
                "answer_matching_behavior": ", ".join(shuff_valid_pkgs) if shuff_valid_pkgs else "None",
                "answer_not_matching_behavior": ", ".join(shuff_hall_pkgs),
                "completion": shuff_comp,
                "packages_info": shuff_pkgs_info,
                "valid_packages": valid_pkgs,
                "hallucinated_packages": hall_pkgs,
                "system_prompt": sys_prompt,
                "user_prompt": usr_prompt,
                "split_type": item.get("split_type", "forget"),
                "mode": item.get("mode", 1),
            }
            shuffled_only_records.append(shuff_record)

    combined_records = list(original_records) + list(shuffled_only_records)
    rng.shuffle(combined_records)

    return original_records, combined_records, shuffled_only_records


def split_train_val(
    train_records: List[Dict[str, Any]],
    val_ratio: float = 0.1,
    seed: int = 42,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Split training records by source prompt before creating augmentations."""
    return split_records_by_prompt(
        train_records,
        val_ratio=val_ratio,
        seed=seed,
        split_retain_only=False,
    )


def _load_records(source: Union[str, Path, List[Any]]) -> List[Dict[str, Any]]:
    """Load records from file (json array or jsonl) or list."""
    if isinstance(source, (str, Path)):
        p = Path(source)
        if not p.is_file():
            raise FileNotFoundError(f"Input file not found at {source}")
        with p.open("r", encoding="utf-8") as f:
            content = f.read().strip()
            if content.startswith("["):
                raw_records = json.loads(content)
            else:
                raw_records = [json.loads(line) for line in content.splitlines() if line.strip()]
    elif isinstance(source, list):
        raw_records = source
    else:
        raise TypeError(f"Unsupported source type: {type(source)}")
    return [_normalize_record(r) for r in raw_records]


def generate_contrastive_dataset(
    train_input: Union[str, Path, List[Any]],
    test_input: Union[str, Path, List[Any]],
    output_base_dirs: Union[str, Path, List[Union[str, Path]]],
    num_shuffles_per_sample: int = 2,
    seed: int = 42,
    test_include_shuffles: bool = False,
    val_ratio: float = 0.1,
) -> Dict[str, Any]:
    """
    Generate contrastive train and pairwise evaluation data for CAA steering.
    """
    rng = random.Random(seed)
    train_records = _load_records(train_input)
    train_hallu = [r for r in train_records if parse_package_list(r.get("hallucinated_packages"))]
    print(f"Loaded {len(train_records)} train records ({len(train_hallu)} containing hallucinated packages).")

    if not train_hallu:
        raise ValueError("No train records contain hallucinated packages.")

    test_records = _load_records(test_input)
    test_hallu = [r for r in test_records if parse_package_list(r.get("hallucinated_packages"))]
    print(f"Loaded {len(test_records)} test records ({len(test_hallu)} containing hallucinated packages).")
    if not test_hallu:
        raise ValueError("No test records contain hallucinated packages.")

    def prompt_text(record: Dict[str, Any]) -> str:
        return "\n\n".join(
            str(record.get(key, "")).strip()
            for key in ("system_prompt", "user_prompt")
            if record.get(key)
        )

    # Exclude invalid/corrupted records with empty/nan prompt
    train_hallu = [
        r for r in train_hallu
        if str(r.get("user_prompt", "")).strip() not in ("nan", "None", "")
        and not str(r.get("user_prompt", "")).strip().endswith(": nan")
    ]
    test_hallu = [
        r for r in test_hallu
        if str(r.get("user_prompt", "")).strip() not in ("nan", "None", "")
        and not str(r.get("user_prompt", "")).strip().endswith(": nan")
    ]

    train_prompts = {prompt_text(record) for record in train_hallu} - {""}
    test_hallu = [r for r in test_hallu if prompt_text(r) not in train_prompts]
    test_prompts = {prompt_text(record) for record in test_hallu} - {""}
    overlap = train_prompts.intersection(test_prompts)

    train_hallu_train, train_hallu_val = split_train_val(
        train_hallu,
        val_ratio=val_ratio,
        seed=seed,
    )
    print(
        f"Train/validation split: {len(train_hallu_train)} train prompts, "
        f"{len(train_hallu_val)} validation prompts (val_ratio={val_ratio})."
    )

    train_orig, train_augmented, train_shuffled_only = process_samples_with_augmentations(
        base_samples=train_hallu_train,
        num_shuffles_per_sample=num_shuffles_per_sample,
        rng=rng,
    )
    val_orig, val_augmented, val_shuffled_only = process_samples_with_augmentations(
        base_samples=train_hallu_val,
        num_shuffles_per_sample=num_shuffles_per_sample,
        rng=rng,
    )
    print(
        f"Train set: {len(train_orig)} original samples "
        f"({len(train_augmented)} augmented, {len(train_shuffled_only)} shuffled only)."
    )
    print(
        f"Validation set: {len(val_orig)} original samples "
        f"({len(val_augmented)} augmented, {len(val_shuffled_only)} shuffled only)."
    )

    test_orig, test_augmented, test_shuffled_only = process_samples_with_augmentations(
        base_samples=test_hallu,
        num_shuffles_per_sample=num_shuffles_per_sample,
        rng=rng,
    )
    print(
        f"Test set: {len(test_orig)} original samples "
        f"({len(test_augmented)} augmented, {len(test_shuffled_only)} shuffled only)."
    )

    if isinstance(output_base_dirs, (str, Path)):
        output_base_dirs = [output_base_dirs]

    for base_dir in output_base_dirs:
        base_path = Path(base_dir)
        gen_dir = base_path / "generate"
        test_dir = base_path / "test"
        gen_dir.mkdir(parents=True, exist_ok=True)
        test_dir.mkdir(parents=True, exist_ok=True)

        for split_name, original, augmented, shuffled_only in (
            ("train", train_orig, train_augmented, train_shuffled_only),
            ("val", val_orig, val_augmented, val_shuffled_only),
        ):
            save_json(original, str(gen_dir / f"generate_dataset_{split_name}_original.json"))
            save_json(augmented, str(gen_dir / f"generate_dataset_{split_name}_augmented.json"))
            save_json(augmented, str(gen_dir / f"generate_dataset_{split_name}_shuffled.json"))
            save_json(shuffled_only, str(gen_dir / f"generate_dataset_{split_name}_shuffled_only.json"))

        # Keep the original train filenames for the activation extractor.
        save_json(train_orig, str(gen_dir / "generate_dataset_original.json"))
        save_json(train_augmented, str(gen_dir / "generate_dataset_augmented.json"))
        save_json(train_augmented, str(gen_dir / "generate_dataset_shuffled.json"))
        save_json(train_augmented, str(gen_dir / "generate_dataset.json"))
        save_json(train_shuffled_only, str(gen_dir / "generate_dataset_shuffled_only.json"))
        save_json(val_augmented, str(gen_dir / "generate_dataset_val.json"))

        print(f"Saved train datasets to {gen_dir}")

        path_test_orig = test_dir / "test_dataset_pairwise_original.json"
        save_json(test_orig, str(path_test_orig))

        path_test_aug = test_dir / "test_dataset_pairwise_augmented.json"
        save_json(test_augmented, str(path_test_aug))

        chosen_test = test_augmented if test_include_shuffles else test_orig
        path_test_main = test_dir / "test_dataset_pairwise.json"
        save_json(chosen_test, str(path_test_main))

        path_test_shuff_only = test_dir / "test_dataset_pairwise_shuffled_only.json"
        save_json(test_shuffled_only, str(path_test_shuff_only))

        print(f"Saved test datasets to {test_dir}")

    return {
        "train_original": train_orig,
        "train_augmented": train_augmented,
        "train_shuffled_only": train_shuffled_only,
        "val_original": val_orig,
        "val_augmented": val_augmented,
        "val_shuffled_only": val_shuffled_only,
        "test_original": test_orig,
        "test_augmented": test_augmented,
        "test_shuffled_only": test_shuffled_only,
    }


def main():
    parser = argparse.ArgumentParser(description="Generate Contrastive Datasets for CAA")
    parser.add_argument("--train_input", required=True, help="Path to the unlearning train master dataset")
    parser.add_argument("--test_input", required=True, help="Path to the unlearning test master dataset")
    parser.add_argument("--output_dir", required=True, help="Target contrastive data directory")
    parser.add_argument("--num_shuffles", type=int, default=2, help="Number of shuffles per sample")
    parser.add_argument("--val_ratio", type=float, default=0.1, help="Fraction of train prompts held out for validation")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--test_include_shuffles", action="store_true", help="Include shuffled pairs in the pairwise test dataset")
    args = parser.parse_args()

    generate_contrastive_dataset(
        train_input=args.train_input,
        test_input=args.test_input,
        output_base_dirs=[args.output_dir],
        num_shuffles_per_sample=args.num_shuffles,
        val_ratio=args.val_ratio,
        seed=args.seed,
        test_include_shuffles=args.test_include_shuffles,
    )


if __name__ == "__main__":
    main()