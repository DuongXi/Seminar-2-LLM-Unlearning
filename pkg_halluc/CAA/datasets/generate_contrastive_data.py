"""
Generate Contrastive Data for CAA Package Hallucination Steering.
Augments the original samples by ADDING shuffled permutations of package sequences.
Target behavior: Non-hallucination / Truthful (recommending only valid packages).
"""

import argparse
import json
import os
import random
import sys
from typing import Any, Dict, List, Optional, Tuple, Set

if sys.stdout.encoding != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(SCRIPT_DIR)))


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

    for idx, pkg in enumerate(packages):
        separator = ", " if idx > 0 else ""
        current_char += len(separator)

        char_start = current_char
        char_end = char_start + len(pkg)
        current_char = char_end

        packages_info.append(
            {
                "package_name": pkg,
                "is_hallucinated": pkg in hallucinated_set,
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
    Returns:
    - original_records: list of original un-shuffled samples
    - shuffled_only_records: list of all generated shuffled variations
    - combined_records: original + shuffled variations combined and shuffled
    """
    original_records: List[Dict[str, Any]] = []
    shuffled_only_records: List[Dict[str, Any]] = []

    for item in base_samples:
        valid_pkgs = [p.strip() for p in item.get("valid_packages", []) if p.strip()]
        hall_pkgs = [p.strip() for p in item.get("hallucinated_packages", []) if p.strip()]
        hall_set = set(hall_pkgs)

        rejected_str = item.get("rejected") or item.get("completion", "")
        pkgs_with_hall = [p.strip() for p in rejected_str.split(",") if p.strip()]
        if not pkgs_with_hall:
            pkgs_with_hall = list(valid_pkgs) + list(hall_pkgs)

        sys_prompt = item.get("system_prompt", "").strip()
        usr_prompt = item.get("user_prompt", "").strip()
        question = f"{sys_prompt}\n\n{usr_prompt}" if sys_prompt else usr_prompt

        # Original Sample
        orig_comp, orig_pkgs_info = locate_packages(pkgs_with_hall, hall_set)
        orig_record = {
            "sample_id": item["sample_id"],
            "base_sample_id": item["sample_id"],
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
                "sample_id": f"{item['sample_id']}_shuff_{shuffles_added}",
                "base_sample_id": item["sample_id"],
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
                "mode": item.get("mode", 1),
            }
            shuffled_only_records.append(shuff_record)

    combined_records = list(original_records) + list(shuffled_only_records)
    rng.shuffle(combined_records)

    return original_records, combined_records


def generate_dataset(
    input_file: str,
    output_base_dirs: List[str],
    n_test: int = 50,
    num_shuffles_per_sample: int = 2,
    seed: int = 42,
    test_include_shuffles: bool = False,
) -> None:
    rng = random.Random(seed)

    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input file not found at {input_file}")

    with open(input_file, "r", encoding="utf-8") as f:
        records = [json.loads(line.strip()) for line in f if line.strip()]

    samples = [r for r in records if r.get("hallucinated_packages")]
    print(f"[*] Found {len(samples)} base samples containing hallucinated packages.")

    # Train / Test Split by base sample_id
    indices = list(range(len(samples)))
    rng.shuffle(indices)

    test_indices = set(indices[-n_test:]) if n_test > 0 else set()
    train_base_samples = [samples[i] for i in indices if i not in test_indices]
    test_base_samples = [samples[i] for i in indices if i in test_indices]

    print(f"[*] Base split: {len(train_base_samples)} train samples | {len(test_base_samples)} test samples")

    gen_orig, gen_augmented = process_samples_with_augmentations(
        base_samples=train_base_samples,
        num_shuffles_per_sample=num_shuffles_per_sample,
        rng=rng,
    )
    print(f"[*] Train set: {len(gen_orig)} original samples z{len(gen_augmented)} with variations).")

    test_orig, test_augmented = process_samples_with_augmentations(
        base_samples=test_base_samples,
        num_shuffles_per_sample=num_shuffles_per_sample,
        rng=rng,
    )
    print(f"[*] Test set: {len(test_orig)} original samples ({len(test_augmented)} with variations).")

    # 4. Save to target output directories
    for base_dir in output_base_dirs:
        gen_dir = os.path.join(base_dir, "generate")
        test_dir = os.path.join(base_dir, "test")
        os.makedirs(gen_dir, exist_ok=True)
        os.makedirs(test_dir, exist_ok=True)

        # Train datasets:
        path_gen_orig = os.path.join(gen_dir, "generate_dataset_original.json")
        save_json(gen_orig, path_gen_orig)

        path_gen_aug = os.path.join(gen_dir, "generate_dataset_augmented.json")
        save_json(gen_augmented, path_gen_aug)

        print(f"[+] Saved train dataset")

        # Test datasets:
        path_test_orig = os.path.join(test_dir, "test_dataset_ab_original.json")
        save_json(test_orig, path_test_orig)

        path_test_aug = os.path.join(test_dir, "test_dataset_ab_augmented.json")
        save_json(test_augmented, path_test_aug)

        chosen_test = test_augmented if test_include_shuffles else test_orig
        path_test_main = os.path.join(test_dir, "test_dataset_ab.json")
        save_json(chosen_test, path_test_main)

        path_test_shuff_alias = os.path.join(test_dir, "test_dataset_ab_shuffled.json")
        save_json(test_augmented, path_test_shuff_alias)

        print(f"[+] Saved test dataset")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate CAA package hallucination dataset by adding shuffled package permutations to original samples."
    )
    parser.add_argument(
        "--input",
        type=str,
        default="data/Llama3_1B/train_test_split/master_train.json",
        help="Path to master_train.json input file",
    )
    parser.add_argument(
        "--num_shuffles",
        type=int,
        default=2,
        help="Number of shuffled permutations to add per original sample (default: 2)",
    )
    parser.add_argument(
        "--n_test",
        type=int,
        default=50,
        help="Number of base samples reserved for test split (default: 50)",
    )
    parser.add_argument(
        "--test_include_shuffles",
        action="store_true",
        default=False,
        help="Include shuffled variations in the default test_dataset_ab.json (default: False, use original)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for shuffling and split")
    args = parser.parse_args()

    default_base_dirs = [
        SCRIPT_DIR,
    ]

    generate_dataset(
        input_file=args.input,
        output_base_dirs=default_base_dirs,
        n_test=args.n_test,
        num_shuffles_per_sample=args.num_shuffles,
        seed=args.seed,
        test_include_shuffles=args.test_include_shuffles,
    )