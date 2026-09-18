"""Eval 1 checkpoint: sinh code, hỏi package (mode 1, 2), chấm hallucination và quét import trần."""
import argparse
import csv
import json
import os
import random

import pandas as pd

from pkg_halluc.evaluation.detection import (
    aggregate_results,
    import_scan,
    package_detection,
)
from pkg_halluc.evaluation.generation import generate_code, generate_package_names


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--method", default="standard", choices=["standard"])
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--tag", required=True, help="Tên biến thể, dùng để đặt tên thư mục kết quả")
    ap.add_argument("--n_prompts", type=int, default=150)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--code_temp", type=float, default=0.7)
    ap.add_argument("--package_temp", type=float, default=0.01)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--pypi_csv", required=True, help="Danh sách package PyPI, dùng cho package_detection và quét bare import")
    ap.add_argument("--false_positive_csv", required=True, help="CSV các tên được coi là false positive")
    ap.add_argument("--package_modes", type=int, nargs="*", default=[1, 2], choices=[1, 2])
    ap.add_argument(
        "--eval_prompts_path", required=True,
        help="Đường dẫn tới prompts.jsonl để lấy eval set",
    )
    ap.add_argument("--out_dir", required=True, help="Thư mục ghi kết quả")
    args = ap.parse_args()
    args.package_modes = set(args.package_modes)

    random.seed(args.seed)

    full_prompts_path = args.eval_prompts_path
    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    with open(full_prompts_path, encoding="utf-8") as f:
        all_prompts = [json.loads(l) for l in f if l.strip()]
    sample = random.sample(all_prompts, args.n_prompts) if args.n_prompts < len(all_prompts) else all_prompts
    sampled_path = os.path.join(out_dir, "prompts.jsonl")
    with open(sampled_path, "w", encoding="utf-8") as f:
        for p in sample:
            f.write(json.dumps(p) + "\n")

    code_out = os.path.join(out_dir, "code.json")
    master_out = os.path.join(out_dir, "master.json")

    print(f"[{args.tag}] Đang sinh code cho {len(sample)} prompt...")
    generate_code.generate_code(
        sampled_path, code_out, args.model_path, "Python",
        args.code_temp, 20, 0.9, args.batch_size, False,
    )
    aggregate_results.combine_code_and_prompt(sampled_path, code_out, master_out)

    n_rows = sum(1 for _ in open(master_out, encoding="utf-8"))
    for mode in (1, 2):
        pkg_out = os.path.join(out_dir, f"packages_{mode}.json")
        if mode not in args.package_modes:
            with open(pkg_out, "w", encoding="utf-8") as f:
                for _ in range(n_rows):
                    json.dump({"prefix": "", "input": "", "full_prompt": "", "response": ""}, f)
                    f.write("\n")
            continue
        print(f"[{args.tag}] Đang hỏi package, mode {mode}...")
        generate_package_names.generate_packages(
            mode, master_out, pkg_out, args.model_path, "Python",
            args.package_temp, 20, 0.9, False,
        )

    package_detection.detect_packages(
        os.path.dirname(args.pypi_csv), out_dir, args.tag, "verbose", "Python", "master.json", "",
    )
    print(f"[{args.tag}] Xong -> {out_dir}/FINAL_RESULTS.csv")

    valid_names = import_scan.load_name_set(args.pypi_csv)
    fps = set()
    with open(args.false_positive_csv, encoding="utf-8", newline="") as f:
        for row in csv.reader(f):
            if len(row) >= 2 and row[1].strip():
                fps.add(import_scan.normalize_python_package(row[1]))
    records, total_valid, total_hall = import_scan.scan_master_file(master_out, valid_names, fps)
    import_csv = os.path.join(out_dir, "import_scan_results.csv")
    pd.DataFrame(records).to_csv(import_csv, index=False)
    total = total_valid + total_hall
    rate = 100.0 * total_hall / total if total else float("nan")
    print(f"[{args.tag}] Import-scan (ngữ cảnh 'bare import', bổ sung cho RQ4): "
          f"valid={total_valid} hallucinated={total_hall} rate={rate:.2f}% -> {import_csv}")


if __name__ == "__main__":
    main()
