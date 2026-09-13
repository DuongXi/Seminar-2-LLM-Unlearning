"""Script đánh giá hallucination dùng chung cho mọi method (base/ga/npo/ga_plain/npo_plain)."""
import argparse
import json
import os
import random
import sys

sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import generate_code
import generate_package_names
import aggregate_results
import package_detection


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--method", default="standard", choices=["standard"])
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--tag", required=True, help="Tên biến thể để đặt tên thư mục kết quả, vd base/ga/npo")
    ap.add_argument("--n_prompts", type=int, default=150)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--code_temp", type=float, default=0.7)
    ap.add_argument("--package_temp", type=float, default=0.01)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--pypi_csv", default=None, help="quét thêm bare import nếu truyền")
    ap.add_argument("--false_positive_csv", default=None)
    ap.add_argument("--package_modes", type=int, nargs="*", default=[1, 2], choices=[1, 2])
    ap.add_argument(
        "--eval_prompts_path", default=None,
        help="Đường dẫn tới prompts.jsonl để lấy mẫu; mặc định Data/prompts.jsonl (tương đối cwd)",
    )
    ap.add_argument(
        "--out_dir", default=None,
        help="Thư mục ghi kết quả; mặc định ./eval_runs/<tag> (tương đối cwd)",
    )
    args = ap.parse_args()
    args.package_modes = set(args.package_modes)

    random.seed(args.seed)

    data_path = os.path.join(os.getcwd(), "Data")
    full_prompts_path = args.eval_prompts_path or os.path.join(data_path, "prompts.jsonl")
    out_dir = args.out_dir or os.path.join(os.getcwd(), "eval_runs", args.tag)
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
    # elif args.method == "steering":
    #     import steering_eval_adapter as steer_adapter
    #     steer_adapter.generate_code_steering(sampled_path, code_out, args.model_path,
    #                                           args.steering_vector_path, ...)

    aggregate_results.combine_code_and_prompt(sampled_path, code_out, master_out)

    # mode bị bỏ qua vẫn ghi placeholder rỗng để package_detection.py không vỡ
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
        data_path, out_dir, args.tag, "verbose", "Python", "master.json", "",
    )
    print(f"[{args.tag}] Xong -> {out_dir}/FINAL_RESULTS.csv")

    if args.pypi_csv:
        import import_scan
        valid_names = import_scan.load_name_set(args.pypi_csv)
        fps = set()
        if args.false_positive_csv:
            import csv as _csv
            with open(args.false_positive_csv, encoding="utf-8", newline="") as f:
                for row in _csv.reader(f):
                    if len(row) >= 2 and row[1].strip():
                        fps.add(import_scan.normalize_python_package(row[1]))
        records, total_valid, total_hall = import_scan.scan_master_file(master_out, valid_names, fps)
        import_csv = os.path.join(out_dir, "import_scan_results.csv")
        import pandas as pd
        pd.DataFrame(records).to_csv(import_csv, index=False)
        total = total_valid + total_hall
        rate = 100.0 * total_hall / total if total else float("nan")
        print(f"[{args.tag}] Import-scan (ngữ cảnh 'bare import', bổ sung cho RQ4): "
              f"valid={total_valid} hallucinated={total_hall} rate={rate:.2f}% -> {import_csv}")


if __name__ == "__main__":
    main()