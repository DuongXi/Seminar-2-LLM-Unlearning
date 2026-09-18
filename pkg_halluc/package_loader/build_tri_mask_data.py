"""Dựng file JSONL tri-mask (retain/forget) từ các CSV kết quả hallucination cho GA/NPO."""
from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict
from pathlib import Path

import torch

from pkg_halluc.common.model_setup import build_model
from pkg_halluc.package_loader.generate_tri_mask import generate_tri_mask_dataset
from pkg_halluc.package_loader.unlearn_loader import PackageUnlearningDataset
from pkg_halluc.package_loader.utils import infer_model, load_csv_data

# Thư mục gốc repo: pkg_halluc/package_loader/<file>.py -> đi lên 2 cấp
REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"


def with_fallback(filenames: list[str], subfolder: str) -> list[Path]:
    """Ưu tiên data/<subfolder>/<file>, không có thì lấy data/<file>."""
    resolved = []
    for fn in filenames:
        sub = DATA_DIR / subfolder / fn
        resolved.append(sub if sub.exists() else DATA_DIR / fn)
    return resolved


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True, help="Tên/đường dẫn model")
    ap.add_argument("--dtype", default="auto", choices=["auto", "bfloat16", "float16", "float32"])
    ap.add_argument("--device-map", default="auto")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-length", type=int, default=2048)
    ap.add_argument(
        "--max-train-samples-per-split", type=int, default=None,
        help="Giới hạn số dòng retain forget",
    )
    ap.add_argument("--retain-file", required=True, help="Đường dẫn ghi file JSONL retain")
    ap.add_argument("--forget-file", required=True, help="Đường dẫn ghi file JSONL forget")
    ap.add_argument("--master-file", default=None, help="Ghi thêm 1 bản dump JSONL của toàn bộ dataset")
    return ap.parse_args()


def main() -> None:
    args = parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    result_files = ["LLM_LY_results.csv", "LLM_AT_results.csv", "SO_LY_results.csv", "SO_AT_results.csv"]
    result_paths = with_fallback(result_files, "Llama3_3_Python")

    model, tok = build_model(model_name_or_path=args.model, dtype=args.dtype, device_map=args.device_map)
    model.eval()

    source_df = load_csv_data([str(p) for p in result_paths])

    unlearn_ds = PackageUnlearningDataset(
        data_source=source_df,
        split_type="all",
        query_modes=[1, 2],
        tokenizer=tok,
        model_family=infer_model(args.model),
        max_length=args.max_length,
    )
    print(f"PackageUnlearningDataset khởi tạo thành công với {len(unlearn_ds)} records!")

    retain_records, forget_records = generate_tri_mask_dataset(
        dataset=unlearn_ds,
        tokenizer=tok,
        suffix="",
        max_length=args.max_length,
    )

    if args.max_train_samples_per_split is not None:
        retain_records = retain_records[: args.max_train_samples_per_split]
        forget_records = forget_records[: args.max_train_samples_per_split]

    if args.master_file:
        output_master = Path(args.master_file)
        output_master.parent.mkdir(parents=True, exist_ok=True)
        with output_master.open("w", encoding="utf-8") as handle:
            for rec in unlearn_ds.records:
                handle.write(json.dumps(asdict(rec), ensure_ascii=False) + "\n")

    for output_path, records in ((args.retain_file, retain_records), (args.forget_file, forget_records)):
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Tri-mask records: {len(retain_records)} retain | {len(forget_records)} forget")
    print(f"Đã ghi: {args.retain_file}, {args.forget_file}")


if __name__ == "__main__":
    main()
