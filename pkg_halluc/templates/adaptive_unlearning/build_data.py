"""
Generate training data using package_loader
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import random
import sys
import torch


# Dynamic sys.path setup to find project or pkg_halluc modules
def _setup_sys_path():
    current = Path(__file__).resolve()
    for candidate in [Path.cwd(), current.parent, *current.parents]:
        proj_pkg = candidate / "project" / "pkg_halluc"
        if proj_pkg.is_dir():
            for p in [str(candidate), str(candidate / "project")]:
                if p not in sys.path:
                    sys.path.insert(0, p)
            break
        pkg = candidate / "pkg_halluc"
        if pkg.is_dir():
            if str(candidate) not in sys.path:
                sys.path.insert(0, str(candidate))
            break

_setup_sys_path()

from dataclasses import asdict
from pkg_halluc.model_setup import build_model
from pkg_halluc.package_loader.config import (
    PACKAGE_PREFIX_1,
    PACKAGE_PREFIX_2,
    PACKAGE_SYSTEM_PROMPT_1,
    PACKAGE_SYSTEM_PROMPT_2,
)
from pkg_halluc.package_loader.generate_tri_mask import (
    generate_tri_mask_dataset,
)
from pkg_halluc.package_loader.unlearn_loader import PackageUnlearningDataset
from pkg_halluc.package_loader.utils import (
    infer_model,
    load_csv_data,
    normalize_package,
    parse_package_list,
)

load_source_data = load_csv_data

def str2bool(v):
    if isinstance(v, bool):
        return v
    if str(v).lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif str(v).lower() in ("no", "false", "f", "n", "0"):
        return False
    return True


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model_path", required=True, help="HF model name or path")
    ap.add_argument("--tri_mask", type=str2bool, nargs="?", const=True, default=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32", "auto"])
    ap.add_argument("--device_map", default="auto")
    ap.add_argument("--max_length", type=int, default=2048)
    ap.add_argument(
        "--max_train_samples_per_split", type=int, default=None,
        help="Cap retain/forget datasets to at most this many rows each (smoke-test option)",
    )
    ap.add_argument("--master_files", nargs="*", default=[])
    ap.add_argument("--result_files", nargs="*", default=[])
    ap.add_argument("--seed_prompts", nargs="*", default=[os.path.join("New_Data_Set", "prompts.jsonl")])
    ap.add_argument("--extra_prompts", nargs="*", default=[os.path.join("Package_Hallucination_Testing", "Data", "prompts.jsonl")])
    ap.add_argument("--pypi_csv", default=os.path.join("Package_Hallucination_Testing", "Data", "pypi_package_names.csv"))
    ap.add_argument("--fp_csv", default=os.path.join("Package_Hallucination_Testing", "Data", "false_positive_packages.csv"))
    ap.add_argument("--out_master", default=os.path.join("New_Data_Set", "master.json"))
    ap.add_argument("--out_retain", default=os.path.join("New_Data_Set", "npo_retain_tok_1B.jsonl"))
    ap.add_argument("--out_forget", default=os.path.join("New_Data_Set", "npo_forget_tok_1B.jsonl"))
    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    model, tok = build_model(
        model_name_or_path=args.model_path,
        dtype=args.dtype,
        device_map=args.device_map,
    )
    model.eval()

    # Load source dataframe
    source_df = load_csv_data(args.result_files)
    
    unlearn_ds = PackageUnlearningDataset(
        data_source=source_df,
        split_type="all",
        query_modes=[1, 2],
        tokenizer=tok,
        model_family=infer_model(args.model_path),
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
        # smoke-test cap: deterministic head-slice, not a random sample
        retain_records = retain_records[: args.max_train_samples_per_split]
        forget_records = forget_records[: args.max_train_samples_per_split]

    # Save master dataset
    if args.out_master:
        output_master = Path(args.out_master)
        output_master.parent.mkdir(parents=True, exist_ok=True)
        with output_master.open("w", encoding="utf-8") as handle:
            if len(unlearn_ds.records) > 0:
                for rec in unlearn_ds.records:
                    handle.write(json.dumps(asdict(rec), ensure_ascii=False) + "\n")
            else:
                for _, row in source_df.iterrows():
                    row_dict = {k: v for k, v in row.to_dict().items() if not k.startswith("_")}
                    handle.write(json.dumps(row_dict, ensure_ascii=False) + "\n")

    # Save retain and forget tri-mask datasets
    for output_path, records in (
        (args.out_retain, retain_records),
        (args.out_forget, forget_records),
    ):
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"Tri-mask records: {len(retain_records)} retain | {len(forget_records)} forget")

if __name__ == "__main__":
    main()