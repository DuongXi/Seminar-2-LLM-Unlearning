"""
Generate training data using package_loader
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
import random

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
from pkg_halluc.common.model_setup import build_model
from pkg_halluc.common.model_presets import MODEL_PRESETS, resolve_model_suffix

from pkg_halluc.package_loader.generate_tri_mask import (
    generate_tri_mask_dataset,
)
from pkg_halluc.package_loader.unlearn_loader import PackageUnlearningDataset
from pkg_halluc.package_loader.utils import (
    infer_model,
    load_csv_data,
    is_preprocessed,
    save_tokenized_records,
    setup_tokenizer,
)

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model_path", required=True, help="HF model name or path")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32", "auto"])
    ap.add_argument("--device_map", default="auto")
    ap.add_argument("--max_length", type=int, default=2048)
    ap.add_argument("--main_path", default="", help="Path to model data directory (from model_config)")
    ap.add_argument("--result_files", nargs="*", default=[])
    ap.add_argument("--out_master", default="")
    ap.add_argument("--out_val_master", default="", help="Path to save held-out validation master records (pre-tri-mask)")
    ap.add_argument("--out_retain_tri_mask", default="")
    ap.add_argument("--out_forget_tri_mask", default="")
    ap.add_argument("--val_ratio", type=float, default=0.0, help="Fraction of retain prompts to hold out for validation")
    ap.add_argument("--out_val_retain_tri_mask", default="", help="Path to output validation retain tri-mask records")
    ap.add_argument("--out_plain_train", default="", help="Path to save pre-tri-mask tokenized train records")
    ap.add_argument("--out_plain_val", default="", help="Path to save pre-tri-mask tokenized val records")
    ap.add_argument("--out_plain_test", default="", help="Path to save pre-tri-mask tokenized test records")
    ap.add_argument("--out_plain_retain", default="", help="Path to save pre-tri-mask tokenized retain records")
    ap.add_argument("--out_plain_forget", default="", help="Path to save pre-tri-mask tokenized forget records")
    ap.add_argument("--max_train_samples_per_split", type=int, default=None, help="Cap samples per split for smoke test")
    args = ap.parse_args()

    suffix = resolve_model_suffix(args.model_path)
    if args.main_path:
        main_dir = Path(args.main_path)
        if not args.result_files:
            args.result_files = [
                str(main_dir / "LLM_LY_results.csv"),
                str(main_dir / "LLM_AT_results.csv"),
                str(main_dir / "SO_LY_results.csv"),
                str(main_dir / "SO_AT_results.csv"),
            ]
        if not args.out_retain_tri_mask:
            args.out_retain_tri_mask = str(main_dir / "tri_mask" / f"npo_retain_tok{suffix}.jsonl")
        if not args.out_forget_tri_mask:
            args.out_forget_tri_mask = str(main_dir / "tri_mask" / f"npo_forget_tok{suffix}.jsonl")
        if not args.out_val_retain_tri_mask and args.val_ratio > 0.0:
            args.out_val_retain_tri_mask = str(main_dir / "tri_mask" / f"npo_val_retain_tok{suffix}.jsonl")
        if not args.out_master:
            args.out_master = str(main_dir / "master_train.json")
        if not args.out_val_master and args.val_ratio > 0.0:
            args.out_val_master = str(main_dir / "master_val.json")
        if not args.out_plain_train:
            args.out_plain_train = str(main_dir / "plain" / f"plain_train_tok{suffix}.jsonl")
        if not args.out_plain_val and args.val_ratio > 0.0:
            args.out_plain_val = str(main_dir / "plain" / f"plain_val_tok{suffix}.jsonl")
        if not args.out_plain_retain:
            args.out_plain_retain = str(main_dir / "plain" / f"plain_retain_tok{suffix}.jsonl")
        if not args.out_plain_forget:
            args.out_plain_forget = str(main_dir / "plain" / f"plain_forget_tok{suffix}.jsonl")
    else:
        if not args.out_retain_tri_mask:
            args.out_retain_tri_mask = str(REPO_ROOT / "data" / "tri_mask" / f"npo_retain_tok{suffix}.jsonl")
        if not args.out_forget_tri_mask:
            args.out_forget_tri_mask = str(REPO_ROOT / "data" / "tri_mask" / f"npo_forget_tok{suffix}.jsonl")
        if not args.out_val_retain_tri_mask and args.val_ratio > 0.0:
            args.out_val_retain_tri_mask = str(REPO_ROOT / "data" / "tri_mask" / f"npo_val_retain_tok{suffix}.jsonl")
        if not args.out_plain_train:
            args.out_plain_train = str(REPO_ROOT / "data" / "plain" / f"plain_train_tok{suffix}.jsonl")
        if not args.out_plain_val and args.val_ratio > 0.0:
            args.out_plain_val = str(REPO_ROOT / "data" / "plain" / f"plain_val_tok{suffix}.jsonl")
        if not args.out_plain_retain:
            args.out_plain_retain = str(REPO_ROOT / "data" / "plain" / f"plain_retain_tok{suffix}.jsonl")
        if not args.out_plain_forget:
            args.out_plain_forget = str(REPO_ROOT / "data" / "plain" / f"plain_forget_tok{suffix}.jsonl")

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    model, tok = build_model(
        model_name_or_path=args.model_path,
        dtype=args.dtype,
        device_map=args.device_map,
    )
    model.eval()
    resolved_path = MODEL_PRESETS.get(args.model_path, args.model_path)
    tok = setup_tokenizer(resolved_path)

    out_master_path = Path(args.out_master) if args.out_master else None
    out_val_master_path = Path(args.out_val_master) if args.out_val_master else None

    # Check if both master train and val already exist
    reusing_master = False
    if out_master_path and out_master_path.is_file() and is_preprocessed(out_master_path):
        if out_val_master_path and out_val_master_path.is_file() and is_preprocessed(out_val_master_path):
            print(f"Found existing master train ({out_master_path}) and master val ({out_val_master_path}). Reusing both!")
            train_ds = PackageUnlearningDataset(
                data_source=str(out_master_path),
                split_type="all",
                query_modes=[1, 2],
                tokenizer=tok,
                model_family=infer_model(args.model_path),
                max_length=args.max_length,
            )
            val_ds = PackageUnlearningDataset(
                data_source=str(out_val_master_path),
                split_type="retain",
                query_modes=[1, 2],
                tokenizer=tok,
                model_family=infer_model(args.model_path),
                max_length=args.max_length,
            )
            reusing_master = True
        else:
            print(f"Found existing dataset pool at {out_master_path}. Loading it for splitting/processing.")
            unlearn_ds = PackageUnlearningDataset(
                data_source=str(out_master_path),
                split_type="all",
                query_modes=[1, 2],
                tokenizer=tok,
                model_family=infer_model(args.model_path),
                max_length=args.max_length,
            )
    else:
        # Load from result_files
        if not args.result_files:
            raise ValueError("No master dataset or result_files provided to build_data!")
        source_df = load_csv_data(args.result_files)
        unlearn_ds = PackageUnlearningDataset(
            data_source=source_df,
            split_type="all",
            query_modes=[1, 2],
            tokenizer=tok,
            model_family=infer_model(args.model_path),
            max_length=args.max_length,
        )

    val_retain_tri_mask_records = []
    if reusing_master:
        print(f"Loaded {len(train_ds)} train records and {len(val_ds)} val records.")
        retain_tri_mask_records, forget_tri_mask_records = generate_tri_mask_dataset(
            dataset=train_ds,
            tokenizer=tok,
            suffix=suffix,
            max_length=args.max_length,
        )
        if args.out_val_retain_tri_mask and len(val_ds) > 0:
            val_retain_tri_mask_records, _ = generate_tri_mask_dataset(
                dataset=val_ds,
                tokenizer=tok,
                suffix=suffix,
                max_length=args.max_length,
            )
    elif args.val_ratio > 0.0:
        train_ds, val_ds = unlearn_ds.split_train_val(
            val_ratio=args.val_ratio,
            seed=args.seed,
            split_retain_only=True,
        )
        print(f"Split dataset with val_ratio={args.val_ratio}: train={len(train_ds)}, val={len(val_ds)}")

        # Save pre-tri-mask master datasets to disk
        if out_master_path:
            out_master_path.parent.mkdir(parents=True, exist_ok=True)
            train_ds.save_to_file(out_master_path)
            print(f"Saved master train records ({len(train_ds)}) to {out_master_path}")
        if out_val_master_path and len(val_ds) > 0:
            out_val_master_path.parent.mkdir(parents=True, exist_ok=True)
            val_ds.save_to_file(out_val_master_path)
            print(f"Saved master val records ({len(val_ds)}) to {out_val_master_path}")

        retain_tri_mask_records, forget_tri_mask_records = generate_tri_mask_dataset(
            dataset=train_ds,
            tokenizer=tok,
            suffix=suffix,
            max_length=args.max_length,
        )
        if args.out_val_retain_tri_mask and len(val_ds) > 0:
            val_retain_tri_mask_records, _ = generate_tri_mask_dataset(
                dataset=val_ds,
                tokenizer=tok,
                suffix=suffix,
                max_length=args.max_length,
            )
    else:
        print(f"Loaded {len(unlearn_ds)} records from dataset (val_ratio=0).")
        if out_master_path and not reusing_master:
            out_master_path.parent.mkdir(parents=True, exist_ok=True)
            unlearn_ds.save_to_file(out_master_path)
        retain_tri_mask_records, forget_tri_mask_records = generate_tri_mask_dataset(
            dataset=unlearn_ds,
            tokenizer=tok,
            suffix=suffix,
            max_length=args.max_length,
        )

    if args.max_train_samples_per_split is not None:
        retain_tri_mask_records = retain_tri_mask_records[: args.max_train_samples_per_split]
        forget_tri_mask_records = forget_tri_mask_records[: args.max_train_samples_per_split]
        if val_retain_tri_mask_records:
            val_retain_tri_mask_records = val_retain_tri_mask_records[: args.max_train_samples_per_split]

    # Save retain, forget, and optional val tri-mask datasets
    save_specs = [
        (args.out_retain_tri_mask, retain_tri_mask_records),
        (args.out_forget_tri_mask, forget_tri_mask_records),
    ]
    if args.out_val_retain_tri_mask and val_retain_tri_mask_records:
        save_specs.append((args.out_val_retain_tri_mask, val_retain_tri_mask_records))

    for output_path, records in save_specs:
        if not output_path:
            continue
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    val_info = f" | {len(val_retain_tri_mask_records)} val_retain" if val_retain_tri_mask_records else ""
    print(f"Tri-mask records saved: {len(retain_tri_mask_records)} retain | {len(forget_tri_mask_records)} forget{val_info}")

    # Save pre-tri-mask pointwise tokenized datasets (sample_id, split_type, mode, input_ids, attention_mask, labels)
    target_train_ds = train_ds if (reusing_master or args.val_ratio > 0.0) else unlearn_ds
    target_val_ds = val_ds if (reusing_master or args.val_ratio > 0.0) else None

    if args.out_plain_train and target_train_ds is not None:
        save_tokenized_records(target_train_ds.records, args.out_plain_train, tokenizer=tok, max_length=args.max_length)

    if args.out_plain_retain and target_train_ds is not None:
        recs = [r for r in target_train_ds.records if (r.split_type if hasattr(r, "split_type") else r.get("split_type")) == "retain"]
        save_tokenized_records(recs, args.out_plain_retain, tokenizer=tok, max_length=args.max_length)

    if args.out_plain_forget and target_train_ds is not None:
        recs = [r for r in target_train_ds.records if (r.split_type if hasattr(r, "split_type") else r.get("split_type")) == "forget"]
        save_tokenized_records(recs, args.out_plain_forget, tokenizer=tok, max_length=args.max_length)

    if args.out_plain_val and target_val_ds is not None and len(target_val_ds) > 0:
        save_tokenized_records(target_val_ds.records, args.out_plain_val, tokenizer=tok, max_length=args.max_length)

    if args.out_plain_test:
        test_file = Path(args.out_master).parent / "master_test.json" if args.out_master else None
        if test_file and test_file.is_file():
            test_ds = PackageUnlearningDataset(str(test_file), split_type="all", tokenizer=tok, max_length=args.max_length)
            save_tokenized_records(test_ds.records, args.out_plain_test, tokenizer=tok, max_length=args.max_length)


if __name__ == "__main__":
    main()