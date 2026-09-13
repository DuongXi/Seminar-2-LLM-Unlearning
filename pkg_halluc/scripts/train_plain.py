"""Train GA-plain / NPO-plain -- no tri-mask, no Adaptive Unlearning code.

Counterpart to methods/ga.py and methods/npo.py (which subprocess into AU's
vendored train.py). This script is entirely this project's own code: loads
the base model via pkg_halluc.model_setup.build_model, and builds retain/
forget sets via PackageUnlearningDataset with return_format="pointwise"
(masks only the prompt, keeps every response token as a label -- no
per-token tri-mask).

Invoked via `python -m pkg_halluc.scripts.train_plain` (see
methods/_finetune_plain_common.py); not meant to be run by hand.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch
from torch.utils.data import ConcatDataset
from transformers import TrainingArguments

from pkg_halluc.methods.plain_trainers import GAPlainTrainer, NPOPlainTrainer
from pkg_halluc.model_setup import build_model
from pkg_halluc.package_loader.unlearn_loader import PackageUnlearningDataset
from pkg_halluc.package_loader.utils import (
    DataCollatorForUnlearning,
    infer_model,
    load_csv_data,
)


def _apply_lora(model, lora_rank: int):
    """Wrap model with LoRA adapters via PEFT (own reimplementation of AU's
    utils.py::apply_lora config, kept independent of the AU repo)."""
    from peft import LoraConfig, TaskType, get_peft_model

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_rank,
        lora_alpha=lora_rank * 2,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.enable_input_require_grads()  # required for gradient checkpointing with frozen base weights
    model.print_trainable_parameters()
    return model


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model_path", required=True, help="Local path to the base model")
    ap.add_argument("--model_name", required=True, help="HF id / preset name, used for model-family inference")
    ap.add_argument("--loss_function", required=True, choices=["ga_plain", "npo_plain"])
    ap.add_argument("--save_tag", required=True)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--num_train_epochs", type=int, default=3)
    ap.add_argument("--lambda_retain", type=float, default=1.0)
    ap.add_argument("--lambda_forget", type=float, default=0.5)
    ap.add_argument("--use_lora", action="store_true", help="Train a LoRA adapter instead of full fine-tuning")
    ap.add_argument("--lora_rank", type=int, default=16)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32", "auto"])
    ap.add_argument("--max_length", type=int, default=2048)
    ap.add_argument(
        "--max_samples_per_split", type=int, default=None,
        help="Cap retain/forget datasets to at most this many rows each (smoke-test option)",
    )
    ap.add_argument("--result_files", nargs="+", required=True, help="Same results CSVs the tri-mask build uses")
    ap.add_argument("--out_dir", required=True)
    return ap.parse_args()


def main() -> None:
    args = parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    model, tok = build_model(
        model_name_or_path=args.model_path,
        dtype=args.dtype,
        device_map=None,
    )
    if args.use_lora:
        model = _apply_lora(model, args.lora_rank)

    source_df = load_csv_data(list(args.result_files))
    family = infer_model(args.model_name)

    retain_ds = PackageUnlearningDataset(
        data_source=source_df,
        split_type="retain",
        query_modes=[1, 2],
        tokenizer=tok,
        model_family=family,
        max_length=args.max_length,
        return_format="pointwise",
    )
    forget_ds = PackageUnlearningDataset(
        data_source=source_df,
        split_type="forget",
        query_modes=[1, 2],
        tokenizer=tok,
        model_family=family,
        max_length=args.max_length,
        return_format="pointwise",
    )

    if args.max_samples_per_split is not None:
        # smoke-test cap: deterministic head-slice, applied to both splits independently
        retain_ds.records = retain_ds.records[: args.max_samples_per_split]
        forget_ds.records = forget_ds.records[: args.max_samples_per_split]

    print(f"PackageUnlearningDataset (plain): retain={len(retain_ds)} forget={len(forget_ds)}")

    train_dataset = ConcatDataset([retain_ds, forget_ds])
    collator = DataCollatorForUnlearning(tok)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=str(out_dir),
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.lr,
        weight_decay=0.01,
        warmup_steps=10,
        logging_strategy="steps",
        logging_dir=str(out_dir / "logs"),
        logging_steps=5,
        eval_strategy="no",
        save_strategy="steps",
        save_steps=25,
        save_total_limit=2,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=16,
        gradient_checkpointing=True,
        bf16=(args.dtype == "bfloat16"),
        fp16=(args.dtype == "float16"),
        max_grad_norm=1.0,
        lr_scheduler_type="constant",
        report_to=["tensorboard"],
        remove_unused_columns=False,
        seed=args.seed,
    )

    vocab_size = len(tok)
    trainer_cls = GAPlainTrainer if args.loss_function == "ga_plain" else NPOPlainTrainer
    trainer = trainer_cls(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tok,
        data_collator=collator,
        lambda_retain=args.lambda_retain,
        lambda_forget=args.lambda_forget,
        vocab_size=vocab_size,
    )

    print(f"Training {args.loss_function} (plain, no tri-mask)...")
    trainer.train()

    print("Saving model...")
    if args.use_lora:
        # Adapter only. Eval loads this via AU's utils.py::load_model_auto(),
        # which auto-detects the adapter and merges with the base model in
        # memory at load time -- no full checkpoint needs to be on disk.
        trainer.model.save_pretrained(str(out_dir))
    else:
        trainer.save_model(str(out_dir))
    tok.save_pretrained(str(out_dir))

    training_info = {
        "loss_function": args.loss_function,
        "tri_mask": False,
        "lambda_retain": args.lambda_retain,
        "lambda_forget": args.lambda_forget,
        "lr": args.lr,
        "num_train_epochs": args.num_train_epochs,
        "use_lora": args.use_lora,
        "lora_rank": args.lora_rank if args.use_lora else None,
        "seed": args.seed,
        "vocab_size": vocab_size,
        "retain_samples": len(retain_ds),
        "forget_samples": len(forget_ds),
        "result_files": list(args.result_files),
    }
    with open(out_dir / "training_info.json", "w", encoding="utf-8") as f:
        json.dump(training_info, f, indent=2)

    print("Done")


if __name__ == "__main__":
    main()
