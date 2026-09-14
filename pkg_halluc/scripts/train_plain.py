"""Train GA-plain / NPO-plain -- no tri-mask

Early stopping: a --val_ratio slice of the *retain* set only is held out
as eval_dataset (the forget set is never held out). GAPlainTrainer /
NPOPlainTrainer's compute_loss (methods/plain_trainers.py) already
zeroes out the forget term whenever a batch has no "forget"-split rows,
so evaluating on a retain-only split makes Trainer's built-in eval_loss
equal to lambda_retain * L_retain -- a bounded signal for whether the
model's general (retain) behavior is degrading. deliberate: the
forget-side term is a *negative* CE (an ascent target), so it keeps
"improving" without bound and would make early stopping on the combined
loss meaningless -- it would never trigger a plateau
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch
from torch.utils.data import ConcatDataset, random_split
from transformers import EarlyStoppingCallback, TrainingArguments

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
    ap.add_argument(
        "--val_ratio", type=float, default=0.1,
        help="Fraction of the retain set held out as an eval split for early stopping "
        "(the forget set is never held out -- see module docstring below)",
    )
    ap.add_argument("--eval_steps", type=int, default=25, help="Evaluate (and save) every N optimizer steps")
    ap.add_argument("--early_stopping_patience", type=int, default=3, help="Stop after this many evals with no improvement")
    ap.add_argument(
        "--early_stopping_threshold", type=float, default=0.0,
        help="Minimum eval_loss decrease to count as an improvement",
    )
    ap.add_argument(
        "--disable_early_stopping", action="store_true",
        help="Train for the full num_train_epochs with no held-out eval split (old behavior)",
    )
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

    # Carve a held-out slice off the retain set for early stopping (see
    # module docstring for why retain-only, not forget). Falls back to
    # training the full retain set with early stopping off if the retain
    # set is too small to split meaningfully
    early_stopping_enabled = not args.disable_early_stopping
    retain_val_ds = None
    retain_train_ds = retain_ds
    if early_stopping_enabled:
        n_val = max(1, round(len(retain_ds) * args.val_ratio))
        if len(retain_ds) - n_val < 1:
            print(
                f"[train_plain] retain set too small ({len(retain_ds)} rows) to hold out "
                f"a val split at val_ratio={args.val_ratio} -- disabling early stopping."
            )
            early_stopping_enabled = False
        else:
            retain_train_ds, retain_val_ds = random_split(
                retain_ds,
                [len(retain_ds) - n_val, n_val],
                generator=torch.Generator().manual_seed(args.seed),
            )
            print(
                f"Early stopping on: retain_train={len(retain_train_ds)} "
                f"retain_val={len(retain_val_ds)} (val_ratio={args.val_ratio}), "
                f"eval_steps={args.eval_steps}, patience={args.early_stopping_patience}, "
                f"threshold={args.early_stopping_threshold}"
            )

    train_dataset = ConcatDataset([retain_train_ds, forget_ds])
    collator = DataCollatorForUnlearning(tok)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    eval_save_steps = args.eval_steps if early_stopping_enabled else 25
    training_args = TrainingArguments(
        output_dir=str(out_dir),
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.lr,
        weight_decay=0.01,
        warmup_steps=10,
        logging_strategy="steps",
        logging_dir=str(out_dir / "logs"),
        logging_steps=5,
        eval_strategy=("steps" if early_stopping_enabled else "no"),
        eval_steps=(eval_save_steps if early_stopping_enabled else None),
        save_strategy="steps",
        save_steps=eval_save_steps,
        save_total_limit=(3 if early_stopping_enabled else 2),
        load_best_model_at_end=early_stopping_enabled,
        metric_for_best_model=("eval_loss" if early_stopping_enabled else None),
        greater_is_better=(False if early_stopping_enabled else None),
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        prediction_loss_only=True,
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
        eval_dataset=(retain_val_ds if early_stopping_enabled else None),
        tokenizer=tok,
        data_collator=collator,
        lambda_retain=args.lambda_retain,
        lambda_forget=args.lambda_forget,
        vocab_size=vocab_size,
        callbacks=(
            [EarlyStoppingCallback(
                early_stopping_patience=args.early_stopping_patience,
                early_stopping_threshold=args.early_stopping_threshold,
            )]
            if early_stopping_enabled else []
        ),
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
        "early_stopping_enabled": early_stopping_enabled,
        "retain_train_samples": len(retain_train_ds),
        "retain_val_samples": (len(retain_val_ds) if early_stopping_enabled else 0),
        "val_ratio": (args.val_ratio if early_stopping_enabled else None),
        "eval_steps": (eval_save_steps if early_stopping_enabled else None),
        "early_stopping_patience": (args.early_stopping_patience if early_stopping_enabled else None),
        "early_stopping_threshold": (args.early_stopping_threshold if early_stopping_enabled else None),
        "best_eval_loss": (trainer.state.best_metric if early_stopping_enabled else None),
        "best_checkpoint": (trainer.state.best_model_checkpoint if early_stopping_enabled else None),
        "total_steps_run": trainer.state.global_step,
    }
    with open(out_dir / "training_info.json", "w", encoding="utf-8") as f:
        json.dump(training_info, f, indent=2)

    print("Done")


if __name__ == "__main__":
    main()
