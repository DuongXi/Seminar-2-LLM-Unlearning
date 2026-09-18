# AU
import os
import json
import wandb
import argparse
import torch
import datetime

from datasets import Dataset
from typing import Tuple

from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments

from pkg_halluc.common.au_utils import (
    load_toklevel_files,
    TokLevelCollator,
    load_hf_token,
    apply_lora,
)
from pkg_halluc.training.au.ga_trainer import GradientAscentTrainer
from pkg_halluc.training.au.npo_trainer import NPOTrainer


def resolve_dtype(dtype_arg: str) -> torch.dtype:
    """"auto" chọn bfloat16 nếu GPU hỗ trợ, không thì float16."""
    if dtype_arg == "bfloat16":
        return torch.bfloat16
    if dtype_arg == "float16":
        return torch.float16
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def main():
    args = parse_arguments()
    save_string = args.save_string
    if args.resume_from_checkpoint:
        save_string = f"{save_string}_reset"
    loss_function = args.loss_function

    if args.seed is None:
        import time

        random_seed = int(time.time() * 1000) % 2**32
    else:
        random_seed = args.seed

    datetime_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    if args.use_wandb:
        wandb.init(
            project="unlearning-fine-tune",
            name=f"{save_string}_{datetime_str}",
            config={
                "model_path": args.model_path,
                "save_string": f"{save_string}_{datetime_str}",
                "loss_function": loss_function,
                "learning_rate": args.lr,
                "datetime": datetime_str,
            },
        )

    TRAINER_REGISTRY = {
        "npo": NPOTrainer,
        "ga": GradientAscentTrainer,
    }

    trainer_class = TRAINER_REGISTRY[loss_function]

    dtype = resolve_dtype(args.dtype)

    hf_token = load_hf_token() if args.use_hf else None
    cache_dir = args.cache_dir

    if args.resume_from_checkpoint:
        ckpt = os.path.abspath(args.resume_from_checkpoint)
        if not os.path.isdir(ckpt):
            raise FileNotFoundError(f"Không tìm thấy checkpoint để resume: {ckpt}")
        print(f"Đang load trọng số full-parameter từ checkpoint: {ckpt}")
        model = AutoModelForCausalLM.from_pretrained(
            ckpt,
            cache_dir=cache_dir,
            dtype=dtype,
            attn_implementation="eager", 
        )
        tok_ckpt = os.path.join(ckpt, "tokenizer_config.json")
        if os.path.isfile(tok_ckpt):
            tokenizer = AutoTokenizer.from_pretrained(ckpt, cache_dir=cache_dir)
        else:
            tokenizer = AutoTokenizer.from_pretrained(
                args.model_path, token=hf_token, cache_dir=cache_dir
            )
    else:
        print(f"Đang load model từ {args.model_path}...")
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            token=hf_token,
            cache_dir=cache_dir,
            dtype=dtype,
            attn_implementation="eager",  
        )
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_path, token=hf_token, cache_dir=cache_dir
        )
    print("Đã load model thành công!")


    if "DeepSeek-Coder-V2-Lite-Instruct" in args.model_path and not args.resume_from_checkpoint:
        frozen_count = 0
        frozen_params = 0
        for name, param in model.named_parameters():
            if ".mlp.gate." in name and ".experts." not in name:
                param.requires_grad = False
                frozen_count += 1
                frozen_params += param.numel()

        assert frozen_count > 0, (
            f""
            f""
        )

    if args.use_lora:
        print(f"Đang áp LoRA với rank={args.lora_rank} ...")
        model = apply_lora(model, lora_rank=args.lora_rank)

    # PARAMETERS
    beta = 0.1 
    lambda_retain = args.lambda_retain
    lambda_forget = args.lambda_forget
    temperature = 2.0 
    alpha = 0.7 
    gamma = 0.3  

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    vocab_size = len(tokenizer)
    print(f"Vocab size: {vocab_size}")

    print("Đang load dataset theo token...")

    full_ds = load_toklevel_files(args.forget_file, args.retain_file)
    split = full_ds.train_test_split(test_size=0.1, seed=42, shuffle=True)

    collator = TokLevelCollator(
        pad_token_id=tokenizer.pad_token_id, label_pad_id=-100
    )

    # Đếm số token ignore/retain/forget
    def count_tri_mask_tokens(ds: Dataset) -> Tuple[int, int, int]:
        n0 = n1 = n2 = 0
        for ex in ds:
            n0 += ex["tri_mask"].count(0)
            n1 += ex["tri_mask"].count(1)
            n2 += ex["tri_mask"].count(2)
        return n0, n1, n2

    n0_tr, n1_tr, n2_tr = count_tri_mask_tokens(split["train"])
    n0_ev, n1_ev, n2_ev = count_tri_mask_tokens(split["test"])
    print(f"Token train – ignore:{n0_tr} retain:{n1_tr} forget:{n2_tr}")
    print(f" Token eval – ignore:{n0_ev} retain:{n1_ev} forget:{n2_ev}")

    # Chỉ report wandb khi được bật
    report_to = ["tensorboard"]
    if args.use_wandb:
        report_to.append("wandb")

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.lr,
        weight_decay=0.01,
        warmup_steps=10,
        logging_strategy="steps",
        logging_dir=os.path.join(args.output_dir, "logs"),
        logging_steps=5,
        eval_strategy="no",
        save_strategy="no",
        save_steps=25,
        save_total_limit=2,
        per_device_train_batch_size=1,  
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=16,
        gradient_checkpointing=True,
        bf16=(dtype == torch.bfloat16),
        fp16=(dtype == torch.float16),
        max_grad_norm=1.0,
        lr_scheduler_type="constant",
        report_to=report_to,
        remove_unused_columns=False,
        deepspeed=args.deepspeed,
        seed=random_seed,
    )

    trainer_kwargs = dict(
        model=model,
        args=training_args,
        train_dataset=split["train"],
        eval_dataset=split["test"],
        tokenizer=tokenizer,
        data_collator=collator,
        lambda_retain=lambda_retain,
        lambda_forget=lambda_forget,
        lambda_eos=args.lambda_eos, 
        vocab_size=vocab_size,
        beta=beta,
    )

    trainer = trainer_class(**trainer_kwargs)

    print("Đang train…")
    trainer.train()

    print("Đang lưu model…")
    if args.use_lora:
        model.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)
        print(f"Đã lưu LoRA adapter vào {args.output_dir}")
    else:
        trainer.save_model(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)

    training_info = {
        "beta": beta,
        "lambda_retain": lambda_retain,
        "vocab_size": vocab_size,
        "use_lora": args.use_lora,
        "lora_rank": args.lora_rank if args.use_lora else None,
        "args": args.__dict__,
        "train_token_counts": {"ignore": n0_tr, "retain": n1_tr, "forget": n2_tr},
        "eval_token_counts": {"ignore": n0_ev, "retain": n1_ev, "forget": n2_ev},
        "temperature": temperature,
        "alpha": alpha,
        "gamma": gamma,
    }

    with open(
        os.path.join(args.output_dir, "training_info.json"), "w", encoding="utf-8"
    ) as f:
        json.dump(training_info, f, indent=2)

    print("Xong")


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True, help="Đường dẫn cục bộ (hoặc id HF) của base model cần fine-tune")
    parser.add_argument("--retain-file", required=True, help="File JSONL retain đã tokenize (tri-mask), dựng bởi scripts/build_data.sh")
    parser.add_argument("--forget-file", required=True, help="File JSONL forget đã tokenize (tri-mask), dựng bởi scripts/build_data.sh")
    parser.add_argument("--output-dir", required=True, help="Thư mục lưu checkpoint")
    parser.add_argument("--cache-dir", default=None, help="Thư mục cache của huggingface_hub (tuỳ chọn)")
    parser.add_argument("--save_string", type=str, help="Chuỗi gắn thêm vào tên run (chỉ dùng để đặt tên trên wandb)")
    parser.add_argument(
        "--resume_from_checkpoint", type=str, default=None,
        help=(
            "Đường dẫn tới 1 thư mục model HF hoặc checkpoint Trainer có đủ trọng số. "
            "Chỉ load trọng số; trạng thái optimizer/trainer không được khôi phục "
        ),
    )
    parser.add_argument("--loss_function", type=str, choices=["ga", "npo"], required=True, help="")
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--num_train_epochs", type=int, default=60, help="Số epoch train")
    parser.add_argument("--use_wandb", action="store_true", help="Bật log wandb")
    parser.add_argument("--deepspeed", type=str, default=None, help="Đường dẫn file config DeepSpeed")
    parser.add_argument("--use_hf", action="store_true", help="Dùng token HF (đọc qua au_utils.load_hf_token) khi tải model/tokenizer")
    parser.add_argument(
        "--dtype", type=str, default="auto", choices=["auto", "bfloat16", "float16"],
        help="auto chọn bfloat16 trên GPU hỗ trợ, không thì float16",
    )
    parser.add_argument("--lambda_retain", type=float, default=1.0, help="Trọng số CE loss cho token retain hợp lệ (mask=1)")
    parser.add_argument("--lambda_forget", type=float, default=0.5, help="Trọng số NPO loss cho token hallucinated (mask=2)")
    parser.add_argument("--lambda_eos", type=int, default=2, help="Trọng số cho token eos để tránh lặp lại")
    parser.add_argument("--seed", type=int, default=None, help="Seed để tái lập kết quả")
    parser.add_argument("--use_lora", action="store_true", help="Train bằng LoRA thay vì fine-tune toàn bộ trọng số")
    parser.add_argument("--lora_rank", type=int, default=16, help="LoRA rank (r)")

    return parser.parse_args()


if __name__ == "__main__":
    main()
