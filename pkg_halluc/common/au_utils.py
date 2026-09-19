# AU
import os
import json
import random
from dataclasses import dataclass
from typing import List, Dict, Optional

from datasets import Dataset
import torch
from peft import LoraConfig, get_peft_model, PeftModel, TaskType
from transformers import AutoModelForCausalLM, AutoTokenizer

MODE1_PREFIXES = [
    "What Python packages are needed to run this code: ",
    "Which pip packages does this code require: ",
    "Name the Python packages this code depends on: ",
    "What packages would I need to pip install to run this: ",
    "Identify the Python package names required by this code: ",
    "What are the package dependencies for this code: ",
    "Which Python packages must be installed to execute this code: ",
    "List the package names needed to run the following code: ",
    "What packages does this code need installed to work: ",
    "Tell me which Python packages this code requires: ",
]

MODE2_PREFIXES = [
    "Which Python packages would help solve this coding problem: ",
    "What Python packages could I use to tackle this problem: ",
    "Name some Python packages that would be relevant for solving this: ",
    "What packages should I consider using for this coding task: ",
    "Which Python packages would be appropriate for this problem: ",
    "What Python packages would you recommend for solving this: ",
    "Suggest Python packages that could help with this coding challenge: ",
    "What packages would be beneficial for implementing a solution to this: ",
    "Which Python packages are well-suited for this problem: ",
    "What Python packages would assist in solving the following task: ",
]


def get_random_prefix(mode: int) -> str:
    """Chọn ngẫu nhiên 1 prefix hỏi package theo mode (1: từ code, 2: từ đề bài)."""
    if mode == 1:
        return random.choice(MODE1_PREFIXES)
    elif mode == 2:
        return random.choice(MODE2_PREFIXES)


def read_jsonl(path: str) -> List[Dict]:
    """Đọc file JSONL thành list dict, bỏ qua dòng trống."""
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    return items


def load_toklevel_files(forget_tok_path: Optional[str], retain_tok_path: Optional[str]) -> Dataset:
    """Đọc 2 file JSONL tri-mask (forget, retain) thành 1 Dataset dùng để train."""
    data = []
    for p in [forget_tok_path, retain_tok_path]:
        if p is None: 
            continue
        for ex in read_jsonl(p):
            # Chỉ giữ các field cần cho việc train
            item = {
                "input_ids": ex["input_ids"],
                "attention_mask": ex["attention_mask"],
                "labels": ex["labels"],
                "tri_mask": ex["tri_mask"],
            }
            # Kiểm tra độ dài các field phải khớp nhau
            assert (
                len(item["input_ids"])
                == len(item["attention_mask"])
                == len(item["labels"])
                == len(item["tri_mask"])
            ), "Độ dài không khớp trong 1 mẫu token-level."
            # tri_mask chỉ nhận 0 (ignore), 1 (retain), 2 (forget)
            if not all(t in (0, 1, 2) for t in item["tri_mask"]):
                raise ValueError("tri_mask chỉ được chứa {0,1,2}.")
            data.append(item)

    return Dataset.from_list(data)


@dataclass
class TokLevelCollator:
    """Collator pad động input_ids, attention_mask, labels và tri_mask."""

    pad_token_id: int
    label_pad_id: int = -100

    def __call__(self, features: List[Dict]) -> Dict[str, torch.Tensor]:
        max_len = max(len(f["input_ids"]) for f in features)

        def pad(seq: List[int], pad_val: int) -> List[int]:
            return seq + [pad_val] * (max_len - len(seq))

        batch = {
            "input_ids": torch.tensor(
                [pad(f["input_ids"], self.pad_token_id) for f in features],
                dtype=torch.long,
            ),
            "attention_mask": torch.tensor(
                [pad(f["attention_mask"], 0) for f in features], dtype=torch.long
            ),
            "labels": torch.tensor(
                [pad(f["labels"], self.label_pad_id) for f in features],
                dtype=torch.long,
            ),
            "tri_mask": torch.tensor(
                [pad(f["tri_mask"], 0) for f in features], dtype=torch.long
            ),
        }
        return batch


def load_hf_token():
    """Lấy token HF từ hf_token.txt (ưu tiên) hoặc biến môi trường HF_TOKEN."""
    token_file_path = os.path.join(os.getcwd(), "hf_token.txt")
    if os.path.exists(token_file_path):
        with open(token_file_path, "r") as f:
            hf_token = f.read().strip()
    else:
        hf_token = os.getenv("HF_TOKEN")
    if hf_token is None:
        raise ValueError("Biến môi trường HF_TOKEN chưa được set")
    return hf_token


def apply_lora(model, lora_rank: int = 16, lora_alpha: int = None, target_modules=None):
    """Bọc causal LM bằng LoRA adapter qua PEFT, chỉ tham số LoRA là trainable."""
    if lora_alpha is None:
        lora_alpha = lora_rank * 2

    if target_modules is None:
        target_modules = [
            # Attention
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            # MLP
            "gate_proj",
            "up_proj",
            "down_proj",
        ]

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=0.05,
        target_modules=target_modules,
        bias="none",
    )

    model = get_peft_model(model, lora_config)
    model.enable_input_require_grads()
    model.print_trainable_parameters()
    return model


def is_lora_model(model_path: str) -> bool:
    """Kiểm tra thư mục model đã lưu có chứa LoRA adapter hay không."""
    return os.path.isfile(os.path.join(model_path, "adapter_config.json"))


def load_model_auto(
    model_path: str, device_map="auto", torch_dtype=None, attn_implementation=None
):
    """Load model, tự nhận diện LoRA adapter rồi merge vào base model, trả (model, tokenizer)."""
    extra_kwargs = {}
    if torch_dtype is not None:
        # transformers 4.57.6 đổi tên tham số torch_dtype= thành dtype=
        extra_kwargs["dtype"] = torch_dtype
    if attn_implementation is not None:
        extra_kwargs["attn_implementation"] = attn_implementation

    if is_lora_model(model_path):
        with open(os.path.join(model_path, "adapter_config.json"), "r") as f:
            adapter_cfg = json.load(f)
        base_model_path = adapter_cfg.get("base_model_name_or_path", model_path)
        print(f"Phát hiện LoRA adapter tại {model_path}")
        print(f"Đang load base model từ {base_model_path} ...")

        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            device_map=device_map,
            **extra_kwargs,
        )
        model = PeftModel.from_pretrained(base_model, model_path)
        model = model.merge_and_unload()
        print("Đã merge LoRA adapter vào base model.")

        tokenizer = AutoTokenizer.from_pretrained(model_path)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map=device_map,
            **extra_kwargs,
        )
        tokenizer = AutoTokenizer.from_pretrained(model_path)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer
