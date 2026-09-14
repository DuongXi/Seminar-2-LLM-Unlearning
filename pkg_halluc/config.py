"""Load and validate the pipeline's JSON config files (see configs/*.json).

Merges a config file over hard-coded defaults, then resolves values that
depend on the running machine (dtype).
"""
from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

MODEL_PRESETS: dict[str, str] = {
    # Qwen 2.5 Coder
    "qwen2.5-coder-0.5b": "Qwen/Qwen2.5-Coder-0.5B-Instruct",
    "qwen2.5-coder-1.5b": "Qwen/Qwen2.5-Coder-1.5B-Instruct",
    "qwen2.5-coder-3b": "Qwen/Qwen2.5-Coder-3B-Instruct",
    "qwen-0.5b": "Qwen/Qwen2.5-Coder-0.5B-Instruct",
    "qwen-1.5b": "Qwen/Qwen2.5-Coder-1.5B-Instruct",
    "qwen-3b": "Qwen/Qwen2.5-Coder-3B-Instruct",
    "qwen2.5-0.5b": "Qwen/Qwen2.5-Coder-0.5B-Instruct",
    "qwen2.5-1.5b": "Qwen/Qwen2.5-Coder-1.5B-Instruct",
    "qwen2.5-3b": "Qwen/Qwen2.5-Coder-3B-Instruct",

    # Llama 3.2
    "llama3.2-1b": "meta-llama/Llama-3.2-1B-Instruct",
    "llama3.2-3b": "meta-llama/Llama-3.2-3B-Instruct",
    "llama-1b": "meta-llama/Llama-3.2-1B-Instruct",
    "llama-3b": "meta-llama/Llama-3.2-3B-Instruct",

    # DeepSeek Coder
    "deepseek-coder-1.3b": "deepseek-ai/deepseek-coder-1.3b-instruct",
    "deepseek-coder-1b": "deepseek-ai/deepseek-coder-1.3b-instruct",
    "deepseek-1.3b": "deepseek-ai/deepseek-coder-1.3b-instruct",
    "deepseek-1b": "deepseek-ai/deepseek-coder-1.3b-instruct",
}

# Every field a config file *can* set. A config JSON only needs to include the
# keys it wants to change; everything else falls back to these defaults.
DEFAULT_CONFIG: dict[str, Any] = {
    "model_name": MODEL_PRESETS["llama3.2-3b"],
    "seed": 42,
    # "auto" resolves to bfloat16 on GPUs that support it, else float16.
    "dtype": "auto",
    "data": {
        # Cap retain/forget datasets to at most this many rows each, for a
        # quick smoke test on real data (None = no cap). Read by
        # tok_data/get_data.py::build_data() (ga/npo) and
        # pkg_halluc/scripts/train_plain.py (ga_plain/npo_plain).
        "max_train_samples_per_split": None,
    },
    "eval": {
        "n_eval_prompts": 150,
        "batch_size": 16,
        "package_modes": [1, 2],
        "eval_prompts_path": None,
    },
    "methods": {
        "base": {"enabled": True},
        # use_lora: train a LoRA adapter instead of full fine-tune (much
        # lighter on GPU memory for bigger models). Saved as an adapter
        # only -- eval merges it with the base model at load time.
        "ga": {"enabled": True, "lr": 1e-5, "num_train_epochs": 3, "use_lora": False, "lora_rank": 16},
        "npo": {"enabled": True, "lr": 1e-5, "num_train_epochs": 3, "use_lora": False, "lora_rank": 16},
        # "Plain" ablations of ga/npo: same retain/forget rows, no tri-mask,
        # no AU code (see methods/plain_trainers.py). lambda_retain/forget
        # match the effective defaults ga/npo already run with.
        "ga_plain": {
            "enabled": True, "lr": 1e-5, "num_train_epochs": 3,
            "lambda_retain": 1.0, "lambda_forget": 0.5,
            "use_lora": False, "lora_rank": 16,
            "val_ratio": 0.1, "eval_steps": 25,
            "early_stopping_patience": 3, "early_stopping_threshold": 0.0,
            "disable_early_stopping": False,
        },
        "npo_plain": {
            "enabled": True, "lr": 1e-5, "num_train_epochs": 3,
            "lambda_retain": 1.0, "lambda_forget": 0.5,
            "use_lora": False, "lora_rank": 16,
            "val_ratio": 0.1, "eval_steps": 25,
            "early_stopping_patience": 3, "early_stopping_threshold": 0.0,
            "disable_early_stopping": False,
        },
        "steering": {"enabled": False},
    },
    "tri-mask": True,
    "deps": {
        "adaptive_unlearning": {
            "source": None,
        },
    },
}


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(path: str | Path | None) -> dict[str, Any]:
    """Load a config JSON and merge it over DEFAULT_CONFIG. path=None returns the defaults."""
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    if path is not None:
        with open(path, encoding="utf-8") as f:
            user_cfg = json.load(f)
        cfg = _deep_merge(cfg, user_cfg)

    # Resolve a friendly model preset name (e.g. "qwen2.5-1.5b") to its full HF id.
    if cfg["model_name"] in MODEL_PRESETS:
        cfg["model_name"] = MODEL_PRESETS[cfg["model_name"]]

    return cfg


def resolve_dtype(cfg: dict[str, Any]) -> str:
    """Return "bfloat16" or "float16", resolving cfg["dtype"] == "auto" via torch."""
    if cfg["dtype"] != "auto":
        assert cfg["dtype"] in ("bfloat16", "float16"), f"Unknown dtype: {cfg['dtype']}"
        return cfg["dtype"]
    import torch

    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return "bfloat16"
    return "float16"


def apply_overrides(cfg: dict[str, Any], *, model_name: str | None = None, seed: int | None = None) -> dict[str, Any]:
    """Apply a handful of common CLI --flag overrides on top of a loaded config."""
    cfg = copy.deepcopy(cfg)
    if model_name is not None:
        cfg["model_name"] = MODEL_PRESETS.get(model_name, model_name)
    if seed is not None:
        cfg["seed"] = seed
    return cfg


def dump(cfg: dict[str, Any]) -> str:
    return json.dumps(cfg, indent=2, ensure_ascii=False)
