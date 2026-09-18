"""Load and validate the pipeline's JSON config files (see ``configs/*.json``).

Config files are plain JSON so they're easy to diff/version/share between
teammates. This module merges a config file over hard-coded defaults (so a
config only needs to specify what it overrides), then resolves a couple of
values that depend on the running machine (``dtype``).
"""
from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

MODEL_PRESETS: dict[str, str] = {
    "qwen2.5-0.5b": "Qwen/Qwen2.5-0.5B-Instruct",
    "qwen2.5-1.5b": "Qwen/Qwen2.5-1.5B-Instruct",
    "qwen2.5-3b": "Qwen/Qwen2.5-3B-Instruct",
    "deepseek-coder-1.3b": "deepseek-ai/deepseek-coder-1.3b-instruct",
}

# Every field a config file *can* set. A config JSON only needs to include the
# keys it wants to change; everything else falls back to these defaults.
DEFAULT_CONFIG: dict[str, Any] = {
    "model_name": MODEL_PRESETS["qwen2.5-0.5b"],
    "seed": 42,
    # "auto" resolves to bfloat16 on GPUs that support it, else float16.
    "dtype": "auto",
    "data": {
        # how many seed prompts to use when building the static GA/NPO tri-mask
        # dataset (see `pkg_halluc data build-seed-data`)
        "n_data_construction_prompts": 10,
    },
    "eval": {
        "n_eval_prompts": 150,
        "batch_size": 16,
        # mode 1 = "which packages does this code need", mode 2 = "which
        # packages would help solve this task" -- used for base/ga/npo.
        # PackMonitor always evaluates with package_modes=[] (see methods/packmonitor.py).
        "package_modes": [1, 2],
        # None (default) = sample from AU's own bundled eval pool
        # (paths.au_eval_prompts_path, ~4,900 prompts). Set this to use your
        # own prompt set instead -- a path relative to the repo root (e.g.
        # "data/my_prompts.jsonl", see data/README.md) or an absolute path.
        "eval_prompts_path": None,
    },
    "methods": {
        "base": {"enabled": True},
        "ga": {"enabled": True, "lr": 1e-5, "num_train_epochs": 3},
        "npo": {"enabled": True, "lr": 1e-5, "num_train_epochs": 3},
        "packmonitor": {"enabled": True, "code_temp": 0.7, "package_temp": 0.01},
        # Representation Steering: mean-difference vector computed from paired
        # valid/hallucinated package-name activations, added to the residual
        # stream at generation time.  Disabled by default; set enabled=true
        # after the vector is computed (or let `pkg_halluc evaluate --method
        # steering` do the prepare step automatically).
        "steering": {
            "enabled": False,
            # Layer range to search for the best steering layer.
            # The prepare step tries every layer from layer_start to layer_end
            # and caches per-layer vectors; the best single layer (lowest
            # hallucination rate on a small validation split) is selected.
            "layer_start": 4,
            "layer_end": 12,
            # Strength λ: h^(l) ← h^(l) + lambda * v
            "steering_strength": 20.0,
            # Where to inject: "res" (residual stream, recommended),
            # "mlp" (after MLP), or "attn" (after attention).
            "component": "res",
            # How to aggregate hidden states across tokens at the package
            # boundary: "first_token" (default), "last_token", or "mean".
            "strategy": "first_token",
            # Batch size used when running the forward pass to extract
            # hidden states during the prepare step.
            "extract_batch_size": 4,
            # Path to a CSV file with columns Prompts / Answers /
            # pip_valid / pip_hallucinated (the LLM_LY_results.csv schema).
            # null = look for the file relative to the repo root under
            # data/steering_train_data.csv, then fall back to the AU eval
            # prompts as a last resort (no pre-computed answers available
            # there, so the fallback is mainly for smoke-test runs).
            "train_csv": None,
        },
    },
    "deps": {
        "adaptive_unlearning": {
            # No default URL on purpose: the paper's code is published anonymously
            # (anonymous.4open.science) for double-blind review, and that service
            # is known to be occasionally unreachable. Point this at a local zip
            # you've already downloaded, an http(s) URL, or leave it null and rely
            # on a Kaggle Dataset / --au-src being passed explicitly.
            # See README.md > "Fetching the third-party code" for details.
            "source": None,
        },
        "packmonitor": {
            # None -- resolution falls through to the Kaggle Dataset search
            # first (if you're on Kaggle), and only then to PackMonitor's
            # official public repo (baked into deps/fetch.py::PACKMONITOR_SPEC,
            # not repeated here) as the last resort. Set this explicitly only
            # if you want to *force* a particular source ahead of that Kaggle
            # check, e.g. a fork or a local mirror.
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
    """Load a config JSON and merge it over :data:`DEFAULT_CONFIG`.

    ``path=None`` returns the defaults untouched (useful for tests).
    """
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
