"""Representation Steering -- mean-difference activation steering.

Workflow:
  prepare()  -- Loads a results CSV, extracts hidden states at package-generation
                boundaries, computes mean-difference vector v_L for each layer in
                [layer_start, layer_end], selects the best layer (highest vector
                norm), and caches:
                  generated_data/steering/per_layer_vectors.pt
                  generated_data/steering/steering_vector.pt
                  generated_data/steering/metadata.json

  evaluate() -- extra_eval_args() passes CLI flags so eval_variant.py calls
                steering_eval_adapter.py which patches the model at generation time.

Config (methods.steering): enabled, layer_start, layer_end, steering_strength,
  component, strategy, extract_batch_size, train_csv  (see config.py)
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import torch

from .spec import MethodContext, MethodSpec

logger = logging.getLogger(__name__)

TAG = "steering"


def _vector_cache_dir(ctx: MethodContext) -> Path:
    return ctx.paths.generated_data_dir / "steering"


def _best_vector_path(ctx: MethodContext) -> Path:
    return _vector_cache_dir(ctx) / "steering_vector.pt"


def _meta_path(ctx: MethodContext) -> Path:
    return _vector_cache_dir(ctx) / "metadata.json"


def _resolve_train_csv(ctx: MethodContext) -> Path:
    explicit = ctx.method_cfg.get("train_csv")
    if explicit:
        p = Path(explicit)
        if not p.is_absolute():
            p = ctx.paths.repo_root / p
        return p

    candidates = [
        ctx.paths.repo_root / "data" / "steering_train_data.csv",
        ctx.paths.repo_root / "data" / "Llama3_3_Python" / "LLM_LY_results.csv",
        ctx.paths.repo_root / "data" / "LLM_LY_results.csv",
    ]
    for candidate in candidates:
        if candidate.exists():
            logger.info("Auto-detected training CSV: %s", candidate)
            return candidate

    raise FileNotFoundError(
        "No training CSV found for Representation Steering. "
        "Set methods.steering.train_csv in your config to the path of "
        "LLM_LY_results.csv from the Steering branch."
    )


def _select_best_layer(per_layer_vectors: dict, layer_start: int, layer_end: int) -> int:
    best_layer, best_norm = layer_start, -1.0
    for layer_idx in range(layer_start, layer_end + 1):
        if layer_idx not in per_layer_vectors:
            continue
        v = per_layer_vectors[layer_idx]
        if v.numel() == 0:
            continue
        norm = v.norm().item()
        if norm > best_norm:
            best_norm = norm
            best_layer = layer_idx
    logger.info("Best layer: %d (norm=%.4f)", best_layer, best_norm)
    return best_layer


def _prepare(ctx: MethodContext) -> None:
    cache_dir = _vector_cache_dir(ctx)
    best_path = _best_vector_path(ctx)
    meta_path = _meta_path(ctx)

    if best_path.exists() and meta_path.exists():
        logger.info("Steering vector already cached -- skipping.")
        return

    cache_dir.mkdir(parents=True, exist_ok=True)

    m = ctx.method_cfg
    layer_start = int(m.get("layer_start", 4))
    layer_end = int(m.get("layer_end", 12))
    extract_bs = int(m.get("extract_batch_size", 4))

    train_csv = _resolve_train_csv(ctx)
    logger.info("Computing steering vectors from %s ...", train_csv)

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from ..config import resolve_dtype
    from .steering_vector_utils import compute_and_cache_steering_vectors

    dtype_str = resolve_dtype(ctx.cfg)
    dtype = torch.bfloat16 if dtype_str == "bfloat16" else torch.float16

    tokenizer = AutoTokenizer.from_pretrained(ctx.cfg["model_name"], padding_side="right")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        ctx.cfg["model_name"], torch_dtype=dtype, device_map="auto"
    )
    model.eval()

    per_layer_pt = compute_and_cache_steering_vectors(
        model=model, tokenizer=tokenizer,
        csv_path=train_csv, output_dir=cache_dir,
        batch_size=extract_bs,
    )

    del model
    torch.cuda.empty_cache()

    per_layer_vectors = torch.load(per_layer_pt, map_location="cpu")
    best_layer = _select_best_layer(per_layer_vectors, layer_start, layer_end)

    best_vec = per_layer_vectors[best_layer]
    torch.save({"vector": best_vec, "layer": best_layer}, best_path)

    meta = {
        "best_layer": best_layer,
        "layer_start": layer_start,
        "layer_end": layer_end,
        "steering_strength": m.get("steering_strength", 20.0),
        "component": m.get("component", "res"),
        "strategy": m.get("strategy", "first_token"),
        "train_csv": str(train_csv),
        "model_name": ctx.cfg["model_name"],
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    logger.info("Steering vector ready: layer=%d norm=%.4f -> %s",
                best_layer, best_vec.norm().item(), best_path)


def _checkpoint_path(ctx: MethodContext) -> Path:
    return ctx.base_model_path


def _package_modes(ctx: MethodContext) -> list:
    return list(ctx.cfg["eval"]["package_modes"])


def _extra_eval_args(ctx: MethodContext) -> list:
    meta_path = _meta_path(ctx)
    if not meta_path.exists():
        raise RuntimeError(
            "Steering metadata not found. Run prepare first:\n"
            "  pkg_halluc prepare --method steering"
        )
    with open(meta_path, encoding="utf-8") as f:
        meta = json.load(f)

    m = ctx.method_cfg
    return [
        "--method", "steering",
        "--steering_vector_path", str(_best_vector_path(ctx)),
        "--str_layer", str(meta["best_layer"]),
        "--steering_strength", str(m.get("steering_strength", meta.get("steering_strength", 20.0))),
        "--component", m.get("component", meta.get("component", "res")),
        "--steering_adapter_dir", str(
            Path(__file__).resolve().parent.parent / "templates" / "steering"
        ),
    ]


def build_spec(enabled: bool) -> MethodSpec:
    return MethodSpec(
        name=TAG,
        display_name="Representation Steering",
        kind="representation_steering",
        enabled=enabled,
        checkpoint_path=_checkpoint_path,
        package_modes=_package_modes,
        train=None,
        prepare=_prepare,
        extra_eval_args=_extra_eval_args,
    )
