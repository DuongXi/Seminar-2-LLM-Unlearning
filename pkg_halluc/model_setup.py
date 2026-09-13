"""Download, build, and configure base models, plus sanity-check generation.

Supports:
  - Qwen 2.5 Coder (1.5B, 3B)
  - Llama 3.2 (1B, 3B)
  - DeepSeek Coder (1.3B / 1B)
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from .paths import Paths


def download_base_model(cfg: dict[str, Any], paths: Paths) -> Path:
    """Download ``cfg["model_name"]`` from the Hub into ``work_dir/models/<name>``
    unless it's already there. Resolves presets (e.g. qwen2.5-coder-1.5b, llama3.2-1b).
    Returns the local path."""
    from huggingface_hub import snapshot_download
    from .config import MODEL_PRESETS

    raw_model_name = cfg["model_name"]
    repo_id = MODEL_PRESETS.get(raw_model_name, raw_model_name)

    model_local_dir = paths.models_dir / repo_id
    if not model_local_dir.exists() or not any(model_local_dir.iterdir()):
        model_local_dir.mkdir(parents=True, exist_ok=True)
        snapshot_download(repo_id=repo_id, local_dir=str(model_local_dir))
    print(f"Base model cached at: {model_local_dir}")
    print(sorted(p.name for p in model_local_dir.iterdir())[:20])
    return model_local_dir


def build_model(
    model_name_or_path: str | Path,
    dtype: str | Any = "auto",
    device_map: str | dict | None = "auto",
    trust_remote_code: bool = True,
    model_family: str = "auto",
    padding_side: str = "right",
) -> tuple[Any, Any]:
    """Build model + tokenizer for Qwen 2.5 Coder / Llama 3.2 / DeepSeek Coder.

    Sets pad_token/eos_token (Llama 3.2 has no default pad_token), chat
    template, dtype, device_map, and syncs pad/eos ids across tokenizer,
    model config, and generation_config.

    Returns (model, tokenizer).
    """
    import torch
    from transformers import AutoModelForCausalLM
    from .config import MODEL_PRESETS
    from .package_loader.utils import setup_tokenizer

    raw_identifier = str(model_name_or_path)
    resolved_id = MODEL_PRESETS.get(raw_identifier, raw_identifier)

    # 1. Setup tokenizer with correct pad_token & chat_template
    tok = setup_tokenizer(
        resolved_id,
        model_family=model_family,
        trust_remote_code=trust_remote_code,
    )
    tok.padding_side = padding_side

    # 2. Resolve torch dtype
    if isinstance(dtype, str):
        if dtype == "auto":
            torch_dtype = (
                torch.bfloat16
                if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
                else torch.float16
            )
        elif dtype == "bfloat16":
            torch_dtype = torch.bfloat16
        elif dtype == "float16":
            torch_dtype = torch.float16
        else:
            torch_dtype = torch.float32
    else:
        torch_dtype = dtype

    # 3. Load causal LM model
    model = AutoModelForCausalLM.from_pretrained(
        resolved_id,
        torch_dtype=torch_dtype,
        device_map=device_map,
        trust_remote_code=trust_remote_code,
    )
    model.eval()

    # 4. Synchronize pad_token_id and eos_token_id
    if tok.pad_token_id is not None:
        model.config.pad_token_id = tok.pad_token_id
        if hasattr(model, "generation_config") and model.generation_config is not None:
            model.generation_config.pad_token_id = tok.pad_token_id
    if tok.eos_token_id is not None:
        model.config.eos_token_id = tok.eos_token_id
        if hasattr(model, "generation_config") and model.generation_config is not None:
            model.generation_config.eos_token_id = tok.eos_token_id

    return model, tok


def sanity_check_generation(cfg: dict[str, Any], model_path: Path) -> str:
    """One short generation, to confirm torch/transformers/CUDA work before
    the heavier fetch-deps / build-data / train stages."""
    import torch
    from transformers import GenerationConfig
    from .config import resolve_dtype

    dtype = resolve_dtype(cfg)
    model, tok = build_model(
        model_name_or_path=model_path,
        dtype=dtype,
        device_map="auto",
        trust_remote_code=True,
    )

    messages = [{"role": "user", "content": "Write one line of Python that imports numpy."}]
    enc = tok.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
    ).to(model.device)
    gen_cfg = GenerationConfig(
        do_sample=False,
        max_new_tokens=40,
        pad_token_id=tok.pad_token_id,
        eos_token_id=tok.eos_token_id,
    )
    with torch.no_grad():
        out = model.generate(**enc, generation_config=gen_cfg)
    text = tok.decode(out[0][enc["input_ids"].shape[1] :], skip_special_tokens=True)
    print(f"Sanity-check generation: {text!r}")

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return text
