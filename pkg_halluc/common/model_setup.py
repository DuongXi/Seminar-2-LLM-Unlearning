"""Download base model, build model + tokenizer"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from pkg_halluc.common.model_presets import MODEL_PRESETS


def apply_hf_token() -> None:
    if os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN"):
        return
    repo_root = Path(__file__).resolve().parents[2]
    for token_file in (Path.cwd() / "hf_token.txt", repo_root / "hf_token.txt"):
        if token_file.is_file():
            token = token_file.read_text(encoding="utf-8-sig").strip()
            if token:
                os.environ["HF_TOKEN"] = token
                return


def download_base_model(model_name: str, models_dir: Path) -> Path:
    """Tải model từ Hugging Face vào models_dir/<id đầy đủ>, đã có thì bỏ qua"""
    from huggingface_hub import snapshot_download

    apply_hf_token()

    repo_id = MODEL_PRESETS.get(model_name, model_name)

    model_local_dir = models_dir / repo_id
    if not model_local_dir.exists() or not any(model_local_dir.iterdir()):
        model_local_dir.mkdir(parents=True, exist_ok=True)
        snapshot_download(repo_id=repo_id, local_dir=str(model_local_dir))
    print(f"Base model đã có tại: {model_local_dir}")
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
    """Build causal LM + tokenizer đã chuẩn hoá pad/eos, trả về (model, tokenizer)"""
    import torch
    from transformers import AutoModelForCausalLM

    from pkg_halluc.package_loader.utils import setup_tokenizer

    apply_hf_token()

    raw_identifier = str(model_name_or_path)
    resolved_id = MODEL_PRESETS.get(raw_identifier, raw_identifier)

    tok = setup_tokenizer(
        resolved_id,
        model_family=model_family,
        trust_remote_code=trust_remote_code,
    )
    tok.padding_side = padding_side

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

    model = AutoModelForCausalLM.from_pretrained(
        resolved_id,
        dtype=torch_dtype,
        device_map=device_map,
        trust_remote_code=trust_remote_code,
    )
    model.eval()

    if tok.pad_token_id is not None:
        model.config.pad_token_id = tok.pad_token_id
        if hasattr(model, "generation_config") and model.generation_config is not None:
            model.generation_config.pad_token_id = tok.pad_token_id
    if tok.eos_token_id is not None:
        model.config.eos_token_id = tok.eos_token_id
        if hasattr(model, "generation_config") and model.generation_config is not None:
            model.generation_config.eos_token_id = tok.eos_token_id

    return model, tok


def sanity_check_generation(model_path: Path, dtype: str = "auto") -> str:
    """Genarate 1 time to check if torch/transformers/CUDA runnable"""
    import torch
    from transformers import GenerationConfig

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
    print(f"Sanity check: {text!r}")

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return text
