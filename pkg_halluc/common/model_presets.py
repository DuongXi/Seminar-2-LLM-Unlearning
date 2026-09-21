"""Preset: short name -> ID Hugging Face"""
from __future__ import annotations

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


def resolve_model_name(name: str) -> str:
    """Preset name to full id HF"""
    return MODEL_PRESETS.get(name, name)


def resolve_model_suffix(name: str, explicit_suffix: str | None = None) -> str:
    """Map model name to suffix for tokenized tri-mask files"""
    if explicit_suffix:
        return explicit_suffix if explicit_suffix.startswith("_") else f"_{explicit_suffix}"

    name_lower = (name or "").lower()

    if "deepseek" in name_lower:
        if any(k in name_lower for k in ("1.3b", "1b", "1.3", "1_3b")):
            return "_deepseek_1B"
        return "_deepseek"

    if "llama" in name_lower:
        if "1b" in name_lower or "1_b" in name_lower:
            return "_llama_1B"
        if "3b" in name_lower or "3_b" in name_lower:
            return "_llama_3B"
        return "_llama"

    if "qwen" in name_lower:
        if "3b" in name_lower:
            return "_qwen_3B"
        if "1.5b" in name_lower or "1_5b" in name_lower:
            return "_qwen_1.5B"
        if "0.5b" in name_lower or "0_5b" in name_lower:
            return "_qwen_0.5B"
        return "_qwen"

    return ""

