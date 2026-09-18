"""Bảng preset: tên ngắn của model -> id Hugging Face đầy đủ."""
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
    """Đổi tên preset thành id HF đầy đủ, không có trong bảng thì giữ nguyên."""
    return MODEL_PRESETS.get(name, name)
