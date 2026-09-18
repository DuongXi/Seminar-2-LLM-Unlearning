"""Steering eval adapter -- same I/O as generate_code.py / generate_package_names.py.

Loads a cached steering vector (steering_vector.pt), patches the model at the
chosen layer via add_tsv_layers, then generates code / package names.
Called by eval_variant.py when --method steering is passed.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

_HERE = Path(__file__).resolve().parent
_PKG_SRC = _HERE.parents[2]
sys.path.insert(0, str(_PKG_SRC))

from pkg_halluc.methods.llm_layers import add_tsv_layers  # noqa: E402

logger = logging.getLogger(__name__)
_MODEL_CACHE: dict = {}


def _get_model_and_tokenizer(
    model_path: str, steering_vector_path: str,
    str_layer: int, steering_strength: float,
    component: str, dtype_str: str,
):
    key = (model_path, steering_vector_path, str_layer, steering_strength, component)
    if key in _MODEL_CACHE:
        return _MODEL_CACHE[key]

    dtype = torch.bfloat16 if dtype_str == "bfloat16" else torch.float16
    tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="right")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=dtype, device_map="auto"
    )
    model.eval()

    vec_dict = torch.load(steering_vector_path, map_location="cpu")
    vector = vec_dict["vector"] if isinstance(vec_dict, dict) else vec_dict

    add_tsv_layers(
        model, tsv=vector.to(dtype),
        alpha=steering_strength,
        str_layer=str_layer,
        component=component,
    )
    _MODEL_CACHE[key] = (model, tokenizer)
    return model, tokenizer


_SYSTEM_CODE = (
    "You are a coding assistant that generates Python code. Provide only the "
    "Python code and add additional explanatory text only when absolutely necessary. "
    "If no code is required to answer the question, simply reply 'None'"
)
_MODE1_PREFIXES = [
    "What Python packages are needed to run this code: ",
    "Which pip packages does this code require: ",
    "Name the Python packages this code depends on: ",
    "What packages would I need to pip install to run this: ",
    "Identify the Python package names required by this code: ",
]
_MODE2_PREFIXES = [
    "Which Python packages would help solve this coding problem: ",
    "What Python packages could I use to tackle this problem: ",
    "Name some Python packages that would be relevant for solving this: ",
    "What packages should I consider using for this coding task: ",
    "Which Python packages would be appropriate for this problem: ",
]


def _chat_prompt(tokenizer, messages: list) -> str:
    try:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    except Exception:
        parts = [f"<|{m['role']}|>\n{m['content']}" for m in messages]
        parts.append("<|assistant|>\n")
        return "\n".join(parts)


def _generate_batch(model, tokenizer, prompts: list, max_new_tokens: int,
                    temperature: float, top_k: int = 20, top_p: float = 0.9) -> list:
    enc = tokenizer(
        prompts, return_tensors="pt", padding=True,
        truncation=True, max_length=2048,
    )
    input_ids = enc["input_ids"].to(model.device)
    attention_mask = enc["attention_mask"].to(model.device)

    with torch.no_grad():
        out_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=(temperature > 0),
            temperature=temperature if temperature > 0 else 1.0,
            top_k=top_k,
            top_p=top_p,
            pad_token_id=tokenizer.pad_token_id,
        )
    new_ids = out_ids[:, input_ids.shape[1]:]
    return tokenizer.batch_decode(new_ids, skip_special_tokens=True)


def generate_code_steering(
    infile: str, outfile: str, model_path: str, steering_vector_path: str,
    str_layer: int, steering_strength: float = 20.0, component: str = "res",
    dtype_str: str = "bfloat16", temperature: float = 0.7,
    batch_size: int = 16, max_new_tokens: int = 2048,
):
    model, tokenizer = _get_model_and_tokenizer(
        model_path, steering_vector_path, str_layer, steering_strength, component, dtype_str
    )
    with open(infile, encoding="utf-8") as f:
        prompts = [json.loads(l) for l in f if l.strip()]
    prompt_texts = [p[0] if isinstance(p, list) else p for p in prompts]

    results = []
    for i in tqdm(range(0, len(prompt_texts), batch_size), desc="Generating code (Steering)"):
        batch = prompt_texts[i: i + batch_size]
        chat_prompts = [
            _chat_prompt(tokenizer, [
                {"role": "system", "content": _SYSTEM_CODE},
                {"role": "user", "content": p},
            ])
            for p in batch
        ]
        results.extend(_generate_batch(model, tokenizer, chat_prompts, max_new_tokens, temperature))

    with open(outfile, "w", encoding="utf-8") as f:
        for r in results:
            json.dump(r, f)
            f.write("\n")


def generate_packages_steering(
    mode: int, infile: str, outfile: str, model_path: str, steering_vector_path: str,
    str_layer: int, steering_strength: float = 20.0, component: str = "res",
    dtype_str: str = "bfloat16", temperature: float = 0.01, batch_size: int = 16,
):
    model, tokenizer = _get_model_and_tokenizer(
        model_path, steering_vector_path, str_layer, steering_strength, component, dtype_str
    )
    with open(infile, encoding="utf-8") as f:
        rows = [json.loads(l) for l in f if l.strip()]

    if mode == 1:
        items = [r.get("Answers", r.get("code", "")) for r in rows]
    else:
        items = [r[0] if isinstance(r, list) else r.get("prompt", "") for r in rows]

    prefixes = _MODE1_PREFIXES if mode == 1 else _MODE2_PREFIXES
    results = []
    for i in tqdm(range(0, len(items), batch_size), desc=f"Generating packages mode {mode} (Steering)"):
        batch_items = items[i: i + batch_size]
        batch_prefixes = [random.choice(prefixes) for _ in batch_items]
        full_prompts = [p + itm for p, itm in zip(batch_prefixes, batch_items)]
        chat_prompts = [
            _chat_prompt(tokenizer, [{"role": "user", "content": fp}])
            for fp in full_prompts
        ]
        responses = _generate_batch(model, tokenizer, chat_prompts,
                                     max_new_tokens=256, temperature=temperature)
        for prefix, item, fp, resp in zip(batch_prefixes, batch_items, full_prompts, responses):
            results.append({"prefix": prefix, "input": item, "full_prompt": fp, "response": resp})

    with open(outfile, "w", encoding="utf-8") as f:
        for r in results:
            json.dump(r, f)
            f.write("\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--step", choices=["code", "packages"], required=True)
    ap.add_argument("--infile", required=True)
    ap.add_argument("--outfile", required=True)
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--steering_vector_path", required=True)
    ap.add_argument("--str_layer", type=int, required=True)
    ap.add_argument("--steering_strength", type=float, default=20.0)
    ap.add_argument("--component", default="res", choices=["res", "mlp", "attn"])
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16"])
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--mode", type=int, default=1, choices=[1, 2])
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    random.seed(args.seed)

    if args.step == "code":
        generate_code_steering(
            args.infile, args.outfile, args.model_path, args.steering_vector_path,
            args.str_layer, steering_strength=args.steering_strength,
            component=args.component, dtype_str=args.dtype,
            temperature=args.temperature, batch_size=args.batch_size,
        )
    else:
        generate_packages_steering(
            args.mode, args.infile, args.outfile, args.model_path, args.steering_vector_path,
            args.str_layer, steering_strength=args.steering_strength,
            component=args.component, dtype_str=args.dtype,
            temperature=args.temperature, batch_size=args.batch_size,
        )


if __name__ == "__main__":
    main()
