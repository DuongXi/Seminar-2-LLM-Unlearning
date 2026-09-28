"""
Generate CAA Steering Vectors for Package Hallucination Unlearning
"""

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Tuple
import torch
from tqdm import tqdm
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

if sys.stdout.encoding != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", ".."))
CAA_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))

for p in [PROJECT_ROOT, CAA_DIR, SCRIPT_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

from pkg_halluc.common.model_presets import resolve_model_name

def make_clean_model_suffix(model_id_or_preset: str) -> str:
    """Generate a clean, unambiguous suffix for saved vector files."""
    name = os.path.basename(model_id_or_preset.rstrip("/\\"))
    name = name.replace("models--", "").replace("--", "_")
    return name

def tokenize_sample(
    tokenizer: Any,
    sys_prompt: str,
    user_prompt: str,
    response_text: str,
) -> Tuple[List[int], int, int]:
    """
    Format prompt with chat template and append response tokens.
    Returns: (full_token_ids, prompt_len, response_len)
    """
    messages = []
    if sys_prompt:
        messages.append({"role": "system", "content": sys_prompt})
    messages.append({"role": "user", "content": user_prompt})

    try:
        prompt_ids = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True
        )
    except Exception:
        # Fallback plain prompt
        raw_prompt = f"{sys_prompt}\n\n{user_prompt}\n\nAssistant: " if sys_prompt else f"{user_prompt}\n\nAssistant: "
        prompt_ids = tokenizer.encode(raw_prompt, add_special_tokens=True)

    if isinstance(prompt_ids, dict) or hasattr(prompt_ids, "input_ids"):
        prompt_ids = prompt_ids["input_ids"]

    resp_ids = tokenizer.encode(response_text, add_special_tokens=False)
    if not resp_ids:
        resp_ids = [tokenizer.eos_token_id]

    full_ids = prompt_ids + resp_ids
    return full_ids, len(prompt_ids), len(resp_ids)

def main():
    parser = argparse.ArgumentParser(
        description="Generate CAA steering vectors tailored for selected models."
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to model config json.",
    )
    parser.add_argument(
        "--main_path",
        type=str,
        default=None,
        help="Path to model data directory",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default=None,
        help="Model preset",
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default=None,
        help="Path to generate dataset",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory to store extracted steering vectors",
    )
    parser.add_argument(
        "--layers",
        nargs="+",
        type=int,
        default=None,
        help="Specific layer indices to compute vectors for",
    )
    parser.add_argument(
        "--token_position",
        type=str,
        choices=["last_token", "mean_response"],
        default="last_token",
        help="Which token activations to extract: 'last_token'",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Limit number of dataset samples to process for quick testing",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run inference",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="auto",
        help="Compute dtype: 'auto', 'bfloat16', 'float16', 'float32'",
    )
    parser.add_argument(
        "--from_config",
        action="store_true",
        default=False,
        help="Initialize model architecture without weights for fast offline pipeline testing",
    )
    parser.add_argument(
        "--local_files_only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Load model files only from the local Hugging Face cache",
    )

    args = parser.parse_args()

    model_name_or_path = args.model_name_or_path
    main_path = args.main_path

    # Read model, data path, and dtype from config unless overridden.
    if args.config:
        if not os.path.exists(args.config):
            raise FileNotFoundError(f"Config file not found at: {args.config}")
        with open(args.config, "r", encoding="utf-8") as f:
            cfg = json.load(f)

        if not model_name_or_path:
            model_name_or_path = cfg.get("model_name")

        if main_path is None:
            main_path = cfg.get("data", {}).get("main_path")

        if args.dtype == "auto" and cfg.get("dtype") and cfg.get("dtype") != "auto":
            args.dtype = cfg["dtype"]

    if not model_name_or_path:
        raise ValueError("Model path/name not specified! Provide --model_name_or_path or --config.")

    if not main_path:
        raise ValueError("Model data main_path not specified! Provide --main_path or --config with data.main_path.")

    model_id = resolve_model_name(model_name_or_path)
    model_key = model_name_or_path
    model_suffix = make_clean_model_suffix(model_id)

    # Select the compute dtype.
    if args.dtype == "auto":
        torch_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
    else:
        torch_dtype = getattr(torch, args.dtype, torch.float16)

    if not args.dataset_path:
        args.dataset_path = os.path.join(main_path, "contrastive", "generate", "generate_dataset_shuffled.json")

    if not args.output_dir:
        args.output_dir = os.path.join(main_path, "contrastive", "vectors")

    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 70, flush=True)
    print("CAA STEERING VECTOR GENERATOR (SELECTED MODELS)", flush=True)
    print(f"- Target Model: {model_id} (preset: {model_name_or_path})", flush=True)
    print(f"- Model Suffix: {model_suffix}", flush=True)
    print(f"- Main Path: {main_path}", flush=True)
    print(f"- Dataset: {args.dataset_path}", flush=True)
    print(f"- Token Extraction: {args.token_position}", flush=True)
    print(f"- Device: {args.device} | Dtype: {torch_dtype}", flush=True)
    print(f"- Output Directory: {args.output_dir}", flush=True)
    print("=" * 70, flush=True)

    # Load the contrastive dataset.
    if not os.path.exists(args.dataset_path):
        raise FileNotFoundError(f"Dataset not found at {args.dataset_path}")

    with open(args.dataset_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if args.max_samples:
        data = data[: args.max_samples]
    print(f"[*] Loaded {len(data)} comparison samples.", flush=True)

    # Load the tokenizer and model.
    print(f"[*] Loading Tokenizer & Model for {model_id}...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        trust_remote_code=True,
        local_files_only=args.local_files_only,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if args.from_config:
        config = AutoConfig.from_pretrained(
            model_id,
            trust_remote_code=True,
            local_files_only=args.local_files_only,
        )
        if args.layers is not None:
            config.num_hidden_layers = max(args.layers) + 1
        model = AutoModelForCausalLM.from_config(config)
        model = model.to(dtype=torch_dtype, device=args.device)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch_dtype,
            device_map=args.device,
            trust_remote_code=True,
            local_files_only=args.local_files_only,
        )

    model.eval()

    num_layers = getattr(model.config, "num_hidden_layers", getattr(model.config, "n_layer", 32))
    if args.layers is None:
        target_layers = list(range(num_layers))
    else:
        target_layers = [l for l in args.layers if 0 <= l < num_layers]

    print(f"[*] Target layers ({len(target_layers)} layers): {target_layers}", flush=True)

    # Accumulators for positive (matching / truthful) and negative (not matching / hallucinated)
    # Stored as list of tensors per layer
    pos_activations: Dict[int, List[torch.Tensor]] = {l: [] for l in target_layers}
    neg_activations: Dict[int, List[torch.Tensor]] = {l: [] for l in target_layers}

    # Extract activations from contrastive pairs.
    for item in tqdm(data, desc="Extracting contrastive activations"):
        sys_prompt = item.get("system_prompt", "").strip()
        user_prompt = item.get("user_prompt", "").strip()
        pos_resp = item["answer_matching_behavior"]       # Truthful (valid packages)
        neg_resp = item["answer_not_matching_behavior"]   # Hallucinated (fake packages)

        # Tokenize positive and negative completions
        pos_ids, pos_prompt_len, pos_resp_len = tokenize_sample(
            tokenizer, sys_prompt, user_prompt, pos_resp
        )
        neg_ids, neg_prompt_len, neg_resp_len = tokenize_sample(
            tokenizer, sys_prompt, user_prompt, neg_resp
        )

        pos_tensor = torch.tensor([pos_ids], device=args.device)
        neg_tensor = torch.tensor([neg_ids], device=args.device)

        # Forward positive completion
        with torch.no_grad():
            pos_out = model(pos_tensor, output_hidden_states=True)
            pos_hidden = pos_out.hidden_states

        # Forward negative completion
        with torch.no_grad():
            neg_out = model(neg_tensor, output_hidden_states=True)
            neg_hidden = neg_out.hidden_states

        num_hidden = len(pos_hidden)
        layer_indices = {l: (l + 1 if (l + 1) < num_hidden else l) for l in target_layers}

        for l in target_layers:
            idx = layer_indices[l]

            # Hidden states for batch 0: [seq_len, hidden_dim]
            p_layer_h = pos_hidden[idx][0]
            n_layer_h = neg_hidden[idx][0]

            if args.token_position == "last_token":
                # Last token of the response
                p_vec = p_layer_h[-1, :].detach().cpu()
                n_vec = n_layer_h[-1, :].detach().cpu()
            else:
                # Average across all response tokens
                p_vec = p_layer_h[pos_prompt_len:, :].mean(dim=0).detach().cpu()
                n_vec = n_layer_h[neg_prompt_len:, :].mean(dim=0).detach().cpu()

            pos_activations[l].append(p_vec)
            neg_activations[l].append(n_vec)

    # Compute and save steering vectors.
    print("\n[*] Computing CAA Steering Vectors (Truthful - Hallucinated)...", flush=True)
    all_steering_vectors = []
    layer_summary = {}

    for l in target_layers:
        all_pos = torch.stack(pos_activations[l]).float()
        all_neg = torch.stack(neg_activations[l]).float()

        mean_pos = all_pos.mean(dim=0)
        mean_neg = all_neg.mean(dim=0)

        # Steering vector: points towards truthful / non-hallucination
        vec = mean_pos - mean_neg
        all_steering_vectors.append(vec)

        norm_val = torch.norm(vec).item()
        cos_sim = torch.cosine_similarity(mean_pos.unsqueeze(0), mean_neg.unsqueeze(0)).item()

        # Save single-layer vector file
        save_file = os.path.join(args.output_dir, f"vec_layer_{l}_{model_suffix}.pt")
        torch.save(vec, save_file)

        layer_summary[l] = {
            "norm": norm_val,
            "cosine_similarity_truthful_vs_hallu": cos_sim,
            "save_path": save_file,
        }

    # Stack and save all-layers tensor
    all_layers_tensor = torch.stack(all_steering_vectors)  # [num_layers, hidden_dim]
    all_layers_path = os.path.join(args.output_dir, f"all_layers_steering_vectors_{model_suffix}.pt")
    torch.save(all_layers_tensor, all_layers_path)

    # Save summary report
    summary_report = {
        "model_preset": model_key,
        "model_id": model_id,
        "model_suffix": model_suffix,
        "dataset_path": args.dataset_path,
        "token_position": args.token_position,
        "num_samples": len(data),
        "target_layers": target_layers,
        "all_layers_file": all_layers_path,
        "layer_stats": layer_summary,
    }
    summary_file = os.path.join(args.output_dir, f"vector_generation_summary_{model_suffix}.json")
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary_report, f, indent=2)

    print(f"[+] Successfully generated steering vectors for {len(target_layers)} layers!")
    print(f"[+] Individual layer vectors: {args.output_dir}/vec_layer_<L>_{model_suffix}.pt")
    print(f"[+] All-layers tensor: {all_layers_path}")
    print(f"[+] Summary report: {summary_file}")
    print("=" * 70, flush=True)


if __name__ == "__main__":
    main()