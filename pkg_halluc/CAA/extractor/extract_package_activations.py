"""
Extract package-token activations in completions across all model layers
(at the first package token or right after each comma), and compute
layer-pair activation differences (Layer-pair differences).
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from pkg_halluc.CAA.extractor.behaviors import get_vector_dir, PACKAGE_HALLUCINATION

if sys.stdout.encoding != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Add CAA to sys.path
CAA_DIR = os.path.dirname(os.path.abspath(__file__))
if CAA_DIR not in sys.path:
    sys.path.append(CAA_DIR)


# Short model presets mapping to Hugging Face IDs
MODEL_PRESETS = {
    "llama3.2-1b": "meta-llama/Llama-3.2-1B-Instruct",
    "llama3.2-3b": "meta-llama/Llama-3.2-3B-Instruct",
    "llama-1b": "meta-llama/Llama-3.2-1B-Instruct",
    "llama-3b": "meta-llama/Llama-3.2-3B-Instruct",
    "qwen2.5-coder-0.5b": "Qwen/Qwen2.5-Coder-0.5B-Instruct",
    "qwen2.5-coder-1.5b": "Qwen/Qwen2.5-Coder-1.5B-Instruct",
    "qwen2.5-coder-3b": "Qwen/Qwen2.5-Coder-3B-Instruct",
    "deepseek-coder-1.3b": "deepseek-ai/deepseek-coder-1.3b-instruct",
}


def resolve_model_name(name: str) -> str:
    return MODEL_PRESETS.get(name.lower(), name)


def make_model_save_suffix(model_name: str) -> str:
    """Generate short model suffix for CAA-compatible vector filenames"""
    clean_name = os.path.basename(model_name.rstrip("/\\"))
    clean_name = clean_name.replace("models--", "").replace("--", "_")
    return clean_name


def parse_layer_pairs(
    pair_specs: Optional[List[str]],
    layers: List[int],
) -> List[Tuple[int, int]]:
    """
    Parse layer pair specifications from CLI arguments:
    - 'adjacent': (0, 1), (1, 2), ..., (L-2, L-1)
    - 'all': all unique pairs (l1, l2) with l1 < l2
    - '0-5', '5-10', '0,15': specific pairs
    """
    if not pair_specs:
        return []

    pairs = []
    layer_set = set(layers)

    for spec in pair_specs:
        spec_lower = spec.lower().strip()
        if spec_lower == "adjacent":
            for i in range(len(layers) - 1):
                pairs.append((layers[i], layers[i + 1]))
        elif spec_lower in ("all", "all_pairs"):
            for i in range(len(layers)):
                for j in range(i + 1, len(layers)):
                    pairs.append((layers[i], layers[j]))
        else:
            # Formats: '0-5', '0,5', '0:5', '0_5'
            parts = re.split(r"[-:,_]", spec)
            if len(parts) == 2:
                try:
                    l1, l2 = int(parts[0].strip()), int(parts[1].strip())
                    if l1 in layer_set and l2 in layer_set and l1 != l2:
                        pairs.append((min(l1, l2), max(l1, l2)))
                    else:
                        print(f"Warning: Layer pair {spec} is outside extracted layers {layers}")
                except ValueError:
                    print(f"Warning: Could not parse layer pair: {spec}")

    # Remove duplicates preserving order
    unique_pairs = []
    seen = set()
    for p in pairs:
        if p not in seen:
            seen.add(p)
            unique_pairs.append(p)
    return unique_pairs


def find_package_token_spans(
    tokenizer: Any,
    response_ids: List[int],
    offset_mapping: Optional[List[Tuple[int, int]]],
    packages_info: List[Dict[str, Any]],
    completion: str,
    prompt_len: int,
) -> List[Dict[str, Any]]:
    """
    Locate precise token indices for each package within the concatenated sequence:
    - First package (idx = 0): first token of the package in completion.
    - Subsequent packages (idx > 0): first token of the package right after comma/space.
    """
    token_spans = []

    # Reconstruct offsets if not provided by fast tokenizer
    if not offset_mapping or all(s == e for s, e in offset_mapping):
        offsets = []
        pos = 0
        for tid in response_ids:
            t_str = tokenizer.decode([tid])
            if not t_str:
                offsets.append((pos, pos))
                continue
            idx = completion.find(t_str, pos)
            if idx != -1:
                offsets.append((idx, idx + len(t_str)))
                pos = idx + len(t_str)
            else:
                offsets.append((pos, pos + len(t_str)))
                pos += len(t_str)
        offset_mapping = offsets

    for p_info in packages_info:
        char_start = p_info["char_start"]
        char_end = p_info["char_end"]
        is_first = p_info.get("is_first", False)

        # Find tokens overlapping with [char_start, char_end]
        matched_token_indices = []
        for i, (tok_s, tok_e) in enumerate(offset_mapping):
            if tok_e <= tok_s:
                continue
            if max(tok_s, char_start) < min(tok_e, char_end):
                matched_token_indices.append(i)

        if not matched_token_indices:
            # Fallback: find nearest token to char_start
            best_idx = 0
            best_diff = float("inf")
            for i, (tok_s, tok_e) in enumerate(offset_mapping):
                diff = abs(tok_s - char_start)
                if diff < best_diff:
                    best_diff = diff
                    best_idx = i
            matched_token_indices = [best_idx]

        first_tok_in_resp = matched_token_indices[0]
        # Token position in full sequence (prompt + response)
        package_start_token_idx = prompt_len + first_tok_in_resp

        # Boundary token immediately preceding the package:
        if is_first:
            boundary_token_idx = max(0, prompt_len - 1)
        else:
            boundary_token_idx = max(0, prompt_len + first_tok_in_resp - 1)

        all_package_token_indices = [prompt_len + i for i in matched_token_indices]

        token_spans.append(
            {
                "package_name": p_info["package_name"],
                "is_hallucinated": p_info["is_hallucinated"],
                "position_idx": p_info["position_idx"],
                "is_first": is_first,
                "char_start": char_start,
                "char_end": char_end,
                "package_start_token_idx": package_start_token_idx,
                "boundary_token_idx": boundary_token_idx,
                "all_package_token_indices": all_package_token_indices,
            }
        )

    return token_spans


def extract_activations_for_sample(
    model: AutoModelForCausalLM,
    tokenizer: Any,
    item: Dict[str, Any],
    layers: List[int],
    position_mode: str = "package_start",
    device: str = "cuda",
) -> List[Dict[str, Any]]:
    """
    Extract multi-layer activation tensor (num_layers, hidden_dim) for each package in a sample:
    - position_mode:
        + 'package_start': first token of the package (at appearance point).
        + 'boundary': boundary token immediately preceding the package (end of prompt or comma).
        + 'mean': average activation across all tokens composing the package.
    """
    sys_prompt = item.get("system_prompt", "").strip()
    user_prompt = item.get("user_prompt", "").strip()
    completion = item.get("completion", "").strip()
    packages_info = item.get("packages_info", [])

    if not completion or not packages_info:
        return []

    # 1. Format prompt using chat template if supported
    messages = []
    if sys_prompt:
        messages.append({"role": "system", "content": sys_prompt})
    messages.append({"role": "user", "content": user_prompt})

    try:
        prompt_ids = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True
        )
    except Exception:
        # Fallback raw text if model does not support chat template
        prompt_text = f"{sys_prompt}\n\n{user_prompt}\n\nAssistant: " if sys_prompt else f"{user_prompt}\n\nAssistant: "
        prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=True)

    if isinstance(prompt_ids, dict) or hasattr(prompt_ids, "input_ids"):
        prompt_ids = prompt_ids["input_ids"]

    # 2. Tokenize completion
    resp_enc = tokenizer(
        completion,
        add_special_tokens=False,
        return_offsets_mapping=True,
    )
    response_ids = resp_enc["input_ids"]
    offset_mapping = [tuple(x) for x in resp_enc.get("offset_mapping", [])]

    if not response_ids:
        return []

    prompt_len = len(prompt_ids)
    full_input_ids = prompt_ids + response_ids
    input_tensor = torch.tensor([full_input_ids], device=device)

    # 3. Locate tokens for each package
    spans = find_package_token_spans(
        tokenizer=tokenizer,
        response_ids=response_ids,
        offset_mapping=offset_mapping,
        packages_info=packages_info,
        completion=completion,
        prompt_len=prompt_len,
    )

    # 4. Forward pass to retrieve hidden_states for all layers
    with torch.no_grad():
        outputs = model(input_tensor, output_hidden_states=True)
        hidden_states = outputs.hidden_states  # Tuple: (num_layers + 1) * [1, seq_len, hidden_dim]

    extracted_results = []
    num_total_hidden = len(hidden_states)

    # Map layer l to hidden_states index (layer 0 is embedding, layer l is hidden_states[l+1])
    layer_indices = [(l + 1 if (l + 1) < num_total_hidden else l) for l in layers]

    for span in spans:
        # Determine target token index by position_mode
        if position_mode == "package_start":
            target_indices = [span["package_start_token_idx"]]
        elif position_mode == "boundary":
            target_indices = [span["boundary_token_idx"]]
        elif position_mode == "mean":
            target_indices = span["all_package_token_indices"]
        else:
            target_indices = [span["package_start_token_idx"]]

        # Extract activation tensor across all specified layers: shape (num_layers, hidden_dim)
        extracted_layers = []
        for idx in layer_indices:
            layer_tensor = hidden_states[idx][0]  # [seq_len, hidden_dim]
            vec = layer_tensor[target_indices].mean(dim=0).detach().cpu()
            extracted_layers.append(vec)

        # Stack into 2D tensor: [num_layers, hidden_dim]
        stacked_layer_activations = torch.stack(extracted_layers)

        extracted_results.append(
            {
                "package_name": span["package_name"],
                "is_hallucinated": span["is_hallucinated"],
                "position_idx": span["position_idx"],
                "is_first": span["is_first"],
                "target_indices": target_indices,
                "layer_activations": stacked_layer_activations,  # [num_layers, hidden_dim]
            }
        )

    return extracted_results


def main():
    parser = argparse.ArgumentParser(
        description="Extract package-token activations across layers and compute layer-pair activation differences"
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="llama3.2-1b",
        help="Model preset name (e.g., llama3.2-1b, qwen2.5-coder-1.5b) or HF model path",
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="CAA/datasets/generate/package_hallucination/generate_dataset.json",
        help="Path to generate_dataset.json file for package_hallucination",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="",
        help="Output directory for steering vectors (default: CAA/vectors/package_hallucination/)",
    )
    parser.add_argument(
        "--layers",
        nargs="+",
        type=int,
        default=None,
        help="List of layer indices to extract (default: all layers of the model)",
    )
    parser.add_argument(
        "--position_mode",
        type=str,
        choices=["package_start", "boundary", "mean"],
        default="package_start",
        help="Activation extraction point: 'package_start' (first package token), 'boundary' (preceding boundary token), 'mean' (full package average)",
    )
    parser.add_argument(
        "--layer_pairs",
        nargs="+",
        type=str,
        default=["adjacent"],
        help="Layer pairs for difference computation: 'adjacent', 'all', or specific pairs like '0-5' '5-10'",
    )
    parser.add_argument(
        "--save_layer_diffs",
        action="store_true",
        default=True,
        help="Save layer-pair difference steering vectors to layer_diffs/ directory",
    )
    parser.add_argument(
        "--direction",
        type=str,
        choices=["non_hallucination", "hallucination"],
        default="non_hallucination",
        help="Steering vector direction: 'non_hallucination' (default, valid - hall) or 'hallucination' (hall - valid)",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Limit number of samples for rapid testing (default: process all)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Compute device ('cuda' or 'cpu')",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16" if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else "float16",
        help="Data type ('bfloat16', 'float16', 'float32')",
    )
    parser.add_argument(
        "--save_activations",
        action="store_true",
        default=False,
        help="Save raw package activation tensors",
    )
    parser.add_argument(
        "--from_config",
        action="store_true",
        default=False,
        help="Initialize model architecture from config without loading weights (rapid offline testing)",
    )
    parser.add_argument(
        "--local_files_only",
        action="store_true",
        default=True,
        help="Load only from local cache without online Hugging Face checks",
    )

    args = parser.parse_args()

    model_id = resolve_model_name(args.model_name_or_path)
    model_suffix = make_model_save_suffix(model_id)

    if not args.output_dir:
        args.output_dir = get_vector_dir(PACKAGE_HALLUCINATION)
    os.makedirs(args.output_dir, exist_ok=True)

    layer_diff_dir = os.path.join(args.output_dir, "layer_diffs")
    if args.save_layer_diffs:
        os.makedirs(layer_diff_dir, exist_ok=True)

    print("=" * 65, flush=True)
    print("CAA MULTI-LAYER & LAYER-PAIR ACTIVATION EXTRACTOR", flush=True)
    print(f"- Model: {model_id} (suffix: {model_suffix})", flush=True)
    print(f"- Dataset: {args.dataset_path}", flush=True)
    print(f"- Position mode: {args.position_mode}", flush=True)
    print(f"- Layer pairs mode: {args.layer_pairs}", flush=True)
    print(f"- Device: {args.device} | Dtype: {args.dtype}", flush=True)
    print(f"- From config only: {args.from_config}", flush=True)
    print(f"- Output dir: {args.output_dir}", flush=True)
    print("=" * 65, flush=True)

    # 1. Load dataset
    if not os.path.exists(args.dataset_path):
        raise FileNotFoundError(f"Dataset not found at {args.dataset_path}")

    with open(args.dataset_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if args.max_samples:
        data = data[: args.max_samples]
    print(f"-> Loaded {len(data)} samples.", flush=True)

    # 2. Initialize Tokenizer & Model
    print(f"-> Loading Tokenizer & Model from {model_id}...", flush=True)
    torch_dtype = getattr(torch, args.dtype, torch.float16)

    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            trust_remote_code=True,
            local_files_only=args.local_files_only,
        )
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if args.from_config:
        try:
            config = AutoConfig.from_pretrained(
                model_id,
                trust_remote_code=True,
                local_files_only=args.local_files_only,
            )
        except Exception:
            config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
        if args.layers is not None:
            config.num_hidden_layers = max(args.layers) + 1
        model = AutoModelForCausalLM.from_config(config)
        model = model.to(dtype=torch_dtype, device=args.device)
    else:
        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=torch_dtype,
                device_map=args.device,
                trust_remote_code=True,
                local_files_only=args.local_files_only,
            )
        except Exception:
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=torch_dtype,
                device_map=args.device,
                trust_remote_code=True,
            )
    model.eval()

    num_layers = model.config.num_hidden_layers
    if args.layers is None:
        target_layers = list(range(num_layers))
    else:
        target_layers = [l for l in args.layers if 0 <= l < num_layers]

    # Parse target layer pairs for difference computation
    target_pairs = parse_layer_pairs(args.layer_pairs, target_layers)

    print(f"-> Extracting {len(target_layers)} layers: {target_layers}", flush=True)
    print(f"-> Computing differences for {len(target_pairs)} layer pairs: {target_pairs[:8]}{' ...' if len(target_pairs) > 8 else ''}", flush=True)

    # 3. Extract activations across samples
    hall_tensors = []  # List of [num_layers, hidden_dim]
    val_tensors = []   # List of [num_layers, hidden_dim]
    meta_records = []

    for item in tqdm(data, desc="Extracting multi-layer activations"):
        sample_results = extract_activations_for_sample(
            model=model,
            tokenizer=tokenizer,
            item=item,
            layers=target_layers,
            position_mode=args.position_mode,
            device=args.device,
        )

        for res in sample_results:
            is_hall = res["is_hallucinated"]
            # Tensor [num_layers, hidden_dim]
            stacked_act = res["layer_activations"]

            if is_hall:
                hall_tensors.append(stacked_act)
            else:
                val_tensors.append(stacked_act)

            meta_records.append(
                {
                    "package_name": res["package_name"],
                    "is_hallucinated": is_hall,
                    "position_idx": res["position_idx"],
                    "is_first": res["is_first"],
                }
            )

    if not hall_tensors or not val_tensors:
        raise ValueError("Not enough hallucinated or valid samples to compute steering vector!")

    # 4. Compute and save multi-layer steering vectors
    print("\n[1/3] Computing & Saving Steering Vectors per layer:", flush=True)
    all_hall = torch.stack(hall_tensors).float()
    all_val = torch.stack(val_tensors).float()

    mean_hall_all_layers = all_hall.mean(dim=0)
    mean_val_all_layers = all_val.mean(dim=0)

    # Multi-layer steering vector:
    if args.direction == "non_hallucination":
        # Target behavior: NON-HALLUCINATION (Steering +v pushes model towards truthful / valid)
        steering_vecs_all_layers = mean_val_all_layers - mean_hall_all_layers  # [num_layers, hidden_dim]
        print("-> Vector direction: NON-HALLUCINATION (valid - hallucinated). Steer +v to suppress hallucination.")
    else:
        # Undesired behavior: HALLUCINATION (Steering -v pushes model away from hallucination)
        steering_vecs_all_layers = mean_hall_all_layers - mean_val_all_layers  # [num_layers, hidden_dim]
        print("-> Vector direction: HALLUCINATION (hallucinated - valid). Steer -v to suppress hallucination.")

    summary_stats = {
        "model": model_id,
        "position_mode": args.position_mode,
        "num_samples": len(data),
        "total_packages": len(meta_records),
        "num_hallucinated": len(hall_tensors),
        "num_valid": len(val_tensors),
        "num_layers": len(target_layers),
        "target_layers": target_layers,
        "layer_stats": {},
        "layer_pairs_stats": {},
    }

    # Map layer number to tensor index (0 .. len(target_layers)-1)
    layer_to_idx = {l: i for i, l in enumerate(target_layers)}

    for l in target_layers:
        l_idx = layer_to_idx[l]
        vec = steering_vecs_all_layers[l_idx]
        mean_h = mean_hall_all_layers[l_idx]
        mean_v = mean_val_all_layers[l_idx]

        norm_steering = torch.norm(vec).item()
        cos_sim = torch.cosine_similarity(mean_h.unsqueeze(0), mean_v.unsqueeze(0)).item()

        save_path = os.path.join(args.output_dir, f"vec_layer_{l}_{model_suffix}.pt")
        torch.save(vec, save_path)

        summary_stats["layer_stats"][l] = {
            "norm": norm_steering,
            "cosine_similarity_pos_neg": cos_sim,
            "save_path": save_path,
        }

    # Save aggregated all-layer steering vector tensor: [num_layers, hidden_dim]
    all_layers_path = os.path.join(args.output_dir, f"all_layers_steering_vectors_{model_suffix}.pt")
    torch.save(steering_vecs_all_layers, all_layers_path)
    print(f"  + Saved all-layers tensor to: {all_layers_path}", flush=True)

    # Optional: save raw activations
    if args.save_activations:
        act_dir = os.path.join(CAA_DIR, "activations", PACKAGE_HALLUCINATION)
        os.makedirs(act_dir, exist_ok=True)
        torch.save(all_hall, os.path.join(act_dir, f"activations_pos_all_layers_{model_suffix}.pt"))
        torch.save(all_val, os.path.join(act_dir, f"activations_neg_all_layers_{model_suffix}.pt"))
        print(f"  + Saved raw activations to: {act_dir}", flush=True)

    # 5. Compute Layer-Pair Activation Differences
    print(f"\n[2/3] Computing layer-pair activation differences ({len(target_pairs)} pairs):", flush=True)

    pair_diff_results = []
    for l1, l2 in target_pairs:
        idx1 = layer_to_idx[l1]
        idx2 = layer_to_idx[l2]

        v1 = steering_vecs_all_layers[idx1]
        v2 = steering_vecs_all_layers[idx2]

        # Difference vector between 2 layers: Delta v = v_l2 - v_l1
        delta_vec = v2 - v1

        delta_hall = mean_hall_all_layers[idx2] - mean_hall_all_layers[idx1]
        delta_val = mean_val_all_layers[idx2] - mean_val_all_layers[idx1]

        norm_delta = torch.norm(delta_vec).item()
        cos_between_layers = torch.cosine_similarity(v1.unsqueeze(0), v2.unsqueeze(0)).item()
        cos_delta_hall_val = torch.cosine_similarity(delta_hall.unsqueeze(0), delta_val.unsqueeze(0)).item()

        diff_save_path = ""
        if args.save_layer_diffs:
            diff_save_path = os.path.join(layer_diff_dir, f"vec_diff_layer_{l1}_to_{l2}_{model_suffix}.pt")
            torch.save(delta_vec, diff_save_path)

        pair_info = {
            "pair": f"layer_{l1}_to_{l2}",
            "layer_from": l1,
            "layer_to": l2,
            "delta_norm": norm_delta,
            "cosine_similarity_between_layers": cos_between_layers,
            "cosine_similarity_hall_vs_val_delta": cos_delta_hall_val,
            "save_path": diff_save_path,
        }
        pair_diff_results.append(pair_info)
        summary_stats["layer_pairs_stats"][f"{l1}->{l2}"] = pair_info

    # Print summary table of layer-pair differences
    print(f"\n{'Layer Pair (L1 -> L2)':<25} | {'Delta Norm':<18} | {'Cos Sim (v_L1, v_L2)':<22} | {'Vector File'}")
    print("-" * 90)
    for p in pair_diff_results[:15]:
        pair_name = f"Layer {p['layer_from']} -> Layer {p['layer_to']}"
        save_name = os.path.basename(p['save_path']) if p['save_path'] else "N/A"
        print(f"{pair_name:<25} | {p['delta_norm']:<18.4f} | {p['cosine_similarity_between_layers']:<22.4f} | {save_name}")
    if len(pair_diff_results) > 15:
        print(f"... (and {len(pair_diff_results) - 15} more pairs)")

    # Identify largest representation transition between adjacent layers
    adj_pairs = [p for p in pair_diff_results if p["layer_to"] == p["layer_from"] + 1]
    if adj_pairs:
        top_jump = max(adj_pairs, key=lambda x: x["delta_norm"])
        print(f"\n Largest representation transition between adjacent layers:")
        print(f"   Layer {top_jump['layer_from']} -> Layer {top_jump['layer_to']} (Norm Delta = {top_jump['delta_norm']:.4f}, Cos Sim = {top_jump['cosine_similarity_between_layers']:.4f})")

    # 6. Save summary reports and metadata
    print("\n[3/3] Saving summary reports & metadata:", flush=True)
    summary_file = os.path.join(args.output_dir, f"extraction_summary_{model_suffix}.json")
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary_stats, f, indent=2)

    diff_summary_file = os.path.join(args.output_dir, f"layer_diff_summary_{model_suffix}.json")
    with open(diff_summary_file, "w", encoding="utf-8") as f:
        json.dump(summary_stats["layer_pairs_stats"], f, indent=2)

    print(f"  + Layer steering vectors: {args.output_dir}")
    if args.save_layer_diffs:
        print(f"  + Layer-pair difference vectors: {layer_diff_dir}")
    print(f"  + Summary files: {summary_file} & {diff_summary_file}")
    print("=" * 65, flush=True)


if __name__ == "__main__":
    main()

