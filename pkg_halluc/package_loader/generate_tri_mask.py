"""
Generate tokenized Tri-Mask datasets
"""

import argparse
import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd
import torch

from pkg_halluc.package_loader.unlearn_loader import PackageUnlearningDataset
from pkg_halluc.package_loader.utils import (
    setup_tokenizer,
    find_package_positions,
    overlaps,
    build_token_to_char_map,
)

logger = logging.getLogger(__name__)

def build_tri_mask_record(
    tokenizer: Any,
    system_prompt: str,
    user_prompt: str,
    response: str,
    valid_pkgs: List[str],
    hall_pkgs: List[str],
    sample_id: str,
    mode_tag: str,
    template_tag: str,
    max_length: int = 2048,
) -> Optional[Dict[str, Any]]:
    """
    Tokenizes (system + user prompt) and response, produce tri-mask record
    """
    # Format prompt via chat template
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_prompt})

    prompt_ids = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True
    )
    if isinstance(prompt_ids, torch.Tensor):
        prompt_ids = prompt_ids.squeeze().tolist()

    # Tokenize response
    response_enc = tokenizer(
        response,
        add_special_tokens=False,
        return_offsets_mapping=True,
    )
    response_ids = response_enc["input_ids"]
    if isinstance(response_ids, torch.Tensor):
        response_ids = response_ids.squeeze().tolist()

    if not response_ids:
        return None

    # Use tokenizer's fast offset mapping if valid, otherwise fallback
    if "offset_mapping" in response_enc and any(
        end > start for start, end in response_enc["offset_mapping"]
    ):
        response_offsets = [tuple(p) for p in response_enc["offset_mapping"]]
    else:
        response_offsets = build_token_to_char_map(tokenizer, response, response_ids)

    # Concatenate and append EOS
    eos_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else tokenizer.pad_token_id
    input_ids = prompt_ids + response_ids + ([eos_id] if eos_id is not None else [])

    # Truncation if needed
    if len(input_ids) > max_length:
        avail_resp = max_length - len(prompt_ids) - 1
        if avail_resp <= 0:
            return None
        response_ids = response_ids[:avail_resp]
        response_offsets = response_offsets[:avail_resp]
        input_ids = prompt_ids + response_ids + ([eos_id] if eos_id is not None else [])

    valid_positions = find_package_positions(response, valid_pkgs)
    hall_positions = find_package_positions(response, hall_pkgs)

    # Build tri_mask
    tri_mask = [0] * len(prompt_ids)

    for offset in response_offsets:
        tok_start, tok_end = offset
        if tok_end <= tok_start:
            tri_mask.append(0)
        elif overlaps((tok_start, tok_end), hall_positions):
            tri_mask.append(2)  # Forget
        elif overlaps((tok_start, tok_end), valid_positions):
            tri_mask.append(1)  # Retain
        else:
            tri_mask.append(0)  # Ignore

    if eos_id is not None:
        tri_mask.append(1)  # Reinforce EOS

    # Build labels (copy input_ids, set -100 where tri_mask == 0)
    labels = list(input_ids)
    for i, m in enumerate(tri_mask):
        if m == 0:
            labels[i] = -100

    attention_mask = [1] * len(input_ids)

    return {
        "id": sample_id,
        "mode": mode_tag,
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "tri_mask": tri_mask,
    }


def resolve_model_suffix(model_name: str, explicit_suffix: Optional[str] = None) -> str:
    """
    Resolve the dataset filename suffix for the model sizes used by this project.
    """
    if explicit_suffix is not None:
        return explicit_suffix
    name = str(model_name).lower()
    if "deepseek" in name and ("1b" in name or "1.3b" in name):
        return "_1B"
    if "llama" in name and "1b" in name:
        return "_1B"
    if "qwen" in name and "0.5b" in name:
        return "_0.5B"
    if "qwen" in name and "1.5b" in name:
        return "_1.5B"
    if "qwen" in name and "3b" in name:
        return "_3B"
    if "llama" in name and "3b" in name:
        return "_3B"

    known_family = any(f in name for f in ("qwen", "llama", "deepseek"))
    size_match = re.search(r"(\d+(?:\.\d+)?)\s*b", name)
    if known_family and size_match is not None:
        return f"_{size_match.group(1)}B"
    raise ValueError(
        f"Cannot resolve dataset suffix for model {model_name!r}; "
        "provide an explicit suffix."
    )

def generate_tri_mask_dataset(
    dataset: Union[PackageUnlearningDataset, str, Path, List[str], pd.DataFrame] = None,
    tokenizer: Optional[Union[str, Any]] = None,
    model_name: str = "auto",
    model_family: str = "auto",
    output_dir: Optional[Union[str, Path]] = None,
    suffix: Optional[str] = None,
    max_length: Optional[int] = None,
    single_objective_forget: bool = True,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Generate retain and forget tri_mask records from a PackageUnlearningDataset
    """
    target_data = dataset
    if target_data is None:
        raise ValueError("dataset must be provided.")

    # Convert raw input to PackageUnlearningDataset if not already one
    if not isinstance(target_data, PackageUnlearningDataset):
        target_dataset = PackageUnlearningDataset(
            data_source=target_data,
            split_type="all",
            query_modes=[1, 2],
        )
    else:
        target_dataset = target_data

    # Resolve Tokenizer
    tok_target = tokenizer or getattr(target_dataset, "tokenizer", None)
    family = model_family if model_family != "auto" else getattr(target_dataset, "model_family", "auto")

    if tok_target is None:
        if model_name != "auto":
            tok = setup_tokenizer(model_name, model_family=family)
        else:
            raise ValueError(
                "A tokenizer, model_name, or a PackageUnlearningDataset with an initialized tokenizer must be provided."
            )
    elif isinstance(tok_target, str):
        tok = setup_tokenizer(tok_target, model_family=family)
    else:
        tok = tok_target

    # Resolve model suffix
    if model_name != "auto":
        model_id = model_name
    elif hasattr(tok, "name_or_path") and tok.name_or_path:
        model_id = tok.name_or_path
    else:
        model_id = "auto"

    resolved_suffix = resolve_model_suffix(model_id, explicit_suffix=suffix)
    max_len = max_length or getattr(target_dataset, "max_length", 2048)

    retain_records: List[Dict[str, Any]] = []
    forget_records: List[Dict[str, Any]] = []

    # Build AU Tri-mask records by iterating over dataset.records
    for record in target_dataset.records:
        template_tag = f"T{record.mode}"

        if record.split_type == "retain":
            rec = build_tri_mask_record(
                tokenizer=tok,
                system_prompt=record.system_prompt,
                user_prompt=record.user_prompt,
                response=record.completion,
                valid_pkgs=record.valid_packages,
                hall_pkgs=[],
                sample_id=record.sample_id,
                mode_tag="retain",
                template_tag=template_tag,
                max_length=max_len,
            )
            if rec:
                retain_records.append(rec)

        elif record.split_type == "forget":
            v_pkgs = [] if single_objective_forget else record.valid_packages
            rec = build_tri_mask_record(
                tokenizer=tok,
                system_prompt=record.system_prompt,
                user_prompt=record.user_prompt,
                response=record.completion,
                valid_pkgs=v_pkgs,
                hall_pkgs=record.hallucinated_packages,
                sample_id=record.sample_id,
                mode_tag="forget",
                template_tag=template_tag,
                max_length=max_len,
            )
            if rec:
                forget_records.append(rec)

    # Write to files if output_dir provided
    if output_dir is not None:
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        retain_file = out_dir / f"npo_retain_tok{resolved_suffix}.jsonl"
        forget_file = out_dir / f"npo_forget_tok{resolved_suffix}.jsonl"
        stats_file = out_dir / f"tri_mask_stats{resolved_suffix}.json"

        with open(retain_file, "w", encoding="utf-8") as f:
            for r in retain_records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

        with open(forget_file, "w", encoding="utf-8") as f:
            for r in forget_records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

        stats = {
            "model_name": str(model_id),
            "model_suffix": resolved_suffix,
            "retain_samples": len(retain_records),
            "forget_samples": len(forget_records),
            "query_modes": getattr(target_dataset, "query_modes", [1, 2]),
            "retain_file": str(retain_file),
            "forget_file": str(forget_file),
        }

        with open(stats_file, "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2)

        print(f"[OK] Tri-mask Retain Dataset : {retain_file} ({len(retain_records):,} samples)")
        print(f"[OK] Tri-mask Forget Dataset : {forget_file} ({len(forget_records):,} samples)")
        print(f"[OK] Metadata & Statistics   : {stats_file}")

    return retain_records, forget_records

def main():
    parser = argparse.ArgumentParser(
        description="Generate tokenized Tri-Mask JSONL datasets (npo_retain_tok / npo_forget_tok) for AU unlearning.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data_source",
        required=True,
        help="Path to CSV file, directory, or benchmark results.",
    )
    parser.add_argument(
        "--model_name",
        required=True,
        help="Model identifier or local checkpoint path (e.g. 'meta-llama/Llama-3.2-1B-Instruct', 'Qwen/Qwen2.5-Coder-3B-Instruct').",
    )
    parser.add_argument(
        "--model_family",
        default="auto",
        choices=["auto", "llama3", "qwen", "deepseek"],
        help="Model family for tokenizer and chat template configuration.",
    )
    parser.add_argument(
        "--output_dir",
        default="./New_Data_Set",
        help="Directory to write npo_retain_tok and npo_forget_tok files.",
    )
    parser.add_argument(
        "--suffix",
        default=None,
        help="Explicit filename suffix (e.g. '_1B', '_1.5B', '_3B'). If omitted, auto-resolved from model_name.",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=2048,
        help="Maximum sequence length for tokenization.",
    )
    parser.add_argument(
        "--query_modes",
        type=int,
        nargs="+",
        default=[1, 2],
        help="Query modes to process (1: Code->Packages, 2: Problem->Packages).",
    )

    args = parser.parse_args()

    # Direct approach: Build dataset then export to tri-mask
    dataset = PackageUnlearningDataset(
        data_source=args.data_source,
        split_type="all",
        query_modes=args.query_modes,
    )
    dataset.to_tri_mask_dataset(
        model_name=args.model_name,
        model_family=args.model_family,
        output_dir=args.output_dir,
        suffix=args.suffix,
        max_length=args.max_length,
    )


if __name__ == "__main__":
    main()
