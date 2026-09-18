"""Compute mean-difference steering vectors from labeled package activations.

Algorithm:
1. Load CSV (LLM_LY_results.csv schema): Prompts, Answers, pip_valid, pip_hallucinated
2. For each (prompt, answer, package_name, label) triple:
   - Locate package_name in Answers to find the boundary character position
   - Tokenize prefix text (prompt + answer[:boundary]) to get boundary token index
   - Run frozen model forward pass with output_hidden_states=True
   - Extract hidden state at boundary position for each layer
3. Steering vector at layer L:
   v_L = mean(real_hidden_states_L) - mean(halluc_hidden_states_L)
4. Save dict {layer_idx: Tensor[hidden_size]} to output_dir/per_layer_vectors.pt
"""
from __future__ import annotations

import ast
import csv
import logging
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from .llm_layers import get_last_non_padded_token_rep

logger = logging.getLogger(__name__)


def _parse_list_field(raw: Any) -> list:
    if isinstance(raw, list):
        return [str(x).strip() for x in raw if str(x).strip()]
    if not isinstance(raw, str) or not raw.strip():
        return []
    s = raw.strip()
    try:
        val = ast.literal_eval(s)
        if isinstance(val, list):
            return [str(x).strip() for x in val if str(x).strip()]
    except (ValueError, SyntaxError):
        pass
    return [x.strip() for x in s.split(",") if x.strip()]


def load_training_records(csv_path) -> list:
    records = []
    with open(csv_path, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            prompt = str(row.get("Prompts", row.get("Questions", ""))).strip()
            answer = str(row.get("Answers", "")).strip()
            if not prompt or not answer:
                continue
            for pkg in _parse_list_field(row.get("pip_valid", [])):
                if pkg:
                    records.append({"prompt": prompt, "answer": answer,
                                    "package_name": pkg, "label": 1})
            for pkg in _parse_list_field(row.get("pip_hallucinated", [])):
                if pkg:
                    records.append({"prompt": prompt, "answer": answer,
                                    "package_name": pkg, "label": 0})
    logger.info("Loaded %d training records from %s", len(records), csv_path)
    return records


def _find_package_boundary(tokenizer, prompt: str, answer: str,
                            package_name: str, max_length: int = 1024):
    pos = answer.find(package_name)
    if pos == -1:
        return None
    prefix_text = prompt + answer[:pos]
    enc = tokenizer(
        prefix_text, return_tensors="pt", truncation=True,
        max_length=max_length, add_special_tokens=True,
    )
    return enc["input_ids"].shape[1] - 1


@torch.no_grad()
def extract_per_layer_activations(
    model, tokenizer, records: list,
    batch_size: int = 4, max_length: int = 1024,
) -> tuple:
    device = next(model.parameters()).device
    n_layers = model.config.num_hidden_layers

    all_per_layer = [[] for _ in range(n_layers + 1)]
    all_labels = []

    valid_records = []
    boundary_positions = []
    for rec in records:
        pos = _find_package_boundary(
            tokenizer, rec["prompt"], rec["answer"], rec["package_name"], max_length
        )
        if pos is not None and pos > 0:
            valid_records.append(rec)
            boundary_positions.append(pos)

    logger.info("%d / %d records have a locatable boundary position",
                len(valid_records), len(records))

    for i in tqdm(range(0, len(valid_records), batch_size), desc="Extracting hidden states"):
        batch_recs = valid_records[i: i + batch_size]
        batch_pos = boundary_positions[i: i + batch_size]
        batch_labels = [r["label"] for r in batch_recs]

        texts = []
        for rec in batch_recs:
            char_pos = rec["answer"].find(rec["package_name"])
            texts.append(rec["prompt"] + rec["answer"][:char_pos])

        enc = tokenizer(
            texts, return_tensors="pt", padding=True,
            truncation=True, max_length=max_length,
        )
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )

        for layer_idx, hs in enumerate(outputs.hidden_states):
            reps = []
            for b_idx, pos in enumerate(batch_pos):
                actual_len = attention_mask[b_idx].sum().item()
                safe_pos = min(pos, int(actual_len) - 1)
                reps.append(hs[b_idx, safe_pos, :].float().cpu())
            all_per_layer[layer_idx].extend(reps)

        all_labels.extend(batch_labels)

    layer_tensors = [
        torch.stack(all_per_layer[l]) if all_per_layer[l] else torch.empty(0)
        for l in range(n_layers + 1)
    ]
    return layer_tensors, all_labels


def compute_mean_difference_vectors(
    layer_activations: list, labels: list, normalize: bool = False,
) -> list:
    label_tensor = torch.tensor(labels, dtype=torch.long)
    real_mask = label_tensor == 1
    hall_mask = label_tensor == 0

    vectors = []
    for hs in layer_activations:
        if hs.numel() == 0 or real_mask.sum() == 0 or hall_mask.sum() == 0:
            vectors.append(torch.zeros(hs.shape[-1]) if hs.numel() > 0 else torch.tensor([]))
            continue
        mean_real = hs[real_mask].mean(dim=0)
        mean_hall = hs[hall_mask].mean(dim=0)
        vec = mean_real - mean_hall
        if normalize:
            vec = F.normalize(vec, dim=0)
        vectors.append(vec)
    return vectors


def compute_and_cache_steering_vectors(
    model, tokenizer, csv_path, output_dir,
    batch_size: int = 4, max_length: int = 1024, normalize: bool = False,
) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    records = load_training_records(csv_path)
    if not records:
        raise ValueError(f"No valid training records found in {csv_path}")

    layer_activations, labels = extract_per_layer_activations(
        model, tokenizer, records, batch_size=batch_size, max_length=max_length
    )
    vectors = compute_mean_difference_vectors(layer_activations, labels, normalize=normalize)

    out = {i: v for i, v in enumerate(vectors)}
    save_path = output_dir / "per_layer_vectors.pt"
    torch.save(out, save_path)
    torch.save(torch.tensor(labels), output_dir / "labels.pt")

    logger.info("Saved %d per-layer vectors to %s", len(out), save_path)
    return save_path
