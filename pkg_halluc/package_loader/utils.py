"""
Utility functions, collators, and data helpers
"""

import ast
import glob
import json
import logging
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union
import re
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer
from .config import (
    CHAT_TEMPLATES,
    MODEL_CONFIGS,
)

logger = logging.getLogger(__name__)

def infer_model(identifier: str) -> str:
    """Infer model family from model name, path, or identifier string."""
    name = str(identifier).lower()
    if "qwen" in name:
        return "qwen"
    elif "deepseek" in name:
        return "deepseek"
    elif "llama" in name:
        return "llama3"
    return "llama3"

infer_model_family = infer_model

def setup_tokenizer(
    tokenizer_or_name: Union[str, Any],
    model_family: str = "auto",
    pad_token: Optional[str] = None,
    trust_remote_code: bool = True,
) -> Any:
    """Loads and configures a tokenizer with chat template and padding tokens."""
    if isinstance(tokenizer_or_name, str):
        tok = AutoTokenizer.from_pretrained(tokenizer_or_name, trust_remote_code=trust_remote_code)
        inferred_family = infer_model(tokenizer_or_name)
    else:
        tok = tokenizer_or_name
        inferred_family = infer_model(getattr(tok, "name_or_path", ""))

    family = inferred_family if model_family == "auto" else model_family.lower()
    family_config = MODEL_CONFIGS.get(family, MODEL_CONFIGS["llama3"])

    # Ensure pad_token is set
    if pad_token is not None:
        tok.pad_token = pad_token
    elif tok.pad_token_id is None:
        target_pad = family_config["pad_token"]
        if target_pad in tok.get_vocab():
            tok.pad_token = target_pad
        elif tok.eos_token is not None:
            tok.pad_token = tok.eos_token
        else:
            tok.pad_token = tok.eos_token = family_config["eos_token"]

    # Ensure chat_template is present
    if getattr(tok, "chat_template", None) is None:
        tok.chat_template = CHAT_TEMPLATES.get(family, CHAT_TEMPLATES["llama3"])

    return tok

def parse_package_list(value: Any) -> List[str]:
    """Parsing package list from string representation, list, set, or tuple."""
    if value is None:
        return []
    if isinstance(value, (list, set, tuple)):
        return [str(x).strip() for x in value if str(x).strip()]
    if isinstance(value, float) and pd.isna(value):
        return []
    if isinstance(value, str):
        val = value.strip()
        if not val or val == "[]" or val.lower() == "none" or val.lower() == "nan":
            return []
        try:
            parsed = ast.literal_eval(val)
            if isinstance(parsed, (list, tuple, set)):
                return [str(x).strip() for x in parsed if str(x).strip()]
            return [str(parsed).strip()]
        except Exception:
            items = val.strip("[]'\"").split(",")
            return [item.strip().strip("'\"") for item in items if item.strip().strip("'\"")]
    return [str(value).strip()]


def format_packages_as_string(packages: List[str], empty_fallback: str = "None") -> str:
    """Formats a list of package names into a comma-separated string."""
    clean_pkgs = [p for p in packages if p]
    if not clean_pkgs:
        return empty_fallback
    return ", ".join(clean_pkgs)


def _load_tabular_file(path: str) -> pd.DataFrame:
    """Load either a benchmark CSV or the project's JSONL master file."""
    if path.lower().endswith((".json", ".jsonl")):
        with open(path, encoding="utf-8") as handle:
            records = [json.loads(line) for line in handle if line.strip()]
        return pd.DataFrame(records)
    return pd.read_csv(path)

def load_csv_data(data_source: Union[str, List[str], pd.DataFrame]) -> pd.DataFrame:
    """
    Loads benchmark CSV files or DataFrames into a single combined pandas DataFrame
    Automatically discovers result CSV files and tracks the source file in `_source_file`
    """
    dfs = []
    if isinstance(data_source, pd.DataFrame):
        df = data_source.copy()
        if "_source_file" not in df.columns:
            df["_source_file"] = "in_memory_dataframe"
        dfs.append(df)
    elif isinstance(data_source, list):
        for path in data_source:
            if os.path.isfile(path):
                df = _load_tabular_file(path)
                df["_source_file"] = os.path.basename(path)
                dfs.append(df)
    elif isinstance(data_source, str):
        if os.path.isdir(data_source):
            csv_files = glob.glob(os.path.join(data_source, "*results*.csv")) or glob.glob(os.path.join(data_source, "*.csv"))
            if not csv_files:
                csv_files = glob.glob(os.path.join(data_source, "**", "*results*.csv"), recursive=True)
            for path in csv_files:
                if "FINAL_RESULTS" in path or "PACKAGE_NAMES" in path:
                    continue
                df = _load_tabular_file(path)
                df["_source_file"] = os.path.basename(path)
                dfs.append(df)
        elif os.path.isfile(data_source):
            df = _load_tabular_file(data_source)
            df["_source_file"] = os.path.basename(data_source)
            dfs.append(df)
        else:
            raise FileNotFoundError(f"Data source path not found: {data_source}")

    if not dfs:
        raise ValueError(f"No valid data loaded from source: {data_source}")

    combined_df = pd.concat(dfs, ignore_index=True)
    return combined_df

def normalize_package(name: str) -> str:
    """Normalize package name"""
    if not name or not isinstance(name, str):
        return ""
    name = re.sub(r"^\d+[\.\)]\s*", "", name)
    name = re.sub(r"(?<=.)\n(?=.)", " ", name)
    name = re.sub(r"\n", "", name)
    name = re.sub(r"[-_.]+", "-", name)
    return name.strip(" `.-_\"'").lower()


def make_package_pattern(normalized_name: str) -> re.Pattern:
    """Create regex pattern matching a package name with flexible separator variants"""
    parts = normalized_name.split("-")
    if len(parts) == 1:
        return re.compile(rf"\b{re.escape(parts[0])}\b", re.IGNORECASE)
    flexible = r"[-_.]+".join(re.escape(p) for p in parts if p)
    return re.compile(rf"\b{flexible}\b", re.IGNORECASE)


def find_package_positions(text: str, packages: List[str]) -> List[Tuple[int, int]]:
    """
    Find non-overlapping character positions (start, end) of package names in text
    Sorts packages longest-first to avoid partial sub-matches
    """
    positions = []
    used = [False] * len(text)

    clean_pkgs = sorted(
        set(normalize_package(p) for p in packages if normalize_package(p)),
        key=len,
        reverse=True,
    )

    for pkg in clean_pkgs:
        pattern = make_package_pattern(pkg)
        matches = list(pattern.finditer(text))

        for match in matches:
            start, end = match.span()
            if not any(used[start:end]):
                positions.append((start, end))
                for i in range(start, end):
                    used[i] = True

    return positions


def build_token_to_char_map(
    tokenizer: Any,
    text: str,
    token_ids: List[int],
) -> List[Tuple[int, int]]:
    """
    Map each token to its (start, end) character span in the original text
    Handles SentencePiece and BPE space absorption
    """
    offsets = []
    pos = 0

    for i, tid in enumerate(token_ids):
        tok_text = tokenizer.decode([tid])
        if not tok_text:
            offsets.append((pos, pos))
            continue

        if text[pos : pos + len(tok_text)] == tok_text:
            offsets.append((pos, pos + len(tok_text)))
            pos += len(tok_text)
            continue

        ws = pos
        while ws < len(text) and text[ws] in " \t\n\r":
            ws += 1

        if text[ws : ws + len(tok_text)] == tok_text:
            offsets.append((ws, ws + len(tok_text)))
            pos = ws + len(tok_text)
            continue

        found = False
        for j in range(pos, min(pos + 30, len(text))):
            if text[j : j + len(tok_text)] == tok_text:
                offsets.append((j, j + len(tok_text)))
                pos = j + len(tok_text)
                found = True
                break

        if not found:
            pre = tokenizer.decode(token_ids[:i], skip_special_tokens=False)
            with_tok = tokenizer.decode(token_ids[: i + 1], skip_special_tokens=False)
            n_chars = max(len(with_tok) - len(pre), 1)

            while pos < len(text) and text[pos] in " \t\n\r":
                pos += 1
            end = min(pos + n_chars, len(text))
            offsets.append((pos, end))
            pos = end

    return offsets


def overlaps(tok_span: Tuple[int, int], positions: List[Tuple[int, int]]) -> bool:
    """Check if token character span overlaps with any target package span."""
    tok_start, tok_end = tok_span
    for pos_start, pos_end in positions:
        if not (tok_end <= pos_start or pos_end <= tok_start):
            return True
    return False

@dataclass
class UnlearningRecord:
    """Represents a single unlearning instance (forget or retain)."""
    sample_id: str
    split_type: str  # 'forget' or 'retain'
    mode: int        # 1 or 2
    system_prompt: str
    user_prompt: str
    completion: str  # Target response for SFT/GA/NPO
    chosen: str      # Preferred response for DPO
    rejected: str    # Unwanted response for DPO
    valid_packages: List[str]
    hallucinated_packages: List[str]

class DataCollatorForUnlearning:
    """
    Collator that dynamically pads batches for Causal LM training
    Pads input_ids with pad_token_id and labels with -100
    """

    def __init__(self, tokenizer: Any, pad_to_multiple_of: Optional[int] = 8):
        self.tokenizer = tokenizer
        self.pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
        self.pad_to_multiple_of = pad_to_multiple_of

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        if "input_ids" not in features[0]:
            # Non-tokenized batch
            batch = {}
            for key in features[0].keys():
                batch[key] = [f[key] for f in features]
            return batch

        max_len = max(len(f["input_ids"]) for f in features)
        if self.pad_to_multiple_of is not None:
            max_len = ((max_len + self.pad_to_multiple_of - 1) // self.pad_to_multiple_of) * self.pad_to_multiple_of

        batch_input_ids = []
        batch_attention_mask = []
        batch_labels = []

        for f in features:
            ids = f["input_ids"]
            mask = f["attention_mask"]
            lab = f["labels"]
            pad_len = max_len - len(ids)

            # Right-padding
            batch_input_ids.append(ids + [self.pad_token_id] * pad_len)
            batch_attention_mask.append(mask + [0] * pad_len)
            batch_labels.append(lab + [-100] * pad_len)

        return {
            "input_ids": torch.tensor(batch_input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(batch_attention_mask, dtype=torch.long),
            "labels": torch.tensor(batch_labels, dtype=torch.long),
            "sample_id": [f["sample_id"] for f in features],
            "split_type": [f["split_type"] for f in features],
            "mode": torch.tensor([f["mode"] for f in features], dtype=torch.long),
        }

def tsv_collate(
    prompts: List[torch.Tensor], labels: Union[List[int], np.ndarray, torch.Tensor]
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Exact collate_fn from TSV (tsv-main/train_utils.py)
    Pads prompts to the maximum sequence length in the batch
    """
    # Ensure all prompts are 2D (1, seq_len)
    normalized_prompts = []
    for p in prompts:
        if p.dim() == 1:
            p = p.unsqueeze(0)
        normalized_prompts.append(p)

    max_seq_len = max(prompt.size(1) for prompt in normalized_prompts)
    batch_size = len(normalized_prompts)
    dtype = normalized_prompts[0].dtype
    device = normalized_prompts[0].device

    prompts_padded = torch.zeros(batch_size, 1, max_seq_len, dtype=dtype, device=device)
    for i, prompt in enumerate(normalized_prompts):
        seq_len = prompt.size(1)
        prompts_padded[i, :, :seq_len] = prompt

    if isinstance(labels, torch.Tensor):
        labels_tensor = labels.to(device=device, dtype=torch.long)
    else:
        labels_tensor = torch.tensor(labels, dtype=torch.long, device=device)

    return prompts_padded, labels_tensor


class TSVBatchCollator:
    """
    Collator class for standard PyTorch DataLoader that produces
    TSV-formatted batches including attention_mask
    """

    def __init__(self, device: Optional[torch.device] = None):
        self.device = device

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        prompts = [item["prompt"] for item in batch]
        labels = [item["label"] for item in batch]

        prompts_padded, labels_tensor = tsv_collate(prompts, labels)
        attention_mask = (prompts_padded != 0).half()

        if self.device is not None:
            prompts_padded = prompts_padded.to(self.device)
            labels_tensor = labels_tensor.to(self.device)
            attention_mask = attention_mask.to(self.device)

        return {
            "prompts": prompts_padded,
            "labels": labels_tensor,
            "attention_mask": attention_mask,
            "sample_ids": [item["sample_id"] for item in batch],
        }