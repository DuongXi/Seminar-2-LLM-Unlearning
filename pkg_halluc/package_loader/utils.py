"""
Utility functions, collators, and data helpers
"""

import ast
import glob
import json
import logging
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import re
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from .config import (
    CHAT_TEMPLATES,
    MODEL_CONFIGS,
    PACKAGE_PREFIX_1,
    PACKAGE_PREFIX_2,
    PACKAGE_SYSTEM_PROMPT_1,
    PACKAGE_SYSTEM_PROMPT_2,
)

logger = logging.getLogger(__name__)

def infer_model(identifier: str) -> str:
    """Infer model family from model name, path, or identifier string"""
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
    """Loads and configures a tokenizer with chat template and padding tokens"""
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
    """Parsing package list from string representation, list, set, or tuple"""
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
    """Formats a list of package names into a comma-separated string"""
    clean_pkgs = [p for p in packages if p]
    if not clean_pkgs:
        return empty_fallback
    return ", ".join(clean_pkgs)


def _load_tabular_file(path: str) -> pd.DataFrame:
    """Load either a benchmark CSV or the project's JSONL master file"""
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
    """Check if token character span overlaps with any target package span"""
    tok_start, tok_end = tok_span
    for pos_start, pos_end in positions:
        if not (tok_end <= pos_start or pos_end <= tok_start):
            return True
    return False

@dataclass
class UnlearningRecord:
    """Represents a single unlearning instance (forget or retain)"""
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

def is_tokenized(file_path: Union[str, Path]) -> bool:
    """Check if a file contains pre-tokenized pointwise records (with input_ids, labels)"""
    p = Path(file_path)
    if not p.is_file():
        return False
    suffix = p.suffix.lower()
    if suffix not in (".json", ".jsonl"):
        return False
    try:
        with p.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                if line.startswith("["):
                    data = json.loads(p.read_text(encoding="utf-8"))
                    if isinstance(data, list) and len(data) > 0 and isinstance(data[0], dict):
                        return "input_ids" in data[0] and "labels" in data[0]
                    return False
                item = json.loads(line)
                if isinstance(item, dict):
                    return "input_ids" in item and "labels" in item
                return False
    except Exception:
        return False
    return False


def load_tokenized_records(
    file_path: Union[str, Path],
    split_type: str = "all",
    query_modes: Optional[List[int]] = None,
) -> List[Dict[str, Any]]:
    """Load tokenized pointwise records from a JSON or JSONL file with optional filtering"""
    p = Path(file_path)
    if not p.is_file():
        raise FileNotFoundError(f"Tokenized records file not found: {file_path}")

    records = []
    split_type = split_type.lower()
    allowed_splits = ("forget", "retain", "val") if split_type == "all" else (split_type,)

    raw_items: List[Dict[str, Any]] = []
    with p.open("r", encoding="utf-8") as f:
        first_char = f.read(1)
        f.seek(0)
        if first_char == "[":
            raw_items = json.load(f)
        else:
            for line in f:
                line = line.strip()
                if line:
                    raw_items.append(json.loads(line))

    for item in raw_items:
        rec_split = str(item.get("split_type", "")).lower()
        rec_mode = int(item.get("mode", 1))

        if rec_split not in allowed_splits:
            continue
        if query_modes is not None and rec_mode not in query_modes:
            continue

        records.append({
            "sample_id": str(item.get("sample_id", "")),
            "split_type": rec_split,
            "mode": rec_mode,
            "input_ids": list(item.get("input_ids", [])),
            "attention_mask": list(item.get("attention_mask", [])),
            "labels": list(item.get("labels", [])),
        })
    return records


def save_tokenized_records(
    records: Union[List[UnlearningRecord], List[Dict[str, Any]]],
    file_path: Union[str, Path],
    tokenizer: Optional[Any] = None,
    max_length: int = 2048,
) -> Path:
    """
    Save records in pointwise tokenized format:
    sample_id, split_type, mode, input_ids, attention_mask, labels
    """
    p = Path(file_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with p.open("w", encoding="utf-8") as f:
        for r in records:
            if isinstance(r, dict) and "input_ids" in r and "labels" in r:
                tok_rec = r
            else:
                tok_rec = format_record(r, return_format="pointwise", tokenizer=tokenizer, max_length=max_length)

            clean_rec = {
                "sample_id": str(tok_rec["sample_id"]),
                "split_type": str(tok_rec["split_type"]),
                "mode": int(tok_rec["mode"]),
                "input_ids": list(tok_rec["input_ids"]),
                "attention_mask": list(tok_rec["attention_mask"]),
                "labels": list(tok_rec["labels"]),
            }
            f.write(json.dumps(clean_rec, ensure_ascii=False) + "\n")
            count += 1
    logger.info("Saved %d tokenized records to %s", count, p)
    print(f"Saved {count} tokenized records to {p}")
    return p


def is_preprocessed(file_path: Union[str, Path]) -> bool:
    """Check if a file contains UnlearningRecord or tokenized records"""
    p = Path(file_path)
    if not p.is_file():
        return False
    suffix = p.suffix.lower()
    if suffix not in (".json", ".jsonl"):
        return False
    try:
        with p.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                if line.startswith("["):
                    data = json.loads(p.read_text(encoding="utf-8"))
                    if isinstance(data, list) and len(data) > 0 and isinstance(data[0], dict):
                        first_item = data[0]
                        return "sample_id" in first_item and "split_type" in first_item
                    return False
                item = json.loads(line)
                if isinstance(item, dict):
                    return "sample_id" in item and "split_type" in item
                return False
    except Exception:
        return False
    return False

def load_files(
    file_path: Union[str, Path],
    split_type: str = "all",
    query_modes: Optional[List[int]] = None,
) -> List[UnlearningRecord]:
    """Load UnlearningRecord objects from a JSON or JSONL file with optional filtering"""
    p = Path(file_path)
    if not p.is_file():
        raise FileNotFoundError(f"Records file not found: {file_path}")

    records: List[UnlearningRecord] = []
    split_type = split_type.lower()
    allowed_splits = ("forget", "retain", "val") if split_type == "all" else (split_type,)

    raw_items: List[Dict[str, Any]] = []
    with p.open("r", encoding="utf-8") as f:
        first_char = f.read(1)
        f.seek(0)
        if first_char == "[":
            raw_items = json.load(f)
        else:
            for line in f:
                line = line.strip()
                if line:
                    raw_items.append(json.loads(line))

    for item in raw_items:
        rec_split = str(item.get("split_type", "")).lower()
        rec_mode = int(item.get("mode", 1))

        if rec_split not in allowed_splits:
            continue
        if query_modes is not None and rec_mode not in query_modes:
            continue

        records.append(
            UnlearningRecord(
                sample_id=str(item.get("sample_id", "")),
                split_type=rec_split,
                mode=rec_mode,
                system_prompt=str(item.get("system_prompt", "")),
                user_prompt=str(item.get("user_prompt", "")),
                completion=str(item.get("completion", "")),
                chosen=str(item.get("chosen", "")),
                rejected=str(item.get("rejected", "")),
                valid_packages=list(item.get("valid_packages", [])),
                hallucinated_packages=list(item.get("hallucinated_packages", [])),
            )
        )
    return records

def save_files(
    records: List[UnlearningRecord],
    file_path: Union[str, Path],
) -> Path:
    """Save UnlearningRecord objects to JSONL file"""
    p = Path(file_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(asdict(rec), ensure_ascii=False) + "\n")
    logger.info("Saved %d records to %s", len(records), p)
    return p

# Backwards compatibility aliases
is_preprocessed_records_file = is_preprocessed
load_records_from_file = load_files
save_records_to_file = save_files

class CombinedUnlearningLoader:
    """
    Yields paired batches:
    {'forget': forget_batch, 'retain': retain_batch}
    at each training step
    """

    def __init__(
        self,
        forget_loader: DataLoader,
        retain_loader: DataLoader,
        cycle_shorter: bool = True,
    ):
        self.forget_loader = forget_loader
        self.retain_loader = retain_loader
        self.cycle_shorter = cycle_shorter

    def __len__(self) -> int:
        if self.cycle_shorter:
            return max(len(self.forget_loader), len(self.retain_loader))
        return min(len(self.forget_loader), len(self.retain_loader))

    def __iter__(self):
        iter_f = iter(self.forget_loader)
        iter_r = iter(self.retain_loader)

        total_steps = len(self)
        for _ in range(total_steps):
            try:
                batch_f = next(iter_f)
            except StopIteration:
                if self.cycle_shorter:
                    iter_f = iter(self.forget_loader)
                    batch_f = next(iter_f)
                else:
                    break

            try:
                batch_r = next(iter_r)
            except StopIteration:
                if self.cycle_shorter:
                    iter_r = iter(self.retain_loader)
                    batch_r = next(iter_r)
                else:
                    break

            yield {"forget": batch_f, "retain": batch_r}

def extract_unlearning_records(
    df: pd.DataFrame,
    query_modes: List[int] = (1, 2),
    mode_prompts: Optional[Dict[int, Dict[str, str]]] = None,
    split_type: str = "all",
    forget_target_type: str = "all_generated",
    empty_package_fallback: str = "None",
) -> List[UnlearningRecord]:
    """Extract UnlearningRecord objects from a benchmark results DataFrame."""
    if mode_prompts is None:
        mode_prompts = {
            1: {"system_prompt": PACKAGE_SYSTEM_PROMPT_1, "prefix": PACKAGE_PREFIX_1.strip()},
            2: {"system_prompt": PACKAGE_SYSTEM_PROMPT_2, "prefix": PACKAGE_PREFIX_2.strip()},
        }

    records: List[UnlearningRecord] = []
    for idx, row in df.iterrows():
        source_file = row.get("_source_file", f"row_{idx}")

        mode_specs = [
            (
                1,
                str(row.get("Answers", "")).strip(),
                parse_package_list(row.get("valid_1", row.get("valid1", []))),
                parse_package_list(row.get("hallucinated_1", row.get("hallucination_1", []))),
                parse_package_list(row.get("Test_1", [])),
            ),
            (
                2,
                str(row.get("Prompts", row.get("Questions", ""))).strip(),
                parse_package_list(row.get("valid_2", row.get("valid2", []))),
                parse_package_list(row.get("hallucinated_2", row.get("hallucination_2", []))),
                parse_package_list(row.get("Test_2", [])),
            ),
        ]

        for mode, content, valid, hall, test_pkgs in mode_specs:
            if mode not in query_modes or not content:
                continue

            mode_info = mode_prompts.get(mode, {})
            sys_prompt = mode_info.get("system_prompt", "")
            user_msg = f"{mode_info.get('prefix', '')} {content}".strip()
            has_hall = len(hall) > 0

            valid_str = format_packages_as_string(valid, empty_package_fallback)
            if forget_target_type == "hallucinated_only":
                target_pkgs = hall
            else:
                target_pkgs = test_pkgs if test_pkgs else (valid + hall)
            forget_str = format_packages_as_string(target_pkgs, empty_package_fallback)

            sample_id = f"{source_file}_m{mode}_idx{idx}"

            # Add to forget set if it has hallucinations
            if has_hall and split_type in ("forget", "all"):
                records.append(
                    UnlearningRecord(
                        sample_id=sample_id,
                        split_type="forget",
                        mode=mode,
                        system_prompt=sys_prompt,
                        user_prompt=user_msg,
                        completion=forget_str,
                        chosen=valid_str,
                        rejected=forget_str,
                        valid_packages=valid,
                        hallucinated_packages=hall,
                    )
                )

            # Add to retain set if it had NO hallucinations and had valid packages
            if (not has_hall) and len(valid) > 0 and split_type in ("retain", "all"):
                records.append(
                    UnlearningRecord(
                        sample_id=sample_id,
                        split_type="retain",
                        mode=mode,
                        system_prompt=sys_prompt,
                        user_prompt=user_msg,
                        completion=valid_str,
                        chosen=valid_str,
                        rejected="",
                        valid_packages=valid,
                        hallucinated_packages=[],
                    )
                )

    return records

def format_record(
    record: Union[UnlearningRecord, Dict[str, Any]],
    return_format: str = "pointwise",
    tokenizer: Optional[Any] = None,
    max_length: int = 2048,
) -> Dict[str, Any]:
    """Format an UnlearningRecord into pointwise, DPO, or text dictionary format."""
    if isinstance(record, dict) and "input_ids" in record and "labels" in record:
        return record

    if return_format == "text" or tokenizer is None:
        return {
            "sample_id": record.sample_id,
            "split_type": record.split_type,
            "mode": record.mode,
            "system_prompt": record.system_prompt,
            "user_prompt": record.user_prompt,
            "completion": record.completion,
            "chosen": record.chosen,
            "rejected": record.rejected,
            "valid_packages": record.valid_packages,
            "hallucinated_packages": record.hallucinated_packages,
            "raw_source": getattr(record, "raw_source", None),
        }

    if return_format == "dpo":
        prompt_msgs = [
            {"role": "system", "content": record.system_prompt},
            {"role": "user", "content": record.user_prompt},
        ]
        prompt_text = tokenizer.apply_chat_template(
            prompt_msgs, add_generation_prompt=True, tokenize=False
        )
        return {
            "sample_id": record.sample_id,
            "prompt": prompt_text,
            "chosen": record.chosen,
            "rejected": record.rejected,
            "split_type": record.split_type,
            "mode": record.mode,
        }

    # 'pointwise' training format (input_ids, attention_mask, labels)
    messages = [
        {"role": "system", "content": record.system_prompt},
        {"role": "user", "content": record.user_prompt},
        {"role": "assistant", "content": record.completion},
    ]
    prompt_msgs = messages[:2]

    prompt_text = tokenizer.apply_chat_template(
        prompt_msgs, add_generation_prompt=True, tokenize=False
    )
    full_text = tokenizer.apply_chat_template(
        messages, add_generation_prompt=False, tokenize=False
    )

    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    full_ids = tokenizer(full_text, add_special_tokens=False)["input_ids"]

    prompt_len = len(prompt_ids)
    labels = [-100] * prompt_len + full_ids[prompt_len:]

    if len(full_ids) > max_length:
        full_ids = full_ids[:max_length]
        labels = labels[:max_length]

    attention_mask = [1] * len(full_ids)

    return {
        "sample_id": record.sample_id,
        "split_type": record.split_type,
        "mode": record.mode,
        "input_ids": full_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }

def split_records_by_prompt(
    records: Union[List[UnlearningRecord], List[Dict[str, Any]]],
    val_ratio: float = 0.1,
    seed: int = 42,
    split_retain_only: bool = True,
) -> Tuple[List[Any], List[Any]]:
    """Split records into train and validation sets grouped by prompt to prevent data leakage."""
    import random

    if val_ratio <= 0.0 or len(records) == 0:
        return list(records), []

    def get_group_key(rec: Any) -> str:
        sample_id = rec.sample_id if hasattr(rec, "sample_id") else rec.get("sample_id", "")
        if "_m" in sample_id:
            return re.sub(r"_m\d+_", "_", sample_id)
        user_prompt = rec.user_prompt if hasattr(rec, "user_prompt") else rec.get("user_prompt", "")
        return user_prompt.strip() if user_prompt else sample_id

    def get_split_type(rec: Any) -> str:
        return rec.split_type if hasattr(rec, "split_type") else rec.get("split_type", "")

    if split_retain_only:
        forget_records = [r for r in records if get_split_type(r) == "forget"]
        retain_records = [r for r in records if get_split_type(r) != "forget"]

        groups: Dict[str, List[UnlearningRecord]] = {}
        for r in retain_records:
            k = get_group_key(r)
            groups.setdefault(k, []).append(r)

        keys = sorted(groups.keys())
        rng = random.Random(seed)
        rng.shuffle(keys)

        n_val = max(1, round(len(keys) * val_ratio)) if len(keys) > 1 else 0
        val_keys = set(keys[:n_val])
        train_keys = set(keys[n_val:])

        retain_train = [r for k in train_keys for r in groups[k]]
        retain_val = [r for k in val_keys for r in groups[k]]

        train_records = retain_train + forget_records
        val_records = retain_val
    else:
        groups: Dict[str, List[UnlearningRecord]] = {}
        for r in records:
            k = get_group_key(r)
            groups.setdefault(k, []).append(r)

        keys = sorted(groups.keys())
        rng = random.Random(seed)
        rng.shuffle(keys)

        n_val = max(1, round(len(keys) * val_ratio)) if len(keys) > 1 else 0
        val_keys = set(keys[:n_val])
        train_keys = set(keys[n_val:])

        train_records = [r for k in train_keys for r in groups[k]]
        val_records = [r for k in val_keys for r in groups[k]]

    return train_records, val_records