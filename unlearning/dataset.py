"""
Dataset and DataLoader implementations for Package Hallucination Unlearning.
Supports:
- Llama 3 / 3.2 / 3.3
- Qwen 2.5 Coder (e.g. 3B, 7B)
- DeepSeekCoder (e.g. 1.3B, 6.7B)
"""

import ast
import glob
import logging
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

from unlearning.config import (
    CHAT_TEMPLATES,
    MODEL_CONFIGS,
    PACKAGE_PREFIX_1,
    PACKAGE_PREFIX_2,
    PACKAGE_SYSTEM_PROMPT_1,
    PACKAGE_SYSTEM_PROMPT_2,
)

logger = logging.getLogger(__name__)


def infer_model_family(identifier: str) -> str:
    """Infer model family ('llama3', 'qwen', 'deepseek') from model name or path."""
    name = str(identifier).lower()
    if "qwen" in name:
        return "qwen"
    elif "deepseek" in name:
        return "deepseek"
    elif "llama" in name:
        return "llama3"
    return "llama3"


def setup_tokenizer(
    tokenizer_or_name: Union[str, Any],
    model_family: str = "auto",
    pad_token: Optional[str] = None,
) -> Any:
    """
    Loads or configures a tokenizer.
    Ensures pad_token and chat_template are properly set.
    """
    if isinstance(tokenizer_or_name, str):
        tok = AutoTokenizer.from_pretrained(tokenizer_or_name, trust_remote_code=True)
        inferred_family = infer_model_family(tokenizer_or_name)
    else:
        tok = tokenizer_or_name
        inferred_family = infer_model_family(getattr(tok, "name_or_path", ""))

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
    """
    Parsing package list from string representation or list object.
    Safely parse package list from string representation or list object.
    Handles '[]', \"['pkg1', 'pkg2']\", list, NaN, etc.
    """
    if pd.isna(value) or value is None:
        return []
    if isinstance(value, (list, set, tuple)):
        return [str(x).strip() for x in value if str(x).strip()]
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


@dataclass
class UnlearningRecord:
    """Represents a single unlearning instance (forget or retain)."""
    sample_id: str
    split_type: str  # 'forget' or 'retain'
    mode: int        # 1 (code->packages) or 2 (problem->packages)
    system_prompt: str
    user_prompt: str
    completion: str  # Target response for SFT/GA/NPO
    chosen: str      # Preferred response for DPO
    rejected: str    # Unwanted response for DPO
    valid_packages: List[str]
    hallucinated_packages: List[str]
    raw_source: Optional[str] = None


class PackageUnlearningDataset(Dataset):
    """
    PyTorch Dataset for Package Hallucination Unlearning.
    
    Compatible with:
    - Llama3.2
    - Qwen 2.5 Coder (e.g. 3B)
    - DeepSeekCoder (e.g. 1.3B)
    
    Supports:
    - Splitting into 'forget' (contains hallucinated packages) and 'retain' (valid packages only).
    - Query 1 (Code -> required packages) and Query 2 (Problem -> helpful packages).
    - Multiple training paradigms:
        * 'pointwise': (input_ids, attention_mask, labels) with prompt masked as -100
        * 'dpo': (prompt, chosen, rejected)
        * 'text': raw structured text dict
    """

    def __init__(
        self,
        data_source: Union[str, List[str], pd.DataFrame],
        split_type: str = "forget",  # 'forget', 'retain', or 'all'
        query_modes: Union[int, List[int]] = (1, 2),
        tokenizer: Optional[Union[str, Any]] = None,
        model_family: str = "auto",  # 'auto', 'llama3', 'qwen', 'deepseek'
        max_length: int = 2048,
        return_format: str = "pointwise",  # 'pointwise', 'dpo', 'text'
        package_system_prompt_1: str = PACKAGE_SYSTEM_PROMPT_1,
        prefix_1: str = PACKAGE_PREFIX_1,
        package_system_prompt_2: str = PACKAGE_SYSTEM_PROMPT_2,
        prefix_2: str = PACKAGE_PREFIX_2,        
        empty_package_fallback: str = "None",
    ):
        super().__init__()
        self.split_type = split_type.lower()
        if self.split_type not in ("forget", "retain", "all"):
            raise ValueError(f"split_type must be 'forget', 'retain', or 'all', got '{split_type}'")

        if isinstance(query_modes, int):
            self.query_modes = [query_modes]
        else:
            self.query_modes = list(query_modes)

        # Configure tokenizer
        if tokenizer is not None:
            self.tokenizer = setup_tokenizer(tokenizer, model_family=model_family)
        else:
            self.tokenizer = None

        self.model_family = model_family
        self.max_length = max_length
        self.return_format = return_format.lower()
        self.package_system_prompt_1 = package_system_prompt_1
        self.prefix_1 = prefix_1.strip()
        self.package_system_prompt_2 = package_system_prompt_2
        self.prefix_2 = prefix_2.strip()
        self.empty_package_fallback = empty_package_fallback

        self.records: List[UnlearningRecord] = []
        self._load_and_process_data(data_source)

    def _load_and_process_data(self, data_source: Union[str, List[str], pd.DataFrame]):
        dfs = []
        if isinstance(data_source, pd.DataFrame):
            dfs.append(data_source)
        elif isinstance(data_source, list):
            for path in data_source:
                if os.path.isfile(path):
                    df = pd.read_csv(path)
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
                    df = pd.read_csv(path)
                    df["_source_file"] = os.path.basename(path)
                    dfs.append(df)
            elif os.path.isfile(data_source):
                df = pd.read_csv(data_source)
                df["_source_file"] = os.path.basename(data_source)
                dfs.append(df)
            else:
                raise FileNotFoundError(f"Data source path not found: {data_source}")

        if not dfs:
            raise ValueError(f"No valid data loaded from source: {data_source}")

        combined_df = pd.concat(dfs, ignore_index=True)
        self._extract_records(combined_df)

    def _extract_records(self, df: pd.DataFrame):
        for idx, row in df.iterrows():
            source_file = row.get("_source_file", f"row_{idx}")

            code = str(row.get("Answers", "")).strip()
            problem = str(row.get("Prompts", row.get("Questions", ""))).strip()

            valid_1 = parse_package_list(row.get("valid_1", row.get("valid1", [])))
            hallucinated_1 = parse_package_list(row.get("hallucinated_1", row.get("hallucination_1", [])))

            valid_2 = parse_package_list(row.get("valid_2", row.get("valid2", [])))
            hallucinated_2 = parse_package_list(row.get("hallucinated_2", row.get("hallucination_2", [])))

            # Test 1 and Test 2 full generated lists (if available)
            test_1_pkgs = parse_package_list(row.get("Test_1", []))
            test_2_pkgs = parse_package_list(row.get("Test_2", []))
            
            # Process Mode 1
            if 1 in self.query_modes and code:
                user_msg_1 = f"{self.prefix_1} {code}"
                has_h1 = len(hallucinated_1) > 0

                valid_str_1 = format_packages_as_string(valid_1, self.empty_package_fallback)
                all_pkgs = test_1_pkgs if test_1_pkgs else (valid_1 + hallucinated_1)
                forget_str_1 = format_packages_as_string(all_pkgs, self.empty_package_fallback)

                if has_h1 and self.split_type in ("forget", "all"):
                    self.records.append(
                        UnlearningRecord(
                            sample_id=f"{source_file}_m1_idx{idx}",
                            split_type="forget",
                            mode=1,
                            system_prompt=self.package_system_prompt_1,
                            user_prompt=user_msg_1,
                            completion=forget_str_1,
                            chosen=valid_str_1,
                            rejected=forget_str_1,
                            valid_packages=valid_1,
                            hallucinated_packages=hallucinated_1,
                            raw_source=source_file,
                        )
                    )

                # Add to retain set if it had NO hallucinations and had valid packages
                if (not has_h1) and len(valid_1) > 0 and self.split_type in ("retain", "all"):
                    self.records.append(
                        UnlearningRecord(
                            sample_id=f"{source_file}_m1_idx{idx}",
                            split_type="retain",
                            mode=1,
                            system_prompt=self.package_system_prompt_1,
                            user_prompt=user_msg_1,
                            completion=valid_str_1,
                            chosen=valid_str_1,
                            rejected="",
                            valid_packages=valid_1,
                            hallucinated_packages=[],
                            raw_source=source_file,
                        )
                    )

            # Process Mode 2
            if 2 in self.query_modes and problem:
                user_msg_2 = f"{self.prefix_2} {problem}"
                has_h2 = len(hallucinated_2) > 0

                valid_str_2 = format_packages_as_string(valid_2, self.empty_package_fallback)
                all_pkgs = test_2_pkgs if test_2_pkgs else (valid_2 + hallucinated_2)
                forget_str_2 = format_packages_as_string(all_pkgs, self.empty_package_fallback)

                if has_h2 and self.split_type in ("forget", "all"):
                    self.records.append(
                        UnlearningRecord(
                            sample_id=f"{source_file}_m2_idx{idx}",
                            split_type="forget",
                            mode=2,
                            system_prompt=self.package_system_prompt_2,
                            user_prompt=user_msg_2,
                            completion=forget_str_2,
                            chosen=valid_str_2,
                            rejected=forget_str_2,
                            valid_packages=valid_2,
                            hallucinated_packages=hallucinated_2,
                            raw_source=source_file,
                        )
                    )

                if (not has_h2) and len(valid_2) > 0 and self.split_type in ("retain", "all"):
                    self.records.append(
                        UnlearningRecord(
                            sample_id=f"{source_file}_m2_idx{idx}",
                            split_type="retain",
                            mode=2,
                            system_prompt=self.package_system_prompt_2,
                            user_prompt=user_msg_2,
                            completion=valid_str_2,
                            chosen=valid_str_2,
                            rejected="",
                            valid_packages=valid_2,
                            hallucinated_packages=[],
                            raw_source=source_file,
                        )
                    )

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        record = self.records[idx]

        if self.return_format == "text" or self.tokenizer is None:
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
                "raw_source": record.raw_source,
            }

        if self.return_format == "dpo":
            prompt_msgs = [
                {"role": "system", "content": record.system_prompt},
                {"role": "user", "content": record.user_prompt},
            ]
            prompt_text = self.tokenizer.apply_chat_template(
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

        # Default: 'pointwise' training format (input_ids, attention_mask, labels)
        messages = [
            {"role": "system", "content": record.system_prompt},
            {"role": "user", "content": record.user_prompt},
            {"role": "assistant", "content": record.completion},
        ]
        prompt_msgs = messages[:2]

        prompt_text = self.tokenizer.apply_chat_template(
            prompt_msgs, add_generation_prompt=True, tokenize=False
        )
        full_text = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=False, tokenize=False
        )

        prompt_ids = self.tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
        full_ids = self.tokenizer(full_text, add_special_tokens=False)["input_ids"]

        prompt_len = len(prompt_ids)
        # Labels: mask prompt tokens with -100 so loss is computed ONLY on completion
        labels = [-100] * prompt_len + full_ids[prompt_len:]

        # Truncate if exceeding max_length
        if len(full_ids) > self.max_length:
            full_ids = full_ids[: self.max_length]
            labels = labels[: self.max_length]

        attention_mask = [1] * len(full_ids)

        return {
            "sample_id": record.sample_id,
            "split_type": record.split_type,
            "mode": record.mode,
            "input_ids": full_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


class DataCollatorForUnlearning:
    """
    Collator that dynamically pads batches for Causal LM training.
    Pads input_ids with pad_token_id and labels with -100.
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


class CombinedUnlearningDataLoader:
    """
    An iterator/DataLoader that yields paired batches:
    {'forget': forget_batch, 'retain': retain_batch}
    at each training step.
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
            # Fetch forget batch
            try:
                batch_f = next(iter_f)
            except StopIteration:
                if self.cycle_shorter:
                    iter_f = iter(self.forget_loader)
                    batch_f = next(iter_f)
                else:
                    break

            # Fetch retain batch
            try:
                batch_r = next(iter_r)
            except StopIteration:
                if self.cycle_shorter:
                    iter_r = iter(self.retain_loader)
                    batch_r = next(iter_r)
                else:
                    break

            yield {"forget": batch_f, "retain": batch_r}


def get_unlearning_dataloaders(
    data_source: Union[str, List[str], pd.DataFrame],
    tokenizer: Optional[Union[str, Any]] = None,
    model_family: str = "auto",  # 'auto', 'llama3', 'qwen', 'deepseek'
    batch_size: int = 4,
    retain_batch_size: Optional[int] = None,
    query_modes: Union[int, List[int]] = (1, 2),
    max_length: int = 2048,
    return_format: str = "pointwise",
    shuffle: bool = True,
    num_workers: int = 0,
    cycle_shorter: bool = True,
    package_system_prompt_1: str = PACKAGE_SYSTEM_PROMPT_1,
    prefix_1: str = PACKAGE_PREFIX_1,
    package_system_prompt_2: str = PACKAGE_SYSTEM_PROMPT_2,
    prefix_2: str = PACKAGE_PREFIX_2,
) -> Tuple[DataLoader, DataLoader, CombinedUnlearningDataLoader]:
    """
    Convenience factory to create Forget, Retain, and Combined DataLoaders.
    Compatible with Llama 3.2, Qwen 2.5 Coder, and DeepSeekCoder.

    Returns:
        (forget_loader, retain_loader, combined_loader)
    """
    if retain_batch_size is None:
        retain_batch_size = batch_size

    if tokenizer is not None:
        configured_tokenizer = setup_tokenizer(tokenizer, model_family=model_family)
    else:
        configured_tokenizer = None

    forget_dataset = PackageUnlearningDataset(
        data_source=data_source,
        split_type="forget",
        query_modes=query_modes,
        tokenizer=configured_tokenizer,
        model_family=model_family,
        max_length=max_length,
        return_format=return_format,
        package_system_prompt_1=package_system_prompt_1,
        prefix_1=prefix_1,
        package_system_prompt_2=package_system_prompt_2,
        prefix_2=prefix_2,
    )

    retain_dataset = PackageUnlearningDataset(
        data_source=data_source,
        split_type="retain",
        query_modes=query_modes,
        tokenizer=configured_tokenizer,
        model_family=model_family,
        max_length=max_length,
        return_format=return_format,
        package_system_prompt_1=package_system_prompt_1,
        prefix_1=prefix_1,
        package_system_prompt_2=package_system_prompt_2,
        prefix_2=prefix_2,
    )

    collate_fn = DataCollatorForUnlearning(configured_tokenizer) if configured_tokenizer is not None else None

    forget_loader = DataLoader(
        forget_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn,
        num_workers=num_workers,
    )

    retain_loader = DataLoader(
        retain_dataset,
        batch_size=retain_batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn,
        num_workers=num_workers,
    )

    combined_loader = CombinedUnlearningDataLoader(
        forget_loader=forget_loader,
        retain_loader=retain_loader,
        cycle_shorter=cycle_shorter,
    )

    return forget_loader, retain_loader, combined_loader