"""
Dataset and DataLoader
"""
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd
from torch.utils.data import DataLoader, Dataset

from pkg_halluc.package_loader.prompt_config import (
    PACKAGE_PREFIX_1,
    PACKAGE_PREFIX_2,
    PACKAGE_SYSTEM_PROMPT_1,
    PACKAGE_SYSTEM_PROMPT_2,
)

from pkg_halluc.package_loader.utils import (
    DataCollatorForUnlearning,
    UnlearningRecord,
    format_packages_as_string,
    load_csv_data,
    parse_package_list,
    setup_tokenizer,
)

logger = logging.getLogger(__name__)

class PackageUnlearningDataset(Dataset):
    """Dataset for PH Unlearning"""

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
        forget_target_type: str = "all_generated",  # 'all_generated' or 'hallucinated_only'
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
        self.mode_prompts = {
            1: {
                "system_prompt": self.package_system_prompt_1,
                "prefix": self.prefix_1,
            },
            2: {
                "system_prompt": self.package_system_prompt_2,
                "prefix": self.prefix_2,
            },
        }
        self.forget_target_type = forget_target_type.lower()
        self.empty_package_fallback = empty_package_fallback
        self.records: List[UnlearningRecord] = []
        self._load_and_process_data(data_source)

    def _load_and_process_data(self, data_source: Union[str, List[str], pd.DataFrame]):
        combined_df = load_csv_data(data_source)
        self._extract_records(combined_df)

    def _extract_records(self, df: pd.DataFrame):
        for idx, row in df.iterrows():
            source_file = row.get("_source_file", f"row_{idx}")

            # Group (mode, content, valid_pkgs, hall_pkgs, test_pkgs)
            # Mode 1: Answer code -> Required packages
            # Mode 2: Problem prompt -> Recommended packages
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
                if mode not in self.query_modes or not content:
                    continue

                mode_info = self.mode_prompts[mode]
                sys_prompt = mode_info["system_prompt"]
                user_msg = f"{mode_info['prefix']} {content}"
                has_hall = len(hall) > 0

                valid_str = format_packages_as_string(valid, self.empty_package_fallback)
                if self.forget_target_type == "hallucinated_only":
                    target_pkgs = hall
                else:
                    target_pkgs = test_pkgs if test_pkgs else (valid + hall)
                forget_str = format_packages_as_string(target_pkgs, self.empty_package_fallback)

                sample_id = f"{source_file}_m{mode}_idx{idx}"

                # Add to forget set if it has hallucinations
                if has_hall and self.split_type in ("forget", "all"):
                    self.records.append(
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
                if (not has_hall) and len(valid) > 0 and self.split_type in ("retain", "all"):
                    self.records.append(
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
                "raw_source": getattr(record, "raw_source", None),
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
        labels = [-100] * prompt_len + full_ids[prompt_len:]

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

    def to_tri_mask_dataset(
        self,
        tokenizer: Optional[Union[str, Any]] = None,
        model_name: str = "auto",
        model_family: str = "auto",
        output_dir: Optional[Union[str, Path]] = None,
        suffix: Optional[str] = None,
        single_objective_forget: bool = True,
        max_length: Optional[int] = None,
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        from pkg_halluc.package_loader.generate_tri_mask import generate_tri_mask_dataset

        return generate_tri_mask_dataset(
            dataset=self,
            tokenizer=tokenizer or self.tokenizer,
            model_name=model_name,
            model_family=model_family if model_family != "auto" else self.model_family,
            output_dir=output_dir,
            suffix=suffix,
            single_objective_forget=single_objective_forget,
            max_length=max_length or self.max_length,
        )


class CombinedUnlearningLoader:

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


def get_full_data(
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
    forget_target_type: str = "all_generated",
) -> Tuple[DataLoader, DataLoader, CombinedUnlearningLoader]:
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
        forget_target_type=forget_target_type,
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

    combined_loader = CombinedUnlearningLoader(
        forget_loader=forget_loader,
        retain_loader=retain_loader,
        cycle_shorter=cycle_shorter,
    )

    return forget_loader, retain_loader, combined_loader


CombinedUnlearningDataLoader = CombinedUnlearningLoader
get_unlearning_dataloaders = get_full_data


def get_forget_set(
    data_source: Union[str, List[str], pd.DataFrame],
    tokenizer: Optional[Union[str, Any]] = None,
    model_family: str = "auto",  # 'auto', 'llama3', 'qwen', 'deepseek'
    batch_size: int = 4,
    query_modes: Union[int, List[int]] = (1, 2),
    max_length: int = 2048,
    return_format: str = "pointwise",  # 'pointwise', 'dpo', 'text'
    shuffle: bool = True,
    num_workers: int = 0,
    package_system_prompt_1: str = PACKAGE_SYSTEM_PROMPT_1,
    prefix_1: str = PACKAGE_PREFIX_1,
    package_system_prompt_2: str = PACKAGE_SYSTEM_PROMPT_2,
    prefix_2: str = PACKAGE_PREFIX_2,
    forget_target_type: str = "all_generated",  # 'all_generated' or 'hallucinated_only'
    empty_package_fallback: str = "None",
) -> DataLoader:
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
        forget_target_type=forget_target_type,
        empty_package_fallback=empty_package_fallback,
    )

    collate_fn = DataCollatorForUnlearning(configured_tokenizer) if configured_tokenizer is not None else None

    forget_loader = DataLoader(
        forget_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn,
        num_workers=num_workers,
    )

    return forget_loader


def get_retain_set(
    data_source: Union[str, List[str], pd.DataFrame],
    tokenizer: Optional[Union[str, Any]] = None,
    model_family: str = "auto",  # 'auto', 'llama3', 'qwen', 'deepseek'
    batch_size: int = 4,
    query_modes: Union[int, List[int]] = (1, 2),
    max_length: int = 2048,
    return_format: str = "pointwise",  # 'pointwise', 'dpo', 'text'
    shuffle: bool = True,
    num_workers: int = 0,
    package_system_prompt_1: str = PACKAGE_SYSTEM_PROMPT_1,
    prefix_1: str = PACKAGE_PREFIX_1,
    package_system_prompt_2: str = PACKAGE_SYSTEM_PROMPT_2,
    prefix_2: str = PACKAGE_PREFIX_2,
    empty_package_fallback: str = "None",
) -> DataLoader:
    if tokenizer is not None:
        configured_tokenizer = setup_tokenizer(tokenizer, model_family=model_family)
    else:
        configured_tokenizer = None

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
        empty_package_fallback=empty_package_fallback,
    )

    collate_fn = DataCollatorForUnlearning(configured_tokenizer) if configured_tokenizer is not None else None

    retain_loader = DataLoader(
        retain_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn,
        num_workers=num_workers,
    )

    return retain_loader


get_forget_dataloader = get_forget_set
get_forget_only_dataloader = get_forget_set
get_retain_dataloader = get_retain_set