"""
Dataset and DataLoader
"""
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd
from torch.utils.data import DataLoader, Dataset

from .config import (
    PACKAGE_PREFIX_1,
    PACKAGE_PREFIX_2,
    PACKAGE_SYSTEM_PROMPT_1,
    PACKAGE_SYSTEM_PROMPT_2,
)
from .utils import (
    CombinedUnlearningLoader,
    DataCollatorForUnlearning,
    UnlearningRecord,
    extract_unlearning_records,
    format_record,
    is_preprocessed,
    is_tokenized,
    load_csv_data,
    load_files,
    load_tokenized_records,
    save_files,
    save_tokenized_records,
    setup_tokenizer,
    split_records_by_prompt,
)

# Backwards compatibility aliases
is_preprocessed_records_file = is_preprocessed
load_records_from_file = load_files
save_records_to_file = save_files
CombinedUnlearningDataLoader = CombinedUnlearningLoader

logger = logging.getLogger(__name__)


class PackageUnlearningDataset(Dataset):
    """Dataset for PH Unlearning"""
    def __init__(
        self,
        data_source: Union[str, List[str], pd.DataFrame, List[UnlearningRecord]],
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
        cache_path: Optional[Union[str, Path]] = None,
        save_path: Optional[Union[str, Path]] = None,
        auto_save: bool = False,
    ):
        super().__init__()
        self.split_type = split_type.lower()
        if self.split_type not in ("forget", "retain", "all"):
            raise ValueError(f"split_type must be 'forget', 'retain', or 'all', got '{split_type}'")

        self.query_modes = [query_modes] if isinstance(query_modes, int) else list(query_modes)
        self.tokenizer = setup_tokenizer(tokenizer, model_family=model_family) if tokenizer is not None else None
        self.model_family = model_family
        self.max_length = max_length
        self.return_format = return_format.lower()
        self.package_system_prompt_1 = package_system_prompt_1
        self.prefix_1 = prefix_1.strip()
        self.package_system_prompt_2 = package_system_prompt_2
        self.prefix_2 = prefix_2.strip()
        self.mode_prompts = {
            1: {"system_prompt": self.package_system_prompt_1, "prefix": self.prefix_1},
            2: {"system_prompt": self.package_system_prompt_2, "prefix": self.prefix_2},
        }
        self.forget_target_type = forget_target_type.lower()
        self.empty_package_fallback = empty_package_fallback
        self.cache_path = Path(cache_path) if cache_path else None
        self.save_path = Path(save_path) if save_path else None
        self.auto_save = auto_save
        self.records: List[Union[UnlearningRecord, Dict[str, Any]]] = []
        self._load_and_process_data(data_source)

    def _load_and_process_data(self, data_source: Union[str, List[str], pd.DataFrame, List[UnlearningRecord]]):
        if isinstance(data_source, list) and (len(data_source) == 0 or isinstance(data_source[0], (UnlearningRecord, dict))):
            allowed = ("forget", "retain", "val") if self.split_type == "all" else (self.split_type,)
            self.records = [
                r for r in data_source
                if (self.split_type == "all" or (r.split_type if hasattr(r, "split_type") else r.get("split_type")) in allowed)
                and ((r.mode if hasattr(r, "mode") else r.get("mode")) in self.query_modes)
            ]
            return

        if self.cache_path is not None and self.cache_path.is_file():
            if is_tokenized(self.cache_path):
                self.records = load_tokenized_records(self.cache_path, split_type=self.split_type, query_modes=self.query_modes)
                print(f"Loaded {len(self.records)} tokenized records from cache: {self.cache_path}")
                return
            elif is_preprocessed(self.cache_path):
                self.records = load_files(self.cache_path, split_type=self.split_type, query_modes=self.query_modes)
                print(f"Loaded {len(self.records)} records from cache: {self.cache_path}")
                return

        if isinstance(data_source, (str, Path)):
            ds_path = Path(data_source)
            if ds_path.is_file() and is_tokenized(ds_path):
                self.records = load_tokenized_records(ds_path, split_type=self.split_type, query_modes=self.query_modes)
                print(f"Loaded {len(self.records)} pre-tokenized pointwise records directly from {ds_path}")
                return
            elif ds_path.is_file() and is_preprocessed(ds_path):
                self.records = load_files(ds_path, split_type=self.split_type, query_modes=self.query_modes)
                print(f"Loaded {len(self.records)} preprocessed records directly from {ds_path}")
                if self.cache_path is not None and not self.cache_path.exists():
                    self.save_to_file(self.cache_path)
                return

        combined_df = load_csv_data(data_source)
        self._extract_records(combined_df)
        print(f"Extracted {len(self.records)} records from raw data (split_type={self.split_type}).")

        target_save = self.save_path or self.cache_path
        if target_save is not None or self.auto_save:
            save_dest = target_save or Path("cached_unlearning_records.jsonl")
            self.save_to_file(save_dest)

    def _extract_records(self, df: pd.DataFrame):
        self.records = extract_unlearning_records(
            df=df,
            query_modes=self.query_modes,
            mode_prompts=self.mode_prompts,
            split_type=self.split_type,
            forget_target_type=self.forget_target_type,
            empty_package_fallback=self.empty_package_fallback,
        )

    def save_to_file(self, file_path: Union[str, Path]) -> Path:
        """Save dataset records to a JSONL file"""
        saved_path = save_files(self.records, file_path)
        print(f"Saved {len(self.records)} records to {saved_path}")
        return saved_path

    def save_tokenized(
        self,
        file_path: Union[str, Path],
        tokenizer: Optional[Union[str, Any]] = None,
        max_length: Optional[int] = None,
    ) -> Path:
        """Save dataset in pointwise tokenized format: sample_id, split_type, mode, input_ids, attention_mask, labels"""
        tok = tokenizer or self.tokenizer
        ml = max_length or self.max_length
        return save_tokenized_records(
            records=self.records,
            file_path=file_path,
            tokenizer=tok,
            max_length=ml,
        )

    def to_tokenized_records(
        self,
        tokenizer: Optional[Union[str, Any]] = None,
        max_length: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Export all records in pointwise tokenized format"""
        tok = tokenizer or self.tokenizer
        ml = max_length or self.max_length
        return [
            format_record(r, return_format="pointwise", tokenizer=tok, max_length=ml)
            for r in self.records
        ]

    def load_from_file(self, file_path: Union[str, Path]) -> int:
        """Load records from a preprocessed file into this dataset, replacing current records"""
        p = Path(file_path)
        if is_tokenized(p):
            self.records = load_tokenized_records(p, split_type=self.split_type, query_modes=self.query_modes)
        else:
            self.records = load_files(p, split_type=self.split_type, query_modes=self.query_modes)
        print(f"Loaded {len(self.records)} records from {file_path}")
        return len(self.records)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return format_record(
            self.records[idx],
            return_format=self.return_format,
            tokenizer=self.tokenizer,
            max_length=self.max_length,
        )

    def _clone_with_records(
        self,
        records: List[UnlearningRecord],
        split_type: Optional[str] = None,
    ) -> "PackageUnlearningDataset":
        """Helper to create a new dataset instance sharing config but with specific records"""
        ds = PackageUnlearningDataset.__new__(PackageUnlearningDataset)
        super(PackageUnlearningDataset, ds).__init__()
        ds.split_type = split_type or self.split_type
        ds.query_modes = list(self.query_modes)
        ds.tokenizer = self.tokenizer
        ds.model_family = self.model_family
        ds.max_length = self.max_length
        ds.return_format = self.return_format
        ds.package_system_prompt_1 = self.package_system_prompt_1
        ds.prefix_1 = self.prefix_1
        ds.package_system_prompt_2 = self.package_system_prompt_2
        ds.prefix_2 = self.prefix_2
        ds.mode_prompts = dict(self.mode_prompts)
        ds.forget_target_type = self.forget_target_type
        ds.empty_package_fallback = self.empty_package_fallback
        ds.cache_path = None
        ds.save_path = None
        ds.auto_save = False
        ds.records = list(records)
        return ds

    def split_train_val(
        self,
        val_ratio: float = 0.1,
        seed: int = 42,
        split_retain_only: bool = True,
    ) -> Tuple["PackageUnlearningDataset", "PackageUnlearningDataset"]:
        """Split dataset into train and validation sets grouped by prompt to prevent data leakage."""
        train_records, val_records = split_records_by_prompt(
            self.records,
            val_ratio=val_ratio,
            seed=seed,
            split_retain_only=split_retain_only,
        )
        train_ds = self._clone_with_records(train_records)
        val_ds = self._clone_with_records(val_records, split_type="val" if not split_retain_only else "retain")
        return train_ds, val_ds

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
        """Exports this PackageUnlearningDataset directly to tri-mask records and JSONL files."""
        from .generate_tri_mask import generate_tri_mask_dataset

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

def _create_single_dataloader(
    split_type: str,
    data_source: Union[str, List[str], pd.DataFrame, List[UnlearningRecord]],
    tokenizer: Optional[Union[str, Any]] = None,
    model_family: str = "auto",
    batch_size: int = 4,
    query_modes: Union[int, List[int]] = (1, 2),
    max_length: int = 2048,
    return_format: str = "pointwise",
    shuffle: bool = True,
    num_workers: int = 0,
    **kwargs,
) -> DataLoader:
    tok = setup_tokenizer(tokenizer, model_family=model_family) if tokenizer is not None else None
    dataset = PackageUnlearningDataset(
        data_source=data_source,
        split_type=split_type,
        query_modes=query_modes,
        tokenizer=tok,
        model_family=model_family,
        max_length=max_length,
        return_format=return_format,
        **kwargs,
    )
    collate_fn = DataCollatorForUnlearning(tok) if tok is not None else None
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn,
        num_workers=num_workers,
    )

def get_forget_dataloader(
    data_source: Union[str, List[str], pd.DataFrame, List[UnlearningRecord]],
    tokenizer: Optional[Union[str, Any]] = None,
    model_family: str = "auto",
    batch_size: int = 4,
    query_modes: Union[int, List[int]] = (1, 2),
    max_length: int = 2048,
    return_format: str = "pointwise",
    shuffle: bool = True,
    num_workers: int = 0,
    **kwargs,
) -> DataLoader:
    """Creates DataLoader for the forget set"""
    return _create_single_dataloader(
        "forget",
        data_source=data_source,
        tokenizer=tokenizer,
        model_family=model_family,
        batch_size=batch_size,
        query_modes=query_modes,
        max_length=max_length,
        return_format=return_format,
        shuffle=shuffle,
        num_workers=num_workers,
        **kwargs,
    )

def get_retain_dataloader(
    data_source: Union[str, List[str], pd.DataFrame, List[UnlearningRecord]],
    tokenizer: Optional[Union[str, Any]] = None,
    model_family: str = "auto",
    batch_size: int = 4,
    query_modes: Union[int, List[int]] = (1, 2),
    max_length: int = 2048,
    return_format: str = "pointwise",
    shuffle: bool = True,
    num_workers: int = 0,
    **kwargs,
) -> DataLoader:
    """Creates DataLoader for the retain set"""
    return _create_single_dataloader(
        "retain",
        data_source=data_source,
        tokenizer=tokenizer,
        model_family=model_family,
        batch_size=batch_size,
        query_modes=query_modes,
        max_length=max_length,
        return_format=return_format,
        shuffle=shuffle,
        num_workers=num_workers,
        **kwargs,
    )

def get_unlearning_dataloaders_with_val(
    data_source: Union[str, List[str], pd.DataFrame, List[UnlearningRecord]],
    val_ratio: float = 0.1,
    seed: int = 42,
    tokenizer: Optional[Union[str, Any]] = None,
    model_family: str = "auto",
    batch_size: int = 4,
    retain_batch_size: Optional[int] = None,
    val_batch_size: Optional[int] = None,
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
) -> Tuple[DataLoader, DataLoader, Optional[DataLoader], CombinedUnlearningLoader]:
    """
    Creates forget, retain_train, retain_val, and combined dataloaders
    with zero prompt leakage between train and validation.
    """
    if retain_batch_size is None:
        retain_batch_size = batch_size
    if val_batch_size is None:
        val_batch_size = retain_batch_size

    tok = setup_tokenizer(tokenizer, model_family=model_family) if tokenizer is not None else None

    # Load full dataset
    ds = PackageUnlearningDataset(
        data_source=data_source,
        split_type="all",
        query_modes=query_modes,
        tokenizer=tok,
        model_family=model_family,
        max_length=max_length,
        return_format=return_format,
        package_system_prompt_1=package_system_prompt_1,
        prefix_1=prefix_1,
        package_system_prompt_2=package_system_prompt_2,
        prefix_2=prefix_2,
        forget_target_type=forget_target_type,
    )

    # Prompt-grouped split holding out retain only
    train_ds, val_ds = ds.split_train_val(val_ratio=val_ratio, seed=seed, split_retain_only=True)

    forget_records = [r for r in train_ds.records if r.split_type == "forget"]
    retain_train_records = [r for r in train_ds.records if r.split_type == "retain"]

    forget_dataset = train_ds._clone_with_records(forget_records, split_type="forget")
    retain_train_dataset = train_ds._clone_with_records(retain_train_records, split_type="retain")

    collate_fn = DataCollatorForUnlearning(tok) if tok is not None else None

    forget_loader = DataLoader(
        forget_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn,
        num_workers=num_workers,
    )
    retain_loader = DataLoader(
        retain_train_dataset,
        batch_size=retain_batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn,
        num_workers=num_workers,
    )
    val_loader = (
        DataLoader(
            val_ds,
            batch_size=val_batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=num_workers,
        )
        if len(val_ds) > 0
        else None
    )

    combined_loader = CombinedUnlearningLoader(
        forget_loader=forget_loader,
        retain_loader=retain_loader,
        cycle_shorter=cycle_shorter,
    )

    return forget_loader, retain_loader, val_loader, combined_loader

def get_full_data(*args, **kwargs) -> Tuple[DataLoader, DataLoader, CombinedUnlearningLoader]:
    """Creates forget, retain, and combined dataloaders without validation split"""
    kwargs["val_ratio"] = 0.0
    forget_loader, retain_loader, _, combined_loader = get_unlearning_dataloaders_with_val(*args, **kwargs)
    return forget_loader, retain_loader, combined_loader


# Aliases
get_unlearning_dataloaders = get_full_data
get_forget_set = get_forget_dataloader
get_forget_only_dataloader = get_forget_dataloader
get_retain_set = get_retain_dataloader