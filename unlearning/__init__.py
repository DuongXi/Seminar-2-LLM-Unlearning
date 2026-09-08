"""
Package Hallucination Unlearning & TSV Hallucination Detection Package.
Supports:
- Llama 3 / 3.2 / 3.3
- Qwen 2.5 Coder
- DeepSeekCoder
"""

from unlearning.config import (
    CHAT_TEMPLATES,
    MODEL_CONFIGS,
    CODE_GENERATION_SYSTEM_PROMPT,
    PACKAGE_PREFIX_1,
    PACKAGE_PREFIX_2,
    PACKAGE_SYSTEM_PROMPT_1,
    PACKAGE_SYSTEM_PROMPT_2,
)
from unlearning.dataset import (
    CombinedUnlearningDataLoader,
    DataCollatorForUnlearning,
    PackageUnlearningDataset,
    UnlearningRecord,
    get_unlearning_dataloaders,
    infer_model_family,
    parse_package_list,
    setup_tokenizer,
)
from unlearning.tsv_dataloader import (
    TSVBatchCollator,
    TSVPackageDataset,
    get_tsv_data_and_loaders,
    tsv_collate_fn,
)

__all__ = [
    "CODE_GENERATION_SYSTEM_PROMPT",
    "PACKAGE_SYSTEM_PROMPT_1",
    "PACKAGE_PREFIX_1",
    "PACKAGE_SYSTEM_PROMPT_2",
    "PACKAGE_PREFIX_2",
    "CHAT_TEMPLATES",
    "MODEL_CONFIGS",
    "PackageUnlearningDataset",
    "DataCollatorForUnlearning",
    "CombinedUnlearningDataLoader",
    "UnlearningRecord",
    "get_unlearning_dataloaders",
    "parse_package_list",
    "infer_model_family",
    "setup_tokenizer",
    # TSV specific
    "TSVPackageDataset",
    "tsv_collate_fn",
    "TSVBatchCollator",
    "get_tsv_data_and_loaders",
]