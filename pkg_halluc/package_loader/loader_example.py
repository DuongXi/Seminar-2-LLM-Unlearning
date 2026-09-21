"""Minimal example for loading the pre-tokenized plain datasets."""

from pathlib import Path
from types import SimpleNamespace

from torch.utils.data import DataLoader

from pkg_halluc.package_loader.unlearn_loader import PackageUnlearningDataset
from pkg_halluc.package_loader.utils import DataCollatorForUnlearning


DATA_DIR = Path("data/Llama3_3B/plain")
TRAIN_FILE = DATA_DIR / "plain_train_tok_llama_3B.jsonl"
EVAL_FILE = DATA_DIR / "plain_val_tok_llama_3B.jsonl"

train_dataset = PackageUnlearningDataset(
    data_source=TRAIN_FILE,
    split_type="all",
    query_modes=[1, 2],
    tokenizer=None,
    return_format="pointwise",
)
forget_dataset = PackageUnlearningDataset(
    data_source=TRAIN_FILE,
    split_type="forget",
    query_modes=[1, 2],
    tokenizer=None,
    return_format="pointwise",
)
retain_dataset = PackageUnlearningDataset(
    data_source=TRAIN_FILE,
    split_type="retain",
    query_modes=[1, 2],
    tokenizer=None,
    return_format="pointwise",
)
eval_dataset = PackageUnlearningDataset(
    data_source=EVAL_FILE,
    split_type="all",
    query_modes=[1, 2],
    tokenizer=None,
    return_format="pointwise",
)

pad_tokenizer = SimpleNamespace(pad_token_id=0, eos_token_id=0)
collator = DataCollatorForUnlearning(pad_tokenizer)

forget_loader = DataLoader(forget_dataset, batch_size=4, shuffle=True, collate_fn=collator)
retain_loader = DataLoader(retain_dataset, batch_size=4, shuffle=True, collate_fn=collator)
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=collator)
eval_loader = DataLoader(eval_dataset, batch_size=4, shuffle=False, collate_fn=collator)
