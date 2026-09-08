"""
TSV-compatible Dataset and DataLoader for Package Hallucination Detection and Latent Steering.
Based on the ICML 2025 paper 'Steer LLM Latents for Hallucination Detection' (TSV).
"""

import ast
import glob
import logging
import os
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from unlearning.config import (
    PACKAGE_PREFIX_1,
    PACKAGE_PREFIX_2,
    PACKAGE_SYSTEM_PROMPT_1,
    PACKAGE_SYSTEM_PROMPT_2,
)
from unlearning.dataset import parse_package_list, setup_tokenizer

logger = logging.getLogger(__name__)


def tsv_collate_fn(
    prompts: List[torch.Tensor], labels: Union[List[int], np.ndarray, torch.Tensor]
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Exact collate_fn from TSV (tsv-main/train_utils.py).
    Pads prompts to the maximum sequence length in the batch.
    
    Args:
        prompts: List of tensors, each of shape (1, seq_len) or (seq_len,)
        labels: List or array of binary labels (0 = hallucinated, 1 = truthful)
        
    Returns:
        prompts_padded: Tensor of shape (batch_size, 1, max_seq_len)
        labels: Tensor of shape (batch_size,)
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
    TSV-formatted batches including attention_mask.
    """

    def __init__(self, device: Optional[torch.device] = None):
        self.device = device

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        prompts = [item["prompt"] for item in batch]
        labels = [item["label"] for item in batch]

        prompts_padded, labels_tensor = tsv_collate_fn(prompts, labels)
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


class TSVPackageDataset(Dataset):
    """
    Dataset that extracts Question + Answer prompts and binary labels (0 = Hallucination, 1 = Truthful)
    from PackageHallucination benchmark results, formatted for TSV.
    """

    def __init__(
        self,
        data_source: Union[str, List[str], pd.DataFrame],
        tokenizer: Any,
        model_family: str = "auto",
        query_modes: Union[int, List[int]] = (1, 2),
        max_length: int = 1024,
        device: str = "cpu",
        system_prompt_1: str = PACKAGE_SYSTEM_PROMPT_1,
        prefix_1: str = PACKAGE_PREFIX_1,
        system_prompt_2: str = PACKAGE_SYSTEM_PROMPT_2,
        prefix_2: str = PACKAGE_PREFIX_2,
    ):
        super().__init__()
        self.tokenizer = setup_tokenizer(tokenizer, model_family=model_family)
        self.model_family = model_family
        self.query_modes = [query_modes] if isinstance(query_modes, int) else list(query_modes)
        self.max_length = max_length
        self.device = torch.device(device)
        self.system_prompt_1 = system_prompt_1
        self.prefix_1 = prefix_1.strip()
        self.system_prompt_2 = system_prompt_2
        self.prefix_2 = prefix_2.strip()

        self.samples: List[Dict[str, Any]] = []
        self._load_and_process(data_source)

    def _load_and_process(self, data_source: Union[str, List[str], pd.DataFrame]):
        dfs = []
        if isinstance(data_source, pd.DataFrame):
            dfs.append(data_source)
        elif isinstance(data_source, list):
            for p in data_source:
                if os.path.isfile(p):
                    d = pd.read_csv(p)
                    d["_source"] = os.path.basename(p)
                    dfs.append(d)
        elif isinstance(data_source, str):
            if os.path.isdir(data_source):
                csvs = glob.glob(os.path.join(data_source, "*results*.csv")) or glob.glob(os.path.join(data_source, "*.csv"))
                if not csvs:
                    csvs = glob.glob(os.path.join(data_source, "**", "*results*.csv"), recursive=True)
                for p in csvs:
                    if "FINAL_RESULTS" in p or "PACKAGE_NAMES" in p:
                        continue
                    d = pd.read_csv(p)
                    d["_source"] = os.path.basename(p)
                    dfs.append(d)
            elif os.path.isfile(data_source):
                d = pd.read_csv(data_source)
                d["_source"] = os.path.basename(data_source)
                dfs.append(d)

        if not dfs:
            raise ValueError(f"No valid data files found in: {data_source}")

        combined_df = pd.concat(dfs, ignore_index=True)

        for idx, row in combined_df.iterrows():
            source = row.get("_source", f"row_{idx}")
            code = str(row.get("Answers", "")).strip()
            problem = str(row.get("Prompts", row.get("Questions", ""))).strip()

            valid_1 = parse_package_list(row.get("valid_1", row.get("valid1", [])))
            hallucinated_1 = parse_package_list(row.get("hallucinated_1", row.get("hallucination_1", [])))

            valid_2 = parse_package_list(row.get("valid_2", row.get("valid2", [])))
            hallucinated_2 = parse_package_list(row.get("hallucinated_2", row.get("hallucination_2", [])))

            test_1 = row.get("Test_1", "")
            test_2 = row.get("Test_2", "")

            # Mode 1: Code -> Required packages
            if 1 in self.query_modes and code:
                # In TSV convention: 0 = Hallucination, 1 = Truthful / Valid
                is_hallu = len(hallucinated_1) > 0
                is_valid = (not is_hallu) and len(valid_1) > 0
                if is_hallu or is_valid:
                    label = 0 if is_hallu else 1
                    ans_str = str(test_1) if test_1 else ", ".join(valid_1 + hallucinated_1)
                    user_text = f"{self.prefix_1} {code}"
                    
                    self._add_sample(
                        sample_id=f"{source}_m1_{idx}",
                        system_prompt=self.system_prompt_1,
                        user_prompt=user_text,
                        answer=ans_str,
                        label=label,
                        mode=1,
                    )

            # Mode 2: Problem -> Helpful packages
            if 2 in self.query_modes and problem:
                is_hallu = len(hallucinated_2) > 0
                is_valid = (not is_hallu) and len(valid_2) > 0
                if is_hallu or is_valid:
                    label = 0 if is_hallu else 1
                    ans_str = str(test_2) if test_2 else ", ".join(valid_2 + hallucinated_2)
                    user_text = f"{self.prefix_2} {problem}"

                    self._add_sample(
                        sample_id=f"{source}_m2_{idx}",
                        system_prompt=self.system_prompt_2,
                        user_prompt=user_text,
                        answer=ans_str,
                        label=label,
                        mode=2,
                    )

    def _add_sample(
        self,
        sample_id: str,
        system_prompt: str,
        user_prompt: str,
        answer: str,
        label: int,
        mode: int,
    ):
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": answer},
        ]

        full_text = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=False, tokenize=False
        )
        token_ids = self.tokenizer(
            full_text,
            max_length=self.max_length,
            truncation=True,
            return_tensors="pt",
        ).input_ids  # shape: (1, seq_len)

        if self.device != torch.device("cpu"):
            token_ids = token_ids.to(self.device)

        self.samples.append({
            "sample_id": sample_id,
            "prompt": token_ids,
            "label": label,
            "mode": mode,
        })

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.samples[idx]

    def get_all_prompts_and_labels(self) -> Tuple[List[torch.Tensor], np.ndarray]:
        """Returns lists of prompts and labels directly for TSV manipulation."""
        prompts = [s["prompt"] for s in self.samples]
        labels = np.array([s["label"] for s in self.samples], dtype=np.int32)
        return prompts, labels


def get_tsv_data_and_loaders(
    data_source: Union[str, List[str], pd.DataFrame],
    tokenizer: Any,
    model_family: str = "auto",
    query_modes: Union[int, List[int]] = (1, 2),
    num_exemplars: int = 32,
    wild_ratio: float = 0.75,
    batch_size: int = 32,
    max_length: int = 1024,
    device: str = "cpu",
    seed: int = 42,
    balanced_exemplars: bool = True,
) -> Dict[str, Any]:
    """
    Creates TSV-compatible data splits and loaders.
    
    Splits:
    1. 'exemplar': Few-shot balanced labeled samples (0 = hallucination, 1 = true)
    2. 'train' / 'wild': Unlabeled wild samples for Sinkhorn-Knopp transport
    3. 'test': Held-out test set for AUROC hallucination evaluation
    
    Returns:
        dict containing:
            - 'prompts': [test_prompts, train_prompts, exemplar_prompts] (plug-and-play for tsv_main.py)
            - 'labels': [gt_label_test, gt_label_wild, gt_label_exemplar]
            - 'test_loader': DataLoader for test set
            - 'train_loader': DataLoader for wild training set
            - 'exemplar_loader': DataLoader for exemplar set
    """
    dataset = TSVPackageDataset(
        data_source=data_source,
        tokenizer=tokenizer,
        model_family=model_family,
        query_modes=query_modes,
        max_length=max_length,
        device=device,
    )

    all_prompts, all_labels = dataset.get_all_prompts_and_labels()
    total_len = len(all_labels)

    rng = np.random.RandomState(seed)
    perm_indices = rng.permutation(total_len)

    # 1. Select exemplars
    if balanced_exemplars:
        hallu_indices = np.where(all_labels == 0)[0]
        true_indices = np.where(all_labels == 1)[0]
        rng.shuffle(hallu_indices)
        rng.shuffle(true_indices)

        half_ex = num_exemplars // 2
        exemplar_indices = np.concatenate([hallu_indices[:half_ex], true_indices[:half_ex]])
        rng.shuffle(exemplar_indices)
    else:
        exemplar_indices = perm_indices[:num_exemplars]

    exemplar_set = set(exemplar_indices.tolist())

    # 2. Divide remaining into wild (train) and test
    remaining_indices = [i for i in perm_indices if i not in exemplar_set]
    num_wild = int(wild_ratio * len(remaining_indices))
    wild_indices = np.array(remaining_indices[:num_wild])
    test_indices = np.array(remaining_indices[num_wild:])

    # 3. Construct TSV prompts and labels lists
    test_prompts = [all_prompts[i] for i in test_indices]
    gt_label_test = all_labels[test_indices]

    train_prompts = [all_prompts[i] for i in wild_indices]
    gt_label_wild = all_labels[wild_indices]

    exemplar_prompts = [all_prompts[i] for i in exemplar_indices]
    gt_label_exemplar = all_labels[exemplar_indices]

    prompts_list = [test_prompts, train_prompts, exemplar_prompts]
    labels_list = [gt_label_test, gt_label_wild, gt_label_exemplar]

    # 4. Build standard PyTorch DataLoaders with TSVBatchCollator
    collator = TSVBatchCollator()

    test_subdataset = [dataset[i] for i in test_indices]
    train_subdataset = [dataset[i] for i in wild_indices]
    exemplar_subdataset = [dataset[i] for i in exemplar_indices]

    test_loader = DataLoader(test_subdataset, batch_size=batch_size, shuffle=False, collate_fn=collator)
    train_loader = DataLoader(train_subdataset, batch_size=batch_size, shuffle=True, collate_fn=collator)
    exemplar_loader = DataLoader(exemplar_subdataset, batch_size=min(batch_size, num_exemplars), shuffle=True, collate_fn=collator)

    logger.info(
        f"TSV Data Prepared: Total={total_len}, Exemplars={len(exemplar_prompts)} "
        f"(Hallu: {(gt_label_exemplar == 0).sum()}, True: {(gt_label_exemplar == 1).sum()}), "
        f"Wild={len(train_prompts)}, Test={len(test_prompts)}"
    )

    return {
        "prompts": prompts_list,
        "labels": labels_list,
        "test_loader": test_loader,
        "train_loader": train_loader,
        "exemplar_loader": exemplar_loader,
        "dataset": dataset,
    }

