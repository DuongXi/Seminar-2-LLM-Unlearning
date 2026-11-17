import torch
import numpy
from sklearn.metrics import roc_curve, auc

from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_from_disk
from torch.utils.data import DataLoader
from transformers import DataCollatorWithPadding
from tqdm import tqdm
import argparse

from utils.model import load_model

def compute_min_k_ppl_acc(selected_log_probs, mask, k, predicts_mask):
    average_log_probs = []
    for sample_log_probs, sample_mask in zip(selected_log_probs, mask):
        sample_log_probs_nonpad = sample_log_probs[sample_mask]
        k_value = int(k * sample_log_probs_nonpad.size(0))
        if k_value > 0:
            min_k_log_probs = torch.topk(sample_log_probs_nonpad, k_value, largest=False).values
            average_log_probs.append(min_k_log_probs.mean())
    return torch.stack(average_log_probs).cpu().numpy()

# def compute_mia_scores(model:AutoModelForCausalLM, 
#                      tokenizer:AutoTokenizer,
#                      dataset, 
#                      batch_size=1
#                      ):
#     data_collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="pt")
#     dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=data_collator)
#     model.eval()

#     selected_log_probs_list, mask_list = [], []

#     with torch.no_grad():
#         for batch in tqdm(dataloader, desc="MIA", unit="batch"):
#             batch = {k: v.to("cuda") for k, v in batch.items()}
#             outputs = model(**batch)
#             logits = outputs.logits

#             labels = batch["labels"][:, 1:]
#             pad_token_mask = labels != -100
#             log_probs = torch.log_softmax(logits, dim=-1)[:, :-1]

#             input_ids_expanded = labels.unsqueeze(-1)
#             input_ids_expanded[input_ids_expanded == -100] = 0
#             selected_log_probs = log_probs.gather(2, input_ids_expanded).squeeze(-1) * pad_token_mask

#             selected_log_probs_list.append(selected_log_probs)
#             mask_list.append(pad_token_mask)

#     selected_log_probs = torch.cat(selected_log_probs_list, dim=0)
#     mask = torch.cat(mask_list, dim=0)

#     mia_scores = {}
#     for ratio in [0.3, 0.4, 0.5, 0.6, 1]:
#         mia_scores[f"min_{int(ratio * 100)}_value"] = compute_min_k_ppl_acc(selected_log_probs, mask, ratio, None)

#     return mia_scores

def compute_mia_scores(model: AutoModelForCausalLM,
                       tokenizer: AutoTokenizer,
                       dataset,
                       batch_size=1):
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="pt")
    dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=data_collator)
    model.eval()

    vocab_size = model.get_output_embeddings().weight.shape[0]
    print(f"[MIA] model vocab_size = {vocab_size}")
    print(f"[MIA] tokenizer len    = {len(tokenizer)}")

    selected_log_probs_list, mask_list = [], []
    n_clamped = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="MIA", unit="batch"):
            batch = {k: v.to("cuda") for k, v in batch.items()}

            # Take labels out of the batch BEFORE the model forward, so the
            # model doesn't try to compute its internal CrossEntropyLoss.
            labels = batch.pop("labels", None)
            if labels is None:
                labels = batch["input_ids"]

            # Clamp input_ids (embedding lookup)
            if "input_ids" in batch:
                ids = batch["input_ids"]
                if ids.min().item() < 0 or ids.max().item() >= vocab_size:
                    batch["input_ids"] = ids.clamp(min=0, max=vocab_size - 1)

            # Clamp attention_mask if it was ever non-binary (defensive)
            if "attention_mask" in batch:
                batch["attention_mask"] = batch["attention_mask"].clamp(min=0, max=1)

            outputs = model(**batch)
            logits = outputs.logits

            # Build the mask / shift labels for our own log-prob extraction
            labels = labels[:, 1:]                     # next-token shift
            pad_token_mask = labels != -100

            safe_labels = labels.clone()
            out_of_range = (safe_labels != -100) & (
                (safe_labels < 0) | (safe_labels >= vocab_size)
            )
            n_clamped += int(out_of_range.sum().item())
            safe_labels[safe_labels == -100] = 0
            safe_labels = safe_labels.clamp(min=0, max=vocab_size - 1)

            log_probs = torch.log_softmax(logits, dim=-1)[:, :-1]
            input_ids_expanded = safe_labels.unsqueeze(-1)
            selected_log_probs = (
                log_probs.gather(2, input_ids_expanded).squeeze(-1)
                * pad_token_mask
            )

            selected_log_probs_list.append(selected_log_probs)
            mask_list.append(pad_token_mask)

    if n_clamped:
        print(f"[MIA] clamped {n_clamped} out-of-range label tokens")

    selected_log_probs = torch.cat(selected_log_probs_list, dim=0)
    mask = torch.cat(mask_list, dim=0)

    mia_scores = {}
    for ratio in [0.3, 0.4, 0.5, 0.6, 1]:
        mia_scores[f"min_{int(ratio * 100)}_value"] = compute_min_k_ppl_acc(
            selected_log_probs, mask, ratio, None)
    return mia_scores

def compute_mia_scores(model, tokenizer, dataset, batch_size=1):
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="pt")
    dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=data_collator)
    model.eval()

    vocab_size = model.get_output_embeddings().weight.shape[0]
    print(f"[MIA] model vocab_size = {vocab_size}")
    print(f"[MIA] tokenizer len    = {len(tokenizer)}")

    per_sample_log_probs = []
    n_clamped = n_empty = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="MIA", unit="batch"):
            batch = {k: v.to("cuda") for k, v in batch.items()}
            labels = batch.pop("labels", None)
            if labels is None:
                raise ValueError("dataset has no 'labels' column")

            if "input_ids" in batch:
                ids = batch["input_ids"]
                if ids.min().item() < 0 or ids.max().item() >= vocab_size:
                    batch["input_ids"] = ids.clamp(min=0, max=vocab_size - 1)
            if "attention_mask" in batch:
                batch["attention_mask"] = batch["attention_mask"].clamp(min=0, max=1)

            logits = model(**batch).logits

            labels = labels[:, 1:]
            pad_token_mask = labels != -100

            safe_labels = labels.clone()
            oob = (safe_labels != -100) & ((safe_labels < 0) | (safe_labels >= vocab_size))
            n_clamped += int(oob.sum().item())
            safe_labels[safe_labels == -100] = 0
            safe_labels = safe_labels.clamp(min=0, max=vocab_size - 1)

            log_probs = torch.log_softmax(logits, dim=-1)[:, :-1]
            selected = log_probs.gather(2, safe_labels.unsqueeze(-1)).squeeze(-1)  # (B, L-1)

            for b in range(selected.size(0)):
                sp = selected[b][pad_token_mask[b]].detach().cpu()
                if sp.numel() == 0:
                    n_empty += 1
                    continue
                per_sample_log_probs.append(sp)

    print(f"[MIA] kept {len(per_sample_log_probs)} samples, "
          f"skipped {n_empty} empty, clamped {n_clamped} labels")
    if not per_sample_log_probs:
        raise RuntimeError("no scorable tokens; check the conversion script's labels")

    mia_scores = {}
    for ratio in [0.3, 0.4, 0.5, 0.6, 1.0]:
        vals = []
        for sp in per_sample_log_probs:
            kv = min(max(1, int(ratio * sp.numel())), sp.numel())
            vals.append(torch.topk(sp, kv, largest=False).values.mean())
        mia_scores[f"min_{int(ratio * 100)}_value"] = torch.stack(vals).numpy()
    return mia_scores