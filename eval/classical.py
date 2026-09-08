import torch
import math

from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_from_disk
from torch.utils.data import DataLoader
from transformers import DataCollatorWithPadding
from tqdm import tqdm
import argparse

from utils.model import load_model

parser = argparse.ArgumentParser(description="Classical evaluation methods:")
parser.add_argument("--path", type=str, 
                     required=True
                     )
parser.add_argument("--forgetset", type=str, 
                     required=True
                     )
parser.add_argument("--retainset", type=str, 
                     required=False
                     )
args = parser.parse_args()

model_path = args.path
forget_set = args.forgetset
forget_dataset = load_from_disk(forget_set)
### TO BE CHANGED LATER
forget_dataset.set_format("torch", columns=["input_ids", 
                                            "attention_mask", 
                                            "labels"
                                            ])
if args.retainset is not None:
    retain_set = args.retainset
    retain_dataset = load_from_disk(retain_set)
    ### TO BE CHANGED LATER
    retain_dataset.set_format("torch", columns=["input_ids", 
                                            "attention_mask", 
                                            "labels"
                                            ])
print(model_path)
model = load_model(model_path=model_path)

def compute_min_k_ppl_acc(selected_log_probs, 
                          mask, 
                          k, 
                          predicts_mask
                          ):
    average_log_probs = []
    average_accs = []
    for sample_log_probs, sample_mask, sample_predicts_mask in zip(
        selected_log_probs, mask, predicts_mask
    ):
        sample_log_probs_nonpad = sample_log_probs[sample_mask]  
        k_value = int(k * sample_log_probs_nonpad.size(0))  
        if k_value > 0:
            topk_results = torch.topk(sample_log_probs_nonpad, k_value, largest=False)
            min_k_log_probs = topk_results.values
            topk_indices = topk_results.indices
            sample_average_log_prob = min_k_log_probs.mean()
            average_log_probs.append(sample_average_log_prob)

            sample_acc = sample_predicts_mask[topk_indices].float().mean()  
            average_accs.append(sample_acc)

    ppl = torch.exp(-torch.stack(average_log_probs).mean())
    acc = (sum(average_accs) / len(average_accs)) * 100
    return ppl.item(), acc.item()

def compute_accuracy(model:AutoModelForCausalLM, 
                     tokenizer:AutoTokenizer,
                     dataset, 
                     batch_size=1
                     ):
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="pt")
    dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=data_collator)
    model.eval()

    selected_log_probs_list, mask_list, predicts_mask_list = [], [], []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="", unit="batch"):
            batch = {k: v.to("cuda") for k, v in batch.items()}
            outputs = model(**batch)
            logits = outputs.logits

            labels = batch["labels"][:, 1:] 
            pad_token_mask = labels != -100 
            log_probs = torch.log_softmax(logits, dim=-1)[:, :-1] 

            input_ids_expanded = labels.unsqueeze(-1)
            input_ids_expanded[input_ids_expanded == -100] = 0
            selected_log_probs = log_probs.gather(2, input_ids_expanded).squeeze(-1) * pad_token_mask

            pred = logits.argmax(dim=-1)[:, :-1]  
            predicts_mask = pred == labels  

            selected_log_probs_list.append(selected_log_probs)
            mask_list.append(pad_token_mask)
            predicts_mask_list.append(predicts_mask)

    selected_log_probs = torch.cat(selected_log_probs_list, dim=0)
    mask = torch.cat(mask_list, dim=0)
    predicts_mask = torch.cat(predicts_mask_list, dim=0)
    ratio = 1
    ppl, accuracy = compute_min_k_ppl_acc(selected_log_probs, mask, ratio, predicts_mask)

    return ppl, accuracy

def dataset_metrics(model:AutoModelForCausalLM,
                    dataset
                    ):
    total_nll, total_tokens = 0.0, 0
    total_correct = 0

    with torch.no_grad():
        for sample in tqdm(dataset, desc="scanning"):
            # ---- 把 list -> tensor 并放到 GPU ----
            inputs = {
                k: torch.tensor(v).unsqueeze(0).to(model.device)
                for k, v in sample.items()
                if k in ["input_ids", "attention_mask"]
            }

            # ---- 前向得到 logits ----
            logits = model(**inputs).logits[:, :-1]          # [1, L-1, V]
            tgt    = inputs["input_ids"][:, 1:]              # [1, L-1]
            mask   = inputs["attention_mask"][:, 1:] == 1    # True at real tokens

            # ---- NLL / PPL ----
            logp   = torch.log_softmax(logits, -1)
            ll     = logp.gather(2, tgt.unsqueeze(-1)).squeeze(-1)
            nll    = -ll[mask]
            total_nll    += nll.sum().item()
            total_tokens += nll.numel()

            # ---- Accuracy ----
            pred = logits.argmax(dim=-1)                     # [1, L-1]
            total_correct += (pred[mask] == tgt[mask]).sum().item()

    mean_nll = total_nll / total_tokens
    ppl  = math.exp(mean_nll)
    acc  = 100.0 * total_correct / total_tokens            # %

    return mean_nll, ppl, acc

ppl_forget, acc_forget = compute_accuracy(model, forget_dataset)
print(f"Forget dataset PPL: {ppl_forget}")
print(f"Forget dataset Acc: {acc_forget}")

mnll, ppl, acc = dataset_metrics(model, forget_dataset)
print(f"{forget_dataset:6} forget  NLL={mnll:.4f}  PPL={ppl:>6.4f}  ACC={acc:5.4f}%")

if retain_set is not None:
    ppl_retain, acc_retain = compute_accuracy(model, retain_dataset)
    print(f"Retain dataset PPL: {ppl_retain}")
    print(f"Retain dataset Acc: {acc_retain}")