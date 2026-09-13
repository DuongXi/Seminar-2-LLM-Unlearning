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
# from eval.classical import compute_min_k_ppl_acc

def compute_min_k_ppl_acc(selected_log_probs, mask, k, predicts_mask):
    average_log_probs = []
    for sample_log_probs, sample_mask in zip(selected_log_probs, mask):
        sample_log_probs_nonpad = sample_log_probs[sample_mask]
        k_value = int(k * sample_log_probs_nonpad.size(0))
        if k_value > 0:
            min_k_log_probs = torch.topk(sample_log_probs_nonpad, k_value, largest=False).values
            average_log_probs.append(min_k_log_probs.mean())
    return torch.stack(average_log_probs).cpu().numpy()

def compute_mia_scores(model:AutoModelForCausalLM, 
                     tokenizer:AutoTokenizer,
                     dataset, 
                     batch_size=1
                     ):
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="pt")
    dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=data_collator)
    model.eval()

    selected_log_probs_list, mask_list = [], []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="MIA", unit="batch"):
            batch = {k: v.to("cuda") for k, v in batch.items()}
            outputs = model(**batch)
            logits = outputs.logits

            labels = batch["labels"][:, 1:]
            pad_token_mask = labels != -100
            log_probs = torch.log_softmax(logits, dim=-1)[:, :-1]

            input_ids_expanded = labels.unsqueeze(-1)
            input_ids_expanded[input_ids_expanded == -100] = 0
            selected_log_probs = log_probs.gather(2, input_ids_expanded).squeeze(-1) * pad_token_mask

            selected_log_probs_list.append(selected_log_probs)
            mask_list.append(pad_token_mask)

    selected_log_probs = torch.cat(selected_log_probs_list, dim=0)
    mask = torch.cat(mask_list, dim=0)

    mia_scores = {}
    for ratio in [0.3, 0.4, 0.5, 0.6, 1]:
        mia_scores[f"min_{int(ratio * 100)}_value"] = compute_min_k_ppl_acc(selected_log_probs, mask, ratio, None)

    return mia_scores

def compute_auc(forget_scores, approximate_scores):
    labels = numpy.concatenate([numpy.ones_like(forget_scores), numpy.zeros_like(approximate_scores)])
    scores = numpy.concatenate([forget_scores, approximate_scores])
    fpr, tpr, _ = roc_curve(labels, scores)
    auc_score = auc(fpr, tpr)
    return fpr, tpr, auc_score

def main():
    parser = argparse.ArgumentParser(description="Classical evaluation methods:")
    parser.add_argument("--path", type=str, required=True)
    parser.add_argument("--forgetset", type=str, required=True)
    parser.add_argument("--approxset", type=str, required=True)
    parser.add_argument("--retainset", type=str, required=False)
    args = parser.parse_args()

    model_path = args.path
    forget_set = args.forgetset
    approx_set = args.approxset
    forget_dataset = load_from_disk(forget_set)
    forget_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
    approx_dataset = load_from_disk(approx_set)
    approx_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
    if args.retainset is not None:
        retain_set = args.retainset
        retain_dataset = load_from_disk(retain_set)
        retain_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
    print(model_path)
    tokenizer, model = load_model(model_path=model_path)

    mia_forget_scores = compute_mia_scores(model, tokenizer, forget_dataset)
    mia_approximate_scores = compute_mia_scores(model,tokenizer, approx_dataset)

    for key in mia_forget_scores.keys():
        auc_result = compute_auc(mia_forget_scores[key], mia_approximate_scores[key])
        print(f"MIA Attack AUC ({key}): {auc_result[2]:.6f}")

if __name__ == "__main__":
    main()