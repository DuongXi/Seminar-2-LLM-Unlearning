import argparse
import math

import torch
from datasets import Dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorWithPadding

from utils.model import load_model
from utils.data import load_split
from pkg_halluc.CAA.wrapper import ModelWrapper


# Uniform forward helper
def _forward(model, inputs):
    return model(**inputs)


def _set_caa_position(caa_wrapper, labels_batch):
    """Point the CAA hook at the last prompt token (answer boundary)."""
    if caa_wrapper is None:
        return
    row = labels_batch[0]
    nz = (row != -100).nonzero()
    if nz.numel() == 0:
        return
    first_answer_pos = int(nz[0].item())
    caa_wrapper.set_from_positions(max(first_answer_pos - 1, 0))


def compute_min_k_ppl_acc(selected_log_probs, mask, k, predicts_mask):
    average_log_probs = []
    average_accs = []
    for sample_log_probs, sample_mask, sample_predicts_mask in zip(
        selected_log_probs, mask, predicts_mask
    ):
        sample_log_probs_nonpad = sample_log_probs[sample_mask]
        sample_predicts_mask_nonpad = sample_predicts_mask[sample_mask]
        k_value = int(k * sample_log_probs_nonpad.size(0))
        if k_value > 0:
            topk_results = torch.topk(sample_log_probs_nonpad,
                                      k_value, largest=False)
            min_k_log_probs = topk_results.values
            topk_indices = topk_results.indices
            average_log_probs.append(min_k_log_probs.mean())
            sample_acc = sample_predicts_mask_nonpad[topk_indices].float().mean()
            average_accs.append(sample_acc)

    ppl = torch.exp(-torch.stack(average_log_probs).mean())
    acc = (sum(average_accs) / len(average_accs)) * 100
    return ppl.item(), acc.item()


def compute_accuracy(model, tokenizer: AutoTokenizer, dataset: Dataset,
                     batch_size=1, caa_wrapper=None):
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer,
                                            return_tensors="pt")
    dataloader = DataLoader(dataset, batch_size=batch_size,
                            collate_fn=data_collator)
    model.eval()

    selected_log_probs_list, mask_list, predicts_mask_list = [], [], []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="", unit="batch"):
            batch = {k: v.to("cuda") for k, v in batch.items()}

            if "labels" in batch:
                _set_caa_position(caa_wrapper, batch["labels"])

            outputs = _forward(model, batch)
            logits = outputs.logits

            labels = batch["labels"][:, 1:]
            pad_token_mask = labels != -100
            log_probs = torch.log_softmax(logits, dim=-1)[:, :-1]

            input_ids_expanded = labels.clone().unsqueeze(-1)
            input_ids_expanded[input_ids_expanded == -100] = 0
            selected_log_probs = (
                log_probs.gather(2, input_ids_expanded).squeeze(-1) * pad_token_mask
            )

            pred = logits.argmax(dim=-1)[:, :-1]
            predicts_mask = pred == labels

            selected_log_probs_list.append(selected_log_probs.squeeze(0))
            mask_list.append(pad_token_mask.squeeze(0))
            predicts_mask_list.append(predicts_mask.squeeze(0))

    ppl, accuracy = compute_min_k_ppl_acc(
        selected_log_probs_list, mask_list, 1, predicts_mask_list
    )
    return ppl, accuracy


def dataset_metrics(model, dataset: Dataset, caa_wrapper=None):
    total_nll, total_tokens = 0.0, 0
    total_correct = 0

    with torch.no_grad():
        for sample in tqdm(dataset, desc="scanning"):
            inputs = {
                k: torch.tensor(v).unsqueeze(0).to(model.device)
                for k, v in sample.items()
                if k in ["input_ids", "attention_mask", "labels"]
            }

            if "labels" in inputs:
                _set_caa_position(caa_wrapper, inputs["labels"])

            forward_inputs = {k: v for k, v in inputs.items() if k != "labels"}
            logits = _forward(model, forward_inputs).logits[:, :-1]
            tgt    = forward_inputs["input_ids"][:, 1:]
            mask   = forward_inputs["attention_mask"][:, 1:] == 1

            logp   = torch.log_softmax(logits, -1)
            ll     = logp.gather(2, tgt.unsqueeze(-1)).squeeze(-1)
            nll    = -ll[mask]
            total_nll    += nll.sum().item()
            total_tokens += nll.numel()

            pred = logits.argmax(dim=-1)
            total_correct += (pred[mask] == tgt[mask]).sum().item()

    mean_nll = total_nll / total_tokens
    ppl = math.exp(mean_nll)
    acc = 100.0 * total_correct / total_tokens
    return mean_nll, ppl, acc


def main():
    parser = argparse.ArgumentParser(description="Classical evaluation methods")
    parser.add_argument("--path", type=str, default=None,
                        help="HF model id/path (vanilla backend)")
    parser.add_argument("--caa_bundle", type=str, default=None,
                        help="path to a CAA steering bundle directory")
    parser.add_argument("--multiplier", type=float, default=None,
                        help="optional CAA multiplier override")
    parser.add_argument("--forgetset", type=str, required=True)
    parser.add_argument("--retainset", type=str, required=False)
    args = parser.parse_args()

    if not args.caa_bundle and not args.path:
        parser.error("provide --path or --caa_bundle")

    forget_set = args.forgetset

    caa_wrapper = None

    if args.caa_bundle:
        caa_wrapper = ModelWrapper.from_steering_bundle(
            bundle_dir=args.caa_bundle,
            device="cuda",
        )
        if args.multiplier is not None:
            for hook in caa_wrapper.hooks.values():
                if hook.add_activations is not None:
                    hook.multiplier = float(args.multiplier)

        model = caa_wrapper.model
        tokenizer = caa_wrapper.tokenizer

        active = [i for i, h in caa_wrapper.hooks.items()
                  if h.add_activations is not None]
        info = ", ".join(
            f"layer={i} mult={caa_wrapper.hooks[i].multiplier}"
            for i in active
        )
        print(f"[backend] caa({info})")
    else:
        print(args.path)
        tokenizer, model = load_model(model_path=args.path)
        print("[backend] vanilla")

    forget_dataset = load_split(forget_set)
    forget_dataset.set_format("torch",
                              columns=["input_ids", "attention_mask", "labels"])

    retain_set = None
    retain_dataset = None
    if args.retainset is not None:
        retain_set = args.retainset
        retain_dataset = load_split(retain_set)
        retain_dataset.set_format("torch",
                                  columns=["input_ids", "attention_mask", "labels"])

    ppl_forget, acc_forget = compute_accuracy(
        model, tokenizer, forget_dataset, caa_wrapper=caa_wrapper)
    print(f"Forget dataset PPL: {ppl_forget}")
    print(f"Forget dataset Acc: {acc_forget}")

    mnll, ppl, acc = dataset_metrics(model, forget_dataset,
                                     caa_wrapper=caa_wrapper)
    print(f"{forget_set:6} forget  NLL={mnll:.4f}  PPL={ppl:>6.4f}  ACC={acc:5.4f}%")

    if retain_dataset is not None:
        ppl_retain, acc_retain = compute_accuracy(
            model, tokenizer, retain_dataset, caa_wrapper=caa_wrapper)
        print(f"Retain dataset PPL: {ppl_retain}")
        print(f"Retain dataset Acc: {acc_retain}")


if __name__ == "__main__":
    main()