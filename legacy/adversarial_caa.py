import argparse

import numpy
import torch
from datasets import load_from_disk
from sklearn.metrics import roc_curve, auc
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorWithPadding

from utils.model import load_model
from pkg_halluc.CAA.wrapper import ModelWrapper


# Uniform forward helper
def _set_caa_position(caa_wrapper, labels_batch):
    """Point the CAA hook at the last prompt token (answer boundary).

    labels_batch is [B, L] with -100 over the prompt. We steer from the
    token just before the first answer label, matching the convention
    ModelWrapper.generate_text uses (prompt_len - 1).
    """
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
    for sample_log_probs, sample_mask in zip(selected_log_probs, mask):
        sample_log_probs_nonpad = sample_log_probs[sample_mask]
        k_value = int(k * sample_log_probs_nonpad.size(0))
        if k_value > 0:
            min_k_log_probs = torch.topk(
                sample_log_probs_nonpad, k_value, largest=False).values
            average_log_probs.append(min_k_log_probs.mean())
    return torch.stack(average_log_probs).cpu().numpy()


def compute_mia_scores(model: AutoModelForCausalLM,
                       tokenizer: AutoTokenizer,
                       dataset,
                       batch_size=1,
                       caa_wrapper=None):
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer,
                                            return_tensors="pt")
    dataloader = DataLoader(dataset, batch_size=batch_size,
                            collate_fn=data_collator)
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

            # CAA: set steering start at the answer boundary before forward.
            _set_caa_position(caa_wrapper, labels)

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


def compute_auc(forget_scores, approximate_scores):
    labels = numpy.concatenate([numpy.ones_like(forget_scores),
                                numpy.zeros_like(approximate_scores)])
    scores = numpy.concatenate([forget_scores, approximate_scores])
    fpr, tpr, _ = roc_curve(labels, scores)
    auc_score = auc(fpr, tpr)
    return fpr, tpr, auc_score


def main():
    parser = argparse.ArgumentParser(description="MIA evaluation methods")
    parser.add_argument("--path", type=str, default=None,
                        help="HF model id/path (vanilla backend)")
    parser.add_argument("--caa_bundle", type=str, default=None,
                        help="path to a CAA steering bundle directory")
    parser.add_argument("--multiplier", type=float, default=None,
                        help="optional CAA multiplier override")
    parser.add_argument("--forgetset", type=str, required=True)
    parser.add_argument("--approxset", type=str, required=True)
    parser.add_argument("--retainset", type=str, required=False)
    args = parser.parse_args()

    if not args.caa_bundle and not args.path:
        parser.error("provide --path or --caa_bundle")

    forget_set = args.forgetset
    approx_set = args.approxset


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
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        tokenizer, model = load_model(model_path=args.path, device_map=device)
        print("[backend] vanilla")


    forget_dataset = load_from_disk(forget_set)
    forget_dataset.set_format("torch",
                              columns=["input_ids", "attention_mask", "labels"])

    approx_dataset = load_from_disk(approx_set)
    approx_dataset.set_format("torch",
                              columns=["input_ids", "attention_mask", "labels"])

    retain_dataset = None
    if args.retainset is not None:
        retain_set = args.retainset
        retain_dataset = load_from_disk(retain_set)
        retain_dataset.set_format("torch",
                                  columns=["input_ids", "attention_mask", "labels"])


    mia_forget_scores = compute_mia_scores(
        model, tokenizer, forget_dataset, caa_wrapper=caa_wrapper)
    mia_approximate_scores = compute_mia_scores(
        model, tokenizer, approx_dataset, caa_wrapper=caa_wrapper)

    for key in mia_forget_scores.keys():
        auc_result = compute_auc(mia_forget_scores[key],
                                 mia_approximate_scores[key])
        print(f"MIA Attack AUC ({key}): {auc_result[2]:.6f}")


if __name__ == "__main__":
    main()