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
from utils.data import _set_caa_position
try:
    from pkg_halluc.CAA.wrapper import ModelWrapper
except ImportError:
    print(
        'PKG_HALLUC + CAA wrapper failed to import!'
    )

def compute_mia_scores(model: AutoModelForCausalLM,
                       tokenizer: AutoTokenizer,
                       dataset,
                       batch_size=1,
                       caa_wrapper=None):
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

            # CAA check
            if caa_wrapper is not None: _set_caa_position(caa_wrapper, labels)

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

def compute_auc(forget_scores, approximate_scores):
    labels = numpy.concatenate([numpy.ones_like(forget_scores), numpy.zeros_like(approximate_scores)])
    scores = numpy.concatenate([forget_scores, approximate_scores])
    fpr, tpr, _ = roc_curve(labels, scores)
    auc_score = auc(fpr, tpr)
    return fpr, tpr, auc_score

def main():
    parser = argparse.ArgumentParser(description="Classical evaluation methods:")
    parser.add_argument("--path", type=str, required=False)
    parser.add_argument("--caa_bundle", type=str, default=None,
                        help="path to a CAA steering bundle directory", required=False)
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
        model_path = args.path
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        tokenizer, model = load_model(model_path=model_path, device_map=device)
        print("[backend] vanilla")

    forget_dataset = load_from_disk(forget_set)
    forget_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
    approx_dataset = load_from_disk(approx_set)
    approx_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
    if args.retainset is not None:
        retain_set = args.retainset
        retain_dataset = load_from_disk(retain_set)
        retain_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

    mia_forget_scores = compute_mia_scores(
        model, tokenizer, forget_dataset, caa_wrapper=caa_wrapper)
    mia_approximate_scores = compute_mia_scores(
        model, tokenizer, approx_dataset, caa_wrapper=caa_wrapper)

    for key in mia_forget_scores.keys():
        auc_result = compute_auc(mia_forget_scores[key], mia_approximate_scores[key])
        print(f"MIA Attack AUC ({key}): {auc_result[2]:.6f}")

if __name__ == "__main__":
    main()