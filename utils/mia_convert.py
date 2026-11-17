#!/usr/bin/env python3
"""
Output layout:
    <out-dir>/forget/      HF Dataset (save_to_disk)
    <out-dir>/retain/      HF Dataset
    <out-dir>/all/         HF Dataset (forget + retain) — for base-model member set

Usage:
    python convert_master_to_eval.py \
        --master LLM_AT_results_train.jsonl \
        --tokenizer meta-llama/Llama-3.2-1B \
        --out-dir eval_data/train \
        --max-length 1024 --set-pad-token-to-eos
"""
import argparse
import json
from pathlib import Path

from datasets import Dataset, load_from_disk
from transformers import AutoTokenizer


def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def build_prompt(rec, tokenizer):
    system = (rec.get("system_prompt") or "").strip()
    user = (rec.get("user_prompt") or "").strip()
    completion = rec.get("completion")
    if completion is None:
        raise ValueError(
            f"row {rec.get('sample_id')} has no 'completion' field")
    completion = completion.strip()

    if tokenizer.chat_template is not None:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": user})
        prompt_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
    else:
        prompt_text = (system + "\n\n" + user).strip() if system else user
    return prompt_text, completion


def encode_example(rec, tokenizer, max_length, add_eos):
    prompt_text, completion_text = build_prompt(rec, tokenizer)

    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    completion_ids = tokenizer(
        completion_text, add_special_tokens=False)["input_ids"]

    if add_eos and tokenizer.eos_token_id is not None:
        completion_ids = completion_ids + [tokenizer.eos_token_id]

    # Add BOS once if the template didn't already add one.
    if tokenizer.bos_token_id is not None and (
            not prompt_ids or prompt_ids[0] != tokenizer.bos_token_id):
        prompt_ids = [tokenizer.bos_token_id] + prompt_ids

    # Truncate prompt from the left so the completion (what we score) survives.
    total = len(prompt_ids) + len(completion_ids)
    if total > max_length:
        overflow = total - max_length
        head_keep = 1 if tokenizer.bos_token_id is not None else 0
        cut = min(overflow, max(0, len(prompt_ids) - head_keep))
        prompt_ids = prompt_ids[cut:]
        if len(prompt_ids) + len(completion_ids) > max_length:
            keep = max_length - len(prompt_ids)
            completion_ids = completion_ids[:keep]

    input_ids = prompt_ids + completion_ids
    labels = [-100] * len(prompt_ids) + list(completion_ids)
    attention_mask = [1] * len(input_ids)

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "sample_id": rec.get("sample_id", ""),
        "mode": rec.get("mode", -1),
        "split_type": rec.get("split_type", ""),
        "prompt_tokens": len(prompt_ids),
        "completion_tokens": len(completion_ids),
    }


def save_split(rows, path, name):
    if not rows:
        print(f"[WARN] {name} is empty, skipping")
        return
    ds = Dataset.from_list(rows)
    ds.save_to_disk(str(path))
    lens = [len(r["input_ids"]) for r in rows]
    comp = [r["completion_tokens"] for r in rows]
    print(f"[{name}] {len(rows)} rows -> {path}")
    print(f"   total len        min/mean/max = "
          f"{min(lens)}/{sum(lens) / len(lens):.1f}/{max(lens)}")
    print(f"   completion tokens min/mean/max = "
          f"{min(comp)}/{sum(comp) / len(comp):.1f}/{max(comp)}")
    if min(comp) <= 1:
        print(f"   [WARN] some completions are <=1 token; "
              f"min-k MIA at k>=0.5 will skip them")


def sanity_check(path, tag):
    ds = load_from_disk(str(path))
    ex = ds[0]
    n_prompt = sum(1 for x in ex["labels"] if x == -100)
    n_scored = len(ex["labels"]) - n_prompt
    print(f"[check {tag}] input_ids[:8]={ex['input_ids'][:8]}")
    print(f"[check {tag}] labels[:8]={ex['labels'][:8]}")
    print(f"[check {tag}] prompt masked={n_prompt} scored={n_scored}")
    if n_scored == 0:
        raise AssertionError(
            f"{tag}: all labels are -100 — completion was truncated away")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--master", required=True,
                    help="JSONL with split_type in {forget, retain}")
    ap.add_argument("--tokenizer", required=True,
                    help="HF tokenizer name or local path (must match model)")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--max-length", type=int, default=1024)
    ap.add_argument("--no-add-eos", dest="add_eos", action="store_false")
    ap.set_defaults(add_eos=True)
    ap.add_argument("--set-pad-token-to-eos", action="store_true",
                    help="If tokenizer has no pad_token, set it to eos_token")
    args = ap.parse_args()

    records = load_jsonl(args.master)
    for r in records:
        if r.get("split_type") not in ("forget", "retain"):
            raise ValueError(
                f"row {r.get('sample_id')} has split_type="
                f"{r.get('split_type')!r}; expected 'forget' or 'retain'")

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True)
    if args.set_pad_token_to_eos and tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
        print(f"[pad] pad_token set to eos_token ({tokenizer.eos_token!r})")
    if tokenizer.pad_token_id is None:
        print("[WARN] tokenizer has no pad_token. The eval script's "
              "DataCollatorWithPadding will pad labels with pad_token_id "
              "(not -100), which corrupts min-k scores when batch_size>1. "
              "Run the eval script with batch_size=1 or pass "
              "--set-pad-token-to-eos and confirm labels masking.")

    encoded = [encode_example(r, tokenizer, args.max_length, args.add_eos)
               for r in records]

    forget = [e for e in encoded if e["split_type"] == "forget"]
    retain = [e for e in encoded if e["split_type"] == "retain"]
    combined = forget + retain

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    save_split(forget,   out / "forget", "forget")
    save_split(retain,   out / "retain", "retain")
    save_split(combined, out / "all",    "all (forget+retain)")

    for tag in ("forget", "retain"):
        p = out / tag
        if p.exists():
            sanity_check(p, tag)

    print("\nHow to plug these into the eval script:")
    print("  Base-model attack (both sets are members):")
    print(f"    --forgetset {out}/forget --approxset <master_test_dir>")
    print("  Retention check (retain=member, master_test=non-member):")
    print(f"    --forgetset {out}/retain --approxset <master_test_dir>")
    print("  Note: master_test must be converted with this same script's")
    print("  tokenizer so labels/input_ids are aligned.")


if __name__ == "__main__":
    main()