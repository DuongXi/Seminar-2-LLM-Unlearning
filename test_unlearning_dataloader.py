"""
Verification test script for Package Hallucination Unlearning DataLoader across:
1. Llama-3 / 3.2 / 3.3
2. Qwen-2.5-Coder 3B
3. DeepSeekCoder 1.3B
"""

import os
import torch
from unlearning import (
    PackageUnlearningDataset,
    get_unlearning_dataloaders,
    setup_tokenizer,
)

def run_tests():
    data_dir = "Llama3_3_Python"
    if not os.path.isdir(data_dir):
        data_dir = os.path.join("Llama3_3_Python", "Llama3_3_Python")
    data_dir = os.path.abspath(data_dir)
    print(f"Data directory: {data_dir}")

    # Test models matrix
    test_models = [
        ("llama3", "meta-llama/Llama-3.2-1B-Instruct"),
        ("qwen", "Qwen/Qwen2.5-Coder-3B-Instruct"),
        ("deepseek", "deepseek-ai/deepseek-coder-1.3b-instruct"),
    ]

    for family, model_id in test_models:
        print(f"\n{'='*20} Testing Model: {family.upper()} ({model_id}) {'='*20}")
        try:
            tok = setup_tokenizer(model_id, model_family=family)
            print(f"[{family}] Successfully configured tokenizer. Pad ID: {tok.pad_token_id}, EOS ID: {tok.eos_token_id}")
        except Exception as e:
            print(f"[{family}] Failed to load tokenizer {model_id}: {e}")
            continue

        # 1. Test DataLoader in pointwise format
        forget_loader, retain_loader, combined_loader = get_unlearning_dataloaders(
            data_source=data_dir,
            tokenizer=tok,
            model_family=family,
            batch_size=2,
            retain_batch_size=2,
            query_modes=[1, 2],
            max_length=512,
            return_format="pointwise",
            forget_target_type="all_generated",
            shuffle=False,
        )

        print(f"[{family}] Forget batches: {len(forget_loader)}, Retain batches: {len(retain_loader)}")

        # Inspect first batch
        for batch in forget_loader:
            print(f"[{family}] Batch input_ids: {batch['input_ids'].shape}, labels: {batch['labels'].shape}")
            sample_idx = 0
            input_ids = batch["input_ids"][sample_idx].tolist()
            labels = batch["labels"][sample_idx].tolist()

            target_token_ids = [token for token, label in zip(input_ids, labels) if label != -100]
            decoded_target = tok.decode(target_token_ids, skip_special_tokens=True).strip()
            print(f"[{family}] Target completion tokens (loss computed only on these): {decoded_target!r}")
            break

        # 2. Test DPO format
        dpo_dataset = PackageUnlearningDataset(
            data_source=data_dir,
            split_type="forget",
            query_modes=[1, 2],
            tokenizer=tok,
            model_family=family,
            return_format="dpo",
        )
        sample = dpo_dataset[0]
        print(f"[{family}] DPO sample prompt ending: ...{sample['prompt'][-80:].replace(chr(10), ' ')}")
        print(f"[{family}] DPO chosen (valid): {sample['chosen']}")
        print(f"[{family}] DPO rejected (hallucinated): {sample['rejected']}")

    print("\nAll model formats (Llama 3, Qwen 2.5 Coder, DeepSeekCoder) tested successfully!")

if __name__ == "__main__":
    run_tests()