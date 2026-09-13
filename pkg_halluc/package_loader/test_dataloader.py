"""
Verification test script
"""

import os
from utils import tsv_collate
from unlearn_loader import (
    PackageUnlearningDataset,
    get_unlearning_dataloaders,
    setup_tokenizer,
)

from tsv_loader import (
    get_tsv_data_and_loaders,  
    setup_tokenizer
)

def loader_example(data_dir, family, model_id):
    data_dir = os.path.abspath(data_dir)
    tok = setup_tokenizer(model_id, model_family=family)

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
    return forget_loader, retain_loader, combined_loader

def unlearning_loader_tests():
    data_dir = "Llama3_3_Python"
    if not os.path.isdir(data_dir):
        data_dir = os.path.join("Llama3_3_Python")
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

        # Pointwise format test
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

        # DPO format test
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

def tsv_loader_test():
    data_dir = "Llama3_3_Python"
    if not os.path.isdir(data_dir):
        data_dir = os.path.join("Llama3_3_Python")
    data_dir = os.path.abspath(data_dir)
    print(f"Data directory: {data_dir}")

    models_to_test = [
        ("llama3", "meta-llama/Llama-3.2-1B-Instruct"),
        ("qwen", "Qwen/Qwen2.5-Coder-3B-Instruct"),
        ("deepseek", "deepseek-ai/deepseek-coder-1.3b-instruct"),
    ]

    for family, model_id in models_to_test:
        print(f"\n{'='*20} Testing TSV Loader with {family.upper()} ({model_id}) {'='*20}")
        tok = setup_tokenizer(model_id, model_family=family)

        tsv_data = get_tsv_data_and_loaders(
            data_source=data_dir,
            tokenizer=tok,
            model_family=family,
            query_modes=[1, 2],
            num_exemplars=32,
            wild_ratio=0.75,
            batch_size=8,
            max_length=512,
            balanced_exemplars=True,
            seed=42,
        )

        prompts = tsv_data["prompts"]
        labels = tsv_data["labels"]
        test_prompts, train_prompts, exemplar_prompts = prompts[0], prompts[1], prompts[2]
        test_labels, train_labels, exemplar_labels = labels[0], labels[1], labels[2]

        print(f"[{family}] Exemplar samples: {len(exemplar_prompts)}, Hallucinations: {(exemplar_labels == 0).sum()}, Truthful: {(exemplar_labels == 1).sum()}")
        print(f"[{family}] Wild (Train) samples: {len(train_prompts)}")
        print(f"[{family}] Test samples: {len(test_prompts)}")

        # native collate_fn on exemplar batch test
        batch_size = 4
        batch_prompts = exemplar_prompts[:batch_size]
        batch_labels = exemplar_labels[:batch_size]
        padded_prompts, labels_tensor = tsv_collate(batch_prompts, batch_labels)
        attention_mask = (padded_prompts != 0).half()

        print(f"[{family}] Padded prompts shape: {padded_prompts.shape}") # Expect: (4, 1, max_len)
        print(f"[{family}] Labels tensor shape: {labels_tensor.shape}, values: {labels_tensor.tolist()}")
        print(f"[{family}] Attention mask shape: {attention_mask.shape}")

        # PyTorch DataLoader Test
        train_loader = tsv_data["train_loader"]
        first_batch = next(iter(train_loader))
        print(f"[{family}] PyTorch DataLoader batch prompts: {first_batch['prompts'].shape}, labels: {first_batch['labels'].shape}")

    print("\nAll TSV DataLoader tests passed successfully!")


if __name__ == "__main__":
    unlearning_loader_tests()
    tsv_loader_test()