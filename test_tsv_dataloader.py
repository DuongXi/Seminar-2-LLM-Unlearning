"""
Verification test for TSV-compatible DataLoader on PackageHallucination dataset.
"""

import os
import torch
from unlearning import get_tsv_data_and_loaders, tsv_collate_fn, setup_tokenizer

def test_tsv_loader():
    data_dir = "Llama3_3_Python"
    if not os.path.isdir(data_dir):
        data_dir = os.path.join("Llama3_3_Python", "Llama3_3_Python")
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

        # 1. Test TSV native collate_fn on exemplar batch
        batch_size = 4
        batch_prompts = exemplar_prompts[:batch_size]
        batch_labels = exemplar_labels[:batch_size]
        padded_prompts, labels_tensor = tsv_collate_fn(batch_prompts, batch_labels)
        attention_mask = (padded_prompts != 0).half()

        print(f"[{family}] Padded prompts shape: {padded_prompts.shape}") # Expect: (4, 1, max_len)
        print(f"[{family}] Labels tensor shape: {labels_tensor.shape}, values: {labels_tensor.tolist()}")
        print(f"[{family}] Attention mask shape: {attention_mask.shape}")

        # 2. Test PyTorch DataLoader
        train_loader = tsv_data["train_loader"]
        first_batch = next(iter(train_loader))
        print(f"[{family}] PyTorch DataLoader batch prompts: {first_batch['prompts'].shape}, labels: {first_batch['labels'].shape}")

    print("\nAll TSV DataLoader tests passed successfully!")

if __name__ == "__main__":
    test_tsv_loader()

