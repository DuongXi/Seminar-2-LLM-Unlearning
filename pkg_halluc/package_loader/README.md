# Dataset and DataLoader

The `package_loader` package preprocesses benchmark data, creates train/validation/test splits, normalizes records, and generates token-level **Tri-Mask** datasets for unlearning.

## Raw Data and Labels

Unlearning data is built from four benchmark CSV files: `LLM_AT_results.csv`, `LLM_LY_results.csv`, `SO_AT_results.csv`, and `SO_LY_results.csv`. They contain two query modes:

### Mode 1: Code to Required Packages

Labels come from `valid_1` (valid packages) and `hallucinated_1` (hallucinated packages).

### Mode 2: Problem Description to Helpful Packages

Labels come from `valid_2` and `hallucinated_2`.

### Record Types

- `forget` records contain at least one hallucinated package. `chosen` contains valid packages; `rejected` contains the generated response with hallucinations.
- `retain` records contain valid packages and preserve correct model behavior.

Records without the required valid or hallucinated package labels are excluded. Tri-Mask generation also drops records that exceed the configured maximum sequence length.

## Train, Validation, and Test Splits

### Train and Test

`train_test_hallu_split.py` samples 100 unique hallucination-producing prompts per CSV by default, for 400 prompts total. It assigns 90% (360 prompts) to train and 10% (40 prompts) to test, then writes the matching CSV rows to:

- `train_csvs/`, including `LLM_AT_results_train.csv` and `LLM_LY_results_train.csv`.
- `test_csvs/`, including the corresponding test CSVs.
- `train_prompts.jsonl` and `test_prompts.jsonl`.

### Train and Validation

Validation is selected from retain records only. Forget records remain in train so hallucinated behavior stays available to the unlearning objective. Records are grouped by prompt so query modes for the same prompt remain in the same split.

## Master Datasets

`master_train.json`, `master_val.json`, and `master_test.json` contain normalized records built from their corresponding CSV splits.

### Record Schema
```json
{
  "sample_id": "LLM_AT_results_train.csv_m2_idx12",
  "split_type": "forget",
  "mode": 2,
  "system_prompt": "You are a coding assistant that recommends Python packages...",
  "user_prompt": "What Python packages would be useful in solving the following coding problem: Generate Python code that connects to an AWS cloud-based pandas cluster...",
  "completion": "boto3, pandas, aws-cluster-manager, s3fs",
  "chosen": "boto3, pandas, s3fs",
  "rejected": "boto3, pandas, aws-cluster-manager, s3fs",
  "valid_packages": ["boto3", "pandas", "s3fs"],
  "hallucinated_packages": ["aws-cluster-manager"]
}
```

`PackageUnlearningDataset` recognizes preprocessed master files with `is_preprocessed()` and loads them directly instead of reparsing the source CSVs.

## Tri-Mask Encoding

| `tri_mask` | Token type | `labels` | Loss behavior |
| :---: | :--- | :---: | :--- |
| `0` | Prompt, system-template, padding, and separators | `-100` | Ignored by the loss |
| `1` | Valid package tokens and the EOS token | Original token ID | Retain cross-entropy loss |
| `2` | Hallucinated package tokens | Original token ID | Forget GA/NPO loss |

`generate_tri_mask.py` tokenizes the prompt with the model's chat template and assigns prompt tokens a mask value of `0`. Response token offsets are matched against valid and hallucinated package spans: valid package tokens receive `1`, hallucinated package tokens receive `2`, and separators receive `0`. The EOS token is appended with mask value `1`.

The pipeline writes these tokenized JSONL files:

- `npo_retain_tok_*.jsonl`: retain samples with mask values `0` and `1`.
- `npo_forget_tok_*.jsonl`: forget samples with mask values `0`, `1`, and `2`.
- `npo_val_retain_tok_*.jsonl`: validation retain samples for early stopping.

## Directory Structure

```
data/<Model_Name>/
├── FINAL_RESULTS.csv                 # Original code-generation benchmark results
├── LLM_AT_results.csv                # Raw CSV benchmark: LLM All Time
├── LLM_LY_results.csv                # Raw CSV benchmark: LLM Last Year
├── SO_AT_results.csv                 # Raw CSV benchmark: Stack Overflow All Time
├── SO_LY_results.csv                 # Raw CSV benchmark: Stack Overflow Last Year
├── train_test_split/                 # Train/test split outputs
│   ├── split_metadata.json           # Split configuration and counts
│   ├── train_prompts.jsonl           # 360 train prompts by default
│   ├── test_prompts.jsonl            # 40 held-out test prompts by default
│   ├── train_csvs/                   # Filtered train rows from each source CSV
│   ├── test_csvs/                    # Filtered test rows from each source CSV
│   ├── master_train.json             # Normalized train records
│   ├── master_val.json               # Held-out retain validation records
│   └── master_test.json              # Normalized test records
└── tri_mask/                         # Tri-Mask generation outputs
    ├── npo_retain_tok_*.jsonl        # Tokenized retain records
    ├── npo_forget_tok_*.jsonl        # Tokenized forget records
    └── npo_val_retain_tok_*.jsonl    # Tokenized validation retain records
```