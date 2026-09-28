
# pkg-halluc: Mitigating Package Hallucinations

| Method | Description | Source |
| --- | --- | --- |
| **Base** | Original model | -- |
| **GA** (Gradient Ascent) | Full fine-tuning on static data with token-level Tri-Mask | |
| **NPO** (Negative Preference Optimization) | Full fine-tuning on static data with token-level Tri-Mask | |
| **GA-plain** | Full fine-tuning on the same static data without Tri-Mask | |
| **NPO-plain** | Full fine-tuning on the same static data without Tri-Mask | |

## Repository Structure

```
├── model_config/     # JSON configs passed to scripts/*.sh; see Configuration
├── data/              # PackageHallucination datasets and evaluation data
├── data_gen/          # Scripts for generating the full PackageHallucination dataset
├── notebooks/         # Kaggle notebooks
├── scripts/
│   ├── common.sh                         # Shared shell setup
│   ├── quickstart.sh                     # Run the full pipeline
│   ├── download_model.sh                # Download a base model and run a sanity check
│   ├── build_data.sh                    # Build unlearning and optional CAA datasets
│   ├── train_ga.sh, train_npo.sh         # Train GA/NPO Tri-Mask models
│   ├── train_ga_plain.sh, train_npo_plain.sh # Train GA-plain/NPO-plain models
│   ├── train_tri_mask_common.sh, train_plain_common.sh # Shared training logic
│   ├── eval.sh                           # Evaluate a base or fine-tuned checkpoint
│   └── report.sh                         # Combine evaluation runs into result tables
└── pkg_halluc/         # Python package; see pkg_halluc/README.md for module details
  ├── common/          # Model presets, setup, config, and Tri-Mask utilities
  ├── package_loader/  # Dataset loading, preprocessing, and Tri-Mask generation
  ├── training/        # Training implementations
  ├── evaluation/      # Checkpoint evaluation, generation, detection, and reporting
```

## Requirements

- Python >= 3.10
- GPU CUDA

## Installation

```bash
git clone https://github.com/DuongXi/Seminar-2-LLM-Unlearning
cd Seminar-2-LLM-Unlearning

pip install -r requirements.txt 
```

## Configuration

Each script in `scripts/` accepts command-line options and has defaults documented by `--help`. Pass `--config <file>` to load defaults from a model configuration:

```bash
bash scripts/train_ga.sh --config model_config/default.json
bash scripts/eval.sh --config model_config/default.json --tag ga --model-path ...
```

Values from `--config` are defaults. Explicit command-line options override them, so one-off changes do not require editing the JSON file:

```bash
bash scripts/train_ga.sh --config model_config/default.json --lr 2e-5
```

| File | Use |
| --- | --- |
| `model_config/default.json` | Qwen2.5-Coder-3B |
| `model_config/smoke_test.json` | Quick run (25 evaluation prompts; retain/forget data capped at 40 records per split) |
| `model_config/qwen2.5-coder-1.5b.json`, `model_config/qwen2.5-coder-3b.json` | Qwen 2.5 Coder presets (1.5B / 3B) |
| `model_config/llama3.2-1b.json`, `model_config/llama3.2-3b.json` | Llama 3.2 presets (1B / 3B) |
| `model_config/deepseek-coder-1.3b.json` | DeepSeek Coder 1.3B preset |


## Training
### Script Help

```bash
bash scripts/train_ga.sh --help
bash scripts/train_ga_plain.sh --help
bash scripts/train_npo.sh --help
bash scripts/train_npo_plain.sh --help
```

### Basic Usage

```bash
bash scripts/train_ga.sh --model deepseek-coder-1.3b
```

The script resolves model presets, checks for a local checkpoint under `models/`, and writes trained checkpoints to `checkpoints/`.

`--model` accepts a preset (for example, `qwen-0.5b`, `qwen-1.5b`, `qwen-3b`, `llama3.2-1b`, or `deepseek-coder-1.3b`) or a full Hugging Face ID. Unknown short names are not treated as presets. The default is `meta-llama/Llama-3.2-1B-Instruct`.

### Defaults

Without `--config`, all options use the defaults in the shell script:

| Option | Default | Description |
| --- | --- | --- |
| `--lr` | `1e-5` | Learning rate |
| `--epochs` | `3` | Training epochs |
| `--seed` | `42` | Random seed |
| `--dtype` | `auto` | Weight precision (bfloat16 when supported) |
| `--use-lora` | off | Full fine-tuning is used by default |
| Early stopping | on | |
| Batch | 1 × 16 | 16 samples per optimizer step (batch size 1, gradient accumulation 16) |

### Use a Model Config

```bash
bash scripts/train_ga.sh --config model_config/deepseek-coder-1.3b.json
bash scripts/train_ga.sh --config model_config/deepseek-coder-1.3b.json --epochs 5   # CLI options override config values
```

The supplied `model_config/*.json` files enable LoRA (`use_lora: true`) to reduce memory and checkpoint size. Without a config, training defaults to full fine-tuning.

### Specify All Options Explicitly

The following command specifies each option explicitly using the defaults:
```bash
bash scripts/train_ga.sh \
  --model deepseek-coder-1.3b \
  --model-path models/deepseek-ai/deepseek-coder-1.3b-instruct \
  --save-tag ga \
  --out-dir checkpoints/deepseek-coder-1.3b-instruct_ga \
  --lr 1e-5 \
  --epochs 3 \
  --seed 42 \
  --dtype auto \
  --val-ratio 0.1 \
  --eval-steps 25 \
  --early-stopping-patience 3 \
  --early-stopping-threshold 0.0
```

Add `--use-lora --lora-rank 16` to train a LoRA adapter, or `--disable-early-stopping` to disable early stopping.

On Kaggle, checkpoints are written under `/kaggle/working/pkg-halluc/checkpoints`. Archive and download them after training if needed.

## Contrastive Activation Addition (CAA)

CAA builds contrastive train and test data from the unlearning split.

### Build Contrastive Data

Set `data.main_path` in the model config, then generate the contrastive datasets:

```bash
bash scripts/build_data.sh --config model_config/default.json --contrastive
```

This reads `train_test_split/master_train.json` and `master_test.json`. It writes training records under `<main_path>/contrastive/generate/` and pairwise test data under `<main_path>/contrastive/test/`. Test data keeps records with hallucinated packages, removes invalid prompts and train/test prompt overlap, and defaults to original (unshuffled) records in `test_dataset_pairwise.json`.

### Extract Vectors

```bash
bash scripts/CAA.sh --config model_config/default.json --stage extract --layers 10 12 14
```

Vectors are written under `<main_path>/contrastive/vectors/` unless `--output-dir` is provided.

### Optional Pairwise Evaluation

The evaluator uses `<main_path>/contrastive/test/test_dataset_pairwise.json` by default. It compares the mean token log-likelihood of `answer_matching_behavior` and `answer_not_matching_behavior`; use `--dataset-path` to supply another JSON dataset with those fields:

```bash
bash scripts/CAA.sh --config model_config/default.json --stage eval \
  --dataset-path path/to/pairwise_evaluation.json \
  --layers 10 12 14 --multipliers 0 0.5 1
```

## Run the Pipeline

The example below uses the matching Llama 3.2 1B config and dataset. Create the split once if `master_train.json` and `master_test.json` are missing:

```bash
bash scripts/download_model.sh --model meta-llama/Llama-3.2-1B-Instruct   # Download the base model and run a sanity check
python -m pkg_halluc.package_loader.train_test_hallu_split --data_dir data/Qwen_3B # Create the 90/10 train/test split
bash scripts/build_data.sh --config model_config/default.json              # Build unlearning data from data.main_path

bash scripts/train_ga.sh        --model **meta-llama/Llama-3.2-1B-Instruct**
bash scripts/train_npo.sh       --model meta-llama/Llama-3.2-1B-Instruct
bash scripts/train_ga_plain.sh  --model meta-llama/Llama-3.2-1B-Instruct
bash scripts/train_npo_plain.sh --model meta-llama/Llama-3.2-1B-Instruct

bash scripts/eval.sh --tag base      --model-path models/meta-llama/Llama-3.2-1B-Instruct
bash scripts/eval.sh --tag ga        --model-path checkpoints/Llama-3.2-1B-Instruct_ga
bash scripts/eval.sh --tag npo       --model-path checkpoints/Llama-3.2-1B-Instruct_npo
bash scripts/eval.sh --tag ga_plain  --model-path checkpoints/Llama-3.2-1B-Instruct_ga_plain
bash scripts/eval.sh --tag npo_plain --model-path checkpoints/Llama-3.2-1B-Instruct_npo_plain
bash scripts/report.sh   # Combine evaluation runs into final tables and CSV files
bash scripts/CAA.sh --config model_config/llama3.2-1b.json --stage all

```

Or run the full pipeline:

```bash
bash scripts/quickstart.sh --config model_config/llama3.2-1b.json
bash scripts/CAA.sh --config model_config/llama3.2-1b.json --stage all \
  --layers 10 12 14 --multipliers 0 0.5 1
```