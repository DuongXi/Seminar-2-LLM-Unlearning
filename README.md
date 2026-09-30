
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

CAA extracts package-level steering vectors from contrastive training records and evaluates pairwise completions. Run commands from the repository root.

### Prepare Data

Set `data.main_path` in the model config. Ensure its `train_test_split/master_train.json` and `master_test.json` exist, then generate CAA data:

```bash
bash scripts/build_data.sh --config model_config/llama3.2-1b.json --contrastive
```

This reads `master_train.json` and `master_test.json`. It splits hallucination-bearing prompts from `master_train.json` into prompt-grouped CAA train/validation sets (90/10 by default) before augmentation. `generate_dataset.json` is used for vector extraction and `generate_dataset_val.json` for model selection; pairwise records from `master_test.json` remain under `<main_path>/contrastive/test/` for held-out evaluation.

### Extract Vectors Only

```bash
bash scripts/CAA.sh --config model_config/llama3.2-1b.json --stage extract
```

Without `--layers`, extraction computes vectors for all model layers. Output defaults to `<main_path>/contrastive/vectors/`; use `--layers 10 12 14` to extract a subset.

### Evaluate Existing Vectors

The evaluator uses `<main_path>/contrastive/generate/generate_dataset_val.json` by default, discovers all available vector layers, and sweeps multipliers `0, 0.5, 1, 1.5, 2`. Specify `--layers` or `--multipliers` to override these defaults:

```bash
bash scripts/CAA.sh --config model_config/llama3.2-1b.json --stage eval \
  --layers 10 12 14 --multipliers 0 0.5 1
```

To evaluate the held-out master test set instead, pass its generated pairwise file:

```bash
bash scripts/CAA.sh --config model_config/llama3.2-1b.json --stage eval \
  --dataset-path data/Llama3_1B/contrastive/test/test_dataset_pairwise.json
```

After a sweep, the best setting (highest pairwise accuracy, then highest mean margin) is saved under `checkpoints/Llama-3.2-1B-Instruct_CAA/best_model_bundle/`. Override the result or bundle location with `--results-dir` or `--best-model-dir`. If using the test file for model selection, reserve a separate untouched set for final reporting.

### Extract and Evaluate

Run both stages in order; with no `--layers`, this extracts and evaluates all available layers:

```bash
bash scripts/CAA.sh --config model_config/llama3.2-1b.json --stage all
```

For a local Hugging Face model directory, pass `--model-path /path/to/model`. `CAA.sh` defaults to local-only loading; pass `--local-files-only false` only when the model should be fetched from Hugging Face.

### Token or Sequence Mode

`--mode token` (default) uses `generate_package_vectors.py` to learn from package-token activations. The default `--position-mode` is `mean`, averaging activations across tokens in each package; use `package_start` or `boundary` to select a single token instead.

`--mode sequence` uses `generate_completion_vectors.py` to learn from response-level activations. Choose `--token-position last_token` or `mean_response`. The two modes write vectors and results to separate directories so their same-named layer vectors do not overwrite each other.

```bash
# Package-token activation at each package's first token
bash scripts/CAA.sh --config model_config/llama3.2-1b.json --mode token --stage all

# Mean activation across each response
bash scripts/CAA.sh --config model_config/llama3.2-1b.json --mode sequence --token-position mean_response --stage all
```

### Use Original or Shuffled Data

Choose a generated data variant with `--variant`; CAA selects the matching train file for extraction and validation file for evaluation:

| Variant | Training input |
| --- | --- |
| Original only | `generate_dataset_original.json` |
| Augmented | `generate_dataset_augmented.json` |
| Shuffled (original + permutations) | `generate_dataset_shuffled.json` |
| Shuffled only | `generate_dataset_shuffled_only.json` |

The default variant is `original`, so the normal command needs no `--variant` flag. The `augmented` and `shuffled` variants currently contain the same records; use `shuffled_only` to exclude original-order examples. Examples:

```bash
bash scripts/CAA.sh --config model_config/llama3.2-1b.json --stage all
bash scripts/CAA.sh --config model_config/llama3.2-1b.json --stage all --variant augmented
bash scripts/CAA.sh --config model_config/llama3.2-1b.json --stage all --variant shuffled_only
```

The same option works with `--stage extract` or `--stage eval`. The default `original` and other non-augmented variants use separate vector folders (`vectors_<variant>`) and result folders under `checkpoints/<model>_CAA/<variant>`; `augmented` keeps the base `vectors/` and `<model>_CAA/` locations. For final held-out evaluation, run eval only and override its dataset:

```bash
bash scripts/CAA.sh --config model_config/llama3.2-1b.json --stage eval \
  --variant original \
  --dataset-path data/Llama3_1B/contrastive/test/test_dataset_pairwise.json
```

`--dataset-path` overrides the input for whichever stages are selected; when using `--stage all`, it is passed to both extraction and evaluation. Use `--variant` for the usual matching train/validation pair.

## Run the Pipeline

The example uses the Llama 3.2 1B config. `build_data.sh` reuses existing master splits, or creates them from the four source CSVs under `data.main_path` when both master files are missing.

```bash
bash scripts/download_model.sh --model meta-llama/Llama-3.2-1B-Instruct   # Download the base model and run a sanity check
bash scripts/build_data.sh --config model_config/llama3.2-1b.json --contrastive # Build unlearning and CAA data

bash scripts/train_ga.sh        --model meta-llama/Llama-3.2-1B-Instruct
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
bash scripts/CAA.sh --config model_config/llama3.2-1b.json --stage all
```