# pkg-halluc: Package Hallucination Mitigation

| Method | Kind | Source |
| --- | --- | --- |
| **Base** | reference (unmodified base model) | -- |
| **GA** (Gradient Ascent) | full fine-tune, static dataset, per-token mask (tri-mask) | from the *Adaptive Unlearning* (AU) paper's code |
| **NPO** (Negative Preference Optimization) | full fine-tune, static dataset, per-token mask (tri-mask) | from AU's code |
| **GA-plain** | full fine-tune, same static dataset, whole-response loss (no tri-mask) | own code |
| **NPO-plain** | full fine-tune, same static dataset, whole-response loss (no tri-mask) | own code |
| **Representation Steering** | *not implemented yet* | |


## Directory structure

```
├── configs/            # JSON config files
├── data/               # Package Hallucination dataset for Python
├── data_gen/           # scripts that generate the full dataset
├── notebooks/          # Kaggle notebook
├── scripts/            # quickstart.sh -- script to run the full pipeline
├── steering_ref/       # archived TSV code (unmodified, from the paper)
├── pkg_halluc/         # the pipeline code
│   ├── paths.py            # Kaggle-vs-local environment detection (run on Kaggle or local)
│   ├── config.py           # config loading
│   ├── cli.py              # the `pkg_halluc` CLI
│   ├── model_setup.py      # base model download
│   ├── package_loader/     # data loader
│   ├── deps/               # fetch + patch + copy the AU paper's code
│   ├── tok_data/           # builds the static training data for GA/NPO
│   ├── scripts/            # train_plain.py -- trains GA-plain/NPO-plain
│   ├── methods/            # per-method definitions (base/ga/npo/ga_plain/npo_plain)
│   ├── eval/               # evaluation for every method
│   ├── report/             # aggregates eval_runs/* into the final tables
│   ├── templates/          # supporting run scripts
│   └── utils/              # subprocess runner shared by every pipeline stage
└── README.md
```

## Prerequisites

- Python >= 3.10
- CUDA GPU

## Installation

```bash
git clone https://github.com/DuongXi/Seminar-2-LLM-Unlearning



pip install -e .          # installs pkg_halluc + the dependencies pinned in pyproject.toml
```

After this, the `pkg_halluc` command is available.

## Fetching the AU paper's code

```bash
# Option A: already have the AU zip locally
pkg_halluc fetch-deps --au-src /path/to/Adaptive-Unlearning-952E.zip

# Option B: on Kaggle, with the repo attached as a Dataset via "Add Input" --
# no --au-src needed, found automatically
pkg_halluc fetch-deps
```

## Configuration

| File | What it's for |
| --- | --- |
| `configs/default.json` | Qwen2.5-Coder-1.5B, realistic eval size (150 prompts) |
| `configs/smoke_test.json` | small (25 eval prompts, every method's retain/forget data capped to 40 rows/split) -- for checking the environment (and code changes) work before a real run |
| `configs/qwen2.5-coder-1.5b.json`, `configs/qwen2.5-coder-3b.json`, `configs/qwen2.5-1.5b.json` | Qwen 2.5 Coder presets (1.5B / 3B) |
| `configs/llama3.2-1b.json`, `configs/llama3.2-3b.json` | Llama 3.2 presets (1B / 3B) |
| `configs/deepseek-coder-1.3b.json` | DeepSeek Coder 1.3B preset |

The fields below are shown at their raw `DEFAULT_CONFIG` defaults (every
preset above overrides `use_lora: true` and a lower `eval.batch_size` --
4 to 8 depending on model size -- these are just the fallback values a
config doesn't have to restate):

```jsonc
{
  "model_name": "qwen2.5-0.5b",      
  "seed": 42,
  "dtype": "auto",                    // "auto" | "bfloat16" | "float16" -- auto picks bf16 if the GPU supports it
  "data": {
    "max_train_samples_per_split": null // set a small number for a fast smoke test
  },
  "eval": {
    "n_eval_prompts": 150,
    "batch_size": 16,                 // try lower batch_size when hit OOM
    "package_modes": [1, 2]          
  },
  "methods": {
    "base": { "enabled": true },
    "ga":   { "enabled": true, "lr": 1e-5, "num_train_epochs": 3, "use_lora": false, "lora_rank": 16 },
    "npo":  { "enabled": true, "lr": 1e-5, "num_train_epochs": 3, "use_lora": false, "lora_rank": 16 },
    "ga_plain":  { "enabled": true, "lr": 1e-5, "num_train_epochs": 3, "lambda_retain": 1.0, "lambda_forget": 0.5, "use_lora": false, "lora_rank": 16 },
    "npo_plain": { "enabled": true, "lr": 1e-5, "num_train_epochs": 3, "lambda_retain": 1.0, "lambda_forget": 0.5, "use_lora": false, "lora_rank": 16 },
    "steering":    { "enabled": false } 
  },
  "deps": {
    "adaptive_unlearning": { "source": null }   // set this, or use --au-src to fetch the AU code
  }
}
```

`use_lora: true` trains a LoRA adapter (via PEFT) instead of full
fine-tuning: only the small adapter needs gradients/optimizer state, the
base model stays frozen. Useful when a bigger model doesn't fit
full fine-tuning in VRAM

## Usage

Every stage is its own subcommand so a run can be stopped and resumed
partway through.

```
pkg_halluc fetch-deps            # fetch + patch the AU code
pkg_halluc download-model        # download the base model, quick-check it can generate
pkg_halluc build-data            # build the prompt + response data to train GA/NPO
pkg_halluc train --method all    # train every method
pkg_halluc evaluate --method all # evaluate hallucination rate (swap all for ga/npo/ga_plain/npo_plain to evaluate just one)
pkg_halluc report                # aggregate eval_runs/* into the final tables + save CSVs

pkg_halluc run-all               # run every step above in order
```

Every subcommand accepts `--config`, `--work-dir`, `--model`, `--seed`; see
`pkg_halluc <subcommand> --help` for more

### Local

```bash
pkg_halluc fetch-deps --config configs/smoke_test.json --au-src /path/to/Adaptive-Unlearning-952E.zip
pkg_halluc download-model --config configs/smoke_test.json
pkg_halluc build-data --config configs/smoke_test.json
pkg_halluc train --config configs/smoke_test.json --method all
pkg_halluc evaluate --config configs/smoke_test.json --method all
pkg_halluc report --config configs/smoke_test.json
```