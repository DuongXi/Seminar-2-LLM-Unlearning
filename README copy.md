# pkg-halluc: Package Hallucination Mitigation

> 🇻🇳 [Bản dịch tiếng Việt / Vietnamese translation: `README.vi.md`](README.vi.md)

Compares mitigation methods for **package hallucination** in code-generating
LLMs -- cases where a model recommends a Python package that doesn't exist,
opening the door to "slopsquatting" supply-chain attacks -- on one shared
evaluation harness:

| Method | Kind | Source |
| --- | --- | --- |
| **Base** | reference (unmodified model) | -- |
| **GA** (Gradient Ascent) | full fine-tune, static dataset, per-token tri-mask | vendored from the *Adaptive Unlearning* (AU) paper's code |
| **NPO** (Negative Preference Optimization) | full fine-tune, static dataset, per-token tri-mask | vendored from AU's code |
| **GA-plain** | full fine-tune, same static dataset, whole-response loss (no tri-mask) | this project's own code, no AU dependency -- ablation of GA |
| **NPO-plain** | full fine-tune, same static dataset, whole-response loss (no tri-mask) | this project's own code, no AU dependency -- ablation of NPO |
| **Representation Steering** | *not implemented yet* | this project's own planned contribution -- see [Adding a new method](#adding-a-new-method) |

`GA-plain`/`NPO-plain` read the exact same rows (`PackageUnlearningDataset`,
built from the same results CSVs as `ga`/`npo`), so the comparison isolates
one variable: AU's per-token package-position tri-mask vs. a plain
whole-response GA/NPO loss. They don't call any AU code at all -- see
`pkg_halluc/methods/plain_trainers.py` and
`pkg_halluc/scripts/train_plain.py`.

For a quick "does the code even run" check on real data without waiting
through a full training run, set `data.max_train_samples_per_split` in your
config (see `configs/smoke_test.json`, which sets it to `40`) -- caps the
retain/forget data to that many rows each for every method (`ga`, `npo`,
`ga_plain`, `npo_plain` alike).

Every method is evaluated on the **same 4 generation contexts**:
`install_cmd` (an explicit `pip install` line in generated code), `nl_query_1`
("which packages does this code need"), `nl_query_2` ("suggest packages for
this task"), and `bare_import` (raw `import` statements, no install command
needed).

GA and NPO here train on a **fixed, static** dataset built once from a fixed
set of seed prompts -- unlike Adaptive Unlearning's own adaptive
prompt-mutation/discovery loop. That's a deliberate scope choice for this
project, not a limitation of the upstream code.

## Table of contents

- [How it's organized](#how-its-organized)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Fetching the third-party code](#fetching-the-third-party-code)
- [Configuration](#configuration)
- [Usage](#usage)
  - [Local](#local)
  - [Kaggle](#kaggle)
- [Understanding the output](#understanding-the-output)
- [Adding a new method](#adding-a-new-method)
- [Known limitations](#known-limitations)
- [License](#license)

## How it's organized

```
configs/            JSON config files (model, hyperparameters, which methods to run)
data/               optional: your own custom prompt sets (empty by default, see data/README.md)
notebooks/          thin Kaggle wrapper notebook -- calls the same CLI as everywhere else
scripts/            quickstart.sh -- convenience wrapper chaining every CLI stage
steering_ref/     archived (unmodified, Apache-2.0) TSV code relevant to
                    the future Representation Steering method -- see
                    steering_ref/NOTES.md, not wired into the pipeline
pkg_halluc/         the actual pipeline (installable Python package)
  paths.py            Kaggle-vs-local environment detection, all runtime paths
  config.py           config loading/merging, model presets, dtype resolution
  cli.py              the `pkg_halluc` command-line entrypoint (staged subcommands)
  model_setup.py      base model download + sanity check
  deps/               fetch + patch + materialize the third-party (AU) code
  tok_data/           builds the static GA/NPO tri-mask training data
  package_loader/     PackageUnlearningDataset, tri-mask generation, tokenizer/collator utils
  methods/            pluggable per-method specs (base/ga/npo/ga_plain/npo_plain/steering)
  scripts/            train_plain.py -- standalone trainer for ga_plain/npo_plain (no AU dependency)
  eval/               runs the shared evaluation script per method
  report/             aggregates eval_runs/* into the final tables
  templates/          this project's own scripts, copied into the fetched AU repo
  utils/              subprocess runner shared by every pipeline stage
```

Everything the pipeline *writes* at runtime -- fetched third-party source,
downloaded model weights, generated training data, evaluation runs, reports
-- lives under one `work_dir` (`/kaggle/working` on Kaggle, `./.workdir`
locally by default) and is git-ignored. Nothing in `pkg_halluc/`, `configs/`, or
`data/` is ever written to by the pipeline itself.

## Prerequisites

- Python >= 3.10
- A CUDA GPU (everything here has been validated on a Kaggle T4; anything
  with a few GB of free VRAM works for the 0.5B/1.3B model presets)
- `torch` matching your CUDA version, installed **separately** -- see
  [Installation](#installation). Kaggle and Colab both come with a working
  one preinstalled.
- Internet access for the one-time setup steps (`fetch-deps`,
  `download-model`) unless you're supplying everything from local
  files/Kaggle Datasets.

## Installation

```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>

# torch: skip this on Kaggle/Colab (already installed); locally, pick the
# build matching your CUDA version from https://pytorch.org/get-started/locally/
pip install torch --index-url https://download.pytorch.org/whl/cu121   # example only

pip install -e .          # installs pkg_halluc + pinned deps from pyproject.toml
```

This gives you the `pkg_halluc` command (and `python -m pkg_halluc`, if you
prefer).

## Fetching the third-party code

This repo does **not** vendor the AU source tree in git -- see
[`THIRD_PARTY_NOTICE.md`](THIRD_PARTY_NOTICE.md) for the full reasoning
(short version: AU's code is currently published anonymously for
double-blind review, so its license/redistribution terms aren't resolved
yet). Instead, `pkg_halluc fetch-deps` fetches it into
`work_dir/third_party/` (git-ignored) and applies a small set of
compatibility patches (see `pkg_halluc/deps/patches.py` -- every patch's
docstring explains exactly what it does and why: a GPU-dtype fallback, two
`transformers`/`pandas` API renames, and disabling AU's
intermediate-checkpoint saving for the baselines here that don't need it).

Resolution order, first match wins:

1. **Explicit source** -- `--au-src` on the CLI, or
   `deps.adaptive_unlearning.source` in your config JSON. Can be a local
   `.zip`, a local already-extracted directory, or an `http(s)://` URL to a
   `.zip`.
2. **A Kaggle Dataset** already attached via "Add Input" (matched by looking
   for the repo's signature files, e.g. `train.py` + `au_trainer.py`).
   This is what lets you keep doing exactly what may already have worked for
   you before: attach the paper's code as a private Kaggle Dataset, no
   network fetch needed.

There's no default URL for AU, on purpose (see above) -- you must supply it
yourself the first time, either way.

```bash
# Option A: you already have the AU zip locally (e.g. downloaded once from
# the anonymous.4open.science link in the paper)
pkg_halluc fetch-deps --au-src /path/to/Adaptive-Unlearning-952E.zip

# Option B: on Kaggle, with the repo attached as a Dataset via "Add Input" --
# no --au-src needed, it's found automatically
pkg_halluc fetch-deps
```

## Configuration

Config files are plain JSON under `configs/`; a file only needs to specify
what it overrides -- everything else falls back to the defaults in
`pkg_halluc/config.py::DEFAULT_CONFIG`.

| File | What it's for |
| --- | --- |
| `configs/default.json` | Qwen2.5-Coder-1.5B, realistic eval size (150 prompts) |
| `configs/smoke_test.json` | tiny (25 eval prompts, every method's retain/forget data capped to 40 rows/split) -- a few minutes end-to-end, for checking the environment (and code changes) work before a real run |
| `configs/qwen2.5-coder-1.5b.json`, `configs/qwen2.5-coder-3b.json`, `configs/qwen2.5-1.5b.json` | Qwen 2.5 Coder presets (1.5B / 3B) |
| `configs/llama3.2-1b.json`, `configs/llama3.2-3b.json` | Llama 3.2 presets (1B / 3B) |
| `configs/deepseek-coder-1.3b.json` | DeepSeek Coder 1.3B preset |

Key fields, shown here at their raw `DEFAULT_CONFIG` schema defaults (every
shipped preset above overrides `use_lora: true` and a lower `eval.batch_size`
-- 4 to 8 depending on model size -- these are just the fallback values a
config doesn't have to restate):

```jsonc
{
  "model_name": "qwen2.5-0.5b",      // preset name (see MODEL_PRESETS in config.py) or a full HF id
  "seed": 42,
  "dtype": "auto",                    // "auto" | "bfloat16" | "float16" -- auto picks bf16 iff the GPU supports it
  "data": {
    "max_train_samples_per_split": null // set to a small int for a fast smoke test of every method (ga, npo, ga_plain, npo_plain)
  },
  "eval": {
    "n_eval_prompts": 150,
    "batch_size": 16,                 // lower this first if eval hits CUDA OOM -- every shipped preset already does
    "package_modes": [1, 2]           // which nl_query_* modes base/GA/NPO are evaluated on
  },
  "methods": {
    "base": { "enabled": true },
    "ga":   { "enabled": true, "lr": 1e-5, "num_train_epochs": 3, "use_lora": false, "lora_rank": 16 },
    "npo":  { "enabled": true, "lr": 1e-5, "num_train_epochs": 3, "use_lora": false, "lora_rank": 16 },
    "ga_plain":  { "enabled": true, "lr": 1e-5, "num_train_epochs": 3, "lambda_retain": 1.0, "lambda_forget": 0.5, "use_lora": false, "lora_rank": 16 },
    "npo_plain": { "enabled": true, "lr": 1e-5, "num_train_epochs": 3, "lambda_retain": 1.0, "lambda_forget": 0.5, "use_lora": false, "lora_rank": 16 },
    "steering":    { "enabled": false }   // stays false until it's implemented, see below
  },
  "deps": {
    "adaptive_unlearning": { "source": null }   // set this, or use --au-src
  }
}
```

`use_lora: true` trains a LoRA adapter (via PEFT) instead of full fine-tuning --
only the small adapter needs gradients/optimizer memory, base weights stay
frozen. Useful when a bigger model (3B+) doesn't fit full-parameter
fine-tuning on your GPU (e.g. a 16GB Kaggle T4). Saved as a small
adapter-only checkpoint (not merged into the base weights) -- every eval
script already goes through AU's `utils.py::load_model_auto`, which
auto-detects and merges an adapter checkpoint in memory at load time, so
there's no full second copy of the model to store per LoRA run (matters on
Kaggle's ~19.5GB working-directory quota).

Override individual fields ad hoc without writing a whole new file:

```bash
pkg_halluc train --config configs/default.json --model qwen2.5-1.5b --seed 7
```

## Usage

Every stage is its own subcommand so a run can be stopped and resumed --
useful on Kaggle, where sessions have a wall-clock limit. Each is safe to
re-run (already-fetched/downloaded/built artifacts are detected and skipped).

```
pkg_halluc fetch-deps            # fetch + patch AU source
pkg_halluc download-model        # download the base model, sanity-check it generates
pkg_halluc build-data            # build the static GA/NPO tri-mask training data
pkg_halluc train --method all    # train every enabled method (ga, npo, ga_plain, npo_plain, ...)
pkg_halluc evaluate --method all # evaluate every enabled method's hallucination rate
pkg_halluc report                # aggregate eval_runs/* into the final tables + save CSVs

pkg_halluc run-all               # all of the above, in order, one command
```

Every subcommand accepts `--config`, `--work-dir`, `--model`, `--seed`; see
`pkg_halluc <subcommand> --help` for the rest.

### Local

```bash
pkg_halluc fetch-deps --config configs/smoke_test.json --au-src /path/to/Adaptive-Unlearning-952E.zip
pkg_halluc download-model --config configs/smoke_test.json
pkg_halluc build-data --config configs/smoke_test.json
pkg_halluc train --config configs/smoke_test.json --method all
pkg_halluc evaluate --config configs/smoke_test.json --method all
pkg_halluc report --config configs/smoke_test.json
```

or, equivalently: `./scripts/quickstart.sh configs/smoke_test.json` (set the
`AU_SRC=...` env var if you need to pass an explicit source). Swap in
`configs/default.json` once you've confirmed the smoke test works
end-to-end.

### Kaggle

Open `notebooks/kaggle_run.ipynb` in a Kaggle Notebook (GPU + Internet
enabled in Settings). It's a thin wrapper -- installs this project, then
calls the exact same CLI subcommands as above.

**Getting the project code onto Kaggle:**

- **Already pushed to GitHub:** set `REPO_URL` in the notebook's first code
  cell; it clones it.
- **Not on GitHub yet:** zip this project folder (the one containing
  `pyproject.toml`) and upload it as a new **Kaggle Dataset** -- the exact
  same flow you already use for `Adaptive-Unlearning-952E.zip` -- then
  attach it to the notebook via **Add Input**. Leave `REPO_URL = None`; the
  cell searches `/kaggle/input` for it automatically (matches by finding
  `pyproject.toml`) and copies it into `/kaggle/working`, no URL needed.

Either way, if you'd rather not re-fetch AU over the network every session,
attach it as a private Kaggle Dataset too via **Add Input**; `fetch-deps`
finds it automatically (see [Fetching the third-party
code](#fetching-the-third-party-code)) -- so a full Kaggle setup is
typically **2 attached Datasets**: this project and AU.

## Understanding the output

`pkg_halluc report` writes two CSVs to `work_dir/outputs/`:

- **`table1_hallucination_rate.csv`** -- one row per method: overall Package
  Hallucination Rate (%) and absolute reduction vs. Base.
- **`table1_by_context.csv`** -- the same, broken out by the 4 generation
  contexts.

## Adding a new method

The pipeline was structured so this is the *only* thing you should need to
touch. Every method (including the built-in ones) is described by one
`MethodSpec` (`pkg_halluc/methods/spec.py`); the CLI, eval loop, and
report code all talk to the registry (`methods/registry.py`), never to an
individual method module.

**Representation Steering** already has a placeholder,
`pkg_halluc/methods/steering.py` -- read its module docstring for the
planned approach and exactly what to fill in (a hyperparameter block in
`config.py`, the vector-computation function, a generation-time hook, and a
generation adapter with the same I/O shape as `generate_code.py` /
`generate_package_names.py`, wired in via `MethodSpec.extra_eval_args`).
It's registered and wired into every CLI command already; it's disabled by
default (`methods.steering.enabled: false`) purely because there's nothing
to run yet. `steering_ref/` has archived (unmodified, Apache-2.0)
utility code from the TSV paper's repo that's relevant to the
generation-time hook step -- see `steering_ref/NOTES.md` for what's
there and what still needs fixing before it's usable.

For any *other* new method: add `methods/<name>.py` with a `build_spec(enabled)
-> MethodSpec` function, add one line to `_BUILDERS` and `DISPLAY_ORDER` in
`methods/registry.py`, and add its config block to `DEFAULT_CONFIG["methods"]`
in `config.py`.

## Known limitations

- **The evaluation prompt pool isn't filtered for "produces >=1
  hallucination".** The AU paper's own evaluation protocol samples from a
  pool pre-filtered to prompts known to elicit at least one hallucination
  from the base model; this pipeline currently samples uniformly from AU's
  full ~4,900-prompt pool instead. Worth revisiting if your rates come out
  lower than the AU paper's reported numbers for comparable settings.
- **KL-divergence and coding-benchmark (HumanEval/MBPP) metrics are out of
  scope here** -- this pipeline focuses exclusively on Package Hallucination
  Rate. AU's own repo (fetched into `third_party/adaptive-unlearning/`) still
  has `benchmarks.py` and `Distribution_Tests/` if you want to run those
  yourself; they're just not wired into `pkg_halluc`.
- **No real multi-GPU support.** `train.py`/`train_plain.py`/`eval_variant.py`
  all assume a single GPU process (plain HF `Trainer`, no `accelerate
  launch`/DDP setup). If more than one GPU is visible (e.g. Kaggle's "GPU T4
  x2"), `utils/proc.py::run()` pins every subprocess it launches to GPU 0
  only (`CUDA_VISIBLE_DEVICES=0`, unless you've already set that variable
  yourself) -- otherwise HF `Trainer` silently falls back to naive
  `torch.nn.DataParallel`, which replicates the model across GPUs and
  funnels gradient reduction through GPU 0, often making GPU 0 OOM *sooner*
  than a clean single-GPU run would. This means a 2-GPU Kaggle session
  currently only ever uses one of the two GPUs.

## License

This repository's own code is MIT-licensed -- see [`LICENSE`](LICENSE). The
third-party code it fetches at setup time is **not** covered by that license
-- see [`THIRD_PARTY_NOTICE.md`](THIRD_PARTY_NOTICE.md).

## Citing the underlying methods

If you use this, please cite the original paper whose methods it wraps:

- Spracklen, Aghazadeh, Koushanfar, Jadliwala. *LLM Ghostbusters: Surgical
  Hallucination Suppression via Adaptive Unlearning* (GA, NPO baselines used
  here are implemented in their released code).
