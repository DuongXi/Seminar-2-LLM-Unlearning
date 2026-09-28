
## `common/` -- shared utilities

| File | Responsibility |
| --- | --- |
| `model_presets.py` | Maps model preset names to Hugging Face IDs |
| `model_setup.py` | Downloads and initializes models/tokenizers; runs a generation sanity check |
| `download_model.py` | Downloads a base model and optionally runs a sanity check |
| `read_config.py` | Reads `model_config/*.json` and exports shell variables |
| `tri_mask_utils.py` | Prompt prefixes, Tri-Mask loading/collation, LoRA, and `load_model_auto` |

## `package_loader/` -- dataset and Tri-Mask data

| File | Responsibility |
| --- | --- |
| `prompt_config.py` | Package-query prompts and model-family token settings |
| `utils.py` | Tokenizer setup, CSV/JSONL loading, package parsing, and token/character mapping |
| `unlearn_loader.py` | `PackageUnlearningDataset` and forget/retain dataloaders |
| `tsv_loader.py` | TSV datasets and dataloaders |
| `generate_tri_mask.py` | Builds Tri-Mask records (0=ignore, 1=retain, 2=forget) |
| `build_tri_mask_data.py` | Writes retain/forget JSONL datasets for GA/NPO |

## `training/` -- training methods

| File | Responsibility |
| --- | --- |
| `tri_mask/train_tri_mask.py` | GA/NPO Tri-Mask training with optional retain-only validation and early stopping |
| `tri_mask/ga_trainer.py` | `GradientAscentTrainer` |
| `tri_mask/npo_trainer.py` | `NPOTrainer` |
| `plain/train_plain.py` | GA-plain/NPO-plain training |
| `plain/plain_trainers.py` | `GAPlainTrainer` and `NPOPlainTrainer` |

## `evaluation/` -- evaluation and reports

