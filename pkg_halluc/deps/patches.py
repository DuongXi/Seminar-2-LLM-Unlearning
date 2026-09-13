"""Text-patch the fetched AU source for compatibility.

Each patch checks whether it's already applied before touching the file,
so apply_all_patches is safe to call every run.
"""
from __future__ import annotations

from pathlib import Path


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _write(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def patch_dtype_bf16_to_fp16(paths: list[Path], use_fp16: bool) -> None:
    """AU hard-codes torch.bfloat16 / bf16=True. Older GPUs (T4, P100) don't
    support bf16 compute -- rewrite to float16 when resolve_dtype() decided against bf16."""
    if not use_fp16:
        return
    for path in paths:
        if not path.exists():
            continue
        src = _read(path)
        new_src = src.replace("torch.bfloat16", "torch.float16").replace("bf16=True", "fp16=True")
        if new_src != src:
            _write(path, new_src)
            print(f"[patch] bf16 -> fp16 in {path}")


def patch_torch_dtype_kwarg(paths: list[Path]) -> None:
    """transformers==4.57.6 renamed from_pretrained(torch_dtype=...) to dtype=."""
    for path in paths:
        if not path.exists():
            continue
        src = _read(path)
        new_src = src.replace("torch_dtype=", "dtype=")
        if new_src != src:
            _write(path, new_src)
            print(f"[patch] torch_dtype= -> dtype= in {path}")


def patch_au_utils_dtype_key(au_repo_dir: Path) -> None:
    """Same rename as patch_torch_dtype_kwarg, but for a dict-key assignment in utils.py::load_model_auto."""
    utils_path = au_repo_dir / "utils.py"
    if not utils_path.exists():
        return
    src = _read(utils_path)
    old_line = 'extra_kwargs["torch_dtype"] = torch_dtype'
    new_line = 'extra_kwargs["dtype"] = torch_dtype'
    if old_line in src:
        _write(utils_path, src.replace(old_line, new_line))
        print(f"[patch] torch_dtype -> dtype key in {utils_path}")


def patch_row_indexing(paths: list[Path]) -> None:
    """generate_code.py / generate_package_names.py index a pandas row with
    row[0], which recent pandas versions broke -- needs row.iloc[0]."""
    for path in paths:
        if not path.exists():
            print(f"[patch][WARN] not found: {path}")
            continue
        src = _read(path)
        if "row.iloc[0]" in src:
            continue
        if "row[0]" not in src:
            print(f"[patch][WARN] 'row[0]' not found in {path} -- nothing to patch")
            continue
        _write(path, src.replace("row[0]", "row.iloc[0]"))
        print(f"[patch] row[0] -> row.iloc[0] in {path}")


def patch_disable_intermediate_checkpoints(au_repo_dir: Path) -> None:
    """train.py saves a checkpoint every 25 steps by default -- needed for
    AU's nested-epoch resampling, not for GA/NPO's single static dataset.
    Switch to save_strategy="no" so only the final model gets saved
    (Kaggle disk quota is tight)."""
    train_py = au_repo_dir / "train.py"
    src = _read(train_py)
    old = 'save_strategy="steps",\n        save_steps=25,'
    new = 'save_strategy="no",\n        save_steps=25,'
    if 'save_strategy="no"' in src:
        return
    if old not in src:
        print(f"[patch][WARN] expected save_strategy block not found in {train_py}")
        return
    _write(train_py, src.replace(old, new, 1))
    print(f"[patch] disabled intermediate checkpointing in {train_py}")


def patch_lora_unmerged_save(au_repo_dir: Path) -> None:
    """--use_lora should save the bare PEFT adapter, not a merged model:
    utils.py::load_model_auto already merges the adapter in memory at eval
    load time, and a merged checkpoint (~6GB/run) doesn't fit Kaggle's disk
    quota. Reverts to unmerged if a prior merge-before-save patch is present."""
    train_py = au_repo_dir / "train.py"
    src = _read(train_py)
    merged = (
        '    if args.use_lora:\n'
        '        model = model.merge_and_unload()\n'
        '        model.save_pretrained(output_dir)\n'
        '        tokenizer.save_pretrained(output_dir)\n'
        '        print(f"LoRA adapter merged into base weights and saved to {output_dir}")\n'
    )
    unmerged = (
        '    if args.use_lora:\n'
        '        model.save_pretrained(output_dir)\n'
        '        tokenizer.save_pretrained(output_dir)\n'
        '        print(f"LoRA adapter saved to {output_dir}")\n'
    )
    if merged in src:
        _write(train_py, src.replace(merged, unmerged, 1))
        print(f"[patch] reverted LoRA merge-before-save (adapter-only save restored) in {train_py}")


def apply_all_patches(au_repo_dir: Path, use_fp16: bool) -> None:
    """Run every patch above, in the right order. Safe to call repeatedly."""
    dtype_targets = [au_repo_dir / "train.py"]
    patch_dtype_bf16_to_fp16(dtype_targets, use_fp16=use_fp16)
    patch_torch_dtype_kwarg(dtype_targets)
    patch_au_utils_dtype_key(au_repo_dir)
    patch_row_indexing(
        [
            au_repo_dir / "Package_Hallucination_Testing" / "generate_code.py",
            au_repo_dir / "Package_Hallucination_Testing" / "generate_package_names.py",
        ]
    )
    patch_disable_intermediate_checkpoints(au_repo_dir)
    patch_lora_unmerged_save(au_repo_dir)
