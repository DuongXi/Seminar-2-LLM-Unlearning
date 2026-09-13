"""Shared plumbing for GA-plain and NPO-plain.

Counterpart to _finetune_common.py: that one subprocesses into AU's
vendored train.py. This one runs pkg_halluc/scripts/train_plain.py via
`python -m` -- no AU code, no dependency on fetch-deps having run. Both
read the same results CSVs (paths.main_hallu_result_paths) the tri-mask
build uses, so the two families train on identical source rows.
"""
from __future__ import annotations

from pathlib import Path

from .spec import MethodContext
from ..config import resolve_dtype
from ..utils.proc import python_run


def checkpoint_path_for_plain(tag: str, ctx: MethodContext) -> Path:
    """Under paths.checkpoints_dir, separate from au_repo_dir/Models (AU-
    delegated ga/npo) and models_dir (base model download)."""
    model_tag = ctx.cfg["model_name"].split("/")[-1]
    return ctx.paths.checkpoints_dir / f"{model_tag}_{tag}"


def train_plain_finetune(tag: str, loss_function: str, ctx: MethodContext) -> None:
    m = ctx.method_cfg
    result_paths = ctx.paths.main_hallu_result_paths
    max_length = ctx.cfg["data"].get("max_length", 2048)
    max_samples = ctx.cfg["data"].get("max_train_samples_per_split")
    out_dir = checkpoint_path_for_plain(tag, ctx)

    cmd = [
        "-m", "pkg_halluc.scripts.train_plain",
        "--model_path", str(ctx.base_model_path),
        "--model_name", ctx.cfg["model_name"],
        "--loss_function", loss_function,
        "--save_tag", tag,
        "--lr", str(m["lr"]),
        "--num_train_epochs", str(m["num_train_epochs"]),
        "--lambda_retain", str(m.get("lambda_retain", 1.0)),
        "--lambda_forget", str(m.get("lambda_forget", 0.5)),
        "--seed", str(ctx.cfg["seed"]),
        "--dtype", resolve_dtype(ctx.cfg),
        "--max_length", str(max_length),
        "--result_files", *[str(p) for p in result_paths],
        "--out_dir", str(out_dir),
    ]
    if max_samples is not None:
        cmd += ["--max_samples_per_split", str(max_samples)]
    if m.get("use_lora", False):
        cmd += ["--use_lora", "--lora_rank", str(m.get("lora_rank", 16))]

    python_run(
        cmd,
        cwd=ctx.paths.repo_root,
        work_dir_for_disk_log=ctx.paths.work_dir,
    )
    print(f"Plain checkpoint ready: {out_dir}")
