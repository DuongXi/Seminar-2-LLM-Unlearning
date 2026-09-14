"""Shared plumbing for the two "weight_finetune" methods, GA and NPO.

Both wrap the *same* upstream entrypoint, ``train.py``, with a different
``--loss_function``. Both train on the fixed tri-mask dataset built once by
`pkg_halluc data build-seed-data` (see data/seed_data.py)
"""
from __future__ import annotations

from pathlib import Path

from .spec import MethodContext
from ..utils.proc import python_run


def checkpoint_path_for(tag: str, ctx: MethodContext) -> Path:
    """Mirrors train.py's own `output_dir` naming exactly:
    ``<au_repo>/Models/<model_name>_<save_string>``."""
    return ctx.paths.au_repo_dir / "Models" / f"{ctx.cfg['model_name']}_{tag}"


def _ensure_base_model_symlinked_for_au(ctx: MethodContext) -> None:
    """train.py resolves its own base-model path as au_repo/Models/<model_name>
    and re-downloads there if missing, unaware that download-model already
    fetched the same model into paths.models_dir. Symlink it in ahead of
    time so train.py finds it and skips its own download."""
    au_model_path = ctx.paths.au_repo_dir / "Models" / ctx.cfg["model_name"]
    if au_model_path.exists() or au_model_path.is_symlink():
        return
    au_model_path.parent.mkdir(parents=True, exist_ok=True)
    au_model_path.symlink_to(ctx.base_model_path, target_is_directory=True)
    print(f"[ga/npo] symlinked {au_model_path} -> {ctx.base_model_path} (reusing the already-downloaded base model)")


def train_full_finetune(tag: str, loss_function: str, ctx: MethodContext) -> None:
    m = ctx.method_cfg
    _ensure_base_model_symlinked_for_au(ctx)
    args = [
        "train.py",
        "--model", ctx.cfg["model_name"],
        "--loss_function", loss_function,
        "--save_string", tag,
        "--lr", str(m["lr"]),
        "--num_train_epochs", str(m["num_train_epochs"]),
        "--seed", str(ctx.cfg["seed"]),
    ]
    if m.get("use_lora", False):
        # Saved as a bare adapter, not merged -- AU's own eval loader
        # (utils.py::load_model_auto) auto-detects and merges it in memory.
        args += ["--use_lora", "--lora_rank", str(m.get("lora_rank", 16))]

    python_run(
        args,
        cwd=ctx.paths.au_repo_dir,
        work_dir_for_disk_log=ctx.paths.work_dir,
    )
