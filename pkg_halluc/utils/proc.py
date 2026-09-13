"""Small process-execution helpers shared by every pipeline stage.

Training/eval steps shell out to scripts (AU's vendored train.py, the
materialized eval_variant.py template, ...) rather than importing them as
Python modules, so each stage stays a resumable, independently-runnable
step with streamed logs.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Sequence


def print_disk_usage(work_dir: Path, label: str = "") -> None:
    total, used, free = shutil.disk_usage(work_dir)
    gb = 1024**3
    tag = f" [{label}]" if label else ""
    print(f"  [disk{tag}] used: {used/gb:.1f} GB / {total/gb:.1f} GB — free: {free/gb:.1f} GB")


def run(cmd: Sequence[str], cwd: Path | str | None = None, work_dir_for_disk_log: Path | None = None) -> int:
    """Run *cmd*, streaming stdout/stderr live, and raise on non-zero exit.

    Pinned to a single GPU (CUDA_VISIBLE_DEVICES=0) unless the caller's
    environment already sets that variable. None of the scripts this
    project shells out to are set up for real multi-GPU (no accelerate
    launch/DDP) -- with more than one GPU visible, HF Trainer falls back to
    plain nn.DataParallel, which often makes GPU 0 OOM sooner than a clean
    single-GPU run would.
    """
    printable = " ".join(str(c) for c in cmd)
    print(f"$ {printable}   (cwd={cwd or Path.cwd()})")
    env = os.environ.copy()
    env.setdefault("CUDA_VISIBLE_DEVICES", "0")
    # Reduces allocator fragmentation on long generations near a full GPU
    # (per the CUDA OOM message's own suggestion) -- doesn't fix runaway
    # KV-cache growth, just gives the allocator more room to reuse blocks.
    env.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
    proc = subprocess.Popen(
        [str(c) for c in cmd],
        cwd=str(cwd) if cwd else None,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        print(line, end="")
    proc.wait()
    if work_dir_for_disk_log is not None:
        print_disk_usage(work_dir_for_disk_log, "after command above")
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {proc.returncode}: {printable}")
    return proc.returncode


def python_run(args: Sequence[str], cwd: Path | str | None = None, work_dir_for_disk_log: Path | None = None) -> int:
    """Shortcut for ``run([sys.executable, *args], ...)``."""
    return run([sys.executable, *args], cwd=cwd, work_dir_for_disk_log=work_dir_for_disk_log)
