"""Build the static GA/NPO training data (retain/forget tri-mask sets).

Unlike AU (which regenerates training data continuously across outer
epochs), GA and NPO here train on a fixed dataset built once by this step,
from the same results CSVs (paths.main_hallu_result_paths) ga_plain/
npo_plain also read.

The actual generation logic lives in
templates/adaptive_unlearning/build_data.py (also materialized into the
fetched AU repo, though this module invokes it straight from templates/)
because it needs this project's own package_loader modules. This module
just invokes it with the right arguments and CWD.

Output path/suffix must match train.py's own hardcoded dataset-file
selection (see _au_train_suffix below) -- train.py doesn't take a data
path argument for ga/npo, it derives one itself from the model name.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from ..config import resolve_dtype
from ..package_loader.generate_tri_mask import resolve_model_suffix
from ..paths import Paths
from ..utils.proc import python_run


def _resolve_path(path_value: str | Path, repo_root: Path) -> Path:
    path = Path(path_value).expanduser()
    return path if path.is_absolute() else (repo_root / path).resolve()


def build_data_au(cfg: dict[str, Any], paths: Paths, model_path: Path) -> None:
    n_prompts = cfg["data"].get("n_data_construction_prompts", 10)
    max_length = cfg["data"].get("max_length", 2048)
    dtype = resolve_dtype(cfg)

    suffix = resolve_model_suffix(cfg["model_name"])

    out_retain = paths.work_dir / "" / f"npo_retain_tok{suffix}.jsonl"
    out_forget = paths.work_dir / "" / f"npo_forget_tok{suffix}.jsonl"
    data_cfg = cfg["data"]
    master_paths = paths.au_seed_prompts_path

    seed_prompts_path = _resolve_path(
        data_cfg.get("seed_prompts_path") or paths.au_seed_prompts_path,
        paths.repo_root,
    )
    eval_prompts_path = _resolve_path(
        cfg["eval"].get("eval_prompts_path") or paths.au_eval_prompts_path,
        paths.repo_root,
    )

    python_run(
        [
            "build_data.py",
            "--model_path", str(model_path),
            "--n_prompts", str(n_prompts),
            "--dtype", dtype,
            "--max_length", str(max_length),
            "--tri_mask", str(cfg.get("tri-mask", True)),
            "--seed", str(cfg["seed"]),
            "--master_files", *[str(path) for path in master_paths],
            "--seed_prompts", str(seed_prompts_path),
            "--extra_prompts", str(eval_prompts_path),
            "--pypi_csv", str(paths.au_pypi_csv_path),
            "--fp_csv", str(paths.au_false_positive_csv_path),
            "--out_master", str(paths.generated_data_dir / "master.json"),
            "--out_retain", str(out_retain),
            "--out_forget", str(out_forget),
        ],
        cwd=paths.au_repo_dir,
        work_dir_for_disk_log=paths.work_dir,
    )
    
    print(f"Seed data ready: {out_retain}, {out_forget}")


def _au_train_suffix(model_name: str) -> str:
    """Mirrors train.py's own hardcoded, case-sensitive dataset-file
    selection: "7b" in name -> `_7b`, "1.3b" -> no suffix, else -> `_16B`
    (coarser than resolve_model_suffix()'s per-size buckets -- train.py
    only knows these 3). build_data() must write to this exact path/suffix,
    or train.py silently loads whatever's already at <au_repo>/New_Data_Set/
    (possibly AU's own shipped example files)."""
    if "7b" in model_name:
        return "_7b"
    if "1.3b" in model_name:
        return ""
    return "_16B"


def build_data(cfg: dict[str, Any], paths: Paths, model_path: Path) -> None:
    max_length = cfg["data"].get("max_length", 2048)
    max_train_samples = cfg["data"].get("max_train_samples_per_split")
    dtype = resolve_dtype(cfg)

    au_suffix = _au_train_suffix(cfg["model_name"])
    out_retain = paths.au_repo_dir / "New_Data_Set" / f"npo_retain_tok{au_suffix}.jsonl"
    out_forget = paths.au_repo_dir / "New_Data_Set" / f"npo_forget_tok{au_suffix}.jsonl"
    data_cfg = cfg["data"]

    master_paths = paths.main_data_paths
    result_path = paths.main_hallu_result_paths
    prompts_path = paths.main_prompts_path
    origin_prompts_path = paths.main_origin_prompts_path

    template_path = paths.templates_dir / "adaptive_unlearning" / "build_data.py"
    args = [
        str(template_path),
        "--model_path", str(model_path),
        "--dtype", dtype,
        "--max_length", str(max_length),
        "--seed", str(cfg["seed"]),
        "--master_files", *[str(path) for path in master_paths],
        "--result_files", *[str(path) for path in result_path],
        "--seed_prompts", *[str(path) for path in prompts_path],
        "--extra_prompts", *[str(path) for path in origin_prompts_path],
        "--out_master", str(paths.generated_data_dir / "master.json"),
        "--out_retain", str(out_retain),
        "--out_forget", str(out_forget),
    ]
    if max_train_samples is not None:
        # smoke-test knob -- configs/smoke_test.json sets this
        args += ["--max_train_samples_per_split", str(max_train_samples)]

    python_run(
        args,
        cwd=paths.repo_root.parent,
        work_dir_for_disk_log=paths.work_dir,
    )

    print(f"Seed data ready: {out_retain}, {out_forget}")
    print(f"(train.py will load these for model_name={cfg['model_name']!r} -- suffix {au_suffix!r})")
