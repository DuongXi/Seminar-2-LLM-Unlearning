"""Environment detection and filesystem layout.

Everything the pipeline writes at runtime (fetched third-party repos,
downloaded model weights, generated datasets, evaluation runs, reports)
lives under a single work_dir. Defaults to /kaggle/working on Kaggle,
./.workdir locally; override with --work-dir or PKG_HALLUC_WORK_DIR.

No other dependencies (no torch/transformers) so this can be imported
cheaply from anywhere, including test collection.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def is_kaggle() -> bool:
    """True when running inside a Kaggle Notebook/Script session."""
    return os.path.isdir("/kaggle/working") and (
        "KAGGLE_KERNEL_RUN_TYPE" in os.environ or os.path.isdir("/kaggle/input")
    )


def default_work_dir() -> Path:
    if is_kaggle():
        return Path("/kaggle/working")
    return Path.cwd() / ".workdir"


def default_input_root() -> Path | None:
    """Kaggle "Add Input" mount point, if any. ``None`` when running locally."""
    if os.path.isdir("/kaggle/input"):
        return Path("/kaggle/input")
    return None


@dataclass
class Paths:
    """Resolved filesystem layout for one pipeline run.

    work_dir: root for all runtime artifacts (override with --work-dir /
        PKG_HALLUC_WORK_DIR).
    input_root: where Kaggle Datasets are mounted (/kaggle/input); None locally.
    repo_root: root of this repo (contains pkg_halluc/, configs/, data/).
    """

    work_dir: Path
    input_root: Path | None
    repo_root: Path

    # -- third-party source code, fetched at setup time (see deps/fetch.py) ----
    @property
    def third_party_dir(self) -> Path:
        return self.work_dir / "third_party"

    @property
    def au_repo_dir(self) -> Path:
        return self.third_party_dir / "adaptive-unlearning"

    # -- models & training checkpoints ------------------------------------------
    @property
    def models_dir(self) -> Path:
        return self.repo_root / "models"

    # -- generated data (tri-mask retain/forget sets for GA/NPO) ----------------
    @property
    def generated_data_dir(self) -> Path:
        return self.repo_root / "generated_data"

    # -- checkpoints for methods that don't depend on the AU repo ---------------
    @property
    def checkpoints_dir(self) -> Path:
        """ga_plain/npo_plain checkpoints -- kept separate from models_dir
        (base model download) and au_repo_dir/Models (ga/npo checkpoints)."""
        return self.repo_root / "checkpoints"

    # -- evaluation runs & final reports -----------------------------------------
    @property
    def eval_runs_dir(self) -> Path:
        return self.work_dir / "eval_runs"

    @property
    def outputs_dir(self) -> Path:
        return self.work_dir / "outputs"

    # -- data assets --------------------------------------------------------
    @property
    def main_data_dir(self) -> Path:
        return self.repo_root / "data"

    @property
    def main_data_paths(self) -> list[Path]:
        """The four master data files used to build the static GA/NPO tri-mask dataset."""
        filenames = [
            "LLM_Recent_Master.json",
            "LLM_All_Time_Master.json",
            "Stack_Overflow_Recent_Master.json",
            "Stack_Overflow_All_Time_Master.json",
        ]
        resolved = []
        for fn in filenames:
            p_sub = self.main_data_dir / "Llama3_3_Python" / fn
            if p_sub.exists():
                resolved.append(p_sub)
            else:
                resolved.append(self.main_data_dir / fn)
        return resolved

    @property
    def main_prompts_path(self) -> Path:
        return [
            self.main_data_dir / "package_prompt" / "LLM_LY.json",
            self.main_data_dir / "package_prompt" / "LLM_AT.json",
            self.main_data_dir / "package_prompt" / "SO_LY.json",
            self.main_data_dir / "package_prompt" / "SO_AT.json"
        ]

    @property
    def main_hallu_result_paths(self) -> list[Path]:
        """The four results CSVs used to build the static GA/NPO tri-mask dataset."""
        filenames = [
            "LLM_LY_results.csv",
            "LLM_AT_results.csv",
            "SO_LY_results.csv",
            "SO_AT_results.csv",
        ]
        resolved = []
        for fn in filenames:
            p_sub = self.main_data_dir / "Llama3_3_Python" / fn
            if p_sub.exists():
                resolved.append(p_sub)
            else:
                resolved.append(self.main_data_dir / fn)
        return resolved

    @property
    def main_origin_prompts_path(self) -> Path:
        return [
            self.main_data_dir / "original_prompt" / "LLM_LY.json",
            self.main_data_dir / "original_prompt" / "LLM_AT.json",
            self.main_data_dir / "original_prompt" / "SO_LY.json",
            self.main_data_dir / "original_prompt" / "SO_AT.json"
        ]

    # -- au_repo data assets --------------------------------------------------
    # Seed prompts, eval prompts, PyPI snapshot, false-positive list: bundled
    # inside the fetched AU repo, read from there instead of duplicating.
    # `data/` in this repo is for custom prompt sets (see data/README.md).
    @property
    def au_seed_prompts_path(self) -> Path:
        return self.au_repo_dir / "New_Data_Set" / "prompts.jsonl"

    @property
    def au_eval_prompts_path(self) -> Path:
        return self.au_repo_dir / "Package_Hallucination_Testing" / "Data" / "prompts.jsonl"

    @property
    def au_pypi_csv_path(self) -> Path:
        return self.au_repo_dir / "Package_Hallucination_Testing" / "Data" / "pypi_package_names.csv"

    @property
    def au_false_positive_csv_path(self) -> Path:
        return self.au_repo_dir / "Package_Hallucination_Testing" / "Data" / "false_positive_packages.csv"

    @property
    def templates_dir(self) -> Path:
        return Path(__file__).resolve().parent / "templates"

    def ensure_dirs(self) -> None:
        for d in (
            self.work_dir,
            self.third_party_dir,
            self.models_dir,
            self.generated_data_dir,
            self.checkpoints_dir,
            self.eval_runs_dir,
            self.outputs_dir,
        ):
            d.mkdir(parents=True, exist_ok=True)


def _find_repo_root(start: Path) -> Path:
    """Walk up from *start* looking for the repo root (has a ``pkg_halluc`` dir)."""
    cur = start.resolve()
    for candidate in (cur, *cur.parents):
        if (candidate / "src" / "pkg_halluc").is_dir() and (candidate / "configs").is_dir():
            return candidate
        if (candidate / "pkg_halluc").is_dir() and (candidate / "configs").is_dir():
            return candidate
    return Path(__file__).resolve().parents[1]


def resolve_paths(work_dir: str | Path | None = None) -> Paths:
    """Build a Paths object, applying the override/env-var/default chain."""
    if work_dir is None:
        work_dir = os.environ.get("PKG_HALLUC_WORK_DIR")
    work_dir_path = Path(work_dir).expanduser().resolve() if work_dir else default_work_dir()
    paths = Paths(
        work_dir=work_dir_path,
        input_root=default_input_root(),
        repo_root=_find_repo_root(Path.cwd()),
    )
    paths.ensure_dirs()
    return paths
