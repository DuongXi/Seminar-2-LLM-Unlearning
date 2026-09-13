"""Fetch the third-party source code this pipeline depends on.

This project doesn't vendor the Adaptive Unlearning (AU) source tree in
git (it's published anonymously for double-blind review, so its license
terms aren't settled). Everything under work_dir/third_party/ is fetched
on demand and git-ignored.

Resolution order for AU, first match wins:
  1. Explicit source (--au-src, or deps.adaptive_unlearning.source in
     config): a local .zip, a local directory, or an http(s) URL to a .zip.
  2. A Kaggle Dataset already mounted under /kaggle/input (matched by the
     repo's signature files).

No default URL for AU -- if neither above finds it, ensure_repo raises.

Nothing below is AU-specific except ensure_adaptive_unlearning_repo at the
bottom -- ensure_repo() itself is generic.
"""
from __future__ import annotations

import glob as globmod
import shutil
import urllib.request
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..paths import Paths


@dataclass(frozen=True)
class RepoSpec:
    name: str  # human-readable, used in log/error messages
    signature_rel_path: str  # a file that must exist, relative to the repo root
    extra_signature_rel_paths: tuple[str, ...] = ()  # additional required files at repo root
    zip_name_globs: tuple[str, ...] = field(default_factory=tuple)  # filename globs to search for
    default_url: str | None = None
    config_key: str = ""  # key under cfg["deps"][...] holding an optional "source"


def _looks_like(directory: Path, spec: RepoSpec) -> bool:
    if not (directory / spec.signature_rel_path).is_file():
        return False
    return all((directory / p).is_file() for p in spec.extra_signature_rel_paths)


def _flatten_in_place(target_dir: Path, spec: RepoSpec) -> None:
    """Hoist a zip's wrapping folder (e.g. Adaptive-Unlearning-main/Adaptive-Unlearning-main/...)
    up one level if that's what got extracted."""
    if _looks_like(target_dir, spec):
        return
    entries = [p for p in target_dir.iterdir() if not p.name.startswith(".")]
    if len(entries) == 1 and entries[0].is_dir() and _looks_like(entries[0], spec):
        nested = entries[0]
        for item in nested.iterdir():
            shutil.move(str(item), str(target_dir / item.name))
        nested.rmdir()


def _extract_zip(zip_path: Path, target_dir: Path) -> None:
    target_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(target_dir)


def _download_zip(url: str, dest_zip_path: Path) -> None:
    dest_zip_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {url}\n         -> {dest_zip_path}")
    urllib.request.urlretrieve(url, dest_zip_path)  # noqa: S310 (trusted, user-configured URL)


def _search_kaggle_input(input_root: Path, spec: RepoSpec) -> tuple[Path | None, Path | None]:
    """Look under /kaggle/input for a matching .zip or an already-extracted
    directory (Kaggle auto-extracts .zip Datasets). Returns (zip_path, dir_path);
    at most one is non-None."""
    for pattern in spec.zip_name_globs:
        matches = sorted(globmod.glob(str(input_root / "**" / pattern), recursive=True))
        if matches:
            return Path(matches[0]), None

    signature_basename = Path(spec.signature_rel_path).name
    signature_depth = len(Path(spec.signature_rel_path).parts) - 1  # levels between repo root and the file
    for path_str in globmod.glob(str(input_root / "**" / signature_basename), recursive=True):
        candidate = Path(path_str).parent
        for _ in range(signature_depth):
            candidate = candidate.parent
        if _looks_like(candidate, spec):
            return None, candidate
    return None, None


def _place_from_source(source: str, target_dir: Path, spec: RepoSpec) -> None:
    """Resolve an explicit source (path or URL) into *target_dir*."""
    if source.startswith("http://") or source.startswith("https://"):
        tmp_zip = target_dir.parent / f"_download_{spec.config_key or 'repo'}.zip"
        _download_zip(source, tmp_zip)
        _extract_zip(tmp_zip, target_dir)
        tmp_zip.unlink(missing_ok=True)
        return

    src_path = Path(source).expanduser()
    if src_path.is_file() and src_path.suffix == ".zip":
        _extract_zip(src_path, target_dir)
        return
    if src_path.is_dir():
        shutil.copytree(src_path, target_dir, dirs_exist_ok=True)
        return
    raise FileNotFoundError(f"--source '{source}' is neither a .zip file, a directory, nor an http(s) URL.")


def _missing_source_message(spec: RepoSpec, paths: Paths) -> str:
    lines = [
        f"Could not find the '{spec.name}' source code anywhere.",
        "Tried, in order:",
        "  1. an explicit --*-src / config deps.*.source",
        f"  2. a Kaggle Dataset under {paths.input_root} matching {spec.zip_name_globs or spec.signature_rel_path}"
        if paths.input_root
        else "  2. (skipped -- not running on Kaggle, no /kaggle/input mount)",
    ]
    if spec.default_url:
        lines.append(f"  3. the default URL ({spec.default_url}) -- download must have failed, check your network")
    else:
        lines.append("  3. (no default URL configured for this dependency -- see README.md)")
    lines.append(
        "Fix: pass --au-src pointing at a local .zip or directory, set "
        "deps.adaptive_unlearning.source in your config JSON, or (on "
        "Kaggle) attach a Dataset containing the code via 'Add Input'."
    )
    return "\n".join(lines)


def ensure_repo(spec: RepoSpec, target_dir: Path, paths: Paths, explicit_source: str | None = None) -> Path:
    """Make sure *target_dir* contains a valid copy of the repo described by
    *spec*, fetching it if necessary. Idempotent: if it's already there and
    valid, does nothing and returns immediately."""
    if _looks_like(target_dir, spec):
        print(f"[{spec.name}] already present at {target_dir}")
        return target_dir

    if explicit_source:
        print(f"[{spec.name}] using explicit source: {explicit_source}")
        _place_from_source(explicit_source, target_dir, spec)
    elif paths.input_root is not None:
        zip_path, dir_path = _search_kaggle_input(paths.input_root, spec)
        if zip_path is not None:
            print(f"[{spec.name}] found zip on Kaggle: {zip_path}")
            _extract_zip(zip_path, target_dir)
        elif dir_path is not None:
            print(f"[{spec.name}] found pre-extracted Kaggle Dataset: {dir_path}")
            shutil.copytree(dir_path, target_dir, dirs_exist_ok=True)
        elif spec.default_url:
            print(f"[{spec.name}] not found on Kaggle, falling back to default URL")
            _place_from_source(spec.default_url, target_dir, spec)
        else:
            raise FileNotFoundError(_missing_source_message(spec, paths))
    elif spec.default_url:
        _place_from_source(spec.default_url, target_dir, spec)
    else:
        raise FileNotFoundError(_missing_source_message(spec, paths))

    _flatten_in_place(target_dir, spec)
    if not _looks_like(target_dir, spec):
        raise RuntimeError(
            f"[{spec.name}] fetched into {target_dir} but it doesn't look like a valid "
            f"repo (missing {spec.signature_rel_path}). Check the source and try again "
            f"after deleting {target_dir}."
        )
    print(f"[{spec.name}] ready at {target_dir}")
    return target_dir


# ---------------------------------------------------------------------------
# AU-specific wrapper
# ---------------------------------------------------------------------------

AU_SPEC = RepoSpec(
    name="Adaptive Unlearning (AU)",
    signature_rel_path="train.py",
    extra_signature_rel_paths=("au_trainer.py", "ga_trainer.py", "npo_trainer.py"),
    zip_name_globs=("Adaptive-Unlearning-952E.zip", "Adaptive*Unlearning*.zip"),
    default_url=None,
    config_key="adaptive_unlearning",
)


def ensure_adaptive_unlearning_repo(cfg: dict[str, Any], paths: Paths, cli_source: str | None = None) -> Path:
    source = cli_source or cfg.get("deps", {}).get("adaptive_unlearning", {}).get("source")
    return ensure_repo(AU_SPEC, paths.au_repo_dir, paths, explicit_source=source)
