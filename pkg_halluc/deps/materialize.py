"""Copy this project's own scripts into the fetched AU source tree.

A few scripts have to physically live inside the AU source tree to work
(they import sibling modules by relative import, like the upstream
authors' own scripts do). Kept as ordinary .py files under
pkg_halluc/templates/ in this repo and copied over on every run, so they
never drift from what's checked in.
"""
from __future__ import annotations

import shutil

from ..paths import Paths

# (relative path under templates/, relative path under the target repo)
_AU_FILES = [
    ("adaptive_unlearning/build_data.py", "build_data.py"),
    (
        "adaptive_unlearning/package_hallucination_testing/import_scan.py",
        "Package_Hallucination_Testing/import_scan.py",
    ),
    (
        "adaptive_unlearning/package_hallucination_testing/eval_variant.py",
        "Package_Hallucination_Testing/eval_variant.py",
    ),
]


def materialize_templates(paths: Paths) -> None:
    """Copy every template into place. Call after ``ensure_adaptive_unlearning_repo()`` + patches."""
    for rel_template, rel_target in _AU_FILES:
        src = paths.templates_dir / rel_template
        dst = paths.au_repo_dir / rel_target
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dst)
        print(f"[materialize] {rel_template} -> {dst}")
