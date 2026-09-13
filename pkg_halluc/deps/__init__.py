from __future__ import annotations

from typing import Any

from ..config import resolve_dtype
from ..paths import Paths
from .fetch import ensure_adaptive_unlearning_repo
from .materialize import materialize_templates
from .patches import apply_all_patches

__all__ = [
    "ensure_adaptive_unlearning_repo",
    "materialize_templates",
    "apply_all_patches",
    "setup_dependencies",
]


def setup_dependencies(
    cfg: dict[str, Any],
    paths: Paths,
    au_src: str | None = None,
) -> None:
    """Fetch AU, patch it for compatibility, drop this project's own scripts
    into place. Idempotent. `pkg_halluc fetch-deps`'s implementation, also
    called from `run-all`."""
    ensure_adaptive_unlearning_repo(cfg, paths, cli_source=au_src)

    dtype = resolve_dtype(cfg)
    apply_all_patches(
        au_repo_dir=paths.au_repo_dir,
        use_fp16=(dtype == "float16"),
    )
    materialize_templates(paths)
    print(f"[setup] dependencies ready (dtype={dtype})")
