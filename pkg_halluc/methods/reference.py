"""The unmodified base model -- the comparison point every other method's
"Abs. Reduction" column is measured against. No training, no decode-time
intervention."""
from __future__ import annotations

from pathlib import Path

from .spec import MethodContext, MethodSpec


def _checkpoint_path(ctx: MethodContext) -> Path:
    return ctx.base_model_path


def _package_modes(ctx: MethodContext) -> list[int]:
    return list(ctx.cfg["eval"]["package_modes"])


def build_spec(enabled: bool) -> MethodSpec:
    return MethodSpec(
        name="base",
        display_name="Base",
        kind="reference",
        enabled=enabled,
        checkpoint_path=_checkpoint_path,
        package_modes=_package_modes,
        train=None,
    )
