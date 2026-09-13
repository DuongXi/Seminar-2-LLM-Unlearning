"""NPO-plain -- Negative Preference Optimization, no tri-mask, no AU code.

Same retain/forget rows as npo (methods/npo.py), but loss over each response
in full instead of AU's per-token tri-mask. Ablation comparison against npo.
"""
from __future__ import annotations

from ._finetune_plain_common import checkpoint_path_for_plain, train_plain_finetune
from .spec import MethodContext, MethodSpec

TAG = "npo_plain"


def _package_modes(ctx: MethodContext) -> list[int]:
    return list(ctx.cfg["eval"]["package_modes"])


def build_spec(enabled: bool) -> MethodSpec:
    return MethodSpec(
        name=TAG,
        display_name="NPO-plain (no tri-mask)",
        kind="weight_finetune",
        enabled=enabled,
        checkpoint_path=lambda ctx: checkpoint_path_for_plain(TAG, ctx),
        package_modes=_package_modes,
        train=lambda ctx: train_plain_finetune(TAG, "npo_plain", ctx),
    )
