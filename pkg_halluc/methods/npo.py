"""NPO -- Negative Preference Optimization (Zhang et al., 2024).

Trains with the upstream ``NPOTrainer`` (AU repo's ``npo_trainer.py``,
unmodified) via ``train.py --loss_function npo``, on the same fixed static
tri-mask dataset GA uses.
"""
from __future__ import annotations

from ._finetune_common import checkpoint_path_for, train_full_finetune
from .spec import MethodContext, MethodSpec

TAG = "npo"


def _package_modes(ctx: MethodContext) -> list[int]:
    return list(ctx.cfg["eval"]["package_modes"])


def build_spec(enabled: bool) -> MethodSpec:
    return MethodSpec(
        name=TAG,
        display_name="NPO",
        kind="weight_finetune",
        enabled=enabled,
        checkpoint_path=lambda ctx: checkpoint_path_for(TAG, ctx),
        package_modes=_package_modes,
        train=lambda ctx: train_full_finetune(TAG, "npo", ctx),
    )
