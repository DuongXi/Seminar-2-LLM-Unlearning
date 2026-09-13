"""GA -- Gradient Ascent, the classic unlearning baseline.

Trains with the upstream ``GradientAscentTrainer`` (AU repo's
``ga_trainer.py``, unmodified) via ``train.py --loss_function ga``, on the
fixed static tri-mask dataset from ``pkg_halluc data build-seed-data``.
"""
from __future__ import annotations

from ._finetune_common import checkpoint_path_for, train_full_finetune
from .spec import MethodContext, MethodSpec

TAG = "ga"


def _package_modes(ctx: MethodContext) -> list[int]:
    return list(ctx.cfg["eval"]["package_modes"])


def build_spec(enabled: bool) -> MethodSpec:
    return MethodSpec(
        name=TAG,
        display_name="GA",
        kind="weight_finetune",
        enabled=enabled,
        checkpoint_path=lambda ctx: checkpoint_path_for(TAG, ctx),
        package_modes=_package_modes,
        train=lambda ctx: train_full_finetune(TAG, "ga", ctx),
    )
