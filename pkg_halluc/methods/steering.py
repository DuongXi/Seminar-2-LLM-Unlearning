"""Representation Steering -- placeholder for this project's own, novel
mitigation method (no upstream code exists for it, unlike GA/NPO).

Planned approach: an ITI / mean-difference steering vector computed from
paired valid/hallucinated package-name activations, added to the residual
stream at generation time.

This is a stub:
  * registered like every other method (registry.py) so wiring it up later
    doesn't touch the CLI, eval loop, or report code -- only this file.
  * disabled by default so `--method all` skips it until implemented.
  * calling train/prepare explicitly raises NotImplementedError with a
    pointer back here.

To implement:
  1. Add a hyperparameter block under methods.steering in config.py
     (layer index, steering strength, which contrastive pairs).
  2. Fill in a vector-computation function, have `prepare` cache the result
     under paths.generated_data_dir.
  3. Fill in a generation-time hook (adds the cached vector to the target
     layer's residual stream) and a generation adapter with the same I/O
     shape as generate_code.py/generate_package_names.py, wired in via
     MethodSpec.extra_eval_args. steering_ref/tsv/ has an archived
     llm_layers.py (Apache-2.0) with a generic version of this hook
     mechanism -- see steering_ref/NOTES.md for what needs fixing first.
  4. kind="representation_steering" is already set below; flip enabled: true.
"""
from __future__ import annotations

from pathlib import Path

from .spec import MethodContext, MethodSpec

TAG = "steering"


def _checkpoint_path(ctx: MethodContext) -> Path:
    # No weight changes -- steering intervenes at generation time via a hook.
    return ctx.base_model_path


def _package_modes(ctx: MethodContext) -> list[int]:
    return list(ctx.cfg["eval"]["package_modes"])


def _not_implemented(ctx: MethodContext) -> None:
    raise NotImplementedError(
        "Representation Steering has no implementation yet -- see the module "
        "docstring in pkg_halluc/methods/steering.py for the planned "
        "approach and what to fill in. It stays disabled "
        "(methods.steering.enabled=false) until then."
    )


def build_spec(enabled: bool) -> MethodSpec:
    if enabled:
        raise NotImplementedError(
            "methods.steering.enabled=true in your config, but Representation "
            "Steering isn't implemented yet -- see pkg_halluc/methods/steering.py."
        )
    return MethodSpec(
        name=TAG,
        display_name="Representation Steering",
        kind="representation_steering",
        enabled=enabled,
        checkpoint_path=_checkpoint_path,
        package_modes=_package_modes,
        train=_not_implemented,
        prepare=_not_implemented,
    )
