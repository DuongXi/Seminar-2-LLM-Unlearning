"""Builds the dict of MethodSpec the rest of the pipeline (train/evaluate/
report CLI commands) operates on.

To add a method: write methods/<name>.py with a build_spec(enabled) ->
MethodSpec function (see methods/ga.py for a simple example), then add one
line to _BUILDERS below.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from . import npo, reference, steering
from ..paths import Paths
from . import ga, ga_plain, npo_plain
from .spec import MethodContext, MethodSpec

_BUILDERS = {
    "base": reference.build_spec,
    "ga": ga.build_spec,
    "npo": npo.build_spec,
    # same retain/forget rows as ga/npo, no AU code, no tri-mask
    "ga_plain": ga_plain.build_spec,
    "npo_plain": npo_plain.build_spec,
    "steering": steering.build_spec,
}

# Display order for tables/reports -- not necessarily the same as _BUILDERS'
# (arbitrary dict) order.
DISPLAY_ORDER = ["base", "ga", "npo", "ga_plain", "npo_plain", "steering"]


def build_registry(cfg: dict[str, Any]) -> dict[str, MethodSpec]:
    registry: dict[str, MethodSpec] = {}
    for name in DISPLAY_ORDER:
        method_cfg = cfg.get("methods", {}).get(name, {})
        enabled = bool(method_cfg.get("enabled", False))
        registry[name] = _BUILDERS[name](enabled)
    return registry


def enabled_methods(cfg: dict[str, Any]) -> list[str]:
    registry = build_registry(cfg)
    return [name for name in DISPLAY_ORDER if registry[name].enabled]


def make_context(cfg: dict[str, Any], paths: Paths, base_model_path: Path, method_name: str) -> MethodContext:
    return MethodContext(
        cfg=cfg,
        paths=paths,
        base_model_path=base_model_path,
        method_cfg=cfg.get("methods", {}).get(method_name, {}),
    )
