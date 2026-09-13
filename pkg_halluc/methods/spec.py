"""Pluggable "method" abstraction.

Every mitigation method compared by this project -- the unmodified model,
GA, NPO, their no-tri-mask "plain" ablations, and (once implemented)
Representation Steering -- is one MethodSpec. registry.py builds the dict
of active specs from the config; cli.py only ever talks to that dict, never
to e.g. ga.py directly.

To add a method: write methods/<name>.py with a build_spec() function and
register it in registry.py. See methods/steering.py for a placeholder.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from ..paths import Paths

# One of:
#   "reference"               - the unmodified base model
#   "weight_finetune"         - full fine-tune on a fixed dataset (GA, NPO, plain ablations)
#   "decode_monitor"          - no weight change, intervenes at decode time
#   "representation_steering" - no weight change, intervenes on hidden states (planned)
MethodKind = str


@dataclass
class MethodContext:
    """Everything a method implementation needs, bundled so MethodSpec
    callables all share one signature."""

    cfg: dict[str, Any]
    paths: Paths
    base_model_path: Path
    method_cfg: dict[str, Any]  # cfg["methods"][<name>]


@dataclass
class MethodSpec:
    name: str
    display_name: str
    kind: MethodKind
    enabled: bool

    # Checkpoint directory to evaluate this method with (== base_model_path
    # for kinds that don't change weights).
    checkpoint_path: Callable[[MethodContext], Path]

    # Which package_modes (nl_query_1 / nl_query_2) to evaluate with.
    package_modes: Callable[[MethodContext], list[int]]

    # None for methods with no training step.
    train: Callable[[MethodContext], None] | None = None

    # One-time preparation that isn't training, run before each evaluate.
    prepare: Callable[[MethodContext], None] | None = None

    # Extra eval_variant.py CLI args specific to this method.
    extra_eval_args: Callable[[MethodContext], list[str]] | None = None
