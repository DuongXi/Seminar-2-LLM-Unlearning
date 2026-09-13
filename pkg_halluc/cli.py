"""``pkg_halluc`` command-line interface.

Every stage of the pipeline is its own subcommand so a run can be stopped and
resumed at any point (handy on Kaggle, where sessions have a wall-clock
limit) -- see README.md "Usage" for the full walkthrough. ``run-all`` chains
every stage for a one-shot convenience run.

    pkg_halluc fetch-deps
    pkg_halluc download-model
    pkg_halluc build-data
    pkg_halluc train --method all
    pkg_halluc evaluate --method all
    pkg_halluc report

    # or, equivalently:
    pkg_halluc run-all
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

from . import deps, model_setup
from .config import dump, load_config
from .tok_data import build_data
from .eval import evaluate_method
from .methods.registry import DISPLAY_ORDER, build_registry, make_context
from .paths import Paths, resolve_paths
from .report import build_report, save_report


def _resolve_config_path(cli_value: str | None, paths: Paths) -> Path | None:
    if cli_value is None:
        default = paths.repo_root / "configs" / "default.json"
        return default if default.exists() else None
    p = Path(cli_value)
    if p.exists():
        return p
    alt = paths.repo_root / "configs" / cli_value
    if alt.exists():
        return alt
    raise FileNotFoundError(f"Config file not found: {cli_value} (also tried {alt})")


def _add_common_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--config", default=None, help="Path to a config JSON (default: configs/default.json)")
    p.add_argument("--work-dir", default=None, help="Override the runtime work directory")
    p.add_argument("--model", default=None, help="Override cfg.model_name (preset name or full HF id)")
    p.add_argument("--seed", type=int, default=None, help="Override cfg.seed")


def _load(args: argparse.Namespace) -> tuple[dict[str, Any], Paths]:
    paths = resolve_paths(args.work_dir)
    cfg = load_config(_resolve_config_path(args.config, paths))
    if args.model is not None:
        from .config import MODEL_PRESETS

        cfg["model_name"] = MODEL_PRESETS.get(args.model, args.model)
    if args.seed is not None:
        cfg["seed"] = args.seed
    return cfg, paths


def _require_base_model(paths: Paths, cfg: dict[str, Any]) -> Path:
    model_path = paths.models_dir / cfg["model_name"]
    if not model_path.exists() or not any(model_path.iterdir()):
        raise FileNotFoundError(
            f"Base model not found at {model_path}. Run `pkg_halluc download-model` first."
        )
    return model_path


def _method_names(arg_value: str) -> list[str]:
    return DISPLAY_ORDER if arg_value == "all" else [arg_value]


# --- subcommands -------------------------------------------------------------


def cmd_fetch_deps(args: argparse.Namespace) -> None:
    cfg, paths = _load(args)
    print(dump(cfg))
    deps.setup_dependencies(cfg, paths, au_src=args.au_src)


def cmd_download_model(args: argparse.Namespace) -> None:
    cfg, paths = _load(args)
    model_path = model_setup.download_base_model(cfg, paths)
    if not args.skip_sanity_check:
        model_setup.sanity_check_generation(cfg, model_path)


def cmd_build_data(args: argparse.Namespace) -> None:
    cfg, paths = _load(args)
    model_path = _require_base_model(paths, cfg)
    build_data(cfg, paths, model_path)


def cmd_train(args: argparse.Namespace) -> None:
    cfg, paths = _load(args)
    model_path = _require_base_model(paths, cfg)
    registry = build_registry(cfg)
    for name in _method_names(args.method):
        spec = registry[name]
        if not spec.enabled:
            print(f"[{name}] skipped (disabled in config)")
            continue
        if spec.train is None:
            print(f"[{name}] nothing to train (kind={spec.kind})")
            continue
        ctx = make_context(cfg, paths, model_path, name)
        print(f"\n===== Training: {name} =====")
        spec.train(ctx)


def cmd_evaluate(args: argparse.Namespace) -> None:
    cfg, paths = _load(args)
    model_path = _require_base_model(paths, cfg)
    registry = build_registry(cfg)
    for name in _method_names(args.method):
        spec = registry[name]
        ctx = make_context(cfg, paths, model_path, name)
        if spec.prepare is not None:
            spec.prepare(ctx)
        evaluate_method(spec, ctx)


def cmd_report(args: argparse.Namespace) -> None:
    cfg, paths = _load(args)
    hall_table, context_table = build_report(cfg, paths)
    print(f"Model: {cfg['model_name']}\n")
    print("Table 1 -- Package Hallucination Rate:")
    print(hall_table.set_index("Variant").to_string())
    print("\nBy generation context:")
    print(context_table.to_string(index=False))
    table1_path, context_path = save_report(hall_table, context_table, paths)
    print(f"\nSaved: {table1_path}\nSaved: {context_path}")


def cmd_run_all(args: argparse.Namespace) -> None:
    cmd_fetch_deps(args)
    cmd_download_model(args)
    cmd_build_data(args)
    args.method = "all"
    cmd_train(args)
    cmd_evaluate(args)
    cmd_report(args)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="pkg_halluc", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("fetch-deps", help="Fetch + patch AU source code")
    _add_common_args(p)
    p.add_argument("--au-src", default=None, help="Local .zip/dir or URL for the AU repo (see README)")
    p.set_defaults(func=cmd_fetch_deps)

    p = sub.add_parser("download-model", help="Download the base model from the Hub")
    _add_common_args(p)
    p.add_argument("--skip-sanity-check", action="store_true", help="Skip the one-shot test generation")
    p.set_defaults(func=cmd_download_model)

    p = sub.add_parser("build-data", help="Build the static GA/NPO tri-mask training data")
    _add_common_args(p)
    p.set_defaults(func=cmd_build_data)

    p = sub.add_parser("train", help="Train one (or all) weight_finetune methods")
    _add_common_args(p)
    p.add_argument("--method", default="all", choices=[*DISPLAY_ORDER, "all"])
    p.set_defaults(func=cmd_train)

    p = sub.add_parser("evaluate", help="Evaluate one (or all) methods' hallucination rate")
    _add_common_args(p)
    p.add_argument("--method", default="all", choices=[*DISPLAY_ORDER, "all"])
    p.set_defaults(func=cmd_evaluate)

    p = sub.add_parser("report", help="Aggregate eval_runs/* into the final tables")
    _add_common_args(p)
    p.set_defaults(func=cmd_report)

    p = sub.add_parser("run-all", help="Run every stage above in sequence")
    _add_common_args(p)
    p.add_argument("--au-src", default=None)
    p.add_argument("--skip-sanity-check", action="store_true")
    p.set_defaults(func=cmd_run_all)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
