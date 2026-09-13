"""Evaluate one method's package hallucination rate across all 4 contexts
(``install_cmd``, ``nl_query_1``, ``nl_query_2``, ``bare_import``).

Thin wrapper around the materialized ``eval_variant.py`` template -- this
module's only job is building the right CLI invocation from a
:class:`~pkg_halluc.methods.spec.MethodSpec` and running it.
"""
from __future__ import annotations

from ..methods.spec import MethodContext, MethodSpec
from ..utils.proc import python_run


def evaluate_method(spec: MethodSpec, ctx: MethodContext) -> None:
    if not spec.enabled:
        print(f"[{spec.name}] skipped (disabled in config)")
        return

    checkpoint = spec.checkpoint_path(ctx)
    package_modes = spec.package_modes(ctx)
    print(f"\n===== Evaluating: {spec.name} ({checkpoint}) -- kind={spec.kind} =====")

    cmd = [
        "eval_variant.py",
        "--tag", spec.name,
        "--model_path", str(checkpoint),
        "--n_prompts", str(ctx.cfg["eval"]["n_eval_prompts"]),
        "--batch_size", str(ctx.cfg["eval"]["batch_size"]),
        "--seed", str(ctx.cfg["seed"]),
        "--package_modes", *[str(m) for m in package_modes],
        "--pypi_csv", str(ctx.paths.au_pypi_csv_path),
        "--false_positive_csv", str(ctx.paths.au_false_positive_csv_path),
        "--out_dir", str(ctx.paths.eval_runs_dir / spec.name),
    ]

    eval_prompts_override = ctx.cfg["eval"].get("eval_prompts_path")
    if eval_prompts_override:
        # Relative paths are resolved against the repo root (e.g.
        # "data/my_prompts.jsonl"); an absolute path is used as-is.
        resolved = (ctx.paths.repo_root / eval_prompts_override).resolve()
        if not resolved.is_file():
            raise FileNotFoundError(
                f"eval.eval_prompts_path is set to {eval_prompts_override!r} but "
                f"{resolved} doesn't exist."
            )
        cmd += ["--eval_prompts_path", str(resolved)]

    if spec.extra_eval_args is not None:
        cmd += spec.extra_eval_args(ctx)
    else:
        cmd += ["--method", "standard"]

    python_run(
        cmd,
        cwd=ctx.paths.au_repo_dir / "Package_Hallucination_Testing",
        work_dir_for_disk_log=ctx.paths.work_dir,
    )
