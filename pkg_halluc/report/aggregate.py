"""Aggregate per-method ``eval_runs/<tag>/FINAL_RESULTS.csv`` (+
``import_scan_results.csv``) into the final comparison tables:

* ``hall_table`` -- one row per method, overall Package Hallucination Rate
  and absolute reduction vs. the base model.
* ``context_table`` -- the same, broken out by generation context
  (``install_cmd``, ``nl_query_1``, ``nl_query_2``, ``bare_import``).
"""
from __future__ import annotations

import ast
from typing import Any

import pandas as pd

from ..methods.registry import DISPLAY_ORDER, build_registry
from ..paths import Paths

CONTEXT_LABELS = {
    "install_cmd": "install_cmd (pip install in generated code)",
    "nl_query_1": "nl_query_1 (follow-up: what packages does this code need)",
    "nl_query_2": "nl_query_2 (follow-up: suggest packages for this task)",
    "bare_import": "bare_import (raw `import` statements, no pip install needed)",
}


def _safe_len(list_repr: Any) -> int:
    try:
        return len(ast.literal_eval(list_repr)) if isinstance(list_repr, str) else len(list_repr)
    except (ValueError, SyntaxError):
        return 0


def _rate(h: float, v: float) -> float:
    t = h + v
    return 100.0 * h / t if t else float("nan")


def read_hallucination_totals(tag: str, paths: Paths) -> tuple[float, dict[str, float]]:
    """Read one method's `eval_runs/<tag>/` output. Returns (overall_rate, context_rates)."""
    run_dir = paths.eval_runs_dir / tag
    df = pd.read_csv(run_dir / "FINAL_RESULTS.csv", index_col=0)
    row = df.loc["Totals"]

    pip_h, pip_v = row.get("pip_hallucinated", 0), row.get("pip_valid", 0)
    h1, v1 = row.get("hallucinated_1", 0), row.get("valid_1", 0)
    h2, v2 = row.get("hallucinated_2", 0), row.get("valid_2", 0)

    import_h = import_v = 0
    import_path = run_dir / "import_scan_results.csv"
    if import_path.exists():
        idf = pd.read_csv(import_path)
        import_h = idf["import_hallucinated"].apply(_safe_len).sum()
        import_v = idf["import_valid"].apply(_safe_len).sum()

    halluc = pip_h + h1 + h2 + import_h
    valid = pip_v + v1 + v2 + import_v
    total = halluc + valid
    overall_rate = 100.0 * halluc / total if total else float("nan")

    context_rates = {
        CONTEXT_LABELS["install_cmd"]: _rate(pip_h, pip_v),
        CONTEXT_LABELS["nl_query_1"]: _rate(h1, v1),
        CONTEXT_LABELS["nl_query_2"]: _rate(h2, v2),
        CONTEXT_LABELS["bare_import"]: _rate(import_h, import_v),
    }
    return overall_rate, context_rates


def build_report(cfg: dict[str, Any], paths: Paths) -> tuple[pd.DataFrame, pd.DataFrame]:
    registry = build_registry(cfg)
    tags = [t for t in DISPLAY_ORDER if registry[t].enabled]

    hall_rates: dict[str, float] = {}
    context_rows: list[dict[str, Any]] = []
    for tag in tags:
        try:
            rate, ctx = read_hallucination_totals(tag, paths)
        except Exception as e:  # noqa: BLE001 -- report generation should degrade, not crash
            print(f"[WARN] could not read results for '{tag}': {e}")
            rate, ctx = float("nan"), {}
        hall_rates[tag] = rate
        context_rows.append({"Variant": registry[tag].display_name, **ctx})

    base_rate = hall_rates.get("base", float("nan"))
    hall_table = pd.DataFrame(
        {
            "Variant": [registry[t].display_name for t in tags],
            "Hallucination Rate (%)": [hall_rates[t] for t in tags],
            "Abs. Reduction": [(base_rate - hall_rates[t]) if t != "base" else 0.0 for t in tags],
        }
    )
    context_table = pd.DataFrame(context_rows).round(2)
    return hall_table.round(2), context_table


def save_report(hall_table: pd.DataFrame, context_table: pd.DataFrame, paths: Paths) -> tuple[str, str]:
    paths.outputs_dir.mkdir(parents=True, exist_ok=True)
    table1_path = paths.outputs_dir / "table1_hallucination_rate.csv"
    context_path = paths.outputs_dir / "table1_by_context.csv"
    hall_table.set_index("Variant").to_csv(table1_path)
    context_table.to_csv(context_path, index=False)
    return str(table1_path), str(context_path)
