"""Combine method evaluation results into overall and context-specific hallucination tables."""
from __future__ import annotations

import argparse
import ast
from pathlib import Path
from typing import Any

import pandas as pd

METHOD_TAGS: list[tuple[str, str]] = [
    ("base", "Base"),
    ("ga", "GA"),
    ("npo", "NPO"),
    ("ga_plain", "GA-plain (no Tri-Mask)"),
    ("npo_plain", "NPO-plain (no Tri-Mask)"),
]

CONTEXT_LABELS = {
    "install_cmd": "install_cmd (pip install trong code sinh ra)",
    "nl_query_1": "nl_query_1 (packages required by this code)",
    "nl_query_2": "nl_query_2 (package recommendations for this task)",
    "bare_import": "bare_import (import statement without pip install)",
}


def safe_len(list_repr: Any) -> int:
    """Count list elements, returning zero if a string representation cannot be parsed."""
    try:
        return len(ast.literal_eval(list_repr)) if isinstance(list_repr, str) else len(list_repr)
    except (ValueError, SyntaxError):
        return 0


def rate(h: float, v: float) -> float:
    """Return the hallucination rate as a percentage, or NaN when no packages exist."""
    t = h + v
    return 100.0 * h / t if t else float("nan")


def read_hallucination_totals(tag: str, eval_runs_dir: Path) -> tuple[float, dict[str, float]]:
    """Read a tag's result files and return overall and context-specific rates."""
    run_dir = eval_runs_dir / tag
    df = pd.read_csv(run_dir / "FINAL_RESULTS.csv", index_col=0)
    row = df.loc["Totals"]

    pip_h, pip_v = row.get("pip_hallucinated", 0), row.get("pip_valid", 0)
    h1, v1 = row.get("hallucinated_1", 0), row.get("valid_1", 0)
    h2, v2 = row.get("hallucinated_2", 0), row.get("valid_2", 0)

    import_h = import_v = 0
    import_path = run_dir / "import_scan_results.csv"
    if import_path.exists():
        idf = pd.read_csv(import_path)
        import_h = idf["import_hallucinated"].apply(safe_len).sum()
        import_v = idf["import_valid"].apply(safe_len).sum()

    halluc = pip_h + h1 + h2 + import_h
    valid = pip_v + v1 + v2 + import_v
    total = halluc + valid
    overall_rate = 100.0 * halluc / total if total else float("nan")

    context_rates = {
        CONTEXT_LABELS["install_cmd"]: rate(pip_h, pip_v),
        CONTEXT_LABELS["nl_query_1"]: rate(h1, v1),
        CONTEXT_LABELS["nl_query_2"]: rate(h2, v2),
        CONTEXT_LABELS["bare_import"]: rate(import_h, import_v),
    }
    return overall_rate, context_rates


def build_report(eval_runs_dir: Path, tags: list[tuple[str, str]] | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build overall and context-specific hallucination tables for available evaluation tags."""
    tags = tags if tags is not None else METHOD_TAGS
    tags = [(t, name) for (t, name) in tags if (eval_runs_dir / t).is_dir()]

    hall_rates: dict[str, float] = {}
    context_rows: list[dict[str, Any]] = []
    for tag, display_name in tags:
        try:
            rate, ctx = read_hallucination_totals(tag, eval_runs_dir)
        except Exception as e:
            print(f"[Warning] Could not read results for '{tag}': {e}")
            rate, ctx = float("nan"), {}
        hall_rates[tag] = rate
        context_rows.append({"Variant": display_name, **ctx})

    base_rate = hall_rates.get("base", float("nan"))
    hall_table = pd.DataFrame(
        {
            "Variant": [name for _, name in tags],
            "Hallucination Rate (%)": [hall_rates[t] for t, _ in tags],
            "Abs. Reduction": [(base_rate - hall_rates[t]) if t != "base" else 0.0 for t, _ in tags],
        }
    )
    context_table = pd.DataFrame(context_rows).round(2)
    return hall_table.round(2), context_table


def save_report(hall_table: pd.DataFrame, context_table: pd.DataFrame, out_dir: Path) -> tuple[str, str]:
    """Save both result tables as CSV files and return their paths."""
    out_dir.mkdir(parents=True, exist_ok=True)
    table1_path = out_dir / "table1_hallucination_rate.csv"
    context_path = out_dir / "table1_by_context.csv"
    hall_table.set_index("Variant").to_csv(table1_path)
    context_table.to_csv(context_path, index=False)
    return str(table1_path), str(context_path)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--eval-runs-dir", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--model-name", default=None, help="Model name shown in the report header")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    hall_table, context_table = build_report(args.eval_runs_dir)
    if args.model_name:
        print(f"Model: {args.model_name}\n")
    print("Table 1 -- Package Hallucination Rate:")
    print(hall_table.set_index("Variant").to_string())
    print("\nBy context:")
    print(context_table.to_string(index=False))
    table1_path, context_path = save_report(hall_table, context_table, args.out_dir)
    print(f"\nSaved: {table1_path}\nSaved: {context_path}")


if __name__ == "__main__":
    main()
