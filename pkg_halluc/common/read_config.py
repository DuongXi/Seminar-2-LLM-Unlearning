"""Đọc file config JSON và in ra các biến shell CFG_<KEY> (scripts/common.sh dùng)"""
from __future__ import annotations

import argparse
import json
import shlex
import sys


def get_section(data: dict, dotted_path: str) -> dict:
    """Get child object"""
    node = data
    if dotted_path:
        for part in dotted_path.split("."):
            if not isinstance(node, dict) or part not in node:
                return {}
            node = node[part]
    return node if isinstance(node, dict) else {}


def to_shell_value(value):
    """Set JSON value to shell string"""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float, str)):
        return str(value)
    if isinstance(value, list) and all(isinstance(v, (int, float, str)) for v in value):
        return " ".join(str(v) for v in value)
    return None  # omit nested dict or complex list


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--file", required=True, help="Path to JSON config file")
    ap.add_argument(
        "--section", action="append", default=None,
        help='Path to a child object, e.g. "methods.ga" or "eval". '
        'Can be duplicatd; subsequent section overwrites the preceding one if the keys overlap. '
        'Leave blank or omit to retrieve all keys at the root.',
    )
    args = ap.parse_args()

    with open(args.file, "r", encoding="utf-8") as f:
        data = json.load(f)

    merged: dict = dict(get_section(data, ""))
    if args.section:
        for sec in args.section:
            if sec:
                merged.update(get_section(data, sec))

    for key, value in merged.items():
        if key.startswith("_"):
            continue
        shell_value = to_shell_value(value)
        if shell_value is None:
            continue
        var_name = "CFG_" + key.upper()
        print(f"{var_name}={shlex.quote(shell_value)}")


if __name__ == "__main__":
    main()
