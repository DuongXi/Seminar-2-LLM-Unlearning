#!/usr/bin/env bash
# Convenience wrapper around the staged `pkg_halluc` subcommands -- equivalent
# to `pkg_halluc run-all`, spelled out one stage per line so you can see (and
# comment out / re-run individually) each step. Prefer this over `run-all`
# once you're past your first end-to-end run and want to re-run only part of
# the pipeline -- see README.md "Usage" for that.
#
# Usage:
#   ./scripts/quickstart.sh                        # configs/smoke_test.json, tiny/fast
#   ./scripts/quickstart.sh configs/default.json    # a real run
#   AU_SRC=/path/to/Adaptive-Unlearning-952E.zip ./scripts/quickstart.sh
set -euo pipefail

CONFIG="${1:-configs/smoke_test.json}"
AU_SRC="${AU_SRC:-}"

FETCH_ARGS=(--config "$CONFIG")
[ -n "$AU_SRC" ] && FETCH_ARGS+=(--au-src "$AU_SRC")

echo "== 1/6 fetch-deps =="
pkg_halluc fetch-deps "${FETCH_ARGS[@]}"

echo "== 2/6 download-model =="
pkg_halluc download-model --config "$CONFIG"

echo "== 3/6 build-data =="
pkg_halluc build-data --config "$CONFIG"

echo "== 4/6 train (all enabled weight_finetune methods) =="
pkg_halluc train --config "$CONFIG" --method all

echo "== 5/6 evaluate (all enabled methods) =="
pkg_halluc evaluate --config "$CONFIG" --method all

echo "== 6/6 report =="
pkg_halluc report --config "$CONFIG"
