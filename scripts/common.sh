# Shared settings
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODELS_DIR="$REPO_ROOT/models"
CHECKPOINTS_DIR="$REPO_ROOT/checkpoints"
EVAL_RUNS_DIR="$REPO_ROOT/eval_runs"
OUTPUTS_DIR="$REPO_ROOT/outputs"

export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"

if [ -z "${PYTHON_BIN:-}" ]; then
    if command -v python3 >/dev/null 2>&1; then
        PYTHON_BIN="python3"
    else
        PYTHON_BIN="python"
    fi
fi

resolve_model_name() {
    "$PYTHON_BIN" -c "
import sys
from pkg_halluc.common.model_presets import resolve_model_name
print(resolve_model_name(sys.argv[1]))
" "$1"
}

resolve_model_suffix() {
    local model="${1:-}"
    local explicit="${2:-}"
    "$PYTHON_BIN" -c "
import sys
from pkg_halluc.common.model_presets import resolve_model_suffix
model = sys.argv[1] if len(sys.argv) > 1 else ''
explicit = sys.argv[2] if len(sys.argv) > 2 and sys.argv[2] else None
print(resolve_model_suffix(model, explicit))
" "$model" "$explicit"
}


resolve_model_path() {
    local model="$1"
    if [ -d "$model" ]; then
        echo "$model"
        return
    fi
    local resolved
    resolved="$(resolve_model_name "$model")"
    if [ -d "$MODELS_DIR/$resolved" ] && [ -n "$(ls -A "$MODELS_DIR/$resolved" 2>/dev/null)" ]; then
        echo "$MODELS_DIR/$resolved"
    elif [ -d "$MODELS_DIR/$model" ] && [ -n "$(ls -A "$MODELS_DIR/$model" 2>/dev/null)" ]; then
        echo "$MODELS_DIR/$model"
    else
        echo "$resolved"
    fi
}

load_config() {
    local file="$1"; shift
    local section_args=()
    local s
    for s in "$@"; do
        [ -n "$s" ] && section_args+=(--section "$s")
    done
    eval "$("$PYTHON_BIN" -m pkg_halluc.common.read_config --file "$file" "${section_args[@]}")"
}

mkdir -p "$MODELS_DIR" "$CHECKPOINTS_DIR" "$EVAL_RUNS_DIR" "$OUTPUTS_DIR"
