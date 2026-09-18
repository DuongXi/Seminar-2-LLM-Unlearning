# Thiết lập dùng chung
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODELS_DIR="$REPO_ROOT/models"
CHECKPOINTS_DIR="$REPO_ROOT/checkpoints"
WORK_DIR="${PKG_HALLUC_WORK_DIR:-$REPO_ROOT/.workdir}"
EVAL_RUNS_DIR="$WORK_DIR/eval_runs"
OUTPUTS_DIR="$WORK_DIR/outputs"
TRI_MASK_DIR="$WORK_DIR/tri_mask_data"  # dữ liệu retain/forget cho GA/NPO

export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"

resolve_model_name() {
    python3 -c "
import sys
from pkg_halluc.common.model_presets import resolve_model_name
print(resolve_model_name(sys.argv[1]))
" "$1"
}

resolve_model_path() {
    local model="$1"
    if [ -d "$model" ]; then
        echo "$model"
        return
    fi
    local resolved
    resolved="$(resolve_model_name "$model")"
    local local_dir="$MODELS_DIR/$resolved"
    if [ ! -d "$local_dir" ] || [ -z "$(ls -A "$local_dir" 2>/dev/null)" ]; then
        echo "loi: khong tim thay base model tai $local_dir" >&2
        echo "  Chay 'bash scripts/download_model.sh --model $model' truoc, hoac" >&2
        echo "  truyen --model-path tro thang vao 1 ban da tai san." >&2
        exit 1
    fi
    echo "$local_dir"
}

load_config() {
    local file="$1"; shift
    local section_args=()
    local s
    for s in "$@"; do
        section_args+=(--section "$s")
    done
    eval "$(python3 -m pkg_halluc.common.read_config --file "$file" "${section_args[@]}")"
}

au_train_suffix() {
    local model="$1"
    if [[ "$model" == *"7b"* ]]; then
        echo "_7b"
    elif [[ "$model" == *"1.3b"* ]]; then
        echo ""
    else
        echo "_16B"
    fi
}

resolve_result_files() {
    local names=(LLM_LY_results.csv LLM_AT_results.csv SO_LY_results.csv SO_AT_results.csv)
    for fn in "${names[@]}"; do
        if [ -f "$REPO_ROOT/data/Llama3_3_Python/$fn" ]; then
            echo "$REPO_ROOT/data/Llama3_3_Python/$fn"
        else
            echo "$REPO_ROOT/data/$fn"
        fi
    done
}

mkdir -p "$MODELS_DIR" "$CHECKPOINTS_DIR" "$WORK_DIR" "$EVAL_RUNS_DIR" "$OUTPUTS_DIR" "$TRI_MASK_DIR"
