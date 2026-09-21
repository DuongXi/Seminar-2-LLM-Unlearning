# Gộp eval_runs/* thành 2 bảng kết quả cuối (pkg_halluc/evaluation/report_build.py), xem --help
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"

MODEL_NAME=""
CONFIG_FILE=""

usage() {
    cat <<USAGE
Cách dùng: report.sh [tuỳ chọn]
  --config FILE       File config JSON (vd model_config/default.json)
  --model-name TEN   
  -h, --help
USAGE
    exit "${1:-0}"
}

# Pass 1:  tìm --config
_args=("$@")
for ((_i = 0; _i < ${#_args[@]}; _i++)); do
    if [ "${_args[$_i]}" = "--config" ]; then
        CONFIG_FILE="${_args[$((_i + 1))]}"
        break
    fi
done

if [ -n "$CONFIG_FILE" ]; then
    load_config "$CONFIG_FILE" ""
    [ -n "${CFG_MODEL_NAME:-}" ] && MODEL_NAME="$CFG_MODEL_NAME"
fi

while [ $# -gt 0 ]; do
    case "$1" in
        --config) shift 2 ;; 
        --model-name) MODEL_NAME="$2"; shift 2 ;;
        -h|--help) usage 0 ;;
        *) echo "tuỳ chọn không rõ: $1" >&2; usage 1 ;;
    esac
done

ARGS=(
    -m pkg_halluc.evaluation.report_build
    --eval-runs-dir "$EVAL_RUNS_DIR"
    --out-dir "$OUTPUTS_DIR"
)
[ -n "$MODEL_NAME" ] && ARGS+=(--model-name "$MODEL_NAME")

echo "[report] \$ python ${ARGS[*]}"
"$PYTHON_BIN" "${ARGS[@]}"
