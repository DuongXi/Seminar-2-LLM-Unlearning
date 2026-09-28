# Download a base model from Hugging Face.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"

MODEL="Qwen/Qwen2.5-Coder-0.5B-Instruct"
DTYPE="auto"
SKIP_SANITY_CHECK="false"
CONFIG_FILE=""

usage() {
    cat <<USAGE
Usage: download_model.sh [options]
    --config FILE              JSON configuration file
    --model HF_ID_OR_PRESET    Preset (e.g. qwen2.5-coder-1.5b) or HF ID (default: $MODEL)
    --dtype auto|bfloat16|float16
    --skip-sanity-check        Skip the generation sanity check
  -h, --help
USAGE
    exit "${1:-0}"
}

# Read the config path before loading config-derived defaults.
_args=("$@")
for ((_i = 0; _i < ${#_args[@]}; _i++)); do
    if [ "${_args[$_i]}" = "--config" ]; then
        CONFIG_FILE="${_args[$((_i + 1))]}"
        break
    fi
done

if [ -n "$CONFIG_FILE" ]; then
    load_config "$CONFIG_FILE" ""
    [ -n "${CFG_MODEL_NAME:-}" ] && MODEL="$CFG_MODEL_NAME"
    [ -n "${CFG_DTYPE:-}" ] && DTYPE="$CFG_DTYPE"
    echo "[download-model] Loaded config: $CONFIG_FILE"
fi

# Apply command-line overrides.
while [ $# -gt 0 ]; do
    case "$1" in
        --config) shift 2 ;; 
        --model) MODEL="$2"; shift 2 ;;
        --dtype) DTYPE="$2"; shift 2 ;;
        --skip-sanity-check) SKIP_SANITY_CHECK="true"; shift ;;
        -h|--help) usage 0 ;;
        *) echo "Unknown option: $1" >&2; usage 1 ;;
    esac
done

ARGS=(
    -m pkg_halluc.common.download_model
    --model "$MODEL"
    --models-dir "$MODELS_DIR"
    --dtype "$DTYPE"
)
[ "$SKIP_SANITY_CHECK" = "true" ] && ARGS+=(--skip-sanity-check)

echo "[download-model] \$ python ${ARGS[*]}"
"$PYTHON_BIN" "${ARGS[@]}"
