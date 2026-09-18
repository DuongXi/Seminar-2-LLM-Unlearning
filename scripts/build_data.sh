# Dựng dữ liệu tri-mask retain/forget cho GA/NPO (pkg_halluc/package_loader/build_tri_mask_data.py), cần tải model trước
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"

MODEL="Qwen/Qwen2.5-Coder-0.5B-Instruct"
DTYPE="auto"
SEED="42"
MAX_LENGTH="2048"
MAX_SAMPLES_PER_SPLIT=""
CONFIG_FILE=""

usage() {
    cat <<USAGE
Cách dùng: build_data.sh [tuỳ chọn]
  --config FILE                    File config JSON (vd configs/default.json) 
  --model TEN_HOAC_ID_HF          Model dùng để sinh code (đã tải sẵn) (mặc định: $MODEL)
  --dtype auto|bfloat16|float16    (mặc định: $DTYPE)
  --seed INT                        (mặc định: $SEED)
  --max-length INT                   (mặc định: $MAX_LENGTH)
  --max-train-samples-per-split INT   Giới hạn số dòng retain/forget (chạy thử nhanh)
  -h, --help
USAGE
    exit "${1:-0}"
}

# Pass 1: tìm --config làm mặc định trước
_args=("$@")
for ((_i = 0; _i < ${#_args[@]}; _i++)); do
    if [ "${_args[$_i]}" = "--config" ]; then
        CONFIG_FILE="${_args[$((_i + 1))]}"
        break
    fi
done

if [ -n "$CONFIG_FILE" ]; then
    load_config "$CONFIG_FILE" "" "data"
    [ -n "${CFG_MODEL_NAME:-}" ] && MODEL="$CFG_MODEL_NAME"
    [ -n "${CFG_SEED:-}" ] && SEED="$CFG_SEED"
    [ -n "${CFG_DTYPE:-}" ] && DTYPE="$CFG_DTYPE"
    [ -n "${CFG_MAX_TRAIN_SAMPLES_PER_SPLIT:-}" ] && MAX_SAMPLES_PER_SPLIT="$CFG_MAX_TRAIN_SAMPLES_PER_SPLIT"
    echo "[build-data] da nap config: $CONFIG_FILE"
fi

# Pass 2: xử lý ghi đè giá trị từ config
while [ $# -gt 0 ]; do
    case "$1" in
        --config) shift 2 ;;
        --model) MODEL="$2"; shift 2 ;;
        --dtype) DTYPE="$2"; shift 2 ;;
        --seed) SEED="$2"; shift 2 ;;
        --max-length) MAX_LENGTH="$2"; shift 2 ;;
        --max-train-samples-per-split) MAX_SAMPLES_PER_SPLIT="$2"; shift 2 ;;
        -h|--help) usage 0 ;;
        *) echo "tuỳ chọn không rõ: $1" >&2; usage 1 ;;
    esac
done

MODEL="$(resolve_model_name "$MODEL")"
SUFFIX="$(au_train_suffix "$MODEL")"
RETAIN_FILE="$TRI_MASK_DIR/npo_retain_tok${SUFFIX}.jsonl"
FORGET_FILE="$TRI_MASK_DIR/npo_forget_tok${SUFFIX}.jsonl"

ARGS=(
    -m pkg_halluc.package_loader.build_tri_mask_data
    --model "$MODEL"
    --dtype "$DTYPE"
    --seed "$SEED"
    --max-length "$MAX_LENGTH"
    --retain-file "$RETAIN_FILE"
    --forget-file "$FORGET_FILE"
)
[ -n "$MAX_SAMPLES_PER_SPLIT" ] && ARGS+=(--max-train-samples-per-split "$MAX_SAMPLES_PER_SPLIT")

echo "[build-data] \$ python ${ARGS[*]}"
python "${ARGS[@]}"
