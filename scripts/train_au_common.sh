# dùng chung của train_ga.sh và train_npo.sh
set -euo pipefail

LOSS_FUNCTION="$1"; shift
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"

MODEL="Qwen/Qwen2.5-Coder-0.5B-Instruct"
MODEL_PATH=""
SAVE_TAG="$LOSS_FUNCTION"
LR="1e-5"
EPOCHS="3"
SEED="42"
DTYPE="auto"   # auto = bfloat16 nếu GPU hỗ trợ, không thì float16 (T4/P100 -> float16)
USE_LORA="false"
LORA_RANK="16"
RESUME_FROM=""
OUT_DIR=""
CONFIG_FILE=""

usage() {
    cat <<USAGE
Cách dùng: train_${LOSS_FUNCTION}.sh [tuỳ chọn]
  --config FILE                 File config JSON (vd configs/default.json)
                                
  --model TEN_HOAC_DUONG_DAN     Id HF hoặc tên preset, phải tải sẵn
                                  trong models/ (mặc định: $MODEL)
  --model-path DUONG_DAN        
  --save-tag TAG                  Tên thư mục checkpoint trong checkpoints/ (mặc định: $LOSS_FUNCTION)
  --lr FLOAT                       Learning rate (mặc định: $LR)
  --epochs INT                      Số epoch train (mặc định: $EPOCHS)
  --seed INT                         Random seed (mặc định: $SEED)
  --dtype auto|bfloat16|float16       dtype (mặc định: $DTYPE)
  --use-lora                           Train LoRA adapter thay vì full fine-tune
  --lora-rank INT                       LoRA rank (mặc định: $LORA_RANK; chỉ dùng khi có --use-lora)
  --resume-from DUONG_DAN                 Resume trọng số full-parameter từ 1 checkpoint có sẵn
  --out-dir DUONG_DAN                       Ghi đè thư mục output mặc định checkpoints/<model>_<tag>
  -h, --help

USAGE
    exit "${1:-0}"
}

# Pass 1: tìm --config làm mặc định 
_args=("$@")
for ((_i = 0; _i < ${#_args[@]}; _i++)); do
    if [ "${_args[$_i]}" = "--config" ]; then
        CONFIG_FILE="${_args[$((_i + 1))]}"
        break
    fi
done

if [ -n "$CONFIG_FILE" ]; then
    load_config "$CONFIG_FILE" "" "methods.$LOSS_FUNCTION"
    [ -n "${CFG_MODEL_NAME:-}" ] && MODEL="$CFG_MODEL_NAME"
    [ -n "${CFG_SEED:-}" ] && SEED="$CFG_SEED"
    [ -n "${CFG_DTYPE:-}" ] && DTYPE="$CFG_DTYPE"
    [ -n "${CFG_LR:-}" ] && LR="$CFG_LR"
    [ -n "${CFG_NUM_TRAIN_EPOCHS:-}" ] && EPOCHS="$CFG_NUM_TRAIN_EPOCHS"
    [ "${CFG_USE_LORA:-}" = "true" ] && USE_LORA="true"
    [ -n "${CFG_LORA_RANK:-}" ] && LORA_RANK="$CFG_LORA_RANK"
    echo "[train_$LOSS_FUNCTION] da nap config: $CONFIG_FILE (methods.$LOSS_FUNCTION)"
fi

# Pass 2: xử lý ghi đè giá trị từ config
while [ $# -gt 0 ]; do
    case "$1" in
        --config) shift 2 ;; 
        --model) MODEL="$2"; shift 2 ;;
        --model-path) MODEL_PATH="$2"; shift 2 ;;
        --save-tag) SAVE_TAG="$2"; shift 2 ;;
        --lr) LR="$2"; shift 2 ;;
        --epochs) EPOCHS="$2"; shift 2 ;;
        --seed) SEED="$2"; shift 2 ;;
        --dtype) DTYPE="$2"; shift 2 ;;
        --use-lora) USE_LORA="true"; shift ;;
        --lora-rank) LORA_RANK="$2"; shift 2 ;;
        --resume-from) RESUME_FROM="$2"; shift 2 ;;
        --out-dir) OUT_DIR="$2"; shift 2 ;;
        -h|--help) usage 0 ;;
        *) echo "tuỳ chọn không rõ: $1" >&2; usage 1 ;;
    esac
done

# Chuẩn hoá về id HF đầy đủ
MODEL="$(resolve_model_name "$MODEL")"

if [ -z "$MODEL_PATH" ]; then
    MODEL_PATH="$(resolve_model_path "$MODEL")"
fi

SUFFIX="$(au_train_suffix "$MODEL")"
RETAIN_FILE="$TRI_MASK_DIR/npo_retain_tok${SUFFIX}.jsonl"
FORGET_FILE="$TRI_MASK_DIR/npo_forget_tok${SUFFIX}.jsonl"
if [ ! -f "$RETAIN_FILE" ] || [ ! -f "$FORGET_FILE" ]; then
    echo "loi: chua co du lieu tri-mask cho model nay ($RETAIN_FILE)" >&2
    echo "  Chay 'bash scripts/build_data.sh --model $MODEL' truoc." >&2
    exit 1
fi

[ -z "$OUT_DIR" ] && OUT_DIR="$CHECKPOINTS_DIR/$(basename "$MODEL")_${SAVE_TAG}"

ARGS=(
    -m pkg_halluc.training.au.train_au
    --model-path "$MODEL_PATH"
    --retain-file "$RETAIN_FILE"
    --forget-file "$FORGET_FILE"
    --output-dir "$OUT_DIR"
    --cache-dir "$MODELS_DIR/cache"
    --loss_function "$LOSS_FUNCTION"
    --save_string "$SAVE_TAG"
    --lr "$LR"
    --num_train_epochs "$EPOCHS"
    --seed "$SEED"
    --dtype "$DTYPE"
)
[ "$USE_LORA" = "true" ] && ARGS+=(--use_lora --lora_rank "$LORA_RANK")
[ -n "$RESUME_FROM" ] && ARGS+=(--resume_from_checkpoint "$RESUME_FROM")

echo "[train_$LOSS_FUNCTION] \$ python ${ARGS[*]}   (cwd=$REPO_ROOT)"
cd "$REPO_ROOT"
python "${ARGS[@]}"

echo "[train_$LOSS_FUNCTION] xong -> $OUT_DIR"
