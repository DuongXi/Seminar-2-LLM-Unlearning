# dùng chung của train_ga_plain.sh và train_npo_plain.sh, cú pháp: train_plain_common.sh <ga_plain|npo_plain> [tuỳ chọn]
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
DTYPE="auto"
LAMBDA_RETAIN="1.0"
LAMBDA_FORGET="0.5"
MAX_LENGTH="2048"
VAL_RATIO="0.1"
EVAL_STEPS="25"
EARLY_STOP_PATIENCE="3"
EARLY_STOP_THRESHOLD="0.0"
DISABLE_EARLY_STOPPING="false"
USE_LORA="false"
LORA_RANK="16"
MAX_SAMPLES_PER_SPLIT=""
OUT_DIR=""
CONFIG_FILE=""

usage() {
    cat <<USAGE
Cách dùng: train_${LOSS_FUNCTION}.sh [tuỳ chọn]
  --config FILE                      File config JSON (vd model_config/default.json) 
                     
  --model TEN_HOAC_DUONG_DAN         Id HF hoặc tên preset (xem pkg_halluc/common/model_presets.py), phải tải sẵn
                                       trong models/ (mặc định: $MODEL)
  --model-path DUONG_DAN            
  --save-tag TAG                      Tên thư mục checkpoint trong checkpoints/ (mặc định: $LOSS_FUNCTION)
  --lr FLOAT                           Learning rate (mặc định: $LR)
  --epochs INT                          Số epoch train (mặc định: $EPOCHS)
  --seed INT                             Random seed (mặc định: $SEED)
  --dtype auto|bfloat16|float16|float32   (mặc định: $DTYPE)
  --lambda-retain FLOAT                    (mặc định: $LAMBDA_RETAIN)
  --lambda-forget FLOAT                    (mặc định: $LAMBDA_FORGET)
  --max-length INT                          Độ dài token tối đa (mặc định: $MAX_LENGTH)
  --val-ratio FLOAT                          Tỉ lệ tách từ retain để early stopping (mặc định: $VAL_RATIO)
  --eval-steps INT                            (mặc định: $EVAL_STEPS)
  --early-stopping-patience INT                (mặc định: $EARLY_STOP_PATIENCE)
  --early-stopping-threshold FLOAT              (mặc định: $EARLY_STOP_THRESHOLD)
  --disable-early-stopping                       Train đủ num_train_epochs, không tách eval split
  --use-lora                                      Train LoRA adapter thay vì full fine-tune
  --lora-rank INT                                  LoRA rank (mặc định: $LORA_RANK; chỉ dùng khi có --use-lora)
  --max-samples-per-split INT                      Giới hạn số dòng retain/forget (for fast test)
  --out-dir DUONG_DAN                               Ghi đè thư mục output mặc định checkpoints/<model>_<tag>
  -h, --help

USAGE
    exit "${1:-0}"
}

MAIN_PATH=""

# Pass 1: tìm --config để làm mặc định trước
_args=("$@")
for ((_i = 0; _i < ${#_args[@]}; _i++)); do
    if [ "${_args[$_i]}" = "--config" ]; then
        CONFIG_FILE="${_args[$((_i + 1))]}"
        break
    fi
done

if [ -n "$CONFIG_FILE" ]; then
    load_config "$CONFIG_FILE" "" "data" "methods.$LOSS_FUNCTION"
    [ -n "${CFG_MODEL_NAME:-}" ] && MODEL="$CFG_MODEL_NAME"
    [ -n "${CFG_MAIN_PATH:-}" ] && MAIN_PATH="$CFG_MAIN_PATH"
    [ -n "${CFG_SEED:-}" ] && SEED="$CFG_SEED"
    [ -n "${CFG_DTYPE:-}" ] && DTYPE="$CFG_DTYPE"
    [ -n "${CFG_LR:-}" ] && LR="$CFG_LR"
    [ -n "${CFG_NUM_TRAIN_EPOCHS:-}" ] && EPOCHS="$CFG_NUM_TRAIN_EPOCHS"
    [ -n "${CFG_LAMBDA_RETAIN:-}" ] && LAMBDA_RETAIN="$CFG_LAMBDA_RETAIN"
    [ -n "${CFG_LAMBDA_FORGET:-}" ] && LAMBDA_FORGET="$CFG_LAMBDA_FORGET"
    [ -n "${CFG_VAL_RATIO:-}" ] && VAL_RATIO="$CFG_VAL_RATIO"
    [ -n "${CFG_EVAL_STEPS:-}" ] && EVAL_STEPS="$CFG_EVAL_STEPS"
    [ -n "${CFG_EARLY_STOPPING_PATIENCE:-}" ] && EARLY_STOP_PATIENCE="$CFG_EARLY_STOPPING_PATIENCE"
    [ -n "${CFG_EARLY_STOPPING_THRESHOLD:-}" ] && EARLY_STOP_THRESHOLD="$CFG_EARLY_STOPPING_THRESHOLD"
    [ "${CFG_USE_LORA:-}" = "true" ] && USE_LORA="true"
    [ -n "${CFG_LORA_RANK:-}" ] && LORA_RANK="$CFG_LORA_RANK"
    [ -n "${CFG_MAX_TRAIN_SAMPLES_PER_SPLIT:-}" ] && MAX_SAMPLES_PER_SPLIT="$CFG_MAX_TRAIN_SAMPLES_PER_SPLIT"
    echo "[train_$LOSS_FUNCTION] da nap config: $CONFIG_FILE (methods.$LOSS_FUNCTION)"
fi

# Pass 2: xử lý ghi đè giá trị từ config
while [ $# -gt 0 ]; do
    case "$1" in
        --config) shift 2 ;;  # đã xử lý ở Pass 1
        --model) MODEL="$2"; shift 2 ;;
        --model-path|--model_path) MODEL_PATH="$2"; shift 2 ;;
        --main-path|--main_path) MAIN_PATH="$2"; shift 2 ;;
        --save-tag) SAVE_TAG="$2"; shift 2 ;;
        --lr) LR="$2"; shift 2 ;;
        --epochs) EPOCHS="$2"; shift 2 ;;
        --seed) SEED="$2"; shift 2 ;;
        --dtype) DTYPE="$2"; shift 2 ;;
        --lambda-retain) LAMBDA_RETAIN="$2"; shift 2 ;;
        --lambda-forget) LAMBDA_FORGET="$2"; shift 2 ;;
        --max-length) MAX_LENGTH="$2"; shift 2 ;;
        --val-ratio) VAL_RATIO="$2"; shift 2 ;;
        --eval-steps) EVAL_STEPS="$2"; shift 2 ;;
        --early-stopping-patience) EARLY_STOP_PATIENCE="$2"; shift 2 ;;
        --early-stopping-threshold) EARLY_STOP_THRESHOLD="$2"; shift 2 ;;
        --disable-early-stopping) DISABLE_EARLY_STOPPING="true"; shift ;;
        --use-lora) USE_LORA="true"; shift ;;
        --lora-rank) LORA_RANK="$2"; shift 2 ;;
        --max-samples-per-split) MAX_SAMPLES_PER_SPLIT="$2"; shift 2 ;;
        --out-dir) OUT_DIR="$2"; shift 2 ;;
        -h|--help) usage 0 ;;
        *) echo "tuỳ chọn không rõ: $1" >&2; usage 1 ;;
    esac
done

MODEL="$(resolve_model_name "$MODEL")"

if [ -z "$MODEL_PATH" ]; then
    MODEL_PATH="$(resolve_model_path "$MODEL")"
fi

mapfile -t RESULT_FILES < <(resolve_result_files)
if [ -n "$MAIN_PATH" ]; then
    MAIN_PATH="${MAIN_PATH%/}"
    [[ "$MAIN_PATH" != /* && "$MAIN_PATH" != [A-Za-z]:* ]] && MAIN_PATH="$REPO_ROOT/$MAIN_PATH"
    RESULT_FILES=(
        "$MAIN_PATH/LLM_LY_results.csv"
        "$MAIN_PATH/LLM_AT_results.csv"
        "$MAIN_PATH/SO_LY_results.csv"
        "$MAIN_PATH/SO_AT_results.csv"
    )
else
    RESULT_FILES=(
        "$REPO_ROOT/data/Llama3_3_Python/LLM_LY_results.csv"
        "$REPO_ROOT/data/Llama3_3_Python/LLM_AT_results.csv"
        "$REPO_ROOT/data/Llama3_3_Python/SO_LY_results.csv"
        "$REPO_ROOT/data/Llama3_3_Python/SO_AT_results.csv"
    )
fi
[ -z "$OUT_DIR" ] && OUT_DIR="$CHECKPOINTS_DIR/$(basename "$MODEL")_${SAVE_TAG}"

ARGS=(
    -m pkg_halluc.training.plain.train_plain
    --model_path "$MODEL_PATH"
    --model_name "$MODEL"
    --loss_function "$LOSS_FUNCTION"
    --save_tag "$SAVE_TAG"
    --lr "$LR"
    --num_train_epochs "$EPOCHS"
    --lambda_retain "$LAMBDA_RETAIN"
    --lambda_forget "$LAMBDA_FORGET"
    --seed "$SEED"
    --dtype "$DTYPE"
    --max_length "$MAX_LENGTH"
    --result_files "${RESULT_FILES[@]}"
    --out_dir "$OUT_DIR"
    --val_ratio "$VAL_RATIO"
    --eval_steps "$EVAL_STEPS"
    --early_stopping_patience "$EARLY_STOP_PATIENCE"
    --early_stopping_threshold "$EARLY_STOP_THRESHOLD"
)
[ "$DISABLE_EARLY_STOPPING" = "true" ] && ARGS+=(--disable_early_stopping)
[ "$USE_LORA" = "true" ] && ARGS+=(--use_lora --lora_rank "$LORA_RANK")
[ -n "$MAX_SAMPLES_PER_SPLIT" ] && ARGS+=(--max_samples_per_split "$MAX_SAMPLES_PER_SPLIT")

echo "[train_$LOSS_FUNCTION] \$ python ${ARGS[*]}   (cwd=$REPO_ROOT)"
cd "$REPO_ROOT"
"$PYTHON_BIN" "${ARGS[@]}"

echo "[train_$LOSS_FUNCTION] xong -> $OUT_DIR"
