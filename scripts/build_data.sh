#!/usr/bin/env bash
# To generate tri-mask retain/forget data for GA/NPO, the model must be loaded first
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"

MODEL="Qwen/Qwen2.5-Coder-0.5B-Instruct"
MODEL_PATH=""
DTYPE="auto"
SEED="42"
DEVICE_MAP="auto"
MAX_LENGTH="2048"
VAL_RATIO="0.0"
MAX_SAMPLES_PER_SPLIT=""
CONFIG_FILE=""
MAIN_PATH=""
RESULT_FILES=()

OUT_MASTER=""
OUT_VAL_MASTER=""
OUT_RETAIN_TRI_MASK=""
OUT_FORGET_TRI_MASK=""
OUT_VAL_RETAIN_TRI_MASK=""
OUT_PLAIN_TRAIN=""
OUT_PLAIN_VAL=""
OUT_PLAIN_TEST=""
OUT_PLAIN_RETAIN=""
OUT_PLAIN_FORGET=""

usage() {
    cat <<USAGE
Cách dùng: build_data.sh [tuỳ chọn]
  --config FILE                       File config JSON
  --model TEN_HOAC_ID_HF              Id HF hoặc preset model (default: $MODEL)
  --model-path DUONG_DAN              Path to model (override with --model if any)
  --dtype auto|bfloat16|float16|float32 (default: $DTYPE)
  --seed INT                          Random seed (default: $SEED)
  --device-map STR                    Device map (default: $DEVICE_MAP)
  --max-length INT                    max length token (default: $MAX_LENGTH)
  --val-ratio FLOAT                   Set the split ratio for the validation set (default: $VAL_RATIO)
  --max-train-samples-per-split INT   Limit the number of samples per split for smoke tests
  --result-files FILE [FILE ...]      Result CSV (default: auto-detect)
  --out-master FILE                   Pre-tri-mask master dataset pipeline/reuse
  --out-val-master FILE               Saving path for master val dataset
  --out-retain-tri-mask FILE          File JSONL retain tri-mask (alias: --retain-file)
  --out-forget-tri-mask FILE          File JSONL forget tri-mask (alias: --forget-file)
  --out-val-retain-tri-mask FILE      File JSONL val retain tri-mask
  --out-plain-train FILE              File tokenized train records (pre-tri-mask)
  --out-plain-val FILE                File tokenized val records
  --out-plain-test FILE               File tokenized test records
  --out-plain-retain FILE             File tokenized retain records
  --out-plain-forget FILE             File tokenized forget records
  -h, --help
USAGE
    exit "${1:-0}"
}

# Pass 1: Look for `--config` as default
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
    [ -n "${CFG_MAX_LENGTH:-}" ] && MAX_LENGTH="$CFG_MAX_LENGTH"
    [ -n "${CFG_VAL_RATIO:-}" ] && VAL_RATIO="$CFG_VAL_RATIO"
    [ -n "${CFG_MAIN_PATH:-}" ] && MAIN_PATH="$CFG_MAIN_PATH"
    echo "Config set: $CONFIG_FILE"
fi

# Pass 2: handle value overrides from the config
while [ $# -gt 0 ]; do
    case "$1" in
        --config) shift 2 ;;
        --model) MODEL="$2"; shift 2 ;;
        --model-path|--model_path) MODEL_PATH="$2"; shift 2 ;;
        --main-path|--main_path) MAIN_PATH="$2"; shift 2 ;;
        --dtype) DTYPE="$2"; shift 2 ;;
        --seed) SEED="$2"; shift 2 ;;
        --device-map|--device_map) DEVICE_MAP="$2"; shift 2 ;;
        --max-length|--max_length) MAX_LENGTH="$2"; shift 2 ;;
        --val-ratio|--val_ratio) VAL_RATIO="$2"; shift 2 ;;
        --max-train-samples-per-split|--max_train_samples_per_split) MAX_SAMPLES_PER_SPLIT="$2"; shift 2 ;;
        --out-master|--out_master) OUT_MASTER="$2"; shift 2 ;;
        --out-val-master|--out_val_master) OUT_VAL_MASTER="$2"; shift 2 ;;
        --out-retain-tri-mask|--out_retain_tri_mask|--retain-file) OUT_RETAIN_TRI_MASK="$2"; shift 2 ;;
        --out-forget-tri-mask|--out_forget_tri_mask|--forget-file) OUT_FORGET_TRI_MASK="$2"; shift 2 ;;
        --out-val-retain-tri-mask|--out_val_retain_tri_mask) OUT_VAL_RETAIN_TRI_MASK="$2"; shift 2 ;;
        --out-plain-train|--out_plain_train) OUT_PLAIN_TRAIN="$2"; shift 2 ;;
        --out-plain-val|--out_plain_val) OUT_PLAIN_VAL="$2"; shift 2 ;;
        --out-plain-test|--out_plain_test) OUT_PLAIN_TEST="$2"; shift 2 ;;
        --out-plain-retain|--out_plain_retain) OUT_PLAIN_RETAIN="$2"; shift 2 ;;
        --out-plain-forget|--out_plain_forget) OUT_PLAIN_FORGET="$2"; shift 2 ;;
        --result-files|--result_files)
            shift
            while [ $# -gt 0 ] && [[ "$1" != --* ]]; do
                RESULT_FILES+=("$1")
                shift
            done
            ;;
        -h|--help) usage 0 ;;
        *) echo "Unknown option: $1" >&2; usage 1 ;;
    esac
done

MODEL="$(resolve_model_name "$MODEL")"

if [ -z "$MODEL_PATH" ]; then
    if [ -d "$MODELS_DIR/$MODEL" ] && [ -n "$(ls -A "$MODELS_DIR/$MODEL" 2>/dev/null)" ]; then
        MODEL_PATH="$MODELS_DIR/$MODEL"
    elif [ -d "$MODEL" ]; then
        MODEL_PATH="$MODEL"
    else
        MODEL_PATH="$MODEL"
    fi
fi

MODEL_SUFFIX="$(resolve_model_suffix "$MODEL")"
[ -z "$MODEL_SUFFIX" ] && MODEL_SUFFIX="$(resolve_model_suffix "$MODEL_PATH")"

MAIN_PATH="${MAIN_PATH%/}"
[[ "$MAIN_PATH" != /* && "$MAIN_PATH" != [A-Za-z]:* ]] && MAIN_PATH="$REPO_ROOT/$MAIN_PATH"
[ -z "$OUT_RETAIN_TRI_MASK" ] && OUT_RETAIN_TRI_MASK="$MAIN_PATH/tri_mask/npo_retain_tok${MODEL_SUFFIX}.jsonl"
[ -z "$OUT_FORGET_TRI_MASK" ] && OUT_FORGET_TRI_MASK="$MAIN_PATH/tri_mask/npo_forget_tok${MODEL_SUFFIX}.jsonl"
if [ -n "$VAL_RATIO" ] && [ "$VAL_RATIO" != "0" ] && [ "$VAL_RATIO" != "0.0" ]; then
    [ -z "$OUT_VAL_RETAIN_TRI_MASK" ] && OUT_VAL_RETAIN_TRI_MASK="$MAIN_PATH/tri_mask/npo_val_retain_tok${MODEL_SUFFIX}.jsonl"
fi
if [ ${#RESULT_FILES[@]} -eq 0 ]; then
    RESULT_FILES=(
        "$MAIN_PATH/LLM_LY_results.csv"
        "$MAIN_PATH/LLM_AT_results.csv"
        "$MAIN_PATH/SO_LY_results.csv"
        "$MAIN_PATH/SO_AT_results.csv"
    )
fi

ARGS=(
    -m pkg_halluc.package_loader.build_data
    --model_path "$MODEL_PATH"
    --dtype "$DTYPE"
    --seed "$SEED"
    --device_map "$DEVICE_MAP"
    --max_length "$MAX_LENGTH"
    --out_retain_tri_mask "$OUT_RETAIN_TRI_MASK"
    --out_forget_tri_mask "$OUT_FORGET_TRI_MASK"
)
[ -n "$MAIN_PATH" ] && ARGS+=(--main_path "$MAIN_PATH")

if [ -n "$VAL_RATIO" ] && [ "$VAL_RATIO" != "0" ] && [ "$VAL_RATIO" != "0.0" ]; then
    ARGS+=(--val_ratio "$VAL_RATIO")
    if [ -n "$OUT_VAL_RETAIN_TRI_MASK" ]; then
        ARGS+=(--out_val_retain_tri_mask "$OUT_VAL_RETAIN_TRI_MASK")
    fi
fi

[ -n "$OUT_MASTER" ] && ARGS+=(--out_master "$OUT_MASTER")
[ -n "$OUT_VAL_MASTER" ] && ARGS+=(--out_val_master "$OUT_VAL_MASTER")
[ -n "$MAX_SAMPLES_PER_SPLIT" ] && ARGS+=(--max_train_samples_per_split "$MAX_SAMPLES_PER_SPLIT")

[ -n "$OUT_PLAIN_TRAIN" ] && ARGS+=(--out_plain_train "$OUT_PLAIN_TRAIN")
[ -n "$OUT_PLAIN_VAL" ] && ARGS+=(--out_plain_val "$OUT_PLAIN_VAL")
[ -n "$OUT_PLAIN_TEST" ] && ARGS+=(--out_plain_test "$OUT_PLAIN_TEST")
[ -n "$OUT_PLAIN_RETAIN" ] && ARGS+=(--out_plain_retain "$OUT_PLAIN_RETAIN")
[ -n "$OUT_PLAIN_FORGET" ] && ARGS+=(--out_plain_forget "$OUT_PLAIN_FORGET")

if [ ${#RESULT_FILES[@]} -gt 0 ]; then
    ARGS+=(--result_files "${RESULT_FILES[@]}")
fi

echo "[build-data] \$ python ${ARGS[*]}   (cwd=$REPO_ROOT)"
cd "$REPO_ROOT"
"$PYTHON_BIN" "${ARGS[@]}"
