#!/usr/bin/env bash
# Prepare unlearning datasets and optional CAA contrastive data.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"

MODEL="Qwen/Qwen2.5-Coder-0.5B-Instruct"
MODEL_PATH=""
SEED="42"
MAX_LENGTH="2048"
VAL_RATIO="0.0"
MAX_SAMPLES_PER_SPLIT=""
CONFIG_FILE=""
MAIN_PATH=""
RESULT_FILES=()
CONTRASTIVE="false"

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
Usage: build_data.sh [options]
    --config FILE                       JSON model configuration
    --model HF_ID_OR_PRESET             Hugging Face ID or model preset (default: $MODEL)
    --model-path PATH                   Model checkpoint path (overrides --model)
    --main-path PATH                    Model data directory (required unless set in config)
    --seed INT                          Random seed (default: $SEED)
    --max-length INT                    Maximum tokenized sequence length (default: $MAX_LENGTH)
    --val-ratio FLOAT                   Retain validation split ratio (default: $VAL_RATIO)
    --max-train-samples-per-split INT   Limit samples per split for smoke tests
    --result-files FILE [FILE ...]      Input result CSV files (default: auto-detect)
    --out-master FILE                   Pre-tri-mask master train dataset
    --out-val-master FILE               Master validation dataset output path
    --out-retain-tri-mask FILE          Retain tri-mask JSONL (alias: --retain-file)
    --out-forget-tri-mask FILE          Forget tri-mask JSONL (alias: --forget-file)
    --out-val-retain-tri-mask FILE      Validation retain tri-mask JSONL
    --out-plain-train FILE              Tokenized train records
    --out-plain-val FILE                Tokenized validation records
    --out-plain-test FILE               Tokenized test records
    --out-plain-retain FILE             Tokenized retain records
    --out-plain-forget FILE             Tokenized forget records
    --contrastive [BOOL]                Generate CAA contrastive data (default: $CONTRASTIVE)
  -h, --help
USAGE
    exit "${1:-0}"
}

# Read the config path before loading defaults from it.
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
    [ -n "${CFG_MAX_TRAIN_SAMPLES_PER_SPLIT:-}" ] && MAX_SAMPLES_PER_SPLIT="$CFG_MAX_TRAIN_SAMPLES_PER_SPLIT"
    [ -n "${CFG_MAX_LENGTH:-}" ] && MAX_LENGTH="$CFG_MAX_LENGTH"
    [ -n "${CFG_VAL_RATIO:-}" ] && VAL_RATIO="$CFG_VAL_RATIO"
    [ -n "${CFG_MAIN_PATH:-}" ] && MAIN_PATH="$CFG_MAIN_PATH"
    [ -n "${CFG_CONTRASTIVE:-}" ] && CONTRASTIVE="$CFG_CONTRASTIVE"
    echo "Config set: $CONFIG_FILE"
fi

# Apply command-line overrides.
while [ $# -gt 0 ]; do
    case "$1" in
        --config) shift 2 ;;
        --model) MODEL="$2"; shift 2 ;;
        --model-path|--model_path) MODEL_PATH="$2"; shift 2 ;;
        --main-path|--main_path) MAIN_PATH="$2"; shift 2 ;;
        --seed) SEED="$2"; shift 2 ;;
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
        --contrastive)
            if [ $# -gt 1 ] && [[ "$2" != --* ]]; then
                CONTRASTIVE="$2"
                shift 2
            else
                CONTRASTIVE="true"
                shift 1
            fi
            ;;
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
    MODEL_PATH="$(resolve_model_path "$MODEL")"
fi

MODEL_SUFFIX="$(resolve_model_suffix "$MODEL")"
[ -z "$MODEL_SUFFIX" ] && MODEL_SUFFIX="$(resolve_model_suffix "$MODEL_PATH")"

MAIN_PATH="${MAIN_PATH%/}"
[[ "$MAIN_PATH" != /* && "$MAIN_PATH" != [A-Za-z]:* ]] && MAIN_PATH="$REPO_ROOT/$MAIN_PATH"
[ -n "$MAIN_PATH" ] || { echo "Error: set data.main_path in the config or pass --main-path." >&2; exit 1; }

SPLIT_DIR="$MAIN_PATH/train_test_split"
if [ -d "$SPLIT_DIR" ]; then
    [ -z "$OUT_MASTER" ] && [ -f "$SPLIT_DIR/master_train.json" ] && OUT_MASTER="$SPLIT_DIR/master_train.json"
    [ -z "$OUT_VAL_MASTER" ] && [ -f "$SPLIT_DIR/master_val.json" ] && OUT_VAL_MASTER="$SPLIT_DIR/master_val.json"
fi

[ -z "$OUT_RETAIN_TRI_MASK" ] && OUT_RETAIN_TRI_MASK="$MAIN_PATH/tri_mask/npo_retain_tok${MODEL_SUFFIX}.jsonl"
[ -z "$OUT_FORGET_TRI_MASK" ] && OUT_FORGET_TRI_MASK="$MAIN_PATH/tri_mask/npo_forget_tok${MODEL_SUFFIX}.jsonl"
if [ -n "$VAL_RATIO" ] && [ "$VAL_RATIO" != "0" ] && [ "$VAL_RATIO" != "0.0" ]; then
    [ -z "$OUT_VAL_RETAIN_TRI_MASK" ] && OUT_VAL_RETAIN_TRI_MASK="$MAIN_PATH/tri_mask/npo_val_retain_tok${MODEL_SUFFIX}.jsonl"
elif [ -f "$SPLIT_DIR/master_val.json" ]; then
    [ -z "$OUT_VAL_RETAIN_TRI_MASK" ] && OUT_VAL_RETAIN_TRI_MASK="$MAIN_PATH/tri_mask/npo_val_retain_tok${MODEL_SUFFIX}.jsonl"
fi
[ -z "$OUT_PLAIN_TRAIN" ] && OUT_PLAIN_TRAIN="$MAIN_PATH/plain/plain_train_tok${MODEL_SUFFIX}.jsonl"
[ -z "$OUT_PLAIN_RETAIN" ] && OUT_PLAIN_RETAIN="$MAIN_PATH/plain/plain_retain_tok${MODEL_SUFFIX}.jsonl"
[ -z "$OUT_PLAIN_FORGET" ] && OUT_PLAIN_FORGET="$MAIN_PATH/plain/plain_forget_tok${MODEL_SUFFIX}.jsonl"
if [ -n "$VAL_RATIO" ] && [ "$VAL_RATIO" != "0" ] && [ "$VAL_RATIO" != "0.0" ]; then
    [ -z "$OUT_PLAIN_VAL" ] && OUT_PLAIN_VAL="$MAIN_PATH/plain/plain_val_tok${MODEL_SUFFIX}.jsonl"
elif [ -f "$SPLIT_DIR/master_val.json" ]; then
    [ -z "$OUT_PLAIN_VAL" ] && OUT_PLAIN_VAL="$MAIN_PATH/plain/plain_val_tok${MODEL_SUFFIX}.jsonl"
fi

ARGS=(
    -m pkg_halluc.package_loader.build_data
    --model_path "$MODEL_PATH"
    --seed "$SEED"
    --max_length "$MAX_LENGTH"
    --out_retain_tri_mask "$OUT_RETAIN_TRI_MASK"
    --out_forget_tri_mask "$OUT_FORGET_TRI_MASK"
)
ARGS+=(--main_path "$MAIN_PATH")

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

if [ -n "$CONTRASTIVE" ] && [ "$CONTRASTIVE" != "0" ] && [ "$CONTRASTIVE" != "false" ]; then
    ARGS+=(--contrastive "$CONTRASTIVE")
fi

if [ ${#RESULT_FILES[@]} -gt 0 ]; then
    ARGS+=(--result_files "${RESULT_FILES[@]}")
fi

echo "[build-data] \$ python ${ARGS[*]}   (cwd=$REPO_ROOT)"
cd "$REPO_ROOT"
"$PYTHON_BIN" "${ARGS[@]}"
