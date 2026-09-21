# Chạy cả pipeline theo từng bước, vd: ./scripts/quickstart.sh --config model_config/default.json
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"

MODEL="Qwen/Qwen2.5-Coder-0.5B-Instruct"
CONFIG_FILE=""

while [ $# -gt 0 ]; do
    case "$1" in
        --config) CONFIG_FILE="$2"; shift 2 ;;
        --model) MODEL="$2"; shift 2 ;;
        *) echo "unknown option: $1" >&2; exit 1 ;;
    esac
done

if [ -n "$CONFIG_FILE" ]; then
    load_config "$CONFIG_FILE" ""
    [ -n "${CFG_MODEL_NAME:-}" ] && MODEL="$CFG_MODEL_NAME"
fi

MODEL_FULL="$(resolve_model_name "$MODEL")"
MODEL_BASENAME="$(basename "$MODEL_FULL")"

OPTS=()
if [ -n "$CONFIG_FILE" ]; then
    OPTS+=(--config "$CONFIG_FILE")
else
    OPTS+=(--model "$MODEL")
fi

echo "== 1/5 download-model =="
bash "$SCRIPT_DIR/download_model.sh" "${OPTS[@]}"

echo "== 2/5 build-data =="
bash "$SCRIPT_DIR/build_data.sh" "${OPTS[@]}"

echo "== 3/5 train (ga, npo, ga_plain, npo_plain) =="
bash "$SCRIPT_DIR/train_ga.sh" "${OPTS[@]}"
bash "$SCRIPT_DIR/train_npo.sh" "${OPTS[@]}"
bash "$SCRIPT_DIR/train_ga_plain.sh" "${OPTS[@]}"
bash "$SCRIPT_DIR/train_npo_plain.sh" "${OPTS[@]}"

echo "== 4/5 evaluate (base, ga, npo, ga_plain, npo_plain) =="
EVAL_OPTS=()
[ -n "$CONFIG_FILE" ] && EVAL_OPTS+=(--config "$CONFIG_FILE")

bash "$SCRIPT_DIR/eval.sh" "${EVAL_OPTS[@]}" --tag base      --model-path "$MODELS_DIR/$MODEL_FULL"
bash "$SCRIPT_DIR/eval.sh" "${EVAL_OPTS[@]}" --tag ga        --model-path "$CHECKPOINTS_DIR/${MODEL_BASENAME}_ga"
bash "$SCRIPT_DIR/eval.sh" "${EVAL_OPTS[@]}" --tag npo       --model-path "$CHECKPOINTS_DIR/${MODEL_BASENAME}_npo"
bash "$SCRIPT_DIR/eval.sh" "${EVAL_OPTS[@]}" --tag ga_plain  --model-path "$CHECKPOINTS_DIR/${MODEL_BASENAME}_ga_plain"
bash "$SCRIPT_DIR/eval.sh" "${EVAL_OPTS[@]}" --tag npo_plain --model-path "$CHECKPOINTS_DIR/${MODEL_BASENAME}_npo_plain"

echo "== 5/5 report =="
bash "$SCRIPT_DIR/report.sh" "${OPTS[@]}"
