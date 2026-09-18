# Chạy cả pipeline theo từng bước, vd: ./scripts/quickstart.sh --model qwen2.5-coder-1.5b
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"

MODEL="Qwen/Qwen2.5-Coder-0.5B-Instruct"

while [ $# -gt 0 ]; do
    case "$1" in
        --model) MODEL="$2"; shift 2 ;;
        *) echo "tuỳ chọn không rõ: $1" >&2; exit 1 ;;
    esac
done

echo "== 1/5 download-model =="
bash "$SCRIPT_DIR/download_model.sh" --model "$MODEL"

echo "== 2/5 build-data =="
bash "$SCRIPT_DIR/build_data.sh" --model "$MODEL"

echo "== 3/5 train (ga, npo, ga_plain, npo_plain) =="
bash "$SCRIPT_DIR/train_ga.sh" --model "$MODEL"
bash "$SCRIPT_DIR/train_npo.sh" --model "$MODEL"
bash "$SCRIPT_DIR/train_ga_plain.sh" --model "$MODEL"
bash "$SCRIPT_DIR/train_npo_plain.sh" --model "$MODEL"

echo "== 4/5 evaluate (base, ga, npo, ga_plain, npo_plain) =="
MODEL_BASENAME="$(basename "$MODEL")"
bash "$SCRIPT_DIR/eval.sh" --tag base      --model-path "$MODELS_DIR/$MODEL"
bash "$SCRIPT_DIR/eval.sh" --tag ga        --model-path "$CHECKPOINTS_DIR/${MODEL_BASENAME}_ga"
bash "$SCRIPT_DIR/eval.sh" --tag npo       --model-path "$CHECKPOINTS_DIR/${MODEL_BASENAME}_npo"
bash "$SCRIPT_DIR/eval.sh" --tag ga_plain  --model-path "$CHECKPOINTS_DIR/${MODEL_BASENAME}_ga_plain"
bash "$SCRIPT_DIR/eval.sh" --tag npo_plain --model-path "$CHECKPOINTS_DIR/${MODEL_BASENAME}_npo_plain"

echo "== 5/5 report =="
bash "$SCRIPT_DIR/report.sh" --model-name "$MODEL"
