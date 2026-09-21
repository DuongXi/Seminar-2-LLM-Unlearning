# Eval package hallucination của 1 checkpoint, dùng chung cho mọi method (pkg_halluc/evaluation/eval_variant.py)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"

TAG=""
MODEL_PATH=""
N_PROMPTS="150"
BATCH_SIZE="16"
SEED="42"
PACKAGE_MODES="1 2"
EVAL_PROMPTS_PATH=""
OUT_DIR=""
CONFIG_FILE=""
EXTRA_ARGS=()

usage() {
    cat <<USAGE
Cách dùng: eval.sh --tag TAG --model-path DUONG_DAN [tuỳ chọn]
  --config FILE                 File config JSON (vd model_config/default.json)
  --tag TAG                    Tên method để đặt tên thư mục kết quả, vd base/ga/npo/ga_plain/npo_plain (bắt buộc)
  --model-path DUONG_DAN        Checkpoint cần eval (bắt buộc)
  --n-prompts INT                 Số prompt lấy mẫu từ pool eval (mặc định: $N_PROMPTS)
  --batch-size INT                 (mặc định: $BATCH_SIZE)
  --seed INT                        (mặc định: $SEED)
  --package-modes "1 2"              Chạy mode hỏi package nào (mặc định: "$PACKAGE_MODES")
  --eval-prompts-path DUONG_DAN        
  --out-dir DUONG_DAN                   (mặc định: $EVAL_RUNS_DIR/<tag>)
  -- CAC_THAM_SO...                      
                                          
  -h, --help

USAGE
    exit "${1:-0}"
}

# Pass 1:  tìm --config để làm mặc định trước
_args=("$@")
for ((_i = 0; _i < ${#_args[@]}; _i++)); do
    if [ "${_args[$_i]}" = "--config" ]; then
        CONFIG_FILE="${_args[$((_i + 1))]}"
        break
    fi
done

if [ -n "$CONFIG_FILE" ]; then
    load_config "$CONFIG_FILE" "" "eval"
    [ -n "${CFG_SEED:-}" ] && SEED="$CFG_SEED"
    [ -n "${CFG_N_EVAL_PROMPTS:-}" ] && N_PROMPTS="$CFG_N_EVAL_PROMPTS"
    [ -n "${CFG_BATCH_SIZE:-}" ] && BATCH_SIZE="$CFG_BATCH_SIZE"
    [ -n "${CFG_PACKAGE_MODES:-}" ] && PACKAGE_MODES="$CFG_PACKAGE_MODES"
    echo "[eval] da nap config: $CONFIG_FILE (eval)"
fi

# Pass 2: xử lý ghi đè giá trị từ config
while [ $# -gt 0 ]; do
    case "$1" in
        --config) shift 2 ;; 
        --tag) TAG="$2"; shift 2 ;;
        --model-path) MODEL_PATH="$2"; shift 2 ;;
        --n-prompts) N_PROMPTS="$2"; shift 2 ;;
        --batch-size) BATCH_SIZE="$2"; shift 2 ;;
        --seed) SEED="$2"; shift 2 ;;
        --package-modes) PACKAGE_MODES="$2"; shift 2 ;;
        --eval-prompts-path) EVAL_PROMPTS_PATH="$2"; shift 2 ;;
        --out-dir) OUT_DIR="$2"; shift 2 ;;
        -h|--help) usage 0 ;;
        --) shift; EXTRA_ARGS=("$@"); break ;;
        *) echo "tuỳ chọn không rõ: $1" >&2; usage 1 ;;
    esac
done

[ -z "$TAG" ] && { echo "loi: --tag la bat buoc" >&2; usage 1; }
[ -z "$MODEL_PATH" ] && { echo "loi: --model-path la bat buoc" >&2; usage 1; }
[ -z "$OUT_DIR" ] && OUT_DIR="$EVAL_RUNS_DIR/$TAG"
[ -z "$EVAL_PROMPTS_PATH" ] && EVAL_PROMPTS_PATH="$REPO_ROOT/data/eval/prompts.jsonl"

MODES_ARR=($PACKAGE_MODES)

ARGS=(
    -m pkg_halluc.evaluation.eval_variant
    --tag "$TAG"
    --model_path "$MODEL_PATH"
    --n_prompts "$N_PROMPTS"
    --batch_size "$BATCH_SIZE"
    --seed "$SEED"
    --package_modes "${MODES_ARR[@]}"
    --pypi_csv "$REPO_ROOT/data/eval/pypi_package_names.csv"
    --false_positive_csv "$REPO_ROOT/data/eval/false_positive_packages.csv"
    --eval_prompts_path "$EVAL_PROMPTS_PATH"
    --out_dir "$OUT_DIR"
    --method standard
)
[ "${#EXTRA_ARGS[@]}" -gt 0 ] && ARGS+=("${EXTRA_ARGS[@]}")

echo "[eval:$TAG] \$ python ${ARGS[*]}   (cwd=$REPO_ROOT)"
cd "$REPO_ROOT"
"$PYTHON_BIN" "${ARGS[@]}"

echo "[eval:$TAG] xong -> $OUT_DIR/FINAL_RESULTS.csv"
