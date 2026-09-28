#!/usr/bin/env bash
# CAA Pipeline Runner: extract steering vectors & evaluate steering
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"

STAGE="extract"
CONFIG_FILE=""
MODEL=""
MODEL_PATH=""
MAIN_PATH=""
DATASET_PATH=""
OUTPUT_DIR=""
VECTORS_DIR=""
RESULTS_DIR=""
LAYERS=()
LAYER_PAIRS="adjacent"
POSITION_MODE="package_start"
DIRECTION="non_hallucination"
MULTIPLIERS=()
OVERWRITE="false"
DEVICE="auto"
DTYPE="auto"
MAX_SAMPLES=""
FROM_CONFIG="false"
LOCAL_FILES_ONLY="true"
SAVE_LAYER_DIFFS="true"
SAVE_ACTIVATIONS="false"

usage() {
    cat <<USAGE
Usage: bash scripts/CAA.sh [options]

Main options:
  --config FILE             Path to JSON config file
  --stage STR               Pipeline stage: extract | eval | all (default: $STAGE)
  --model STR               Model name or preset (overrides config)
  --model-path PATH         Path to checkpoint / weights
  --main-path PATH          Model data directory
    --dataset-path PATH       Override generated pairwise test data
  --output-dir PATH         Directory to save extracted vectors
  --vectors-dir PATH        Directory containing vectors for eval
  --results-dir PATH        Directory to save eval results
  --layers INT...           Layer indices to extract / steer
  --layer-pairs STR         Layer pairs for difference vectors: adjacent | all | spec
  --position-mode STR       Token position mode
  --direction STR           Steering direction
  --save-layer-diffs BOOL   Save layer-pair difference vectors
  --save-activations        Save raw activation tensors
  --multipliers FLOAT...    Steering vector multipliers
  --overwrite               Overwrite existing result files
  --device STR              Compute device
  --dtype STR               Compute dtype
  --max-samples INT         Limit number of processed samples
  --from-config             Initialize model architecture without loading weights
  --local-files-only BOOL   Load strictly from local cache without network calls
  -h, --help                Show this help message and exit

Examples:
  1. Extract steering vectors for Llama 3.2 1B:
     bash scripts/CAA.sh --config model_config/llama3.2-1b.json --stage extract

  2. Evaluate pairwise package completions:
      bash scripts/CAA.sh --config model_config/llama3.2-1b.json --stage eval --layers 10 12 14 --multipliers 0.0 0.5 1.0 1.5 2.0

  3. Run complete pipeline (extract -> eval):
    bash scripts/CAA.sh --config model_config/llama3.2-1b.json --stage all --layers 10 12 --multipliers 0.0 1.0
USAGE
    exit "${1:-0}"
}

# Read the config path before loading defaults from it.
_args=("$@")
for ((_i = 0; _i < ${#_args[@]}; _i++)); do
    if [ "${_args[$_i]}" = "--config" ] && [ $((_i + 1)) -lt ${#_args[@]} ]; then
        CONFIG_FILE="${_args[$((_i + 1))]}"
        break
    fi
done

if [ -n "$CONFIG_FILE" ]; then
    if [ ! -f "$CONFIG_FILE" ]; then
        echo "Error: Config file not found at: $CONFIG_FILE" >&2
        exit 1
    fi
    load_config "$CONFIG_FILE" "" "data" "eval"
    [ -n "${CFG_MODEL_NAME:-}" ] && MODEL="$CFG_MODEL_NAME"
    [ -n "${CFG_MODEL_PATH:-}" ] && MODEL_PATH="$CFG_MODEL_PATH"
    [ -n "${CFG_MAIN_PATH:-}" ] && MAIN_PATH="$CFG_MAIN_PATH"
    [ -n "${CFG_DTYPE:-}" ] && [ "$CFG_DTYPE" != "auto" ] && DTYPE="$CFG_DTYPE"
    echo "[CAA] Loaded config: $CONFIG_FILE (model=${MODEL:-$MODEL_PATH}, main_path=${MAIN_PATH:-N/A})"
fi

# Apply command-line overrides.
while [ $# -gt 0 ]; do
    case "$1" in
        --config) shift 2 ;;
        --stage) STAGE="$2"; shift 2 ;;
        --model) MODEL="$2"; shift 2 ;;
        --model-path) MODEL_PATH="$2"; shift 2 ;;
        --main-path) MAIN_PATH="$2"; shift 2 ;;
        --dataset-path) DATASET_PATH="$2"; shift 2 ;;
        --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
        --vectors-dir) VECTORS_DIR="$2"; shift 2 ;;
        --results-dir) RESULTS_DIR="$2"; shift 2 ;;
        --layer-pairs) LAYER_PAIRS="$2"; shift 2 ;;
        --position-mode) POSITION_MODE="$2"; shift 2 ;;
        --direction) DIRECTION="$2"; shift 2 ;;
        --device) DEVICE="$2"; shift 2 ;;
        --dtype) DTYPE="$2"; shift 2 ;;
        --max-samples) MAX_SAMPLES="$2"; shift 2 ;;
        --from-config) FROM_CONFIG="true"; shift ;;
        --overwrite) OVERWRITE="true"; shift ;;
        --save-activations) SAVE_ACTIVATIONS="true"; shift ;;
        --save-layer-diffs)
            if [ $# -ge 2 ] && [[ "$2" =~ ^(true|false)$ ]]; then
                SAVE_LAYER_DIFFS="$2"; shift 2
            else
                SAVE_LAYER_DIFFS="true"; shift
            fi
            ;;
        --local-files-only)
            if [ $# -ge 2 ] && [[ "$2" =~ ^(true|false)$ ]]; then
                LOCAL_FILES_ONLY="$2"; shift 2
            else
                LOCAL_FILES_ONLY="true"; shift
            fi
            ;;
        --layers)
            shift
            LAYERS=()
            while [ $# -gt 0 ] && [[ ! "$1" =~ ^-- ]]; do
                LAYERS+=("$1")
                shift
            done
            ;;
        --multipliers)
            shift
            MULTIPLIERS=()
            while [ $# -gt 0 ] && [[ ! "$1" =~ ^-- ]]; do
                MULTIPLIERS+=("$1")
                shift
            done
            ;;
        -h|--help) usage 0 ;;
        *)
            echo "Error: Unrecognized option: $1" >&2
            usage 1
            ;;
    esac
done

TARGET_MODEL="${MODEL_PATH:-$MODEL}"
if [ -z "$TARGET_MODEL" ] && [ -z "$CONFIG_FILE" ]; then
    echo "Error: Must provide --config or --model/--model-path!" >&2
    exit 1
fi

if [ -z "$MAIN_PATH" ] && [ -z "$CONFIG_FILE" ]; then
    echo "Error: Must provide --config or --main-path!" >&2
    exit 1
fi

if [ "$DEVICE" = "auto" ]; then
    DEVICE="cuda"
fi

cd "$REPO_ROOT"

if [ "$STAGE" = "extract" ] || [ "$STAGE" = "all" ]; then
    echo "  Extracting steering vectors for model: ${TARGET_MODEL:-$MODEL} (main_path=${MAIN_PATH:-N/A})"

    EXTRACT_CMD=(
        "$PYTHON_BIN" -u "$REPO_ROOT/pkg_halluc/CAA/extractor/extract_package_activations.py"
        --device "$DEVICE"
        --dtype "$DTYPE"
        --position_mode "$POSITION_MODE"
        --layer_pairs "$LAYER_PAIRS"
        --direction "$DIRECTION"
    )

    [ -n "$CONFIG_FILE" ] && EXTRACT_CMD+=(--config "$CONFIG_FILE")
    [ -n "$TARGET_MODEL" ] && EXTRACT_CMD+=(--model_name_or_path "$TARGET_MODEL")
    [ -n "$MAIN_PATH" ] && EXTRACT_CMD+=(--main_path "$MAIN_PATH")
    [ -n "$DATASET_PATH" ] && EXTRACT_CMD+=(--dataset_path "$DATASET_PATH")
    [ -n "$OUTPUT_DIR" ] && EXTRACT_CMD+=(--output_dir "$OUTPUT_DIR")
    [ -n "$MAX_SAMPLES" ] && EXTRACT_CMD+=(--max_samples "$MAX_SAMPLES")
    [ ${#LAYERS[@]} -gt 0 ] && EXTRACT_CMD+=(--layers "${LAYERS[@]}")
    [ "$FROM_CONFIG" = "true" ] && EXTRACT_CMD+=(--from_config)
    [ "$SAVE_LAYER_DIFFS" = "true" ] && EXTRACT_CMD+=(--save_layer_diffs)
    [ "$SAVE_ACTIVATIONS" = "true" ] && EXTRACT_CMD+=(--save_activations)
    if [ "$LOCAL_FILES_ONLY" = "true" ]; then
        EXTRACT_CMD+=(--local_files_only)
    else
        EXTRACT_CMD+=(--no-local_files_only)
    fi

    echo "[exec] ${EXTRACT_CMD[*]}"
    "${EXTRACT_CMD[@]}"
fi

if [ "$STAGE" = "eval" ] || [ "$STAGE" = "all" ]; then
    echo "  Evaluating model with steering vectors..."

    if [ ${#LAYERS[@]} -eq 0 ]; then
        LAYERS=(10 12 14)
        echo "Notice: No --layers specified for eval, using default: ${LAYERS[*]}"
    fi

    if [ ${#MULTIPLIERS[@]} -eq 0 ]; then
        MULTIPLIERS=(0.0 0.5 1.0 1.5 2.0)
        echo "Notice: No --multipliers specified, using default: ${MULTIPLIERS[*]}"
    fi

    EVAL_CMD=(
        "$PYTHON_BIN" -u "$REPO_ROOT/pkg_halluc/CAA/eval/prompting_with_steering.py"
        --layers "${LAYERS[@]}"
        --multipliers "${MULTIPLIERS[@]}"
    )

    [ -n "$CONFIG_FILE" ] && EVAL_CMD+=(--config "$CONFIG_FILE")
    [ -n "$TARGET_MODEL" ] && EVAL_CMD+=(--model_name_or_path "$TARGET_MODEL")
    [ -n "$MAIN_PATH" ] && EVAL_CMD+=(--main_path "$MAIN_PATH")
    [ -n "$DATASET_PATH" ] && EVAL_CMD+=(--dataset_path "$DATASET_PATH")
    [ -n "$VECTORS_DIR" ] && EVAL_CMD+=(--vectors_dir "$VECTORS_DIR")
    [ -n "$RESULTS_DIR" ] && EVAL_CMD+=(--results_dir "$RESULTS_DIR")
    [ -n "$MAX_SAMPLES" ] && EVAL_CMD+=(--max_samples "$MAX_SAMPLES")
    [ "$OVERWRITE" = "true" ] && EVAL_CMD+=(--overwrite)
    [ "$FROM_CONFIG" = "true" ] && EVAL_CMD+=(--from_config)
    if [ "$LOCAL_FILES_ONLY" = "true" ]; then
        EVAL_CMD+=(--local_files_only)
    else
        EVAL_CMD+=(--no-local_files_only)
    fi

    echo "[exec] ${EVAL_CMD[*]}"
    "${EVAL_CMD[@]}"
fi

echo "[CAA] Pipeline Done!"