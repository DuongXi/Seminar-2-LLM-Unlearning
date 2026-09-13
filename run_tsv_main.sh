#!/usr/bin/env bash
# run_tsv_main.sh
#
# Convenience script to train a steering vector with tsv_main.py.
# Edit the variables below, then run in a bash terminal (Git Bash, WSL, etc.)
# from the project folder with:
#
#     bash run_tsv_main.sh
#
# (or `chmod +x run_tsv_main.sh` once, then `./run_tsv_main.sh`)

set -e  # stop immediately if any command fails

# ---- Edit these ----
MODEL_PATH="/content/drive/MyDrive/Package_Unlearning_Project/Data/PackageHallucination/Models/Qwen_3"
DATA_PATH="LLM_LY_results.csv"
OUTPUT_DIR="tsv_output"
BATCH_SIZE=4
NUM_EPOCHS=5
LAYER_START=4
LAYER_END=8
# ---------------------

# Activate the virtual environment automatically if it isn't already active.
# Checks both venv layouts: Windows-style (.venv/Scripts/activate, used by
# Git Bash on Windows) and Linux-style (.venv/bin/activate, used by WSL/Linux).
if [ -z "$VIRTUAL_ENV" ]; then
    if [ -f ".venv/Scripts/activate" ]; then
        echo "Activating .venv (Windows-style)..."
        source .venv/Scripts/activate
    elif [ -f ".venv/bin/activate" ]; then
        echo "Activating .venv (Linux-style)..."
        source .venv/bin/activate
    else
        echo "Warning: no .venv found in this folder - continuing with whatever 'python' resolves to on PATH." >&2
    fi
fi

if [ ! -f "$DATA_PATH" ]; then
    echo "Error: data file not found: $DATA_PATH (check the path, or that you're running this from the project folder)" >&2
    exit 1
fi

echo "Running tsv_main.py"
echo "  model_path  : $MODEL_PATH"
echo "  data_path   : $DATA_PATH"
echo "  output_dir  : $OUTPUT_DIR"
echo "  batch_size  : $BATCH_SIZE"
echo "  num_epochs  : $NUM_EPOCHS"
echo "  layer_range : $LAYER_START - $LAYER_END"
echo ""

python tsv_main.py \
    --model_path "$MODEL_PATH" \
    --data_path "$DATA_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --batch_size "$BATCH_SIZE" \
    --num_epochs "$NUM_EPOCHS" \
    --layer_start "$LAYER_START" \
    --layer_end "$LAYER_END"

echo ""
echo "Done. Steering vector saved to $OUTPUT_DIR/steering_vector.pt"
