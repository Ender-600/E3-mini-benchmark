#!/bin/bash

# SuperGLUE fine-tuning script
# Usage: ./scripts/finetune_superglue.sh [model] [task] [train_config]

set -e

# Default configurations
MODEL_CFG=${1:-"configs/model/bert-base.yaml"}
TASK_CFG=${2:-"configs/task/superglue.yaml"}
TRAIN_CFG=${3:-"configs/train/lora.yaml"}

echo "Starting SuperGLUE fine-tuning..."
echo "Model config: $MODEL_CFG"
echo "Task config: $TASK_CFG"
echo "Training config: $TRAIN_CFG"

# Run fine-tuning
python -m src.e3bench.train.finetune_glue \
    --model_cfg "$MODEL_CFG" \
    --task_cfg "$TASK_CFG" \
    --train_cfg "$TRAIN_CFG" \
    --output_dir results

echo "SuperGLUE fine-tuning completed!"
