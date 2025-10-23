#!/bin/bash

# Continued pretraining script
# Usage: ./scripts/cont_pretrain.sh [model] [train_config] [dataset] [target_loss]

set -e

# Default configurations
MODEL_CFG=${1:-"configs/model/bert-base.yaml"}
TRAIN_CFG=${2:-"configs/train/lora.yaml"}
DATASET=${3:-"wikitext"}
TARGET_LOSS=${4:-"2.0"}

echo "Starting continued pretraining..."
echo "Model config: $MODEL_CFG"
echo "Training config: $TRAIN_CFG"
echo "Dataset: $DATASET"
echo "Target loss: $TARGET_LOSS"

# Run continued pretraining
python -m src.e3bench.train.cont_pretrain \
    --model_cfg "$MODEL_CFG" \
    --train_cfg "$TRAIN_CFG" \
    --dataset "$DATASET" \
    --target_loss "$TARGET_LOSS" \
    --output_dir results

echo "Continued pretraining completed!"
