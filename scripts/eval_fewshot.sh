#!/bin/bash

# Few-shot evaluation script
# Usage: ./scripts/eval_fewshot.sh [model] [eval_config]

set -e

# Default configurations
MODEL_CFG=${1:-"configs/model/gpt2-medium.yaml"}
EVAL_CFG=${2:-"configs/eval/fewshot_5.yaml"}

echo "Starting few-shot evaluation..."
echo "Model config: $MODEL_CFG"
echo "Evaluation config: $EVAL_CFG"

# Run few-shot evaluation
python -m src.e3bench.eval.eval_fewshot \
    --model_cfg "$MODEL_CFG" \
    --eval_cfg "$EVAL_CFG" \
    --output_dir results

echo "Few-shot evaluation completed!"
