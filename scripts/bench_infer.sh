#!/bin/bash

# Inference benchmarking script
# Usage: ./scripts/bench_infer.sh [model] [bench_config]

set -e

# Default configurations
MODEL_CFG=${1:-"configs/model/t5-base.yaml"}
BENCH_CFG=${2:-"configs/bench/infer_seq2seq.yaml"}

echo "Starting inference benchmarking..."
echo "Model config: $MODEL_CFG"
echo "Benchmark config: $BENCH_CFG"

# Run inference benchmark
python -m src.e3bench.eval.bench_infer \
    --model_cfg "$MODEL_CFG" \
    --bench_cfg "$BENCH_CFG" \
    --output_dir results

echo "Inference benchmarking completed!"
