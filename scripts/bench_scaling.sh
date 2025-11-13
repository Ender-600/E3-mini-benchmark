#!/bin/bash
# Test inference scaling benchmarks across different architectures

set -e

echo "=========================================="
echo "Inference Scaling Benchmark Test"
echo "=========================================="

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    echo "Activating virtual environment..."
    source venv/bin/activate
fi

# Test 1: Decoder (GPT-2) scaling
echo ""
echo "1/3: Testing Decoder (GPT-2) scaling..."
python -m src.e3bench.eval.bench_infer \
    --model_cfg configs/model/gpt2-medium.yaml \
    --bench_cfg configs/bench/infer_decoder_scaling.yaml \
    --output_dir results

# Test 2: Encoder-Decoder (T5) scaling
echo ""
echo "2/3: Testing Encoder-Decoder (T5) scaling..."
python -m src.e3bench.eval.bench_infer \
    --model_cfg configs/model/t5-base.yaml \
    --bench_cfg configs/bench/infer_seq2seq_scaling.yaml \
    --output_dir results

# Test 3: Encoder (BERT) scaling
echo ""
echo "3/3: Testing Encoder (BERT) scaling..."
python -m src.e3bench.eval.bench_infer \
    --model_cfg configs/model/bert-base.yaml \
    --bench_cfg configs/bench/infer_encoder_scaling.yaml \
    --output_dir results

echo ""
echo "=========================================="
echo "All scaling tests completed!"
echo "=========================================="
echo ""
echo "Aggregating results..."
python -m src.e3bench.report.aggregate --results_dir latest --out_dir tables

echo ""
echo "Generating latency curve plot..."
python -m src.e3bench.report.plots --tables tables --out figs

echo ""
echo "=========================================="
echo "Complete! Check:"
echo "  - tables/inference_results.csv for data"
echo "  - figs/latency_curve.png for visualization"
echo "=========================================="

