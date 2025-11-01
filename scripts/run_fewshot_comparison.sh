#!/bin/bash
# Batch script for comprehensive few-shot model comparison
# This script runs all evaluation combinations and generates reports

set -e

echo "=========================================="
echo "E³ Mini-Benchmark: Few-Shot Comparison"
echo "=========================================="
echo ""

# Check if virtual environment exists
if [ -d "venv" ]; then
    echo "Activating virtual environment..."
    source venv/bin/activate
fi

# Run comprehensive comparison
echo "Running comprehensive few-shot comparison..."
echo "This will evaluate:"
echo "  - BERT (encoder)"
echo "  - GPT-2 (decoder)"
echo "  - T5 (encoder-decoder)"
echo "  With 0, 5, and 10 shot settings"
echo ""

# Run all evaluations
make fewshot-comparison

echo ""
echo "=========================================="
echo "Comparison complete!"
echo "=========================================="
echo ""
echo "Results saved to:"
echo "  - tables/fewshot_results.csv (aggregated data)"
echo "  - figs/fewshot_curves.png (learning curves)"
echo "  - figs/fewshot_heatmap.png (model comparison)"
echo ""

