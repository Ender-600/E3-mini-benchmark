# Few-Shot Comparison Guide

This document explains how to use E³ Mini-Benchmark for cross-model zero-shot/few-shot evaluation comparison.

## Overview

The project now supports comprehensive few-shot evaluation and comparison across multiple model architectures (BERT, GPT-2, T5). You can:

1. **Individual evaluation**: Evaluate specific models with different shot counts
2. **Batch comparison**: Run comprehensive comparison across all models and shot settings
3. **Automated reporting**: Automatically generate visualization and comparison charts

## Quick Start

### Method 1: Using Makefile (Recommended)

```bash
# Activate virtual environment
source venv/bin/activate

# Run full comparison (all models × all shots)
make fewshot-comparison

# Or run only 5-shot evaluation
make eval-all-models
```

### Method 2: Using Batch Script

```bash
source venv/bin/activate
./scripts/run_fewshot_comparison.sh
```

### Method 3: Running Individual Evaluations

```bash
source venv/bin/activate

# BERT evaluation
make eval-bert-0shot    # BERT 0-shot
make eval-bert-5shot    # BERT 5-shot
make eval-bert-10shot   # BERT 10-shot

# GPT-2 evaluation
make eval-gpt2-0shot    # GPT-2 0-shot
make eval-gpt2-5shot    # GPT-2 5-shot
make eval-gpt2-10shot   # GPT-2 10-shot

# T5 evaluation
make eval-t5-0shot      # T5 0-shot
make eval-t5-5shot      # T5 5-shot
make eval-t5-10shot     # T5 10-shot
```

## Viewing Results

After running `make fewshot-comparison`, results are automatically saved in the following locations:

### 1. Aggregated Data (CSV)

```bash
# View aggregated few-shot results
cat tables/fewshot_results.csv

# Columns include:
# - model: Model name
# - arch: Architecture (encoder/decoder/encdec)
# - task: Task name
# - accuracy: Accuracy
# - num_fewshot: Number of few-shot examples
# - max_length: Maximum length
# - duration_seconds: Execution time
# - max_memory_gb: Maximum memory usage
# - avg_watt: Average power consumption
# - kwh: Total energy consumption
```

### 2. Visualization Charts

```bash
# View generated charts
ls -lh figs/

# Main charts:
# - fewshot_curves.png: Learning curves for different models across shots
# - fewshot_heatmap.png: Model performance heatmap (architecture vs task)
```

### 3. Detailed Result Files

```bash
# Each evaluation generates a JSON file
ls -lh results/fewshot_*.json

# View specific result
cat results/fewshot_20251029_235049.json
```

## Available Makefile Targets

### Individual Evaluation Targets

#### BERT (Encoder-only)
- `eval-bert-0shot`: 0-shot evaluation
- `eval-bert-5shot`: 5-shot evaluation
- `eval-bert-10shot`: 10-shot evaluation

#### GPT-2 (Decoder-only)
- `eval-gpt2-0shot`: 0-shot evaluation
- `eval-gpt2-5shot`: 5-shot evaluation
- `eval-gpt2-10shot`: 10-shot evaluation

#### T5 (Encoder-decoder)
- `eval-t5-0shot`: 0-shot evaluation
- `eval-t5-5shot`: 5-shot evaluation
- `eval-t5-10shot`: 10-shot evaluation

### Batch Evaluation Targets

- `eval-all-models`: Run 5-shot evaluation on all models
- `fewshot-comparison`: Run full comparison (all models × all shots) + auto-generate reports and charts

### Post-processing Targets

- `report`: Aggregate results and run significance tests
- `figs`: Generate comparison charts

## Usage Examples

### Example 1: Quick Comparison of GPT-2 and T5 5-shot Performance

```bash
source venv/bin/activate

# Evaluate two models
make eval-gpt2-5shot
make eval-t5-5shot

# Aggregate results
make report

# Generate comparison charts
make figs

# View results
cat tables/fewshot_results.csv | grep -E "(gpt2|t5)" | grep "5$"
```

### Example 2: Analyzing the Effect of Different Shot Counts

```bash
source venv/bin/activate

# Run all shot settings for a specific model
make eval-gpt2-0shot
make eval-gpt2-5shot
make eval-gpt2-10shot

# View GPT-2 performance across different shots
cat tables/fewshot_results.csv | grep "gpt2"
```

### Example 3: Complete Comparison Study

```bash
source venv/bin/activate

# One-click full comparison
make fewshot-comparison

# This will:
# 1. Run BERT × {0,5,10} shots = 3 evaluations
# 2. Run GPT-2 × {0,5,10} shots = 3 evaluations
# 3. Run T5 × {0,5,10} shots = 3 evaluations
# 4. Automatically aggregate results
# 5. Automatically generate comparison charts

# Total time: ~30-60 minutes (depending on hardware)
```

## Result Interpretation

### fewshot_curves.png
Learning curve chart showing:
- X-axis: Few-shot example count (0, 5, 10)
- Y-axis: Accuracy
- Different colored lines: Different model architectures
- Subplots: Different tasks

This chart allows you to observe:
- Which architecture performs best in zero-shot
- Sensitivity of different architectures to few-shot examples
- Performance gains from adding examples

### fewshot_heatmap.png
Heatmap showing:
- Rows: Model architectures
- Columns: Tasks
- Color intensity: Accuracy

This allows you to see:
- Which architecture performs best on which tasks
- Difficulty differences between tasks

## Customization

### Modifying Evaluation Tasks

Edit `configs/eval/fewshot_*.yaml`:

```yaml
name: "fewshot_5"
num_fewshot: 5
max_length: 2048
batch_size: 1
limit: 100
tasks: ["mmlu", "arc_challenge", "hellaswag"]  # Can modify task list
```

### Adding New Models

1. Create new model config file in `configs/model/`
2. Add corresponding evaluation targets in Makefile

## Notes

1. **Virtual environment**: Always activate virtual environment before running
2. **Time consumption**: Full comparison takes considerable time, recommended to run on GPU
3. **Memory usage**: Different models have different memory requirements, be aware of hardware limits
4. **Result retention**: All results saved in `results/` directory, won't be overwritten

## Troubleshooting

### Issue 1: "python: No such file or directory"
**Solution**: Activate virtual environment `source venv/bin/activate`

### Issue 2: Evaluation fails, metrics are empty
**Reason**: Possible lm-eval-harness version compatibility issue
**Solution**: Check parameter configuration in `src/e3bench/eval/eval_fewshot.py`

### Issue 3: Some models cannot be evaluated
**Reason**: Some architectures may not be suitable for certain tasks
**Solution**: Check model and task compatibility

## Contributing

To add new evaluation metrics, models, or tasks:
1. Modify corresponding configuration files
2. Update Makefile to add new targets
3. Update this document

## References

- [lm-eval-harness documentation](https://github.com/EleutherAI/lm-evaluation-harness)
- Project main README: `README.md`
- Evaluation result format: `src/e3bench/utils/io.py`
