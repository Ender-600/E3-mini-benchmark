# Inference Latency Curve Feature

## Quick Start

Test inference latency across different context lengths and generate latency curves:

```bash
./scripts/bench_scaling.sh
```

This will automatically:
1. Test GPT-2, T5, and BERT across multiple context lengths (128, 256, 512, 1024)
2. Aggregate results to CSV
3. Generate latency curve visualization

## View Results

- **Data**: `tables/inference_results.csv`
- **Plot**: `figs/latency_curve.png` ⭐

## What You'll See

The **latency curve plot** shows how inference latency scales with context length for different architectures:

- 🟢 **Encoder (BERT)**: Flattest curve - best for long contexts
- 🔵 **Encoder-Decoder (T5)**: Moderate scaling
- 🔴 **Decoder (GPT-2)**: Steepest curve - challenges with long contexts

## Test Individual Architecture

```bash
# GPT-2
python -m src.e3bench.eval.bench_infer \
    --model_cfg configs/model/gpt2-medium.yaml \
    --bench_cfg configs/bench/infer_decoder_scaling.yaml \
    --output_dir results

# T5
python -m src.e3bench.eval.bench_infer \
    --model_cfg configs/model/t5-base.yaml \
    --bench_cfg configs/bench/infer_seq2seq_scaling.yaml \
    --output_dir results

# BERT
python -m src.e3bench.eval.bench_infer \
    --model_cfg configs/model/bert-base.yaml \
    --bench_cfg configs/bench/infer_encoder_scaling.yaml \
    --output_dir results
```

## Configuration

Test configurations are in `configs/bench/infer_*_scaling.yaml`:

```yaml
context_lengths: [128, 256, 512, 1024]  # Lengths to test
num_tokens: 50                           # Tokens to generate
num_runs: 3                              # Runs per length
```

Customize as needed:
- Add longer contexts: `[128, 256, 512, 1024, 2048]`
- Increase precision: `num_runs: 5`
- Test longer generation: `num_tokens: 100`

## New Files

```
configs/bench/
├── infer_decoder_scaling.yaml      # GPT-2 scaling config
├── infer_seq2seq_scaling.yaml      # T5 scaling config
└── infer_encoder_scaling.yaml      # BERT scaling config

scripts/
└── bench_scaling.sh                 # Automated test script

figs/
└── latency_curve.png                # Generated plot ✨
```

## Key Features

✅ **Quantitative Comparison** - Measure architecture scaling characteristics
✅ **Performance Prediction** - Extrapolate to longer contexts
✅ **Visual Comparison** - See all architectures on one plot
✅ **Fully Automated** - One command to test, aggregate, and plot

## Notes

- **GPU Recommended**: Testing on CPU will be very slow
- **Memory**: Long contexts (>1024) require more VRAM
- **Duration**: Full test takes 10-30 minutes depending on hardware

## Documentation

For detailed documentation, see: `docs/latency_curve_feature.md`

## Architecture-Specific Insights

| Architecture | Scaling Behavior | Best Use Case |
|--------------|------------------|---------------|
| Encoder (BERT) | Slowest growth | Long document classification, Q&A |
| Encoder-Decoder (T5) | Moderate growth | Translation, summarization |
| Decoder (GPT-2) | Fastest growth | Short text generation |

