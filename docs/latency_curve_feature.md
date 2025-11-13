# Inference Latency Curve Feature

## Feature Overview

This feature enables testing inference latency across different context lengths and generates **latency curves** to visualize how different architectures scale with increasing input length.

## Why This Feature?

Different architectures show significant performance differences when processing various input lengths:

| Architecture | Context Length Scaling | Reason |
|--------------|------------------------|--------|
| **Decoder** (GPT-2) | ⚠️ Approximately linear growth | Auto-regressive generation, KV cache grows linearly with context |
| **Encoder-Decoder** (T5) | 📈 Moderate growth | Encoder processes once, Decoder generates incrementally |
| **Encoder** (BERT) | ✅ Flattest curve | Single forward pass, no generation steps |

## New Files

### 1. Configuration Files

Three scaling test configurations:

```
configs/bench/infer_decoder_scaling.yaml     # GPT-2 scaling test
configs/bench/infer_seq2seq_scaling.yaml     # T5 scaling test
configs/bench/infer_encoder_scaling.yaml     # BERT scaling test
```

**Configuration Example**:
```yaml
name: "infer_decoder_scaling"
arch: "decoder"
context_lengths: [128, 256, 512, 1024]  # Test multiple lengths
num_tokens: 50
num_runs: 3
warmup_runs: 1
```

### 2. Test Script

```
scripts/bench_scaling.sh  # Run scaling tests for all architectures
```

## Usage

### Method 1: Use Test Script (Recommended)

Run scaling tests for all architectures:

```bash
./scripts/bench_scaling.sh
```

This will:
1. Test GPT-2, T5, and BERT scaling performance
2. Automatically aggregate results
3. Generate latency curve plots

### Method 2: Test Individual Architecture

#### Test GPT-2 (Decoder)
```bash
python -m src.e3bench.eval.bench_infer \
    --model_cfg configs/model/gpt2-medium.yaml \
    --bench_cfg configs/bench/infer_decoder_scaling.yaml \
    --output_dir results
```

#### Test T5 (Encoder-Decoder)
```bash
python -m src.e3bench.eval.bench_infer \
    --model_cfg configs/model/t5-base.yaml \
    --bench_cfg configs/bench/infer_seq2seq_scaling.yaml \
    --output_dir results
```

#### Test BERT (Encoder)
```bash
python -m src.e3bench.eval.bench_infer \
    --model_cfg configs/model/bert-base.yaml \
    --bench_cfg configs/bench/infer_encoder_scaling.yaml \
    --output_dir results
```

### Method 3: Manual Aggregation and Plotting

```bash
# Aggregate results
python -m src.e3bench.report.aggregate --results_dir latest --out_dir tables

# Generate plots
python -m src.e3bench.report.plots --tables tables --out figs
```

## Output Results

### 1. CSV Data

`tables/inference_results.csv` contains detailed data for each context length:

| exp_id | model | arch | context_length | latency_ms | throughput | max_memory_gb |
|--------|-------|------|----------------|------------|------------|---------------|
| inference_scaling_xxx | gpt2 | decoder | 128 | 5.2 | 96.2 | 0.25 |
| inference_scaling_xxx | gpt2 | decoder | 256 | 8.7 | 87.3 | 0.28 |
| inference_scaling_xxx | gpt2 | decoder | 512 | 15.4 | 64.9 | 0.35 |
| ... | ... | ... | ... | ... | ... | ... |

### 2. Latency Curve Plot

`figs/latency_curve.png` contains two subplots:

**Left: Latency vs Context Length**
- X-axis: Context Length (tokens)
- Y-axis: Latency (ms)
- Three curves for Encoder, Decoder, and Encoder-Decoder

**Right: Memory Usage vs Context Length**
- X-axis: Context Length (tokens)
- Y-axis: Peak Memory (GB)
- Shows memory growth with context length

### 3. JSON Raw Data

`results/inference_scaling_YYYYMMDD_HHMMSS.json`:
```json
{
  "exp_id": "inference_scaling_20251111_120000",
  "metrics": {
    "128": {
      "first_token_latency_ms": 5.23,
      "throughput_tokens_per_sec": 96.15,
      "max_memory_gb": 0.25
    },
    "256": {
      "first_token_latency_ms": 8.67,
      "throughput_tokens_per_sec": 87.31,
      "max_memory_gb": 0.28
    },
    ...
  }
}
```

## Code Modifications

### 1. `src/e3bench/eval/bench_infer.py`

**New Functions**:
- `benchmark_with_context_scaling()` - Test across multiple context lengths
- Modified `benchmark_inference()` - Auto-detect scaling tests

**Key Logic**:
```python
# Detect scaling mode
is_scaling = "context_lengths" in bench_config

if is_scaling:
    # Loop through multiple lengths
    for ctx_len in context_lengths:
        metrics = benchmark_decoder_inference(model, tokenizer, ctx_config)
        results_by_length[ctx_len] = metrics
```

### 2. `src/e3bench/report/aggregate.py`

**Modified Functions**:
- `aggregate_inference_results()` - Handle scaling data format

**Key Logic**:
```python
# Detect scaling data
is_scaling = "inference_scaling" in exp_id or \
             any(isinstance(v, dict) for v in metrics.values())

if is_scaling:
    # Generate one row per context length
    for ctx_len, ctx_metrics in metrics.items():
        inference_data.append({
            "context_length": int(ctx_len),
            "latency_ms": ctx_metrics.get("latency_ms", 0),
            ...
        })
```

### 3. `src/e3bench/report/plots.py`

**New Functions**:
- `generate_latency_curve()` - Generate latency curve plots

**Key Features**:
- Auto-detect scaling data
- Different colors and markers for each architecture
- Show error bands
- Dual subplots: latency + memory

## Expected Results

After running scaling tests, you will see:

```
Context Length Scaling Results:
  128 tokens: 5.23ms
  256 tokens: 8.67ms
  512 tokens: 15.41ms
  1024 tokens: 28.93ms
```

**Latency Curve Plot** will clearly show:
- ✅ **BERT (Encoder)**: Flattest curve, slowest growth with length
- 📈 **T5 (Encoder-Decoder)**: Moderate growth rate
- ⚠️ **GPT-2 (Decoder)**: Fastest growth, higher pressure for long texts

## Custom Configuration

### Test Longer Contexts

Modify `context_lengths` in config files:

```yaml
context_lengths: [128, 256, 512, 1024, 2048, 4096]  # Test longer contexts
```

### Adjust Test Precision

```yaml
num_runs: 5        # More runs for better statistics
warmup_runs: 2     # More warmup to reduce initial variance
```

### Generate More Tokens

```yaml
num_tokens: 100    # Generate more tokens to test longer text generation
```

## Important Notes

⚠️ **Memory Limitations**: Testing long contexts (e.g., 4096) may require substantial VRAM
⚠️ **Test Duration**: Testing multiple lengths × multiple runs takes considerable time
⚠️ **GPU Recommended**: CPU testing will be very slow

## Compatibility

✅ **Backward Compatible**: Original single-point inference tests remain unaffected
✅ **Mixed Data**: Can retain both single-point and scaling test results
✅ **Auto-Detection**: Reporting tools automatically identify data types

## Troubleshooting

### Issue 1: Module Not Found
```bash
ModuleNotFoundError: No module named 'e3bench'
```
**Solution**: Use `src.e3bench` instead of `e3bench`

### Issue 2: Out of Memory
```bash
CUDA out of memory
```
**Solution**: Reduce maximum context length tested, or decrease batch_size

### Issue 3: No Latency Curve Generated
**Check**:
1. Do you have scaling data (multiple context lengths)?
2. Does `tables/inference_results.csv` contain `context_length` column?
3. Are there `inference_scaling` experiment IDs?

## Example Output Structure

After complete run:

```
E3-mini-benchmark/
├── results/
│   ├── inference_scaling_20251111_120000/  # GPT-2 scaling
│   ├── inference_scaling_20251111_120530/  # T5 scaling
│   └── inference_scaling_20251111_121045/  # BERT scaling
├── latest/inference/
│   ├── gpt2.json                           # Latest GPT-2 results
│   ├── t5-base.json                        # Latest T5 results
│   └── bert-base-uncased.json              # Latest BERT results
├── tables/
│   └── inference_results.csv               # Aggregated data (all lengths)
└── figs/
    ├── latency_curve.png                   # NEW: Latency curve plot ✨
    ├── inference_performance.png           # Existing: Performance comparison
    └── inference_memory_tradeoff.png       # Existing: Memory vs performance
```

## Summary

This feature enables you to:

✅ **Quantitative Assessment** - Precisely measure scaling characteristics of different architectures
✅ **Performance Prediction** - Predict performance for longer contexts based on curves
✅ **Architecture Comparison** - Visually compare architectures for different use cases
✅ **Memory Planning** - Understand memory requirements as input length grows

This perfectly aligns with the E³ Benchmark **Efficiency Evaluation** goals!
