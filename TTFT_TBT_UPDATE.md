# TTFT/TBT Latency Metrics Update

## Update Overview

This update adds fine-grained inference latency metrics to E3-mini-benchmark: **TTFT** and **TBT**, for more accurate evaluation of LLM inference performance.

## Core Changes

### 1. New Latency Metrics

- **TTFT (Time-To-First-Token)**: Time from request to first token
  - Includes: Queue/Scheduling + Prefill (full context attention) + First token sampling
  - Reflects user-perceived "response speed"

- **TBT (Time-Between-Tokens)**: Average interval for subsequent tokens
  - Includes: Single decode step (incremental attention + sampling)
  - Reflects "generation fluency"

- **E2E (End-to-End)**: Total latency
  - Formula: `E2E ≈ TTFT + (m-1) × TBT`

### 2. Modified Files

#### Core Code
- ✅ `src/e3bench/eval/bench_infer.py`
  - Rewrote `benchmark_decoder_inference()` - Token-by-token measurement
  - Rewrote `benchmark_seq2seq_inference()` - Separate encoder and decoder measurement
  - Added TTFT, TBT, E2E metric outputs

- ✅ `src/e3bench/report/aggregate.py`
  - Updated `aggregate_inference_results()` - Support for new metrics
  - Added fields: `ttft_ms`, `tbt_ms`, `e2e_latency_ms`, `encoder_latency_ms`

- ✅ `src/e3bench/report/plots.py`
  - Added `generate_ttft_tbt_curves()` - Triple plot (TTFT/TBT/E2E)
  - Added `generate_latency_composition_chart()` - Latency composition stacked chart

#### Documentation
- ✅ `docs/ttft_tbt_metrics.md` - Complete metrics documentation (NEW)
- ✅ `TTFT_TBT_UPDATE.md` - This update note (NEW)

## Backward Compatibility

✅ **Fully Backward Compatible**
- Retained `first_token_latency_ms` (now equivalent to `ttft_ms`)
- Retained `latency_ms` (for traditional aggregation)
- Old CSV columns still exist
- Old charts still generated

## Usage

### Quick Start

```bash
# 1. Run inference test (automatically uses new TTFT/TBT measurement)
python -m src.e3bench.eval.bench_infer \
    --model_cfg configs/model/gpt2-medium.yaml \
    --bench_cfg configs/bench/infer_decoder_scaling.yaml \
    --output_dir results

# 2. Aggregate results
python -m src.e3bench.report.aggregate --results_dir latest --out_dir tables

# 3. Generate plots
python -m src.e3bench.report.plots --tables tables --out figs
```

### New Charts

After running, the following will be generated in `figs/`:
- **`ttft_tbt_curves.png`** - TTFT/TBT/E2E triple curve plot
- **`latency_composition.png`** - Latency composition stacked bar chart

### Output Metrics Example

```json
{
  "ttft_ms": 145.23,              // NEW: Time-To-First-Token
  "ttft_std_ms": 5.67,            // NEW: TTFT standard deviation
  "tbt_ms": 8.45,                 // NEW: Time-Between-Tokens
  "tbt_std_ms": 0.32,             // NEW: TBT standard deviation
  "e2e_latency_ms": 567.89,       // NEW: End-to-End latency
  "e2e_std_ms": 12.34,            // NEW: E2E standard deviation
  "encoder_latency_ms": 98.76,    // NEW: Encoder time (seq2seq only)
  "throughput_tokens_per_sec": 118.3,
  "first_token_latency_ms": 145.23,  // Backward compatible (equals ttft_ms)
  "latency_std_ms": 5.67          // Backward compatible
}
```

## Expected Performance Characteristics

### Latency Characteristics by Architecture

| Architecture | TTFT Characteristic | TBT Characteristic | Context Impact |
|--------------|---------------------|--------------------| ---------------|
| **Decoder (GPT-2)** | Medium | Stable | TTFT linear growth ⚠️ |
| **Encoder-Decoder (T5)** | Higher | Medium | TTFT dominated by Encoder 📈 |
| **Encoder (BERT)** | N/A | N/A | Forward pass only ✅ |

### Example Data (GPT-2, 50 tokens generated)

```
Context Length    TTFT      TBT      E2E
128 tokens       45ms     8ms      437ms
512 tokens       180ms    9ms      621ms
1024 tokens      360ms    10ms     850ms
```

Formula verification: `437 ≈ 45 + 49 × 8 = 437ms` ✅

## Technical Details

### Token-by-Token Measurement

Previous implementation:
```python
# Old method: Generate all at once, only get average latency
outputs = model.generate(max_new_tokens=50)
avg_latency = total_time / 50  # Cannot distinguish TTFT and TBT
```

New implementation:
```python
# New method: Generate token-by-token, accurately measure each token
for i in range(num_tokens):
    token_start = time.time()
    outputs = model(input_ids)
    next_token = sample(outputs)
    token_times.append(time.time() - token_start)
    input_ids = append(input_ids, next_token)

ttft = token_times[0]        # First token (includes prefill)
tbt = mean(token_times[1:])  # Subsequent tokens average (pure decode)
```

### Seq2seq Special Handling

For T5 and similar models, additionally measure Encoder time:

```python
# 1. Separately measure Encoder
encoder_start = time.time()
encoder_outputs = model.encoder(input_ids)
encoder_latency = time.time() - encoder_start

# 2. Measure Decoder token-by-token
for i in range(num_tokens):
    token_start = time.time()
    outputs = model.decoder(decoder_input_ids, encoder_outputs)
    # ...
    token_times.append(time.time() - token_start)

# TTFT = Encoder + First Decoder token
ttft = encoder_latency + token_times[0]
```

## Performance Optimization Recommendations

### How to Use These Metrics to Optimize Performance

**Scenario 1: TTFT is Bottleneck (Long user wait)**
- Optimize prefill: Use FlashAttention, reduce context length
- Use batching to increase parallelism
- Consider KV-cache prefilling

**Scenario 2: TBT is Bottleneck (Slow generation)**
- Operator fusion, quantization (INT8/FP16)
- Optimize KV-cache management
- Use greedy decoding instead of beam search

**Optimization Priorities by Application:**
- Chatbot: Prioritize TTFT (reduce first word wait)
- Code Generation: Prioritize TBT (ensure smooth output)
- Batch Processing: Prioritize throughput (TBT × batch_size)

## Known Limitations

1. ⚠️ **Performance Overhead**: Token-by-token measurement is slightly slower than one-shot generation (some optimizations disabled)
2. ⚠️ **KV-cache**: Current implementation uses `use_cache=False` in some places to simplify timing
3. ⚠️ **Encoder-only**: BERT and similar models have no TTFT/TBT (don't generate sequences)

Future versions will address these limitations.

## Test Verification

### Run Tests

```bash
# Quick test (small model, short context)
./scripts/bench_scaling.sh

# Expected output
INFO Running inference benchmark with TTFT/TBT measurement...
INFO Generated 50 tokens
INFO TTFT: 45.23ms, TBT: 8.45ms, E2E: 437.89ms

# Check generated charts
ls figs/
# Should see:
#   ttft_tbt_curves.png
#   latency_composition.png
```

### Validate Formula

Check if E2E follows the formula:
```python
# Read from CSV
df = pd.read_csv('tables/inference_results.csv')

# Validate: E2E ≈ TTFT + (m-1) × TBT
num_tokens = 50
expected_e2e = df['ttft_ms'] + (num_tokens - 1) * df['tbt_ms']
actual_e2e = df['e2e_latency_ms']

# Error should be < 5%
error = abs(expected_e2e - actual_e2e) / actual_e2e
assert error.mean() < 0.05
```

## Related Documentation

- 📖 [TTFT/TBT Metrics Guide](docs/ttft_tbt_metrics.md) - Complete documentation
- 📖 [Latency Curve Feature](docs/latency_curve_feature.md) - Context scaling tests
- 📖 [README](README.md) - Main project documentation

## Version Information

- **Update Date**: 2025-11-13
- **Version**: v1.0
- **Compatibility**: Backward compatible with all old configurations and scripts

---

For questions or suggestions, please refer to [docs/ttft_tbt_metrics.md](docs/ttft_tbt_metrics.md) or submit an issue.
