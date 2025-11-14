# TTFT and TBT Latency Metrics Guide

## Overview

Traditional average latency metrics cannot adequately reflect the performance characteristics of different inference stages in LLM evaluation. This document details the new latency breakdown metrics: **TTFT (Time-To-First-Token)** and **TBT (Time-Between-Tokens)**.

## Core Concepts

### 1. TTFT (Time-To-First-Token)

**Definition**: The time from sending a request until the first generated token appears.

**Components**:
```
TTFT = Queue/Scheduling + Prefill (full context attention) + First token sampling
```

- **Queue/Scheduling**: Request queuing and scheduling time on the server
- **Prefill**: Encode the entire prompt into the model, perform full context attention
- **First Token Sampling**: Sample and generate the first token based on prefill results

**Influencing Factors**:
- ✅ Context Length: Most significant impact
- ✅ Batch Concurrency: Effect when processing multiple requests simultaneously
- ✅ KV-cache Hit Rate: Cache hits can significantly reduce prefill cost
- ✅ Compilation/Cold Start: Additional overhead on first run

**Physical Meaning**:
- Reflects user-perceived "response speed"
- Greatest impact on interactive applications (e.g., chatbots)
- In long-context scenarios, TTFT can become the bottleneck

### 2. TBT (Time-Between-Tokens)

**Definition**: Average generation interval for each subsequent token (also called inter-token latency).

**Components**:
```
TBT = Single decode step = Single token attention + Sampling
```

- **Decode Phase**: Only needs to look at "next-step attention" with existing KV-cache
- **Complexity**: Approximately O(L·d), where L is number of layers, d is model width
- **Context Impact**: Relatively small impact from context length (due to KV-cache)

**Influencing Factors**:
- ✅ Sampling Strategy (top-k/p, beam search)
- ✅ Parallelism (batch processing, tensor parallelism)
- ✅ Operator Fusion
- ✅ KV-cache Management Efficiency

**Physical Meaning**:
- Reflects "generation fluency"
- For streaming output scenarios, TBT determines the speed users see text
- Stable TBT means better user experience

### 3. E2E (End-to-End Latency)

**Definition**: Total latency to generate m new tokens.

**Formula**:
```
E2E ≈ TTFT + (m-1) × TBT + Overhead
```

Where:
- `m` is the number of generated tokens
- Overhead includes final token decoding, stop condition checking, etc.

### 4. Throughput

**Calculation**:
```
Throughput (tokens/s) ≈ 1 / TBT  (single request, steady state)
```

For batch processing scenarios:
```
Batch Throughput = batch_size / TBT
```

## Architecture Differences

Different model architectures show significant differences in TTFT and TBT:

### Decoder-only (GPT-2)

```
TTFT: Medium
TBT:  Stable and fast
E2E:  Linear growth (with generation length)
```

**Characteristics**:
- Prefill and Decode use the same architecture
- TTFT grows linearly with context length
- TBT relatively stable, less affected by context length
- KV-cache is crucial for performance

### Encoder-Decoder (T5)

```
TTFT: Higher (includes Encoder processing time)
TBT:  Medium (Decoder-only)
E2E:  Moderate growth
```

**Characteristics**:
- TTFT = Encoder time + First Decoder token
- Encoder only needs to run once (on entire input)
- Decoder generates step-by-step, each step only looks at Decoder history
- Suitable for tasks with long input, short output (e.g., summarization)

**Additional Metrics**:
- `encoder_latency_ms`: Standalone Encoder processing time
- Can analyze Encoder and Decoder performance separately

### Encoder-only (BERT)

```
TTFT: Lowest (no generation process)
TBT:  N/A (doesn't generate tokens)
E2E:  Single forward pass
```

**Characteristics**:
- Only has `forward_pass_latency_ms`
- No TTFT/TBT concept (doesn't generate sequences)
- Latency grows with input length, but no KV-cache

**⚠️ Important**: BERT's `forward_pass_latency_ms` is **NOT comparable** to GPT-2/T5's TTFT:
- BERT measures: Complete forward pass (get all token representations)
- GPT-2/T5 measure: Prefill + generate first token (preparation for generation)
- They measure fundamentally different operations!

## Metric Details

### Output Metrics

After running inference tests, you'll get the following metrics:

#### Decoder / Encoder-Decoder Models

```json
{
  "ttft_ms": 145.23,              // Time-To-First-Token (ms)
  "ttft_std_ms": 5.67,            // TTFT standard deviation
  "tbt_ms": 8.45,                 // Time-Between-Tokens (ms)
  "tbt_std_ms": 0.32,             // TBT standard deviation
  "e2e_latency_ms": 567.89,       // End-to-End total latency (ms)
  "e2e_std_ms": 12.34,            // E2E standard deviation
  "throughput_tokens_per_sec": 118.3,  // Throughput (tokens/s)
  "throughput_std": 4.2,          // Throughput standard deviation
  
  // Backward compatibility
  "first_token_latency_ms": 145.23,    // Equivalent to ttft_ms
  "latency_std_ms": 5.67          // Equivalent to ttft_std_ms
}
```

#### Encoder-Decoder Additional Metrics

```json
{
  "encoder_latency_ms": 98.76,    // Encoder processing time
  "encoder_std_ms": 3.21          // Encoder time standard deviation
}
```

Note: `TTFT = encoder_latency_ms + first_decoder_token_time`

#### Encoder-only Models

```json
{
  "forward_pass_latency_ms": 23.45,  // Forward pass latency
  "latency_std_ms": 1.23         // Standard deviation
}
```

## Usage

### 1. Run Inference Benchmarks

#### Test Single Model

```bash
# Decoder (GPT-2)
python -m src.e3bench.eval.bench_infer \
    --model_cfg configs/model/gpt2-medium.yaml \
    --bench_cfg configs/bench/infer_decoder.yaml \
    --output_dir results

# Encoder-Decoder (T5)
python -m src.e3bench.eval.bench_infer \
    --model_cfg configs/model/t5-base.yaml \
    --bench_cfg configs/bench/infer_seq2seq.yaml \
    --output_dir results
```

#### Test Context Length Scaling

```bash
# GPT-2 context scaling test
python -m src.e3bench.eval.bench_infer \
    --model_cfg configs/model/gpt2-medium.yaml \
    --bench_cfg configs/bench/infer_decoder_scaling.yaml \
    --output_dir results
```

`infer_decoder_scaling.yaml` configuration example:
```yaml
name: "infer_decoder_scaling"
arch: "decoder"
context_lengths: [128, 256, 512, 1024, 2048]  # Multiple context lengths
num_tokens: 50              # Number of tokens to generate
num_runs: 3                 # Number of repeated runs
warmup_runs: 1              # Number of warmup runs
```

### 2. Aggregate Results

```bash
python -m src.e3bench.report.aggregate --results_dir latest --out_dir tables
```

Output: `tables/inference_results.csv`

CSV contains columns:
- `ttft_ms`, `ttft_std_ms`
- `tbt_ms`, `tbt_std_ms`
- `e2e_latency_ms`, `e2e_std_ms`
- `encoder_latency_ms`, `encoder_std_ms` (seq2seq only)
- Plus traditional `latency_ms`, `throughput_tokens_per_sec`, `max_memory_gb`, etc.

### 3. Generate Visualizations

```bash
python -m src.e3bench.report.plots --tables tables --out figs
```

#### Generated Plots

1. **`ttft_tbt_curves.png`** - TTFT/TBT/E2E triple plot
   - Left: TTFT vs Context Length
   - Middle: TBT vs Context Length
   - Right: E2E vs Context Length

2. **`latency_composition.png`** - Latency composition stacked bar chart
   - Shows proportion of TTFT (Prefill) and Decode (TBT × tokens)
   - Grouped by architecture
   - Intuitively shows which phase is the bottleneck

3. **`latency_curve.png`** - Traditional latency curve (backward compatible)
   - Latency vs Context Length
   - Memory vs Context Length

## Performance Optimization Guide

### Optimizing TTFT

If TTFT is the bottleneck (long user wait time):

1. **Reduce Prefill Computation**
   - Use smaller models
   - Reduce input context length
   - Use KV-cache prefilling

2. **Increase Prefill Parallelism**
   - Use larger batch size (batch multiple requests)
   - Tensor Parallelism
   - Increase GPU compute power

3. **Architecture Selection**
   - For long input, short output tasks, consider Encoder-Decoder
   - Avoid Decoder-only for long context scenarios

### Optimizing TBT

If TBT is the bottleneck (slow generation speed):

1. **Improve Decode Efficiency**
   - Operator Fusion
   - Flash Attention / Paged Attention
   - Quantization (INT8/FP16)

2. **KV-cache Optimization**
   - Use efficient KV-cache management (e.g., vLLM)
   - Reduce memory access latency

3. **Sampling Strategy**
   - Use greedy decoding (fastest)
   - Avoid complex beam search

### Balancing TTFT and TBT

For different application scenarios, optimization priorities differ:

| Scenario | TTFT Importance | TBT Importance | Optimization Advice |
|----------|-----------------|----------------|---------------------|
| Chatbot | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | Prioritize TTFT, reduce first word wait |
| Code Generation | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | Prioritize TBT, ensure smooth output |
| Document Summary | ⭐⭐⭐⭐ | ⭐⭐ | Balanced optimization, short output |
| Batch Processing | ⭐⭐ | ⭐⭐⭐⭐⭐ | Prioritize throughput, TBT determines total time |

## Example Results

### Expected Performance Characteristics

#### Changes with Context Length

```
Context: 128 → 512 → 1024 tokens

GPT-2 (Decoder):
  TTFT:  45ms → 180ms → 360ms  (Linear growth ⚠️)
  TBT:   8ms  → 9ms   → 10ms   (Basically stable ✅)

T5 (Encoder-Decoder):
  TTFT:  120ms → 280ms → 520ms (Encoder dominates 📈)
  TBT:   12ms  → 13ms  → 14ms  (Decoder relatively independent ✅)

BERT (Encoder):
  Forward: 15ms → 45ms → 90ms  (No generation process)
```

### Log Output Example

```
INFO Running inference benchmark with TTFT/TBT measurement...
INFO Input: The future of artificial intelligence is...
INFO Generated 50 tokens
INFO TTFT: 145.23ms, TBT: 8.45ms, E2E: 567.89ms
INFO Encoder: 98.76ms, TTFT: 145.23ms, TBT: 12.34ms, E2E: 789.01ms
```

## Technical Implementation Details

### Token-by-Token Generation

To accurately measure TTFT and TBT, we use token-by-token generation:

```python
# Pseudocode
token_times = []
for i in range(num_tokens):
    token_start = time.time()
    
    # Forward pass + sampling
    outputs = model(input_ids)
    next_token = sample(outputs)
    
    token_times.append(time.time() - token_start)
    input_ids = append(input_ids, next_token)

# TTFT: Time for first token (includes prefill)
ttft = token_times[0]

# TBT: Average time for subsequent tokens (pure decode)
tbt = mean(token_times[1:])

# E2E: Total time
e2e = sum(token_times)
```

### Encoder-Decoder Special Handling

For T5 and other seq2seq models:

```python
# 1. Separately measure Encoder
encoder_start = time.time()
encoder_outputs = model.encoder(input_ids)
encoder_time = time.time() - encoder_start

# 2. Measure Decoder token-by-token
for i in range(num_tokens):
    token_start = time.time()
    outputs = model.decoder(decoder_input_ids, encoder_outputs)
    next_token = sample(outputs)
    token_times.append(time.time() - token_start)
    decoder_input_ids = append(decoder_input_ids, next_token)

# TTFT = Encoder time + First Decoder token
ttft = encoder_time + token_times[0]

# TBT = Average time for subsequent Decoder tokens
tbt = mean(token_times[1:])
```

## Backward Compatibility

To maintain compatibility with older versions:

- ✅ Retained `first_token_latency_ms` field (now equivalent to `ttft_ms`)
- ✅ Retained `latency_ms` field (for traditional latency aggregation)
- ✅ Old visualization charts can still be generated
- ✅ CSV tables include all new and old fields

## FAQ

### Q1: Why is there no TBT data?

**A**: TBT only makes sense when generating multiple tokens. If `num_tokens` is set to 1, or all generations encounter EOS at the first token, there will be no TBT data.

### Q2: Why is TTFT higher than traditional first_token_latency?

**A**: The old `first_token_latency_ms` was actually `total_time / num_generated_tokens`, an average. The new TTFT is the true "first token latency", including the complete prefill cost.

### Q3: Do Encoder-only models have TTFT/TBT?

**A**: No. BERT and other Encoder-only models don't generate sequences, only have `forward_pass_latency_ms`.

### Q4: How to understand E2E ≈ TTFT + (m-1) × TBT?

**A**: This is an approximate formula:
- First token: TTFT (prefill + decode)
- Subsequent m-1 tokens: Each takes TBT (decode only)
- Total: TTFT + (m-1) × TBT + small overhead

### Q5: Does use_cache=False affect performance?

**A**: Yes. In the current implementation, to ensure accurate timing, KV-cache is disabled in some places (`use_cache=False`). This causes performance to be slightly lower than production environments. Future versions will optimize this.

## Related Documentation

- [Latency Curve Feature](latency_curve_feature.md) - Context length scaling tests
- [Inference Benchmarking](../README.md#inference-benchmarking) - Inference benchmark overview

## References

- [vLLM: Fast and Easy-to-use LLM Serving](https://github.com/vllm-project/vllm)
- [The Illustrated GPT-2](http://jalammar.github.io/illustrated-gpt2/)
- [T5: Text-to-Text Transfer Transformer](https://arxiv.org/abs/1910.10683)

---

**Last Updated**: 2025-11-13  
**Version**: v1.0
