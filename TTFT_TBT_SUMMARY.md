# TTFT/TBT Implementation Summary

## What's New

Added fine-grained latency breakdown for LLM inference benchmarking:

### New Metrics
- **TTFT (Time-To-First-Token)**: Request → First token (includes prefill)
- **TBT (Time-Between-Tokens)**: Average time per subsequent token (decode only)
- **E2E (End-to-End)**: Total generation time = TTFT + (m-1) × TBT

### Key Files Modified
1. **`src/e3bench/eval/bench_infer.py`** - Token-by-token measurement implementation
2. **`src/e3bench/report/aggregate.py`** - New metrics in CSV output
3. **`src/e3bench/report/plots.py`** - TTFT/TBT visualization charts

### New Documentation
- **`docs/ttft_tbt_metrics.md`** - Complete guide (436 lines)
- **`TTFT_TBT_UPDATE.md`** - Quick update reference
- **`test_ttft_tbt.sh`** - Automated testing script

## Quick Test

```bash
# Run quick validation test
./test_ttft_tbt.sh

# Or run full scaling test
./scripts/bench_scaling.sh
```

## Architecture Comparison

| Architecture | TTFT Growth | TBT Stability | Best Use Case |
|--------------|-------------|---------------|---------------|
| GPT-2 (Decoder) | Linear with context ⚠️ | Very stable ✅ | General generation |
| T5 (Encoder-Decoder) | High (encoder cost) 📈 | Moderate | Long input → Short output |
| BERT (Encoder) | N/A | N/A | Classification only |

## Output Example

```json
{
  "ttft_ms": 145.23,
  "tbt_ms": 8.45,
  "e2e_latency_ms": 567.89,
  "encoder_latency_ms": 98.76,  // T5 only
  "throughput_tokens_per_sec": 118.3
}
```

## Backward Compatibility

✅ All existing code continues to work  
✅ Old metrics (`first_token_latency_ms`) still present  
✅ All previous charts still generated

## Next Steps

1. Read full docs: `docs/ttft_tbt_metrics.md`
2. Run tests: `./test_ttft_tbt.sh`
3. View results: Check `figs/ttft_tbt_curves.png`

---

**Version**: 1.0 | **Date**: 2025-11-13

