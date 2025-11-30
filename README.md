# E³ Mini-Benchmark

A general-purpose benchmark designed for evaluating **Small Language Models (SLMs)** across three "E" axes:

1. **Efficiency** - Time dimension (Training speed, Inference latency/throughput)
2. **Energy** - Energy dimension (Training kWh, Inference Joules)
3. **Effectiveness** - Quality (SuperGLUE, MMLU, Generation Quality)

One of the primary tasks demonstrated with this benchmark is the comparison of **encoder-only**, **encoder-decoder**, and **decoder-only** architectures, highlighting tradeoffs in efficiency, energy, and effectiveness. You can view the experimental results for this comparison at [https://ender-600.github.io/llm-arch-compare/](https://ender-600.github.io/llm-arch-compare/).

## Hardware Requirements

- Single/Four/Eight NVIDIA V100 32GB
- CUDA 12.x
- Uses fp16 (no bf16)
- No FlashAttention
- Prefers LoRA for fine-tuning to save VRAM

## Quick Start

### Installation

```bash
# Install dependencies
make env

# Or manually:
pip install -r requirements.txt
```

### Quick Reproduction

1. **SuperGLUE Fine-tuning (LoRA)**
   ```bash
   make superglue-finetune
   ```

2. **Few-shot Evaluation**
   ```bash
   make eval
   ```

3. **Inference Benchmarking**
   ```bash
   make infer
   ```

4. **Generate Reports and Figures**
   ```bash
   make report
   make figs
   ```

## Project Structure

```
E3-mini-benchmark/
├─ configs/          # YAML configuration files
├─ src/e3bench/      # Main Python package
├─ scripts/          # Shell wrapper scripts
├─ results/          # JSON outputs (created at runtime)
├─ tables/           # Aggregated CSV tables (created at runtime)
└─ figs/             # Generated figures (created at runtime)
```

## Outputs

- **results/**: Individual experiment JSON files with metrics, timing, and resource usage
- **tables/**: Aggregated CSV tables for analysis
- **figs/**: Visualization plots for efficiency and effectiveness comparisons

## E³ Evaluation Framework

Efficiency: Training Tokens/sec, Inference Latency (TTFT, TBT).

Energy: Training to Convergence Energy (kWh), Inference Energy per Sample (Joules).

Effectiveness: SuperGLUE (NLU), MMLU (Knowledge), Generation Quality.

While Efficiency and Energy are physically related, Efficiency focuses on the "time dimension" (latency, throughput, time-to-X), whereas Energy focuses specifically on the "energy consumption dimension" (kWh, J/token). Although related, they answer different questions: "How fast can it finish?" vs "How much electricity/cost does it take to finish?"

## License

MIT License - see LICENSE file for details.
