# E³ Mini-Benchmark

A comprehensive benchmark for comparing **encoder-only**, **encoder-decoder**, and **decoder-only** language models across three "E" axes:

1. **Efficiency (Training)** - Continued pretraining proxy with equal tokens
2. **Efficiency (Inference)** - Latency/throughput/VRAM benchmarking  
3. **Effectiveness** - NLU fine-tuning (SuperGLUE) and zero/few-shot evaluation (MMLU/ARC/HellaSwag)

## Hardware Requirements

- Single NVIDIA V100 32GB
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

### Quick Reproduction (3 Commands)

1. **SuperGLUE Fine-tuning (LoRA)**
   ```bash
   make train
   ```

2. **Few-shot Evaluation**
   ```bash
   make eval
   ```

3. **Inference Benchmarking**
   ```bash
   make bench
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

## E³ Definition

- **Efficiency (Training)**: Time and energy to reach target validation loss during continued pretraining
- **Efficiency (Inference)**: Latency, throughput, and VRAM usage during inference
- **Effectiveness**: Performance on downstream tasks via fine-tuning and few-shot evaluation

## License

MIT License - see LICENSE file for details.
