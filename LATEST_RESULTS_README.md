# Latest Results Directory

## Overview

The `latest/` directory maintains only the most recent experiment results for each model/architecture combination. This prevents duplicate entries in aggregated tables and plots.

## Directory Structure

```
latest/
├── cont_pretrain/
│   ├── bert-base-uncased.json
│   ├── gpt2.json
│   └── t5-base.json
├── superglue/
│   ├── bert-base-uncased.json
│   ├── gpt2.json
│   └── t5-base.json
├── fewshot/
│   ├── bert-base-uncased-0shot.json
│   ├── bert-base-uncased-5shot.json
│   ├── bert-base-uncased-10shot.json
│   ├── gpt2-0shot.json
│   ├── gpt2-5shot.json
│   ├── gpt2-10shot.json
│   ├── t5-base-0shot.json
│   ├── t5-base-5shot.json
│   └── t5-base-10shot.json
└── inference/
    ├── bert-base-uncased.json
    ├── gpt2.json
    └── t5-base.json
```

## How It Works

### Automatic Syncing

All experiment scripts automatically sync their results to `latest/` after saving to `results/`:

- **Continued Pretraining** (`cont_pretrain.py`): Syncs to `latest/cont_pretrain/{model_name}.json`
- **SuperGLUE Fine-tuning** (`finetune_glue.py`): Syncs to `latest/superglue/{model_name}.json`
- **Few-shot Evaluation** (`eval_fewshot.py`): Syncs to `latest/fewshot/{model_name}-{num_fewshot}shot.json`
- **Inference Benchmarking** (`bench_infer.py`): Syncs to `latest/inference/{model_name}.json`

### File Naming Convention

Files are named by model and experiment type to enable automatic overwriting:

- **Continued Pretraining**: `{model_name}.json` (e.g., `bert-base-uncased.json`)
- **SuperGLUE**: `{model_name}.json`
- **Few-shot**: `{model_name}-{num_fewshot}shot.json` (e.g., `bert-base-uncased-5shot.json`)
- **Inference**: `{model_name}.json`

When you run a new experiment with the same model/type, it automatically replaces the old file in `latest/`.

## Usage

### Running Experiments

Just run experiments as usual. Results are automatically synced to `latest/`:

```bash
make pretrain-bert    # Syncs to latest/cont_pretrain/bert-base-uncased.json
make eval-bert-5shot  # Syncs to latest/fewshot/bert-base-uncased-5shot.json
make train-gpt2       # Syncs to latest/superglue/gpt2.json
```

### Generating Reports

By default, `make report` now uses the `latest/` directory:

```bash
make report    # Aggregates from latest/ (no duplicates)
```

To aggregate all historical results (including duplicates):

```bash
make report-all    # Aggregates from results/ (all history)
```

### Manual Sync

To manually sync existing results to `latest/`, use the provided script:

```bash
python3 scripts/sync_to_latest.py
```

Or use the Python API:

```python
from src.e3bench.utils.io import load_json, sync_to_latest

result = load_json("results/cont_pretrain_20251110_182137.json")
latest_path = sync_to_latest(result)
print(f"Synced to: {latest_path}")
```

## Benefits

1. **No Duplicates**: Each model/experiment combination appears only once in tables and plots
2. **Clean Visualizations**: Plots show single data points per model/shot combination
3. **Historical Preservation**: All runs are still saved in `results/` for reference
4. **Automatic Updates**: New experiments automatically replace old ones in `latest/`

## Comparison: Before vs After

### Before (using `results/`)
- `pretraining_results.csv`: 14 rows (many duplicates)
- `fewshot_results.csv`: 2482 rows (6+ duplicates per model/shot)
- Plots show multiple overlapping points at same x-coordinate

### After (using `latest/`)
- `pretraining_results.csv`: 2 rows (one per architecture)
- `fewshot_results.csv`: 558 rows (one exp_id per model/shot)
- Plots show clean single points per model/shot combination

## Implementation Details

The implementation consists of:

1. **`sync_to_latest()` function** in `src/e3bench/utils/io.py`: Core syncing logic
2. **Experiment scripts**: All training/eval scripts call `sync_to_latest()` after saving
3. **`load_experiment_results()` enhancement** in `src/e3bench/report/aggregate.py`: Supports both flat and organized directory structures
4. **Makefile updates**: `report` uses `latest/`, `report-all` uses `results/`

## Notes

- The `results/` directory is never modified - all historical data is preserved
- The `latest/` directory is automatically created when needed
- Files in `latest/` are overwritten when newer experiments complete
- The sync happens automatically - no manual intervention required


