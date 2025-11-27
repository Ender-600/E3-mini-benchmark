# E³ Mini-Benchmark Makefile

.PHONY: help env superglue-finetune train eval infer report report-all figs clean all \
	eval-all-models eval-all-open-models-5shot eval-all-new-models fewshot-comparison \
	superglue-bert superglue-t5 superglue-gpt2 \
	superglue-distilbert superglue-roberta superglue-modernbert superglue-deberta-v3 superglue-bert-large \
	superglue-bart superglue-flan-t5 superglue-flan-t5-large superglue-t5-large \
	superglue-gpt2-large superglue-gpt-neo-125m superglue-opt-125m superglue-opt-350m \
	superglue-qwen2.5-0.5b superglue-qwen2.5-1.5b superglue-llama3.2-1b superglue-llama3.2-3b superglue-gemma2-2b \
	superglue-open-models superglue-gated-models superglue-all-models \
	eval-bert-0shot eval-bert-5shot eval-bert-10shot \
	eval-gpt2-0shot eval-gpt2-5shot eval-gpt2-10shot \
	eval-t5-0shot eval-t5-5shot eval-t5-10shot \
	eval-distilbert-0shot eval-distilbert-5shot eval-distilbert-10shot \
	eval-roberta-0shot eval-roberta-5shot eval-roberta-10shot \
	infer-new-models infer-all infer-open-models infer-gated-models \
	infer-distilbert infer-roberta infer-modernbert infer-debertav3 infer-bertlarge \
	infer-bart infer-flant5 infer-flant5-large infer-switch-base-8 \
	infer-gpt2-large infer-opt-125m infer-opt-350m infer-qwen2.5-1.5b infer-llama3.2-1b infer-llama3.2-3b infer-gemma2-2b \
	infer-final-selection infer-gpt-neo-125m infer-qwen2.5-0.5b infer-t5-large \
	pretrain-bert pretrain-t5 pretrain-gpt2 \
	pretrain-distilbert pretrain-roberta pretrain-modernbert pretrain-deberta-v3 pretrain-bert-large \
	pretrain-bart pretrain-flan-t5 pretrain-flan-t5-large pretrain-t5-large \
	pretrain-gpt2-large pretrain-gpt-neo-125m pretrain-opt-125m pretrain-opt-350m \
	pretrain-qwen2.5-0.5b pretrain-qwen2.5-1.5b pretrain-llama3.2-1b pretrain-llama3.2-3b pretrain-gemma2-2b \
	pretrain-open-models pretrain-gated-models pretrain-all-models

# Default target
help:
	@echo "E³ Mini-Benchmark - Available targets:"
	@echo "  env              - Install dependencies"
	@echo "  superglue-finetune - Run SuperGLUE fine-tuning (LoRA)"
	@echo "  eval             - Run few-shot evaluation"
	@echo "  infer            - Run inference benchmarking"
	@echo "  report           - Aggregate latest results and run significance tests"
	@echo "  report-all       - Aggregate all historical results"
	@echo "  figs             - Generate plots"
	@echo "  clean            - Clean up generated files"
	@echo "  all              - Run complete benchmark pipeline"
	@echo ""
	@echo "SuperGLUE Fine-tuning (with train/eval power breakdown):"
	@echo "  superglue-bert         - BERT Base"
	@echo "  superglue-t5           - T5 Base"
	@echo "  superglue-gpt2         - GPT-2 Medium"
	@echo "  superglue-distilbert   - DistilBERT"
	@echo "  superglue-roberta      - RoBERTa"
	@echo "  superglue-modernbert   - ModernBERT"
	@echo "  superglue-deberta-v3   - DeBERTa-v3"
	@echo "  superglue-bert-large   - BERT Large"
	@echo "  superglue-bart         - BART"
	@echo "  superglue-flan-t5      - Flan-T5 Base"
	@echo "  superglue-flan-t5-large - Flan-T5 Large"
	@echo "  superglue-t5-large     - T5 Large"
	@echo "  superglue-gpt2-large   - GPT-2 Large"
	@echo "  superglue-gpt-neo-125m - GPT-Neo 125M"
	@echo "  superglue-opt-125m     - OPT 125M"
	@echo "  superglue-opt-350m     - OPT 350M"
	@echo "  superglue-qwen2.5-0.5b - Qwen2.5 0.5B"
	@echo "  superglue-qwen2.5-1.5b - Qwen2.5 1.5B"
	@echo "  superglue-llama3.2-1b  - Llama 3.2 1B (⚠️ Gated: requires HF auth)"
	@echo "  superglue-llama3.2-3b  - Llama 3.2 3B (⚠️ Gated: requires HF auth)"
	@echo "  superglue-gemma2-2b    - Gemma 2 2B (⚠️ Gated: requires HF auth)"
	@echo ""
	@echo "⚠️  Gated models require Hugging Face authentication:"
	@echo "    1. Request access at huggingface.co/MODEL_NAME"
	@echo "    2. Run: huggingface-cli login"
	@echo "    3. Or set: export HF_TOKEN=your_token"
	@echo ""
	@echo "Batch Operations (run multiple models at once):"
	@echo "  superglue-open-models  - Fine-tune all open-access models"
	@echo "  superglue-gated-models - Fine-tune gated models (requires auth)"
	@echo "  superglue-all-models   - Fine-tune ALL models"
	@echo "  infer-open-models      - Benchmark all open-access models"
	@echo "  infer-gated-models     - Benchmark gated models (requires auth)"
	@echo "  pretrain-open-models   - Pretrain all open-access models"
	@echo "  pretrain-gated-models  - Pretrain gated models (requires auth)"
	@echo "  pretrain-all-models    - Pretrain ALL models"
	@echo ""
	@echo "Cross-model comparison:"
	@echo "  eval-all-models        - Run few-shot eval on all models (BERT, GPT-2, T5)"
	@echo "  eval-all-open-models-5shot - Run 5-shot eval on all open models"
	@echo "  eval-all-new-models    - Run eval on all new models (0/5/10 shot)"
	@echo "  fewshot-comparison     - Full few-shot comparison with all models and shots"
	@echo ""
	@echo "Final Selection:"
	@echo "  infer-final-selection  - Run inference benchmark on YOUR SELECTED 9 models"
	@echo ""
	@echo "Inference Benchmark Targets:"
	@echo "  infer-distilbert    - Benchmark DistilBERT inference"
	@echo "  infer-roberta       - Benchmark RoBERTa inference"
	@echo "  infer-modernbert    - Benchmark ModernBERT inference"
	@echo "  infer-bertlarge     - Benchmark BERT Large inference"
	@echo "  infer-bart          - Benchmark BART inference"
	@echo "  infer-flant5        - Benchmark Flan-T5 inference"
	@echo "  infer-switch-base-8 - Benchmark Switch Transformer (MoE) inference"
	@echo "  infer-qwen2.5-1.5b  - Benchmark Qwen2.5 1.5B inference"
	@echo "  infer-llama3.2-1b   - Benchmark Llama 3.2 1B inference"
	@echo "  infer-llama3.2-3b   - Benchmark Llama 3.2 3B inference"
	@echo "  infer-gemma2-2b     - Benchmark Gemma 2 2B inference"
	@echo "  (and more... see Makefile)"
	@echo ""
	@echo "Individual targets:"
	@echo "  eval-bert-*      - BERT few-shot eval (0/5/10 shot)"
	@echo "  eval-gpt2-*      - GPT-2 few-shot eval (0/5/10 shot)"
	@echo "  eval-t5-*        - T5 few-shot eval (0/5/10 shot)"

# Install dependencies
env:
	pip install -r requirements.txt

# Create necessary directories
results/ tables/ figs/:
	mkdir -p $@

# SuperGLUE fine-tuning (default: BERT with LoRA)
superglue-finetune: results/
	@echo "Running SuperGLUE fine-tuning..."
	python -m src.e3bench.train.finetune_glue \
		--model_cfg configs/model/bert-base.yaml \
		--task_cfg configs/task/superglue.yaml \
		--train_cfg configs/train/lora.yaml \
		--output_dir results

# Backward-compatible alias (deprecated)
train: superglue-finetune
	@echo "⚠️  'make train' is deprecated; use 'make superglue-finetune' instead."

# Few-shot evaluation (default: GPT-2 base)
eval: results/
	@echo "Running few-shot evaluation..."
	python -m src.e3bench.eval.eval_fewshot \
		--model_cfg configs/model/gpt2-medium.yaml \
		--eval_cfg configs/eval/fewshot_5.yaml \
		--output_dir results

# Inference benchmarking (default: T5-base)
infer: results/
	@echo "Running inference benchmarking..."
	python -m src.e3bench.eval.bench_infer \
		--model_cfg configs/model/t5-base.yaml \
		--bench_cfg configs/bench/infer_seq2seq.yaml \
		--output_dir results

# Aggregate results and run significance tests (using latest/ directory)
report: tables/
	@echo "Aggregating latest results..."
	python -m src.e3bench.report.aggregate \
		--results_dir latest \
		--out_dir tables
	@echo "Running significance tests..."
	python -m src.e3bench.report.signif_test \
		--tables tables

# Aggregate all historical results (using results/ directory)
report-all: tables/
	@echo "Aggregating all historical results..."
	python -m src.e3bench.report.aggregate \
		--results_dir results \
		--out_dir tables
	@echo "Running significance tests..."
	python -m src.e3bench.report.signif_test \
		--tables tables

# Generate plots
figs: figs/
	@echo "Generating plots..."
	python -m src.e3bench.report.plots \
		--tables tables \
		--out figs

# Complete benchmark pipeline
all: superglue-finetune eval infer report figs
	@echo "Complete benchmark pipeline finished!"

# Clean up generated files
clean:
	rm -rf results/ tables/ figs/
	rm -f temp_lm_eval_results.json
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

# ============================================================
# Batch Targets: Run multiple models at once
# ============================================================

# Run all open-access models (no HF auth required)
superglue-open-models: results/
	@echo "Running SuperGLUE fine-tuning on all open-access models..."
	$(MAKE) superglue-bert
	$(MAKE) superglue-gpt2
	$(MAKE) superglue-t5
	$(MAKE) superglue-distilbert
	$(MAKE) superglue-roberta
	$(MAKE) superglue-modernbert
	$(MAKE) superglue-deberta-v3
	$(MAKE) superglue-bert-large
	$(MAKE) superglue-bart
	$(MAKE) superglue-flan-t5
	$(MAKE) superglue-flan-t5-large
	$(MAKE) superglue-t5-large
	$(MAKE) superglue-gpt2-large
	$(MAKE) superglue-gpt-neo-125m
	$(MAKE) superglue-opt-125m
	$(MAKE) superglue-opt-350m
	$(MAKE) superglue-qwen2.5-0.5b
	$(MAKE) superglue-qwen2.5-1.5b
	@echo "All open-access models completed!"

# Run gated models (requires HF authentication)
superglue-gated-models: results/
	@echo "⚠️  Running gated models - make sure you're authenticated!"
	@echo "If not authenticated, run: huggingface-cli login"
	@read -p "Press Enter to continue or Ctrl+C to cancel..."
	$(MAKE) superglue-llama3.2-1b
	$(MAKE) superglue-llama3.2-3b
	$(MAKE) superglue-gemma2-2b
	@echo "All gated models completed!"

# Run all models (both open and gated)
superglue-all-models: results/
	@echo "Running SuperGLUE fine-tuning on ALL models..."
	$(MAKE) superglue-open-models
	$(MAKE) superglue-gated-models
	@echo "All models completed!"

# ============================================================
# Individual SuperGLUE fine-tuning targets
# ============================================================
superglue-bert: results/
	@echo "SuperGLUE fine-tuning BERT (encoder-only)..."
	python -m src.e3bench.train.finetune_glue \
		--model_cfg configs/model/bert-base.yaml \
		--task_cfg configs/task/superglue.yaml \
		--train_cfg configs/train/lora.yaml \
		--output_dir results
train-bert: superglue-bert
	@echo "⚠️  'make train-bert' is deprecated; use 'make superglue-bert'."

superglue-t5: results/
	@echo "SuperGLUE fine-tuning T5 (encoder-decoder)..."
	python -m src.e3bench.train.finetune_glue \
		--model_cfg configs/model/t5-base.yaml \
		--task_cfg configs/task/superglue.yaml \
		--train_cfg configs/train/lora.yaml \
		--output_dir results
train-t5: superglue-t5
	@echo "⚠️  'make train-t5' is deprecated; use 'make superglue-t5'."

superglue-gpt2: results/
	@echo "SuperGLUE fine-tuning GPT-2 (decoder-only)..."
	python -m src.e3bench.train.finetune_glue \
		--model_cfg configs/model/gpt2-medium.yaml \
		--task_cfg configs/task/superglue.yaml \
		--train_cfg configs/train/lora.yaml \
		--output_dir results
train-gpt2: superglue-gpt2
	@echo "⚠️  'make train-gpt2' is deprecated; use 'make superglue-gpt2'."

# Encoder-only models
superglue-distilbert: results/
	@echo "SuperGLUE fine-tuning DistilBERT..."
	python -m src.e3bench.train.finetune_glue \
		--model_cfg configs/model/distilbert-base.yaml \
		--task_cfg configs/task/superglue.yaml \
		--train_cfg configs/train/lora.yaml \
		--output_dir results

superglue-roberta: results/
	@echo "SuperGLUE fine-tuning RoBERTa..."
	python -m src.e3bench.train.finetune_glue \
		--model_cfg configs/model/roberta-base.yaml \
		--task_cfg configs/task/superglue.yaml \
		--train_cfg configs/train/lora.yaml \
		--output_dir results

superglue-modernbert: results/
	@echo "SuperGLUE fine-tuning ModernBERT..."
	python -m src.e3bench.train.finetune_glue \
		--model_cfg configs/model/modernbert-base.yaml \
		--task_cfg configs/task/superglue.yaml \
		--train_cfg configs/train/lora.yaml \
		--output_dir results

superglue-deberta-v3: results/
	@echo "SuperGLUE fine-tuning DeBERTa-v3..."
	python -m src.e3bench.train.finetune_glue \
		--model_cfg configs/model/deberta-v3-base.yaml \
		--task_cfg configs/task/superglue.yaml \
		--train_cfg configs/train/lora.yaml \
		--output_dir results

superglue-bert-large: results/
	@echo "SuperGLUE fine-tuning BERT Large..."
	python -m src.e3bench.train.finetune_glue \
		--model_cfg configs/model/bert-large.yaml \
		--task_cfg configs/task/superglue.yaml \
		--train_cfg configs/train/lora.yaml \
		--output_dir results

# Encoder-decoder models
superglue-bart: results/
	@echo "SuperGLUE fine-tuning BART..."
	python -m src.e3bench.train.finetune_glue \
		--model_cfg configs/model/bart-base.yaml \
		--task_cfg configs/task/superglue.yaml \
		--train_cfg configs/train/lora.yaml \
		--output_dir results

superglue-flan-t5: results/
	@echo "SuperGLUE fine-tuning Flan-T5 Base..."
	python -m src.e3bench.train.finetune_glue \
		--model_cfg configs/model/flan-t5-base.yaml \
		--task_cfg configs/task/superglue.yaml \
		--train_cfg configs/train/lora.yaml \
		--output_dir results

superglue-flan-t5-large: results/
	@echo "SuperGLUE fine-tuning Flan-T5 Large..."
	python -m src.e3bench.train.finetune_glue \
		--model_cfg configs/model/flan-t5-large.yaml \
		--task_cfg configs/task/superglue.yaml \
		--train_cfg configs/train/lora.yaml \
		--output_dir results

superglue-t5-large: results/
	@echo "SuperGLUE fine-tuning T5 Large..."
	python -m src.e3bench.train.finetune_glue \
		--model_cfg configs/model/t5-large.yaml \
		--task_cfg configs/task/superglue.yaml \
		--train_cfg configs/train/lora.yaml \
		--output_dir results

# Decoder-only models
superglue-gpt2-large: results/
	@echo "SuperGLUE fine-tuning GPT-2 Large..."
	python -m src.e3bench.train.finetune_glue \
		--model_cfg configs/model/gpt2-large.yaml \
		--task_cfg configs/task/superglue.yaml \
		--train_cfg configs/train/lora.yaml \
		--output_dir results

superglue-gpt-neo-125m: results/
	@echo "SuperGLUE fine-tuning GPT-Neo 125M..."
	python -m src.e3bench.train.finetune_glue \
		--model_cfg configs/model/gpt-neo-125m.yaml \
		--task_cfg configs/task/superglue.yaml \
		--train_cfg configs/train/lora.yaml \
		--output_dir results

superglue-opt-125m: results/
	@echo "SuperGLUE fine-tuning OPT 125M..."
	python -m src.e3bench.train.finetune_glue \
		--model_cfg configs/model/opt-125m.yaml \
		--task_cfg configs/task/superglue.yaml \
		--train_cfg configs/train/lora.yaml \
		--output_dir results

superglue-opt-350m: results/
	@echo "SuperGLUE fine-tuning OPT 350M..."
	python -m src.e3bench.train.finetune_glue \
		--model_cfg configs/model/opt-350m.yaml \
		--task_cfg configs/task/superglue.yaml \
		--train_cfg configs/train/lora.yaml \
		--output_dir results

superglue-qwen2.5-0.5b: results/
	@echo "SuperGLUE fine-tuning Qwen2.5 0.5B..."
	python -m src.e3bench.train.finetune_glue \
		--model_cfg configs/model/qwen2.5-0.5b.yaml \
		--task_cfg configs/task/superglue.yaml \
		--train_cfg configs/train/lora.yaml \
		--output_dir results

superglue-qwen2.5-1.5b: results/
	@echo "SuperGLUE fine-tuning Qwen2.5 1.5B..."
	python -m src.e3bench.train.finetune_glue \
		--model_cfg configs/model/qwen2.5-1.5b.yaml \
		--task_cfg configs/task/superglue.yaml \
		--train_cfg configs/train/lora.yaml \
		--output_dir results

superglue-llama3.2-1b: results/
	@echo "SuperGLUE fine-tuning Llama 3.2 1B..."
	python -m src.e3bench.train.finetune_glue \
		--model_cfg configs/model/llama-3.2-1b.yaml \
		--task_cfg configs/task/superglue.yaml \
		--train_cfg configs/train/lora.yaml \
		--output_dir results

superglue-llama3.2-3b: results/
	@echo "SuperGLUE fine-tuning Llama 3.2 3B..."
	python -m src.e3bench.train.finetune_glue \
		--model_cfg configs/model/llama-3.2-3b.yaml \
		--task_cfg configs/task/superglue.yaml \
		--train_cfg configs/train/lora.yaml \
		--output_dir results

superglue-gemma2-2b: results/
	@echo "SuperGLUE fine-tuning Gemma 2 2B..."
	python -m src.e3bench.train.finetune_glue \
		--model_cfg configs/model/gemma-2-2b.yaml \
		--task_cfg configs/task/superglue.yaml \
		--train_cfg configs/train/lora.yaml \
		--output_dir results

# ============================================================
# Cross-Model Few-Shot Evaluation Targets
# ============================================================

# BERT evaluation targets
eval-bert-0shot: results/
	@echo "Running BERT 0-shot evaluation..."
	python -m src.e3bench.eval.eval_fewshot \
		--model_cfg configs/model/bert-base.yaml \
		--eval_cfg configs/eval/fewshot_0.yaml \
		--output_dir results

eval-bert-5shot: results/
	@echo "Running BERT 5-shot evaluation..."
	python -m src.e3bench.eval.eval_fewshot \
		--model_cfg configs/model/bert-base.yaml \
		--eval_cfg configs/eval/fewshot_5.yaml \
		--output_dir results

eval-bert-10shot: results/
	@echo "Running BERT 10-shot evaluation..."
	python -m src.e3bench.eval.eval_fewshot \
		--model_cfg configs/model/bert-base.yaml \
		--eval_cfg configs/eval/fewshot_10.yaml \
		--output_dir results

# GPT-2 evaluation targets
eval-gpt2-0shot: results/
	@echo "Running GPT-2 0-shot evaluation..."
	python -m src.e3bench.eval.eval_fewshot \
		--model_cfg configs/model/gpt2-medium.yaml \
		--eval_cfg configs/eval/fewshot_0.yaml \
		--output_dir results

eval-gpt2-5shot: results/
	@echo "Running GPT-2 5-shot evaluation..."
	python -m src.e3bench.eval.eval_fewshot \
		--model_cfg configs/model/gpt2-medium.yaml \
		--eval_cfg configs/eval/fewshot_5.yaml \
		--output_dir results

eval-gpt2-10shot: results/
	@echo "Running GPT-2 10-shot evaluation..."
	python -m src.e3bench.eval.eval_fewshot \
		--model_cfg configs/model/gpt2-medium.yaml \
		--eval_cfg configs/eval/fewshot_10.yaml \
		--output_dir results

# T5 evaluation targets
eval-t5-0shot: results/
	@echo "Running T5 0-shot evaluation..."
	python -m src.e3bench.eval.eval_fewshot \
		--model_cfg configs/model/t5-base.yaml \
		--eval_cfg configs/eval/fewshot_0.yaml \
		--output_dir results

eval-t5-5shot: results/
	@echo "Running T5 5-shot evaluation..."
	python -m src.e3bench.eval.eval_fewshot \
		--model_cfg configs/model/t5-base.yaml \
		--eval_cfg configs/eval/fewshot_5.yaml \
		--output_dir results

eval-t5-10shot: results/
	@echo "Running T5 10-shot evaluation..."
	python -m src.e3bench.eval.eval_fewshot \
		--model_cfg configs/model/t5-base.yaml \
		--eval_cfg configs/eval/fewshot_10.yaml \
		--output_dir results

# Additional encoder-only model evaluations
eval-distilbert-0shot: results/
	@echo "Running DistilBERT 0-shot evaluation..."
	python -m src.e3bench.eval.eval_fewshot \
		--model_cfg configs/model/distilbert-base.yaml \
		--eval_cfg configs/eval/fewshot_0.yaml \
		--output_dir results

eval-distilbert-5shot: results/
	@echo "Running DistilBERT 5-shot evaluation..."
	python -m src.e3bench.eval.eval_fewshot \
		--model_cfg configs/model/distilbert-base.yaml \
		--eval_cfg configs/eval/fewshot_5.yaml \
		--output_dir results

eval-distilbert-10shot: results/
	@echo "Running DistilBERT 10-shot evaluation..."
	python -m src.e3bench.eval.eval_fewshot \
		--model_cfg configs/model/distilbert-base.yaml \
		--eval_cfg configs/eval/fewshot_10.yaml \
		--output_dir results

eval-roberta-0shot: results/
	@echo "Running RoBERTa 0-shot evaluation..."
	python -m src.e3bench.eval.eval_fewshot \
		--model_cfg configs/model/roberta-base.yaml \
		--eval_cfg configs/eval/fewshot_0.yaml \
		--output_dir results

eval-roberta-5shot: results/
	@echo "Running RoBERTa 5-shot evaluation..."
	python -m src.e3bench.eval.eval_fewshot \
		--model_cfg configs/model/roberta-base.yaml \
		--eval_cfg configs/eval/fewshot_5.yaml \
		--output_dir results

eval-roberta-10shot: results/
	@echo "Running RoBERTa 10-shot evaluation..."
	python -m src.e3bench.eval.eval_fewshot \
		--model_cfg configs/model/roberta-base.yaml \
		--eval_cfg configs/eval/fewshot_10.yaml \
		--output_dir results

# Batch evaluation targets for new models
eval-all-open-models-5shot: results/
	@echo "Running 5-shot evaluation on all open-access models..."
	$(MAKE) eval-bert-5shot
	$(MAKE) eval-gpt2-5shot
	$(MAKE) eval-t5-5shot
	$(MAKE) eval-distilbert-5shot
	$(MAKE) eval-roberta-5shot
	@echo "All open model 5-shot evaluations completed!"

eval-all-new-models: results/
	@echo "Running evaluations on all new models..."
	$(MAKE) eval-distilbert-0shot
	$(MAKE) eval-distilbert-5shot
	$(MAKE) eval-distilbert-10shot
	$(MAKE) eval-roberta-0shot
	$(MAKE) eval-roberta-5shot
	$(MAKE) eval-roberta-10shot
	@echo "All new model evaluations completed!"

# Comprehensive evaluation: all models with 5-shot
eval-all-models: results/
	@echo "Running few-shot evaluation on all models (5-shot)..."
	$(MAKE) eval-bert-5shot
	$(MAKE) eval-gpt2-5shot
	$(MAKE) eval-t5-5shot
	@echo "All models evaluation completed!"

# Full few-shot comparison: all models with all shots
fewshot-comparison: results/
	@echo "Running comprehensive few-shot comparison..."
	@echo "This will evaluate BERT, GPT-2, and T5 with 0, 5, and 10 shots each..."
	$(MAKE) eval-bert-0shot
	$(MAKE) eval-bert-5shot
	$(MAKE) eval-bert-10shot
	$(MAKE) eval-gpt2-0shot
	$(MAKE) eval-gpt2-5shot
	$(MAKE) eval-gpt2-10shot
	$(MAKE) eval-t5-0shot
	$(MAKE) eval-t5-5shot
	$(MAKE) eval-t5-10shot
	@echo "Few-shot comparison completed!"
	@echo "Now aggregating results..."
	$(MAKE) report
	@echo "Generating comparison plots..."
	$(MAKE) figs
	@echo "Complete! Check tables/ and figs/ directories for results."

# Individual inference benchmarking targets
infer-decoder: results/
	@echo "Benchmarking decoder-only model inference..."
	python -m src.e3bench.eval.bench_infer \
		--model_cfg configs/model/gpt2-medium.yaml \
		--bench_cfg configs/bench/infer_decoder.yaml \
		--output_dir results

infer-seq2seq: results/
	@echo "Benchmarking encoder-decoder model inference..."
	python -m src.e3bench.eval.bench_infer \
		--model_cfg configs/model/t5-base.yaml \
		--bench_cfg configs/bench/infer_seq2seq.yaml \
		--output_dir results

infer-encoder: results/
	@echo "Benchmarking encoder-only model inference..."
	python -m src.e3bench.eval.bench_infer \
		--model_cfg configs/model/bert-base.yaml \
		--bench_cfg configs/bench/infer_encoder.yaml \
		--output_dir results

# ============================================================
# Individual Model Inference Benchmark Targets
# ============================================================

# Encoder Models
infer-distilbert: results/
	@echo "Benchmarking DistilBERT inference..."
	python -m src.e3bench.eval.bench_infer \
		--model_cfg configs/model/distilbert-base.yaml \
		--bench_cfg configs/bench/infer_encoder.yaml \
		--output_dir results

infer-roberta: results/
	@echo "Benchmarking RoBERTa inference..."
	python -m src.e3bench.eval.bench_infer \
		--model_cfg configs/model/roberta-base.yaml \
		--bench_cfg configs/bench/infer_encoder.yaml \
		--output_dir results

infer-modernbert: results/
	@echo "Benchmarking ModernBERT inference..."
	python -m src.e3bench.eval.bench_infer \
		--model_cfg configs/model/modernbert-base.yaml \
		--bench_cfg configs/bench/infer_encoder.yaml \
		--output_dir results

infer-bertlarge: results/
	@echo "Benchmarking BERT Large inference..."
	python -m src.e3bench.eval.bench_infer \
		--model_cfg configs/model/bert-large.yaml \
		--bench_cfg configs/bench/infer_encoder.yaml \
		--output_dir results

infer-debertav3: results/
	@echo "Benchmarking DeBERTa-v3 inference..."
	python -m src.e3bench.eval.bench_infer \
		--model_cfg configs/model/deberta-v3-base.yaml \
		--bench_cfg configs/bench/infer_encoder.yaml \
		--output_dir results

# Decoder Models
infer-gpt2-large: results/
	@echo "Benchmarking GPT-2 Large inference..."
	python -m src.e3bench.eval.bench_infer \
		--model_cfg configs/model/gpt2-large.yaml \
		--bench_cfg configs/bench/infer_decoder.yaml \
		--output_dir results

infer-opt-125m: results/
	@echo "Benchmarking OPT-125m inference..."
	python -m src.e3bench.eval.bench_infer \
		--model_cfg configs/model/opt-125m.yaml \
		--bench_cfg configs/bench/infer_decoder.yaml \
		--output_dir results

infer-opt-350m: results/
	@echo "Benchmarking OPT-350m inference..."
	python -m src.e3bench.eval.bench_infer \
		--model_cfg configs/model/opt-350m.yaml \
		--bench_cfg configs/bench/infer_decoder.yaml \
		--output_dir results

infer-qwen2.5-1.5b: results/
	@echo "Benchmarking Qwen2.5-1.5B inference..."
	python -m src.e3bench.eval.bench_infer \
		--model_cfg configs/model/qwen2.5-1.5b.yaml \
		--bench_cfg configs/bench/infer_decoder.yaml \
		--output_dir results

infer-llama3.2-1b: results/
	@echo "Benchmarking Llama-3.2-1B inference..."
	python -m src.e3bench.eval.bench_infer \
		--model_cfg configs/model/llama-3.2-1b.yaml \
		--bench_cfg configs/bench/infer_decoder.yaml \
		--output_dir results

infer-llama3.2-3b: results/
	@echo "Benchmarking Llama-3.2-3B inference..."
	python -m src.e3bench.eval.bench_infer \
		--model_cfg configs/model/llama-3.2-3b.yaml \
		--bench_cfg configs/bench/infer_decoder.yaml \
		--output_dir results

infer-gemma2-2b: results/
	@echo "Benchmarking Gemma-2-2B inference..."
	python -m src.e3bench.eval.bench_infer \
		--model_cfg configs/model/gemma-2-2b.yaml \
		--bench_cfg configs/bench/infer_decoder.yaml \
		--output_dir results

infer-gpt-neo-125m: results/
	@echo "Benchmarking GPT-Neo 125M inference..."
	python -m src.e3bench.eval.bench_infer \
		--model_cfg configs/model/gpt-neo-125m.yaml \
		--bench_cfg configs/bench/infer_decoder.yaml \
		--output_dir results

infer-qwen2.5-0.5b: results/
	@echo "Benchmarking Qwen2.5 0.5B inference..."
	python -m src.e3bench.eval.bench_infer \
		--model_cfg configs/model/qwen2.5-0.5b.yaml \
		--bench_cfg configs/bench/infer_decoder.yaml \
		--output_dir results

# Encoder-Decoder Models
infer-bart: results/
	@echo "Benchmarking BART inference..."
	python -m src.e3bench.eval.bench_infer \
		--model_cfg configs/model/bart-base.yaml \
		--bench_cfg configs/bench/infer_seq2seq.yaml \
		--output_dir results

infer-flant5: results/
	@echo "Benchmarking Flan-T5 inference..."
	python -m src.e3bench.eval.bench_infer \
		--model_cfg configs/model/flan-t5-base.yaml \
		--bench_cfg configs/bench/infer_seq2seq.yaml \
		--output_dir results

infer-flant5-large: results/
	@echo "Benchmarking Flan-T5 Large inference..."
	python -m src.e3bench.eval.bench_infer \
		--model_cfg configs/model/flan-t5-large.yaml \
		--bench_cfg configs/bench/infer_seq2seq.yaml \
		--output_dir results

infer-switch-base-8: results/
	@echo "Benchmarking Switch Transformer (MoE) inference..."
	python -m src.e3bench.eval.bench_infer \
		--model_cfg configs/model/switch-base-8.yaml \
		--bench_cfg configs/bench/infer_seq2seq.yaml \
		--output_dir results

infer-t5-large: results/
	@echo "Benchmarking T5 Large inference..."
	python -m src.e3bench.eval.bench_infer \
		--model_cfg configs/model/t5-large.yaml \
		--bench_cfg configs/bench/infer_seq2seq.yaml \
		--output_dir results

# Comprehensive inference benchmark for NEW models (calls individual targets)
infer-new-models: results/
	$(MAKE) infer-distilbert
	$(MAKE) infer-roberta
	$(MAKE) infer-modernbert
	$(MAKE) infer-bertlarge
	$(MAKE) infer-debertav3
	$(MAKE) infer-gpt2-large
	$(MAKE) infer-opt-125m
	$(MAKE) infer-opt-350m
	$(MAKE) infer-qwen2.5-1.5b
	$(MAKE) infer-llama3.2-1b
	$(MAKE) infer-llama3.2-3b
	$(MAKE) infer-gemma2-2b
	$(MAKE) infer-bart
	$(MAKE) infer-flant5
	$(MAKE) infer-flant5-large
	$(MAKE) infer-switch-base-8
	@echo "Inference benchmarking of NEW models completed!"

# ============================================================
# FINAL SELECTION INFERENCE BENCHMARK (The 9 chosen models)
# ============================================================
infer-final-selection: results/
	@echo "=== Running Final Selection Inference Benchmark (9 Models) ==="
	# Encoder-only (3)
	@echo "--- 1/9: BERT Base ---"
	$(MAKE) infer-encoder
	@echo "--- 2/9: RoBERTa Base ---"
	$(MAKE) infer-roberta
	@echo "--- 3/9: ModernBERT Base ---"
	$(MAKE) infer-modernbert
	
	# Decoder-only (3)
	@echo "--- 4/9: GPT-2 Medium ---"
	$(MAKE) infer-decoder
	@echo "--- 5/9: GPT-Neo 125M ---"
	$(MAKE) infer-gpt-neo-125m
	@echo "--- 6/9: Qwen2.5 0.5B ---"
	$(MAKE) infer-qwen2.5-0.5b
	
	# Encoder-Decoder (3)
	@echo "--- 7/9: T5 Base ---"
	$(MAKE) infer-seq2seq
	@echo "--- 8/9: T5 Large ---"
	$(MAKE) infer-t5-large
	@echo "--- 9/9: BART Base ---"
	$(MAKE) infer-bart
	@echo "=== Final Selection Inference Benchmark Completed! ==="

# Comprehensive inference benchmark for ALL models
infer-all: results/
	$(MAKE) infer-encoder
	$(MAKE) infer-decoder
	$(MAKE) infer-seq2seq
	$(MAKE) infer-new-models
	@echo "Inference benchmarking of ALL models completed!"

# Inference benchmark for open-access models only
infer-open-models: results/
	@echo "Running inference benchmark on open-access models..."
	$(MAKE) infer-encoder
	$(MAKE) infer-decoder
	$(MAKE) infer-seq2seq
	$(MAKE) infer-distilbert
	$(MAKE) infer-roberta
	$(MAKE) infer-modernbert
	$(MAKE) infer-debertav3
	$(MAKE) infer-bertlarge
	$(MAKE) infer-bart
	$(MAKE) infer-flant5
	$(MAKE) infer-flant5-large
	$(MAKE) infer-t5-large
	$(MAKE) infer-gpt2-large
	$(MAKE) infer-gpt-neo-125m
	$(MAKE) infer-opt-125m
	$(MAKE) infer-opt-350m
	$(MAKE) infer-qwen2.5-0.5b
	$(MAKE) infer-qwen2.5-1.5b
	@echo "Open-access model inference completed!"

# Inference benchmark for gated models only
infer-gated-models: results/
	@echo "⚠️  Running inference on gated models - make sure you're authenticated!"
	@echo "If not authenticated, run: huggingface-cli login"
	@read -p "Press Enter to continue or Ctrl+C to cancel..."
	$(MAKE) infer-llama3.2-1b
	$(MAKE) infer-llama3.2-3b
	$(MAKE) infer-gemma2-2b
	@echo "Gated model inference completed!"

# Continued pretraining targets
# All models use the same token budget (1M tokens) for fair comparison
pretrain-bert: results/
	@echo "Continued pretraining BERT..."
	python -m src.e3bench.train.cont_pretrain \
		--model_cfg configs/model/bert-base.yaml \
		--train_cfg configs/train/lora.yaml \
		--dataset wikitext \
		--target_loss 2.0 \
		--token_budget 1000000 \
		--output_dir results

pretrain-t5: results/
	@echo "Continued pretraining T5..."
	python -m src.e3bench.train.cont_pretrain \
		--model_cfg configs/model/t5-base.yaml \
		--train_cfg configs/train/lora.yaml \
		--dataset wikitext \
		--target_loss 2.0 \
		--token_budget 1000000 \
		--output_dir results

pretrain-gpt2: results/
	@echo "Continued pretraining GPT-2..."
	python -m src.e3bench.train.cont_pretrain \
		--model_cfg configs/model/gpt2-medium.yaml \
		--train_cfg configs/train/lora.yaml \
		--dataset wikitext \
		--target_loss 2.0 \
		--token_budget 1000000 \
		--output_dir results

# Encoder-only models pretraining
pretrain-distilbert: results/
	@echo "Continued pretraining DistilBERT..."
	python -m src.e3bench.train.cont_pretrain \
		--model_cfg configs/model/distilbert-base.yaml \
		--train_cfg configs/train/lora.yaml \
		--dataset wikitext \
		--target_loss 2.0 \
		--token_budget 1000000 \
		--output_dir results

pretrain-roberta: results/
	@echo "Continued pretraining RoBERTa..."
	python -m src.e3bench.train.cont_pretrain \
		--model_cfg configs/model/roberta-base.yaml \
		--train_cfg configs/train/lora.yaml \
		--dataset wikitext \
		--target_loss 2.0 \
		--token_budget 1000000 \
		--output_dir results

pretrain-modernbert: results/
	@echo "Continued pretraining ModernBERT..."
	python -m src.e3bench.train.cont_pretrain \
		--model_cfg configs/model/modernbert-base.yaml \
		--train_cfg configs/train/lora.yaml \
		--dataset wikitext \
		--target_loss 2.0 \
		--token_budget 1000000 \
		--output_dir results

pretrain-deberta-v3: results/
	@echo "Continued pretraining DeBERTa-v3..."
	python -m src.e3bench.train.cont_pretrain \
		--model_cfg configs/model/deberta-v3-base.yaml \
		--train_cfg configs/train/lora.yaml \
		--dataset wikitext \
		--target_loss 2.0 \
		--token_budget 1000000 \
		--output_dir results

pretrain-bert-large: results/
	@echo "Continued pretraining BERT Large..."
	python -m src.e3bench.train.cont_pretrain \
		--model_cfg configs/model/bert-large.yaml \
		--train_cfg configs/train/lora.yaml \
		--dataset wikitext \
		--target_loss 2.0 \
		--token_budget 1000000 \
		--output_dir results

# Encoder-decoder models pretraining
pretrain-bart: results/
	@echo "Continued pretraining BART..."
	python -m src.e3bench.train.cont_pretrain \
		--model_cfg configs/model/bart-base.yaml \
		--train_cfg configs/train/lora.yaml \
		--dataset wikitext \
		--target_loss 2.0 \
		--token_budget 1000000 \
		--output_dir results

pretrain-flan-t5: results/
	@echo "Continued pretraining Flan-T5..."
	python -m src.e3bench.train.cont_pretrain \
		--model_cfg configs/model/flan-t5-base.yaml \
		--train_cfg configs/train/lora.yaml \
		--dataset wikitext \
		--target_loss 2.0 \
		--token_budget 1000000 \
		--output_dir results

pretrain-flan-t5-large: results/
	@echo "Continued pretraining Flan-T5 Large..."
	python -m src.e3bench.train.cont_pretrain \
		--model_cfg configs/model/flan-t5-large.yaml \
		--train_cfg configs/train/lora.yaml \
		--dataset wikitext \
		--target_loss 2.0 \
		--token_budget 1000000 \
		--output_dir results

pretrain-t5-large: results/
	@echo "Continued pretraining T5 Large..."
	python -m src.e3bench.train.cont_pretrain \
		--model_cfg configs/model/t5-large.yaml \
		--train_cfg configs/train/lora.yaml \
		--dataset wikitext \
		--target_loss 2.0 \
		--token_budget 1000000 \
		--output_dir results

# Decoder-only models pretraining
pretrain-gpt2-large: results/
	@echo "Continued pretraining GPT-2 Large..."
	python -m src.e3bench.train.cont_pretrain \
		--model_cfg configs/model/gpt2-large.yaml \
		--train_cfg configs/train/lora.yaml \
		--dataset wikitext \
		--target_loss 2.0 \
		--token_budget 1000000 \
		--output_dir results

pretrain-gpt-neo-125m: results/
	@echo "Continued pretraining GPT-Neo 125M..."
	python -m src.e3bench.train.cont_pretrain \
		--model_cfg configs/model/gpt-neo-125m.yaml \
		--train_cfg configs/train/lora.yaml \
		--dataset wikitext \
		--target_loss 2.0 \
		--token_budget 1000000 \
		--output_dir results

pretrain-opt-125m: results/
	@echo "Continued pretraining OPT 125M..."
	python -m src.e3bench.train.cont_pretrain \
		--model_cfg configs/model/opt-125m.yaml \
		--train_cfg configs/train/lora.yaml \
		--dataset wikitext \
		--target_loss 2.0 \
		--token_budget 1000000 \
		--output_dir results

pretrain-opt-350m: results/
	@echo "Continued pretraining OPT 350M..."
	python -m src.e3bench.train.cont_pretrain \
		--model_cfg configs/model/opt-350m.yaml \
		--train_cfg configs/train/lora.yaml \
		--dataset wikitext \
		--target_loss 2.0 \
		--token_budget 1000000 \
		--output_dir results

pretrain-qwen2.5-0.5b: results/
	@echo "Continued pretraining Qwen2.5 0.5B..."
	python -m src.e3bench.train.cont_pretrain \
		--model_cfg configs/model/qwen2.5-0.5b.yaml \
		--train_cfg configs/train/lora.yaml \
		--dataset wikitext \
		--target_loss 2.0 \
		--token_budget 1000000 \
		--output_dir results

pretrain-qwen2.5-1.5b: results/
	@echo "Continued pretraining Qwen2.5 1.5B..."
	python -m src.e3bench.train.cont_pretrain \
		--model_cfg configs/model/qwen2.5-1.5b.yaml \
		--train_cfg configs/train/lora.yaml \
		--dataset wikitext \
		--target_loss 2.0 \
		--token_budget 1000000 \
		--output_dir results

# Gated models pretraining (requires HF authentication)
pretrain-llama3.2-1b: results/
	@echo "Continued pretraining Llama 3.2 1B..."
	python -m src.e3bench.train.cont_pretrain \
		--model_cfg configs/model/llama-3.2-1b.yaml \
		--train_cfg configs/train/lora.yaml \
		--dataset wikitext \
		--target_loss 2.0 \
		--token_budget 1000000 \
		--output_dir results

pretrain-llama3.2-3b: results/
	@echo "Continued pretraining Llama 3.2 3B..."
	python -m src.e3bench.train.cont_pretrain \
		--model_cfg configs/model/llama-3.2-3b.yaml \
		--train_cfg configs/train/lora.yaml \
		--dataset wikitext \
		--target_loss 2.0 \
		--token_budget 1000000 \
		--output_dir results

pretrain-gemma2-2b: results/
	@echo "Continued pretraining Gemma 2 2B..."
	python -m src.e3bench.train.cont_pretrain \
		--model_cfg configs/model/gemma-2-2b.yaml \
		--train_cfg configs/train/lora.yaml \
		--dataset wikitext \
		--target_loss 2.0 \
		--token_budget 1000000 \
		--output_dir results

# Batch pretraining targets
pretrain-open-models: results/
	@echo "Running continued pretraining on open-access models..."
	$(MAKE) pretrain-bert
	$(MAKE) pretrain-gpt2
	$(MAKE) pretrain-t5
	$(MAKE) pretrain-distilbert
	$(MAKE) pretrain-roberta
	$(MAKE) pretrain-modernbert
	$(MAKE) pretrain-deberta-v3
	$(MAKE) pretrain-bert-large
	$(MAKE) pretrain-bart
	$(MAKE) pretrain-flan-t5
	$(MAKE) pretrain-flan-t5-large
	$(MAKE) pretrain-t5-large
	$(MAKE) pretrain-gpt2-large
	$(MAKE) pretrain-gpt-neo-125m
	$(MAKE) pretrain-opt-125m
	$(MAKE) pretrain-opt-350m
	$(MAKE) pretrain-qwen2.5-0.5b
	$(MAKE) pretrain-qwen2.5-1.5b
	@echo "All open-access model pretraining completed!"

pretrain-gated-models: results/
	@echo "⚠️  Running pretraining on gated models - make sure you're authenticated!"
	@echo "If not authenticated, run: huggingface-cli login"
	@read -p "Press Enter to continue or Ctrl+C to cancel..."
	$(MAKE) pretrain-llama3.2-1b
	$(MAKE) pretrain-llama3.2-3b
	$(MAKE) pretrain-gemma2-2b
	@echo "All gated model pretraining completed!"

pretrain-all-models: results/
	@echo "Running continued pretraining on ALL models..."
	$(MAKE) pretrain-open-models
	$(MAKE) pretrain-gated-models
	@echo "All model pretraining completed!"
