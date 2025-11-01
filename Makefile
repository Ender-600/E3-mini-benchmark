# E³ Mini-Benchmark Makefile

.PHONY: help env train eval bench report figs clean all eval-all-models fewshot-comparison \
	eval-bert-0shot eval-bert-5shot eval-bert-10shot \
	eval-gpt2-0shot eval-gpt2-5shot eval-gpt2-10shot \
	eval-t5-0shot eval-t5-5shot eval-t5-10shot

# Default target
help:
	@echo "E³ Mini-Benchmark - Available targets:"
	@echo "  env              - Install dependencies"
	@echo "  train            - Run SuperGLUE fine-tuning (LoRA)"
	@echo "  eval             - Run few-shot evaluation"
	@echo "  bench            - Run inference benchmarking"
	@echo "  report           - Aggregate results and run significance tests"
	@echo "  figs             - Generate plots"
	@echo "  clean            - Clean up generated files"
	@echo "  all              - Run complete benchmark pipeline"
	@echo ""
	@echo "Cross-model comparison:"
	@echo "  eval-all-models  - Run few-shot eval on all models (BERT, GPT-2, T5)"
	@echo "  fewshot-comparison - Full few-shot comparison with all models and shots"
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
train: results/
	@echo "Running SuperGLUE fine-tuning..."
	python -m src.e3bench.train.finetune_glue \
		--model_cfg configs/model/bert-base.yaml \
		--task_cfg configs/task/superglue.yaml \
		--train_cfg configs/train/lora.yaml \
		--output_dir results

# Few-shot evaluation (default: GPT-2 base)
eval: results/
	@echo "Running few-shot evaluation..."
	python -m src.e3bench.eval.eval_fewshot \
		--model_cfg configs/model/gpt2-medium.yaml \
		--eval_cfg configs/eval/fewshot_5.yaml \
		--output_dir results

# Inference benchmarking (default: T5-base)
bench: results/
	@echo "Running inference benchmarking..."
	python -m src.e3bench.eval.bench_infer \
		--model_cfg configs/model/t5-base.yaml \
		--bench_cfg configs/bench/infer_seq2seq.yaml \
		--output_dir results

# Aggregate results and run significance tests
report: tables/
	@echo "Aggregating results..."
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
all: train eval bench report figs
	@echo "Complete benchmark pipeline finished!"

# Clean up generated files
clean:
	rm -rf results/ tables/ figs/
	rm -f temp_lm_eval_results.json
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

# Individual model training targets
train-bert: results/
	@echo "Training BERT (encoder-only)..."
	python -m src.e3bench.train.finetune_glue \
		--model_cfg configs/model/bert-base.yaml \
		--task_cfg configs/task/superglue.yaml \
		--train_cfg configs/train/lora.yaml \
		--output_dir results

train-t5: results/
	@echo "Training T5 (encoder-decoder)..."
	python -m src.e3bench.train.finetune_glue \
		--model_cfg configs/model/t5-base.yaml \
		--task_cfg configs/task/superglue.yaml \
		--train_cfg configs/train/lora.yaml \
		--output_dir results

train-gpt2: results/
	@echo "Training GPT-2 (decoder-only)..."
	python -m src.e3bench.train.finetune_glue \
		--model_cfg configs/model/gpt2-medium.yaml \
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

# Individual benchmarking targets
bench-decoder: results/
	@echo "Benchmarking decoder-only models..."
	python -m src.e3bench.eval.bench_infer \
		--model_cfg configs/model/gpt2-medium.yaml \
		--bench_cfg configs/bench/infer_decoder.yaml \
		--output_dir results

bench-seq2seq: results/
	@echo "Benchmarking encoder-decoder models..."
	python -m src.e3bench.eval.bench_infer \
		--model_cfg configs/model/t5-base.yaml \
		--bench_cfg configs/bench/infer_seq2seq.yaml \
		--output_dir results

bench-encoder: results/
	@echo "Benchmarking encoder-only models..."
	python -m src.e3bench.eval.bench_infer \
		--model_cfg configs/model/bert-base.yaml \
		--bench_cfg configs/bench/infer_encoder.yaml \
		--output_dir results

# Continued pretraining targets
pretrain-bert: results/
	@echo "Continued pretraining BERT..."
	python -m src.e3bench.train.cont_pretrain \
		--model_cfg configs/model/bert-base.yaml \
		--train_cfg configs/train/lora.yaml \
		--dataset wikitext \
		--target_loss 2.0 \
		--output_dir results

pretrain-t5: results/
	@echo "Continued pretraining T5..."
	python -m src.e3bench.train.cont_pretrain \
		--model_cfg configs/model/t5-base.yaml \
		--train_cfg configs/train/lora.yaml \
		--dataset wikitext \
		--target_loss 2.0 \
		--output_dir results

pretrain-gpt2: results/
	@echo "Continued pretraining GPT-2..."
	python -m src.e3bench.train.cont_pretrain \
		--model_cfg configs/model/gpt2-medium.yaml \
		--train_cfg configs/train/lora.yaml \
		--dataset wikitext \
		--target_loss 2.0 \
		--output_dir results
