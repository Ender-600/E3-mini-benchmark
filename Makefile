# E³ Mini-Benchmark Makefile

.PHONY: help env train eval bench report figs clean all

# Default target
help:
	@echo "E³ Mini-Benchmark - Available targets:"
	@echo "  env     - Install dependencies"
	@echo "  train   - Run SuperGLUE fine-tuning (LoRA)"
	@echo "  eval    - Run few-shot evaluation"
	@echo "  bench   - Run inference benchmarking"
	@echo "  report  - Aggregate results and run significance tests"
	@echo "  figs    - Generate plots"
	@echo "  clean   - Clean up generated files"
	@echo "  all     - Run complete benchmark pipeline"

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

# Few-shot evaluation (default: GPT-2 medium)
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

# Individual evaluation targets
eval-0shot: results/
	@echo "Running 0-shot evaluation..."
	python -m src.e3bench.eval.eval_fewshot \
		--model_cfg configs/model/gpt2-medium.yaml \
		--eval_cfg configs/eval/fewshot_0.yaml \
		--output_dir results

eval-5shot: results/
	@echo "Running 5-shot evaluation..."
	python -m src.e3bench.eval.eval_fewshot \
		--model_cfg configs/model/gpt2-medium.yaml \
		--eval_cfg configs/eval/fewshot_5.yaml \
		--output_dir results

eval-10shot: results/
	@echo "Running 10-shot evaluation..."
	python -m src.e3bench.eval.eval_fewshot \
		--model_cfg configs/model/gpt2-medium.yaml \
		--eval_cfg configs/eval/fewshot_10.yaml \
		--output_dir results

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
