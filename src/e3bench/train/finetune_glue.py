"""
SuperGLUE fine-tuning script for E³ Benchmark NLU (fine-tuned) evaluation.

This script evaluates "NLU (fine-tuned)" effectiveness by fine-tuning models on SuperGLUE tasks.
- Encoder-only (BERT-style) and decoder-only (GPT2-as-classifier) models are fine-tuned as classifiers.
- Encoder-decoder models (T5-style) are fine-tuned as seq2seq label generators, evaluated via generated label tokens.
- Multiple seeds are run and mean/std are reported for stability.
- Compute cost (time, VRAM, energy) is logged per seed run.

Architecture-specific handling:
- seq2seq models use Seq2SeqTrainer with processing_class parameter
- classification models use standard Trainer
- Task-aware metrics: accuracy for BoolQ/RTE/WiC, accuracy + macro-F1 for CB
"""

import argparse
import yaml
import time
import math
import torch
import numpy as np
from transformers import (
    TrainingArguments, 
    Trainer,
    Seq2SeqTrainer,
    DataCollatorWithPadding,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback
)
from datasets import Dataset
from typing import Dict, Any, List, Optional
import logging
import os
from sklearn.metrics import f1_score

from ..utils.logging import setup_logging, PowerMonitor
from ..utils.seed import set_seed
from ..utils.io import generate_exp_id, save_experiment_result, sync_to_latest
from ..data.superglue import load_superglue_data, get_superglue_example_format
from ..models.load_hf import load_model_and_tokenizer
from ..models.lora import setup_lora, save_lora_weights, get_lora_parameters

logger = logging.getLogger(__name__)


def count_trainable_params(model: Any) -> int:
    """Count total trainable parameters in the model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def aggregate_runs(runs: List[Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    """
    Aggregate multiple runs (seeds) into mean and std statistics.
    
    Args:
        runs: List of dictionaries with metric keys
        
    Returns:
        Dictionary with 'mean' and 'std' keys containing aggregated metrics
    """
    if not runs:
        return {"mean": {}, "std": {}}
    
    # Get all metric keys from first run (exclude resource/seed fields)
    metric_keys = [k for k in runs[0].keys() if k not in ["seed", "train_time_seconds", "max_memory_gb", "gpu_name"]]
    
    mean_dict = {}
    std_dict = {}
    
    for key in metric_keys:
        values = [run[key] for run in runs if key in run]
        if values:
            mean_dict[key] = float(np.mean(values))
            std_dict[key] = float(np.std(values))
    
    return {"mean": mean_dict, "std": std_dict}


def task_metrics(
    task_name: str,
    predictions: np.ndarray,
    labels: np.ndarray,
    tokenizer: Optional[Any] = None,
    label_names: Optional[List[str]] = None
) -> Dict[str, float]:
    """
    Compute task-aware metrics for SuperGLUE tasks.
    
    Args:
        task_name: Name of the task (BoolQ, RTE, WiC, CB)
        predictions: Model predictions (logits for classification, token IDs for seq2seq)
        labels: Ground truth labels
        tokenizer: Tokenizer for seq2seq decoding
        label_names: List of label name strings for seq2seq
        
    Returns:
        Dictionary of metrics (accuracy, macro_f1 for CB)
    """
    metrics = {}
    
    # Handle seq2seq models
    if tokenizer is not None and label_names is not None:
        # Convert to numpy if needed
        if isinstance(predictions, torch.Tensor):
            predictions = predictions.cpu().numpy()
        if isinstance(labels, torch.Tensor):
            labels = labels.cpu().numpy()
        
        def decode_sequence(tokens):
            """Decode token sequences, handling padding properly."""
            decoded_list = []
            if tokens.ndim == 1:
                # Single sequence: filter padding and decode
                valid = tokens[tokens != -100]
                if len(valid) > 0:
                    decoded = tokenizer.decode(valid.tolist(), skip_special_tokens=True)
                    decoded_list.append(decoded)
                else:
                    decoded_list.append("")
            else:
                # Multiple sequences: process each
                for row in tokens:
                    valid = row[row != -100]
                    if len(valid) > 0:
                        decoded = tokenizer.decode(valid.tolist(), skip_special_tokens=True)
                        decoded_list.append(decoded)
                    else:
                        decoded_list.append("")
            return decoded_list
        
        # Decode predictions and labels
        decoded_preds = decode_sequence(predictions)
        decoded_labels = decode_sequence(labels)
        
        # Normalize: strip and lowercase
        decoded_preds = [p.strip().lower() for p in decoded_preds]
        decoded_labels = [l.strip().lower() for l in decoded_labels]
        
        # Create label mapping (e.g., for RTE: "not_entailment"->0, "entailment"->1)
        label_to_id = {label.lower(): idx for idx, label in enumerate(label_names)}
        
        # Map decoded predictions to label IDs with fuzzy matching
        pred_label_ids = []
        for pred in decoded_preds:
            matched = False
            # First try exact match
            if pred in label_to_id:
                pred_label_ids.append(label_to_id[pred])
                matched = True
            else:
                # Try fuzzy matching (check if any label is contained in pred or vice versa)
                for label_name, label_idx in label_to_id.items():
                    # Check if label appears in prediction or prediction appears in label
                    if label_name in pred or pred in label_name:
                        pred_label_ids.append(label_idx)
                        matched = True
                        break
                # If still no match, try matching partial words
                if not matched:
                    for label_name, label_idx in label_to_id.items():
                        # Split both strings and check for word overlap
                        pred_words = set(pred.split())
                        label_words = set(label_name.split())
                        if pred_words & label_words:  # Check for intersection
                            pred_label_ids.append(label_idx)
                            matched = True
                            break
                if not matched:
                    # Fallback: pick first class
                    pred_label_ids.append(0)
        
        # Map labels with same logic
        true_label_ids = []
        for label in decoded_labels:
            matched = False
            if label in label_to_id:
                true_label_ids.append(label_to_id[label])
                matched = True
            else:
                for label_name, label_idx in label_to_id.items():
                    if label_name in label or label in label_name:
                        true_label_ids.append(label_idx)
                        matched = True
                        break
                if not matched:
                    for label_name, label_idx in label_to_id.items():
                        label_words = set(label.split())
                        label_name_words = set(label_name.split())
                        if label_words & label_name_words:
                            true_label_ids.append(label_idx)
                            matched = True
                            break
                if not matched:
                    true_label_ids.append(0)
        
        pred_label_ids = np.array(pred_label_ids)
        true_label_ids = np.array(true_label_ids)
    else:
        # Classification models: predictions are logits
        if isinstance(predictions, torch.Tensor):
            predictions = predictions.cpu().numpy()
        if isinstance(labels, torch.Tensor):
            labels = labels.cpu().numpy()
        
        # Get predicted class IDs
        if predictions.ndim > 1:
            pred_label_ids = np.argmax(predictions, axis=-1)
        else:
            pred_label_ids = predictions
        
        true_label_ids = labels
    
    # Compute accuracy
    accuracy = float(np.mean(pred_label_ids == true_label_ids))
    metrics["accuracy"] = accuracy
    
    # Compute macro-F1 for CB
    if task_name == "CB":
        macro_f1 = float(f1_score(true_label_ids, pred_label_ids, average="macro"))
        metrics["macro_f1"] = macro_f1
    
    return metrics


def preprocess_superglue_data(
    dataset: Dataset,
    tokenizer: Any,
    task: str,
    max_length: int = 256,
    arch: str = "encoder"
) -> Dataset:
    """Preprocess SuperGLUE data for training."""
    
    format_info = get_superglue_example_format(task)
    
    def tokenize_function(examples):
        # Create input text using proper prompt format for seq2seq training
        if task == "BoolQ":
            texts = [f"Passage: {p}\nQuestion: {q}\nAnswer:" for p, q in zip(examples["passage"], examples["question"])]
        elif task == "RTE":
            texts = [f"Premise: {p}\nHypothesis: {h}\nEntailment:" for p, h in zip(examples["premise"], examples["hypothesis"])]
        elif task == "WiC":
            texts = [f"Word: {w}\nSentence 1: {s1}\nSentence 2: {s2}\nSame meaning:" for w, s1, s2 in zip(examples["word"], examples["sentence1"], examples["sentence2"])]
        elif task == "CB":
            texts = [f"Premise: {p}\nHypothesis: {h}\nRelationship:" for p, h in zip(examples["premise"], examples["hypothesis"])]
        else:
            raise ValueError(f"Unknown task: {task}")
        
        # Tokenize input texts
        tokenized = tokenizer(
            texts,
            truncation=True,
            padding=False,
            max_length=max_length,
            return_tensors=None
        )
        
        # Handle labels based on architecture
        if arch == "encdec":
            # For seq2seq models like T5, convert integer labels to tokenized sequences
            label_names = format_info["label_names"]
            labels_as_text = [label_names[label] for label in examples["label"]]
            # Tokenize labels as sequences
            label_tokenized = tokenizer(
                labels_as_text,
                truncation=True,
                padding=False,
                max_length=10,  # Short labels should fit
                return_tensors=None
            )
            tokenized["labels"] = label_tokenized["input_ids"]
        else:
            # For encoder/decoder classification models, use integer labels
            tokenized["labels"] = examples["label"]
        
        return tokenized
    
    # Tokenize and remove original columns
    dataset = dataset.map(tokenize_function, batched=True, remove_columns=dataset.column_names)
    
    return dataset




def finetune_superglue(
    model_cfg_path: str,
    task_cfg_path: str, 
    train_cfg_path: str,
    output_dir: str = "results"
) -> Dict[str, Any]:
    """
    Fine-tune model on SuperGLUE tasks with multiple seeds and proper architecture handling.
    
    Returns comprehensive experiment log with per-seed runs and aggregated statistics.
    """
    
    # Setup logging
    logger = setup_logging()
    
    # Load configurations
    with open(model_cfg_path, 'r') as f:
        model_config = yaml.safe_load(f)
    with open(task_cfg_path, 'r') as f:
        task_config = yaml.safe_load(f)
    with open(train_cfg_path, 'r') as f:
        train_config = yaml.safe_load(f)
    
    # Generate experiment ID
    exp_id = generate_exp_id("superglue")
    logger.info(f"Starting experiment: {exp_id}")
    
    # Start power monitoring
    power_monitor = PowerMonitor()
    power_monitor.start()
    
    start_time = time.time()
    
    try:
        # Load data once for all seeds
        logger.info("Loading SuperGLUE data...")
        datasets = load_superglue_data(
            tasks=task_config["tasks"],
            max_length=task_config.get("max_length", 256)
        )
        
        # Get seeds from config
        seeds = train_config.get("seed", [42])
        if not isinstance(seeds, list):
            seeds = [seeds]
        
        logger.info(f"Running {len(seeds)} seeds: {seeds}")
        
        results = {}
        
        # Process each task
        for task_name, task_data in datasets.items():
            logger.info(f"Processing task: {task_name}")
            
            # Get config for this task
            num_labels = len(get_superglue_example_format(task_name)["label_names"])
            use_lora = model_config.get("use_lora", False)
            arch = model_config["arch"]
            
            # Load model once to count parameters (before seed loop)
            model_temp, _ = load_model_and_tokenizer(model_config, num_labels=num_labels)
            if use_lora:
                if arch == "encdec":
                    task_type = "seq2seq"
                else:
                    task_type = "classification"
                model_temp = setup_lora(model_temp, train_config, task_type, arch=arch)
            
            trainable_params = count_trainable_params(model_temp)
            lora_params = get_lora_parameters(model_temp) if use_lora else None
            del model_temp  # Free memory
            
            # Store per-seed runs
            runs = []
            
            # Loop over seeds
            for seed in seeds:
                logger.info(f"Task {task_name}, Seed {seed}")
                
                # Set seed
                set_seed(seed)
                
                # Reinitialize model for each seed
                model, tokenizer = load_model_and_tokenizer(
                    model_config,
                    num_labels=num_labels
                )
                
                # Ensure model dtype matches training fp16 setting for consistency
                # Convert after loading to ensure proper dtype alignment (like cont_pretrain.py)
                train_fp16 = train_config.get("fp16", False)
                model_dtype = next(model.parameters()).dtype
                if train_fp16 and model_dtype != torch.float16:
                    model = model.half()
                    logger.info(f"Converted model to fp16 for training (was {model_dtype})")
                elif not train_fp16 and model_dtype != torch.float32:
                    model = model.float()
                    logger.info(f"Converted model to fp32 for training (was {model_dtype})")
                
                # Reapply LoRA if needed (after dtype conversion)
                if use_lora:
                    if arch == "encdec":
                        task_type = "seq2seq"
                    else:
                        task_type = "classification"
                    model = setup_lora(model, train_config, task_type, arch=arch)
                    
                    # Verify LoRA model dtype matches training setting
                    lora_model_dtype = next(model.parameters()).dtype
                    if train_fp16 and lora_model_dtype != torch.float16:
                        logger.warning(f"LoRA model dtype ({lora_model_dtype}) doesn't match fp16 training setting. Converting...")
                        model = model.half()
                    elif not train_fp16 and lora_model_dtype != torch.float32:
                        logger.warning(f"LoRA model dtype ({lora_model_dtype}) doesn't match fp32 training setting. Converting...")
                        model = model.float()
                
                # Log model configuration before preprocessing
                final_model_dtype = next(model.parameters()).dtype
                logger.info(f"{task_name} seed {seed} - Model dtype: {final_model_dtype}, Training fp16: {train_fp16}, "
                           f"LoRA: {use_lora}, Num labels: {num_labels}")
                
                # Preprocess data
                train_dataset = preprocess_superglue_data(
                    task_data["train"], tokenizer, task_name, 
                    task_config.get("max_length", 256), arch=arch
                )
                eval_dataset = preprocess_superglue_data(
                    task_data["validation"], tokenizer, task_name, 
                    task_config.get("max_length", 256), arch=arch
                )
                
                logger.info(f"{task_name} seed {seed} - Dataset sizes: train={len(train_dataset)}, eval={len(eval_dataset)}")
                
                # Setup data collator based on architecture
                # pad_to_multiple_of=8 is for fp16 optimization, only needed when fp16=True
                pad_multiple = 8 if train_fp16 else None
                
                if arch == "encdec":
                    data_collator = DataCollatorForSeq2Seq(
                        tokenizer=tokenizer,
                        model=model,  # Pass model to ensure proper padding behavior
                        pad_to_multiple_of=pad_multiple,
                        label_pad_token_id=-100,
                        return_tensors="pt"
                    )
                else:
                    data_collator = DataCollatorWithPadding(
                        tokenizer=tokenizer,
                        pad_to_multiple_of=pad_multiple,
                        return_tensors="pt"
                    )
                
                # Setup training arguments
                training_args_dict = {
                    "output_dir": f"{output_dir}/{exp_id}/{task_name}/seed_{seed}",
                    "per_device_train_batch_size": train_config.get("per_device_train_batch_size", 16),
                    "per_device_eval_batch_size": train_config.get("per_device_eval_batch_size", 32),
                    "gradient_accumulation_steps": train_config.get("gradient_accumulation_steps", 2),
                    "num_train_epochs": train_config.get("num_train_epochs", 5),
                    "learning_rate": train_config.get("learning_rate", 5e-4),
                    "weight_decay": train_config.get("weight_decay", 0.01),
                    "warmup_ratio": train_config.get("warmup_ratio", 0.1),
                    "fp16": train_fp16,  # Use the same fp16 setting we determined above
                    "gradient_checkpointing": train_config.get("grad_checkpointing", True),
                    "save_strategy": train_config.get("save_strategy", "epoch"),
                    "eval_strategy": train_config.get("eval_strategy", "epoch"),
                    "logging_steps": train_config.get("logging_steps", 10),
                    "save_total_limit": train_config.get("save_total_limit", 2),
                    "load_best_model_at_end": True,
                    "metric_for_best_model": "eval_accuracy",
                    "greater_is_better": True,
                    "report_to": [],  # Disable wandb by default
                    "remove_unused_columns": False  # Keep label fields we prepared manually
                }
                
                training_args = TrainingArguments(**training_args_dict)
                
                # Add generation-related attributes for seq2seq models to avoid AttributeError
                if arch == "encdec":
                    # Set default attributes on training_args to avoid errors in Seq2SeqTrainer
                    if not hasattr(training_args, 'generation_config'):
                        training_args.generation_config = None
                    if not hasattr(training_args, 'generation_max_length'):
                        training_args.generation_max_length = 10
                    if not hasattr(training_args, 'generation_num_beams'):
                        training_args.generation_num_beams = None
                    if not hasattr(training_args, 'predict_with_generate'):
                        training_args.predict_with_generate = True
                
                # Create compute_metrics wrapper
                label_names = get_superglue_example_format(task_name)["label_names"]
                
                def compute_metrics_fn(eval_pred):
                    predictions, labels = eval_pred
                    return task_metrics(task_name, predictions, labels, 
                                       tokenizer=tokenizer if arch == "encdec" else None,
                                       label_names=label_names if arch == "encdec" else None)
                
                # Create trainer (architecture-specific)
                if arch == "encdec":
                    trainer = Seq2SeqTrainer(
                        model=model,
                        args=training_args,
                        train_dataset=train_dataset,
                        eval_dataset=eval_dataset,
                        data_collator=data_collator,
                        compute_metrics=compute_metrics_fn,
                        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
                        processing_class=tokenizer
                    )
                else:
                    trainer = Trainer(
                        model=model,
                        args=training_args,
                        train_dataset=train_dataset,
                        eval_dataset=eval_dataset,
                        data_collator=data_collator,
                        compute_metrics=compute_metrics_fn,
                        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
                    )
                
                # Reset GPU stats before training
                if torch.cuda.is_available():
                    torch.cuda.reset_peak_memory_stats()
                
                # Track training time
                train_start = time.time()
                
                # Train
                logger.info(f"Training {task_name} with seed {seed}...")
                train_result = trainer.train()
                
                # Check training loss for numerical issues
                train_loss = train_result.training_loss
                if math.isnan(train_loss) or math.isinf(train_loss):
                    logger.error(f"{task_name} seed {seed} - Training loss is NaN or Inf: {train_loss}. Skipping this run.")
                    continue
                if train_loss == 0.0:
                    logger.warning(f"{task_name} seed {seed} - Training loss is exactly 0.0. This may indicate model collapse.")
                
                # Evaluate
                eval_result = trainer.evaluate()
                eval_loss = eval_result.get("eval_loss", float('inf'))
                
                # Check eval loss for numerical issues
                if math.isnan(eval_loss) or math.isinf(eval_loss):
                    logger.error(f"{task_name} seed {seed} - Eval loss is NaN or Inf: {eval_loss}. Skipping this run.")
                    continue
                
                # Warn about unusually high eval loss for classification tasks
                # For n-class classification, random baseline loss is approximately log(n)
                if num_labels == 2 and eval_loss > 5.0:  # Binary classification, random ~0.69
                    logger.warning(f"{task_name} seed {seed} - Eval loss ({eval_loss:.4f}) is unusually high for binary classification (random baseline ~0.69)")
                elif num_labels == 3 and eval_loss > 5.0:  # 3-class (CB), random ~1.1
                    logger.warning(f"{task_name} seed {seed} - Eval loss ({eval_loss:.4f}) is unusually high for 3-class classification (random baseline ~1.1)")
                
                train_time = time.time() - train_start
                
                # Get GPU info
                gpu_name = None
                max_memory_gb = 0.0
                if torch.cuda.is_available():
                    gpu_name = torch.cuda.get_device_name(0)
                    max_memory_gb = torch.cuda.max_memory_allocated() / 1e9
                
                # Save LoRA weights if used (only for first seed per task)
                if use_lora and seed == seeds[0]:
                    lora_path = save_lora_weights(model, f"{output_dir}/{exp_id}/{task_name}/lora")
                
                # Store run results
                run_log = {
                    "seed": seed,
                    "train_time_seconds": train_time,
                    "max_memory_gb": max_memory_gb,
                    "gpu_name": gpu_name,
                    **eval_result  # Include all eval metrics
                }
                runs.append(run_log)
                
                logger.info(f"{task_name} seed {seed} - Train Loss: {train_loss:.4f}, "
                           f"Eval Loss: {eval_loss:.4f}, Eval Accuracy: {eval_result.get('eval_accuracy', 0):.4f}")
            
            # Aggregate runs
            aggregated = aggregate_runs(runs)
            
            # Store task results
            results[task_name] = {
                "runs": runs,
                "mean": aggregated["mean"],
                "std": aggregated["std"],
                "trainable_params": trainable_params,
                "lora_params": lora_params
            }
            
            logger.info(f"{task_name} complete - Mean Accuracy: {aggregated['mean'].get('eval_accuracy', 0):.4f}")
        
        # Stop power monitoring
        end_time = time.time()
        power_stats = power_monitor.stop()
        power_stats["duration_seconds"] = end_time - start_time
        
        # Build final experiment log
        experiment_log = {
            "exp_id": exp_id,
            "model": model_config,
            "task_cfg": task_config,
            "train_cfg": train_config,
            "results": results,
            "power": power_stats,
            "start_time": start_time,
            "end_time": end_time
        }
        
        # Save results
        save_experiment_result(exp_id, experiment_log, output_dir)
        
        # Sync to latest directory
        latest_path = sync_to_latest(experiment_log)
        logger.info(f"Synced to latest: {latest_path}")
        
        logger.info(f"Experiment completed: {exp_id}")
        return experiment_log
        
    except Exception as e:
        logger.error(f"Experiment failed: {e}", exc_info=True)
        raise
    finally:
        power_monitor.stop()


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(description="Fine-tune on SuperGLUE")
    parser.add_argument("--model_cfg", required=True, help="Model config path")
    parser.add_argument("--task_cfg", required=True, help="Task config path")
    parser.add_argument("--train_cfg", required=True, help="Training config path")
    parser.add_argument("--output_dir", default="results", help="Output directory")
    
    args = parser.parse_args()
    
    finetune_superglue(
        args.model_cfg,
        args.task_cfg,
        args.train_cfg,
        args.output_dir
    )


if __name__ == "__main__":
    main()
