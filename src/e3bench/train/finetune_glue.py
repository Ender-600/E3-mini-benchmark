"""SuperGLUE fine-tuning script."""

import argparse
import yaml
import time
import torch
from transformers import (
    TrainingArguments, 
    Trainer,
    DataCollatorWithPadding,
    EarlyStoppingCallback
)
from datasets import Dataset
from typing import Dict, Any, List
import logging
import os

from ..utils.logging import setup_logging, log_experiment_info, PowerMonitor
from ..utils.seed import set_seed
from ..utils.io import generate_exp_id, save_experiment_result
from ..data.superglue import load_superglue_data, get_superglue_example_format
from ..models.load_hf import load_model_and_tokenizer
from ..models.lora import setup_lora, save_lora_weights, get_lora_parameters

logger = logging.getLogger(__name__)


def preprocess_superglue_data(
    dataset: Dataset,
    tokenizer: Any,
    task: str,
    max_length: int = 256
) -> Dataset:
    """Preprocess SuperGLUE data for training."""
    
    format_info = get_superglue_example_format(task)
    
    def tokenize_function(examples):
        # Create input text based on task format
        if task == "BoolQ":
            texts = [f"{p} {q}" for p, q in zip(examples["passage"], examples["question"])]
        elif task == "RTE":
            texts = [f"{p} {h}" for p, h in zip(examples["premise"], examples["hypothesis"])]
        elif task == "WiC":
            texts = [f"{s1} {s2}" for s1, s2 in zip(examples["sentence1"], examples["sentence2"])]
        elif task == "CB":
            texts = [f"{p} {h}" for p, h in zip(examples["premise"], examples["hypothesis"])]
        else:
            raise ValueError(f"Unknown task: {task}")
        
        # Tokenize
        tokenized = tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=max_length,
            return_tensors="pt"
        )
        
        # Add labels
        tokenized["labels"] = examples["label"]
        
        return tokenized
    
    return dataset.map(tokenize_function, batched=True)


def compute_metrics(eval_pred):
    """Compute evaluation metrics."""
    predictions, labels = eval_pred
    predictions = torch.argmax(torch.tensor(predictions), dim=-1)
    
    # Simple accuracy
    accuracy = (predictions == labels).float().mean().item()
    
    return {"accuracy": accuracy}


def finetune_superglue(
    model_cfg_path: str,
    task_cfg_path: str, 
    train_cfg_path: str,
    output_dir: str = "results"
) -> Dict[str, Any]:
    """Fine-tune model on SuperGLUE tasks."""
    
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
        # Set seed
        seed = train_config.get("seed", [42])[0]  # Use first seed
        set_seed(seed)
        
        # Load data
        logger.info("Loading SuperGLUE data...")
        datasets = load_superglue_data(
            tasks=task_config["tasks"],
            max_length=task_config.get("max_length", 256)
        )
        
        results = {}
        
        for task_name, task_data in datasets.items():
            logger.info(f"Fine-tuning on {task_name}")
            
            # Load model and tokenizer
            model, tokenizer = load_model_and_tokenizer(
                model_config,
                num_labels=len(get_superglue_example_format(task_name)["label_names"])
            )
            
            # Setup LoRA if requested
            if model_config.get("use_lora", False):
                task_type = "classification"
                if model_config["arch"] == "decoder":
                    task_type = "causal_lm"
                elif model_config["arch"] == "encdec":
                    task_type = "seq2seq"
                
                model = setup_lora(model, train_config, task_type)
            
            # Preprocess data
            train_dataset = preprocess_superglue_data(
                task_data["train"], tokenizer, task_name, task_config.get("max_length", 256)
            )
            eval_dataset = preprocess_superglue_data(
                task_data["validation"], tokenizer, task_name, task_config.get("max_length", 256)
            )
            
            # Data collator
            data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
            
            # Training arguments
            training_args = TrainingArguments(
                output_dir=f"{output_dir}/{exp_id}/{task_name}",
                per_device_train_batch_size=train_config.get("per_device_train_batch_size", 16),
                per_device_eval_batch_size=train_config.get("per_device_eval_batch_size", 32),
                gradient_accumulation_steps=train_config.get("gradient_accumulation_steps", 2),
                num_train_epochs=train_config.get("num_train_epochs", 5),
                learning_rate=train_config.get("learning_rate", 5e-4),
                weight_decay=train_config.get("weight_decay", 0.01),
                warmup_ratio=train_config.get("warmup_ratio", 0.1),
                fp16=train_config.get("fp16", True),
                gradient_checkpointing=train_config.get("grad_checkpointing", True),
                save_strategy=train_config.get("save_strategy", "epoch"),
                eval_strategy=train_config.get("eval_strategy", "epoch"),
                logging_steps=train_config.get("logging_steps", 10),
                save_total_limit=train_config.get("save_total_limit", 2),
                load_best_model_at_end=True,
                metric_for_best_model="eval_accuracy",
                greater_is_better=True,
                report_to=None,  # Disable wandb
                remove_unused_columns=False
            )
            
            # Trainer
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                data_collator=data_collator,
                compute_metrics=compute_metrics,
                callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
            )
            
            # Train
            logger.info(f"Training {task_name}...")
            train_result = trainer.train()
            
            # Evaluate
            eval_result = trainer.evaluate()
            
            # Save LoRA weights if used
            if model_config.get("use_lora", False):
                lora_path = save_lora_weights(model, f"{output_dir}/{exp_id}/{task_name}/lora")
            
            # Store results
            results[task_name] = {
                "train_loss": train_result.training_loss,
                "eval_accuracy": eval_result["eval_accuracy"],
                "eval_loss": eval_result["eval_loss"],
                "lora_params": get_lora_parameters(model) if model_config.get("use_lora", False) else None
            }
            
            logger.info(f"{task_name} - Train Loss: {train_result.training_loss:.4f}, Eval Accuracy: {eval_result['eval_accuracy']:.4f}")
        
        # Log experiment
        end_time = time.time()
        experiment_log = log_experiment_info(
            exp_id, 
            {"model": model_config, "task": task_config, "train": train_config},
            results,
            start_time,
            end_time,
            power_monitor
        )
        
        # Save results
        save_experiment_result(exp_id, experiment_log, output_dir)
        
        logger.info(f"Experiment completed: {exp_id}")
        return experiment_log
        
    except Exception as e:
        logger.error(f"Experiment failed: {e}")
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
