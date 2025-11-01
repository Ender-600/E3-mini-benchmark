"""Continued pretraining script for efficiency measurement."""

import argparse
import yaml
import time
import torch
from transformers import (
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    DataCollatorForSeq2Seq
)
from datasets import load_dataset, Dataset
from typing import Dict, Any, List
import logging
import os

from ..utils.logging import setup_logging, log_experiment_info, PowerMonitor
from ..utils.seed import set_seed
from ..utils.io import generate_exp_id, save_experiment_result
from ..models.load_hf import load_model_and_tokenizer
from ..models.lora import setup_lora, save_lora_weights, get_lora_parameters

logger = logging.getLogger(__name__)


def load_pretraining_data(
    dataset_name: str = "wikitext",
    dataset_config: str = "wikitext-2-raw-v1",
    max_length: int = 256,
    num_samples: int = 1000
) -> Dataset:
    """Load dataset for continued pretraining."""
    
    logger.info(f"Loading pretraining data: {dataset_name}")
    
    if dataset_name == "wikitext":
        dataset = load_dataset(dataset_name, dataset_config, split="train")
    elif dataset_name == "c4":
        dataset = load_dataset("allenai/c4", "realnewslike", split="train", streaming=True)
        # Take a subset for efficiency
        dataset = dataset.take(num_samples)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    # Convert to regular dataset if streaming
    if hasattr(dataset, 'take'):
        dataset = Dataset.from_list(list(dataset))
    
    # Take subset for efficiency
    if len(dataset) > num_samples:
        dataset = dataset.select(range(num_samples))
    
    logger.info(f"Loaded {len(dataset)} samples")
    return dataset


def preprocess_pretraining_data(
    dataset: Dataset,
    tokenizer: Any,
    arch: str,
    max_length: int = 256
) -> Dataset:
    """Preprocess data for continued pretraining."""
    
    def tokenize_function(examples):
        texts = examples["text"] if "text" in examples else examples["content"]
        
        if arch == "encoder":
            # Masked language modeling - let collator handle labels
            tokenized = tokenizer(
                texts,
                truncation=True,
                padding=False,
                max_length=max_length,
            )
            # Don't set labels here - DataCollatorForLanguageModeling will handle it
        elif arch == "decoder":
            # Causal language modeling - let collator handle labels
            tokenized = tokenizer(
                texts,
                truncation=True,
                padding=False,
                max_length=max_length,
            )
            # Don't set labels here - DataCollatorForLanguageModeling will handle it
            
        elif arch == "encdec":
            # Span corruption (T5 style)
            tokenized = tokenizer(
                texts,
                truncation=True,
                padding=False,
                max_length=max_length,
            )
            # For T5, we need to create input and target sequences
            # This is simplified - in practice you'd use proper span corruption
            tokenized["labels"] = [seq[:] for seq in tokenized["input_ids"]]
        
        return tokenized
    
    # Tokenize and remove original columns
    dataset = dataset.map(tokenize_function, batched=True, remove_columns=dataset.column_names)
    
    return dataset


def continued_pretraining(
    model_cfg_path: str,
    train_cfg_path: str,
    dataset_name: str = "wikitext",
    target_loss: float = 2.0,
    output_dir: str = "results"
) -> Dict[str, Any]:
    """Continued pretraining for efficiency measurement."""
    
    # Setup logging
    logger = setup_logging()
    
    # Load configurations
    with open(model_cfg_path, 'r') as f:
        model_config = yaml.safe_load(f)
    with open(train_cfg_path, 'r') as f:
        train_config = yaml.safe_load(f)
    
    # Generate experiment ID
    exp_id = generate_exp_id("cont_pretrain")
    logger.info(f"Starting continued pretraining: {exp_id}")
    
    # Start power monitoring
    power_monitor = PowerMonitor()
    power_monitor.start()
    
    start_time = time.time()
    
    try:
        # Set seed
        seed = train_config.get("seed", [42])[0]
        set_seed(seed)
        
        # Load data
        logger.info("Loading pretraining data...")
        dataset = load_pretraining_data(
            dataset_name=dataset_name,
            max_length=model_config.get("max_length", 256),
            num_samples=1000  # Limit for efficiency
        )
        
        # Load model and tokenizer for pretraining
        model, tokenizer = load_model_and_tokenizer(model_config, for_pretraining=True)
        
        # Setup LoRA if requested
        if model_config.get("use_lora", False):
            task_type = "causal_lm"
            if model_config["arch"] == "encoder":
                task_type = "feature_extraction"  # For MLM pretraining
            elif model_config["arch"] == "encdec":
                task_type = "seq2seq"
            
            model = setup_lora(model, train_config, task_type, arch=model_config["arch"])
        
        # Preprocess data
        processed_dataset = preprocess_pretraining_data(
            dataset, tokenizer, model_config["arch"], model_config.get("max_length", 256)
        )
        
        # Split data
        train_size = int(0.8 * len(processed_dataset))
        eval_size = len(processed_dataset) - train_size
        
        train_dataset = processed_dataset.select(range(train_size))
        eval_dataset = processed_dataset.select(range(train_size, train_size + eval_size))
        
        # Data collator - use built-in collators
        if model_config["arch"] == "decoder":
            data_collator = DataCollatorForLanguageModeling(
                tokenizer=tokenizer,
                mlm=False  # Causal LM for decoder
            )
        elif model_config["arch"] == "encoder":
            data_collator = DataCollatorForLanguageModeling(
                tokenizer=tokenizer,
                mlm=True  # Masked LM for encoder
            )
        else:  # encdec
            from transformers import DataCollatorForSeq2Seq
            data_collator = DataCollatorForSeq2Seq(
                tokenizer=tokenizer,
                pad_to_multiple_of=8,
                label_pad_token_id=-100,
                return_tensors="pt"
            )
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=f"{output_dir}/{exp_id}",
            per_device_train_batch_size=train_config.get("per_device_train_batch_size", 8),
            per_device_eval_batch_size=train_config.get("per_device_eval_batch_size", 16),
            gradient_accumulation_steps=train_config.get("gradient_accumulation_steps", 4),
            num_train_epochs=train_config.get("num_train_epochs", 3),
            learning_rate=train_config.get("learning_rate", 5e-5),
            weight_decay=train_config.get("weight_decay", 0.01),
            warmup_ratio=train_config.get("warmup_ratio", 0.1),
            fp16=train_config.get("fp16", True),
            gradient_checkpointing=train_config.get("grad_checkpointing", True),
            max_grad_norm=1.0,  # Gradient clipping to prevent gradient explosion
            save_strategy="steps",
            eval_strategy="steps",
            save_steps=100,
            eval_steps=100,
            logging_steps=10,
            save_total_limit=2,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            report_to=["wandb"],  # Enable wandb logging
            remove_unused_columns=True  # Remove original text fields
        )
        
        # Trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
        )
        
        # Train until target loss is reached
        logger.info(f"Training until target loss {target_loss} is reached...")
        
        best_eval_loss = float('inf')
        epochs_trained = 0
        total_tokens = 0
        
        for epoch in range(train_config.get("num_train_epochs", 3)):
            logger.info(f"Epoch {epoch + 1}")
            
            # Train one epoch
            train_result = trainer.train()
            epochs_trained += 1
            
            # Evaluate
            eval_result = trainer.evaluate()
            current_eval_loss = eval_result["eval_loss"]
            
            # Count tokens (rough estimate)
            total_tokens += len(train_dataset) * model_config.get("max_length", 256)
            
            logger.info(f"Epoch {epoch + 1} - Train Loss: {train_result.training_loss:.4f}, Eval Loss: {current_eval_loss:.4f}")
            
            # Check if target loss reached
            if current_eval_loss <= target_loss:
                logger.info(f"Target loss {target_loss} reached after {epochs_trained} epochs")
                break
            
            best_eval_loss = min(best_eval_loss, current_eval_loss)
        
        # Final evaluation
        final_eval = trainer.evaluate()
        
        # Save LoRA weights if used
        if model_config.get("use_lora", False):
            lora_path = save_lora_weights(model, f"{output_dir}/{exp_id}/lora")
        
        # Compute efficiency metrics
        duration = time.time() - start_time
        tokens_per_second = total_tokens / duration if duration > 0 else 0
        
        results = {
            "final_eval_loss": final_eval["eval_loss"],
            "best_eval_loss": best_eval_loss,
            "epochs_trained": epochs_trained,
            "total_tokens": total_tokens,
            "duration_seconds": duration,
            "tokens_per_second": tokens_per_second,
            "target_reached": final_eval["eval_loss"] <= target_loss,
            "lora_params": get_lora_parameters(model) if model_config.get("use_lora", False) else None
        }
        
        # Log experiment
        experiment_log = log_experiment_info(
            exp_id,
            {"model": model_config, "train": train_config, "dataset": dataset_name, "target_loss": target_loss},
            results,
            start_time,
            time.time(),
            power_monitor
        )
        
        # Save results
        save_experiment_result(exp_id, experiment_log, output_dir)
        
        logger.info(f"Continued pretraining completed: {exp_id}")
        logger.info(f"Final loss: {final_eval['eval_loss']:.4f}, Tokens/sec: {tokens_per_second:.2f}")
        
        return experiment_log
        
    except Exception as e:
        logger.error(f"Continued pretraining failed: {e}")
        raise
    finally:
        power_monitor.stop()


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(description="Continued pretraining")
    parser.add_argument("--model_cfg", required=True, help="Model config path")
    parser.add_argument("--train_cfg", required=True, help="Training config path")
    parser.add_argument("--dataset", default="wikitext", help="Dataset name")
    parser.add_argument("--target_loss", type=float, default=2.0, help="Target validation loss")
    parser.add_argument("--output_dir", default="results", help="Output directory")
    
    args = parser.parse_args()
    
    continued_pretraining(
        args.model_cfg,
        args.train_cfg,
        args.dataset,
        args.target_loss,
        args.output_dir
    )


if __name__ == "__main__":
    main()
