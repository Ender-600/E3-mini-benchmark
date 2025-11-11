"""Continued pretraining script for efficiency measurement."""

import argparse
import yaml
import time
import math
import torch
import random
import numpy as np
from transformers import (
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    DataCollatorForSeq2Seq
)
from datasets import load_dataset, Dataset
from typing import Dict, Any, List, Tuple
import logging
import os

from ..utils.logging import setup_logging, log_experiment_info, PowerMonitor
from ..utils.seed import set_seed
from ..utils.io import generate_exp_id, save_experiment_result, sync_to_latest
from ..models.load_hf import load_model_and_tokenizer
from ..models.lora import setup_lora, save_lora_weights, get_lora_parameters

logger = logging.getLogger(__name__)


def apply_t5_span_corruption(
    input_ids: List[int],
    tokenizer: Any,
    corruption_rate: float = 0.15,
    mean_span_length: float = 3.0
) -> Tuple[List[int], List[int]]:
    """
    Apply T5-style span corruption to a sequence.
    
    Args:
        input_ids: List of token IDs (excluding special tokens like BOS/EOS)
        tokenizer: T5 tokenizer (must support extra_id tokens)
        corruption_rate: Proportion of tokens to corrupt (default 0.15)
        mean_span_length: Mean length of corrupted spans (default 3.0)
    
    Returns:
        Tuple of (corrupted_input_ids, target_ids)
        - corrupted_input_ids: Input sequence with spans replaced by sentinel tokens
        - target_ids: Target sequence with sentinel tokens followed by corrupted spans
    """
    # Filter out special tokens that shouldn't be corrupted
    # Keep only the main content tokens
    seq = input_ids[:]
    
    # Get sentinel token IDs (<extra_id_0>, <extra_id_1>, ...)
    # T5 has 100 sentinel tokens: extra_id_0 to extra_id_99
    sentinel_ids = []
    for i in range(100):
        sentinel_token = f"<extra_id_{i}>"
        if hasattr(tokenizer, 'convert_tokens_to_ids'):
            sentinel_id = tokenizer.convert_tokens_to_ids(sentinel_token)
            if sentinel_id != tokenizer.unk_token_id:
                sentinel_ids.append(sentinel_id)
            else:
                break
        else:
            # Fallback: try to encode
            try:
                encoded = tokenizer.encode(sentinel_token, add_special_tokens=False)
                if len(encoded) == 1:
                    sentinel_ids.append(encoded[0])
                else:
                    break
            except:
                break
    
    if not sentinel_ids:
        raise ValueError("Tokenizer does not have sentinel tokens (extra_id_*)")
    
    seq_length = len(seq)
    if seq_length < 1:
        # Empty sequence
        return seq, []
    
    # For very short sequences, corrupt at least 1 token if possible
    if seq_length == 1:
        # Single token: corrupt it
        return [sentinel_ids[0]], [sentinel_ids[0], seq[0], sentinel_ids[1]]
    
    # Calculate number of tokens to corrupt
    num_to_corrupt = max(1, int(seq_length * corruption_rate))
    
    # Generate spans: select random contiguous spans to corrupt
    # Use geometric distribution for span lengths (mean = mean_span_length)
    corrupted_indices = set()
    spans = []  # List of (start, end) tuples
    sentinel_counter = 0
    
    remaining = num_to_corrupt
    max_attempts = seq_length * 10  # Prevent infinite loops
    attempts = 0
    
    while remaining > 0 and attempts < max_attempts and sentinel_counter < len(sentinel_ids):
        attempts += 1
        
        # Sample span length using geometric distribution
        # p = 1/mean_span_length, so E[length] = mean_span_length
        p = 1.0 / mean_span_length
        span_length = np.random.geometric(p)
        span_length = max(1, min(span_length, remaining, seq_length))
        
        # Random start position
        max_start = seq_length - span_length
        if max_start < 0:
            break
        
        start = random.randint(0, max_start)
        end = start + span_length
        
        # Check for overlap with existing spans
        overlap = False
        for s, e in spans:
            if not (end <= s or start >= e):  # Overlaps
                overlap = True
                break
        
        if overlap:
            continue
        
        # Check if this span would corrupt too many tokens
        new_corrupted = set(range(start, end))
        if len(corrupted_indices | new_corrupted) > num_to_corrupt * 1.5:  # Allow some overshoot
            continue
        
        # Accept this span
        spans.append((start, end))
        corrupted_indices.update(new_corrupted)
        remaining -= span_length
        sentinel_counter += 1
    
    # Fallback: if no spans were generated, corrupt at least one token
    if not spans and seq_length > 0:
        # Corrupt a single token at a random position
        corrupt_pos = random.randint(0, seq_length - 1)
        spans = [(corrupt_pos, corrupt_pos + 1)]
    
    # Sort spans by start position
    spans.sort(key=lambda x: x[0])
    
    # Build corrupted input sequence
    corrupted_input = []
    last_end = 0
    
    for i, (start, end) in enumerate(spans):
        # Add tokens before this span
        corrupted_input.extend(seq[last_end:start])
        # Add sentinel token
        corrupted_input.append(sentinel_ids[i % len(sentinel_ids)])
        last_end = end
    
    # Add remaining tokens after last span
    corrupted_input.extend(seq[last_end:])
    
    # Build target sequence: sentinel tokens followed by their corresponding spans
    target = []
    for i, (start, end) in enumerate(spans):
        # Add sentinel token
        target.append(sentinel_ids[i % len(sentinel_ids)])
        # Add the corrupted span content
        target.extend(seq[start:end])
    
    # Add final sentinel token to indicate end
    if spans:
        target.append(sentinel_ids[len(spans) % len(sentinel_ids)])
    
    return corrupted_input, target


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
            # T5-style span corruption (denoising objective)
            # Tokenize first (this will add special tokens like pad_token, eos_token if configured)
            tokenized = tokenizer(
                texts,
                truncation=True,
                padding=False,
                max_length=max_length,
                add_special_tokens=True,
            )
            
            # Apply span corruption to each sequence
            corrupted_inputs = []
            targets = []
            
            for input_ids in tokenized["input_ids"]:
                # T5 tokenizer may add special tokens during tokenization
                # We want to corrupt only the actual content tokens
                # Filter out pad tokens
                seq = [tid for tid in input_ids if tid != tokenizer.pad_token_id]
                
                # Handle EOS token: T5 typically adds it at the end
                # For span corruption, we corrupt content and keep EOS at the end of input
                # and add EOS to the target
                has_eos = False
                if tokenizer.eos_token_id is not None and seq and seq[-1] == tokenizer.eos_token_id:
                    seq = seq[:-1]
                    has_eos = True
                
                # Apply span corruption (corrupts content tokens only)
                if len(seq) > 0:
                    corrupted_input, target = apply_t5_span_corruption(
                        seq, tokenizer, corruption_rate=0.15, mean_span_length=3.0
                    )
                    
                    # Add EOS token back to input if it was present
                    if has_eos and tokenizer.eos_token_id is not None:
                        corrupted_input.append(tokenizer.eos_token_id)
                    
                    # Add EOS token to target
                    if tokenizer.eos_token_id is not None:
                        target.append(tokenizer.eos_token_id)
                    
                    # Truncate if necessary to respect max_length
                    # Input and target can have different lengths, but both should be <= max_length
                    if len(corrupted_input) > max_length:
                        corrupted_input = corrupted_input[:max_length]
                        # Ensure EOS is at the end if it exists
                        if tokenizer.eos_token_id is not None and corrupted_input[-1] != tokenizer.eos_token_id:
                            corrupted_input[-1] = tokenizer.eos_token_id
                    
                    if len(target) > max_length:
                        target = target[:max_length]
                        # Ensure EOS is at the end if it exists
                        if tokenizer.eos_token_id is not None and target[-1] != tokenizer.eos_token_id:
                            target[-1] = tokenizer.eos_token_id
                else:
                    # Empty sequence (shouldn't happen, but handle gracefully)
                    corrupted_input = seq
                    target = [tokenizer.eos_token_id] if tokenizer.eos_token_id else []
                
                corrupted_inputs.append(corrupted_input)
                targets.append(target)
            
            # Update input_ids and labels with corrupted versions
            # Create a clean dict with only the fields we need
            # This ensures no length mismatches with old attention_mask
            clean_tokenized = {
                "input_ids": corrupted_inputs,
                "labels": targets
            }
            
            # DataCollatorForSeq2Seq will handle padding and generate attention_mask
            tokenized = clean_tokenized
        
        return tokenized
    
    # Tokenize and remove original columns
    dataset = dataset.map(tokenize_function, batched=True, remove_columns=dataset.column_names)
    
    # Filter out empty sequences which can lead to NaN losses during training
    pad_token_id = tokenizer.pad_token_id
    
    def has_valid_tokens(example):
        input_ids = example.get("input_ids", [])
        if isinstance(input_ids, list):
            has_tokens = len(input_ids) > 0 and any(token != pad_token_id for token in input_ids)
        else:
            has_tokens = False
        
        if arch == "encdec":
            labels = example.get("labels", [])
            if isinstance(labels, list):
                has_label_tokens = len(labels) > 0 and any(token != pad_token_id for token in labels)
            else:
                has_label_tokens = False
            return has_tokens and has_label_tokens
        
        return has_tokens
    
    dataset = dataset.filter(has_valid_tokens)
    
    return dataset


def continued_pretraining(
    model_cfg_path: str,
    train_cfg_path: str,
    dataset_name: str = "wikitext",
    target_loss: float = 2.0,
    token_budget: int = 1000000,  # Default: 1M tokens for fair comparison
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
        
        # Ensure model dtype matches training config to avoid numerical issues
        use_fp16 = train_config.get("fp16", False)
        if use_fp16 and next(model.parameters()).dtype != torch.float16:
            model = model.half()
            logger.info("Converted model to fp16 for training")
        elif not use_fp16 and next(model.parameters()).dtype != torch.float32:
            model = model.float()
            logger.info("Converted model to fp32 for training")
        
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
            # For causal LM, ensure padding tokens are ignored in loss calculation
            data_collator = DataCollatorForLanguageModeling(
                tokenizer=tokenizer,
                mlm=False,  # Causal LM for decoder
                pad_to_multiple_of=8 if use_fp16 else None
            )
        elif model_config["arch"] == "encoder":
            data_collator = DataCollatorForLanguageModeling(
                tokenizer=tokenizer,
                mlm=True,  # Masked LM for encoder
                pad_to_multiple_of=8 if use_fp16 else None
            )
        else:  # encdec
            from transformers import DataCollatorForSeq2Seq
            data_collator = DataCollatorForSeq2Seq(
                tokenizer=tokenizer,
                model=model,  # Pass model to ensure proper padding
                pad_to_multiple_of=8 if use_fp16 else None,
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
            save_strategy="epoch",
            eval_strategy="epoch",
            # Don't save/eval during epoch to avoid confusion
            save_steps=10000,  # Set to high number to use epoch strategy
            eval_steps=10000,  # Set to high number to use epoch strategy
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
        
        # Train until target loss is reached or token budget is exhausted
        logger.info(f"Training with token budget: {token_budget:,} tokens, target loss: {target_loss}")
        
        # Calculate tokens per epoch for budget tracking
        tokens_per_epoch = len(train_dataset) * model_config.get("max_length", 256)
        logger.info(f"Estimated tokens per epoch: {tokens_per_epoch:,}")
        
        best_eval_loss = float('inf')
        epochs_trained = 0
        total_tokens = 0
        max_epochs = train_config.get("num_train_epochs", 5)
        target_reached = False
        
        for epoch in range(max_epochs):
            # Check if we would exceed token budget by training another epoch
            if total_tokens + tokens_per_epoch > token_budget:
                remaining_budget = token_budget - total_tokens
                if remaining_budget > 0:
                    logger.info(f"Token budget remaining: {remaining_budget:,} tokens (less than one epoch). "
                              f"Stopping training to stay within budget.")
                else:
                    logger.info(f"Token budget exhausted. Stopping training.")
                break
            
            logger.info(f"Epoch {epoch + 1} (Token budget remaining: {token_budget - total_tokens:,})")
            
            # Train one epoch
            train_result = trainer.train()
            epochs_trained += 1
            
            # Count tokens for this epoch
            epoch_tokens = tokens_per_epoch
            total_tokens += epoch_tokens
            
            # Check for NaN in training loss
            train_loss = train_result.training_loss
            if math.isnan(train_loss) or math.isinf(train_loss):
                logger.error(f"Training loss is NaN or Inf after epoch {epoch + 1}. Stopping training.")
                raise ValueError(f"Training instability detected: loss={train_loss}")
            if train_loss == 0.0:
                logger.warning(f"Training loss is exactly 0.0 after epoch {epoch + 1}. This may indicate model collapse.")
                # Don't stop immediately, but log a warning
                
            # Evaluate
            eval_result = trainer.evaluate()
            current_eval_loss = eval_result["eval_loss"]
            
            # Check for NaN in eval loss
            if math.isnan(current_eval_loss) or math.isinf(current_eval_loss):
                logger.warning(f"Eval loss is NaN or Inf after epoch {epoch + 1}. Using previous best loss.")
                # Skip this epoch's eval loss, keep previous best
                if best_eval_loss == float('inf'):
                    raise ValueError("Eval loss is NaN/Inf and no previous valid loss available")
                continue
            
            # Update best_eval_loss BEFORE checking target (so we track the best even if we break early)
            best_eval_loss = min(best_eval_loss, current_eval_loss)
            
            logger.info(f"Epoch {epoch + 1} - Train Loss: {train_loss:.4f}, Eval Loss: {current_eval_loss:.4f}, "
                       f"Best Eval Loss: {best_eval_loss:.4f}, Total Tokens: {total_tokens:,}/{token_budget:,}")
            
            # Check if target loss reached
            if current_eval_loss <= target_loss:
                logger.info(f"Target loss {target_loss} reached after {epochs_trained} epochs ({total_tokens:,} tokens)")
                target_reached = True
                break
            
            # Check if we've reached the token budget (shouldn't happen here, but check for safety)
            if total_tokens >= token_budget:
                logger.info(f"Token budget {token_budget:,} reached. Stopping training.")
                break
        
        # Final evaluation
        # Since load_best_model_at_end=True, Trainer has already loaded the best model
        # Get the true best eval loss from trainer state (more reliable than our manual tracking)
        final_eval = trainer.evaluate()
        final_eval_loss = final_eval["eval_loss"]
        
        # Get the actual best eval loss from trainer's state
        # Trainer.state.best_metric contains the best eval_loss when load_best_model_at_end=True
        trainer_best_eval_loss = None
        if hasattr(trainer.state, 'best_metric') and trainer.state.best_metric is not None:
            trainer_best_eval_loss = trainer.state.best_metric
            logger.info(f"Trainer's tracked best eval loss: {trainer_best_eval_loss:.4f}")
        
        # Use trainer's best_metric if available, otherwise use our manually tracked best
        # Since we're using load_best_model_at_end=True, trainer's best_metric is the authoritative source
        if trainer_best_eval_loss is not None and not math.isnan(trainer_best_eval_loss) and not math.isinf(trainer_best_eval_loss):
            # Use trainer's best_metric as the authoritative best_eval_loss
            best_eval_loss = trainer_best_eval_loss
        elif best_eval_loss != float('inf') and not math.isnan(best_eval_loss):
            # Fall back to our manual tracking if trainer's state is not available
            logger.warning("Using manually tracked best_eval_loss as trainer.state.best_metric is not available")
        else:
            # If both are invalid, use final_eval_loss
            logger.warning("Both trainer's best_metric and manual tracking are invalid. Using final_eval_loss as best.")
            best_eval_loss = final_eval_loss if not math.isnan(final_eval_loss) and not math.isinf(final_eval_loss) else float('nan')
        
        # Handle NaN in final eval loss
        if math.isnan(final_eval_loss) or math.isinf(final_eval_loss):
            logger.warning("Final eval loss is NaN/Inf, using best_eval_loss instead")
            final_eval_loss = best_eval_loss if best_eval_loss != float('inf') and not math.isnan(best_eval_loss) else float('nan')
        
        # Final sanity check: best_eval_loss should be <= final_eval_loss
        # If not, there might be an issue with how Trainer loaded the best model
        if best_eval_loss != float('inf') and not math.isnan(best_eval_loss) and not math.isnan(final_eval_loss):
            if best_eval_loss > final_eval_loss:
                logger.warning(f"best_eval_loss ({best_eval_loss:.4f}) > final_eval_loss ({final_eval_loss:.4f}). "
                             f"This suggests the best model may not have been loaded correctly. "
                             f"Using final_eval_loss as best.")
                best_eval_loss = final_eval_loss
        
        # Save LoRA weights if used
        if model_config.get("use_lora", False):
            lora_path = save_lora_weights(model, f"{output_dir}/{exp_id}/lora")
        
        # Compute efficiency metrics
        duration = time.time() - start_time
        tokens_per_second = total_tokens / duration if duration > 0 else 0
        
        results = {
            "final_eval_loss": float(final_eval_loss) if not math.isnan(final_eval_loss) and not math.isinf(final_eval_loss) else float('nan'),
            "best_eval_loss": float(best_eval_loss) if best_eval_loss != float('inf') else float('nan'),
            "epochs_trained": epochs_trained,
            "total_tokens": total_tokens,
            "token_budget": token_budget,
            "duration_seconds": duration,
            "tokens_per_second": tokens_per_second,
            "target_reached": target_reached,
            "lora_params": get_lora_parameters(model) if model_config.get("use_lora", False) else None
        }
        
        # Log experiment
        experiment_log = log_experiment_info(
            exp_id,
            {"model": model_config, "train": train_config, "dataset": dataset_name, 
             "target_loss": target_loss, "token_budget": token_budget},
            results,
            start_time,
            time.time(),
            power_monitor
        )
        
        # Save results
        save_experiment_result(exp_id, experiment_log, output_dir)
        
        # Sync to latest directory
        latest_path = sync_to_latest(experiment_log)
        logger.info(f"Synced to latest: {latest_path}")
        
        logger.info(f"Continued pretraining completed: {exp_id}")
        final_loss_str = f"{final_eval_loss:.4f}" if not math.isnan(final_eval_loss) and not math.isinf(final_eval_loss) else "NaN/Inf"
        logger.info(f"Final loss: {final_loss_str}, Tokens/sec: {tokens_per_second:.2f}")
        
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
    parser.add_argument("--token_budget", type=int, default=1000000, help="Token budget for training (default: 1M tokens)")
    parser.add_argument("--output_dir", default="results", help="Output directory")
    
    args = parser.parse_args()
    
    continued_pretraining(
        args.model_cfg,
        args.train_cfg,
        args.dataset,
        args.target_loss,
        args.token_budget,
        args.output_dir
    )


if __name__ == "__main__":
    main()
