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


def generate_spans(
    seq_length: int,
    corruption_rate: float = 0.15,
    mean_span_length: float = 3.0
) -> List[Tuple[int, int]]:
    """Generate spans to corrupt."""
    if seq_length < 1:
        return []
    
    if seq_length == 1:
        return [(0, 1)]
    
    num_to_corrupt = max(1, int(seq_length * corruption_rate))
    corrupted_indices = set()
    spans = []
    
    remaining = num_to_corrupt
    max_attempts = seq_length * 10
    attempts = 0
    
    while remaining > 0 and attempts < max_attempts:
        attempts += 1
        
        p = 1.0 / mean_span_length
        span_length = np.random.geometric(p)
        span_length = max(1, min(span_length, remaining, seq_length))
        
        max_start = seq_length - span_length
        if max_start < 0:
            break
        
        start = random.randint(0, max_start)
        end = start + span_length
        
        overlap = False
        for s, e in spans:
            if not (end <= s or start >= e):
                overlap = True
                break
        
        if overlap:
            continue
        
        new_corrupted = set(range(start, end))
        if len(corrupted_indices | new_corrupted) > num_to_corrupt * 1.5:
            continue
        
        spans.append((start, end))
        corrupted_indices.update(new_corrupted)
        remaining -= span_length
    
    if not spans and seq_length > 0:
        corrupt_pos = random.randint(0, seq_length - 1)
        spans = [(corrupt_pos, corrupt_pos + 1)]
    
    spans.sort(key=lambda x: x[0])
    return spans


def apply_bart_span_corruption(
    input_ids: List[int],
    tokenizer: Any,
    corruption_rate: float = 0.15,
    mean_span_length: float = 3.0
) -> Tuple[List[int], List[int]]:
    """
    Apply BART-style span corruption (text infilling).
    """
    seq = input_ids[:]
    seq_length = len(seq)
    
    if seq_length < 1:
        return seq, seq
        
    mask_token_id = tokenizer.mask_token_id
    if mask_token_id is None:
        # Fallback: try to find it
        if hasattr(tokenizer, 'convert_tokens_to_ids'):
            mask_token_id = tokenizer.convert_tokens_to_ids("<mask>")
            if mask_token_id == tokenizer.unk_token_id:
                raise ValueError("Tokenizer does not have a mask token")
        else:
            raise ValueError("Tokenizer does not have a mask token")
            
    spans = generate_spans(seq_length, corruption_rate, mean_span_length)
    
    # Build corrupted input
    corrupted_input = []
    last_end = 0
    for start, end in spans:
        corrupted_input.extend(seq[last_end:start])
        corrupted_input.append(mask_token_id)
        last_end = end
    corrupted_input.extend(seq[last_end:])
    
    # Target is the original sequence
    target = seq[:]
    
    return corrupted_input, target


def apply_t5_span_corruption(
    input_ids: List[int],
    tokenizer: Any,
    corruption_rate: float = 0.15,
    mean_span_length: float = 3.0
) -> Tuple[List[int], List[int]]:
    """
    Apply T5-style span corruption to a sequence.
    """
    seq = input_ids[:]
    
    # Get sentinel token IDs
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
        return seq, []
    
    # For very short sequences, corrupt at least 1 token if possible
    if seq_length == 1:
        return [sentinel_ids[0]], [sentinel_ids[0], seq[0], sentinel_ids[1]]
    
    spans = generate_spans(seq_length, corruption_rate, mean_span_length)
    
    # Truncate spans if we run out of sentinels
    if len(spans) > len(sentinel_ids) - 1: # Reserve one for end
        spans = spans[:len(sentinel_ids) - 1]

    # Build corrupted input sequence
    corrupted_input = []
    last_end = 0
    
    for i, (start, end) in enumerate(spans):
        corrupted_input.extend(seq[last_end:start])
        corrupted_input.append(sentinel_ids[i % len(sentinel_ids)])
        last_end = end
    
    corrupted_input.extend(seq[last_end:])
    
    # Build target sequence
    target = []
    for i, (start, end) in enumerate(spans):
        target.append(sentinel_ids[i % len(sentinel_ids)])
        target.extend(seq[start:end])
    
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
    
    # Check for T5 sentinels once
    has_t5_sentinels = False
    if arch == "encdec":
        try:
            # Check for extra_id_0
            test_id = tokenizer.convert_tokens_to_ids("<extra_id_0>")
            if test_id != tokenizer.unk_token_id:
                has_t5_sentinels = True
        except:
            pass
        
        if not has_t5_sentinels:
             logger.info("Tokenizer does not have T5 sentinels. Using BART-style masking.")

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
            # T5-style span corruption or BART-style denoising
            # Tokenize first
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
                # Filter out pad tokens
                seq = [tid for tid in input_ids if tid != tokenizer.pad_token_id]
                
                has_eos = False
                if tokenizer.eos_token_id is not None and seq and seq[-1] == tokenizer.eos_token_id:
                    seq = seq[:-1]
                    has_eos = True
                
                # Apply span corruption
                if len(seq) > 0:
                    if has_t5_sentinels:
                        corrupted_input, target = apply_t5_span_corruption(
                            seq, tokenizer, corruption_rate=0.15, mean_span_length=3.0
                        )
                    else:
                        corrupted_input, target = apply_bart_span_corruption(
                            seq, tokenizer, corruption_rate=0.15, mean_span_length=3.0
                        )
                    
                    # Add EOS token back to input if it was present
                    if has_eos and tokenizer.eos_token_id is not None:
                        corrupted_input.append(tokenizer.eos_token_id)
                        if not has_t5_sentinels: # For BART, target is full seq, so add EOS
                             target.append(tokenizer.eos_token_id)
                    
                    # Add EOS to target for T5 if it doesn't have it (T5 corruption function adds sentinel at end, but maybe we need EOS too?)
                    # T5 usually ends with a sentinel.
                    # apply_t5_span_corruption adds final sentinel.
                    # apply_bart_span_corruption returns original sequence (without EOS if we stripped it).
                    
                    if has_t5_sentinels:
                        if tokenizer.eos_token_id is not None:
                             target.append(tokenizer.eos_token_id)
                    
                    # Truncate
                    if len(corrupted_input) > max_length:
                        corrupted_input = corrupted_input[:max_length]
                        if tokenizer.eos_token_id is not None and corrupted_input[-1] != tokenizer.eos_token_id:
                            corrupted_input[-1] = tokenizer.eos_token_id
                    
                    if len(target) > max_length:
                        target = target[:max_length]
                        if tokenizer.eos_token_id is not None and target[-1] != tokenizer.eos_token_id:
                            target[-1] = tokenizer.eos_token_id
                else:
                    corrupted_input = seq
                    target = [tokenizer.eos_token_id] if tokenizer.eos_token_id else []
                
                corrupted_inputs.append(corrupted_input)
                targets.append(target)
            
            clean_tokenized = {
                "input_ids": corrupted_inputs,
                "labels": targets
            }
            tokenized = clean_tokenized
        
        return tokenized
    
    # Tokenize and remove original columns
    dataset = dataset.map(tokenize_function, batched=True, remove_columns=dataset.column_names)
    
    # Filter out empty sequences
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
    token_budget: int = 1000000,
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
    
    exp_id = generate_exp_id("cont_pretrain")
    logger.info(f"Starting continued pretraining: {exp_id}")
    
    power_monitor = PowerMonitor()
    power_monitor.start()
    
    start_time = time.time()
    
    try:
        # Set seed
        seed = train_config.get("seed", [42])[0]
        set_seed(seed)
        
        # Load data
        logger.info("Loading pretraining data...")
        # Determine max_length: train_config takes precedence over model_config
        max_length = train_config.get("max_length", model_config.get("max_length", 256))
        
        dataset = load_pretraining_data(
            dataset_name=dataset_name,
            max_length=max_length,
            num_samples=1000
        )
        
        # Load model and tokenizer
        model, tokenizer = load_model_and_tokenizer(model_config, for_pretraining=True)
        
        # Ensure model dtype matches training config
        use_fp16 = train_config.get("fp16", False)
        if use_fp16 and next(model.parameters()).dtype != torch.float16:
            model = model.half()
            logger.info("Converted model to fp16 for training")
        elif not use_fp16 and next(model.parameters()).dtype != torch.float32:
            model = model.float()
            logger.info("Converted model to fp32 for training")
        
        # Setup LoRA
        if model_config.get("use_lora", False):
            task_type = "causal_lm"
            if model_config["arch"] == "encoder":
                task_type = "feature_extraction"
            elif model_config["arch"] == "encdec":
                task_type = "seq2seq"
            
            model = setup_lora(model, train_config, task_type, arch=model_config["arch"])
        
        # Preprocess data
        processed_dataset = preprocess_pretraining_data(
            dataset, tokenizer, model_config["arch"], max_length
        )
        
        # Split data
        train_size = int(0.8 * len(processed_dataset))
        eval_size = len(processed_dataset) - train_size
        
        train_dataset = processed_dataset.select(range(train_size))
        eval_dataset = processed_dataset.select(range(train_size, train_size + eval_size))
        
        # Data collator
        if model_config["arch"] == "decoder":
            data_collator = DataCollatorForLanguageModeling(
                tokenizer=tokenizer,
                mlm=False,
                pad_to_multiple_of=8 if use_fp16 else None
            )
        elif model_config["arch"] == "encoder":
            data_collator = DataCollatorForLanguageModeling(
                tokenizer=tokenizer,
                mlm=True,
                pad_to_multiple_of=8 if use_fp16 else None
            )
        else:  # encdec
            from transformers import DataCollatorForSeq2Seq
            data_collator = DataCollatorForSeq2Seq(
                tokenizer=tokenizer,
                model=model,
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
            max_grad_norm=1.0,
            save_strategy="epoch",
            eval_strategy="epoch",
            save_steps=10000,
            eval_steps=10000,
            logging_steps=10,
            save_total_limit=2,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            report_to=["wandb"],
            remove_unused_columns=True
        )
        
        # Trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
        )
        
        # Train
        logger.info(f"Training with token budget: {token_budget:,} tokens, target loss: {target_loss}")
        
        tokens_per_epoch = len(train_dataset) * max_length
        logger.info(f"Estimated tokens per epoch: {tokens_per_epoch:,}")
        
        best_eval_loss = float('inf')
        epochs_trained = 0
        total_tokens = 0
        max_epochs = train_config.get("num_train_epochs", 5)
        target_reached = False
        
        for epoch in range(max_epochs):
            if total_tokens + tokens_per_epoch > token_budget:
                remaining_budget = token_budget - total_tokens
                if remaining_budget > 0:
                    logger.info(f"Token budget remaining: {remaining_budget:,} tokens (less than one epoch). Stopping training.")
                else:
                    logger.info(f"Token budget exhausted. Stopping training.")
                break
            
            logger.info(f"Epoch {epoch + 1} (Token budget remaining: {token_budget - total_tokens:,})")
            
            train_result = trainer.train()
            epochs_trained += 1
            
            epoch_tokens = tokens_per_epoch
            total_tokens += epoch_tokens
            
            train_loss = train_result.training_loss
            if math.isnan(train_loss) or math.isinf(train_loss):
                logger.error(f"Training loss is NaN or Inf. Stopping.")
                raise ValueError(f"Training instability detected: loss={train_loss}")
                
            eval_result = trainer.evaluate()
            current_eval_loss = eval_result["eval_loss"]
            
            if math.isnan(current_eval_loss) or math.isinf(current_eval_loss):
                logger.warning(f"Eval loss is NaN or Inf. Using previous best.")
                if best_eval_loss == float('inf'):
                    raise ValueError("Eval loss is NaN/Inf and no previous valid loss")
                continue
            
            best_eval_loss = min(best_eval_loss, current_eval_loss)
            
            logger.info(f"Epoch {epoch + 1} - Train Loss: {train_loss:.4f}, Eval Loss: {current_eval_loss:.4f}, Best: {best_eval_loss:.4f}")
            
            if current_eval_loss <= target_loss:
                logger.info(f"Target loss {target_loss} reached.")
                target_reached = True
                break
            
            if total_tokens >= token_budget:
                logger.info(f"Token budget reached. Stopping.")
                break
        
        final_eval = trainer.evaluate()
        final_eval_loss = final_eval["eval_loss"]
        
        trainer_best = getattr(trainer.state, 'best_metric', None)
        if trainer_best is not None and not math.isnan(trainer_best):
            best_eval_loss = trainer_best
        elif not math.isnan(final_eval_loss):
             # Fallback if manual tracking used
             pass
        
        if math.isnan(final_eval_loss):
            final_eval_loss = best_eval_loss
            
        if model_config.get("use_lora", False):
            save_lora_weights(model, f"{output_dir}/{exp_id}/lora")
        
        duration = time.time() - start_time
        tokens_per_second = total_tokens / duration if duration > 0 else 0
        
        results = {
            "final_eval_loss": float(final_eval_loss),
            "best_eval_loss": float(best_eval_loss),
            "epochs_trained": epochs_trained,
            "total_tokens": total_tokens,
            "token_budget": token_budget,
            "duration_seconds": duration,
            "tokens_per_second": tokens_per_second,
            "target_reached": target_reached,
            "lora_params": get_lora_parameters(model) if model_config.get("use_lora", False) else None
        }
        
        experiment_log = log_experiment_info(
            exp_id,
            {"model": model_config, "train": train_config, "dataset": dataset_name, 
             "target_loss": target_loss, "token_budget": token_budget},
            results,
            start_time,
            time.time(),
            power_monitor
        )
        
        save_experiment_result(exp_id, experiment_log, output_dir)
        latest_path = sync_to_latest(experiment_log)
        logger.info(f"Synced to latest: {latest_path}")
        
        logger.info(f"Completed: {exp_id}, Final Loss: {final_eval_loss:.4f}, Tokens/sec: {tokens_per_second:.2f}")
        
        return experiment_log
        
    except Exception as e:
        logger.error(f"Continued pretraining failed: {e}")
        raise
    finally:
        power_monitor.stop()

def main():
    parser = argparse.ArgumentParser(description="Continued pretraining")
    parser.add_argument("--model_cfg", required=True, help="Model config path")
    parser.add_argument("--train_cfg", required=True, help="Training config path")
    parser.add_argument("--dataset", default="wikitext", help="Dataset name")
    parser.add_argument("--target_loss", type=float, default=2.0, help="Target validation loss")
    parser.add_argument("--token_budget", type=int, default=1000000, help="Token budget")
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
