"""LoRA (Low-Rank Adaptation) utilities for efficient fine-tuning."""

import torch
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
from typing import Dict, Any, Optional
import logging
import os

logger = logging.getLogger(__name__)


def setup_lora(
    model: Any,
    config: Dict[str, Any],
    task_type: str = "classification"
) -> Any:
    """Setup LoRA configuration and apply to model."""
    
    # Map task types
    task_type_map = {
        "classification": TaskType.SEQ_CLS,
        "causal_lm": TaskType.CAUSAL_LM,
        "seq2seq": TaskType.SEQ_2_SEQ_LM
    }
    
    peft_task_type = task_type_map.get(task_type, TaskType.SEQ_CLS)
    
    # LoRA configuration
    lora_config = LoraConfig(
        r=config.get("lora_r", 16),
        lora_alpha=config.get("lora_alpha", 32),
        target_modules=config.get("lora_target_modules", ["q_proj", "v_proj"]),
        lora_dropout=config.get("lora_dropout", 0.05),
        bias="none",
        task_type=peft_task_type
    )
    
    # Apply LoRA to model
    model = get_peft_model(model, lora_config)
    
    # Print trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    
    logger.info(f"LoRA applied: {trainable_params:,} trainable parameters ({100 * trainable_params / total_params:.2f}%)")
    
    return model


def save_lora_weights(model: Any, output_dir: str) -> str:
    """Save LoRA adapter weights."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Save only the LoRA weights
    model.save_pretrained(output_dir)
    
    logger.info(f"LoRA weights saved to {output_dir}")
    return output_dir


def load_lora_weights(model: Any, adapter_path: str) -> Any:
    """Load LoRA adapter weights."""
    if os.path.exists(adapter_path):
        model = PeftModel.from_pretrained(model, adapter_path)
        logger.info(f"LoRA weights loaded from {adapter_path}")
    else:
        logger.warning(f"LoRA adapter not found at {adapter_path}")
    
    return model


def get_lora_parameters(model: Any) -> Dict[str, int]:
    """Get LoRA parameter statistics."""
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    
    return {
        "trainable_params": trainable_params,
        "total_params": total_params,
        "trainable_percentage": 100 * trainable_params / total_params
    }
