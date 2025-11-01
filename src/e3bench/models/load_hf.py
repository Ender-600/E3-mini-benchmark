"""HuggingFace model loading utilities."""

import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    BitsAndBytesConfig
)
from typing import Dict, Any, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


def load_model_and_tokenizer(
    config: Dict[str, Any],
    num_labels: Optional[int] = None,
    use_8bit: bool = False,
    for_pretraining: bool = False
) -> Tuple[Any, Any]:
    """Load model and tokenizer from HuggingFace."""
    
    model_name = config["pretrained"]
    arch = config["arch"]
    dtype = config.get("dtype", "fp16")
    max_length = config.get("max_length", 256)
    
    logger.info(f"Loading {arch} model: {model_name}")
    
    # Configure quantization if requested
    quantization_config = None
    if use_8bit:
        try:
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_threshold=6.0
            )
            logger.info("Using 8-bit quantization")
        except ImportError:
            logger.warning("bitsandbytes not available, falling back to fp16")
            use_8bit = False
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Add padding token if not present
    if tokenizer.pad_token is None:
        if tokenizer.eos_token:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
        else:
            # For GPT-2 and similar models, add a pad token
            tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids('[PAD]')
            # For decoder models (GPT-2), update model embeddings to include new pad token
            if arch == "decoder" and num_labels is not None:
                # GPT2ForSequenceClassification will be resized after loading
                pass
    
    # Load model based on architecture
    if arch == "encoder":
        if for_pretraining:
            # For pretraining, use masked language modeling
            from transformers import AutoModelForMaskedLM
            model = AutoModelForMaskedLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if dtype == "fp16" else torch.float32,
                quantization_config=quantization_config,
                attn_implementation=config.get("attn_impl", "standard")
            )
        else:
            model = AutoModelForSequenceClassification.from_pretrained(
                model_name,
                num_labels=num_labels or 2,
                torch_dtype=torch.float16 if dtype == "fp16" else torch.float32,
                quantization_config=quantization_config,
                attn_implementation=config.get("attn_impl", "standard")
            )
    elif arch == "decoder":
        # For decoder models used in classification tasks, use sequence classification wrapper
        if for_pretraining:
            # For pretraining, always use causal LM
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if dtype == "fp16" else torch.float32,
                quantization_config=quantization_config,
                attn_implementation=config.get("attn_impl", "standard")
            )
        elif num_labels is not None:
            from transformers import GPT2ForSequenceClassification
            model = GPT2ForSequenceClassification.from_pretrained(
                model_name,
                num_labels=num_labels,
                torch_dtype=torch.float16 if dtype == "fp16" else torch.float32,
                quantization_config=quantization_config,
                attn_implementation=config.get("attn_impl", "standard")
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if dtype == "fp16" else torch.float32,
                quantization_config=quantization_config,
                attn_implementation=config.get("attn_impl", "standard")
            )
    elif arch == "encdec":
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if dtype == "fp16" else torch.float32,
            quantization_config=quantization_config,
            attn_implementation=config.get("attn_impl", "standard")
        )
    else:
        raise ValueError(f"Unknown architecture: {arch}")
    
    # Resize token embeddings if new tokens were added
    if len(tokenizer) > model.config.vocab_size:
        model.resize_token_embeddings(len(tokenizer))
    
    # Set pad_token_id in model config
    if hasattr(model.config, 'pad_token_id') and tokenizer.pad_token_id is not None:
        model.config.pad_token_id = tokenizer.pad_token_id
    
    # Set model to evaluation mode by default
    model.eval()
    
    logger.info(f"Loaded {arch} model with {model.num_parameters():,} parameters")
    
    return model, tokenizer


def get_model_device(model: Any) -> torch.device:
    """Get the device of the model."""
    return next(model.parameters()).device


def move_model_to_device(model: Any, device: torch.device) -> Any:
    """Move model to specified device."""
    return model.to(device)
