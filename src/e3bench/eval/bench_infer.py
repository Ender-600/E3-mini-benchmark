"""Inference benchmarking script."""

import argparse
import yaml
import time
import torch
import os
from typing import Dict, Any, List, Tuple
import logging
import statistics

from ..utils.logging import setup_logging, log_experiment_info, PowerMonitor
from ..utils.seed import set_seed
from ..utils.io import generate_exp_id, save_experiment_result, sync_to_latest
from ..models.load_hf import load_model_and_tokenizer
from ..models.lora import load_lora_weights

logger = logging.getLogger(__name__)


def benchmark_decoder_inference(
    model: Any,
    tokenizer: Any,
    config: Dict[str, Any]
) -> Dict[str, float]:
    """Benchmark decoder-only model inference."""
    
    max_length = config.get("max_length", 512)
    num_tokens = config.get("num_tokens", 100)
    batch_size = config.get("batch_size", 1)
    num_runs = config.get("num_runs", 5)
    warmup_runs = config.get("warmup_runs", 2)
    
    # Test prompts
    prompts = [
        "The future of artificial intelligence is",
        "In a world where technology advances rapidly,",
        "The key to success in machine learning is",
        "Understanding natural language processing requires",
        "The development of large language models has"
    ]
    
    # Warmup runs
    logger.info("Running warmup...")
    for _ in range(warmup_runs):
        for prompt in prompts[:2]:  # Use fewer prompts for warmup
            inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                _ = model.generate(
                    **inputs,
                    max_new_tokens=min(num_tokens, 50),
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id
                )
    
    # Benchmark runs
    logger.info("Running inference benchmark...")
    latencies = []
    throughputs = []
    
    for run in range(num_runs):
        run_latencies = []
        run_throughputs = []
        
        for prompt in prompts:
            inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            
            # Measure first token latency
            start_time = time.time()
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=num_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                    use_cache=True
                )
            
            end_time = time.time()
            total_time = end_time - start_time
            
            # FIXED: Calculate metrics for seq2seq models
            # For T5 and other encoder-decoder models, outputs.shape[1] is the total generated length
            input_length = inputs["input_ids"].shape[1]
            output_length = outputs.shape[1]
            
            # For seq2seq models, the output is the generated sequence
            # We need to count actual generated tokens by decoding
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            generated_tokens = len(tokenizer.encode(generated_text, add_special_tokens=False))
            
            # Debug logging
            if run == 0:  # Only log for first run to avoid spam
                logger.info(f"Input: {prompt[:50]}...")
                logger.info(f"Generated: {generated_text[:50]}...")
                logger.info(f"Input length: {input_length}, Output length: {output_length}, Generated tokens: {generated_tokens}")
            
            if generated_tokens <= 0:
                logger.warning(f"No tokens generated for input: {prompt[:50]}...")
                continue
                
            latency = total_time / generated_tokens
            throughput = generated_tokens / total_time
            
            run_latencies.append(latency)
            run_throughputs.append(throughput)
        
        latencies.extend(run_latencies)
        throughputs.extend(run_throughputs)
    
    # Calculate statistics
    return {
        "first_token_latency_ms": statistics.mean(latencies) * 1000,
        "latency_std_ms": statistics.stdev(latencies) * 1000,
        "throughput_tokens_per_sec": statistics.mean(throughputs),
        "throughput_std": statistics.stdev(throughputs),
        "total_runs": len(latencies)
    }


def benchmark_seq2seq_inference(
    model: Any,
    tokenizer: Any,
    config: Dict[str, Any]
) -> Dict[str, float]:
    """Benchmark encoder-decoder model inference."""
    
    max_length = config.get("max_length", 512)
    num_tokens = config.get("num_tokens", 100)
    batch_size = config.get("batch_size", 1)
    num_runs = config.get("num_runs", 5)
    warmup_runs = config.get("warmup_runs", 2)
    
    # Test input-output pairs
    test_cases = [
        ("Translate to French: Hello world", "Bonjour le monde"),
        ("Summarize: The quick brown fox jumps over the lazy dog.", "A fox jumps over a dog."),
        ("Answer: What is the capital of France?", "Paris"),
        ("Complete: The weather today is", "sunny and warm"),
        ("Explain: Machine learning is", "a subset of artificial intelligence")
    ]
    
    # Warmup runs
    logger.info("Running warmup...")
    for _ in range(warmup_runs):
        for input_text, _ in test_cases[:2]:
            inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                _ = model.generate(
                    **inputs,
                    max_new_tokens=min(num_tokens, 50),
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id
                )
    
    # Benchmark runs
    logger.info("Running inference benchmark...")
    latencies = []
    throughputs = []
    
    for run in range(num_runs):
        run_latencies = []
        run_throughputs = []
        
        for input_text, _ in test_cases:
            inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            
            start_time = time.time()
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=num_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                    use_cache=True
                )
            
            end_time = time.time()
            total_time = end_time - start_time
            
            # FIXED: Calculate metrics for seq2seq models
            # For T5 and other encoder-decoder models, outputs.shape[1] is the total generated length
            input_length = inputs["input_ids"].shape[1]
            output_length = outputs.shape[1]
            
            # For seq2seq models, the output is the generated sequence
            # We need to count actual generated tokens by decoding
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            generated_tokens = len(tokenizer.encode(generated_text, add_special_tokens=False))
            
            # Debug logging
            if run == 0:  # Only log for first run to avoid spam
                logger.info(f"Input: {input_text[:50]}...")
                logger.info(f"Generated: {generated_text[:50]}...")
                logger.info(f"Input length: {input_length}, Output length: {output_length}, Generated tokens: {generated_tokens}")
            
            if generated_tokens <= 0:
                logger.warning(f"No tokens generated for input: {input_text[:50]}...")
                continue
                
            latency = total_time / generated_tokens
            throughput = generated_tokens / total_time
            
            run_latencies.append(latency)
            run_throughputs.append(throughput)
        
        latencies.extend(run_latencies)
        throughputs.extend(run_throughputs)
    
    # Calculate statistics
    return {
        "first_token_latency_ms": statistics.mean(latencies) * 1000,
        "latency_std_ms": statistics.stdev(latencies) * 1000,
        "throughput_tokens_per_sec": statistics.mean(throughputs),
        "throughput_std": statistics.stdev(throughputs),
        "total_runs": len(latencies)
    }


def benchmark_encoder_inference(
    model: Any,
    tokenizer: Any,
    config: Dict[str, Any]
) -> Dict[str, float]:
    """Benchmark encoder-only model inference (forward pass only)."""
    
    max_length = config.get("max_length", 512)
    num_runs = config.get("num_runs", 5)
    warmup_runs = config.get("warmup_runs", 2)
    
    # Test texts
    texts = [
        "This is a positive review of the product.",
        "I hate this movie, it was terrible.",
        "The weather is nice today.",
        "This is a neutral statement about technology.",
        "I love this book, it's amazing."
    ]
    
    # Warmup runs
    logger.info("Running warmup...")
    for _ in range(warmup_runs):
        for text in texts[:2]:
            inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                _ = model(**inputs)
    
    # Benchmark runs
    logger.info("Running inference benchmark...")
    latencies = []
    
    for run in range(num_runs):
        run_latencies = []
        
        for text in texts:
            inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            
            start_time = time.time()
            
            with torch.no_grad():
                outputs = model(**inputs)
            
            end_time = time.time()
            total_time = end_time - start_time
            
            run_latencies.append(total_time)
        
        latencies.extend(run_latencies)
    
    # Calculate statistics
    return {
        "forward_pass_latency_ms": statistics.mean(latencies) * 1000,
        "latency_std_ms": statistics.stdev(latencies) * 1000,
        "total_runs": len(latencies)
    }


def benchmark_inference(
    model_cfg_path: str,
    bench_cfg_path: str,
    output_dir: str = "results"
) -> Dict[str, Any]:
    """Benchmark model inference performance."""
    
    # Setup logging
    logger = setup_logging()
    
    # Load configurations
    with open(model_cfg_path, 'r') as f:
        model_config = yaml.safe_load(f)
    with open(bench_cfg_path, 'r') as f:
        bench_config = yaml.safe_load(f)
    
    # Generate experiment ID
    exp_id = generate_exp_id("inference")
    logger.info(f"Starting inference benchmark: {exp_id}")
    
    # Start power monitoring
    power_monitor = PowerMonitor()
    power_monitor.start()
    
    start_time = time.time()
    
    try:
        # Set seed
        set_seed(42)
        
        # Load model and tokenizer
        model, tokenizer = load_model_and_tokenizer(model_config)
        
        # Move model to GPU if available
        if torch.cuda.is_available():
            device = torch.device("cuda")
            model = model.to(device)
            logger.info(f"Moved model to {device}")
        else:
            logger.warning("CUDA not available, using CPU")
        
        # Load LoRA weights if available
        if model_config.get("use_lora", False):
            lora_path = f"{output_dir}/superglue_*/{model_config['name']}/lora"
            import glob
            lora_dirs = glob.glob(lora_path)
            if lora_dirs:
                latest_lora = max(lora_dirs, key=os.path.getctime)
                model = load_lora_weights(model, latest_lora)
                logger.info(f"Loaded LoRA weights from {latest_lora}")
        
        # Run appropriate benchmark based on architecture
        arch = model_config["arch"]
        logger.info(f"Benchmarking {arch} model...")
        
        if arch == "decoder":
            metrics = benchmark_decoder_inference(model, tokenizer, bench_config)
        elif arch == "encdec":
            metrics = benchmark_seq2seq_inference(model, tokenizer, bench_config)
        elif arch == "encoder":
            metrics = benchmark_encoder_inference(model, tokenizer, bench_config)
        else:
            raise ValueError(f"Unknown architecture: {arch}")
        
        # Add VRAM usage
        if torch.cuda.is_available():
            metrics["max_memory_gb"] = torch.cuda.max_memory_allocated(0) / 1024**3
            metrics["current_memory_gb"] = torch.cuda.memory_allocated(0) / 1024**3
        
        # Log experiment
        end_time = time.time()
        experiment_log = log_experiment_info(
            exp_id,
            {"model": model_config, "bench": bench_config},
            metrics,
            start_time,
            end_time,
            power_monitor
        )
        
        # Save results
        save_experiment_result(exp_id, experiment_log, output_dir)
        
        # Sync to latest directory
        latest_path = sync_to_latest(experiment_log)
        logger.info(f"Synced to latest: {latest_path}")
        
        logger.info(f"Inference benchmark completed: {exp_id}")
        for metric, value in metrics.items():
            logger.info(f"{metric}: {value:.4f}")
        
        return experiment_log
        
    except Exception as e:
        logger.error(f"Inference benchmark failed: {e}")
        raise
    finally:
        power_monitor.stop()


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(description="Inference benchmarking")
    parser.add_argument("--model_cfg", required=True, help="Model config path")
    parser.add_argument("--bench_cfg", required=True, help="Benchmark config path")
    parser.add_argument("--output_dir", default="results", help="Output directory")
    
    args = parser.parse_args()
    
    benchmark_inference(
        args.model_cfg,
        args.bench_cfg,
        args.output_dir
    )


if __name__ == "__main__":
    main()
