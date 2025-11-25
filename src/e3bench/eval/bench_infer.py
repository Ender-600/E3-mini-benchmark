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
    config: Dict[str, Any],
    power_monitor: Any = None
) -> Dict[str, float]:
    """Benchmark decoder-only model inference with TTFT and TBT metrics."""
    
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
    
    # Start power monitoring after warmup
    if power_monitor:
        power_monitor.start()

    # Benchmark runs with token-by-token timing
    logger.info("Running inference benchmark with TTFT/TBT measurement...")
    ttft_list = []  # Time-To-First-Token
    tbt_list = []   # Time-Between-Tokens
    e2e_list = []   # End-to-End latency
    throughputs = []
    
    for run in range(num_runs):
        for prompt in prompts:
            inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            
            # Measure token-by-token generation
            input_ids = inputs["input_ids"]
            attention_mask = inputs.get("attention_mask", None)
            
            token_times = []
            generated_ids = input_ids.clone()
            
            start_time = time.time()
            
            with torch.no_grad():
                # Generate tokens one by one to measure TTFT and TBT
                for i in range(num_tokens):
                    token_start = time.time()
                    
                    outputs = model(
                        input_ids=generated_ids,
                        attention_mask=attention_mask,
                        use_cache=False  # Simpler for timing, could optimize later
                    )
                    
                    # Get next token (greedy decoding)
                    next_token_logits = outputs.logits[:, -1, :]
                    next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                    
                    token_end = time.time()
                    token_times.append(token_end - token_start)
                    
                    # Append new token
                    generated_ids = torch.cat([generated_ids, next_token], dim=-1)
                    
                    # Update attention mask
                    if attention_mask is not None:
                        attention_mask = torch.cat([
                            attention_mask,
                            torch.ones((attention_mask.shape[0], 1), device=attention_mask.device, dtype=attention_mask.dtype)
                        ], dim=-1)
                    
                    # Check for EOS token
                    if next_token.item() == tokenizer.eos_token_id:
                        break
            
            end_time = time.time()
            total_time = end_time - start_time
            actual_tokens = len(token_times)
            
            if actual_tokens > 0:
                # TTFT: Time to first token (prefill + first decode)
                ttft = token_times[0]
                ttft_list.append(ttft)
                
                # TBT: Average time between subsequent tokens (decode phase)
                if actual_tokens > 1:
                    tbt = statistics.mean(token_times[1:])
                    tbt_list.append(tbt)
                
                # E2E: End-to-end latency
                e2e_list.append(total_time)
                
                # Throughput: tokens per second
                throughput = actual_tokens / total_time
                throughputs.append(throughput)
                
                # Debug logging
                if run == 0:
                    logger.info(f"Input: {prompt[:50]}...")
                    logger.info(f"Generated {actual_tokens} tokens")
                    logger.info(f"TTFT: {ttft*1000:.2f}ms, TBT: {tbt*1000 if actual_tokens > 1 else 0:.2f}ms, E2E: {total_time*1000:.2f}ms")
    
    # Calculate statistics
    results = {
        "ttft_ms": statistics.mean(ttft_list) * 1000,
        "ttft_std_ms": statistics.stdev(ttft_list) * 1000 if len(ttft_list) > 1 else 0,
        "throughput_tokens_per_sec": statistics.mean(throughputs),
        "throughput_std": statistics.stdev(throughputs) if len(throughputs) > 1 else 0,
        "e2e_latency_ms": statistics.mean(e2e_list) * 1000,
        "e2e_std_ms": statistics.stdev(e2e_list) * 1000 if len(e2e_list) > 1 else 0,
        "total_runs": len(ttft_list)
    }

    # Calculate energy stats
    if power_monitor:
        power_stats = power_monitor.stop()
        if power_stats["avg_watt"] is not None:
            # Energy = Power * Time
            total_energy_joules = power_stats["avg_watt"] * power_stats["duration_seconds"]
            # Energy per sample
            total_samples = num_runs * len(prompts)
            if total_samples > 0:
                results["inference_energy_per_sample_joules"] = total_energy_joules / total_samples
            results["avg_power_watts"] = power_stats["avg_watt"]
    
    # Add TBT if we have multiple tokens
    if len(tbt_list) > 0:
        results["tbt_ms"] = statistics.mean(tbt_list) * 1000
        results["tbt_std_ms"] = statistics.stdev(tbt_list) * 1000 if len(tbt_list) > 1 else 0
    
    # Backward compatibility: keep old metric names
    results["first_token_latency_ms"] = results["ttft_ms"]
    results["latency_std_ms"] = results["ttft_std_ms"]
    
    return results


def benchmark_seq2seq_inference(
    model: Any,
    tokenizer: Any,
    config: Dict[str, Any],
    power_monitor: Any = None
) -> Dict[str, float]:
    """Benchmark encoder-decoder model inference with TTFT and TBT metrics."""
    
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
    
    # Start power monitoring after warmup
    if power_monitor:
        power_monitor.start()

    # Benchmark runs with token-by-token timing
    logger.info("Running inference benchmark with TTFT/TBT measurement...")
    ttft_list = []  # Time-To-First-Token (includes encoder + first decoder step)
    tbt_list = []   # Time-Between-Tokens (decoder-only)
    e2e_list = []   # End-to-End latency
    encoder_list = []  # Encoder processing time
    throughputs = []
    
    for run in range(num_runs):
        for input_text, _ in test_cases:
            inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            
            # Measure encoder time separately for seq2seq models
            encoder_start = time.time()
            with torch.no_grad():
                encoder_outputs = model.get_encoder()(**inputs)
            encoder_time = time.time() - encoder_start
            encoder_list.append(encoder_time)
            
            # Now measure decoder token-by-token
            token_times = []
            decoder_input_ids = torch.tensor([[tokenizer.pad_token_id]], device=model.device)
            
            start_time = time.time()
            
            with torch.no_grad():
                # Generate tokens one by one to measure TTFT and TBT
                for i in range(num_tokens):
                    token_start = time.time()
                    
                    outputs = model(
                        encoder_outputs=encoder_outputs,
                        decoder_input_ids=decoder_input_ids,
                        use_cache=False
                    )
                    
                    # Get next token (greedy decoding)
                    next_token_logits = outputs.logits[:, -1, :]
                    next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                    
                    token_end = time.time()
                    token_times.append(token_end - token_start)
                    
                    # Append new token
                    decoder_input_ids = torch.cat([decoder_input_ids, next_token], dim=-1)
                    
                    # Check for EOS token
                    if next_token.item() == tokenizer.eos_token_id:
                        break
            
            end_time = time.time()
            decoder_time = end_time - start_time
            total_time = encoder_time + decoder_time
            actual_tokens = len(token_times)
            
            if actual_tokens > 0:
                # TTFT: Encoder time + first decoder token
                ttft = encoder_time + token_times[0]
                ttft_list.append(ttft)
                
                # TBT: Average time between subsequent tokens (decoder-only)
                if actual_tokens > 1:
                    tbt = statistics.mean(token_times[1:])
                    tbt_list.append(tbt)
                
                # E2E: End-to-end latency
                e2e_list.append(total_time)
                
                # Throughput: tokens per second
                throughput = actual_tokens / total_time
                throughputs.append(throughput)
                
                # Debug logging
                if run == 0:
                    logger.info(f"Input: {input_text[:50]}...")
                    logger.info(f"Generated {actual_tokens} tokens")
                    logger.info(f"Encoder: {encoder_time*1000:.2f}ms, TTFT: {ttft*1000:.2f}ms, TBT: {tbt*1000 if actual_tokens > 1 else 0:.2f}ms, E2E: {total_time*1000:.2f}ms")
    
    # Calculate statistics
    results = {
        "ttft_ms": statistics.mean(ttft_list) * 1000,
        "ttft_std_ms": statistics.stdev(ttft_list) * 1000 if len(ttft_list) > 1 else 0,
        "encoder_latency_ms": statistics.mean(encoder_list) * 1000,
        "encoder_std_ms": statistics.stdev(encoder_list) * 1000 if len(encoder_list) > 1 else 0,
        "throughput_tokens_per_sec": statistics.mean(throughputs),
        "throughput_std": statistics.stdev(throughputs) if len(throughputs) > 1 else 0,
        "e2e_latency_ms": statistics.mean(e2e_list) * 1000,
        "e2e_std_ms": statistics.stdev(e2e_list) * 1000 if len(e2e_list) > 1 else 0,
        "total_runs": len(ttft_list)
    }

    # Calculate energy stats
    if power_monitor:
        power_stats = power_monitor.stop()
        if power_stats["avg_watt"] is not None:
            # Energy = Power * Time
            total_energy_joules = power_stats["avg_watt"] * power_stats["duration_seconds"]
            # Energy per sample
            total_samples = num_runs * len(test_cases)
            if total_samples > 0:
                results["inference_energy_per_sample_joules"] = total_energy_joules / total_samples
            results["avg_power_watts"] = power_stats["avg_watt"]
    
    # Add TBT if we have multiple tokens
    if len(tbt_list) > 0:
        results["tbt_ms"] = statistics.mean(tbt_list) * 1000
        results["tbt_std_ms"] = statistics.stdev(tbt_list) * 1000 if len(tbt_list) > 1 else 0
    
    # Backward compatibility: keep old metric names
    results["first_token_latency_ms"] = results["ttft_ms"]
    results["latency_std_ms"] = results["ttft_std_ms"]
    
    return results


def benchmark_encoder_inference(
    model: Any,
    tokenizer: Any,
    config: Dict[str, Any],
    power_monitor: Any = None
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
    
    # Start power monitoring after warmup
    if power_monitor:
        power_monitor.start()

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
    results = {
        "forward_pass_latency_ms": statistics.mean(latencies) * 1000,
        "latency_std_ms": statistics.stdev(latencies) * 1000,
        "total_runs": len(latencies)
    }

    # Calculate energy stats
    if power_monitor:
        power_stats = power_monitor.stop()
        if power_stats["avg_watt"] is not None:
            # Energy = Power * Time
            total_energy_joules = power_stats["avg_watt"] * power_stats["duration_seconds"]
            # Energy per sample
            total_samples = num_runs * len(texts)
            if total_samples > 0:
                results["inference_energy_per_sample_joules"] = total_energy_joules / total_samples
            results["avg_power_watts"] = power_stats["avg_watt"]

    return results


def benchmark_with_context_scaling(
    model: Any,
    tokenizer: Any,
    config: Dict[str, Any],
    arch: str,
    power_monitor: Any = None
) -> Dict[str, Any]:
    """
    Benchmark model across multiple context lengths to generate latency curve.
    
    Args:
        model: The model to benchmark
        tokenizer: The tokenizer
        config: Benchmark configuration with context_lengths list
        arch: Model architecture (decoder, encdec, encoder)
        power_monitor: Power monitor instance
    
    Returns:
        Dictionary with per-context-length results
    """
    context_lengths = config.get("context_lengths", [128, 256, 512, 1024])
    num_tokens = config.get("num_tokens", 50)
    num_runs = config.get("num_runs", 3)
    warmup_runs = config.get("warmup_runs", 1)
    
    results_by_length = {}
    
    logger.info(f"Running context length scaling test: {context_lengths}")
    
    for ctx_len in context_lengths:
        logger.info(f"Testing context length: {ctx_len}")
        
        # Create modified config for this context length
        ctx_config = config.copy()
        ctx_config["max_length"] = ctx_len
        ctx_config["num_runs"] = num_runs
        ctx_config["warmup_runs"] = warmup_runs
        
        # Reset GPU stats for each context length
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        
        # Run benchmark for this context length
        if arch == "decoder":
            metrics = benchmark_decoder_inference(model, tokenizer, ctx_config, power_monitor)
        elif arch == "encdec":
            metrics = benchmark_seq2seq_inference(model, tokenizer, ctx_config, power_monitor)
        elif arch == "encoder":
            metrics = benchmark_encoder_inference(model, tokenizer, ctx_config, power_monitor)
        else:
            raise ValueError(f"Unknown architecture: {arch}")
        
        # Add memory usage for this context length
        if torch.cuda.is_available():
            metrics["max_memory_gb"] = torch.cuda.max_memory_allocated(0) / 1024**3
            metrics["current_memory_gb"] = torch.cuda.memory_allocated(0) / 1024**3
        
        # Store results
        results_by_length[ctx_len] = metrics
        
        logger.info(f"Context {ctx_len}: Latency={metrics.get('first_token_latency_ms', metrics.get('latency_ms', 0)):.2f}ms")
    
    return results_by_length


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
    
    # Check if this is a scaling benchmark
    is_scaling = "context_lengths" in bench_config
    if is_scaling:
        exp_id = generate_exp_id("inference_scaling")
        logger.info(f"Running context length scaling benchmark: {exp_id}")
    
    # Start power monitoring
    # Note: We initialize it here but let the specific benchmark functions start/stop it
    # to capture only the inference phase (excluding model loading/warmup)
    power_monitor = PowerMonitor()
    
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
        
        # Choose between scaling and single-point benchmark
        if is_scaling:
            metrics = benchmark_with_context_scaling(model, tokenizer, bench_config, arch, power_monitor)
        else:
            if arch == "decoder":
                metrics = benchmark_decoder_inference(model, tokenizer, bench_config, power_monitor)
            elif arch == "encdec":
                metrics = benchmark_seq2seq_inference(model, tokenizer, bench_config, power_monitor)
            elif arch == "encoder":
                metrics = benchmark_encoder_inference(model, tokenizer, bench_config, power_monitor)
            else:
                raise ValueError(f"Unknown architecture: {arch}")
        
        # Add VRAM usage for non-scaling benchmarks
        if not is_scaling and torch.cuda.is_available():
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
        
        # Log results (handle both scaling and non-scaling formats)
        if is_scaling:
            logger.info("Context Length Scaling Results:")
            for ctx_len, ctx_metrics in metrics.items():
                latency_key = 'first_token_latency_ms' if 'first_token_latency_ms' in ctx_metrics else 'latency_ms'
                latency = ctx_metrics.get(latency_key, 0)
                logger.info(f"  {ctx_len} tokens: {latency:.2f}ms")
        else:
            for metric, value in metrics.items():
                if isinstance(value, (int, float)):
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
