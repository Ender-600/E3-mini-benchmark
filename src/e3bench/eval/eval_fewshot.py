"""Few-shot evaluation script using lm-eval-harness."""

import argparse
import yaml
import time
import torch
from typing import Dict, Any, List
import logging
import os
import subprocess
import json

from ..utils.logging import setup_logging, log_experiment_info, PowerMonitor
from ..utils.seed import set_seed
from ..utils.io import generate_exp_id, save_experiment_result
from ..models.load_hf import load_model_and_tokenizer
from ..models.lora import load_lora_weights

logger = logging.getLogger(__name__)


def run_lm_eval(
    model_name: str,
    tasks: List[str],
    num_fewshot: int = 5,
    batch_size: int = 1,
    limit: int = 100,
    max_length: int = 512
) -> Dict[str, Any]:
    """Run lm-eval-harness evaluation."""
    
    # Build lm-eval command
    # Use context_window to limit sequence length and prevent CUDA errors
    cmd = [
        "lm_eval",
        "--model", "hf",
        "--model_args", f"pretrained={model_name},max_length={max_length}",
        "--tasks", ",".join(tasks),
        "--num_fewshot", str(num_fewshot),
        "--batch_size", str(batch_size),
        "--limit", str(limit),
        "--output_path", "temp_lm_eval_results.json"
    ]
    
    logger.info(f"Running lm-eval: {' '.join(cmd)}")
    
    try:
        # Run lm-eval
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=3600  # 1 hour timeout
        )
        
        if result.returncode != 0:
            logger.error(f"lm-eval failed: {result.stderr}")
            return {}
        
        # Load results
        if os.path.exists("temp_lm_eval_results.json"):
            with open("temp_lm_eval_results.json", 'r') as f:
                results = json.load(f)
            os.remove("temp_lm_eval_results.json")
            return results
        else:
            logger.error("lm-eval results file not found")
            return {}
            
    except subprocess.TimeoutExpired:
        logger.error("lm-eval timed out")
        return {}
    except Exception as e:
        logger.error(f"lm-eval failed: {e}")
        return {}


def evaluate_fewshot(
    model_cfg_path: str,
    eval_cfg_path: str,
    output_dir: str = "results"
) -> Dict[str, Any]:
    """Evaluate model in few-shot setting."""
    
    # Setup logging
    logger = setup_logging()
    
    # Load configurations
    with open(model_cfg_path, 'r') as f:
        model_config = yaml.safe_load(f)
    with open(eval_cfg_path, 'r') as f:
        eval_config = yaml.safe_load(f)
    
    # Generate experiment ID
    exp_id = generate_exp_id("fewshot")
    logger.info(f"Starting few-shot evaluation: {exp_id}")
    
    # Start power monitoring
    power_monitor = PowerMonitor()
    power_monitor.start()
    
    start_time = time.time()
    
    try:
        # Set seed
        set_seed(42)
        
        # Load model and tokenizer
        model, tokenizer = load_model_and_tokenizer(model_config)
        
        # Load LoRA weights if available
        if model_config.get("use_lora", False):
            lora_path = f"{output_dir}/superglue_*/{model_config['name']}/lora"
            # Find the most recent LoRA weights
            import glob
            lora_dirs = glob.glob(lora_path)
            if lora_dirs:
                latest_lora = max(lora_dirs, key=os.path.getctime)
                model = load_lora_weights(model, latest_lora)
                logger.info(f"Loaded LoRA weights from {latest_lora}")
        
        # Run evaluation
        logger.info("Running few-shot evaluation...")
        
        # Map task names for lm-eval
        task_mapping = {
            "mmlu": "mmlu",
            "arc_challenge": "arc_challenge", 
            "hellaswag": "hellaswag"
        }
        
        tasks = eval_config.get("tasks", ["mmlu"])
        lm_eval_tasks = [task_mapping.get(task, task) for task in tasks]
        
        results = run_lm_eval(
            model_name=model_config["pretrained"],
            tasks=lm_eval_tasks,
            num_fewshot=eval_config.get("num_fewshot", 5),
            batch_size=eval_config.get("batch_size", 1),
            limit=eval_config.get("limit", 100),
            max_length=eval_config.get("max_length", 512)
        )
        
        # Extract metrics
        metrics = {}
        if "results" in results:
            for task_name, task_results in results["results"].items():
                # Handle different key formats from lm-eval
                # Old format: "acc", "acc_norm"
                # New format (0.4+): "acc,none", "acc_norm,none", etc.
                for key, value in task_results.items():
                    if key.startswith("acc,"):
                        # Handle new format: "acc,none" -> extract as "acc"
                        metric_name = key.split(",")[0]  # Get "acc" from "acc,none"
                        if metric_name == "acc":
                            metrics[f"{task_name}_accuracy"] = value
                        elif metric_name == "acc_norm":
                            metrics[f"{task_name}_accuracy_norm"] = value
                    elif key == "acc":
                        metrics[f"{task_name}_accuracy"] = value
                    elif key == "acc_norm":
                        metrics[f"{task_name}_accuracy_norm"] = value
        
        # Log experiment
        end_time = time.time()
        experiment_log = log_experiment_info(
            exp_id,
            {"model": model_config, "eval": eval_config},
            metrics,
            start_time,
            end_time,
            power_monitor
        )
        
        # Save results
        save_experiment_result(exp_id, experiment_log, output_dir)
        
        logger.info(f"Few-shot evaluation completed: {exp_id}")
        for metric, value in metrics.items():
            logger.info(f"{metric}: {value:.4f}")
        
        return experiment_log
        
    except Exception as e:
        logger.error(f"Few-shot evaluation failed: {e}")
        raise
    finally:
        power_monitor.stop()


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(description="Few-shot evaluation")
    parser.add_argument("--model_cfg", required=True, help="Model config path")
    parser.add_argument("--eval_cfg", required=True, help="Evaluation config path")
    parser.add_argument("--output_dir", default="results", help="Output directory")
    
    args = parser.parse_args()
    
    evaluate_fewshot(
        args.model_cfg,
        args.eval_cfg,
        args.output_dir
    )


if __name__ == "__main__":
    main()
