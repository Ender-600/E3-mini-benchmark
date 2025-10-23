"""Aggregate experiment results into CSV tables."""

import argparse
import json
import os
import pandas as pd
from typing import Dict, Any, List
import logging
import glob

logger = logging.getLogger(__name__)


def load_experiment_results(results_dir: str) -> List[Dict[str, Any]]:
    """Load all experiment results from JSON files."""
    
    results = []
    json_files = glob.glob(os.path.join(results_dir, "*.json"))
    
    logger.info(f"Found {len(json_files)} result files")
    
    for json_file in json_files:
        try:
            with open(json_file, 'r') as f:
                result = json.load(f)
                results.append(result)
        except Exception as e:
            logger.warning(f"Failed to load {json_file}: {e}")
            continue
    
    return results


def aggregate_training_results(results: List[Dict[str, Any]]) -> pd.DataFrame:
    """Aggregate SuperGLUE fine-tuning results."""
    
    training_data = []
    
    for result in results:
        if "superglue" not in result.get("exp_id", ""):
            continue
        
        exp_id = result["exp_id"]
        model_config = result["config"]["model"]
        task_config = result["config"]["task"]
        train_config = result["config"]["train"]
        metrics = result["metrics"]
        timing = result["timing"]
        resources = result.get("resources", {})
        power = result.get("power", {})
        
        # Extract per-task results
        for task_name, task_metrics in metrics.items():
            if isinstance(task_metrics, dict) and "eval_accuracy" in task_metrics:
                training_data.append({
                    "exp_id": exp_id,
                    "model": model_config["name"],
                    "arch": model_config["arch"],
                    "task": task_name,
                    "accuracy": task_metrics["eval_accuracy"],
                    "train_loss": task_metrics.get("train_loss", 0),
                    "eval_loss": task_metrics.get("eval_loss", 0),
                    "duration_seconds": timing["duration_seconds"],
                    "max_memory_gb": resources.get("max_memory_gb", 0),
                    "avg_watt": power.get("avg_watt", 0),
                    "kwh": power.get("kwh", 0),
                    "use_lora": model_config.get("use_lora", False),
                    "lora_r": train_config.get("lora_r", 0),
                    "lora_alpha": train_config.get("lora_alpha", 0),
                    "batch_size": train_config.get("per_device_train_batch_size", 0),
                    "learning_rate": train_config.get("learning_rate", 0),
                    "epochs": train_config.get("num_train_epochs", 0)
                })
    
    return pd.DataFrame(training_data)


def aggregate_fewshot_results(results: List[Dict[str, Any]]) -> pd.DataFrame:
    """Aggregate few-shot evaluation results."""
    
    fewshot_data = []
    
    for result in results:
        if "fewshot" not in result.get("exp_id", ""):
            continue
        
        exp_id = result["exp_id"]
        model_config = result["config"]["model"]
        eval_config = result["config"]["eval"]
        metrics = result["metrics"]
        timing = result["timing"]
        resources = result.get("resources", {})
        power = result.get("power", {})
        
        # Extract task-specific metrics
        for metric_name, value in metrics.items():
            if "_accuracy" in metric_name:
                task = metric_name.replace("_accuracy", "").replace("_accuracy_norm", "")
                fewshot_data.append({
                    "exp_id": exp_id,
                    "model": model_config["name"],
                    "arch": model_config["arch"],
                    "task": task,
                    "accuracy": value,
                    "num_fewshot": eval_config.get("num_fewshot", 0),
                    "max_length": eval_config.get("max_length", 2048),
                    "duration_seconds": timing["duration_seconds"],
                    "max_memory_gb": resources.get("max_memory_gb", 0),
                    "avg_watt": power.get("avg_watt", 0),
                    "kwh": power.get("kwh", 0)
                })
    
    return pd.DataFrame(fewshot_data)


def aggregate_inference_results(results: List[Dict[str, Any]]) -> pd.DataFrame:
    """Aggregate inference benchmarking results."""
    
    inference_data = []
    
    for result in results:
        if "inference" not in result.get("exp_id", ""):
            continue
        
        exp_id = result["exp_id"]
        model_config = result["config"]["model"]
        bench_config = result["config"]["bench"]
        metrics = result["metrics"]
        timing = result["timing"]
        resources = result.get("resources", {})
        power = result.get("power", {})
        
        inference_data.append({
            "exp_id": exp_id,
            "model": model_config["name"],
            "arch": model_config["arch"],
            "latency_ms": metrics.get("first_token_latency_ms", metrics.get("forward_pass_latency_ms", 0)),
            "latency_std_ms": metrics.get("latency_std_ms", 0),
            "throughput_tokens_per_sec": metrics.get("throughput_tokens_per_sec", 0),
            "throughput_std": metrics.get("throughput_std", 0),
            "max_memory_gb": metrics.get("max_memory_gb", resources.get("max_memory_gb", 0)),
            "current_memory_gb": metrics.get("current_memory_gb", 0),
            "duration_seconds": timing["duration_seconds"],
            "avg_watt": power.get("avg_watt", 0),
            "kwh": power.get("kwh", 0),
            "num_runs": metrics.get("total_runs", 0)
        })
    
    return pd.DataFrame(inference_data)


def aggregate_pretraining_results(results: List[Dict[str, Any]]) -> pd.DataFrame:
    """Aggregate continued pretraining results."""
    
    pretraining_data = []
    
    for result in results:
        if "cont_pretrain" not in result.get("exp_id", ""):
            continue
        
        exp_id = result["exp_id"]
        model_config = result["config"]["model"]
        train_config = result["config"]["train"]
        metrics = result["metrics"]
        timing = result["timing"]
        resources = result.get("resources", {})
        power = result.get("power", {})
        
        pretraining_data.append({
            "exp_id": exp_id,
            "model": model_config["name"],
            "arch": model_config["arch"],
            "final_eval_loss": metrics.get("final_eval_loss", 0),
            "best_eval_loss": metrics.get("best_eval_loss", 0),
            "epochs_trained": metrics.get("epochs_trained", 0),
            "total_tokens": metrics.get("total_tokens", 0),
            "tokens_per_second": metrics.get("tokens_per_second", 0),
            "target_reached": metrics.get("target_reached", False),
            "duration_seconds": timing["duration_seconds"],
            "max_memory_gb": resources.get("max_memory_gb", 0),
            "avg_watt": power.get("avg_watt", 0),
            "kwh": power.get("kwh", 0),
            "use_lora": model_config.get("use_lora", False),
            "target_loss": result["config"].get("target_loss", 2.0)
        })
    
    return pd.DataFrame(pretraining_data)


def aggregate_results(
    results_dir: str = "results",
    output_dir: str = "tables"
) -> Dict[str, pd.DataFrame]:
    """Aggregate all experiment results into CSV tables."""
    
    logger.info(f"Aggregating results from {results_dir}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load all results
    results = load_experiment_results(results_dir)
    
    if not results:
        logger.warning("No results found")
        return {}
    
    # Aggregate by experiment type
    aggregated = {}
    
    # Training results
    training_df = aggregate_training_results(results)
    if not training_df.empty:
        training_path = os.path.join(output_dir, "training_results.csv")
        training_df.to_csv(training_path, index=False)
        aggregated["training"] = training_df
        logger.info(f"Saved training results: {training_path}")
    
    # Few-shot results
    fewshot_df = aggregate_fewshot_results(results)
    if not fewshot_df.empty:
        fewshot_path = os.path.join(output_dir, "fewshot_results.csv")
        fewshot_df.to_csv(fewshot_path, index=False)
        aggregated["fewshot"] = fewshot_df
        logger.info(f"Saved few-shot results: {fewshot_path}")
    
    # Inference results
    inference_df = aggregate_inference_results(results)
    if not inference_df.empty:
        inference_path = os.path.join(output_dir, "inference_results.csv")
        inference_df.to_csv(inference_path, index=False)
        aggregated["inference"] = inference_df
        logger.info(f"Saved inference results: {inference_path}")
    
    # Pretraining results
    pretraining_df = aggregate_pretraining_results(results)
    if not pretraining_df.empty:
        pretraining_path = os.path.join(output_dir, "pretraining_results.csv")
        pretraining_df.to_csv(pretraining_path, index=False)
        aggregated["pretraining"] = pretraining_df
        logger.info(f"Saved pretraining results: {pretraining_path}")
    
    # Summary statistics
    summary_data = []
    for exp_type, df in aggregated.items():
        summary_data.append({
            "experiment_type": exp_type,
            "num_experiments": len(df),
            "unique_models": df["model"].nunique() if "model" in df.columns else 0,
            "unique_architectures": df["arch"].nunique() if "arch" in df.columns else 0
        })
    
    summary_df = pd.DataFrame(summary_data)
    summary_path = os.path.join(output_dir, "summary.csv")
    summary_df.to_csv(summary_path, index=False)
    logger.info(f"Saved summary: {summary_path}")
    
    return aggregated


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(description="Aggregate experiment results")
    parser.add_argument("--results_dir", default="results", help="Results directory")
    parser.add_argument("--out_dir", default="tables", help="Output directory")
    
    args = parser.parse_args()
    
    aggregate_results(args.results_dir, args.out_dir)


if __name__ == "__main__":
    main()
