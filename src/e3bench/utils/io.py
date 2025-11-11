"""IO utilities for saving and loading experiment data."""

import json
import csv
import os
from datetime import datetime
from typing import Dict, Any, List, Union
import pandas as pd


def save_json(data: Dict[str, Any], filepath: str) -> None:
    """Save data to JSON file with proper formatting."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2, default=str)


def load_json(filepath: str) -> Dict[str, Any]:
    """Load data from JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)


def save_csv(data: List[Dict[str, Any]], filepath: str) -> None:
    """Save list of dictionaries to CSV file."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    if not data:
        return
        
    df = pd.DataFrame(data)
    df.to_csv(filepath, index=False)


def load_csv(filepath: str) -> pd.DataFrame:
    """Load CSV file as pandas DataFrame."""
    return pd.read_csv(filepath)


def generate_exp_id(prefix: str = "exp") -> str:
    """Generate unique experiment ID with timestamp."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{prefix}_{timestamp}"


def save_experiment_result(
    exp_id: str,
    result: Dict[str, Any],
    results_dir: str = "results"
) -> str:
    """Save experiment result to JSON file."""
    filepath = os.path.join(results_dir, f"{exp_id}.json")
    save_json(result, filepath)
    return filepath


def sync_to_latest(
    result: Dict[str, Any],
    latest_dir: str = "latest"
) -> str:
    """
    Sync experiment result to latest/ directory.
    
    This maintains only the most recent experiment of each type/model combination.
    The filename is determined by experiment type and model name to enable automatic
    overwriting of previous results.
    
    Args:
        result: Experiment result dictionary containing exp_id, config, etc.
        latest_dir: Base directory for latest results (default: "latest")
    
    Returns:
        Path to the saved file in latest/
    """
    exp_id = result.get("exp_id", "unknown")
    
    # Determine experiment type from exp_id
    if "cont_pretrain" in exp_id:
        exp_type = "cont_pretrain"
    elif "superglue" in exp_id:
        exp_type = "superglue"
    elif "fewshot" in exp_id:
        exp_type = "fewshot"
    elif "inference" in exp_id:
        exp_type = "inference"
    else:
        exp_type = "other"
    
    # Extract model name from config (support both old and new format)
    config = result.get("config", {})
    model_config = config.get("model", {})
    
    # If model_config is empty, try old format (model at top level)
    if not model_config:
        model_config = result.get("model", {})
    
    model_name = model_config.get("name", "unknown")
    
    # For few-shot, include num_fewshot in filename to distinguish 0/5/10 shot
    if exp_type == "fewshot":
        eval_config = config.get("eval", {})
        num_fewshot = eval_config.get("num_fewshot", 0)
        filename = f"{model_name}-{num_fewshot}shot.json"
    else:
        filename = f"{model_name}.json"
    
    # Create directory structure: latest/<exp_type>/
    exp_type_dir = os.path.join(latest_dir, exp_type)
    os.makedirs(exp_type_dir, exist_ok=True)
    
    # Save to latest directory
    filepath = os.path.join(exp_type_dir, filename)
    save_json(result, filepath)
    
    return filepath
