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
