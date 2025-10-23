"""Statistical metrics and significance testing utilities."""

import numpy as np
from typing import List, Tuple, Dict, Any
from scipy import stats
import pandas as pd


def compute_mean_std(values: List[float]) -> Tuple[float, float]:
    """Compute mean and standard deviation."""
    if not values:
        return 0.0, 0.0
    return float(np.mean(values)), float(np.std(values))


def bootstrap_p_value(
    scores_a: List[float],
    scores_b: List[float],
    n_bootstrap: int = 1000,
    alternative: str = "two-sided"
) -> float:
    """Compute bootstrap p-value for comparing two sets of scores."""
    
    if len(scores_a) != len(scores_b):
        raise ValueError("Scores must have the same length for paired comparison")
    
    n = len(scores_a)
    differences = np.array(scores_a) - np.array(scores_b)
    observed_diff = np.mean(differences)
    
    # Bootstrap resampling
    bootstrap_diffs = []
    for _ in range(n_bootstrap):
        # Resample with replacement
        indices = np.random.choice(n, size=n, replace=True)
        bootstrap_diff = np.mean(differences[indices])
        bootstrap_diffs.append(bootstrap_diff)
    
    bootstrap_diffs = np.array(bootstrap_diffs)
    
    # Compute p-value based on alternative hypothesis
    if alternative == "two-sided":
        p_value = 2 * min(
            np.mean(bootstrap_diffs >= observed_diff),
            np.mean(bootstrap_diffs <= observed_diff)
        )
    elif alternative == "greater":
        p_value = np.mean(bootstrap_diffs >= observed_diff)
    elif alternative == "less":
        p_value = np.mean(bootstrap_diffs <= observed_diff)
    else:
        raise ValueError("Alternative must be 'two-sided', 'greater', or 'less'")
    
    return float(p_value)


def compute_confidence_interval(
    values: List[float],
    confidence: float = 0.95
) -> Tuple[float, float]:
    """Compute confidence interval for a list of values."""
    if not values:
        return 0.0, 0.0
        
    alpha = 1 - confidence
    lower = np.percentile(values, (alpha / 2) * 100)
    upper = np.percentile(values, (1 - alpha / 2) * 100)
    
    return float(lower), float(upper)


def aggregate_seed_results(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate results across multiple seeds."""
    if not results:
        return {}
    
    # Extract metrics from all results
    all_metrics = {}
    for result in results:
        for metric, value in result.get("metrics", {}).items():
            if metric not in all_metrics:
                all_metrics[metric] = []
            all_metrics[metric].append(value)
    
    # Compute statistics
    aggregated = {}
    for metric, values in all_metrics.items():
        mean, std = compute_mean_std(values)
        lower, upper = compute_confidence_interval(values)
        
        aggregated[metric] = {
            "mean": mean,
            "std": std,
            "values": values,
            "ci_lower": lower,
            "ci_upper": upper
        }
    
    return aggregated
