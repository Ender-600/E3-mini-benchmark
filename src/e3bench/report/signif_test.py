"""Statistical significance testing for experiment results."""

import argparse
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple
import logging
import os
from scipy import stats

logger = logging.getLogger(__name__)


def run_significance_tests(
    tables_dir: str = "tables",
    output_dir: str = "tables"
) -> Dict[str, pd.DataFrame]:
    """Run statistical significance tests on aggregated results."""
    
    logger.info(f"Running significance tests on {tables_dir}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    results = {}
    
    # Load aggregated results
    training_path = os.path.join(tables_dir, "training_results.csv")
    fewshot_path = os.path.join(tables_dir, "fewshot_results.csv")
    inference_path = os.path.join(tables_dir, "inference_results.csv")
    
    # Training significance tests
    if os.path.exists(training_path):
        training_df = pd.read_csv(training_path)
        training_tests = run_training_significance_tests(training_df)
        
        if not training_tests.empty:
            training_output_path = os.path.join(output_dir, "training_significance.csv")
            training_tests.to_csv(training_output_path, index=False)
            results["training"] = training_tests
            logger.info(f"Saved training significance tests: {training_output_path}")
    
    # Few-shot significance tests
    if os.path.exists(fewshot_path):
        fewshot_df = pd.read_csv(fewshot_path)
        fewshot_tests = run_fewshot_significance_tests(fewshot_df)
        
        if not fewshot_tests.empty:
            fewshot_output_path = os.path.join(output_dir, "fewshot_significance.csv")
            fewshot_tests.to_csv(fewshot_output_path, index=False)
            results["fewshot"] = fewshot_tests
            logger.info(f"Saved few-shot significance tests: {fewshot_output_path}")
    
    # Inference significance tests
    if os.path.exists(inference_path):
        inference_df = pd.read_csv(inference_path)
        inference_tests = run_inference_significance_tests(inference_df)
        
        if not inference_tests.empty:
            inference_output_path = os.path.join(output_dir, "inference_significance.csv")
            inference_tests.to_csv(inference_output_path, index=False)
            results["inference"] = inference_tests
            logger.info(f"Saved inference significance tests: {inference_output_path}")
    
    return results


def run_training_significance_tests(df: pd.DataFrame) -> pd.DataFrame:
    """Run significance tests on training results."""
    
    tests = []
    
    # Group by task and compare architectures
    for task in df["task"].unique():
        task_df = df[df["task"] == task]
        
        # Compare architectures
        archs = task_df["arch"].unique()
        if len(archs) >= 2:
            for i, arch1 in enumerate(archs):
                for arch2 in archs[i+1:]:
                    arch1_scores = task_df[task_df["arch"] == arch1]["accuracy"].values
                    arch2_scores = task_df[task_df["arch"] == arch2]["accuracy"].values
                    
                    if len(arch1_scores) > 0 and len(arch2_scores) > 0:
                        # Paired t-test
                        if len(arch1_scores) == len(arch2_scores):
                            t_stat, p_value = stats.ttest_rel(arch1_scores, arch2_scores)
                        else:
                            t_stat, p_value = stats.ttest_ind(arch1_scores, arch2_scores)
                        
                        # Effect size (Cohen's d)
                        pooled_std = np.sqrt(((len(arch1_scores) - 1) * np.var(arch1_scores, ddof=1) + 
                                            (len(arch2_scores) - 1) * np.var(arch2_scores, ddof=1)) / 
                                           (len(arch1_scores) + len(arch2_scores) - 2))
                        cohens_d = (np.mean(arch1_scores) - np.mean(arch2_scores)) / pooled_std if pooled_std > 0 else 0
                        
                        tests.append({
                            "task": task,
                            "metric": "accuracy",
                            "group1": arch1,
                            "group2": arch2,
                            "group1_mean": np.mean(arch1_scores),
                            "group2_mean": np.mean(arch2_scores),
                            "group1_std": np.std(arch1_scores),
                            "group2_std": np.std(arch2_scores),
                            "t_statistic": t_stat,
                            "p_value": p_value,
                            "cohens_d": cohens_d,
                            "significant": p_value < 0.05,
                            "effect_size": "small" if abs(cohens_d) < 0.5 else "medium" if abs(cohens_d) < 0.8 else "large"
                        })
    
    return pd.DataFrame(tests)


def run_fewshot_significance_tests(df: pd.DataFrame) -> pd.DataFrame:
    """Run significance tests on few-shot results."""
    
    tests = []
    
    # Group by task and compare architectures
    for task in df["task"].unique():
        task_df = df[df["task"] == task]
        
        # Compare architectures
        archs = task_df["arch"].unique()
        if len(archs) >= 2:
            for i, arch1 in enumerate(archs):
                for arch2 in archs[i+1:]:
                    arch1_scores = task_df[task_df["arch"] == arch1]["accuracy"].values
                    arch2_scores = task_df[task_df["arch"] == arch2]["accuracy"].values
                    
                    if len(arch1_scores) > 0 and len(arch2_scores) > 0:
                        # Paired t-test
                        if len(arch1_scores) == len(arch2_scores):
                            t_stat, p_value = stats.ttest_rel(arch1_scores, arch2_scores)
                        else:
                            t_stat, p_value = stats.ttest_ind(arch1_scores, arch2_scores)
                        
                        # Effect size
                        pooled_std = np.sqrt(((len(arch1_scores) - 1) * np.var(arch1_scores, ddof=1) + 
                                            (len(arch2_scores) - 1) * np.var(arch2_scores, ddof=1)) / 
                                           (len(arch1_scores) + len(arch2_scores) - 2))
                        cohens_d = (np.mean(arch1_scores) - np.mean(arch2_scores)) / pooled_std if pooled_std > 0 else 0
                        
                        tests.append({
                            "task": task,
                            "metric": "accuracy",
                            "group1": arch1,
                            "group2": arch2,
                            "group1_mean": np.mean(arch1_scores),
                            "group2_mean": np.mean(arch2_scores),
                            "group1_std": np.std(arch1_scores),
                            "group2_std": np.std(arch2_scores),
                            "t_statistic": t_stat,
                            "p_value": p_value,
                            "cohens_d": cohens_d,
                            "significant": p_value < 0.05,
                            "effect_size": "small" if abs(cohens_d) < 0.5 else "medium" if abs(cohens_d) < 0.8 else "large"
                        })
    
    return pd.DataFrame(tests)


def run_inference_significance_tests(df: pd.DataFrame) -> pd.DataFrame:
    """Run significance tests on inference results."""
    
    tests = []
    
    # Compare architectures on latency and throughput
    metrics = ["latency_ms", "throughput_tokens_per_sec"]
    
    for metric in metrics:
        if metric not in df.columns:
            continue
            
        # Compare architectures
        archs = df["arch"].unique()
        if len(archs) >= 2:
            for i, arch1 in enumerate(archs):
                for arch2 in archs[i+1:]:
                    arch1_scores = df[df["arch"] == arch1][metric].values
                    arch2_scores = df[df["arch"] == arch2][metric].values
                    
                    if len(arch1_scores) > 0 and len(arch2_scores) > 0:
                        # Paired t-test
                        if len(arch1_scores) == len(arch2_scores):
                            t_stat, p_value = stats.ttest_rel(arch1_scores, arch2_scores)
                        else:
                            t_stat, p_value = stats.ttest_ind(arch1_scores, arch2_scores)
                        
                        # Effect size
                        pooled_std = np.sqrt(((len(arch1_scores) - 1) * np.var(arch1_scores, ddof=1) + 
                                            (len(arch2_scores) - 1) * np.var(arch2_scores, ddof=1)) / 
                                           (len(arch1_scores) + len(arch2_scores) - 2))
                        cohens_d = (np.mean(arch1_scores) - np.mean(arch2_scores)) / pooled_std if pooled_std > 0 else 0
                        
                        tests.append({
                            "metric": metric,
                            "group1": arch1,
                            "group2": arch2,
                            "group1_mean": np.mean(arch1_scores),
                            "group2_mean": np.mean(arch2_scores),
                            "group1_std": np.std(arch1_scores),
                            "group2_std": np.std(arch2_scores),
                            "t_statistic": t_stat,
                            "p_value": p_value,
                            "cohens_d": cohens_d,
                            "significant": p_value < 0.05,
                            "effect_size": "small" if abs(cohens_d) < 0.5 else "medium" if abs(cohens_d) < 0.8 else "large"
                        })
    
    return pd.DataFrame(tests)


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(description="Run significance tests")
    parser.add_argument("--tables", default="tables", help="Tables directory")
    parser.add_argument("--output_dir", default="tables", help="Output directory")
    
    args = parser.parse_args()
    
    run_significance_tests(args.tables, args.output_dir)


if __name__ == "__main__":
    main()
