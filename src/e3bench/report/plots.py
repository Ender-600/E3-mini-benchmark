"""Generate plots for E³ Mini-Benchmark results."""

import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, Any, List
import logging
import os
import seaborn as sns
import math

logger = logging.getLogger(__name__)


def generate_plots(
    tables_dir: str = "tables",
    output_dir: str = "figs"
) -> None:
    """Generate all plots from aggregated results."""
    
    logger.info(f"Generating plots from {tables_dir}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Set style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Load data
    training_path = os.path.join(tables_dir, "training_results.csv")
    fewshot_path = os.path.join(tables_dir, "fewshot_results.csv")
    inference_path = os.path.join(tables_dir, "inference_results.csv")
    pretraining_path = os.path.join(tables_dir, "pretraining_results.csv")
    
    # Generate plots
    if os.path.exists(training_path):
        generate_training_plots(pd.read_csv(training_path), output_dir)
    
    if os.path.exists(fewshot_path):
        generate_fewshot_plots(pd.read_csv(fewshot_path), output_dir)
    
    if os.path.exists(inference_path):
        generate_inference_plots(pd.read_csv(inference_path), output_dir)
    
    if os.path.exists(pretraining_path):
        generate_pretraining_plots(pd.read_csv(pretraining_path), output_dir)
    
    logger.info(f"Plots saved to {output_dir}")


def generate_training_plots(df: pd.DataFrame, output_dir: str) -> None:
    """Generate training performance plots."""
    
    # 1. Accuracy by architecture and task
    plt.figure(figsize=(12, 8))
    
    # Group by task and architecture
    task_arch_means = df.groupby(['task', 'arch'])['accuracy'].mean().unstack()
    
    ax = task_arch_means.plot(kind='bar', figsize=(12, 8))
    plt.title('SuperGLUE Fine-tuning Accuracy by Architecture', fontsize=14, fontweight='bold')
    plt.xlabel('Task', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.legend(title='Architecture', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_accuracy_by_arch.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Training efficiency (time vs accuracy)
    plt.figure(figsize=(10, 6))
    
    # Use different markers for each architecture
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h']
    
    for idx, arch in enumerate(df['arch'].unique()):
        arch_df = df[df['arch'] == arch]
        plt.scatter(arch_df['duration_seconds'], arch_df['accuracy'], 
                   label=arch, alpha=0.7, s=80, marker=markers[idx % len(markers)])
    
    plt.xlabel('Training Time (seconds)', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.title('Training Efficiency: Time vs Accuracy', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_efficiency.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Memory usage by architecture
    plt.figure(figsize=(10, 6))
    
    memory_data = df.groupby('arch')['max_memory_gb'].agg(['mean', 'std']).reset_index()
    
    bars = plt.bar(memory_data['arch'], memory_data['mean'], 
                   yerr=memory_data['std'], capsize=5, alpha=0.7)
    plt.xlabel('Architecture', fontsize=12)
    plt.ylabel('Max Memory Usage (GB)', fontsize=12)
    plt.title('Memory Usage by Architecture', fontsize=14, fontweight='bold')
    
    # Add value labels on bars
    for bar, mean_val in zip(bars, memory_data['mean']):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                f'{mean_val:.1f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_memory_usage.png'), dpi=300, bbox_inches='tight')
    plt.close()


def generate_fewshot_plots(df: pd.DataFrame, output_dir: str) -> None:
    """Generate few-shot evaluation plots."""
    
    # 1. Few-shot accuracy curves
    unique_tasks = df['task'].unique()
    num_tasks = len(unique_tasks)
    
    # Dynamically calculate subplot layout with better spacing
    if num_tasks <= 4:
        nrows, ncols = 2, 2
    elif num_tasks <= 6:
        nrows, ncols = 2, 3
    elif num_tasks <= 9:
        nrows, ncols = 3, 3
    elif num_tasks <= 12:
        nrows, ncols = 3, 4
    else:
        # For more than 12 tasks, use a more flexible layout
        ncols = 5  # Increased from 4 to make each subplot wider
        nrows = math.ceil(num_tasks / ncols)
    
    # Adjust figure size based on number of subplots - make it much larger for many tasks
    fig_width = max(20, ncols * 5)  # Increased from 4 to 5
    fig_height = max(12, nrows * 4)  # Increased from 3 to 4
    
    fig = plt.figure(figsize=(fig_width, fig_height))
    
    # Group by task and architecture
    for idx, task in enumerate(unique_tasks):
        task_df = df[df['task'] == task]
        
        ax = plt.subplot(nrows, ncols, idx + 1)
        
        for arch in task_df['arch'].unique():
            arch_df = task_df[task_df['arch'] == arch]
            num_fewshot = arch_df['num_fewshot'].values
            accuracy = arch_df['accuracy'].values
            
            # Sort by num_fewshot for proper line plotting
            sorted_data = sorted(zip(num_fewshot, accuracy))
            num_fewshot_sorted, accuracy_sorted = zip(*sorted_data)
            
            ax.plot(num_fewshot_sorted, accuracy_sorted, marker='o', label=arch, linewidth=2, markersize=6)
        
        # Smaller font sizes for many subplots
        title_fontsize = 8 if num_tasks > 20 else 10
        label_fontsize = 7 if num_tasks > 20 else 9
        legend_fontsize = 6 if num_tasks > 20 else 8
        
        ax.set_xlabel('Few-shot Examples', fontsize=label_fontsize)
        ax.set_ylabel('Accuracy', fontsize=label_fontsize)
        ax.set_title(task, fontsize=title_fontsize, fontweight='bold', pad=3)
        ax.legend(fontsize=legend_fontsize, loc='best')
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=label_fontsize-1)
    
    plt.suptitle('Few-shot Learning Curves by Task and Architecture', 
                 fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.99], h_pad=2.5, w_pad=2.0)
    plt.savefig(os.path.join(output_dir, 'fewshot_curves.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Architecture comparison heatmap
    # Create pivot table for heatmap
    pivot_data = df.groupby(['arch', 'task'])['accuracy'].mean().unstack()
    
    # Adjust figure size based on number of tasks
    num_tasks_heatmap = len(pivot_data.columns)
    fig_width = max(16, num_tasks_heatmap * 0.4)  # Scale width with number of tasks
    fig_height = max(6, len(pivot_data.index) * 1.5)  # Scale height with number of architectures
    
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    
    # Adjust annotation font size based on number of tasks
    annot_fontsize = 6 if num_tasks_heatmap > 30 else 8
    
    sns.heatmap(pivot_data, annot=True, fmt='.3f', cmap='viridis', 
                cbar_kws={'label': 'Accuracy'}, ax=ax,
                annot_kws={'fontsize': annot_fontsize})
    
    ax.set_title('Few-shot Accuracy Heatmap: Architecture vs Task', 
                 fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel('Task', fontsize=12)
    ax.set_ylabel('Architecture', fontsize=12)
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=90, ha='right', fontsize=8)
    plt.yticks(rotation=0, fontsize=10)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fewshot_heatmap.png'), dpi=300, bbox_inches='tight')
    plt.close()


def generate_inference_plots(df: pd.DataFrame, output_dir: str) -> None:
    """Generate inference benchmarking plots."""
    
    # 1. Latency comparison
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    latency_data = df.groupby('arch')['latency_ms'].agg(['mean', 'std']).reset_index()
    
    bars = plt.bar(latency_data['arch'], latency_data['mean'], 
                   yerr=latency_data['std'], capsize=5, alpha=0.7)
    plt.xlabel('Architecture', fontsize=12)
    plt.ylabel('Latency (ms)', fontsize=12)
    plt.title('Inference Latency by Architecture', fontsize=14, fontweight='bold')
    
    # Add value labels
    for bar, mean_val in zip(bars, latency_data['mean']):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                f'{mean_val:.1f}', ha='center', va='bottom')
    
    # 2. Throughput comparison
    plt.subplot(1, 2, 2)
    throughput_data = df.groupby('arch')['throughput_tokens_per_sec'].agg(['mean', 'std']).reset_index()
    
    bars = plt.bar(throughput_data['arch'], throughput_data['mean'], 
                   yerr=throughput_data['std'], capsize=5, alpha=0.7)
    plt.xlabel('Architecture', fontsize=12)
    plt.ylabel('Throughput (tokens/sec)', fontsize=12)
    plt.title('Inference Throughput by Architecture', fontsize=14, fontweight='bold')
    
    # Add value labels
    for bar, mean_val in zip(bars, throughput_data['mean']):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                f'{mean_val:.1f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'inference_performance.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Memory vs Performance trade-off
    plt.figure(figsize=(10, 6))
    
    # Use different markers for each architecture
    markers = ['o', 's', '^', 'D', 'v']
    
    for idx, arch in enumerate(df['arch'].unique()):
        arch_df = df[df['arch'] == arch]
        plt.scatter(arch_df['max_memory_gb'], arch_df['throughput_tokens_per_sec'], 
                   label=arch, alpha=0.7, s=120, marker=markers[idx % len(markers)])
    
    plt.xlabel('Max Memory Usage (GB)', fontsize=12)
    plt.ylabel('Throughput (tokens/sec)', fontsize=12)
    plt.title('Memory vs Performance Trade-off', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'inference_memory_tradeoff.png'), dpi=300, bbox_inches='tight')
    plt.close()


def generate_pretraining_plots(df: pd.DataFrame, output_dir: str) -> None:
    """Generate continued pretraining plots."""
    
    # 1. Training efficiency (tokens/sec vs epochs)
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    # Use different markers for each architecture
    markers = ['o', 's', '^', 'D', 'v']
    
    for idx, arch in enumerate(df['arch'].unique()):
        arch_df = df[df['arch'] == arch]
        plt.scatter(arch_df['epochs_trained'], arch_df['tokens_per_second'], 
                   label=arch, alpha=0.7, s=120, marker=markers[idx % len(markers)])
    
    plt.xlabel('Epochs Trained', fontsize=12)
    plt.ylabel('Tokens per Second', fontsize=12)
    plt.title('Training Efficiency: Epochs vs Speed', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    
    # 2. Target loss achievement
    plt.subplot(1, 2, 2)
    target_achievement = df.groupby('arch')['target_reached'].mean()
    
    bars = plt.bar(target_achievement.index, target_achievement.values, alpha=0.7)
    plt.xlabel('Architecture', fontsize=12)
    plt.ylabel('Target Loss Achievement Rate', fontsize=12)
    plt.title('Target Loss Achievement by Architecture', fontsize=14, fontweight='bold')
    plt.ylim(0, 1)
    
    # Add value labels
    for bar, value in zip(bars, target_achievement.values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{value:.2f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'pretraining_efficiency.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Energy efficiency
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Check if we have any valid kwh data
    has_data = False
    markers = ['o', 's', '^', 'D', 'v']
    
    for idx, arch in enumerate(df['arch'].unique()):
        arch_df = df[df['arch'] == arch]
        if 'kwh' in arch_df.columns:
            # Filter out None/NaN values
            valid_df = arch_df[arch_df['kwh'].notna()]
            if len(valid_df) > 0:
                ax.scatter(valid_df['kwh'], valid_df['tokens_per_second'], 
                          label=arch, alpha=0.7, s=120, marker=markers[idx % len(markers)])
                has_data = True
    
    if has_data:
        ax.set_xlabel('Energy Consumption (kWh)', fontsize=12)
        ax.set_ylabel('Training Speed (tokens/sec)', fontsize=12)
        ax.set_title('Energy Efficiency: Power vs Speed', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
    else:
        # If no data, show a message
        ax.text(0.5, 0.5, 'No energy consumption data available\n(Power monitoring may have failed)', 
               ha='center', va='center', fontsize=14, transform=ax.transAxes,
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        ax.set_xlabel('Energy Consumption (kWh)', fontsize=12)
        ax.set_ylabel('Training Speed (tokens/sec)', fontsize=12)
        ax.set_title('Energy Efficiency: Power vs Speed', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'pretraining_energy_efficiency.png'), dpi=300, bbox_inches='tight')
    plt.close()


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(description="Generate plots")
    parser.add_argument("--tables", default="tables", help="Tables directory")
    parser.add_argument("--out", default="figs", help="Output directory")
    
    args = parser.parse_args()
    
    generate_plots(args.tables, args.out)


if __name__ == "__main__":
    main()
