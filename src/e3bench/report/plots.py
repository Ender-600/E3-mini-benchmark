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


# Global style registry for unified visualization
STYLE_CONFIG = {
    'encoder': {'color': '#2ecc71', 'marker': 'o', 'label': 'Encoder-only'},
    'decoder': {'color': '#e74c3c', 'marker': '^', 'label': 'Decoder-only'},
    'encdec':  {'color': '#3498db', 'marker': 's', 'label': 'Encoder-Decoder'}
}


def get_style(arch: str) -> Dict[str, Any]:
    """Get style configuration for a given architecture."""
    return STYLE_CONFIG.get(arch, {'color': 'gray', 'marker': 'o', 'label': arch})



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
    
    # Generate unified E3 trade-off plots
    if os.path.exists(inference_path) and os.path.exists(fewshot_path):
        generate_tradeoff_plots(tables_dir, output_dir)
    
    # Generate E³ trade-off analysis (NEW: 5 core plots for report)
    if os.path.exists(training_path) and os.path.exists(inference_path):
        generate_e3_tradeoff_analysis(tables_dir, output_dir)
    
    # Generate training bubble chart (companion to inference bubble chart)
    if os.path.exists(training_path):
        generate_training_bubble_chart(tables_dir, output_dir)
    
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

    # 1.5 SuperGLUE Accuracy Bar Chart by Model
    # Filter for SuperGLUE tasks if possible, or just use all training results as they seem to be SuperGLUE fine-tuning
    plt.figure(figsize=(14, 7))
    
    # Calculate mean accuracy per model (across all tasks)
    model_acc = df.groupby(['model', 'arch'])['accuracy'].mean().reset_index()
    model_acc = model_acc.sort_values('accuracy', ascending=True)
    
    # Create colors list based on architecture
    colors = [get_style(arch)['color'] for arch in model_acc['arch']]
    
    bars = plt.barh(model_acc['model'], model_acc['accuracy'], color=colors, alpha=0.8)
    
    plt.xlabel('Average Accuracy', fontsize=12)
    plt.ylabel('Model', fontsize=12)
    plt.title('SuperGLUE Average Performance by Model', fontsize=14, fontweight='bold')
    plt.grid(axis='x', alpha=0.3, linestyle='--')
    
    # Add legend manually
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color=get_style(arch)['color'], lw=4, label=get_style(arch)['label'])
        for arch in ['encoder', 'decoder', 'encdec']
    ]
    plt.legend(handles=legend_elements, loc='lower right')
    
    # Add value labels
    for i, v in enumerate(model_acc['accuracy']):
        plt.text(v + 0.005, i, f'{v:.3f}', va='center', fontsize=9)
        
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_superglue_perf.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Training efficiency (time vs accuracy)
    plt.figure(figsize=(10, 6))
    
    for idx, arch in enumerate(df['arch'].unique()):
        arch_df = df[df['arch'] == arch]
        style = get_style(arch)
        plt.scatter(arch_df['duration_seconds'], arch_df['accuracy'], 
                   label=style['label'], alpha=0.7, s=80, 
                   marker=style['marker'], color=style['color'])
    
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
    
    # Filter for key models comparison only: bert-base-uncased, gpt2, t5-base
    # and key tasks (e.g. mmlu main score, not subtasks) for cleaner curves
    
    key_models = ['bert-base-uncased', 'gpt2', 't5-base']
    # Filter dataset to include only these models if they exist
    # (We keep all if none of them exist, to avoid empty plots)
    if any(model in df['model'].unique() for model in key_models):
        curve_df = df[df['model'].isin(key_models)].copy()
    else:
        curve_df = df.copy()
        
    # Also filter out MMLU subtasks for the curves, keep only 'mmlu' aggregate if present
    # Assuming subtasks have underscores like 'mmlu_abstract_algebra'
    # We want to keep 'mmlu' but drop 'mmlu_...'
    # But we also want to keep other tasks like 'superglue_...' or 'hellaswag'
    # Simple heuristic: if task starts with mmlu_ and is not mmlu, drop it for the curve plot
    
    # Get list of tasks to keep
    tasks_to_keep = []
    for task in curve_df['task'].unique():
        if task.startswith('mmlu_') and task != 'mmlu':
            continue
        tasks_to_keep.append(task)
        
    curve_df = curve_df[curve_df['task'].isin(tasks_to_keep)]
    
    # 1. Few-shot accuracy curves
    unique_tasks = curve_df['task'].unique()
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
    fig_width = max(12, ncols * 4)  
    fig_height = max(8, nrows * 3) 
    
    fig = plt.figure(figsize=(fig_width, fig_height))
    
    # Group by task and architecture
    for idx, task in enumerate(unique_tasks):
        task_df = curve_df[curve_df['task'] == task]
        
        ax = plt.subplot(nrows, ncols, idx + 1)
        
        for arch in task_df['arch'].unique():
            arch_df = task_df[task_df['arch'] == arch]
            # Further breakdown by model to show specific model curves
            for model in arch_df['model'].unique():
                model_df = arch_df[arch_df['model'] == model]
                
                num_fewshot = model_df['num_fewshot'].values
                accuracy = model_df['accuracy'].values
                
                # Sort by num_fewshot for proper line plotting
                sorted_data = sorted(zip(num_fewshot, accuracy))
                num_fewshot_sorted, accuracy_sorted = zip(*sorted_data)
                
                style = get_style(arch)
                # Use model name in label for these specific curves
                label = f"{model} ({style['label']})"
                
                ax.plot(num_fewshot_sorted, accuracy_sorted, 
                       marker=style['marker'], color=style['color'], label=label, 
                       linewidth=2, markersize=6)
        
        # Smaller font sizes for many subplots
        title_fontsize = 10
        label_fontsize = 9
        legend_fontsize = 8
        
        ax.set_xlabel('Few-shot Examples', fontsize=label_fontsize)
        ax.set_ylabel('Accuracy', fontsize=label_fontsize)
        ax.set_title(task, fontsize=title_fontsize, fontweight='bold', pad=3)
        ax.legend(fontsize=legend_fontsize, loc='best')
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=label_fontsize-1)
    
    plt.suptitle('Few-shot Learning Curves (Representative Models)', 
                 fontsize=14, fontweight='bold', y=0.98, x=0.5)
    plt.tight_layout(rect=[0, 0, 1, 0.96], h_pad=2.5, w_pad=2.0)
    plt.savefig(os.path.join(output_dir, 'fewshot_curves.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Generate full fewshot curves with ALL models and ALL tasks
    generate_fewshot_curves_full(df, output_dir)
    
    # 2. Architecture comparison heatmap (Keep original full data for heatmap or filter?)
    # User asked for simplified curves, heatmap typically shows broader picture.
    # Let's keep heatmap as is for now, or we can filter it too if requested.
    # Assuming user only complained about curves being cluttered.
    
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

    # 3. MMLU Zero-shot Bar Chart (Reference to SuperGLUE style)
    generate_mmlu_bar_chart(df, output_dir, shot=0)
    
    # 4. MMLU 5-shot Bar Chart
    generate_mmlu_bar_chart(df, output_dir, shot=5)


def generate_fewshot_curves_full(df: pd.DataFrame, output_dir: str) -> None:
    """Generate few-shot curves with traditional baseline models (BERT, GPT2, T5) across ALL tasks."""
    
    # Filter for the three traditional baseline models
    baseline_models = ['bert-base-uncased', 'gpt2', 't5-base']
    full_df = df[df['model'].isin(baseline_models)].copy()
    
    if full_df.empty:
        logger.warning("No baseline models (bert-base-uncased, gpt2, t5-base) found in fewshot data.")
        return
    
    # 1. Few-shot accuracy curves (baseline models, all tasks)
    unique_tasks = full_df['task'].unique()
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
        ncols = 5
        nrows = math.ceil(num_tasks / ncols)
    
    # Adjust figure size based on number of subplots
    fig_width = max(20, ncols * 5)
    fig_height = max(12, nrows * 4)
    
    fig = plt.figure(figsize=(fig_width, fig_height))
    
    # Group by task and model
    for idx, task in enumerate(unique_tasks):
        task_df = full_df[full_df['task'] == task]
        
        ax = plt.subplot(nrows, ncols, idx + 1)
        
        for arch in task_df['arch'].unique():
            arch_df = task_df[task_df['arch'] == arch]
            # Plot each model separately
            for model in arch_df['model'].unique():
                model_df = arch_df[arch_df['model'] == model]
                
                num_fewshot = model_df['num_fewshot'].values
                accuracy = model_df['accuracy'].values
                
                # Sort by num_fewshot for proper line plotting
                sorted_data = sorted(zip(num_fewshot, accuracy))
                num_fewshot_sorted, accuracy_sorted = zip(*sorted_data)
                
                style = get_style(arch)
                label = f"{model}"
                
                ax.plot(num_fewshot_sorted, accuracy_sorted, 
                       marker=style['marker'], color=style['color'], label=label, 
                       linewidth=2, markersize=6)
        
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
    
    plt.suptitle('Few-shot Learning Curves (BERT, GPT2, T5 - All Tasks)', 
                 fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.99], h_pad=2.5, w_pad=2.0)
    plt.savefig(os.path.join(output_dir, 'fewshot_curves_full.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info("Generated full few-shot curves plot")


def generate_mmlu_bar_chart(df: pd.DataFrame, output_dir: str, shot: int) -> None:
    """Generate bar chart for MMLU performance at specific shot count."""
    
    # Filter for MMLU task and specific shot count
    # Note: df might contain 'mmlu' aggregate task or subtasks. 
    # We prefer the 'mmlu' aggregate task if available.
    
    target_task = 'mmlu'
    if target_task not in df['task'].unique():
        # Fallback: check for any mmlu-like task if aggregate missing?
        # Or maybe the user hasn't run MMLU yet.
        # Let's try to filter any task containing 'mmlu' and take mean if aggregate missing
        mmlu_df = df[(df['task'].str.contains('mmlu')) & (df['num_fewshot'] == shot)]
    else:
        mmlu_df = df[(df['task'] == target_task) & (df['num_fewshot'] == shot)]
        
    if mmlu_df.empty:
        logger.warning(f"No MMLU data found for {shot}-shot. Skipping bar chart.")
        return

    # Calculate mean accuracy per model (in case of multiple runs or subtask aggregation)
    model_acc = mmlu_df.groupby(['model', 'arch'])['accuracy'].mean().reset_index()
    model_acc = model_acc.sort_values('accuracy', ascending=True)
    
    plt.figure(figsize=(14, 7))
    
    # Create colors list based on architecture
    colors = [get_style(arch)['color'] for arch in model_acc['arch']]
    
    bars = plt.barh(model_acc['model'], model_acc['accuracy'], color=colors, alpha=0.8)
    
    plt.xlabel('Accuracy', fontsize=12)
    plt.ylabel('Model', fontsize=12)
    plt.title(f'MMLU {shot}-shot Performance by Model', fontsize=14, fontweight='bold')
    plt.grid(axis='x', alpha=0.3, linestyle='--')
    plt.xlim(0, 1.0) # Accuracy is 0-1
    
    # Add legend manually
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color=get_style(arch)['color'], lw=4, label=get_style(arch)['label'])
        for arch in ['encoder', 'decoder', 'encdec']
    ]
    plt.legend(handles=legend_elements, loc='lower right')
    
    # Add value labels
    for i, v in enumerate(model_acc['accuracy']):
        plt.text(v + 0.005, i, f'{v:.3f}', va='center', fontsize=9)
        
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'fewshot_mmlu_{shot}shot_perf.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Generated MMLU {shot}-shot bar chart")


def generate_inference_plots(df: pd.DataFrame, output_dir: str) -> None:
    """Generate inference benchmarking plots."""
    
    # Check if we have scaling data (multiple context lengths per architecture)
    has_scaling_data = 'context_length' in df.columns and len(df.groupby(['arch', 'context_length'])) > len(df['arch'].unique())
    
    if has_scaling_data:
        # Generate latency curve for scaling data
        generate_latency_curve(df, output_dir)
        # Generate TTFT/TBT breakdown if available
        if 'ttft_ms' in df.columns and df['ttft_ms'].sum() > 0:
            generate_ttft_tbt_curves(df, output_dir)
    
    # 1. Latency comparison (aggregate by architecture)
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
    
    for idx, arch in enumerate(df['arch'].unique()):
        arch_df = df[df['arch'] == arch]
        style = get_style(arch)
        plt.scatter(arch_df['max_memory_gb'], arch_df['throughput_tokens_per_sec'], 
                   label=style['label'], alpha=0.7, s=120, 
                   marker=style['marker'], color=style['color'])
    
    plt.xlabel('Max Memory Usage (GB)', fontsize=12)
    plt.ylabel('Throughput (tokens/sec)', fontsize=12)
    plt.title('Memory vs Performance Trade-off', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'inference_memory_tradeoff.png'), dpi=300, bbox_inches='tight')
    plt.close()


def generate_latency_curve(df: pd.DataFrame, output_dir: str) -> None:
    """Generate latency vs context length curve."""
    
    # Filter to only scaling experiments
    scaling_df = df[df['context_length'].notna()].copy()
    
    if scaling_df.empty:
        logger.warning("No scaling data found for latency curve")
        return
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot 1: End-to-End Latency vs Context Length
    for arch in scaling_df['arch'].unique():
        arch_df = scaling_df[scaling_df['arch'] == arch]
        
        # Breakdown by model to show individual curves
        for model in arch_df['model'].unique():
            model_df = arch_df[arch_df['model'] == model]
            
            # Use E2E latency if available, otherwise fall back
            if 'e2e_latency_ms' in model_df.columns and model_df['e2e_latency_ms'].sum() > 0:
                latency_col = 'e2e_latency_ms'
                std_col = 'e2e_std_ms'
            else:
                latency_col = 'latency_ms'
                std_col = 'latency_std_ms'
            
            # Group by context_length
            agg_dict = {latency_col: ['mean', 'std']}
            if std_col in model_df.columns:
                agg_dict[std_col] = 'first'
                
            grouped = model_df.groupby('context_length').agg(agg_dict).reset_index()
            
            if std_col in model_df.columns:
                grouped.columns = ['context_length', 'latency_ms', 'latency_calculated_std', 'latency_std_ms']
                grouped['final_std'] = grouped['latency_calculated_std'].fillna(grouped['latency_std_ms']).fillna(0)
            else:
                grouped.columns = ['context_length', 'latency_ms', 'latency_calculated_std']
                grouped['final_std'] = grouped['latency_calculated_std'].fillna(0)
                
            style = get_style(arch)
            label = f"{model} ({style['label']})"
            
            ax1.plot(grouped['context_length'], grouped['latency_ms'], 
                    marker=style['marker'], color=style['color'], 
                    label=label, linewidth=2, markersize=8)
            
            if grouped['final_std'].sum() > 0:
                ax1.fill_between(grouped['context_length'], 
                                grouped['latency_ms'] - grouped['final_std'],
                                grouped['latency_ms'] + grouped['final_std'],
                                color=style['color'], alpha=0.1)
            
            # Add text label at the end of the line
            last_point = grouped.iloc[-1]
            ax1.text(last_point['context_length'], last_point['latency_ms'], 
                    f' {model}', fontsize=7, va='center', color=style['color'])

    ax1.set_xlabel('Context Length (tokens)', fontsize=12)

    ax1.set_ylabel('E2E Latency (ms)', fontsize=12)
    ax1.set_title('End-to-End Inference Latency vs Context Length', fontsize=14, fontweight='bold')
    # Create custom legend for architectures only
    from matplotlib.lines import Line2D
    arch_legend = []
    present_archs = scaling_df['arch'].unique()
    for arch in ['encoder', 'decoder', 'encdec']:
        if arch in present_archs:
            style = get_style(arch)
            arch_legend.append(Line2D([0], [0], color=style['color'], marker=style['marker'], 
                                     linestyle='-', markersize=8, linewidth=2, label=style['label']))
    ax1.legend(handles=arch_legend, fontsize=10, loc='best', framealpha=0.9)
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.set_xscale('linear')
    
    # Plot 2: Memory Usage vs Context Length
    for arch in scaling_df['arch'].unique():
        arch_df = scaling_df[scaling_df['arch'] == arch]
        
        # Breakdown by model
        for model in arch_df['model'].unique():
            model_df = arch_df[arch_df['model'] == model]
            
            grouped = model_df.groupby('context_length').agg({
                'max_memory_gb': 'mean'
            }).reset_index()
            
            style = get_style(arch)
            label = f"{model} ({style['label']})"
            
            ax2.plot(grouped['context_length'], grouped['max_memory_gb'], 
                    marker=style['marker'], color=style['color'], 
                    label=label, linewidth=2, markersize=8)
            
            # Add text label at the end of the line
            last_point = grouped.iloc[-1]
            ax2.text(last_point['context_length'], last_point['max_memory_gb'], 
                    f' {model}', fontsize=7, va='center', color=style['color'])
    
    ax2.set_xlabel('Context Length (tokens)', fontsize=12)
    ax2.set_ylabel('Peak Memory (GB)', fontsize=12)
    ax2.set_title('Memory Usage vs Context Length', fontsize=14, fontweight='bold')
    # Create custom legend for architectures only
    from matplotlib.lines import Line2D
    arch_legend = []
    present_archs = scaling_df['arch'].unique()
    for arch in ['encoder', 'decoder', 'encdec']:
        if arch in present_archs:
            style = get_style(arch)
            arch_legend.append(Line2D([0], [0], color=style['color'], marker=style['marker'], 
                                     linestyle='-', markersize=8, linewidth=2, label=style['label']))
    ax2.legend(handles=arch_legend, fontsize=10, loc='best', framealpha=0.9)
    ax2.grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'latency_curve.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info("Generated latency curve plot")


def generate_ttft_tbt_curves(df: pd.DataFrame, output_dir: str) -> None:
    """
    Generate TTFT (Time-To-First-Token) and TBT (Time-Between-Tokens) curves.
    
    This shows the breakdown of latency into:
    - TTFT: Prefill + first decode (includes context encoding)
    - TBT: Average decode time for subsequent tokens
    - E2E: End-to-end latency
    """
    
    # Filter to only scaling experiments with TTFT/TBT data
    scaling_df = df[(df['context_length'].notna()) & (df['ttft_ms'] > 0)].copy()
    
    if scaling_df.empty:
        logger.warning("No TTFT/TBT data found for latency breakdown")
        return
    
    # Create figure with three subplots
    fig = plt.figure(figsize=(20, 6))
    
    # Plot 1: TTFT vs Context Length
    ax1 = plt.subplot(1, 3, 1)
    for arch in scaling_df['arch'].unique():
        arch_df = scaling_df[scaling_df['arch'] == arch]
        
        # Breakdown by model
        for model in arch_df['model'].unique():
            model_df = arch_df[arch_df['model'] == model]
            
            # Calculate std
            grouped = model_df.groupby('context_length').agg({
                'ttft_ms': ['mean', 'std'],
                'ttft_std_ms': 'first'
            }).reset_index()
            grouped.columns = ['context_length', 'ttft_ms', 'ttft_calculated_std', 'ttft_std_ms']
            grouped['final_std'] = grouped['ttft_calculated_std'].fillna(grouped['ttft_std_ms']).fillna(0)
            
            style = get_style(arch)
            label = f"{model}"
            
            ax1.plot(grouped['context_length'], grouped['ttft_ms'], 
                    marker=style['marker'], color=style['color'], 
                    label=label, linewidth=2, markersize=8)
            
            if grouped['final_std'].sum() > 0:
                ax1.fill_between(grouped['context_length'], 
                                grouped['ttft_ms'] - grouped['final_std'],
                                grouped['ttft_ms'] + grouped['final_std'],
                                color=style['color'], alpha=0.1)
            
            # Add text label at the end of the line
            last_point = grouped.iloc[-1]
            ax1.text(last_point['context_length'], last_point['ttft_ms'], 
                    f' {model}', fontsize=7, va='center', color=style['color'])
    
    ax1.set_xlabel('Context Length (tokens)', fontsize=12)
    ax1.set_ylabel('TTFT (ms)', fontsize=12)
    ax1.set_title('Time-To-First-Token (TTFT)\nPrefill + First Decode', fontsize=13, fontweight='bold')
    # Create custom legend for architectures only (no individual models)
    from matplotlib.lines import Line2D
    arch_legend = []
    present_archs = scaling_df['arch'].unique()
    for arch in ['encoder', 'decoder', 'encdec']:
        if arch in present_archs:
            style = get_style(arch)
            arch_legend.append(Line2D([0], [0], color=style['color'], marker=style['marker'], 
                                     linestyle='-', markersize=8, linewidth=2, label=style['label']))
    ax1.legend(handles=arch_legend, fontsize=10, loc='best', framealpha=0.9)
    ax1.grid(True, alpha=0.3, linestyle='--')
    
    # Plot 2: TBT vs Context Length
    ax2 = plt.subplot(1, 3, 2)
    for arch in scaling_df['arch'].unique():
        arch_df = scaling_df[scaling_df['arch'] == arch]
        
        # Breakdown by model
        for model in arch_df['model'].unique():
            model_df = arch_df[arch_df['model'] == model]
            
            # Only plot if TBT data is available
            if 'tbt_ms' in model_df.columns and model_df['tbt_ms'].sum() > 0:
                grouped = model_df.groupby('context_length').agg({
                    'tbt_ms': ['mean', 'std'],
                    'tbt_std_ms': 'first'
                }).reset_index()
                grouped.columns = ['context_length', 'tbt_ms', 'tbt_calculated_std', 'tbt_std_ms']
                grouped['final_std'] = grouped['tbt_calculated_std'].fillna(grouped['tbt_std_ms']).fillna(0)
                
                style = get_style(arch)
                label = f"{model}"
                
                ax2.plot(grouped['context_length'], grouped['tbt_ms'], 
                        marker=style['marker'], color=style['color'], 
                        label=label, linewidth=2, markersize=8)
                
                if grouped['final_std'].sum() > 0:
                    ax2.fill_between(grouped['context_length'], 
                                    grouped['tbt_ms'] - grouped['final_std'],
                                    grouped['tbt_ms'] + grouped['final_std'],
                                    color=style['color'], alpha=0.1)
                
                # Add text label at the end of the line
                last_point = grouped.iloc[-1]
                ax2.text(last_point['context_length'], last_point['tbt_ms'], 
                        f' {model}', fontsize=7, va='center', color=style['color'])
    
    ax2.set_xlabel('Context Length (tokens)', fontsize=12)
    ax2.set_ylabel('TBT (ms)', fontsize=12)
    ax2.set_title('Time-Between-Tokens (TBT)\nAverage Decode Latency', fontsize=13, fontweight='bold')
    # Create custom legend for architectures only
    from matplotlib.lines import Line2D
    arch_legend = []
    # For TBT, only decoder and encdec are relevant
    present_archs = scaling_df[scaling_df['tbt_ms'] > 0]['arch'].unique()
    for arch in ['decoder', 'encdec']:
        if arch in present_archs:
            style = get_style(arch)
            arch_legend.append(Line2D([0], [0], color=style['color'], marker=style['marker'], 
                                     linestyle='-', markersize=8, linewidth=2, label=style['label']))
    ax2.legend(handles=arch_legend, fontsize=10, loc='best', framealpha=0.9)
    ax2.grid(True, alpha=0.3, linestyle='--')
    
    # Plot 3: E2E Latency vs Context Length
    ax3 = plt.subplot(1, 3, 3)
    for arch in scaling_df['arch'].unique():
        arch_df = scaling_df[scaling_df['arch'] == arch]
        
        # Breakdown by model
        for model in arch_df['model'].unique():
            model_df = arch_df[arch_df['model'] == model]
            
            if 'e2e_latency_ms' in model_df.columns and model_df['e2e_latency_ms'].sum() > 0:
                grouped = model_df.groupby('context_length').agg({
                    'e2e_latency_ms': ['mean', 'std'],
                    'e2e_std_ms': 'first'
                }).reset_index()
                grouped.columns = ['context_length', 'e2e_latency_ms', 'e2e_calculated_std', 'e2e_std_ms']
                grouped['final_std'] = grouped['e2e_calculated_std'].fillna(grouped['e2e_std_ms']).fillna(0)
                
                style = get_style(arch)
                label = f"{model}"
                
                ax3.plot(grouped['context_length'], grouped['e2e_latency_ms'], 
                        marker=style['marker'], color=style['color'], 
                        label=label, linewidth=2, markersize=8)
                
                if grouped['final_std'].sum() > 0:
                    ax3.fill_between(grouped['context_length'], 
                                    grouped['e2e_latency_ms'] - grouped['final_std'],
                                    grouped['e2e_latency_ms'] + grouped['final_std'],
                                    color=style['color'], alpha=0.1)
                
                # Add text label at the end of the line
                last_point = grouped.iloc[-1]
                ax3.text(last_point['context_length'], last_point['e2e_latency_ms'], 
                        f' {model}', fontsize=7, va='center', color=style['color'])
    
    ax3.set_xlabel('Context Length (tokens)', fontsize=12)
    ax3.set_ylabel('E2E Latency (ms)', fontsize=12)
    ax3.set_title('End-to-End (E2E) Latency\nTotal Generation Time', fontsize=13, fontweight='bold')
    # Create custom legend for architectures only
    from matplotlib.lines import Line2D
    arch_legend = []
    present_archs = scaling_df[scaling_df['e2e_latency_ms'] > 0]['arch'].unique()
    for arch in ['encoder', 'decoder', 'encdec']:
        if arch in present_archs:
            style = get_style(arch)
            arch_legend.append(Line2D([0], [0], color=style['color'], marker=style['marker'], 
                                     linestyle='-', markersize=8, linewidth=2, label=style['label']))
    ax3.legend(handles=arch_legend, fontsize=10, loc='best', framealpha=0.9)
    ax3.grid(True, alpha=0.3, linestyle='--')
    
    plt.suptitle('Latency Breakdown: TTFT vs TBT vs E2E', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'ttft_tbt_curves.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info("Generated TTFT/TBT breakdown plot")
    
    # Generate additional plot: Stacked bar chart for latency composition
    generate_latency_composition_chart(scaling_df, output_dir)
    
    # Generate bar chart for specific context length (512)
    generate_latency_bar_chart(scaling_df, output_dir, target_ctx=512)


def generate_latency_composition_chart(df: pd.DataFrame, output_dir: str) -> None:
    """
    Generate stacked bar chart showing latency composition (TTFT vs TBT contribution).
    """
    
    fig, axes = plt.subplots(1, len(df['arch'].unique()), figsize=(18, 6), sharey=True)
    
    if len(df['arch'].unique()) == 1:
        axes = [axes]
    
    for idx, arch in enumerate(df['arch'].unique()):
        arch_df = df[df['arch'] == arch]
        
        # Group by context length
        grouped = arch_df.groupby('context_length').agg({
            'ttft_ms': 'mean',
            'tbt_ms': 'mean',
            'e2e_latency_ms': 'mean'
        }).reset_index()
        
        # Calculate contribution of TBT to total time
        # E2E ≈ TTFT + (m-1) * TBT, we'll estimate based on available data
        context_lengths = grouped['context_length'].values
        ttft = grouped['ttft_ms'].values
        tbt = grouped['tbt_ms'].values if 'tbt_ms' in grouped.columns else np.zeros_like(ttft)
        
        # Calculate decode contribution (rough estimate)
        # Assuming we generated ~50 tokens on average
        decode_contribution = tbt * 49  # (m-1) where m=50
        
        # Create stacked bar chart
        x = np.arange(len(context_lengths))
        width = 0.6
        
        style = get_style(arch)
        base_color = style['color']
        
        # TTFT (prefill + first decode)
        axes[idx].bar(x, ttft, width, label='TTFT (Prefill)', color=base_color, alpha=0.8)
        # Decode contribution (subsequent tokens)
        axes[idx].bar(x, decode_contribution, width, bottom=ttft, 
                     label='Decode (TBT × tokens)', color=base_color, alpha=0.4)
        
        axes[idx].set_xlabel('Context Length (tokens)', fontsize=11)
        axes[idx].set_ylabel('Latency (ms)', fontsize=11) if idx == 0 else None
        axes[idx].set_title(f'{style["label"]}\nLatency Composition', fontsize=12, fontweight='bold')
        axes[idx].set_xticks(x)
        axes[idx].set_xticklabels(context_lengths, rotation=45)
        axes[idx].legend(fontsize=9, loc='upper left')
        axes[idx].grid(True, alpha=0.3, axis='y')
    
    plt.suptitle('Latency Composition: Prefill (TTFT) vs Decode (TBT)', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'latency_composition.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info("Generated latency composition chart")


def generate_latency_bar_chart(df: pd.DataFrame, output_dir: str, target_ctx: int = 512) -> None:
    """
    Generate bar chart comparing TTFT, TBT, and E2E latency for all models at a specific context length.
    """
    
    # Filter to target context length
    ctx_df = df[df['context_length'] == target_ctx].copy()
    
    if ctx_df.empty:
        logger.warning(f"No data found for context length {target_ctx}")
        return
    
    # Aggregate by model
    agg_data = ctx_df.groupby(['model', 'arch']).agg({
        'ttft_ms': 'mean',
        'tbt_ms': 'mean',
        'e2e_latency_ms': 'mean'
    }).reset_index()
    
    # Sort by e2e latency for better visualization
    agg_data = agg_data.sort_values('e2e_latency_ms', ascending=True)
    
    # Create figure with three subplots
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 7))
    
    # Get colors based on architecture
    colors = [get_style(arch)['color'] for arch in agg_data['arch']]
    
    # Plot 1: TTFT Bar Chart
    ax1.barh(agg_data['model'], agg_data['ttft_ms'], color=colors, alpha=0.8)
    ax1.set_xlabel('TTFT (ms)', fontsize=12)
    ax1.set_title(f'Time-To-First-Token @ {target_ctx} tokens', fontsize=13, fontweight='bold')
    ax1.grid(axis='x', alpha=0.3, linestyle='--')
    
    # Add value labels
    for i, v in enumerate(agg_data['ttft_ms']):
        ax1.text(v + 0.5, i, f'{v:.1f}', va='center', fontsize=8)
    
    # Plot 2: TBT Bar Chart
    # Filter out models with no TBT (encoders)
    tbt_data = agg_data[agg_data['tbt_ms'] > 0]
    if not tbt_data.empty:
        # Re-sort by tbt_ms for this subplot
        tbt_colors = [get_style(arch)['color'] for arch in tbt_data['arch']]
        ax2.barh(tbt_data['model'], tbt_data['tbt_ms'], color=tbt_colors, alpha=0.8)
        ax2.set_xlabel('TBT (ms)', fontsize=12)
        ax2.set_title(f'Time-Between-Tokens @ {target_ctx} tokens\n(Decoder/EncDec only)', fontsize=13, fontweight='bold')
        ax2.grid(axis='x', alpha=0.3, linestyle='--')
        
        # Add value labels
        for i, v in enumerate(tbt_data['tbt_ms']):
            ax2.text(v + 0.02, i, f'{v:.2f}', va='center', fontsize=8)
    else:
        ax2.text(0.5, 0.5, 'No TBT data\n(Encoder models only)', 
                ha='center', va='center', transform=ax2.transAxes, fontsize=12)
        ax2.set_title(f'Time-Between-Tokens @ {target_ctx} tokens', fontsize=13, fontweight='bold')
    
    # Plot 3: E2E Latency Bar Chart
    e2e_data = agg_data[agg_data['e2e_latency_ms'] > 0]
    if not e2e_data.empty:
        e2e_colors = [get_style(arch)['color'] for arch in e2e_data['arch']]
        ax3.barh(e2e_data['model'], e2e_data['e2e_latency_ms'], color=e2e_colors, alpha=0.8)
        ax3.set_xlabel('E2E Latency (ms)', fontsize=12)
        ax3.set_title(f'End-to-End Latency @ {target_ctx} tokens', fontsize=13, fontweight='bold')
        ax3.grid(axis='x', alpha=0.3, linestyle='--')
        
        # Add value labels
        for i, v in enumerate(e2e_data['e2e_latency_ms']):
            ax3.text(v + 5, i, f'{v:.1f}', va='center', fontsize=8)
    else:
        ax3.text(0.5, 0.5, 'No E2E data available', 
                ha='center', va='center', transform=ax3.transAxes, fontsize=12)
        ax3.set_title(f'End-to-End Latency @ {target_ctx} tokens', fontsize=13, fontweight='bold')
    
    # Add legend - only show architectures that are actually present in the data
    from matplotlib.lines import Line2D
    present_archs = agg_data['arch'].unique()
    legend_elements = [
        Line2D([0], [0], color=get_style(arch)['color'], lw=4, label=get_style(arch)['label'])
        for arch in ['encoder', 'decoder', 'encdec'] if arch in present_archs
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=len(legend_elements), 
              bbox_to_anchor=(0.5, -0.02), fontsize=11, frameon=False)
    
    plt.tight_layout(rect=[0, 0.02, 1, 1])
    plt.savefig(os.path.join(output_dir, f'latency_bar_chart_{target_ctx}.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Generated latency bar chart for context length {target_ctx}")


def generate_pretraining_plots(df: pd.DataFrame, output_dir: str) -> None:
    """Generate continued pretraining plots."""
    
    # 1. Training efficiency (tokens/sec vs epochs)
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    
    for idx, arch in enumerate(df['arch'].unique()):
        arch_df = df[df['arch'] == arch]
        style = get_style(arch)
        plt.scatter(arch_df['epochs_trained'], arch_df['tokens_per_second'], 
                   label=style['label'], alpha=0.7, s=120, 
                   marker=style['marker'], color=style['color'])
    
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
    
    for idx, arch in enumerate(df['arch'].unique()):
        arch_df = df[df['arch'] == arch]
        if 'kwh' in arch_df.columns:
            # Filter out None/NaN values
            valid_df = arch_df[arch_df['kwh'].notna()]
            if len(valid_df) > 0:
                style = get_style(arch)
                ax.scatter(valid_df['kwh'], valid_df['tokens_per_second'], 
                          label=style['label'], alpha=0.7, s=120, 
                          marker=style['marker'], color=style['color'])
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


def prepare_tradeoff_data(tables_dir: str) -> pd.DataFrame:
    """
    Merge inference and few-shot results to create a unified dataset.
    Returns a DataFrame with columns: [model, arch, latency_ms, energy_joules, accuracy]
    """
    inf_path = os.path.join(tables_dir, "inference_results.csv")
    few_path = os.path.join(tables_dir, "fewshot_results.csv")
    
    try:
        inf_df = pd.read_csv(inf_path)
        few_df = pd.read_csv(few_path)
    except Exception as e:
        logger.warning(f"Could not read CSVs for tradeoff plot: {e}")
        return pd.DataFrame()

    # 1. Process Inference Data (Rigorous alignment)
    # We prioritize a standard context length of 512 for fair comparison.
    target_ctx = 512
    if 'context_length' in inf_df.columns:
        if (inf_df['context_length'] == target_ctx).any():
            inf_df = inf_df[inf_df['context_length'] == target_ctx]
            logger.info(f"Tradeoff Plot: Aligned inference data to context length {target_ctx}")
        else:
            # If 512 is missing, warn the user and use mean (suboptimal)
            logger.warning(f"Tradeoff Plot: Target context length {target_ctx} not found. Using MEAN across all lengths (CAUTION: May be unfair).")
    
    # Aggregate inference metrics by model
    # We take the mean in case there are multiple runs for the same model
    inf_agg = inf_df.groupby(['model', 'arch']).agg({
        'latency_ms': 'mean',
        'inference_energy_per_sample_joules': 'mean',
        'throughput_tokens_per_sec': 'mean'
    }).reset_index()

    # 2. Process Few-shot Data
    # Calculate average accuracy across tasks.
    # Note: We aggregate per model first.
    # We should NOT blindly average all tasks if the distribution is skewed,
    # but for this benchmark we assume a standard set of tasks (MMLU, SuperGLUE).
    # We filter to ensure we are looking at the overall 'mmlu' score if available, 
    # instead of subtasks to avoid skewing.
    
    # Check if we have 'mmlu' as a main task vs subtasks
    # If 'mmlu' exists, we might want to exclude 'mmlu_...' subtasks from the average
    # to avoid double counting or skewing, depending on how data is logged.
    # For now, we take a simple mean but logging a warning if many subtasks exist.
    
    # Ideally, we focus on MMLU score for the trade-off plot as requested.
    # If 'mmlu' task is present, use that. Otherwise use mean of all.
    
    # Filter for MMLU only if available for tradeoff
    mmlu_data = few_df[few_df['task'] == 'mmlu']
    if not mmlu_data.empty:
        # If we have explicit 'mmlu' task (aggregate), use it
        few_agg = mmlu_data.groupby(['model', 'arch'])['accuracy'].mean().reset_index()
        logger.info("Tradeoff Plot: Using 'mmlu' task accuracy.")
    else:
        # Fallback to average of all tasks
        few_agg = few_df.groupby(['model', 'arch'])['accuracy'].mean().reset_index()
        logger.info("Tradeoff Plot: Using average accuracy of all tasks (MMLU aggregate not found).")

    few_agg.rename(columns={'accuracy': 'avg_accuracy'}, inplace=True)

    # 3. Merge on Model and Architecture
    merged = pd.merge(inf_agg, few_agg, on=['model', 'arch'], how='inner')
    
    return merged


def generate_tradeoff_plots(tables_dir: str, output_dir: str) -> None:
    """Generate unified trade-off plots (Efficiency vs Effectiveness vs Energy)."""
    
    df = prepare_tradeoff_data(tables_dir)
    
    if df.empty:
        logger.warning("No overlapping models found between inference and fewshot results. Skipping tradeoff plots.")
        return

    # --- Plot: The "E3" Bubble Chart ---
    # X: Latency (Efficiency) - Lower is better
    # Y: Accuracy (Effectiveness) - Higher is better
    # Size: Energy (Joules) - Smaller is better (represented by bubble size)
    # Color/Shape: Architecture
    
    plt.figure(figsize=(14, 10))
    
    # Normalize energy for bubble sizing
    # Min size 100, Max size 1000
    if 'inference_energy_per_sample_joules' in df.columns:
        energy = df['inference_energy_per_sample_joules'].fillna(0)
        # Use log scale for size if range is large, otherwise linear
        # Add small epsilon to avoid log(0)
        epsilon = 1e-6
        log_energy = np.log1p(energy + epsilon)
        
        if log_energy.max() > log_energy.min():
            sizes = 100 + (log_energy - log_energy.min()) / (log_energy.max() - log_energy.min()) * 900
        else:
            sizes = 300
    else:
        sizes = 300

    for arch in df['arch'].unique():
        subset = df[df['arch'] == arch]
        if subset.empty:
            continue
            
        style = get_style(arch)
        # Match indices for sizes
        subset_sizes = sizes[subset.index]
        
        plt.scatter(
            subset['latency_ms'], 
            subset['avg_accuracy'],
            s=subset_sizes,
            c=style['color'],
            marker=style['marker'],
            label=style['label'],
            alpha=0.6,
            edgecolors='w',
            linewidth=1.5
        )
        
        # Add labels to points
        for _, row in subset.iterrows():
            plt.text(
                row['latency_ms'], 
                row['avg_accuracy'], 
                f"  {row['model']}", 
                fontsize=9, 
                alpha=0.8,
                va='center'
            )

    # Calculate Pareto Frontier (Efficiency vs Effectiveness)
    # We want High Accuracy (Max) and Low Latency (Min)
    # Sort by latency ascending
    sorted_df = df.sort_values('latency_ms')
    pareto_points = []
    max_acc_so_far = -1.0
    
    for _, row in sorted_df.iterrows():
        if row['avg_accuracy'] > max_acc_so_far:
            pareto_points.append((row['latency_ms'], row['avg_accuracy']))
            max_acc_so_far = row['avg_accuracy']
            
    if pareto_points:
        px, py = zip(*pareto_points)
        plt.plot(px, py, 'k--', alpha=0.4, linewidth=1.5, linestyle='--', label='Pareto Frontier')

    plt.xlabel('Inference Latency (ms) [Efficiency] \n(Lower is Better)', fontsize=12, fontweight='bold')
    plt.ylabel('Average Accuracy [Effectiveness] \n(Higher is Better)', fontsize=12, fontweight='bold')
    plt.title('The E³ Trade-off: Efficiency vs Effectiveness vs Energy\n(Bubble Size represents Energy Consumption)', fontsize=16, fontweight='bold', pad=20)
    
    # Create two separate legends to avoid overlap
    # Legend 1: Architecture types (top left)
    from matplotlib.lines import Line2D
    arch_legend_elements = []
    for arch in df['arch'].unique():
        style = get_style(arch)
        arch_legend_elements.append(
            Line2D([0], [0], marker=style['marker'], color='w', 
                   markerfacecolor=style['color'], markersize=10, 
                   label=style['label'], linestyle='None', markeredgewidth=0)
        )
    
    # Add Pareto Frontier to arch legend
    if pareto_points:
        arch_legend_elements.append(
            Line2D([0], [0], color='k', linestyle='--', linewidth=1.5, label='Pareto Frontier')
        )
    
    legend1 = plt.legend(handles=arch_legend_elements, loc='upper left', 
                        title='Architecture', fontsize=10, framealpha=0.9)
    plt.gca().add_artist(legend1)  # Keep first legend when adding second
    
    # Legend 2: Energy sizes (outside plot area on the right)
    if 'inference_energy_per_sample_joules' in df.columns:
        min_e = df['inference_energy_per_sample_joules'].min()
        max_e = df['inference_energy_per_sample_joules'].max()
        mid_e = (min_e + max_e) / 2
        
        energy_legend_elements = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', 
                   markersize=6, label=f'{min_e:.1f} J', linestyle='None', alpha=0.5),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', 
                   markersize=12, label=f'{mid_e:.1f} J', linestyle='None', alpha=0.5),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', 
                   markersize=18, label=f'{max_e:.1f} J', linestyle='None', alpha=0.5)
        ]
        
        plt.legend(handles=energy_legend_elements, loc='center left', 
                  bbox_to_anchor=(0.01, 0.5),
                  title='Energy per Sample', fontsize=9, framealpha=0.9)
    
    plt.grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'tradeoff_bubble_chart.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info("Generated unified trade-off bubble chart")


def generate_e3_tradeoff_analysis(tables_dir: str, output_dir: str) -> None:
    """Generate E³ trade-off analysis plots for the report.
    
    Creates 5 core visualizations:
    1. 3D Training Trade-off Space
    2. 3D Inference Trade-off Space
    3. Efficiency-Energy Decoupling
    4. Pareto Frontiers (multi-objective)
    5. Lifecycle Energy Breakdown
    """
    from mpl_toolkits.mplot3d import Axes3D
    
    logger.info("Generating E³ trade-off analysis plots...")
    
    # Load data
    training_path = os.path.join(tables_dir, "training_results.csv")
    inference_path = os.path.join(tables_dir, "inference_results.csv")
    
    if not os.path.exists(training_path):
        logger.warning(f"Training results not found: {training_path}")
        return
    
    if not os.path.exists(inference_path):
        logger.warning(f"Inference results not found: {inference_path}")
        return
    
    train_df = pd.read_csv(training_path)
    infer_df = pd.read_csv(inference_path)
    
    # Aggregate training by model to get average performance
    train_agg = train_df.groupby(['model', 'arch']).agg({
        'duration_seconds': 'mean',
        'kwh': 'mean',
        'accuracy': 'mean',
        'avg_watt': 'mean'
    }).reset_index()
    
    # Aggregate inference by model (average across context lengths)
    infer_agg = infer_df.groupby(['model', 'arch']).agg({
        'ttft_ms': 'mean',
        'tbt_ms': 'mean',
        'e2e_latency_ms': 'mean',
        'inference_energy_per_sample_joules': 'mean',
        'throughput_tokens_per_sec': 'mean'
    }).reset_index()
    
    # Join inference with training accuracy (as proxy for effectiveness)
    model_acc = train_df.groupby('model')['accuracy'].mean().reset_index()
    infer_agg = infer_agg.merge(model_acc, on='model', how='left')
    
    # ========== PLOT 1: 3D Training Trade-off Space ==========
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    for arch in ['encoder', 'decoder', 'encdec']:
        data = train_agg[train_agg['arch'] == arch]
        if len(data) == 0:
            continue
        
        style = get_style(arch)
        ax.scatter(
            data['duration_seconds'],      # X: Efficiency (time)
            data['kwh'] * 1000,           # Y: Energy (convert to Wh for readability)
            data['accuracy'] * 100,        # Z: Effectiveness (percentage)
            label=style['label'],
            color=style['color'],
            marker=style['marker'],
            s=150, alpha=0.7, edgecolors='black', linewidth=0.5
        )
        
        # Add model labels
        for _, row in data.iterrows():
            ax.text(row['duration_seconds'], row['kwh']*1000, row['accuracy']*100,
                   f"  {row['model']}", fontsize=7, alpha=0.7)
    
    ax.set_xlabel('Training Time (seconds)\n← Better (faster)', fontsize=11, labelpad=10)
    ax.set_ylabel('Energy (Wh)\n← Better (less energy)', fontsize=11, labelpad=10)
    ax.set_zlabel('Accuracy (%)\n→ Better (higher)', fontsize=11, labelpad=10)
    ax.set_title('E³ Training Trade-off Space\n(Efficiency × Energy × Effectiveness)', 
                 fontsize=14, fontweight='bold', pad=20)
    ax.legend(fontsize=10, loc='upper left')
    ax.view_init(elev=20, azim=45)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'e3_training_3d_scatter.png'), dpi=300, bbox_inches='tight')
    plt.close()
    logger.info("✓ Generated e3_training_3d_scatter.png")
    
    # ========== PLOT 2: 3D Inference Trade-off Space ==========
    # Filter out models without inference data or accuracy
    infer_valid = infer_agg.dropna(subset=['accuracy', 'inference_energy_per_sample_joules'])
    
    if len(infer_valid) > 0:
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        for arch in ['encoder', 'decoder', 'encdec']:
            data = infer_valid[infer_valid['arch'] == arch]
            if len(data) == 0:
                continue
            
            style = get_style(arch)
            ax.scatter(
                data['ttft_ms'],                              # X: Efficiency (latency)
                data['inference_energy_per_sample_joules'],  # Y: Energy per sample
                data['accuracy'] * 100,                       # Z: Effectiveness
                label=style['label'],
                color=style['color'],
                marker=style['marker'],
                s=150, alpha=0.7, edgecolors='black', linewidth=0.5
            )
            
            # Add model labels
            for _, row in data.iterrows():
                ax.text(row['ttft_ms'], row['inference_energy_per_sample_joules'], 
                       row['accuracy']*100, f"  {row['model']}", fontsize=7, alpha=0.7)
        
        ax.set_xlabel('TTFT (ms)\n← Better (lower latency)', fontsize=11, labelpad=10)
        ax.set_ylabel('Energy per Sample (J)\n← Better (less energy)', fontsize=11, labelpad=10)
        ax.set_zlabel('Accuracy (%)\n→ Better (higher)', fontsize=11, labelpad=10)
        ax.set_title('E³ Inference Trade-off Space\n(Efficiency × Energy × Effectiveness)', 
                     fontsize=14, fontweight='bold', pad=20)
        ax.legend(fontsize=10, loc='upper left')
        ax.view_init(elev=20, azim=45)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'e3_inference_3d_scatter.png'), dpi=300, bbox_inches='tight')
        plt.close()
        logger.info("✓ Generated e3_inference_3d_scatter.png")
    else:
        logger.warning("Skipping e3_inference_3d_scatter.png (no valid data)")
    
    # ========== PLOT 3: Efficiency-Energy Decoupling ==========
    # Show that faster doesn't mean more energy-efficient
    infer_gen = infer_valid[infer_valid['arch'].isin(['decoder', 'encdec'])]  # encoder doesn't generate
    
    if len(infer_gen) > 0:
        fig, ax = plt.subplots(figsize=(10, 7))
        
        for arch in ['decoder', 'encdec']:
            data = infer_gen[infer_gen['arch'] == arch]
            if len(data) == 0:
                continue
            
            style = get_style(arch)
            # Energy per token (assuming 50 tokens generated)
            energy_per_token = data['inference_energy_per_sample_joules'] / 50
            
            ax.scatter(
                data['throughput_tokens_per_sec'],
                energy_per_token,
                label=style['label'],
                color=style['color'],
                marker=style['marker'],
                s=200, alpha=0.7, edgecolors='black', linewidth=1.5
            )
            
            # Add model labels
            for _, row in data.iterrows():
                energy_tok = row['inference_energy_per_sample_joules'] / 50
                ax.annotate(
                    row['model'], 
                    (row['throughput_tokens_per_sec'], energy_tok),
                    xytext=(5, 5), textcoords='offset points',
                    fontsize=9, alpha=0.8,
                    bbox=dict(boxstyle='round,pad=0.3', facecolor=style['color'], alpha=0.2)
                )
        
        ax.set_xlabel('Throughput (tokens/sec)\n→ Higher is Better (Faster)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Energy per Token (J)\n← Lower is Better (Greener)', fontsize=12, fontweight='bold')
        ax.set_title('Efficiency-Energy Decoupling\nFaster ≠ More Energy-Efficient', 
                     fontsize=14, fontweight='bold')
        
        # Add quadrant labels
        x_mid = ax.get_xlim()[0] + (ax.get_xlim()[1] - ax.get_xlim()[0]) * 0.5
        y_mid = ax.get_ylim()[0] + (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.5
        ax.axhline(y_mid, color='gray', linestyle='--', alpha=0.3, linewidth=1)
        ax.axvline(x_mid, color='gray', linestyle='--', alpha=0.3, linewidth=1)
        
        ax.text(ax.get_xlim()[1]*0.95, ax.get_ylim()[0]*1.05, 'Fast & Green\n(Ideal)', 
                ha='right', va='bottom', fontsize=9, style='italic', alpha=0.5,
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.3))
        ax.text(ax.get_xlim()[0]*1.05, ax.get_ylim()[1]*0.95, 'Slow & Wasteful\n(Worst)', 
                ha='left', va='top', fontsize=9, style='italic', alpha=0.5,
                bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.3))
        
        ax.legend(fontsize=11, loc='upper right')
        ax.grid(True, alpha=0.3, linestyle='--')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'efficiency_energy_decoupling.png'), dpi=300, bbox_inches='tight')
        plt.close()
        logger.info("✓ Generated efficiency_energy_decoupling.png")
    
    # ========== PLOT 4: Pareto Frontiers (2x2 subplots) ==========
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # Training: Efficiency vs Effectiveness
    ax = axes[0, 0]
    for arch in ['encoder', 'decoder', 'encdec']:
        data = train_agg[train_agg['arch'] == arch]
        if len(data) == 0:
            continue
        style = get_style(arch)
        ax.scatter(data['duration_seconds'], data['accuracy']*100, 
                  label=style['label'],
                  color=style['color'], marker=style['marker'],
                  s=100, alpha=0.7, edgecolors='black', linewidth=0.5)
    ax.set_xlabel('Training Time (seconds) →', fontsize=11)
    ax.set_ylabel('Accuracy (%) →', fontsize=11)
    ax.set_title('Training: Efficiency vs Effectiveness\n(Lower-Left is Better)', fontsize=11, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Training: Energy vs Effectiveness
    ax = axes[0, 1]
    for arch in ['encoder', 'decoder', 'encdec']:
        data = train_agg[train_agg['arch'] == arch]
        if len(data) == 0:
            continue
        style = get_style(arch)
        ax.scatter(data['kwh']*1000, data['accuracy']*100,
                  label=style['label'],
                  color=style['color'], marker=style['marker'],
                  s=100, alpha=0.7, edgecolors='black', linewidth=0.5)
    ax.set_xlabel('Energy (Wh) →', fontsize=11)
    ax.set_ylabel('Accuracy (%) →', fontsize=11)
    ax.set_title('Training: Energy vs Effectiveness\n(Lower-Left is Better)', fontsize=11, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Inference: Latency vs Energy
    ax = axes[1, 0]
    for arch in ['decoder', 'encdec']:
        data = infer_valid[infer_valid['arch'] == arch]
        if len(data) == 0:
            continue
        style = get_style(arch)
        ax.scatter(data['ttft_ms'], data['inference_energy_per_sample_joules'],
                  label=style['label'],
                  color=style['color'], marker=style['marker'],
                  s=100, alpha=0.7, edgecolors='black', linewidth=0.5)
    ax.set_xlabel('TTFT (ms) →', fontsize=11)
    ax.set_ylabel('Energy per Sample (J) →', fontsize=11)
    ax.set_title('Inference: Latency vs Energy\n(Lower-Left is Better)', fontsize=11, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Inference: Latency vs Effectiveness
    ax = axes[1, 1]
    for arch in ['decoder', 'encdec']:
        data = infer_valid[infer_valid['arch'] == arch]
        if len(data) == 0:
            continue
        style = get_style(arch)
        ax.scatter(data['ttft_ms'], data['accuracy']*100,
                  label=style['label'],
                  color=style['color'], marker=style['marker'],
                  s=100, alpha=0.7, edgecolors='black', linewidth=0.5)
    ax.set_xlabel('TTFT (ms) →', fontsize=11)
    ax.set_ylabel('Accuracy (%) →', fontsize=11)
    ax.set_title('Inference: Latency vs Effectiveness\n(Lower-Left is Better)', fontsize=11, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    plt.suptitle('Pareto Frontiers: Multi-objective Trade-offs\n' + 
                 'Ideal points are in lower-left (low cost, high benefit)', 
                 fontsize=15, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'pareto_frontier_combined.png'), dpi=300, bbox_inches='tight')
    plt.close()
    logger.info("✓ Generated pareto_frontier_combined.png")
    
    # ========== PLOT 5: Lifecycle Energy Breakdown ==========
    # Simulate 1M inferences to show training energy becomes negligible
    n_inferences = 1_000_000
    
    lifecycle_data = []
    for _, train_row in train_agg.iterrows():
        model = train_row['model']
        arch = train_row['arch']
        train_energy_kwh = train_row['kwh']
        
        # Get inference energy
        infer_row = infer_agg[infer_agg['model'] == model]
        if len(infer_row) > 0 and not pd.isna(infer_row['inference_energy_per_sample_joules'].values[0]):
            infer_energy_per_sample = infer_row['inference_energy_per_sample_joules'].values[0]
            total_infer_energy_kwh = (infer_energy_per_sample * n_inferences) / 3_600_000  # J to kWh
            
            lifecycle_data.append({
                'model': model,
                'arch': arch,
                'Training': train_energy_kwh,
                'Inference': total_infer_energy_kwh,
                'Total': train_energy_kwh + total_infer_energy_kwh
            })
    
    if len(lifecycle_data) > 0:
        lifecycle_df = pd.DataFrame(lifecycle_data)
        lifecycle_df = lifecycle_df.sort_values('Total', ascending=True)
        
        fig, ax = plt.subplots(figsize=(12, 7))
        
        # Create stacked bar chart
        x_pos = np.arange(len(lifecycle_df))
        
        # Training bars
        bars1 = ax.bar(x_pos, lifecycle_df['Training'], 
                      label='Training (one-time)', color='#3498db', alpha=0.8)
        
        # Inference bars (stacked on top)
        bars2 = ax.bar(x_pos, lifecycle_df['Inference'], 
                      bottom=lifecycle_df['Training'],
                      label=f'Inference ({n_inferences:,} samples)', color='#e74c3c', alpha=0.8)
        
        # Add percentage labels
        for i, (_, row) in enumerate(lifecycle_df.iterrows()):
            total = row['Total']
            train_pct = (row['Training'] / total) * 100
            infer_pct = (row['Inference'] / total) * 100
            
            # Label on training section (only if visible enough)
            if train_pct > 5:
                ax.text(i, row['Training']/2, f'{train_pct:.1f}%', 
                       ha='center', va='center', fontsize=8, fontweight='bold', color='white')
            
            # Label on inference section
            ax.text(i, row['Training'] + row['Inference']/2, f'{infer_pct:.1f}%', 
                   ha='center', va='center', fontsize=8, fontweight='bold', color='white')
            
            # Total energy on top
            ax.text(i, total, f'{total:.1f} kWh', 
                   ha='center', va='bottom', fontsize=8, fontweight='bold')
        
        ax.set_ylabel('Energy (kWh)', fontsize=12, fontweight='bold')
        ax.set_xlabel('Model', fontsize=12, fontweight='bold')
        ax.set_title(f'Lifecycle Energy Breakdown\n(Training + {n_inferences:,} Inferences)\n' +
                     'Inference energy dominates at scale!', 
                     fontsize=14, fontweight='bold')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(lifecycle_df['model'], rotation=45, ha='right')
        ax.legend(fontsize=11, loc='upper left')
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        
        # Add architecture color coding on x-axis labels
        for i, (_, row) in enumerate(lifecycle_df.iterrows()):
            style = get_style(row['arch'])
            ax.get_xticklabels()[i].set_color(style['color'])
            ax.get_xticklabels()[i].set_fontweight('bold')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'lifecycle_energy_breakdown.png'), dpi=300, bbox_inches='tight')
        plt.close()
        logger.info("✓ Generated lifecycle_energy_breakdown.png")
    
    logger.info("✅ Completed E³ trade-off analysis (5 plots generated)")


def generate_training_bubble_chart(tables_dir: str, output_dir: str) -> None:
    """Generate training E³ bubble chart (similar to inference bubble chart).
    
    X-axis: Training Time (Efficiency) - Lower is better
    Y-axis: Accuracy (Effectiveness) - Higher is better
    Bubble Size: Energy (kWh) - Smaller is better
    Color/Shape: Architecture
    """
    
    logger.info("Generating training E³ bubble chart...")
    
    training_path = os.path.join(tables_dir, "training_results.csv")
    
    if not os.path.exists(training_path):
        logger.warning(f"Training results not found: {training_path}")
        return
    
    train_df = pd.read_csv(training_path)
    
    # Aggregate training by model
    train_agg = train_df.groupby(['model', 'arch']).agg({
        'duration_seconds': 'mean',
        'kwh': 'mean',
        'accuracy': 'mean',
        'avg_watt': 'mean'
    }).reset_index()
    
    # Create bubble chart
    plt.figure(figsize=(14, 10))
    
    # Normalize energy for bubble sizing (kwh -> Wh)
    if 'kwh' in train_agg.columns:
        energy_wh = train_agg['kwh'] * 1000  # Convert to Wh
        epsilon = 1e-6
        log_energy = np.log1p(energy_wh + epsilon)
        
        if log_energy.max() > log_energy.min():
            sizes = 100 + (log_energy - log_energy.min()) / (log_energy.max() - log_energy.min()) * 900
        else:
            sizes = pd.Series([300] * len(train_agg), index=train_agg.index)
    else:
        sizes = pd.Series([300] * len(train_agg), index=train_agg.index)
    
    # Plot by architecture
    for arch in ['encoder', 'decoder', 'encdec']:
        data = train_agg[train_agg['arch'] == arch]
        if len(data) == 0:
            continue
        
        style = get_style(arch)
        subset_sizes = sizes[data.index]
        
        plt.scatter(
            data['duration_seconds'],
            data['accuracy'] * 100,  # Convert to percentage
            s=subset_sizes,
            c=style['color'],
            marker=style['marker'],
            label=style['label'],
            alpha=0.6,
            edgecolors='w',
            linewidth=1.5
        )
        
        # Add model labels
        for _, row in data.iterrows():
            plt.text(
                row['duration_seconds'], 
                row['accuracy'] * 100, 
                f"  {row['model']}", 
                fontsize=9, 
                alpha=0.8,
                va='center'
            )
    
    # Calculate Pareto Frontier (Lower time, Higher accuracy)
    sorted_train = train_agg.sort_values('duration_seconds')
    pareto_points = []
    max_acc = -1.0
    
    for _, row in sorted_train.iterrows():
        if row['accuracy'] > max_acc:
            pareto_points.append((row['duration_seconds'], row['accuracy'] * 100))
            max_acc = row['accuracy']
    
    if pareto_points:
        px, py = zip(*pareto_points)
        plt.plot(px, py, 'k--', alpha=0.4, linewidth=1.5, label='Pareto Frontier')
    
    plt.xlabel('Training Time (seconds) [Efficiency]\n(Lower is Better)', 
               fontsize=12, fontweight='bold')
    plt.ylabel('Average Accuracy (%) [Effectiveness]\n(Higher is Better)', 
               fontsize=12, fontweight='bold')
    plt.title('Training E³ Trade-off: Efficiency vs Effectiveness vs Energy\n' +
              '(Bubble Size = Energy Consumption)', 
              fontsize=16, fontweight='bold', pad=20)
    
    # Create legends
    from matplotlib.lines import Line2D
    arch_legend_elements = []
    for arch in train_agg['arch'].unique():
        style = get_style(arch)
        arch_legend_elements.append(
            Line2D([0], [0], marker=style['marker'], color='w', 
                   markerfacecolor=style['color'], markersize=10, 
                   label=style['label'], linestyle='None', markeredgewidth=0)
        )
    
    if pareto_points:
        arch_legend_elements.append(
            Line2D([0], [0], color='k', linestyle='--', linewidth=1.5, 
                   label='Pareto Frontier')
        )
    
    legend1 = plt.legend(handles=arch_legend_elements, loc='upper right', 
                        title='Architecture', fontsize=10, framealpha=0.9)
    plt.gca().add_artist(legend1)
    
    # Energy size legend
    if 'kwh' in train_agg.columns:
        energy_wh = train_agg['kwh'] * 1000
        min_e = energy_wh.min()
        max_e = energy_wh.max()
        mid_e = (min_e + max_e) / 2
        
        energy_legend_elements = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', 
                   markersize=6, label=f'{min_e:.1f} Wh', linestyle='None', alpha=0.5),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', 
                   markersize=12, label=f'{mid_e:.1f} Wh', linestyle='None', alpha=0.5),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', 
                   markersize=18, label=f'{max_e:.1f} Wh', linestyle='None', alpha=0.5)
        ]
        
        plt.legend(handles=energy_legend_elements, loc='center left', 
                  bbox_to_anchor=(0.01, 0.5),
                  title='Energy Consumption', fontsize=9, framealpha=0.9)
    
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'e3_training_bubble_chart.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    logger.info("✓ Generated e3_training_bubble_chart.png")


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(description="Generate plots")
    parser.add_argument("--tables", default="tables", help="Tables directory")
    parser.add_argument("--out", default="figs", help="Output directory")
    
    args = parser.parse_args()
    
    generate_plots(args.tables, args.out)


if __name__ == "__main__":
    main()
