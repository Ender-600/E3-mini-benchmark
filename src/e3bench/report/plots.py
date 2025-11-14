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


def generate_latency_curve(df: pd.DataFrame, output_dir: str) -> None:
    """Generate latency vs context length curve."""
    
    # Filter to only scaling experiments
    scaling_df = df[df['context_length'].notna()].copy()
    
    if scaling_df.empty:
        logger.warning("No scaling data found for latency curve")
        return
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Define colors and markers for architectures
    arch_styles = {
        'encoder': {'color': '#2ecc71', 'marker': 'o', 'label': 'Encoder (BERT)'},
        'decoder': {'color': '#e74c3c', 'marker': 's', 'label': 'Decoder (GPT-2)'},
        'encdec': {'color': '#3498db', 'marker': '^', 'label': 'Encoder-Decoder (T5)'}
    }
    
    # Plot 1: End-to-End Latency vs Context Length
    for arch in scaling_df['arch'].unique():
        arch_df = scaling_df[scaling_df['arch'] == arch]
        
        # Use E2E latency if available, otherwise fall back to other metrics
        # For BERT (encoder): E2E = forward_pass_latency
        # For GPT-2/T5: E2E = full generation time
        if 'e2e_latency_ms' in arch_df.columns and arch_df['e2e_latency_ms'].sum() > 0:
            latency_col = 'e2e_latency_ms'
            std_col = 'e2e_std_ms'
        else:
            # Fallback to latency_ms (TTFT or forward_pass)
            latency_col = 'latency_ms'
            std_col = 'latency_std_ms'
        
        # Group by context_length and aggregate
        agg_dict = {latency_col: ['mean', 'std']}
        if std_col in arch_df.columns:
            agg_dict[std_col] = 'first'
        
        grouped = arch_df.groupby('context_length').agg(agg_dict).reset_index()
        
        # Flatten column names
        if std_col in arch_df.columns:
            grouped.columns = ['context_length', 'latency_ms', 'latency_calculated_std', 'latency_std_ms']
            grouped['final_std'] = grouped['latency_calculated_std'].fillna(grouped['latency_std_ms']).fillna(0)
        else:
            grouped.columns = ['context_length', 'latency_ms', 'latency_calculated_std']
            grouped['final_std'] = grouped['latency_calculated_std'].fillna(0)
        
        style = arch_styles.get(arch, {'color': 'gray', 'marker': 'o', 'label': arch})
        
        ax1.plot(grouped['context_length'], grouped['latency_ms'], 
                marker=style['marker'], color=style['color'], 
                label=style['label'], linewidth=2, markersize=8)
        
        # Add error bars if available
        if grouped['final_std'].sum() > 0:
            ax1.fill_between(grouped['context_length'], 
                            grouped['latency_ms'] - grouped['final_std'],
                            grouped['latency_ms'] + grouped['final_std'],
                            color=style['color'], alpha=0.2)
    
    ax1.set_xlabel('Context Length (tokens)', fontsize=12)
    ax1.set_ylabel('E2E Latency (ms)', fontsize=12)
    ax1.set_title('End-to-End Inference Latency vs Context Length', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10, loc='best', framealpha=0.9)
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.set_xscale('linear')
    
    # Plot 2: Memory Usage vs Context Length
    for arch in scaling_df['arch'].unique():
        arch_df = scaling_df[scaling_df['arch'] == arch]
        grouped = arch_df.groupby('context_length').agg({
            'max_memory_gb': 'mean'
        }).reset_index()
        
        style = arch_styles.get(arch, {'color': 'gray', 'marker': 'o', 'label': arch})
        
        ax2.plot(grouped['context_length'], grouped['max_memory_gb'], 
                marker=style['marker'], color=style['color'], 
                label=style['label'], linewidth=2, markersize=8)
    
    ax2.set_xlabel('Context Length (tokens)', fontsize=12)
    ax2.set_ylabel('Peak Memory (GB)', fontsize=12)
    ax2.set_title('Memory Usage vs Context Length', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10, loc='best', framealpha=0.9)
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
    
    # Define colors and markers for architectures
    arch_styles = {
        'encoder': {'color': '#2ecc71', 'marker': 'o', 'label': 'Encoder (BERT)'},
        'decoder': {'color': '#e74c3c', 'marker': 's', 'label': 'Decoder (GPT-2)'},
        'encdec': {'color': '#3498db', 'marker': '^', 'label': 'Encoder-Decoder (T5)'}
    }
    
    # Plot 1: TTFT vs Context Length
    ax1 = plt.subplot(1, 3, 1)
    for arch in scaling_df['arch'].unique():
        arch_df = scaling_df[scaling_df['arch'] == arch]
        # Calculate std from the ttft values themselves, with fallback to pre-computed std
        grouped = arch_df.groupby('context_length').agg({
            'ttft_ms': ['mean', 'std'],
            'ttft_std_ms': 'first'  # Keep the pre-computed std as fallback
        }).reset_index()
        grouped.columns = ['context_length', 'ttft_ms', 'ttft_calculated_std', 'ttft_std_ms']
        
        # Use calculated std if available, otherwise use pre-computed std
        grouped['final_std'] = grouped['ttft_calculated_std'].fillna(grouped['ttft_std_ms']).fillna(0)
        
        style = arch_styles.get(arch, {'color': 'gray', 'marker': 'o', 'label': arch})
        
        ax1.plot(grouped['context_length'], grouped['ttft_ms'], 
                marker=style['marker'], color=style['color'], 
                label=style['label'], linewidth=2, markersize=8)
        
        # Add error bars if available
        if grouped['final_std'].sum() > 0:
            ax1.fill_between(grouped['context_length'], 
                            grouped['ttft_ms'] - grouped['final_std'],
                            grouped['ttft_ms'] + grouped['final_std'],
                            color=style['color'], alpha=0.2)
    
    ax1.set_xlabel('Context Length (tokens)', fontsize=12)
    ax1.set_ylabel('TTFT (ms)', fontsize=12)
    ax1.set_title('Time-To-First-Token (TTFT)\nPrefill + First Decode', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=10, loc='best', framealpha=0.9)
    ax1.grid(True, alpha=0.3, linestyle='--')
    
    # Plot 2: TBT vs Context Length
    ax2 = plt.subplot(1, 3, 2)
    for arch in scaling_df['arch'].unique():
        arch_df = scaling_df[scaling_df['arch'] == arch]
        # Only plot if TBT data is available
        if 'tbt_ms' in arch_df.columns and arch_df['tbt_ms'].sum() > 0:
            # Calculate std from the tbt values themselves, with fallback
            grouped = arch_df.groupby('context_length').agg({
                'tbt_ms': ['mean', 'std'],
                'tbt_std_ms': 'first'  # Keep the pre-computed std as fallback
            }).reset_index()
            grouped.columns = ['context_length', 'tbt_ms', 'tbt_calculated_std', 'tbt_std_ms']
            
            # Use calculated std if available, otherwise use pre-computed std
            grouped['final_std'] = grouped['tbt_calculated_std'].fillna(grouped['tbt_std_ms']).fillna(0)
            
            style = arch_styles.get(arch, {'color': 'gray', 'marker': 'o', 'label': arch})
            
            ax2.plot(grouped['context_length'], grouped['tbt_ms'], 
                    marker=style['marker'], color=style['color'], 
                    label=style['label'], linewidth=2, markersize=8)
            
            # Add error bars if available
            if grouped['final_std'].sum() > 0:
                ax2.fill_between(grouped['context_length'], 
                                grouped['tbt_ms'] - grouped['final_std'],
                                grouped['tbt_ms'] + grouped['final_std'],
                                color=style['color'], alpha=0.2)
    
    ax2.set_xlabel('Context Length (tokens)', fontsize=12)
    ax2.set_ylabel('TBT (ms)', fontsize=12)
    ax2.set_title('Time-Between-Tokens (TBT)\nAverage Decode Latency', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=10, loc='best', framealpha=0.9)
    ax2.grid(True, alpha=0.3, linestyle='--')
    
    # Plot 3: E2E Latency vs Context Length
    ax3 = plt.subplot(1, 3, 3)
    for arch in scaling_df['arch'].unique():
        arch_df = scaling_df[scaling_df['arch'] == arch]
        if 'e2e_latency_ms' in arch_df.columns and arch_df['e2e_latency_ms'].sum() > 0:
            # Calculate std from the e2e values themselves, with fallback
            grouped = arch_df.groupby('context_length').agg({
                'e2e_latency_ms': ['mean', 'std'],
                'e2e_std_ms': 'first'  # Keep the pre-computed std as fallback
            }).reset_index()
            grouped.columns = ['context_length', 'e2e_latency_ms', 'e2e_calculated_std', 'e2e_std_ms']
            
            # Use calculated std if available, otherwise use pre-computed std
            grouped['final_std'] = grouped['e2e_calculated_std'].fillna(grouped['e2e_std_ms']).fillna(0)
            
            style = arch_styles.get(arch, {'color': 'gray', 'marker': 'o', 'label': arch})
            
            ax3.plot(grouped['context_length'], grouped['e2e_latency_ms'], 
                    marker=style['marker'], color=style['color'], 
                    label=style['label'], linewidth=2, markersize=8)
            
            # Add error bars if available
            if grouped['final_std'].sum() > 0:
                ax3.fill_between(grouped['context_length'], 
                                grouped['e2e_latency_ms'] - grouped['final_std'],
                                grouped['e2e_latency_ms'] + grouped['final_std'],
                                color=style['color'], alpha=0.2)
    
    ax3.set_xlabel('Context Length (tokens)', fontsize=12)
    ax3.set_ylabel('E2E Latency (ms)', fontsize=12)
    ax3.set_title('End-to-End (E2E) Latency\nTotal Generation Time', fontsize=13, fontweight='bold')
    ax3.legend(fontsize=10, loc='best', framealpha=0.9)
    ax3.grid(True, alpha=0.3, linestyle='--')
    
    plt.suptitle('Latency Breakdown: TTFT vs TBT vs E2E', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'ttft_tbt_curves.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info("Generated TTFT/TBT breakdown plot")
    
    # Generate additional plot: Stacked bar chart for latency composition
    generate_latency_composition_chart(scaling_df, output_dir, arch_styles)


def generate_latency_composition_chart(df: pd.DataFrame, output_dir: str, arch_styles: dict) -> None:
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
        
        style = arch_styles.get(arch, {'color': 'gray', 'label': arch})
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
