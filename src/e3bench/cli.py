"""Command-line interface for E³ Mini-Benchmark."""

import argparse
import sys
import logging
from typing import List

from .train.finetune_glue import finetune_superglue
from .train.cont_pretrain import continued_pretraining
from .eval.eval_fewshot import evaluate_fewshot
from .eval.bench_infer import benchmark_inference
from .report.aggregate import aggregate_results
from .report.signif_test import run_significance_tests
from .report.plots import generate_plots

logger = logging.getLogger(__name__)


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="E³ Mini-Benchmark: Comprehensive evaluation framework for language models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run complete benchmark pipeline
  e3bench all
  
  # Fine-tune on SuperGLUE
  e3bench train --model configs/model/bert-base.yaml
  
  # Few-shot evaluation
  e3bench eval --model configs/model/gpt2-medium.yaml
  
  # Inference benchmarking
  e3bench bench --model configs/model/t5-base.yaml
  
  # Generate reports and plots
  e3bench report
  e3bench figs
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Fine-tune on SuperGLUE')
    train_parser.add_argument('--model', default='configs/model/bert-base.yaml', help='Model config')
    train_parser.add_argument('--task', default='configs/task/superglue.yaml', help='Task config')
    train_parser.add_argument('--train', default='configs/train/lora.yaml', help='Training config')
    train_parser.add_argument('--output', default='results', help='Output directory')
    
    # Eval command
    eval_parser = subparsers.add_parser('eval', help='Few-shot evaluation')
    eval_parser.add_argument('--model', default='configs/model/gpt2-medium.yaml', help='Model config')
    eval_parser.add_argument('--eval', default='configs/eval/fewshot_5.yaml', help='Evaluation config')
    eval_parser.add_argument('--output', default='results', help='Output directory')
    
    # Bench command
    bench_parser = subparsers.add_parser('bench', help='Inference benchmarking')
    bench_parser.add_argument('--model', default='configs/model/t5-base.yaml', help='Model config')
    bench_parser.add_argument('--bench', default='configs/bench/infer_seq2seq.yaml', help='Benchmark config')
    bench_parser.add_argument('--output', default='results', help='Output directory')
    
    # Report command
    report_parser = subparsers.add_parser('report', help='Aggregate results and run significance tests')
    report_parser.add_argument('--results', default='results', help='Results directory')
    report_parser.add_argument('--tables', default='tables', help='Tables directory')
    
    # Figs command
    figs_parser = subparsers.add_parser('figs', help='Generate plots')
    figs_parser.add_argument('--tables', default='tables', help='Tables directory')
    figs_parser.add_argument('--output', default='figs', help='Output directory')
    
    # All command
    all_parser = subparsers.add_parser('all', help='Run complete benchmark pipeline')
    all_parser.add_argument('--output', default='results', help='Output directory')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    try:
        if args.command == 'train':
            finetune_superglue(args.model, args.task, args.train, args.output)
        elif args.command == 'eval':
            evaluate_fewshot(args.model, args.eval, args.output)
        elif args.command == 'bench':
            benchmark_inference(args.model, args.bench, args.output)
        elif args.command == 'report':
            aggregate_results(args.results, args.tables)
            run_significance_tests(args.tables, args.tables)
        elif args.command == 'figs':
            generate_plots(args.tables, args.output)
        elif args.command == 'all':
            # Run complete pipeline
            print("Running complete E³ Mini-Benchmark pipeline...")
            
            # Training
            print("1. Fine-tuning on SuperGLUE...")
            finetune_superglue(
                'configs/model/bert-base.yaml',
                'configs/task/superglue.yaml', 
                'configs/train/lora.yaml',
                args.output
            )
            
            # Evaluation
            print("2. Few-shot evaluation...")
            evaluate_fewshot(
                'configs/model/gpt2-medium.yaml',
                'configs/eval/fewshot_5.yaml',
                args.output
            )
            
            # Benchmarking
            print("3. Inference benchmarking...")
            benchmark_inference(
                'configs/model/t5-base.yaml',
                'configs/bench/infer_seq2seq.yaml',
                args.output
            )
            
            # Reporting
            print("4. Aggregating results...")
            aggregate_results(args.output, 'tables')
            run_significance_tests('tables', 'tables')
            
            # Plots
            print("5. Generating plots...")
            generate_plots('tables', 'figs')
            
            print("Complete benchmark pipeline finished!")
            
    except Exception as e:
        logger.error(f"Command failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
