#!/usr/bin/env python3
"""
Verification script for E³ Mini-Benchmark setup.
Run this to verify that all components are properly installed and configured.
"""

import sys
import os
import importlib
import yaml
from pathlib import Path

def check_imports():
    """Check if all required packages can be imported."""
    required_packages = [
        'torch', 'transformers', 'datasets', 'accelerate', 'peft', 
        'evaluate', 'numpy', 'pandas', 'scipy', 'matplotlib'
    ]
    
    print("Checking required packages...")
    missing_packages = []
    
    for package in required_packages:
        try:
            importlib.import_module(package)
            print(f"✓ {package}")
        except ImportError:
            print(f"✗ {package} - MISSING")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\nMissing packages: {', '.join(missing_packages)}")
        print("Run: pip install -r requirements.txt")
        return False
    
    return True

def check_configs():
    """Check if all configuration files exist and are valid."""
    print("\nChecking configuration files...")
    
    config_files = [
        "configs/model/bert-base.yaml",
        "configs/model/t5-base.yaml", 
        "configs/model/gpt2-medium.yaml",
        "configs/task/superglue.yaml",
        "configs/task/mmlu.yaml",
        "configs/train/lora.yaml",
        "configs/eval/fewshot_5.yaml",
        "configs/bench/infer_seq2seq.yaml"
    ]
    
    missing_configs = []
    invalid_configs = []
    
    for config_file in config_files:
        if not os.path.exists(config_file):
            print(f"✗ {config_file} - MISSING")
            missing_configs.append(config_file)
        else:
            try:
                with open(config_file, 'r') as f:
                    yaml.safe_load(f)
                print(f"✓ {config_file}")
            except yaml.YAMLError as e:
                print(f"✗ {config_file} - INVALID YAML: {e}")
                invalid_configs.append(config_file)
    
    if missing_configs or invalid_configs:
        return False
    
    return True

def check_scripts():
    """Check if all scripts exist and are executable."""
    print("\nChecking scripts...")
    
    scripts = [
        "scripts/finetune_superglue.sh",
        "scripts/eval_fewshot.sh", 
        "scripts/bench_infer.sh",
        "scripts/cont_pretrain.sh"
    ]
    
    missing_scripts = []
    non_executable = []
    
    for script in scripts:
        if not os.path.exists(script):
            print(f"✗ {script} - MISSING")
            missing_scripts.append(script)
        elif not os.access(script, os.X_OK):
            print(f"✗ {script} - NOT EXECUTABLE")
            non_executable.append(script)
        else:
            print(f"✓ {script}")
    
    if missing_scripts or non_executable:
        return False
    
    return True

def check_package_structure():
    """Check if the package structure is correct."""
    print("\nChecking package structure...")
    
    required_dirs = [
        "src/e3bench",
        "src/e3bench/utils",
        "src/e3bench/data", 
        "src/e3bench/models",
        "src/e3bench/train",
        "src/e3bench/eval",
        "src/e3bench/report"
    ]
    
    missing_dirs = []
    
    for dir_path in required_dirs:
        if not os.path.exists(dir_path):
            print(f"✗ {dir_path} - MISSING")
            missing_dirs.append(dir_path)
        else:
            print(f"✓ {dir_path}")
    
    if missing_dirs:
        return False
    
    return True

def check_cuda():
    """Check CUDA availability."""
    print("\nChecking CUDA availability...")
    
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✓ CUDA available - {torch.cuda.get_device_name(0)}")
            print(f"✓ CUDA version: {torch.version.cuda}")
            return True
        else:
            print("✗ CUDA not available")
            return False
    except ImportError:
        print("✗ PyTorch not installed")
        return False

def main():
    """Run all verification checks."""
    print("E³ Mini-Benchmark Setup Verification")
    print("=" * 40)
    
    checks = [
        ("Package imports", check_imports),
        ("Configuration files", check_configs),
        ("Scripts", check_scripts),
        ("Package structure", check_package_structure),
        ("CUDA availability", check_cuda)
    ]
    
    all_passed = True
    
    for check_name, check_func in checks:
        try:
            if not check_func():
                all_passed = False
        except Exception as e:
            print(f"✗ {check_name} - ERROR: {e}")
            all_passed = False
    
    print("\n" + "=" * 40)
    if all_passed:
        print("✓ All checks passed! E³ Mini-Benchmark is ready to use.")
        print("\nQuick start:")
        print("  make env              # Install dependencies")
        print("  make superglue-finetune # Run SuperGLUE fine-tuning")
        print("  make eval             # Run few-shot evaluation") 
        print("  make infer            # Run inference benchmarking")
        print("  make report           # Aggregate results")
        print("  make figs             # Generate plots")
        print("  make all              # Run complete pipeline")
    else:
        print("✗ Some checks failed. Please fix the issues above.")
        sys.exit(1)

if __name__ == "__main__":
    main()
