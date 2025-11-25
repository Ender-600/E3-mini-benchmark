#!/usr/bin/env python3
"""
Simple script to show train/eval power breakdown from SuperGLUE results.
"""
import json
import sys

def show_power_breakdown(json_file):
    with open(json_file) as f:
        data = json.load(f)
    
    print(f"Experiment: {data.get('exp_id', 'N/A')}")
    print(f"Model: {data.get('model', {}).get('name', 'N/A')}")
    print("=" * 80)
    
    results = data.get('results', {})
    
    for task_name, task_data in results.items():
        if not isinstance(task_data, dict):
            continue
        
        power = task_data.get('power')
        if not power:
            print(f"\n{task_name}: No power data")
            continue
        
        # Check if it's the new format with train/eval split
        if 'train' in power and 'eval' in power:
            train = power['train']
            eval_p = power['eval']
            total = power.get('total', {})
            
            print(f"\n{task_name}:")
            print(f"  Training:")
            print(f"    Duration: {train.get('duration_seconds', 0):.2f}s")
            print(f"    Avg Power: {train.get('avg_watt', 'N/A')} W")
            print(f"    Energy: {train.get('kwh', 'N/A')} kWh")
            
            print(f"  Evaluation:")
            print(f"    Duration: {eval_p.get('duration_seconds', 0):.2f}s")
            print(f"    Avg Power: {eval_p.get('avg_watt', 'N/A')} W")
            print(f"    Energy: {eval_p.get('kwh', 'N/A')} kWh")
            
            print(f"  Total:")
            print(f"    Duration: {total.get('duration_seconds', 0):.2f}s")
            print(f"    Avg Power: {total.get('avg_watt', 'N/A')} W")
            print(f"    Energy: {total.get('kwh', 'N/A')} kWh")
            
            # Calculate train/eval ratio
            train_kwh = train.get('kwh', 0) or 0
            eval_kwh = eval_p.get('kwh', 0) or 0
            if train_kwh > 0 and eval_kwh > 0:
                ratio = train_kwh / eval_kwh
                train_pct = train_kwh / (train_kwh + eval_kwh) * 100
                eval_pct = eval_kwh / (train_kwh + eval_kwh) * 100
                print(f"  Energy Distribution: Train {train_pct:.1f}% / Eval {eval_pct:.1f}%")
        else:
            # Old format (task-level but no train/eval split)
            print(f"\n{task_name}:")
            print(f"  Total Energy: {power.get('kwh', 'N/A')} kWh")
            print(f"  (No train/eval breakdown)")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 show_power_breakdown.py <result.json>")
        sys.exit(1)
    show_power_breakdown(sys.argv[1])

