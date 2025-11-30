import pandas as pd
import os
import sys

# 模拟 prepare_tradeoff_data 的逻辑
def debug_data():
    tables_dir = "tables"
    inf_path = os.path.join(tables_dir, "inference_results.csv")
    few_path = os.path.join(tables_dir, "fewshot_results.csv")
    
    if not os.path.exists(inf_path) or not os.path.exists(few_path):
        print(f"Error: Files not found in {tables_dir}")
        return

    print("--- Loading Data ---")
    inf_df = pd.read_csv(inf_path)
    few_df = pd.read_csv(few_path)
    
    print(f"Inference Rows: {len(inf_df)}")
    print(f"Fewshot Rows: {len(few_df)}")
    
    # Check context length
    print("\n--- Context Lengths in Inference Data ---")
    if 'context_length' in inf_df.columns:
        print(inf_df['context_length'].value_counts())
    else:
        print("Column 'context_length' not found!")
    
    target_ctx = 512
    if 'context_length' in inf_df.columns and (inf_df['context_length'] == target_ctx).any():
        print(f"Filtering context_length={target_ctx}")
        inf_df = inf_df[inf_df['context_length'] == target_ctx]
    else:
        print(f"WARNING: context_length={target_ctx} not found! Using all data.")

    inf_agg = inf_df.groupby(['model', 'arch']).agg({
        'latency_ms': 'mean', 
        'inference_energy_per_sample_joules': 'mean'
    }).reset_index()
    
    few_agg = few_df.groupby(['model', 'arch'])['accuracy'].mean().reset_index()
    
    print("\n--- Merging ---")
    merged = pd.merge(inf_agg, few_agg, on=['model', 'arch'], how='inner')
    
    if merged.empty:
        print("\nCRITICAL: Merged DataFrame is empty! Check model names:")
        print("Inference Models (First 10):", inf_agg['model'].unique()[:10])
        print("Fewshot Models (First 10):", few_agg['model'].unique()[:10])
        
        # Check for intersection
        inf_models = set(inf_agg['model'].unique())
        few_models = set(few_agg['model'].unique())
        common = inf_models.intersection(few_models)
        print(f"\nCommon Models: {common}")
    else:
        print("\nMerged Data Preview:")
        print(merged[['model', 'arch', 'latency_ms', 'inference_energy_per_sample_joules', 'accuracy']])
        
        print("\n--- Energy Stats ---")
        print(merged['inference_energy_per_sample_joules'].describe())

if __name__ == "__main__":
    debug_data()

