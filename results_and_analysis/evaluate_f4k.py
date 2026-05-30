#!/usr/bin/env python3

"""
Extracts and visualizes test accuracy statistics (using linear evaluation and full fine tuning) 
of a SimCLR-trained ResNet on the Fish4Knowledge dataset across all saved checkpoints.

Usage: python3 evaluate_f4k.py
"""

import pandas as pd #type: ignore
import numpy as np
import json
from pathlib import Path
import matplotlib.pyplot as plt #type: ignore

# Configuration
BASE_DIR = "fish4k_eval_results"
LINEAR_RUN = "run_20250903_103737"
PARTIAL_RUN = "run_20250903_112823"

def extract_checkpoint_data(base_path, mode, run_name):
    """Extract test accuracy for all checkpoints from detailed results files."""
    run_path = Path(base_path) / mode / run_name
    checkpoint_data = []
    
    # Find all detailed results files
    for detail_file in sorted(run_path.glob(f'{mode}_detailed_results_*.json')):
        checkpoint = int(detail_file.stem.split('_')[-1])
        
        with open(detail_file, 'r') as f:
            data = json.load(f)
            test_acc = data.get('test_acc', data.get('test_accuracy', 0))
            checkpoint_data.append({
                'checkpoint': checkpoint,
                'test_acc': test_acc
            })
    
    return pd.DataFrame(checkpoint_data).sort_values('checkpoint')

def create_comparison_plot(df_linear, df_partial):
    """Create grouped bar chart comparing linear and partial evaluation."""
    fig, ax = plt.subplots(figsize=(12, 7))
    
    checkpoints = df_linear['checkpoint'].values
    x_pos = np.arange(len(checkpoints))
    width = 0.35
    
    # Create bars
    ax.bar(x_pos - width/2, df_linear['test_acc'].values, width, 
           label='Linear Evaluation', color='#3f9adb', alpha=0.8)
    ax.bar(x_pos + width/2, df_partial['test_acc'].values, width,
           label='Partial Fine-Tuning', color='#092285', alpha=0.8)
    
    # Customize plot
    ax.set_xlabel('Checkpoint Iterations', fontweight='bold', fontsize=25)
    ax.set_ylabel('Top-1 test Accuracy (%)', fontweight='bold', fontsize=25)
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f'{c//1000}k' for c in checkpoints], rotation=45, ha='right')
    ax.legend(loc='upper right', fontsize=20)

    ax.tick_params(axis='both', which='major', labelsize=20)
    
    ax.grid(True, axis='y', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)
    
    # Set y-axis limits
    y_min = min(df_linear['test_acc'].min(), df_partial['test_acc'].min()) - 5
    y_max = max(df_linear['test_acc'].max(), df_partial['test_acc'].max()) + 5
    ax.set_ylim([max(0, y_min), min(100, y_max)])
    
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['cmss10']
    plt.rcParams['text.usetex'] = True
    
    plt.tight_layout()
    plt.savefig('plots/fish4k.png', dpi=300, bbox_inches='tight')
    plt.show()

def print_summary(df_linear, df_partial):
    """Print summary statistics for both evaluation modes."""
    linear_best_idx = df_linear['test_acc'].idxmax()
    partial_best_idx = df_partial['test_acc'].idxmax()
    
    print(f"\nLinear Evaluation: Best {df_linear['test_acc'].max():.2f}% at checkpoint {df_linear.loc[linear_best_idx, 'checkpoint']}")
    print(f"Partial Evaluation: Best {df_partial['test_acc'].max():.2f}% at checkpoint {df_partial.loc[partial_best_idx, 'checkpoint']}")
    print(f"Difference: {(df_linear['test_acc'].max() - df_partial['test_acc'].max()):.2f}%")

def main():
    """Main function to run F4K evaluation analysis."""
    # Create output directory
    Path('plots').mkdir(exist_ok=True)
    
    # Extract data for both evaluation modes
    df_linear = extract_checkpoint_data(BASE_DIR, 'linear', LINEAR_RUN)
    df_partial = extract_checkpoint_data(BASE_DIR, 'partial', PARTIAL_RUN)
    
    # Create visualization
    create_comparison_plot(df_linear, df_partial)
    
    # Print summary
    print(f"Figure saved as plots/fish4k.png")
    print_summary(df_linear, df_partial)

if __name__ == "__main__":
    main()