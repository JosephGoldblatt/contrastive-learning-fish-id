#!/usr/bin/env python3

"""
Extracts and visualizes test accuracy statistics from the evaluation of SimCLR-trained models
across a range of training checkpoints. 
Extracts and represents four evaluation results (linear and non-linear eval, partial and full fine tuning).
Visualizes the evaluation results averaged over the results produces using ten random seeds.

Usage: python3 simclr_eval_data_extractor.py
"""

import os
import pandas as pd #type: ignore
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt #type: ignore

BASE_DIR = 'simclr_eval_results'

def extract_statistics(base_dir=BASE_DIR):
    """Extract mean and std of test accuracy across all runs."""
    data = {}
    
    # Walk through all test directories
    for test_dir in Path(base_dir).glob('test_all_*'):
        run_dir = list(test_dir.glob('run_*'))[0] if list(test_dir.glob('run_*')) else None
        if not run_dir:
            continue
            
        # Process each training mode
        for mode in ['full', 'linear', 'nonlinear', 'partial']:
            csv_file = run_dir / mode / f'{mode}_batch_results.csv'
            
            if csv_file.exists():
                df = pd.read_csv(csv_file)
                
                if mode not in data:
                    data[mode] = {}
                
                # Store test accuracy values by checkpoint
                for _, row in df.iterrows():
                    checkpoint = row['checkpoint_iter']
                    test_acc = row['test_acc']
                    
                    if checkpoint not in data[mode]:
                        data[mode][checkpoint] = []
                    
                    data[mode][checkpoint].append(test_acc)
    
    # Calculate statistics
    results = []
    for mode in sorted(data.keys()):
        for checkpoint in sorted(data[mode].keys()):
            values = data[mode][checkpoint]
            results.append({
                'mode': mode,
                'checkpoint': checkpoint,
                'mean_test_acc': np.mean(values),
                'std_test_acc': np.std(values, ddof=1),
                'n_samples': len(values)
            })
    
    df_results = pd.DataFrame(results)
    df_results = df_results.sort_values(['mode', 'checkpoint'])
    
    # Create output directories
    Path('data').mkdir(exist_ok=True)
    Path('plots').mkdir(exist_ok=True)
    
    # Save CSV files
    df_results.to_csv('data/simclr_eval_stats.csv', index=False)
    
    # Create pivot tables
    pivot_mean = df_results.pivot(index='checkpoint', columns='mode', values='mean_test_acc')
    pivot_std = df_results.pivot(index='checkpoint', columns='mode', values='std_test_acc')
    pivot_mean.to_csv('data/simclr_eval_means.csv')
    pivot_std.to_csv('data/simclr_eval_stds.csv')
    
    # Create visualization
    create_bar_graph(df_results)
    
    print("Statistics extracted successfully!")
    
    return df_results

def create_bar_graph(df_results):
    """Create a bar graph with error bars for test accuracy."""
    fig, ax = plt.subplots(figsize=(14, 8))
    
    checkpoints = sorted(df_results['checkpoint'].unique())
    modes = ['nonlinear', 'linear', 'partial', 'full']
    
    # Blue color palette
    blue_colors = {
        'nonlinear': '#08519c',
        'linear': '#3182bd',
        'partial': '#6baed6',
        'full': '#9ecae1'
    }
    
    bar_width = 0.2
    x = np.arange(len(checkpoints))
    
    # Plot bars for each mode
    for i, mode in enumerate(modes):
        mode_data = df_results[df_results['mode'] == mode]
        
        means = []
        stds = []
        for checkpoint in checkpoints:
            checkpoint_data = mode_data[mode_data['checkpoint'] == checkpoint]
            if not checkpoint_data.empty:
                means.append(checkpoint_data['mean_test_acc'].values[0])
                stds.append(checkpoint_data['std_test_acc'].values[0])
            else:
                means.append(0)
                stds.append(0)
        
        positions = x + (i - len(modes)/2 + 0.5) * bar_width
        
        ax.bar(positions, means, bar_width, 
               label=mode.capitalize(), 
               color=blue_colors[mode],
               yerr=stds,
               capsize=3,
               error_kw={'linewidth': 1, 'ecolor': 'black', 'alpha': 0.7})
    
    # Customize plot
    ax.set_xlabel('Checkpoint Iterations', fontsize=25, fontweight='bold')
    ax.set_ylabel('Top-1 Test Accuracy (%)', fontsize=25, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f'{int(cp/1000)}k' for cp in checkpoints])
    ax.grid(True, axis='y', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)
    ax.legend(loc='upper right', fontsize=20)

    ax.tick_params(axis='both', which='major', labelsize=20)
    
    # Set y-axis limits
    y_min = df_results['mean_test_acc'].min() - 10
    y_max = df_results['mean_test_acc'].max() + 10
    ax.set_ylim([max(0, y_min), min(100, y_max)])

    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['cmss10']
    plt.rcParams['text.usetex'] = True
    
    plt.tight_layout()
    plt.savefig('plots/simclr_eval_results.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("\nGraph saved as 'plots/simclr_eval_results.png'")

if __name__ == "__main__":
    df = extract_statistics()
    print(f"\nResults saved!")
