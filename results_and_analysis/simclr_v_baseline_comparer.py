#!/usr/bin/env python3

"""
Produces a chart comparing the linear evaluation results of a baseline ImageNet trained ResNet against 
a specific SimCLR-trained model across a range of different random seeds.

Usage: python3 simclr_v_baseline_comparer.py
"""

import pandas as pd #type: ignore
import numpy as np
import json
from pathlib import Path
import matplotlib.pyplot as plt #type: ignore

# Configuration
RESNET_DIR = "resnet_baseline_results"
SIMCLR_DIR = "simclr_eval_results"
CHECKPOINT = 25000
MODE = 'linear'

def extract_resnet_baseline_data():
    """Extract ResNet baseline linear evaluation results from seed folders."""
    resnet_data = []
    for seed_folder in sorted(Path(RESNET_DIR).glob('seed_*')):
        with open(seed_folder / 'linear_eval' / 'summary.json', 'r') as f:
            resnet_data.append({
                'seed': int(seed_folder.name.split('_')[1]), 
                'test_acc': json.load(f)['test_acc']
            })
    return pd.DataFrame(resnet_data).sort_values('seed')

def extract_simclr_data():
    """Extract SimCLR linear evaluation results at specified checkpoint."""
    simclr_data = []
    for test_dir in sorted(Path(SIMCLR_DIR).glob('test_all_*')):
        csv_file = list(test_dir.glob(f'run_*/{MODE}/{MODE}_batch_results.csv'))[0]
        df = pd.read_csv(csv_file)
        simclr_data.append({
            'seed': int(test_dir.name.split('_')[-1]),
            'test_acc': df[df['checkpoint_iter'] == CHECKPOINT]['test_acc'].values[0]
        })
    return pd.DataFrame(simclr_data).sort_values('seed')

def create_comparison_plot(stats):
    """Create error bar plot comparing ResNet baseline and SimCLR results."""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    labels = ['ResNet Baseline\n(Linear Eval)', 
              f'SimCLR\n(Linear Eval, Checkpoint: {CHECKPOINT//1000}k)']
    colors = ["#3f9adb", "#092285"]
    
    for i, ((mean, std), label, color) in enumerate(zip(stats, labels, colors), 1):
        ax.errorbar(i, mean, yerr=std, fmt='o', markersize=10, capsize=5, 
                   capthick=2, color=color)
    
    ax.set_xlim([0.5, 2.5])
    ax.set_ylim([40, 70])
    ax.set_xticks([1, 2])
    ax.set_xticklabels(labels, fontsize=20)
    ax.set_ylabel('Test Accuracy (%)', fontweight='bold', fontsize=20)
    ax.grid(True, axis='y', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)
    
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['cmss10']
    plt.rcParams['text.usetex'] = True
    
    plt.tight_layout()
    plt.savefig('plots/linear_eval_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """Main function to run comparison analysis."""
    # Create output directories
    Path('data').mkdir(exist_ok=True)
    Path('plots').mkdir(exist_ok=True)
    
    # Extract data
    df_resnet = extract_resnet_baseline_data()
    df_simclr = extract_simclr_data()
    
    # Calculate statistics
    stats = [
        (df_resnet['test_acc'].mean(), df_resnet['test_acc'].std()),
        (df_simclr['test_acc'].mean(), df_simclr['test_acc'].std())
    ]
    
    # Create plot
    create_comparison_plot(stats)
    
    # Save data
    df_resnet.to_csv('data/resnet_baseline_linear_eval.csv', index=False)
    df_simclr.to_csv(f'data/simclr_{MODE}_checkpoint_{CHECKPOINT}.csv', index=False)
    
    # Print summary
    print(f"ResNet Baseline: {stats[0][0]:.2f}% ± {stats[0][1]:.2f}%")
    print(f"SimCLR ({MODE}, {CHECKPOINT//1000}k): {stats[1][0]:.2f}% ± {stats[1][1]:.2f}%")
    print(f"Difference: {(stats[1][0] - stats[0][0]):.2f}%")

if __name__ == "__main__":
    main()