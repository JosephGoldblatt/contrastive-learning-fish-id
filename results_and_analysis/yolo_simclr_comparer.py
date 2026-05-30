#!/usr/bin/env python3

"""
Produces a visualization comparing the performance of a SimCLR-trained ResNet model
against a YOLO classifier model across a range of different random seeds.

Usage: python3 yolo_simclr_comparer.py
"""

import pandas as pd #type: ignore
import numpy as np
import json
from pathlib import Path
import matplotlib.pyplot as plt #type: ignore

# Configuration
SIMCLR_BASE_DIR = "simclr_eval_results"
YOLO_JSON_FILE = "yolo_eval_results/aggregate_results.json"
CHECKPOINT = 25000
MODE = 'full'

def extract_simclr_results():
    """Extract SimCLR test accuracy results for each seed."""
    simclr_results = []
    for test_dir in sorted(Path(SIMCLR_BASE_DIR).glob('test_all_*')):
        seed = int(test_dir.name.split('_')[-1])
        csv_file = list(test_dir.glob(f'run_*/{MODE}/{MODE}_batch_results.csv'))[0]
        if csv_file.exists():
            df = pd.read_csv(csv_file)
            test_acc = df[df['checkpoint_iter'] == CHECKPOINT]['test_acc'].values[0]
            simclr_results.append({'seed': seed, 'simclr_acc': test_acc})
    
    return pd.DataFrame(simclr_results).sort_values('seed')

def extract_yolo_results():
    """Extract YOLO test accuracy results from aggregate JSON."""
    with open(YOLO_JSON_FILE, 'r') as f:
        yolo_data = json.load(f)
    
    yolo_results = [{'seed': run['seed'], 'yolo_acc': run['test']['top1_accuracy'] * 100} 
                    for run in yolo_data]
    return pd.DataFrame(yolo_results).sort_values('seed')

def create_comparison_plot(df):
    """Create bar plot comparing SimCLR and YOLO performance."""
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(df))
    width = 0.35
    
    ax.bar(x - width/2, df['simclr_acc'], width, 
           label=f'SimCLR (Linear Eval, {CHECKPOINT//1000}k)', color="#0A289F")
    ax.bar(x + width/2, df['yolo_acc'], width, 
           label='YOLO', color="#579bea")
    
    ax.set_ylim([60, 80])
    ax.set_xlabel('Seed', fontsize=25, fontweight='bold')
    ax.set_ylabel('Top-1 test Accuracy (%)', fontsize=25, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(df['seed'])
    ax.legend(fontsize=20)

    ax.tick_params(axis='both', which='major', labelsize=20)
    
    ax.grid(True, axis='y', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)
    
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['cmss10']
    plt.rcParams['text.usetex'] = True
    
    plt.tight_layout()
    plt.savefig('plots/model_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """Main function to run model comparison."""
    # Create output directories
    Path('data').mkdir(exist_ok=True)
    Path('plots').mkdir(exist_ok=True)
    
    # Extract results
    df_simclr = extract_simclr_results()
    df_yolo = extract_yolo_results()
    
    # Merge dataframes
    df = pd.merge(df_simclr, df_yolo, on='seed')
    
    # Create comparison plot
    create_comparison_plot(df)
    
    # Calculate and print statistics
    print(f"SimCLR: Mean={df['simclr_acc'].mean():.2f}%, Std={df['simclr_acc'].std():.2f}%")
    print(f"YOLO:   Mean={df['yolo_acc'].mean():.2f}%, Std={df['yolo_acc'].std():.2f}%")
    print(f"Difference (YOLO-SimCLR): {(df['yolo_acc'].mean() - df['simclr_acc'].mean()):.2f}%")
    
    # Save results
    df.to_csv('data/comparison_results.csv', index=False)

if __name__ == "__main__":
    main()