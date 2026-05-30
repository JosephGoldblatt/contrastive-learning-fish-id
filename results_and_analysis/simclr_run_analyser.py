#!/usr/bin/env python3

"""
Analyzes the predictions produced during evaluation of a specific SimCLR-trained model 
and generates confusion matrices and outputs classification metrics.

Usage: python3 simclr_run_analyser.py
"""

import json
import numpy as np
import pandas as pd #type: ignore
import matplotlib.pyplot as plt #type: ignore
import seaborn as sns #type: ignore
from pathlib import Path
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score #type: ignore

# Configuration
JSON_FILE = "simclr_eval_results/test_all_5/run_20250901_055406/full/full_detailed_results_25000.json"

def load_and_process_data(json_file):
    """Load predictions data and remap to sorted class indices."""
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    # Extract data
    preds = np.array(data['predictions_data']['test']['predictions'])
    true = np.array(data['predictions_data']['test']['true_labels'])
    classes_original = data['predictions_data']['class_names']
    
    # Sort classes numerically
    classes = sorted(classes_original, key=lambda x: int(x))
    
    # Create mapping from original to sorted indices
    original_to_sorted = {i: classes.index(classes_original[i]) for i in range(len(classes_original))}
    
    # Remap to sorted indices
    preds_sorted = np.array([original_to_sorted[pred] for pred in preds])
    true_sorted = np.array([original_to_sorted[label] for label in true])
    
    return preds_sorted, true_sorted, classes

def create_confusion_matrices(true_sorted, preds_sorted, classes):
    """Generate standard and log-scaled confusion matrices."""
    cm = confusion_matrix(true_sorted, preds_sorted)
    
    # Standard confusion matrix
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.savefig('plots/confusion_matrix.png', dpi=150)
    plt.close()
    
    # Log-scaled confusion matrix
    plt.figure(figsize=(12, 10))

    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['cmss10']
    plt.rcParams['text.usetex'] = True

    cm_log = np.log10(cm + 1)  # Add 1 to avoid log(0)
    annot = np.array([[str(int(v)) if v > 0 else '' for v in row] for row in cm])
    ax = sns.heatmap(cm_log, annot=annot, fmt='', cmap='PuRd', 
                xticklabels=classes, yticklabels=classes,
                cbar_kws={'label': 'log(count)'})
    
    # Set colorbar label font size
    cbar = ax.collections[0].colorbar
    cbar.set_label('log(count)', fontsize=35)

    plt.xlabel('Predicted', fontsize =35)
    plt.ylabel('Actual', fontsize =35)
    plt.tight_layout()
    plt.savefig('plots/log_confusion_matrix.png', dpi=150)
    plt.close()

def calculate_and_save_metrics(true_sorted, preds_sorted):
    """Calculate classification metrics and save to CSV."""
    accuracy = accuracy_score(true_sorted, preds_sorted)
    precision = precision_score(true_sorted, preds_sorted, average='weighted', zero_division=0)
    recall = recall_score(true_sorted, preds_sorted, average='weighted', zero_division=0)
    f1 = f1_score(true_sorted, preds_sorted, average='weighted', zero_division=0)
    
    # Create metrics dataframe
    metrics_df = pd.DataFrame({
        'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score'],
        'Value': [accuracy, precision, recall, f1]
    })
    metrics_df.to_csv('data/simclr_classification_metrics.csv', index=False)
    
    return accuracy, precision, recall, f1

def main():
    """Main function to run SimCLR analysis."""
    # Create output directories
    Path('data').mkdir(exist_ok=True)
    Path('plots').mkdir(exist_ok=True)
    
    # Load and process data
    preds_sorted, true_sorted, classes = load_and_process_data(JSON_FILE)
    
    # Generate confusion matrices
    create_confusion_matrices(true_sorted, preds_sorted, classes)
    
    # Calculate and save metrics
    accuracy, precision, recall, f1 = calculate_and_save_metrics(true_sorted, preds_sorted)
    
    # Print results
    print(f"Results saved.")
    print(f"- Metrics: data/simclr_classification_metrics.csv")
    print(f"\nMetrics:")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")

if __name__ == "__main__":
    main()