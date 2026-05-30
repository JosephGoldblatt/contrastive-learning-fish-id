#!/usr/bin/env python3

"""
Analyzes YOLO model predictions and generates confusion matrices and classification metrics.

Usage: python3 yolo_run_analyser.py
"""

import json
import numpy as np
import pandas as pd #type: ignore
import matplotlib.pyplot as plt #type: ignore
import seaborn as sns #type: ignore
from pathlib import Path
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score #type: ignore

# Configuration
JSON_FILE = "yolo_eval_results/run_8_seed_8/test_predictions.json"

def load_and_process_data(json_file):
    """Load YOLO predictions and construct confusion matrix."""
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    classes = data['model_info']['class_names']
    cm_dict = data['confusion_matrix']
    
    # Create confusion matrix array
    cm = np.zeros((len(classes), len(classes)), dtype=int)
    class_to_idx = {cls: idx for idx, cls in enumerate(classes)}
    
    # Fill confusion matrix
    for true_class, predictions in cm_dict.items():
        if true_class in class_to_idx:
            true_idx = class_to_idx[true_class]
            for pred_class, count in predictions.items():
                if pred_class in class_to_idx:
                    pred_idx = class_to_idx[pred_class]
                    cm[true_idx, pred_idx] = count
    
    # Generate prediction arrays for metrics
    preds = []
    true = []
    for true_class, predictions in cm_dict.items():
        if true_class in class_to_idx:
            true_idx = class_to_idx[true_class]
            for pred_class, count in predictions.items():
                if pred_class in class_to_idx:
                    pred_idx = class_to_idx[pred_class]
                    preds.extend([pred_idx] * count)
                    true.extend([true_idx] * count)
    
    return np.array(preds), np.array(true), cm, classes

def create_confusion_matrices(cm, classes):
    """Generate standard and log-scaled confusion matrices."""

    # Standard confusion matrix
    plt.figure(figsize=(12, 10))

    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['cmss10']
    plt.rcParams['text.usetex'] = True

    sns.heatmap(cm, annot=True, fmt='d', cmap='PuRd', 
                xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig('plots/yolo_confusion_matrix.png', dpi=150)
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


    plt.xlabel('Predicted', fontsize=35)
    plt.ylabel('Actual', fontsize=35)
    plt.tight_layout()
    plt.savefig('plots/yolo_log_confusion_matrix.png', dpi=150)
    plt.close()

def calculate_and_save_metrics(true, preds):
    """Calculate classification metrics and save to CSV."""
    accuracy = accuracy_score(true, preds)
    precision = precision_score(true, preds, average='weighted', zero_division=0)
    recall = recall_score(true, preds, average='weighted', zero_division=0)
    f1 = f1_score(true, preds, average='weighted', zero_division=0)
    
    # Create metrics dataframe
    metrics_df = pd.DataFrame({
        'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score'],
        'Value': [accuracy, precision, recall, f1]
    })
    metrics_df.to_csv('data/yolo_classification_metrics.csv', index=False)
    
    return accuracy, precision, recall, f1

def main():
    """Main function to run YOLO analysis."""
    # Create output directories
    Path('data').mkdir(exist_ok=True)
    Path('plots').mkdir(exist_ok=True)
    
    # Load and process data
    preds, true, cm, classes = load_and_process_data(JSON_FILE)
    
    # Generate confusion matrices
    create_confusion_matrices(cm, classes)
    
    # Calculate and save metrics
    accuracy, precision, recall, f1 = calculate_and_save_metrics(true, preds)
    
    # Print results
    print(f"Results saved.")
    print(f"- Confusion Matrix: plots/yolo_confusion_matrix.png")
    print(f"- Log Confusion Matrix: plots/yolo_log_confusion_matrix.png")
    print(f"- Metrics: data/yolo_classification_metrics.csv")
    print(f"\nMetrics:")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")

if __name__ == "__main__":
    main()