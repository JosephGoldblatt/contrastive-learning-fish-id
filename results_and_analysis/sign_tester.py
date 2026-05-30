#!/usr/bin/env python3

"""
Performs a basic sign test to test for statistically significant difference between 
a SimCLR-trained ResNet and a YOLO classifier model performance.

Usage: python3 sign_tester.py
"""

import pandas as pd #type: ignore
from pathlib import Path
from scipy import stats #type: ignore

def main():
    """Run sign test analysis."""
    # Create output directory
    Path('data').mkdir(exist_ok=True)
    
    # Read data and calculate differences
    df = pd.read_csv('data/comparison_results.csv')
    df['diff'] = df['simclr_acc'] - df['yolo_acc']
    
    # Count differences
    n_positive = (df['diff'] > 0).sum()
    n_negative = (df['diff'] < 0).sum()
    n_ties = (df['diff'] == 0).sum()
    
    # Perform sign test
    n_total = n_positive + n_negative
    p_value = stats.binomtest(n_positive, n_total, p=0.5, alternative='two-sided').pvalue
    
    # Save results to CSV
    results_df = pd.DataFrame({
        'Metric': ['Positive differences', 'Negative differences', 'Ties', 
                   'P-value', 'Mean SimCLR accuracy', 'Mean YOLO accuracy'],
        'Value': [n_positive, n_negative, n_ties, p_value,
                  df['simclr_acc'].mean(), df['yolo_acc'].mean()]
    })
    results_df.to_csv('data/sign_test_results.csv', index=False)
    
    # Print results
    print("Sign Test Results:")
    print(f"Positive differences (SimCLR > YOLO): {n_positive}")
    print(f"Negative differences (SimCLR < YOLO): {n_negative}")
    print(f"Ties: {n_ties}")
    print(f"P-value (two-tailed): {p_value:.4f}")

    print(f"\nResults saved to: data/sign_test_results.csv")

if __name__ == "__main__":
    main()