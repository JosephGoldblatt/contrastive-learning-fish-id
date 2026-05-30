#!/usr/bin/env python3

"""
Trains and evaluates a YOLOv11-n-cls model multiple times across a range of random seeds.

Usage: python3 yolo_trainer.py --dataset-dir <dir> --output-dir <dir>
"""

import os
import json
import shutil
import argparse
from pathlib import Path
from ultralytics import YOLO #type: ignore
import torch #type: ignore
import numpy as np
import random
from datetime import datetime


def set_seed(seed):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)


def verify_dataset(dataset_path):
    """Verify dataset structure."""
    for split in ['train', 'val', 'test']:
        split_path = Path(dataset_path) / split
        if not split_path.exists():
            raise ValueError(f"{split} directory not found")
    
    train_path = Path(dataset_path) / 'train'
    num_classes = len([d for d in train_path.iterdir() if d.is_dir()])
    return num_classes


def get_class_names(dataset_path):
    """Extract sorted class names from directory structure."""
    train_path = Path(dataset_path) / 'train'
    class_dirs = [d.name for d in train_path.iterdir() if d.is_dir()]
    return sorted(class_dirs, key=int)


def train_model(dataset_dir, output_dir, run_number, seed, epochs, batch_size, img_size, lr):
    """Train YOLOv11 classification model."""
    set_seed(seed)
    
    model = YOLO('yolo11n-cls.pt')
    
    results = model.train(
        data=dataset_dir,
        epochs=epochs,
        imgsz=img_size,
        batch=batch_size,
        lr0=lr,
        device=0 if torch.cuda.is_available() else 'cpu',
        project=output_dir,
        name=f'run_{run_number}_seed_{seed}',
        seed=seed,
        deterministic=True,
        save=True,
        plots=False,
        verbose=False
    )
    
    return results


def evaluate_and_save_predictions(model_path, dataset_dir, class_names, seed, epochs, batch_size, img_size, lr):
    """Evaluate model and save detailed predictions."""
    set_seed(seed)
    
    model = YOLO(model_path)
    
    # Get YOLO's internal class mapping
    yolo_class_mapping = model.model.names
    
    # Evaluate on validation and test sets
    val_results = model.val(data=dataset_dir, split='val')
    test_results = model.val(data=dataset_dir, split='test')
    
    # Generate detailed predictions for test set
    test_predictions = []
    test_dir = Path(dataset_dir) / 'test'
    
    for class_dir in test_dir.iterdir():
        if class_dir.is_dir():
            true_class = class_dir.name
            
            for image_path in list(class_dir.glob('*.jpg')) + list(class_dir.glob('*.png')) + list(class_dir.glob('*.jpeg')):
                results = model(str(image_path))
                probs = results[0].probs
                
                # Get top-5 predictions
                top5_indices = probs.top5
                top5_probs = [probs.data[idx].item() for idx in top5_indices]
                top5_classes = [yolo_class_mapping[int(idx)] for idx in top5_indices]
                
                prediction_data = {
                    'image_path': str(image_path.relative_to(dataset_dir)),
                    'true_class': true_class,
                    'predicted_class': yolo_class_mapping[int(probs.top1)],
                    'confidence': probs.top1conf.item(),
                    'top5_predictions': {
                        'classes': top5_classes,
                        'probabilities': top5_probs
                    }
                }
                test_predictions.append(prediction_data)
    
    # Calculate confusion matrix
    confusion_matrix = {}
    for pred in test_predictions:
        true_class = pred['true_class']
        pred_class = pred['predicted_class']
        
        if true_class not in confusion_matrix:
            confusion_matrix[true_class] = {}
        if pred_class not in confusion_matrix[true_class]:
            confusion_matrix[true_class][pred_class] = 0
        
        confusion_matrix[true_class][pred_class] += 1
    
    # Compile results
    results_summary = {
        'model_info': {
            'model_path': model_path,
            'random_seed': seed,
            'num_classes': len(class_names),
            'class_names': class_names,
            'training_params': {
                'epochs': epochs,
                'image_size': img_size,
                'batch_size': batch_size,
                'learning_rate': lr
            }
        },
        'performance_metrics': {
            'validation': {
                'top1_accuracy': float(val_results.top1),
                'top5_accuracy': float(val_results.top5)
            },
            'test': {
                'top1_accuracy': float(test_results.top1),
                'top5_accuracy': float(test_results.top5)
            }
        },
        'confusion_matrix': confusion_matrix,
        'predictions': test_predictions
    }
    
    return results_summary, val_results, test_results


def save_run_results(results_summary, model_path, run_dir, output_base, run_number, seed):
    """Save results for a single run."""
    run_results_path = Path(output_base) / f'run_{run_number}_seed_{seed}'
    run_results_path.mkdir(parents=True, exist_ok=True)
    
    # Copy training outputs directory if exists
    if run_dir.exists():
        results_destination = run_results_path / 'training_outputs'
        shutil.copytree(run_dir, results_destination, dirs_exist_ok=True)
    
    # Copy best model
    if Path(model_path).exists():
        best_model_destination = run_results_path / 'best_model.pt'
        shutil.copy2(model_path, best_model_destination)
    
    # Save predictions JSON
    predictions_file = run_results_path / 'test_predictions.json'
    with open(predictions_file, 'w') as f:
        json.dump(results_summary, f, indent=2)
    
    # Save summary text file
    summary_file = run_results_path / 'training_summary.txt'
    with open(summary_file, 'w') as f:
        f.write(f"YOLOv11 Classification Training Summary - Run {run_number}\n")
        f.write("=" * 50 + "\n")
        f.write(f"Random Seed: {seed}\n")
        f.write(f"Dataset: {Path(model_path).parent.parent.parent}\n")
        f.write(f"Number of classes: {len(results_summary['model_info']['class_names'])}\n")
        f.write(f"Training parameters:\n")
        for param, value in results_summary['model_info']['training_params'].items():
            f.write(f"  {param}: {value}\n")
        f.write(f"\nPerformance Metrics:\n")
        f.write(f"  Validation Top-1: {results_summary['performance_metrics']['validation']['top1_accuracy']:.4f}\n")
        f.write(f"  Validation Top-5: {results_summary['performance_metrics']['validation']['top5_accuracy']:.4f}\n")
        f.write(f"  Test Top-1: {results_summary['performance_metrics']['test']['top1_accuracy']:.4f}\n")
        f.write(f"  Test Top-5: {results_summary['performance_metrics']['test']['top5_accuracy']:.4f}\n")
    
    return str(run_results_path)


def save_aggregate_results(all_runs_metrics, output_dir):
    """Save aggregate results."""
    output_path = Path(output_dir)
    
    # Save aggregate results
    aggregate_file = output_path / 'aggregate_results.json'
    with open(aggregate_file, 'w') as f:
        json.dump(all_runs_metrics, f, indent=2)
    
    # Calculate and save statistics
    stats_file = output_path / 'performance_statistics.txt'
    
    val_top1_scores = [run['validation']['top1_accuracy'] for run in all_runs_metrics]
    val_top5_scores = [run['validation']['top5_accuracy'] for run in all_runs_metrics]
    test_top1_scores = [run['test']['top1_accuracy'] for run in all_runs_metrics]
    test_top5_scores = [run['test']['top5_accuracy'] for run in all_runs_metrics]
    
    with open(stats_file, 'w') as f:
        f.write("Performance Statistics Across All Runs\n")
        f.write("=" * 50 + "\n\n")
        
        f.write("Validation Top-1 Accuracy:\n")
        f.write(f"  Mean: {np.mean(val_top1_scores):.4f}\n")
        f.write(f"  Std:  {np.std(val_top1_scores):.4f}\n")
        f.write(f"  Min:  {np.min(val_top1_scores):.4f}\n")
        f.write(f"  Max:  {np.max(val_top1_scores):.4f}\n\n")
        
        f.write("Validation Top-5 Accuracy:\n")
        f.write(f"  Mean: {np.mean(val_top5_scores):.4f}\n")
        f.write(f"  Std:  {np.std(val_top5_scores):.4f}\n")
        f.write(f"  Min:  {np.min(val_top5_scores):.4f}\n")
        f.write(f"  Max:  {np.max(val_top5_scores):.4f}\n\n")
        
        f.write("Test Top-1 Accuracy:\n")
        f.write(f"  Mean: {np.mean(test_top1_scores):.4f}\n")
        f.write(f"  Std:  {np.std(test_top1_scores):.4f}\n")
        f.write(f"  Min:  {np.min(test_top1_scores):.4f}\n")
        f.write(f"  Max:  {np.max(test_top1_scores):.4f}\n\n")
        
        f.write("Test Top-5 Accuracy:\n")
        f.write(f"  Mean: {np.mean(test_top5_scores):.4f}\n")
        f.write(f"  Std:  {np.std(test_top5_scores):.4f}\n")
        f.write(f"  Min:  {np.min(test_top5_scores):.4f}\n")
        f.write(f"  Max:  {np.max(test_top5_scores):.4f}\n\n")
        
        f.write("Individual Run Results:\n")
        f.write("-" * 50 + "\n")
        for i, run in enumerate(all_runs_metrics):
            f.write(f"Run {run['run_number']} (Seed {run['seed']}):\n")
            f.write(f"  Val Top-1: {run['validation']['top1_accuracy']:.4f}, ")
            f.write(f"Val Top-5: {run['validation']['top5_accuracy']:.4f}\n")
            f.write(f"  Test Top-1: {run['test']['top1_accuracy']:.4f}, ")
            f.write(f"Test Top-5: {run['test']['top5_accuracy']:.4f}\n")
    
    print(f"Test accuracy: {np.mean(test_top1_scores):.2f}% ± {np.std(test_top1_scores):.2f}%")


def main():
    parser = argparse.ArgumentParser(description='Train YOLOv11 classifier')
    parser.add_argument('--dataset-dir', type=str, required=True,
                       help='Path to dataset directory')
    parser.add_argument('--output-dir', type=str, required=True,
                       help='Path to output directory')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--img-size', type=int, default=224)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--seeds', type=int, nargs='+', 
                       default=list(range(1, 11)),
                       help='Random seeds to use')
    
    args = parser.parse_args()
    
    # Setup
    dataset_path = Path(args.dataset_dir)
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Verify dataset
    num_classes = verify_dataset(dataset_path)
    class_names = get_class_names(dataset_path)
    
    print(f"Dataset: {num_classes} classes")
    print(f"Training {len(args.seeds)} runs with seeds: {args.seeds}")
    
    all_runs_metrics = []
    
    for run_idx, seed in enumerate(args.seeds, 1):
        print(f"\n[{run_idx}/{len(args.seeds)}] Training with seed {seed}")
        
        # Train model
        train_model(
            str(dataset_path),
            str(output_path / 'runs'),
            run_idx,
            seed,
            args.epochs,
            args.batch_size,
            args.img_size,
            args.lr
        )
        
        # Find model path
        run_name = f'run_{run_idx}_seed_{seed}'
        run_dirs = list((output_path / 'runs').glob(f'{run_name}*'))
        
        if not run_dirs:
            print(f"Warning: No run directory found for run {run_idx}")
            continue
            
        run_dir = sorted(run_dirs)[-1]
        best_model_path = run_dir / 'weights' / 'best.pt'
        
        if not best_model_path.exists():
            print(f"Warning: Best model not found for run {run_idx}")
            continue
        
        # Evaluate and save predictions
        results_summary, val_results, test_results = evaluate_and_save_predictions(
            str(best_model_path),
            str(dataset_path),
            class_names,
            seed,
            args.epochs,
            args.batch_size,
            args.img_size,
            args.lr
        )
        
        # Save run results
        run_results_path = save_run_results(
            results_summary,
            str(best_model_path),
            run_dir,
            str(output_path),
            run_idx,
            seed
        )
        
        # Store metrics for aggregate analysis 
        run_metrics = {
            'run_number': run_idx,
            'seed': seed,
            'validation': results_summary['performance_metrics']['validation'],
            'test': results_summary['performance_metrics']['test'],
            'results_path': run_results_path
        }
        all_runs_metrics.append(run_metrics)
        
        print(f"Val: {val_results.top1:.2f}% Test: {test_results.top1:.2f}%")
        
        # Clean up run directory to save space
        if run_dir.exists():
            shutil.rmtree(run_dir)
    
    # Save aggregate results
    if all_runs_metrics:
        save_aggregate_results(all_runs_metrics, output_path)
    
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()