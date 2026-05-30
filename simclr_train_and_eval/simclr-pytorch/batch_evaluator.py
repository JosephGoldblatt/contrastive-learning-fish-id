#!/usr/bin/env python

"""
Evaluates all ResNet-50 models produced during a SimCLR training run. 
Evaluates the accuracy of these models across the following test modes: linear and non-linear
evaluation and partial and full fine tuning.

Usage: python batch_evaluator.py --pretrain_dir <dir> --dataset_path <path> --output_dir <dir>
"""

import os
import time
import torch #type:ignore
import torch.nn as nn #type:ignore
import torch.optim as optim #type:ignore
from torch.utils.data import DataLoader #type:ignore
from torchvision import datasets, transforms, models #type:ignore
import pandas as pd #type:ignore
from pathlib import Path
import json
import argparse
import gc
import numpy as np
from collections import defaultdict


def load_simclr_encoder(checkpoint_path, device='cpu', use_cifar_head=False):
    """Load SimCLR pretrained encoder from checkpoint."""
    ckpt = torch.load(checkpoint_path, map_location=device)
    state_dict = ckpt['state_dict']
    
    encoder = models.resnet50(pretrained=False)
    
    if use_cifar_head:
        encoder.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    
    encoder.fc = nn.Identity()
    
    encoder_state = {}
    for key, value in state_dict.items():
        if key.startswith('convnet.'):
            new_key = key.replace('convnet.', '')
            encoder_state[new_key] = value
    
    encoder.load_state_dict(encoder_state, strict=False)
    return encoder


def create_classifier(encoder, num_classes, freeze_mode='partial'):
    """Create classifier with specified freeze mode."""
    with torch.no_grad():
        dummy_input = torch.randn(1, 3, 224, 224)
        features = encoder(dummy_input)
        feature_dim = features.shape[1]
    
    if freeze_mode in ['linear', 'nonlinear']:
        # Freeze entire encoder
        for param in encoder.parameters():
            param.requires_grad = False
            
        if freeze_mode == 'linear':
            classifier = nn.Sequential(
                encoder,
                nn.Linear(feature_dim, num_classes)
            )
        else:  # nonlinear
            hidden_dim = 2048
            classifier = nn.Sequential(
                encoder,
                nn.Linear(feature_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, num_classes)
            )
            
    elif freeze_mode == 'partial':
        # Freeze early layers
        layers_to_freeze = ['conv1', 'bn1', 'layer1', 'layer2']
        for name, param in encoder.named_parameters():
            if any(layer in name for layer in layers_to_freeze):
                param.requires_grad = False
            else:
                param.requires_grad = True
        
        classifier = nn.Sequential(
            encoder,
            nn.Linear(feature_dim, num_classes)
        )
        
    else:  # full
        # Unfreeze all parameters
        for param in encoder.parameters():
            param.requires_grad = True
        
        classifier = nn.Sequential(
            encoder,
            nn.Linear(feature_dim, num_classes)
        )
    
    return classifier


def get_transforms():
    """Get train and test transforms."""
    train_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.1, 0.1, 0.1, 0.05),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return train_transform, test_transform


def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    
    accuracy = 100. * correct / total
    avg_loss = running_loss / len(dataloader)
    return avg_loss, accuracy


def evaluate(model, dataloader, criterion, device):
    """Evaluate model on dataset."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    accuracy = 100. * correct / total
    avg_loss = running_loss / len(dataloader)
    return avg_loss, accuracy


def evaluate_with_predictions(model, dataloader, criterion, device):
    """Evaluate model and return predictions."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    all_predictions = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            probs = torch.softmax(outputs, dim=1)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    accuracy = 100. * correct / total
    avg_loss = running_loss / len(dataloader)
    
    return avg_loss, accuracy, np.array(all_predictions), np.array(all_labels), np.array(all_probs)


def evaluate_single_checkpoint(checkpoint_path, checkpoint_iter, data_loaders, device, args, 
                              num_classes, class_names, freeze_mode):
    """Evaluate a single checkpoint with specified freeze mode."""
    start_time = time.time()
    
    try:
        encoder = load_simclr_encoder(checkpoint_path, device=device, 
                                    use_cifar_head=not args.use_imagenet_head)
        
        model = create_classifier(encoder, num_classes, freeze_mode=freeze_mode)
        model = model.to(device)
        
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        criterion = nn.CrossEntropyLoss()
        
        # Use higher learning rate for linear/nonlinear evaluation
        if freeze_mode in ['linear', 'nonlinear']:
            lr = args.linear_lr
        else:
            lr = args.lr
            
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), 
                              lr=lr, weight_decay=args.weight_decay)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
        
        best_val_acc = 0
        training_results = []
        
        for epoch in range(args.epochs):
            train_loss, train_acc = train_epoch(model, data_loaders['train'], criterion, optimizer, device)
            val_loss, val_acc = evaluate(model, data_loaders['val'], criterion, device)
            scheduler.step()
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_model_state = model.state_dict().copy()
            
            training_results.append({
                'epoch': epoch + 1,
                'train_loss': train_loss,
                'train_acc': train_acc,
                'val_loss': val_loss,
                'val_acc': val_acc,
                'lr': optimizer.param_groups[0]['lr']
            })
        
        model.load_state_dict(best_model_state)
        test_loss, test_acc, predictions, true_labels, probabilities = evaluate_with_predictions(
            model, data_loaders['test'], criterion, device
        )
        
        val_loss_final, val_acc_final, val_predictions, val_true_labels, val_probabilities = evaluate_with_predictions(
            model, data_loaders['val'], criterion, device
        )
        
        predictions_data = {
            'test': {
                'predictions': predictions.tolist(),
                'true_labels': true_labels.tolist(),
                'probabilities': probabilities.tolist(),
                'accuracy': test_acc,
                'num_samples': len(predictions)
            },
            'validation': {
                'predictions': val_predictions.tolist(),
                'true_labels': val_true_labels.tolist(),
                'probabilities': val_probabilities.tolist(),
                'accuracy': val_acc_final,
                'num_samples': len(val_predictions)
            },
            'class_names': class_names,
            'num_classes': num_classes
        }
        
        # Calculate per-class metrics
        per_class_metrics = []
        for class_idx in range(num_classes):
            class_mask = true_labels == class_idx
            if class_mask.sum() > 0:
                class_correct = (predictions[class_mask] == true_labels[class_mask]).sum()
                class_total = class_mask.sum()
                class_acc = 100.0 * class_correct / class_total
                
                true_positives = ((predictions == class_idx) & (true_labels == class_idx)).sum()
                false_positives = ((predictions == class_idx) & (true_labels != class_idx)).sum()
                false_negatives = ((predictions != class_idx) & (true_labels == class_idx)).sum()
                
                precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
                recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
                
                per_class_metrics.append({
                    'class_idx': class_idx,
                    'class_name': class_names[class_idx],
                    'accuracy': float(class_acc),
                    'precision': float(precision),
                    'recall': float(recall),
                    'support': int(class_total)
                })
        
        predictions_data['test']['per_class_metrics'] = per_class_metrics
        
        print(f"[{freeze_mode}] Iter {checkpoint_iter}: Val={best_val_acc:.1f}% Test={test_acc:.1f}%")
        
        return {
            'checkpoint_iter': checkpoint_iter,
            'checkpoint_path': str(checkpoint_path),
            'freeze_mode': freeze_mode,
            'best_val_acc': best_val_acc,
            'test_acc': test_acc,
            'total_params': total_params,
            'trainable_params': trainable_params,
            'training_time': time.time() - start_time,
            'status': 'success',
            'training_results': training_results,
            'predictions_data': predictions_data
        }
        
    except Exception as e:
        print(f"Error: {checkpoint_iter} - {freeze_mode}")
        return {
            'checkpoint_iter': checkpoint_iter,
            'checkpoint_path': str(checkpoint_path),
            'freeze_mode': freeze_mode,
            'status': 'error',
            'error': str(e)[:200],
            'training_time': time.time() - start_time
        }
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()


def create_combined_summary(results_by_mode, class_names):
    """Create summary comparing all evaluation modes."""
    summary = {}
    
    # Process each mode's results
    for mode in ['linear', 'nonlinear', 'partial', 'full']:
        if mode in results_by_mode and results_by_mode[mode]:
            successful = [r for r in results_by_mode[mode] if r['status'] == 'success']
            if successful:
                best = max(successful, key=lambda x: x['test_acc'])
                test_accs = [r['test_acc'] for r in successful]
                
                summary[f'{mode}_evaluation'] = {
                    'best_checkpoint': {
                        'iteration': best['checkpoint_iter'],
                        'test_acc': best['test_acc'],
                        'val_acc': best['best_val_acc']
                    },
                    'test_acc_stats': {
                        'mean': np.mean(test_accs),
                        'std': np.std(test_accs),
                        'min': min(test_accs),
                        'max': max(test_accs)
                    },
                    'num_successful': len(successful),
                    'trainable_params': best['trainable_params'],
                    'total_params': best['total_params']
                }
    
    # Create comparison if multiple modes present
    modes_with_results = [mode for mode in ['linear', 'nonlinear', 'partial', 'full'] 
                         if f'{mode}_evaluation' in summary]
    
    if len(modes_with_results) > 1:
        summary['comparison'] = {}
        
        # Find overall best mode
        best_mode = max(modes_with_results, 
                       key=lambda m: summary[f'{m}_evaluation']['best_checkpoint']['test_acc'])
        summary['comparison']['best_mode'] = best_mode
        summary['comparison']['best_test_acc'] = summary[f'{best_mode}_evaluation']['best_checkpoint']['test_acc']
        
        # Ranking of all modes
        ranked_modes = sorted(modes_with_results, 
                            key=lambda m: summary[f'{m}_evaluation']['best_checkpoint']['test_acc'], 
                            reverse=True)
        summary['comparison']['ranking'] = [
            {'mode': m, 'test_acc': summary[f'{m}_evaluation']['best_checkpoint']['test_acc']} 
            for m in ranked_modes
        ]
        
        # Pairwise comparisons
        for i, mode1 in enumerate(modes_with_results):
            for mode2 in modes_with_results[i+1:]:
                key = f'{mode1}_vs_{mode2}'
                acc1 = summary[f'{mode1}_evaluation']['best_checkpoint']['test_acc']
                acc2 = summary[f'{mode2}_evaluation']['best_checkpoint']['test_acc']
                summary['comparison'][key] = {
                    'difference': acc2 - acc1,
                    'better': mode2 if acc2 > acc1 else mode1
                }
        
        # Per-checkpoint comparison
        checkpoint_comparison = []
        all_checkpoints = set()
        for mode in modes_with_results:
            if mode in results_by_mode:
                successful = [r for r in results_by_mode[mode] if r['status'] == 'success']
                all_checkpoints.update([r['checkpoint_iter'] for r in successful])
        
        for ckpt_iter in sorted(all_checkpoints):
            ckpt_results = {'checkpoint_iter': ckpt_iter}
            for mode in modes_with_results:
                if mode in results_by_mode:
                    result = next((r for r in results_by_mode[mode] 
                                 if r['status'] == 'success' and r['checkpoint_iter'] == ckpt_iter), None)
                    if result:
                        ckpt_results[f'{mode}_test_acc'] = result['test_acc']
            
            if len(ckpt_results) > 1:
                checkpoint_comparison.append(ckpt_results)
        
        summary['comparison']['per_checkpoint'] = checkpoint_comparison
    
    return summary


def main():
    parser = argparse.ArgumentParser(description='Batch SimCLR Checkpoint Finetuning')
    parser.add_argument('--pretrain_dir', type=str, required=True)
    parser.add_argument('--dataset_path', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--checkpoint_range', type=str, default='5000-50000')
    parser.add_argument('--checkpoint_step', type=int, default=5000)
    parser.add_argument('--run_name', type=str, default='batch_finetune')
    parser.add_argument('--finetune_mode', type=str, default='partial',
                       choices=['linear', 'nonlinear', 'partial', 'full', 'all'])
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--linear_lr', type=float, default=0.01)
    parser.add_argument('--weight_decay', type=float, default=1e-3)
    parser.add_argument('--use_imagenet_head', action='store_true')
    parser.add_argument('--seed', type=int, default=42)
    
    args = parser.parse_args()
    
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    try:
        start_iter, end_iter = map(int, args.checkpoint_range.split('-'))
    except ValueError:
        print(f"Error: Invalid checkpoint range format: {args.checkpoint_range}")
        return 1
    
    output_dir = Path(args.output_dir) / args.run_name / f'run_{time.strftime("%Y%m%d_%H%M%S")}'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Determine modes to run
    if args.finetune_mode == 'all':
        modes_to_run = ['linear', 'nonlinear', 'partial', 'full']
        for mode in modes_to_run:
            (output_dir / mode).mkdir(exist_ok=True)
    else:
        modes_to_run = [args.finetune_mode]
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    pretrain_dir = Path(args.pretrain_dir)
    if not pretrain_dir.exists():
        print(f"Error: Pretrain directory not found: {pretrain_dir}")
        return 1
    
    # Find checkpoints
    checkpoints = []
    for iter_num in range(start_iter, end_iter + 1, args.checkpoint_step):
        checkpoint_path = pretrain_dir / f'checkpoint-{iter_num}.pth.tar'
        if checkpoint_path.exists():
            checkpoints.append((iter_num, checkpoint_path))
    
    if not checkpoints:
        print(f"Error: No checkpoints found in range {start_iter}-{end_iter}")
        return 1
    
    print(f"Found {len(checkpoints)} checkpoints")
    
    # Setup transforms
    train_transform, test_transform = get_transforms()
    
    # Validate dataset structure
    for split in ['train', 'val', 'test']:
        split_path = Path(args.dataset_path) / split
        if not split_path.exists():
            print(f"Error: {split} directory not found")
            return 1
    
    # Load datasets
    train_dataset = datasets.ImageFolder(Path(args.dataset_path) / 'train', transform=train_transform)
    val_dataset = datasets.ImageFolder(Path(args.dataset_path) / 'val', transform=test_transform)
    test_dataset = datasets.ImageFolder(Path(args.dataset_path) / 'test', transform=test_transform)
    
    num_classes = len(train_dataset.classes)
    class_names = train_dataset.classes
    print(f"Dataset: {num_classes} classes")
    
    data_loaders = {
        'train': DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, 
                           num_workers=4, pin_memory=True),
        'val': DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, 
                         num_workers=4, pin_memory=True),
        'test': DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, 
                          num_workers=4, pin_memory=True)
    }
    
    all_results_by_mode = {}
    
    print(f"Starting evaluation: {', '.join(modes_to_run)}")
    
    for mode in modes_to_run:
        print(f"\nEvaluating {mode} mode")
        
        mode_results = []
        mode_output_dir = output_dir / mode if args.finetune_mode == 'all' else output_dir
        
        for i, (iter_num, checkpoint_path) in enumerate(checkpoints, 1):
            print(f"[{i}/{len(checkpoints)}] Processing checkpoint {iter_num}")
            
            result = evaluate_single_checkpoint(
                checkpoint_path, iter_num, data_loaders, device, args, 
                num_classes, class_names, freeze_mode=mode
            )
            
            mode_results.append(result)
            
            # Save intermediate results
            csv_results = []
            for r in mode_results:
                csv_result = r.copy()
                csv_result.pop('training_results', None)
                csv_result.pop('predictions_data', None)
                csv_results.append(csv_result)
            
            results_df = pd.DataFrame(csv_results)
            results_df.to_csv(mode_output_dir / f'{mode}_batch_results.csv', index=False)
            
            with open(mode_output_dir / f'{mode}_detailed_results_{iter_num}.json', 'w') as f:
                json.dump(result, f, indent=2)
        
        all_results_by_mode[mode] = mode_results
        
        # Process mode results
        successful_results = [r for r in mode_results if r['status'] == 'success']
        
        if successful_results:
            best_result = max(successful_results, key=lambda x: x['test_acc'])
            
            # Save best model predictions
            best_predictions_file = mode_output_dir / f'{mode}_best_model_predictions.json'
            with open(best_predictions_file, 'w') as f:
                json.dump({
                    'checkpoint_iter': best_result['checkpoint_iter'],
                    'checkpoint_path': best_result['checkpoint_path'],
                    'freeze_mode': mode,
                    'test_acc': best_result['test_acc'],
                    'best_val_acc': best_result['best_val_acc'],
                    'predictions_data': best_result['predictions_data']
                }, f, indent=2)
            
            test_accs = [r['test_acc'] for r in successful_results]
            
            # Save mode summary
            mode_summary = {
                'mode': mode,
                'run_name': args.run_name,
                'pretrain_dir': args.pretrain_dir,
                'dataset_path': args.dataset_path,
                'num_checkpoints': len(checkpoints),
                'successful_evals': len(successful_results),
                'failed_evals': len(mode_results) - len(successful_results),
                'best_checkpoint': {
                    'iteration': best_result['checkpoint_iter'],
                    'val_acc': best_result['best_val_acc'],
                    'test_acc': best_result['test_acc']
                },
                'test_acc_stats': {
                    'mean': np.mean(test_accs),
                    'std': np.std(test_accs),
                    'min': min(test_accs),
                    'max': max(test_accs)
                },
                'class_names': class_names,
                'num_classes': num_classes,
                'args': vars(args),
                'timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
            }
            
            with open(mode_output_dir / f'{mode}_summary.json', 'w') as f:
                json.dump(mode_summary, f, indent=2)
    
    # Create combined summary if multiple modes
    if len(modes_to_run) > 1:
        combined_summary = create_combined_summary(all_results_by_mode, class_names)
        
        with open(output_dir / 'combined_summary.json', 'w') as f:
            json.dump(combined_summary, f, indent=2)
        
        if 'comparison' in combined_summary:
            comp = combined_summary['comparison']
            print(f"\nBest mode: {comp['best_mode']} ({comp['best_test_acc']:.1f}%)")
    
    print(f"\nResults saved to: {output_dir}")
    return 0


if __name__ == "__main__":
    exit(main())