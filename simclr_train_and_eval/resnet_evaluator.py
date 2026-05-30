#!/usr/bin/env python

"""
Performs linear evaluation of a baseline supervised ImageNet-trained ResNet-50 model.

Usage: python resnet_evaluator.py --data-dir <dir> --output-dir <dir> --run-name <name>
"""

import os
import time
import argparse
import torch #type:ignore
import torch.nn as nn #type: ignore
import torch.optim as optim #type: ignore
from torch.utils.data import DataLoader #type: ignore
from torchvision import datasets, transforms, models #type: ignore
import pandas as pd #type: ignore
from pathlib import Path
import json
from datetime import datetime


def get_transforms():
    """Get train and test transforms."""
    train_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    return train_transform, test_transform


def create_model(num_classes, freeze_backbone=True):
    """Create ResNet50 model with optional frozen backbone."""
    model = models.resnet50(pretrained=True)
    
    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False
    
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    
    return model


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


def main():
    parser = argparse.ArgumentParser(description='ResNet baseline linear evaluation')
    
    parser.add_argument('--data-dir', type=str, required=True,
                        help='Path to dataset directory')
    parser.add_argument('--output-dir', type=str, required=True,
                        help='Path to output directory')
    parser.add_argument('--run-name', type=str, required=True,
                        help='Name for this run')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    
    args = parser.parse_args()
    
    # Set random seed
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    # Create output directory
    output_dir = Path(args.output_dir) / args.run_name / 'linear_eval'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Load datasets
    train_transform, test_transform = get_transforms()
    
    train_dataset = datasets.ImageFolder(
        os.path.join(args.data_dir, 'train'), 
        transform=train_transform
    )
    val_dataset = datasets.ImageFolder(
        os.path.join(args.data_dir, 'val'), 
        transform=test_transform
    )
    test_dataset = datasets.ImageFolder(
        os.path.join(args.data_dir, 'test'), 
        transform=test_transform
    )
    
    num_classes = len(train_dataset.classes)
    print(f"Dataset: {num_classes} classes")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size,
        shuffle=True, 
        num_workers=args.num_workers
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size,
        shuffle=False, 
        num_workers=args.num_workers
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=args.batch_size,
        shuffle=False, 
        num_workers=args.num_workers
    )
    
    # Create model with frozen backbone
    model = create_model(num_classes, freeze_backbone=True)
    model = model.to(device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.fc.parameters(), 
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    # Training loop
    results = []
    best_val_acc = 0
    
    print(f"Starting linear evaluation: {args.epochs} epochs")
    
    for epoch in range(args.epochs):
        start_time = time.time()
        
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device
        )
        
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        
        epoch_time = time.time() - start_time
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), output_dir / 'best_model.pth')
        
        results.append({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_loss': val_loss,
            'val_acc': val_acc,
            'epoch_time': epoch_time
        })
        
        if (epoch + 1) % 10 == 0 or epoch == args.epochs - 1:
            print(f"Epoch {epoch+1}: Train={train_acc:.1f}% Val={val_acc:.1f}%")
    
    # Final test evaluation
    model.load_state_dict(torch.load(output_dir / 'best_model.pth'))
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    
    print(f"Final: Val={best_val_acc:.1f}% Test={test_acc:.1f}%")
    
    # Save results
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_dir / 'training_results.csv', index=False)
    
    # Save summary
    summary = {
        'method': 'Linear Evaluation',
        'num_classes': num_classes,
        'classes': train_dataset.classes,
        'train_size': len(train_dataset),
        'val_size': len(val_dataset),
        'test_size': len(test_dataset),
        'epochs': args.epochs,
        'lr': args.lr,
        'weight_decay': args.weight_decay,
        'batch_size': args.batch_size,
        'best_val_acc': best_val_acc,
        'test_acc': test_acc,
        'total_params': sum(p.numel() for p in model.parameters()),
        'trainable_params': sum(p.numel() for p in model.parameters() if p.requires_grad),
        'seed': args.seed,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    with open(output_dir / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"Results saved to {output_dir}")


if __name__ == "__main__":
    main()