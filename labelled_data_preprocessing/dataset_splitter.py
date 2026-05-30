#!/usr/bin/env python3

"""
Splits class-organized dataset into train/test/val subdirectories with fixed 70/15/15 ratios.
Ensuring that each class has >=min-per-split samples in each split. 

Usage: python3 dataset_splitter.py <source_dir> <target_dir> [--min-per-split N]
"""

import os
import shutil
import random
from pathlib import Path
from collections import defaultdict
import argparse


def split_dataset(source_dir, target_dir, min_per_split=1):
    """Split dataset into train/test/val maintaining class distribution."""
    # Fixed split ratios
    train_ratio = 0.7
    test_ratio = 0.15
    val_ratio = 0.15
    seed = 42
    
    source_path = Path(source_dir)
    target_path = Path(target_dir)
    
    if not source_path.exists():
        raise ValueError(f"Source directory does not exist: {source_dir}")
    
    # Set random seed for reproducibility
    random.seed(seed)
    
    # Create target directory structure
    splits = ['train', 'test', 'val']
    for split in splits:
        (target_path / split).mkdir(parents=True, exist_ok=True)
    
    # Common image extensions
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'}
    
    # Track statistics
    total_images = 0
    split_counts = defaultdict(int)
    excluded_classes = []
    
    # Collect class information
    class_info = []
    for class_dir in source_path.iterdir():
        if not class_dir.is_dir():
            continue
        
        class_name = class_dir.name
        
        # Collect all valid images
        class_images = [f for f in class_dir.iterdir() 
                       if f.suffix.lower() in image_extensions]
        
        if not class_images:
            continue
        
        class_info.append({
            'name': class_name,
            'images': class_images,
            'count': len(class_images)
        })
    
    # Filter classes with sufficient samples
    min_samples_needed = min_per_split * 3
    valid_classes = []
    
    for class_data in class_info:
        if class_data['count'] >= min_samples_needed:
            valid_classes.append(class_data)
        else:
            excluded_classes.append(class_data['name'])
    
    if not valid_classes:
        raise ValueError(f"No classes have enough samples (need ≥{min_samples_needed} per class)")
    
    # Process each valid class
    for class_data in valid_classes:
        class_name = class_data['name']
        class_images = class_data['images']
        num_images = len(class_images)
        
        # Shuffle images
        random.shuffle(class_images)
        
        # Reserve minimum samples for each split
        reserved_train = class_images[:min_per_split]
        reserved_test = class_images[min_per_split:min_per_split*2]
        reserved_val = class_images[min_per_split*2:min_per_split*3]
        
        # Distribute remaining images by ratios
        remaining_images = class_images[min_per_split*3:]
        remaining_count = len(remaining_images)
        
        if remaining_count > 0:
            additional_train = int(remaining_count * train_ratio)
            additional_test = int(remaining_count * test_ratio)
            additional_val = remaining_count - additional_train - additional_test
            
            idx = 0
            train_additional = remaining_images[idx:idx + additional_train]
            idx += additional_train
            test_additional = remaining_images[idx:idx + additional_test]
            idx += additional_test
            val_additional = remaining_images[idx:]
        else:
            train_additional = []
            test_additional = []
            val_additional = []
        
        # Combine reserved and additional samples
        final_train = reserved_train + train_additional
        final_test = reserved_test + test_additional
        final_val = reserved_val + val_additional
        
        # Create class directories and copy images
        for split in splits:
            (target_path / split / class_name).mkdir(exist_ok=True)
        
        for img in final_train:
            shutil.copy2(img, target_path / 'train' / class_name / img.name)
        split_counts['train'] += len(final_train)
        
        for img in final_test:
            shutil.copy2(img, target_path / 'test' / class_name / img.name)
        split_counts['test'] += len(final_test)
        
        for img in final_val:
            shutil.copy2(img, target_path / 'val' / class_name / img.name)
        split_counts['val'] += len(final_val)
        
        total_images += num_images
    
    # Print summary
    print(f"Split complete: {total_images} images, {len(valid_classes)} classes")
    print(f"Distribution: Train={split_counts['train']}, Test={split_counts['test']}, Val={split_counts['val']}")
    if excluded_classes:
        print(f"Excluded {len(excluded_classes)} classes with insufficient samples")


def main():
    parser = argparse.ArgumentParser(
        description="Split dataset into train/test/val with 70/15/15 ratios"
    )
    
    parser.add_argument("source_dir", help="Source dataset directory")
    parser.add_argument("target_dir", help="Target directory for split dataset")
    parser.add_argument("--min-per-split", type=int, default=1,
                       help="Minimum samples per class per split (default: 1)")
    
    args = parser.parse_args()
    
    try:
        split_dataset(args.source_dir, args.target_dir, args.min_per_split)
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())