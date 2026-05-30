#!/usr/bin/env python3

"""
Splits unlabelled image dataset into train/val directories for SimCLR training, 
ensuring dataset is structured for PyTorch compatibility.

Usage: python3 unlabelled_dataset_formatter.py <source_dir> <output_dir> [train_ratio]
"""

import sys
import shutil
import random
from pathlib import Path


def split_images(source_dir, output_dir, train_ratio=0.8):
    """Split images from source directory into train/val sets."""
    source_path = Path(source_dir)
    output_path = Path(output_dir)
    
    if not source_path.exists():
        raise FileNotFoundError(f"Source directory '{source_dir}' does not exist")
    
    # Common image extensions
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.gif', '.webp'}
    
    # Get all image files
    image_files = [f for f in source_path.iterdir() 
                   if f.is_file() and f.suffix.lower() in image_extensions]
    
    total_images = len(image_files)
    if total_images == 0:
        print("No images found in source directory")
        return
    
    # Shuffle with fixed seed for reproducibility
    random.seed(42)
    random.shuffle(image_files)
    
    # Calculate split
    train_count = int(total_images * train_ratio)
    val_count = total_images - train_count
    
    # Create output directories
    train_dir = output_path / "train" / "images"
    val_dir = output_path / "val" / "images"
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)
    
    # Copy files to respective directories
    for img_file in image_files[:train_count]:
        shutil.copy2(img_file, train_dir / img_file.name)
    
    for img_file in image_files[train_count:]:
        shutil.copy2(img_file, val_dir / img_file.name)
    
    print(f"Split complete: {train_count} train, {val_count} val images")


def main():
    if len(sys.argv) < 3:
        print("Usage: python3 format_dataset.py <source_dir> <output_dir> [train_ratio]")
        sys.exit(1)
    
    source_dir = sys.argv[1]
    output_dir = sys.argv[2]
    train_ratio = float(sys.argv[3]) if len(sys.argv) > 3 else 0.8
    
    try:
        split_images(source_dir, output_dir, train_ratio)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()