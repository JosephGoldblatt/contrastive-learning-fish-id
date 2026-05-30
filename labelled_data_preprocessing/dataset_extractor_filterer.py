#!/usr/bin/env python3

"""
Extracts bounding box cutouts from labelled frames using per frame .txt annotation files and stores these cutouts,
in class separated sub-directories, in a new directory. Filters out classes with fewer than three cutouts.
Visualizes the composition of this dataset as a pie charts, representing the dataset before and after filtering.

Usage: python3 dataset_extractor_filterer.py <input_dir> <output_dir> <csv_mapping> [--min-images N]
"""

import os
import cv2
import pandas as pd #type: ignore
import matplotlib.pyplot as plt #type: ignore
import matplotlib.patches as patches #type: ignore
import numpy as np
import argparse
import math
import colorsys
import shutil
from pathlib import Path
from collections import defaultdict


class DatasetProcessor:
    def __init__(self, input_dir, output_dir, csv_mapping_path, min_images_per_class=3):
        """Initialize the dataset processor."""
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.csv_mapping_path = csv_mapping_path
        self.min_images_per_class = min_images_per_class
        self.class_mapping = {}
        self.allowed_classes = set()
        self.allowed_videos = [
            'WILD', 'Pondo', '17-04_NSC-S_012', '17-04_NSC-S_019', '17-04_NSC-S_024',
            '17-04_NSC-S_036', '17-04_NSC-S_038', '17-04_NSC-S_041', '17-04_NSC-S_043',
            '17-04_NSC-S_050', '17-04_NSC-S_051', '17-04_NSC-S_053', '17-04_NSC-S_055',
            '17-04_NSC-S_058', '17-04_NSC-S_059', '17-04_NSC-S_061', '17-04_NSC-S_062',
            '17-04_NSC-S_067', '17-04_NSC-S_068', '17-04_NSC-S_082'
        ]
        
    def load_class_mapping(self):
        """Load class mapping from CSV file."""
        try:
            df = pd.read_csv(self.csv_mapping_path)
            for _, row in df.iterrows():
                species_name = row['species']
                class_number = int(row['number'])
                self.class_mapping[str(class_number)] = species_name
                self.allowed_classes.add(class_number)
        except Exception as e:
            raise ValueError(f"Error loading class mapping: {e}")
    
    def extract_cutouts(self):
        """Extract bounding box cutouts from annotated images."""
        class_cutout_counts = defaultdict(int)
        temp_output_dir = self.output_dir / "temp_cutouts"
        temp_output_dir.mkdir(parents=True, exist_ok=True)
        
        total_cutouts = 0
        
        for image_file in self.input_dir.rglob("*.jpg"):
            annotation_file = image_file.with_suffix('.txt')
            if not annotation_file.exists():
                continue
            
            # Check if from allowed videos
            if not any(name in str(image_file) for name in self.allowed_videos):
                continue
            
            image = cv2.imread(str(image_file))
            if image is None:
                continue
            
            img_height, img_width = image.shape[:2]
            cutout_counter = 1
            
            with open(annotation_file, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 5:
                        continue
                    
                    class_id = int(parts[0])
                    if class_id == 0 or class_id not in self.allowed_classes:
                        continue
                    
                    # Convert YOLO format to pixel coordinates
                    x_center, y_center, width, height = map(float, parts[1:5])
                    x_center *= img_width
                    y_center *= img_height
                    width *= img_width
                    height *= img_height
                    
                    x_min = max(0, int(x_center - width/2))
                    y_min = max(0, int(y_center - height/2))
                    x_max = min(img_width, int(x_center + width/2))
                    y_max = min(img_height, int(y_center + height/2))
                    
                    cutout = image[y_min:y_max, x_min:x_max]
                    if cutout.shape[0] > 0 and cutout.shape[1] > 0:
                        class_dir = temp_output_dir / str(class_id)
                        class_dir.mkdir(exist_ok=True)
                        
                        output_path = class_dir / f"{image_file.stem}_cutout_{cutout_counter}.jpg"
                        cv2.imwrite(str(output_path), cutout)
                        
                        class_cutout_counts[str(class_id)] += 1
                        cutout_counter += 1
                        total_cutouts += 1
        
        print(f"Extracted {total_cutouts} cutouts from {len(class_cutout_counts)} classes")
        return temp_output_dir, class_cutout_counts
    
    def filter_by_class_size(self, temp_dir, class_counts):
        """Filter classes by minimum image count."""
        final_output_dir = self.output_dir
        final_output_dir.mkdir(parents=True, exist_ok=True)
        
        filtered_classes = 0
        filtered_images = 0
        
        for class_id, count in class_counts.items():
            if count >= self.min_images_per_class:
                source_class_dir = temp_dir / class_id
                target_class_dir = final_output_dir / class_id
                target_class_dir.mkdir(exist_ok=True)
                
                for image_file in source_class_dir.glob("*.jpg"):
                    target_path = target_class_dir / image_file.name
                    img = cv2.imread(str(image_file))
                    cv2.imwrite(str(target_path), img)
                    filtered_images += 1
                
                filtered_classes += 1
        
        # Clean up temp directory
        shutil.rmtree(temp_dir)
        
        print(f"Filtered to {filtered_classes} classes, {filtered_images} images")
        return final_output_dir, filtered_classes, filtered_images
    
    def count_dataset_classes(self, dataset_dir):
        """Count images per class in dataset directory."""
        dataset_path = Path(dataset_dir)
        class_counts = {}
        
        for class_dir in dataset_path.iterdir():
            if class_dir.is_dir():
                class_name = class_dir.name
                image_count = len(list(class_dir.glob("*.jpg")))
                if image_count > 0:
                    class_counts[class_name] = image_count
        
        return class_counts
    
    def generate_colors(self, n_colors, seed=42):
        """Generate distinct colors for visualization."""
        np.random.seed(seed)
        colors = []
        
        for i in range(n_colors):
            hue = np.random.random()
            saturation = np.random.uniform(0.4, 1.0)
            value = np.random.uniform(0.5, 0.9)
            rgb = colorsys.hsv_to_rgb(hue, saturation, value)
            colors.append(rgb)
        
        return colors
    
    def create_visualization(self, temp_counts, final_dir):
        """Create visualization comparing original and filtered datasets."""
        # Setup matplotlib
        plt.rcParams['font.family'] = 'sans-serif'
        plt.rcParams['font.sans-serif'] = ['cmss10']
        
        final_counts = self.count_dataset_classes(final_dir)
        original_counts = temp_counts
        
        # Get all unique classes
        all_classes = sorted(set(list(original_counts.keys()) + list(final_counts.keys())), key=int)
        
        # Generate colors
        colors = self.generate_colors(len(all_classes), seed=42)
        np.random.seed(42)
        shuffled_indices = np.random.permutation(len(all_classes))
        class_colors = {all_classes[i]: colors[shuffled_indices[i]] for i in range(len(all_classes))}
        
        # Create legend and pie charts
        self.create_legend(all_classes, class_colors)
        self.create_pie_charts(original_counts, final_counts, class_colors, all_classes)
    
    def create_legend(self, all_classes, class_colors):
        """Create a separate legend file."""
        n_classes = len(all_classes)
        cols_per_row = min(8, n_classes)
        n_rows = math.ceil(n_classes / cols_per_row)
        
        fig, ax = plt.subplots(figsize=(max(12, cols_per_row * 1.5), n_rows * 2))
        ax.axis('off')
        ax.set_xlim(0, cols_per_row)
        ax.set_ylim(0, n_rows * 2)
        
        for i, cls in enumerate(all_classes):
            row = i // cols_per_row
            col = i % cols_per_row
            
            x = col + 0.5
            y = (n_rows - row - 1) * 2 + 1.5
            
            # Draw colored rectangle
            rect = patches.Rectangle((x - 0.3, y - 0.1), 0.6, 0.2,
                                    facecolor=class_colors[cls],
                                    edgecolor='black', linewidth=0.5)
            ax.add_patch(rect)
            
            # Add class number
            ax.text(x, y, cls, ha='center', va='center',
                   fontsize=8, fontweight='normal')
            
            # Add species name
            species_name = self.class_mapping.get(cls, f"Class {cls}")
            if len(species_name) > 25:
                species_name = species_name[:22] + "..."
            
            ax.text(x, y - 0.6, species_name, ha='center', va='top',
                   fontsize=12, rotation=0, wrap=True, fontweight='normal')
        
        plt.tight_layout()
        plots_dir = Path("plots")
        plots_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(plots_dir/"legend.png", dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
    
    def create_pie_charts(self, original_counts, final_counts, class_colors, all_classes):
        """Create pie chart visualization."""
        fig, ax = plt.subplots(figsize=(18, 10))
        ax.set_xlim(-4, 4)
        ax.set_ylim(-3, 3)
        ax.axis('off')
        
        # Calculate proportional sizes
        original_total = sum(original_counts.values())
        filtered_total = sum(final_counts.values())
        max_total = max(original_total, filtered_total) if max(original_total, filtered_total) > 0 else 1
        
        original_scale = math.sqrt(original_total / max_total) * 1.8 if original_total > 0 else 0
        filtered_scale = math.sqrt(filtered_total / max_total) * 1.8 if filtered_total > 0 else 0
        
        # Original pie chart
        original_data = [(original_counts[cls], class_colors[cls]) 
                        for cls in all_classes if cls in original_counts]
        if original_data:
            original_data.sort(key=lambda x: x[0], reverse=True)
            original_values, original_colors = zip(*original_data)
            ax.pie(original_values, colors=original_colors,
                  startangle=90, radius=original_scale, center=(-2, 0))
        
        # Filtered pie chart
        filtered_data = [(final_counts[cls], class_colors[cls])
                        for cls in all_classes if cls in final_counts]
        if filtered_data:
            filtered_data.sort(key=lambda x: x[0], reverse=True)
            filtered_values, filtered_colors = zip(*filtered_data)
            ax.pie(filtered_values, colors=filtered_colors,
                  startangle=90, radius=filtered_scale, center=(2, 0))
        
        # Add titles
        ax.text(-2, 2.1, f'Original Dataset\n({original_total} images, {len(original_counts)} classes)',
               ha='center', va='center', fontsize=30, fontweight='normal')
        ax.text(2, 2.1, f'Filtered Dataset\n({filtered_total} images, {len(final_counts)} classes)',
               ha='center', va='center', fontsize=30, fontweight='normal')
        
        plt.tight_layout()
        plots_dir = Path("plots")
        plots_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(plots_dir/"dataset_composition.png", dpi=300, bbox_inches='tight', facecolor='white')
        plt.show()
    
    def process(self):
        """Run the complete processing pipeline."""
        # Load class mapping
        self.load_class_mapping()
        
        # Extract cutouts
        temp_dir, class_counts = self.extract_cutouts()
        
        # Filter by class size
        final_dir, n_classes, n_images = self.filter_by_class_size(temp_dir, class_counts)
        
        # Create visualization
        self.create_visualization(class_counts, final_dir)
        
        print(f"Processing complete: {final_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Extract cutouts, filter by class size, and visualize dataset"
    )
    
    parser.add_argument("input_dir", help="Input directory with images and YOLO annotations")
    parser.add_argument("output_dir", help="Output directory for filtered dataset")
    parser.add_argument("csv_mapping", help="CSV file with species,number columns")
    parser.add_argument("--min-images", type=int, default=3,
                       help="Minimum images per class (default: 3)")
    
    args = parser.parse_args()
    
    try:
        processor = DatasetProcessor(
            args.input_dir,
            args.output_dir,
            args.csv_mapping,
            args.min_images
        )
        processor.process()
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())