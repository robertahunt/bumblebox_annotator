#!/usr/bin/env python3
"""Check validation dataset size for YOLO SAHI training"""

import sys
from pathlib import Path
import yaml

def check_validation_dataset(project_path):
    """Check if validation dataset is large enough for batch plots"""
    project = Path(project_path)
    yolo_sahi = project / 'yolo_sahi'
    
    # Check if dataset exists
    if not yolo_sahi.exists():
        print(f"❌ YOLO SAHI dataset not found at: {yolo_sahi}")
        return False
    
    # Load dataset config
    dataset_yaml = yolo_sahi / 'dataset.yaml'
    if not dataset_yaml.exists():
        print(f"❌ dataset.yaml not found")
        return False
    
    with open(dataset_yaml) as f:
        config = yaml.safe_load(f)
    
    # Count validation images
    val_images_dir = yolo_sahi / 'images' / 'val'
    if not val_images_dir.exists():
        print(f"❌ Validation images directory not found: {val_images_dir}")
        return False
    
    val_images = list(val_images_dir.glob('*.*'))
    num_val = len(val_images)
    
    print(f"\n=== Validation Dataset Check ===")
    print(f"Project: {project.name}")
    print(f"Validation images: {num_val}")
    
    # Check if enough for batch plots (need at least 3 batches)
    # Typical batch size is 8
    min_images_for_plots = 8 * 3  # 3 batches of size 8
    
    if num_val >= min_images_for_plots:
        print(f"✓ Should generate validation batch plots (need >={min_images_for_plots})")
        return True
    else:
        print(f"⚠ May NOT generate validation batch plots!")
        print(f"  - Have: {num_val} images")
        print(f"  - Need: At least {min_images_for_plots} images (for 3 batches of size 8)")
        print(f"  - Or reduce batch size to {num_val // 3} or lower")
        return False

if __name__ == '__main__':
    if len(sys.argv) > 1:
        project_path = sys.argv[1]
    else:
        # Default to test project
        project_path = Path(__file__).parent / 'projects' / 'test'
    
    check_validation_dataset(project_path)
