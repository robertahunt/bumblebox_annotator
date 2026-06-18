#!/usr/bin/env python3
"""
Create dummy COCO dataset for testing Detectron2 training
"""

import json
import numpy as np
from PIL import Image, ImageDraw
from pathlib import Path
import random


def create_dummy_image_with_annotations(image_id, width=640, height=480, num_bees=3):
    """Create a dummy image with simple bee-like objects"""
    # Create base image with random background
    img = Image.new('RGB', (width, height), color=(
        random.randint(100, 200),
        random.randint(150, 255),
        random.randint(100, 200)
    ))
    draw = ImageDraw.Draw(img)
    
    annotations = []
    
    for i in range(num_bees):
        # Random position and size
        x = random.randint(50, width - 100)
        y = random.randint(50, height - 100)
        w = random.randint(30, 80)
        h = random.randint(20, 60)
        
        # Draw ellipse (bee-like shape)
        bbox = [x, y, x + w, y + h]
        draw.ellipse(bbox, fill=(255, 200, 0), outline=(0, 0, 0), width=2)
        
        # Add some stripes
        for j in range(3):
            stripe_y = y + (h * j // 3)
            draw.line([(x, stripe_y), (x + w, stripe_y)], fill=(50, 50, 50), width=3)
        
        # Create segmentation (simple polygon around bbox)
        segmentation = [
            x, y,
            x + w, y,
            x + w, y + h,
            x, y + h
        ]
        
        # Create annotation
        annotation = {
            "id": image_id * 100 + i,
            "image_id": image_id,
            "category_id": 1,  # Will be remapped to 0 by our loader
            "bbox": [x, y, w, h],
            "area": w * h,
            "segmentation": [segmentation],
            "iscrowd": 0
        }
        annotations.append(annotation)
    
    return img, annotations


def create_dummy_dataset(output_dir, num_train=20, num_val=5):
    """
    Create a dummy dataset for testing
    
    Args:
        output_dir: Directory to create dataset in
        num_train: Number of training images
        num_val: Number of validation images
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Creating dummy dataset in {output_dir}")
    print(f"  - {num_train} training images")
    print(f"  - {num_val} validation images")
    
    # Create COCO structure
    coco_train = {
        "images": [],
        "annotations": [],
        "categories": [
            {
                "id": 1,
                "name": "bee",
                "supercategory": "insect"
            }
        ]
    }
    
    coco_val = {
        "images": [],
        "annotations": [],
        "categories": [
            {
                "id": 1,
                "name": "bee",
                "supercategory": "insect"
            }
        ]
    }
    
    # Generate training images
    print("\nGenerating training images...")
    for i in range(num_train):
        image_id = i + 1
        filename = f"frame_{image_id:04d}.jpg"
        filepath = output_dir / filename
        
        # Create image with annotations
        img, annotations = create_dummy_image_with_annotations(image_id)
        
        # Save image
        img.save(filepath, quality=90)
        
        # Add to COCO
        coco_train["images"].append({
            "id": image_id,
            "file_name": filename,
            "width": img.width,
            "height": img.height
        })
        coco_train["annotations"].extend(annotations)
        
        if (i + 1) % 5 == 0:
            print(f"  Created {i + 1}/{num_train} images")
    
    # Generate validation images
    print("\nGenerating validation images...")
    for i in range(num_val):
        image_id = num_train + i + 1
        filename = f"frame_{image_id:04d}.jpg"
        filepath = output_dir / filename
        
        # Create image with annotations
        img, annotations = create_dummy_image_with_annotations(image_id)
        
        # Save image
        img.save(filepath, quality=90)
        
        # Add to COCO
        coco_val["images"].append({
            "id": image_id,
            "file_name": filename,
            "width": img.width,
            "height": img.height
        })
        coco_val["annotations"].extend(annotations)
    
    print(f"  Created {num_val}/{num_val} images")
    
    # Save COCO JSON files
    train_json = output_dir / "coco_train.json"
    val_json = output_dir / "coco_val.json"
    
    print(f"\nWriting COCO annotations...")
    with open(train_json, 'w') as f:
        json.dump(coco_train, f, indent=2)
    print(f"  Saved: {train_json}")
    
    with open(val_json, 'w') as f:
        json.dump(coco_val, f, indent=2)
    print(f"  Saved: {val_json}")
    
    # Print summary
    print("\n" + "="*60)
    print("DATASET CREATED SUCCESSFULLY")
    print("="*60)
    print(f"Location: {output_dir}")
    print(f"Training: {len(coco_train['images'])} images, {len(coco_train['annotations'])} annotations")
    print(f"Validation: {len(coco_val['images'])} images, {len(coco_val['annotations'])} annotations")
    print("\nTo test training, run:")
    print(f"  python test_training.py {output_dir}")
    
    return output_dir


def main():
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python create_dummy_data.py <output_directory>")
        print()
        print("Example: python create_dummy_data.py projects/dummy_test")
        print()
        print("This will create:")
        print("  - Dummy images with bee-like objects")
        print("  - coco_train.json (20 training images)")
        print("  - coco_val.json (5 validation images)")
        sys.exit(1)
    
    output_dir = sys.argv[1]
    num_train = int(sys.argv[2]) if len(sys.argv) > 2 else 20
    num_val = int(sys.argv[3]) if len(sys.argv) > 3 else 5
    
    create_dummy_dataset(output_dir, num_train, num_val)


if __name__ == "__main__":
    main()
