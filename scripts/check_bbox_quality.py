#!/usr/bin/env python3
"""
Check the quality of bounding box annotations in training data.
Visualizes bounding boxes overlaid on images to verify they're correct.
"""

import argparse
import json
import sys
import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def visualize_coco_annotations(json_path, output_dir=None, max_images=10):
    """
    Visualize COCO format bounding box annotations
    
    Args:
        json_path: Path to COCO JSON file
        output_dir: Directory to save visualizations (if None, displays only)
        max_images: Maximum number of images to visualize
    """    
    # Load COCO JSON
    with open(json_path, 'r') as f:
        coco_data = json.load(f)
    
    # Create image id to file mapping
    image_map = {img['id']: img for img in coco_data['images']}
    
    # Create lookup for annotations by image_id
    annotations_by_image = {}
    for ann in coco_data['annotations']:
        image_id = ann['image_id']
        if image_id not in annotations_by_image:
            annotations_by_image[image_id] = []
        annotations_by_image[image_id].append(ann)
    
    print(f"Found {len(image_map)} images with {len(coco_data['annotations'])} annotations")
    
    # Analyze bounding box statistics
    bbox_widths = []
    bbox_heights = []
    bbox_areas = []
    bbox_aspect_ratios = []
    
    for ann in coco_data['annotations']:
        if 'bbox' in ann:
            x, y, w, h = ann['bbox']
            bbox_widths.append(w)
            bbox_heights.append(h)
            bbox_areas.append(w * h)
            if h > 0:
                bbox_aspect_ratios.append(w / h)
    
    print(f"\nBounding Box Statistics:")
    print(f"  Width:  min={min(bbox_widths):.1f}, max={max(bbox_widths):.1f}, mean={np.mean(bbox_widths):.1f}")
    print(f"  Height: min={min(bbox_heights):.1f}, max={max(bbox_heights):.1f}, mean={np.mean(bbox_heights):.1f}")
    print(f"  Area:   min={min(bbox_areas):.1f}, max={max(bbox_areas):.1f}, mean={np.mean(bbox_areas):.1f}")
    print(f"  Aspect Ratio (W/H): min={min(bbox_aspect_ratios):.2f}, max={max(bbox_aspect_ratios):.2f}, mean={np.mean(bbox_aspect_ratios):.2f}")
    
    # Check for anomalies
    anomalies = []
    for ann in coco_data['annotations']:
        if 'bbox' in ann:
            x, y, w, h = ann['bbox']
            issues = []
            
            # Check for zero or negative dimensions
            if w <= 0:
                issues.append(f"width={w}")
            if h <= 0:
                issues.append(f"height={h}")
            
            # Check for extremely small boxes (likely errors)
            if w < 5 or h < 5:
                issues.append(f"very small: {w}x{h}")
            
            # Check for extreme aspect ratios
            aspect_ratio = w / h if h > 0 else float('inf')
            if aspect_ratio > 5 or aspect_ratio < 0.2:
                issues.append(f"extreme aspect ratio: {aspect_ratio:.2f}")
            
            # Check if bbox is outside image bounds
            img = image_map.get(ann['image_id'])
            if img:
                if x < 0 or y < 0:
                    issues.append(f"negative coords: x={x}, y={y}")
                if x + w > img['width'] or y + h > img['height']:
                    issues.append(f"exceeds image bounds: {img['width']}x{img['height']}")
            
            if issues:
                anomalies.append({
                    'annotation_id': ann['id'],
                    'image_id': ann['image_id'],
                    'bbox': ann['bbox'],
                    'issues': issues
                })
    
    if anomalies:
        print(f"\n⚠️  Found {len(anomalies)} anomalous bounding boxes:")
        for i, anom in enumerate(anomalies[:20]):  # Show first 20
            print(f"  {i+1}. Ann ID {anom['annotation_id']}, Img ID {anom['image_id']}")
            print(f"     BBox: {anom['bbox']}")
            print(f"     Issues: {', '.join(anom['issues'])}")
    else:
        print("\n✓ No obvious anomalies detected in bounding boxes")
    
    # Visualize sample images
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    
    json_dir = Path(json_path).parent
    images_to_viz = list(annotations_by_image.keys())[:max_images]
    
    print(f"\nVisualizing {len(images_to_viz)} sample images...")
    
    for idx, image_id in enumerate(images_to_viz):
        img_info = image_map[image_id]
        img_path = Path(img_info['file_name'])
        
        # Try to find the image
        if not img_path.is_absolute():
            # Try relative to JSON file
            img_path = json_dir / img_path
            if not img_path.exists():
                # Try going up to find images
                img_path = json_dir.parent / img_info['file_name']
        
        if not img_path.exists():
            print(f"  Warning: Image not found: {img_path}")
            continue
        
        # Load image
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"  Warning: Failed to load image: {img_path}")
            continue
        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Create figure
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        ax.imshow(img)
        
        # Draw bounding boxes
        annotations = annotations_by_image.get(image_id, [])
        for ann in annotations:
            if 'bbox' in ann:
                x, y, w, h = ann['bbox']
                rect = patches.Rectangle(
                    (x, y), w, h,
                    linewidth=2, edgecolor='lime', facecolor='none'
                )
                ax.add_patch(rect)
                
                # Add annotation ID label
                ax.text(x, y - 5, f"ID:{ann['id']}", 
                       color='lime', fontsize=8, 
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.5))
        
        ax.set_title(f"Image {image_id}: {img_path.name} ({len(annotations)} bees)")
        ax.axis('off')
        
        if output_dir:
            out_file = output_dir / f"bbox_check_{idx:03d}_{img_path.stem}.png"
            plt.savefig(out_file, bbox_inches='tight', dpi=150)
            print(f"  Saved: {out_file}")
            plt.close()
        else:
            plt.show()
    
    return anomalies


def main():
    parser = argparse.ArgumentParser(
        description='Check quality of bounding box annotations in COCO format'
    )
    parser.add_argument('--json', required=True, help='Path to COCO JSON file')
    parser.add_argument('--output', help='Output directory for visualizations')
    parser.add_argument('--max-images', type=int, default=10, 
                       help='Maximum number of images to visualize (default: 10)')
    
    args = parser.parse_args()
    
    anomalies = visualize_coco_annotations(
        args.json,
        output_dir=args.output,
        max_images=args.max_images
    )
    
    if anomalies:
        print(f"\n⚠️  Total anomalies found: {len(anomalies)}")
        print("Review the visualizations to verify bounding box accuracy.")
    else:
        print("\n✓ All bounding boxes appear valid!")


if __name__ == '__main__':
    main()
