#!/usr/bin/env python3
"""
Visualize YOLO format annotations on images for manual inspection
"""
import argparse
import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def visualize_yolo_dataset(yolo_dir, output_dir, split='train', max_images=50):
    """
    Visualize YOLO annotations overlaid on images
    
    Args:
        yolo_dir: Path to YOLO format directory (contains train/val subdirs)
        output_dir: Where to save visualizations
        split: 'train' or 'val'
        max_images: Maximum number of images to visualize
    """
    yolo_dir = Path(yolo_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    images_dir = yolo_dir / split / 'images'
    labels_dir = yolo_dir / split / 'labels'
    
    if not images_dir.exists():
        print(f"Images directory not found: {images_dir}")
        return
    
    if not labels_dir.exists():
        print(f"Labels directory not found: {labels_dir}")
        return
    
    # Get all images
    image_files = sorted(list(images_dir.glob('*.jpg')))
    
    if not image_files:
        print(f"No images found in {images_dir}")
        return
    
    print(f"Found {len(image_files)} images in {split} split")
    print(f"Visualizing up to {max_images} images...")
    
    issues_found = []
    
    for idx, img_path in enumerate(image_files[:max_images]):
        # Load image
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"  Warning: Failed to load {img_path.name}")
            continue
        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_height, img_width = img.shape[:2]
        
        # Load corresponding label file
        label_path = labels_dir / f"{img_path.stem}.txt"
        
        annotations = []
        if label_path.exists():
            with open(label_path, 'r') as f:
                for line in f:
                    if line.strip():
                        parts = line.strip().split()
                        if len(parts) >= 5:
                            class_id = int(parts[0])
                            x_center = float(parts[1])
                            y_center = float(parts[2])
                            width = float(parts[3])
                            height = float(parts[4])
                            annotations.append({
                                'class_id': class_id,
                                'x_center': x_center,
                                'y_center': y_center,
                                'width': width,
                                'height': height
                            })
        
        # Create visualization
        fig, ax = plt.subplots(1, 1, figsize=(14, 10))
        ax.imshow(img)
        
        # Draw bounding boxes
        for i, ann in enumerate(annotations):
            # Convert YOLO format (normalized) to pixel coordinates
            x_center_px = ann['x_center'] * img_width
            y_center_px = ann['y_center'] * img_height
            w_px = ann['width'] * img_width
            h_px = ann['height'] * img_height
            
            # Top-left corner
            x = x_center_px - w_px / 2
            y = y_center_px - h_px / 2
            
            # Check for anomalies
            is_anomaly = False
            if x < 0 or y < 0 or x + w_px > img_width or y + h_px > img_height:
                is_anomaly = True
                issues_found.append({
                    'image': img_path.name,
                    'annotation': i,
                    'issue': 'bbox_outside_image',
                    'bbox': [x, y, w_px, h_px],
                    'image_size': [img_width, img_height]
                })
                color = 'red'
                linewidth = 3
            elif w_px < 5 or h_px < 5:
                is_anomaly = True
                issues_found.append({
                    'image': img_path.name,
                    'annotation': i,
                    'issue': 'very_small_bbox',
                    'bbox': [x, y, w_px, h_px]
                })
                color = 'orange'
                linewidth = 2
            else:
                color = 'lime'
                linewidth = 2
            
            # Draw rectangle
            rect = patches.Rectangle(
                (x, y), w_px, h_px,
                linewidth=linewidth,
                edgecolor=color,
                facecolor='none'
            )
            ax.add_patch(rect)
            
            # Add label
            label_text = f"{i}"
            if is_anomaly:
                label_text = f"{i}⚠"
            
            ax.text(
                x, y - 5,
                label_text,
                color=color,
                fontsize=10,
                weight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.7)
            )
        
        # Title with info
        title = f"{img_path.name}\n{len(annotations)} annotations | Image: {img_width}x{img_height}"
        if not annotations:
            title += " | ⚠️ NO LABELS"
        ax.set_title(title, fontsize=12, pad=10)
        ax.axis('off')
        
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='none', edgecolor='lime', label='Normal bbox'),
            Patch(facecolor='none', edgecolor='orange', label='Very small bbox'),
            Patch(facecolor='none', edgecolor='red', label='Outside image')
        ]
        ax.legend(handles=legend_elements, loc='upper right', fontsize=10)
        
        # Save
        output_file = output_dir / f"{idx:04d}_{img_path.stem}.png"
        plt.savefig(output_file, dpi=100, bbox_inches='tight')
        plt.close()
        
        if (idx + 1) % 10 == 0:
            print(f"  Processed {idx + 1}/{min(len(image_files), max_images)}...")
    
    print(f"\n✓ Saved {min(len(image_files), max_images)} visualizations to {output_dir}")
    
    # Summary
    if issues_found:
        print(f"\n⚠️  Found {len(issues_found)} potential issues:")
        issue_types = {}
        for issue in issues_found:
            issue_type = issue['issue']
            issue_types[issue_type] = issue_types.get(issue_type, 0) + 1
        
        for issue_type, count in issue_types.items():
            print(f"  - {issue_type}: {count}")
        
        # Show first few issues
        print("\nFirst 5 issues:")
        for i, issue in enumerate(issues_found[:5]):
            print(f"  {i+1}. {issue['image']}, ann {issue['annotation']}: {issue['issue']}")
            if 'bbox' in issue:
                print(f"     BBox: {issue['bbox']}")
    else:
        print("\n✓ No obvious issues detected!")
    
    return issues_found


def main():
    parser = argparse.ArgumentParser(
        description='Visualize YOLO format annotations for manual inspection'
    )
    parser.add_argument(
        '--yolo-dir',
        default='projects/test/yolo_bbox_format',
        help='Path to YOLO format directory (default: projects/test/yolo_bbox_format)'
    )
    parser.add_argument(
        '--output',
        default='/tmp/yolo_visualization',
        help='Output directory for visualizations (default: /tmp/yolo_visualization)'
    )
    parser.add_argument(
        '--split',
        choices=['train', 'val'],
        default='train',
        help='Which split to visualize (default: train)'
    )
    parser.add_argument(
        '--max-images',
        type=int,
        default=50,
        help='Maximum number of images to visualize (default: 50)'
    )
    
    args = parser.parse_args()
    
    issues = visualize_yolo_dataset(
        args.yolo_dir,
        args.output,
        split=args.split,
        max_images=args.max_images
    )
    
    print(f"\nVisualizations saved to: {args.output}")
    print("You can now manually inspect the images to verify annotations are correct.")


if __name__ == '__main__':
    main()
