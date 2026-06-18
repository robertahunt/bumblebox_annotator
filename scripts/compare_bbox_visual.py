#!/usr/bin/env python3
"""
Visually compare COCO and YOLO bounding boxes on an image
"""
import argparse
import json
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path


def compare_bboxes(project_path, video_name, frame_name, output_path=None):
    """
    Compare COCO and YOLO bboxes visually
    
    Args:
        project_path: Path to project
        video_name: Video name
        frame_name: Frame filename (e.g., frame_000052.jpg)
        output_path: Where to save comparison image
    """
    project_path = Path(project_path)
    
    # Find COCO JSON
    coco_pattern = f"{video_name}.json"
    coco_files = list(project_path.glob(f"annotations/coco/**/{coco_pattern}"))
    
    if not coco_files:
        print(f"COCO file not found for {video_name}")
        return
    
    coco_path = coco_files[0]
    print(f"COCO file: {coco_path}")
    
    # Load COCO data
    with open(coco_path) as f:
        coco_data = json.load(f)
    
    # Find the image
    target_img = None
    for img in coco_data['images']:
        if frame_name in img['file_name']:
            target_img = img
            break
    
    if not target_img:
        print(f"Frame {frame_name} not found in COCO data")
        return
    
    img_width = target_img['width']
    img_height = target_img['height']
    
    # Get COCO annotations
    coco_anns = [a for a in coco_data['annotations'] if a['image_id'] == target_img['id']]
    print(f"COCO annotations: {len(coco_anns)}")
    
    # Load YOLO annotations
    yolo_label_file = project_path / 'yolo_bbox_format' / 'train' / 'labels' / f"{video_name}_{frame_name.replace('.jpg', '.txt')}"
    
    if not yolo_label_file.exists():
        # Try val split
        yolo_label_file = project_path / 'yolo_bbox_format' / 'val' / 'labels' / f"{video_name}_{frame_name.replace('.jpg', '.txt')}"
    
    if not yolo_label_file.exists():
        print(f"YOLO file not found: {yolo_label_file}")
        return
    
    print(f"YOLO file: {yolo_label_file}")
    
    # Parse YOLO annotations
    yolo_anns = []
    with open(yolo_label_file) as f:
        for line in f:
            parts = line.strip().split()
            yolo_anns.append({
                'x_center': float(parts[1]),
                'y_center': float(parts[2]),
                'width': float(parts[3]),
                'height': float(parts[4])
            })
    
    print(f"YOLO annotations: {len(yolo_anns)}")
    
    # Load image
    img_path = project_path / target_img['file_name']
    if not img_path.exists():
        print(f"Image not found: {img_path}")
        return
    
    img = cv2.imread(str(img_path))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Create comparison
    fig, axes = plt.subplots(1, 3, figsize=(20, 7))
    
    # 1. COCO bboxes (red)
    axes[0].imshow(img)
    for i, ann in enumerate(coco_anns):
        x, y, w, h = ann['bbox']
        rect = patches.Rectangle((x, y), w, h, linewidth=2, edgecolor='red', facecolor='none')
        axes[0].add_patch(rect)
        axes[0].text(x, y-5, str(i), color='red', fontsize=10, weight='bold')
    axes[0].set_title(f'COCO Annotations ({len(coco_anns)} boxes)', fontsize=14)
    axes[0].axis('off')
    
    # 2. YOLO bboxes converted back to pixels (green)
    axes[1].imshow(img)
    for i, yolo_ann in enumerate(yolo_anns):
        # Convert back to pixels
        x_center = yolo_ann['x_center'] * img_width
        y_center = yolo_ann['y_center'] * img_height
        w = yolo_ann['width'] * img_width
        h = yolo_ann['height'] * img_height
        x = x_center - w / 2
        y = y_center - h / 2
        
        rect = patches.Rectangle((x, y), w, h, linewidth=2, edgecolor='lime', facecolor='none')
        axes[1].add_patch(rect)
        axes[1].text(x, y-5, str(i), color='lime', fontsize=10, weight='bold')
    axes[1].set_title(f'YOLO Annotations ({len(yolo_anns)} boxes)', fontsize=14)
    axes[1].axis('off')
    
    # 3. Overlay both (red=COCO, green=YOLO)
    axes[2].imshow(img)
    for ann in coco_anns:
        x, y, w, h = ann['bbox']
        rect = patches.Rectangle((x, y), w, h, linewidth=2, edgecolor='red', facecolor='none', label='COCO')
        axes[2].add_patch(rect)
    for yolo_ann in yolo_anns:
        x_center = yolo_ann['x_center'] * img_width
        y_center = yolo_ann['y_center'] * img_height
        w = yolo_ann['width'] * img_width
        h = yolo_ann['height'] * img_height
        x = x_center - w / 2
        y = y_center - h / 2
        rect = patches.Rectangle((x, y), w, h, linewidth=2, edgecolor='lime', facecolor='none', linestyle='--', label='YOLO')
        axes[2].add_patch(rect)
    
    # Remove duplicate labels
    handles, labels = axes[2].get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    axes[2].legend(by_label.values(), by_label.keys(), loc='upper right')
    axes[2].set_title('Overlay (if aligned, they match)', fontsize=14)
    axes[2].axis('off')
    
    fig.suptitle(f'{video_name} - {frame_name}', fontsize=16)
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved to: {output_path}")
    else:
        plt.show()
    
    plt.close()
    
    # Print numeric comparison
    print(f"\nNumeric comparison:")
    for i in range(min(len(coco_anns), len(yolo_anns))):
        coco_bbox = coco_anns[i]['bbox']
        yolo_ann = yolo_anns[i]
        
        # Expected YOLO from COCO
        x, y, w, h = coco_bbox
        expected_x_center = (x + w / 2) / img_width
        expected_y_center = (y + h / 2) / img_height
        expected_w_norm = w / img_width
        expected_h_norm = h / img_height
        
        print(f"\nBox {i}:")
        print(f"  COCO:     [{x}, {y}, {w}, {h}]")
        print(f"  Expected: [{expected_x_center:.6f}, {expected_y_center:.6f}, {expected_w_norm:.6f}, {expected_h_norm:.6f}]")
        print(f"  Actual:   [{yolo_ann['x_center']:.6f}, {yolo_ann['y_center']:.6f}, {yolo_ann['width']:.6f}, {yolo_ann['height']:.6f}]")
        
        if abs(expected_x_center - yolo_ann['x_center']) < 0.000001:
            print(f"  ✓ Match")
        else:
            print(f"  ✗ MISMATCH!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Compare COCO and YOLO bboxes visually')
    parser.add_argument('--project', default='projects/test', help='Project path')
    parser.add_argument('--video', required=True, help='Video name')
    parser.add_argument('--frame', required=True, help='Frame filename (e.g., frame_000052.jpg)')
    parser.add_argument('--output', help='Output image path')
    
    args = parser.parse_args()
    
    compare_bboxes(args.project, args.video, args.frame, args.output)
