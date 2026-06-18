#!/usr/bin/env python3
"""
Check if COCO to YOLO bbox conversion is correct
"""
import json
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path

def check_coco_to_yolo_conversion(coco_json_path, yolo_format_dir, num_samples=5):
    """
    Compare COCO and YOLO format annotations
    
    Args:
        coco_json_path: Path to COCO JSON file
        yolo_format_dir: Path to YOLO format directory (with images/ and labels/)
        num_samples: Number of samples to visualize
    """
    # Load COCO JSON
    with open(coco_json_path, 'r') as f:
        coco_data = json.load(f)
    
    print(f"\nCOCO JSON: {coco_json_path}")
    print(f"  Images: {len(coco_data['images'])}")
    print(f"  Annotations: {len(coco_data['annotations'])}")
    
    # Create image id to file mapping
    image_map = {img['id']: img for img in coco_data['images']}
    
    # Create lookup for annotations by image_id
    annotations_by_image = {}
    for ann in coco_data['annotations']:
        image_id = ann['image_id']
        if image_id not in annotations_by_image:
            annotations_by_image[image_id] = []
        annotations_by_image[image_id].append(ann)
    
    yolo_format_dir = Path(yolo_format_dir)
    yolo_labels_dir = yolo_format_dir / 'labels'
    yolo_images_dir = yolo_format_dir / 'images'
    
    # Check conversion for first few images
    issues_found = []
    
    for idx, (image_id, image_info) in enumerate(list(image_map.items())[:num_samples]):
        img_width = image_info['width']
        img_height = image_info['height']
        img_filename = Path(image_info['file_name']).name
        
        print(f"\n--- Image {idx}: {img_filename} ---")
        print(f"  Dimensions: {img_width} x {img_height}")
        
        # Get COCO annotations
        coco_anns = annotations_by_image.get(image_id, [])
        print(f"  COCO annotations: {len(coco_anns)}")
        
        # Load YOLO annotations
        yolo_label_file = yolo_labels_dir / f"{Path(img_filename).stem}.txt"
        
        if not yolo_label_file.exists():
            print(f"  ⚠️  YOLO label file not found: {yolo_label_file}")
            continue
        
        with open(yolo_label_file, 'r') as f:
            yolo_lines = f.read().strip().split('\n')
            yolo_anns = []
            for line in yolo_lines:
                if line.strip():
                    parts = line.strip().split()
                    yolo_anns.append({
                        'class_id': int(parts[0]),
                        'x_center': float(parts[1]),
                        'y_center': float(parts[2]),
                        'width': float(parts[3]),
                        'height': float(parts[4])
                    })
        
        print(f"  YOLO annotations: {len(yolo_anns)}")
        
        if len(coco_anns) != len(yolo_anns):
            print(f"  ⚠️  Annotation count mismatch!")
            issues_found.append({
                'image': img_filename,
                'issue': 'count_mismatch',
                'coco_count': len(coco_anns),
                'yolo_count': len(yolo_anns)
            })
            continue
        
        # Compare each annotation
        for ann_idx, (coco_ann, yolo_ann) in enumerate(zip(coco_anns, yolo_anns)):
            # Get COCO bbox [x, y, width, height]
            coco_x, coco_y, coco_w, coco_h = coco_ann['bbox']
            
            # Convert COCO to YOLO format (normalized)
            expected_x_center = (coco_x + coco_w / 2) / img_width
            expected_y_center = (coco_y + coco_h / 2) / img_height
            expected_width = coco_w / img_width
            expected_height = coco_h / img_height
            
            # Get actual YOLO values
            actual_x_center = yolo_ann['x_center']
            actual_y_center = yolo_ann['y_center']
            actual_width = yolo_ann['width']
            actual_height = yolo_ann['height']
            
            # Compare with tolerance
            tolerance = 0.001  # 0.1% tolerance for floating point
            
            diff_x = abs(expected_x_center - actual_x_center)
            diff_y = abs(expected_y_center - actual_y_center)
            diff_w = abs(expected_width - actual_width)
            diff_h = abs(expected_height - actual_height)
            
            if max(diff_x, diff_y, diff_w, diff_h) > tolerance:
                print(f"  ⚠️  Ann {ann_idx}: Conversion mismatch!")
                print(f"    COCO bbox: {coco_ann['bbox']}")
                print(f"    Expected YOLO: [{expected_x_center:.6f}, {expected_y_center:.6f}, {expected_width:.6f}, {expected_height:.6f}]")
                print(f"    Actual YOLO:   [{actual_x_center:.6f}, {actual_y_center:.6f}, {actual_width:.6f}, {actual_height:.6f}]")
                print(f"    Differences: dx={diff_x:.6f}, dy={diff_y:.6f}, dw={diff_w:.6f}, dh={diff_h:.6f}")
                
                issues_found.append({
                    'image': img_filename,
                    'annotation': ann_idx,
                    'issue': 'conversion_error',
                    'coco_bbox': coco_ann['bbox'],
                    'expected_yolo': [expected_x_center, expected_y_center, expected_width, expected_height],
                    'actual_yolo': [actual_x_center, actual_y_center, actual_width, actual_height]
                })
            else:
                print(f"  ✓ Ann {ann_idx}: Conversion correct")
    
    print(f"\n{'='*60}")
    print(f"Summary: {len(issues_found)} issues found")
    
    if issues_found:
        print("\n⚠️  Issues detected:")
        for issue in issues_found:
            print(f"  - {issue['image']}: {issue['issue']}")
    else:
        print("\n✓ All checked conversions are correct!")
    
    return issues_found


def visualize_bbox_on_image(coco_json_path, yolo_format_dir, project_path, image_idx=0):
    """Visualize bboxes from both formats on an actual image"""
    # Load COCO JSON
    with open(coco_json_path, 'r') as f:
        coco_data = json.load(f)
    
    image_info = coco_data['images'][image_idx]
    img_width = image_info['width']
    img_height = image_info['height']
    img_filename = Path(image_info['file_name']).name
    
    # Find the image
    project_path = Path(project_path)
    img_path = project_path / image_info['file_name']
    
    if not img_path.exists():
        print(f"Image not found: {img_path}")
        return
    
    # Load image
    img = cv2.imread(str(img_path))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Get COCO annotations
    image_id = image_info['id']
    coco_anns = [ann for ann in coco_data['annotations'] if ann['image_id'] == image_id]
    
    # Load YOLO annotations
    yolo_format_dir = Path(yolo_format_dir)
    yolo_label_file = yolo_format_dir / 'labels' / f"{Path(img_filename).stem}.txt"
    
    yolo_anns = []
    if yolo_label_file.exists():
        with open(yolo_label_file, 'r') as f:
            for line in f:
                if line.strip():
                    parts = line.strip().split()
                    yolo_anns.append({
                        'class_id': int(parts[0]),
                        'x_center': float(parts[1]),
                        'y_center': float(parts[2]),
                        'width': float(parts[3]),
                        'height': float(parts[4])
                    })
    
    # Create visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Show COCO bboxes
    ax1.imshow(img)
    for ann in coco_anns:
        x, y, w, h = ann['bbox']
        rect = patches.Rectangle((x, y), w, h, linewidth=2, edgecolor='red', facecolor='none')
        ax1.add_patch(rect)
    ax1.set_title(f'COCO Format ({len(coco_anns)} bees)\nBBox: [x, y, w, h]')
    ax1.axis('off')
    
    # Show YOLO bboxes (converted back to pixel coordinates)
    ax2.imshow(img)
    for yolo_ann in yolo_anns:
        # Convert YOLO format back to pixel coordinates
        x_center_norm = yolo_ann['x_center']
        y_center_norm = yolo_ann['y_center']
        w_norm = yolo_ann['width']
        h_norm = yolo_ann['height']
        
        x_center = x_center_norm * img_width
        y_center = y_center_norm * img_height
        w = w_norm * img_width
        h = h_norm * img_height
        
        x = x_center - w / 2
        y = y_center - h / 2
        
        rect = patches.Rectangle((x, y), w, h, linewidth=2, edgecolor='lime', facecolor='none')
        ax2.add_patch(rect)
    ax2.set_title(f'YOLO Format ({len(yolo_anns)} bees)\nBBox: [x_c, y_c, w, h] (normalized)')
    ax2.axis('off')
    
    plt.tight_layout()
    plt.savefig('/tmp/coco_vs_yolo_bbox.png', dpi=150)
    print(f"Saved visualization to /tmp/coco_vs_yolo_bbox.png")
    plt.show()


if __name__ == '__main__':
    project_path = 'projects/test'
    coco_json = 'projects/test/annotations/coco/train/bumblebox-01_2024-07-18_16_00_03.json'
    yolo_dir = 'projects/test/yolo_bbox_format/train'
    
    issues = check_coco_to_yolo_conversion(coco_json, yolo_dir, num_samples=5)
    
    print("\nGenerating visual comparison...")
    visualize_bbox_on_image(coco_json, yolo_dir, project_path, image_idx=0)
