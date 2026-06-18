#!/usr/bin/env python3
"""
Test if the bbox computation from masks is correct
"""
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.annotation import AnnotationManager

# Import the function from the script
sys.path.insert(0, str(Path(__file__).parent))
from scripts.generate_bbox_annotations import compute_bbox_from_mask


def compute_bbox_method_1(mask):
    """Method used in generate_bbox_annotations.py"""
    y_indices, x_indices = np.where(mask > 0)
    if len(y_indices) > 0:
        x_min = int(x_indices.min())
        x_max = int(x_indices.max())
        y_min = int(y_indices.min())
        y_max = int(y_indices.max())
        return [x_min, y_min, x_max - x_min + 1, y_max - y_min + 1]
    return [0, 0, 0, 0]


def compute_bbox_method_2(mask):
    """Method used in core/annotation.py COCO export"""
    coords = np.where(mask > 0)
    if len(coords[0]) > 0:
        y_min, y_max = coords[0].min(), coords[0].max()
        x_min, x_max = coords[1].min(), coords[1].max()
        bbox_width = int(x_max - x_min + 1)
        bbox_height = int(y_max - y_min + 1)
        return [int(x_min), int(y_min), bbox_width, bbox_height]
    return [0, 0, 0, 0]


def test_bbox_computation():
    """Test both bbox computation methods"""
    project_path = Path('projects/test')
    video_id = 'worker24-2022-08-18_20-12-01'
    frame_idx = 0
    
    ann_manager = AnnotationManager()
    annotations = ann_manager.load_frame_annotations(project_path, video_id, frame_idx)
    
    print(f"\nTesting bbox computation on {len(annotations)} annotations\n")
    print("="*80)
    
    mismatches = 0
    
    for i, ann in enumerate(annotations[:3]):  # Test first 3
        mask = ann['mask']
        
        bbox1 = compute_bbox_method_1(mask)
        bbox2 = compute_bbox_method_2(mask)
        
        print(f"\nAnnotation {i}:")
        print(f"  Mask shape: {mask.shape}")
        print(f"  Method 1 (generate_bbox_annotations.py): {bbox1}")
        print(f"  Method 2 (core/annotation.py):           {bbox2}")
        
        if bbox1 != bbox2:
            print(f"  ⚠️  MISMATCH!")
            mismatches += 1
            
            # Visualize the difference
            fig, axes = plt.subplots(1, 3, figsize=(18, 6))
            
            # Show mask
            axes[0].imshow(mask, cmap='gray')
            axes[0].set_title(f'Mask {i}')
            axes[0].axis('off')
            
            # Show with bbox1
            axes[1].imshow(mask, cmap='gray')
            x, y, w, h = bbox1
            rect1 = patches.Rectangle((x, y), w, h, linewidth=3, edgecolor='red', facecolor='none')
            axes[1].add_patch(rect1)
            axes[1].set_title(f'Method 1 (RED)\n{bbox1}')
            axes[1].axis('off')
            
            # Show with bbox2
            axes[2].imshow(mask, cmap='gray')
            x2, y2, w2, h2 = bbox2
            rect2 = patches.Rectangle((x2, y2), w2, h2, linewidth=3, edgecolor='lime', facecolor='none')
            axes[2].add_patch(rect2)
            axes[2].set_title(f'Method 2 (GREEN)\n{bbox2}')
            axes[2].axis('off')
            
            plt.tight_layout()
            plt.savefig(f'/tmp/bbox_comparison_ann{i}.png', dpi=150)
            print(f"  Saved visualization to /tmp/bbox_comparison_ann{i}.png")
            plt.close()
        else:
            print(f"  ✓ Both methods agree")
    
    print("\n" + "="*80)
    print(f"\nSummary: {mismatches} mismatches found out of {min(3, len(annotations))} tested")
    
    if mismatches == 0:
        print("✓ Both bbox computation methods produce identical results")
    else:
        print("⚠️  Methods produce different results - there's a bug!")


if __name__ == '__main__':
    test_bbox_computation()
