#!/usr/bin/env python3
"""
Comprehensive check of bounding box quality by comparing to original segmentation masks.
This can detect coordinate swap issues, off-by-one errors, etc.
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import cv2
import json
import matplotlib.pyplot as plt
import matplotlib.patches as patches

sys.path.insert(0, str(Path(__file__).parent.parent))
from core.annotation import AnnotationManager


def check_bbox_matches_mask(mask, bbox, tolerance=2):
    """
    Check if a bounding box correctly bounds a mask
    
    Args:
        mask: Binary mask (H, W)
        bbox: [x, y, w, h] in COCO format
        tolerance: Allowed pixel difference
    
    Returns:
        dict with check results
    """
    # Compute bbox from mask
    coords = np.where(mask > 0)
    if len(coords[0]) == 0:
        return {
            'valid': False,
            'error': 'empty_mask',
            'message': 'Mask has no pixels'
        }
    
    # Calculate expected bbox from mask
    y_min, y_max = coords[0].min(), coords[0].max()
    x_min, x_max = coords[1].min(), coords[1].max()
    expected_bbox = [int(x_min), int(y_min), int(x_max - x_min + 1), int(y_max - y_min + 1)]
    
    x, y, w, h = bbox
    expected_x, expected_y, expected_w, expected_h = expected_bbox
    
    # Check for coordinate swap (x and y swapped)
    swapped_bbox = [y, x, h, w]
    
    # Calculate differences
    diff_x = abs(x - expected_x)
    diff_y = abs(y - expected_y)
    diff_w = abs(w - expected_w)
    diff_h = abs(h - expected_h)
    
    # Check if swapped coordinates match better
    diff_x_swapped = abs(swapped_bbox[0] - expected_x)
    diff_y_swapped = abs(swapped_bbox[1] - expected_y)
    
    result = {
        'actual_bbox': bbox,
        'expected_bbox': expected_bbox,
        'diff_x': diff_x,
        'diff_y': diff_y,
        'diff_w': diff_w,
        'diff_h': diff_h,
        'mask_coverage': 0,
        'bbox_coverage': 0
    }
    
    # Check if coordinates are swapped
    if (diff_x_swapped + diff_y_swapped) < (diff_x + diff_y) / 2:
        result['valid'] = False
        result['error'] = 'coordinates_swapped'
        result['message'] = f'X and Y coordinates appear to be swapped. Expected {expected_bbox}, got {bbox}'
        result['suggested_fix'] = swapped_bbox
        return result
    
    # Check if within tolerance
    if diff_x <= tolerance and diff_y <= tolerance and diff_w <= tolerance and diff_h <= tolerance:
        result['valid'] = True
        result['message'] = 'Bbox matches mask within tolerance'
    else:
        result['valid'] = False
        result['error'] = 'bbox_mismatch'
        result['message'] = f'Bbox does not match mask. Expected {expected_bbox}, got {bbox}'
        result['message'] += f'\nDifferences: x={diff_x}, y={diff_y}, w={diff_w}, h={diff_h}'
    
    # Calculate coverage metrics
    mask_area = np.sum(mask > 0)
    
    # Create bbox mask
    bbox_mask = np.zeros_like(mask)
    y1, y2 = max(0, y), min(mask.shape[0], y + h)
    x1, x2 = max(0, x), min(mask.shape[1], x + w)
    bbox_mask[y1:y2, x1:x2] = 1
    
    bbox_area = np.sum(bbox_mask > 0)
    
    if mask_area > 0:
        overlap = np.sum((mask > 0) & (bbox_mask > 0))
        result['mask_coverage'] = overlap / mask_area  # What fraction of mask is covered by bbox
    
    if bbox_area > 0:
        overlap = np.sum((mask > 0) & (bbox_mask > 0))
        result['bbox_coverage'] = overlap / bbox_area  # What fraction of bbox contains mask
    
    return result


def check_project_annotations(project_path, max_frames=20, visualize=True, output_dir=None):
    """
    Check bbox annotations across project
    
    Args:
        project_path: Path to project directory
        max_frames: Max frames to check
        visualize: Whether to create visualizations
        output_dir: Where to save visualizations
    
    Returns:
        dict with summary statistics
    """
    project_path = Path(project_path)
    ann_manager = AnnotationManager()
    
    # Load video metadata
    videos_dir = project_path / 'frames'
    video_dirs = [d for d in videos_dir.iterdir() if d.is_dir()]
    
    if not video_dirs:
        print("No video directories found")
        return None
    
    print(f"Checking annotations in {len(video_dirs)} videos...")
    
    issues = []
    stats = {
        'total_checked': 0,
        'valid': 0,
        'coordinate_swapped': 0,
        'bbox_mismatch': 0,
        'empty_mask': 0,
        'other_errors': 0
    }
    
    checked_count = 0
    
    for video_dir in video_dirs:
        video_id = video_dir.name
        
        # Get annotated frames by scanning annotation directories
        frame_indices = set()
        
        # Check PNG annotations
        png_video_dir = project_path / 'annotations' / 'png' / video_id
        if png_video_dir.exists():
            for png_file in png_video_dir.glob('*.png'):
                # Extract frame index from filename (e.g., frame_000042.png)
                frame_idx_str = png_file.stem.split('_')[-1]
                try:
                    frame_indices.add(int(frame_idx_str))
                except ValueError:
                    pass
        
        # Check PKL annotations
        pkl_video_dir = project_path / 'annotations' / 'pkl' / video_id
        if pkl_video_dir.exists():
            for pkl_file in pkl_video_dir.glob('*.pkl'):
                frame_idx_str = pkl_file.stem.split('_')[-1]
                try:
                    frame_indices.add(int(frame_idx_str))
                except ValueError:
                    pass
        
        frame_indices = sorted(list(frame_indices))
        
        if not frame_indices:
            continue
        
        print(f"\nChecking video: {video_id} ({len(frame_indices)} annotated frames)")
        
        for frame_idx in sorted(frame_indices)[:max_frames]:
            # Load annotations
            try:
                annotations = ann_manager.load_frame_annotations(project_path, video_id, frame_idx)
            except Exception as e:
                print(f"  Error loading frame {frame_idx}: {e}")
                continue
            
            if not annotations:
                continue
            
            # Check each annotation
            for ann_idx, ann in enumerate(annotations):
                if 'mask' not in ann or 'bbox' not in ann:
                    continue
                
                mask = ann['mask']
                bbox = ann['bbox']
                
                result = check_bbox_matches_mask(mask, bbox)
                stats['total_checked'] += 1
                
                if result['valid']:
                    stats['valid'] += 1
                else:
                    error_type = result.get('error', 'other')
                    if error_type == 'coordinates_swapped':
                        stats['coordinate_swapped'] += 1
                    elif error_type == 'bbox_mismatch':
                        stats['bbox_mismatch'] += 1
                    elif error_type == 'empty_mask':
                        stats['empty_mask'] += 1
                    else:
                        stats['other_errors'] += 1
                    
                    issues.append({
                        'video': video_id,
                        'frame': frame_idx,
                        'annotation': ann_idx,
                        'result': result
                    })
                    
                    # Print first few issues
                    if len(issues) <= 5:
                        print(f"  ⚠️  Frame {frame_idx}, Ann {ann_idx}: {result['message']}")
                
                # Visualize first few with issues
                if visualize and not result['valid'] and len(issues) <= 10 and output_dir:
                    visualize_bbox_issue(
                        mask, bbox, result,
                        f"{video_id}_frame_{frame_idx:06d}_ann_{ann_idx}",
                        output_dir
                    )
            
            checked_count += 1
            if checked_count >= max_frames:
                break
        
        if checked_count >= max_frames:
            break
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"SUMMARY:")
    print(f"  Total annotations checked: {stats['total_checked']}")
    print(f"  ✓ Valid: {stats['valid']} ({stats['valid']/max(stats['total_checked'],1)*100:.1f}%)")
    print(f"  ✗ Coordinates swapped: {stats['coordinate_swapped']}")
    print(f"  ✗ Bbox mismatch: {stats['bbox_mismatch']}")
    print(f"  ✗ Empty mask: {stats['empty_mask']}")
    print(f"  ✗ Other errors: {stats['other_errors']}")
    print(f"{'='*60}")
    
    if stats['coordinate_swapped'] > 0:
        print(f"\n🔴 CRITICAL: {stats['coordinate_swapped']} annotations have swapped X/Y coordinates!")
        print("This is likely a bug in the bbox calculation code.")
    
    return {'stats': stats, 'issues': issues}


def visualize_bbox_issue(mask, bbox, result, name, output_dir):
    """Create visualization of bbox issue"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Show mask with actual bbox
    ax1.imshow(mask, cmap='gray')
    x, y, w, h = bbox
    rect = patches.Rectangle((x, y), w, h, linewidth=2, edgecolor='red', facecolor='none', label='Actual')
    ax1.add_patch(rect)
    ax1.set_title(f'Actual BBox (red)\nBBox: {bbox}')
    ax1.legend()
    ax1.axis('off')
    
    # Show mask with expected bbox
    ax2.imshow(mask, cmap='gray')
    expected = result['expected_bbox']
    x2, y2, w2, h2 = expected
    rect2 = patches.Rectangle((x2, y2), w2, h2, linewidth=2, edgecolor='lime', facecolor='none', label='Expected')
    ax2.add_patch(rect2)
    ax2.set_title(f'Expected BBox (green)\nBBox: {expected}')
    ax2.legend()
    ax2.axis('off')
    
    fig.suptitle(f"{name}\n{result.get('message', '')}", fontsize=10)
    
    output_file = output_dir / f"{name}.png"
    plt.savefig(output_file, bbox_inches='tight', dpi=100)
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description='Check bbox annotation quality against segmentation masks'
    )
    parser.add_argument('--project', required=True, help='Path to project directory')
    parser.add_argument('--max-frames', type=int, default=20, help='Max frames to check')
    parser.add_argument('--output', help='Output directory for visualizations')
    parser.add_argument('--no-visualize', action='store_true', help='Skip visualizations')
    
    args = parser.parse_args()
    
    result = check_project_annotations(
        args.project,
        max_frames=args.max_frames,
        visualize=not args.no_visualize,
        output_dir=args.output
    )


if __name__ == '__main__':
    main()
