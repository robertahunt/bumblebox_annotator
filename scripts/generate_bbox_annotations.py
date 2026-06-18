#!/usr/bin/env python3
"""
Generate bounding box annotations from existing segmentation annotations.

This script scans all videos in a project and generates bbox-only annotations
for frames that have segmentation annotations. The bboxes are computed from
the segmentation masks and saved separately in annotations/bbox/.

Usage:
    python scripts/generate_bbox_annotations.py --project /path/to/project
"""

import argparse
import sys
from pathlib import Path
import json
import numpy as np

# Add parent directory to path to import from core
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.annotation import AnnotationManager


def compute_bbox_from_mask(mask):
    """Compute bounding box from a binary mask
    
    Args:
        mask: Binary mask array (H, W)
        
    Returns:
        [x, y, w, h] bounding box in COCO format, or [0, 0, 0, 0] if mask is empty
    """
    y_indices, x_indices = np.where(mask > 0)
    if len(y_indices) > 0:
        x_min = int(x_indices.min())
        x_max = int(x_indices.max())
        y_min = int(y_indices.min())
        y_max = int(y_indices.max())
        return [x_min, y_min, x_max - x_min + 1, y_max - y_min + 1]
    return [0, 0, 0, 0]


def generate_bbox_annotations(project_path, overwrite=False, verbose=True):
    """Generate bounding box annotations from segmentations
    
    Args:
        project_path: Path to project directory
        overwrite: If True, regenerate all bbox annotations even if they exist
        verbose: Print progress information
        
    Returns:
        dict with statistics: {video_id: {frames_processed, bboxes_generated}}
    """
    project_path = Path(project_path)
    
    if not project_path.exists():
        raise ValueError(f"Project path does not exist: {project_path}")
    
    # Initialize annotation manager
    ann_manager = AnnotationManager()
    
    # Find all videos with annotations
    annotations_base = project_path / 'annotations'
    
    # Check both pkl and png annotation directories
    video_ids = set()
    
    pkl_dir = annotations_base / 'pkl'
    if pkl_dir.exists():
        for video_dir in pkl_dir.iterdir():
            if video_dir.is_dir():
                video_ids.add(video_dir.name)
    
    png_dir = annotations_base / 'png'
    if png_dir.exists():
        for video_dir in png_dir.iterdir():
            if video_dir.is_dir():
                video_ids.add(video_dir.name)
    
    if not video_ids:
        print("No annotated videos found in project")
        return {}
    
    print(f"Found {len(video_ids)} videos with annotations")
    
    stats = {}
    
    for video_id in sorted(video_ids):
        if verbose:
            print(f"\nProcessing video: {video_id}")
        
        frames_processed = 0
        bboxes_generated = 0
        
        # Find all annotation files for this video
        frame_indices = set()
        
        # Check PKL annotations
        pkl_video_dir = pkl_dir / video_id
        if pkl_video_dir.exists():
            for ann_file in pkl_video_dir.glob('frame_*.pkl'):
                frame_idx = int(ann_file.stem.split('_')[1])
                frame_indices.add(frame_idx)
        
        # Check PNG annotations
        png_video_dir = png_dir / video_id
        if png_video_dir.exists():
            for ann_file in png_video_dir.glob('frame_*.png'):
                frame_idx = int(ann_file.stem.split('_')[1])
                frame_indices.add(frame_idx)
        
        if not frame_indices:
            if verbose:
                print(f"  No annotation frames found, skipping")
            continue
        
        if verbose:
            print(f"  Found {len(frame_indices)} annotated frames")
        
        for frame_idx in sorted(frame_indices):
            # Check if bbox annotations already exist
            bbox_file = annotations_base / 'bbox' / video_id / f'frame_{frame_idx:06d}.json'
            
            if bbox_file.exists() and not overwrite:
                if verbose and frames_processed == 0:
                    print(f"  Skipping (bbox annotations already exist, use --overwrite to regenerate)")
                continue
            
            # Load segmentation annotations
            try:
                segmentation_anns = ann_manager.load_frame_annotations(
                    project_path, video_id, frame_idx
                )
                
                if not segmentation_anns:
                    continue
                
                # Generate bbox annotations
                bbox_anns = []
                for ann in segmentation_anns:
                    # Create bbox-only annotation (no mask)
                    bbox_ann = {k: v for k, v in ann.items() if k not in ['mask', 'mask_rle']}
                    
                    # Compute bbox from mask if not present or invalid
                    if 'bbox' not in bbox_ann or bbox_ann.get('bbox') == [0, 0, 0, 0]:
                        if 'mask' in ann:
                            bbox_ann['bbox'] = compute_bbox_from_mask(ann['mask'])
                            bboxes_generated += 1
                        else:
                            bbox_ann['bbox'] = [0, 0, 0, 0]
                    else:
                        # Bbox already exists in annotation
                        bboxes_generated += 1
                    
                    bbox_anns.append(bbox_ann)
                
                # Save bbox annotations
                ann_manager.save_bbox_annotations(
                    project_path, video_id, frame_idx, bbox_anns
                )
                
                frames_processed += 1
                
                if verbose and frames_processed % 10 == 0:
                    print(f"  Processed {frames_processed}/{len(frame_indices)} frames...")
                
            except Exception as e:
                print(f"  Error processing frame {frame_idx}: {e}")
                continue
        
        stats[video_id] = {
            'frames_processed': frames_processed,
            'bboxes_generated': bboxes_generated
        }
        
        if verbose:
            print(f"  ✓ Completed: {frames_processed} frames, {bboxes_generated} bboxes generated")
    
    return stats


def main():
    parser = argparse.ArgumentParser(
        description='Generate bounding box annotations from segmentation annotations',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate bbox annotations for all annotated frames
  python scripts/generate_bbox_annotations.py --project /path/to/project
  
  # Regenerate all bbox annotations (overwrite existing)
  python scripts/generate_bbox_annotations.py --project /path/to/project --overwrite
  
  # Silent mode (only show errors)
  python scripts/generate_bbox_annotations.py --project /path/to/project --quiet
"""
    )
    
    parser.add_argument(
        '--project', '-p',
        type=str,
        required=True,
        help='Path to project directory'
    )
    
    parser.add_argument(
        '--overwrite',
        action='store_true',
        help='Regenerate bbox annotations even if they already exist'
    )
    
    parser.add_argument(
        '--quiet', '-q',
        action='store_true',
        help='Suppress progress output (only show summary)'
    )
    
    args = parser.parse_args()
    
    try:
        print(f"Generating bounding box annotations for project: {args.project}")
        print(f"Overwrite existing: {args.overwrite}")
        print()
        
        stats = generate_bbox_annotations(
            args.project,
            overwrite=args.overwrite,
            verbose=not args.quiet
        )
        
        print("\n" + "="*60)
        print("SUMMARY")
        print("="*60)
        total_frames = sum(s['frames_processed'] for s in stats.values())
        total_bboxes = sum(s['bboxes_generated'] for s in stats.values())
        print(f"Videos processed: {len(stats)}")
        print(f"Frames processed: {total_frames}")
        print(f"Bboxes generated: {total_bboxes}")
        
        if stats:
            print("\nPer-video breakdown:")
            for video_id, video_stats in sorted(stats.items()):
                print(f"  {video_id}: {video_stats['frames_processed']} frames, {video_stats['bboxes_generated']} bboxes")
        
        print("\n✓ Generation complete!")
        
    except Exception as e:
        print(f"\nError: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
