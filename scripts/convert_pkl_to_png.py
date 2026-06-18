#!/usr/bin/env python3
"""
Convert PKL annotations to PNG+JSON format.

This script scans all PKL annotation files in a project and converts them to the newer
PNG+JSON format. It preserves all annotation metadata while storing instance masks
as PNG images for faster loading.

Usage:
    python scripts/convert_pkl_to_png.py --project projects/test
    python scripts/convert_pkl_to_png.py --project projects/test --verbose
    python scripts/convert_pkl_to_png.py --project projects/test --force  # Overwrite existing
"""

import argparse
import pickle
import json
import numpy as np
import cv2
from pathlib import Path
import tempfile
import os
import sys


def rle_to_mask(rle):
    """Convert RLE encoding back to binary mask
    
    Args:
        rle: Dictionary with 'counts' and 'size' keys
        
    Returns:
        Binary mask as numpy array
    """
    h, w = rle['size']
    mask = np.zeros(h * w, dtype=np.uint8)
    
    counts = rle['counts']
    position = 0
    for i, count in enumerate(counts):
        if i % 2 == 1:  # Odd indices are 1s
            mask[position:position + count] = 1
        position += count
    
    return mask.reshape((h, w), order='F')


def sanitize_for_json(obj):
    """Convert numpy types and handle NaN/Inf for JSON serialization"""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        if np.isnan(obj) or np.isinf(obj):
            return None
        return float(obj)
    elif isinstance(obj, dict):
        return {k: sanitize_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [sanitize_for_json(item) for item in obj]
    else:
        return obj


def load_pkl_annotations(pkl_path):
    """Load annotations from a PKL file
    
    Args:
        pkl_path: Path to PKL file
        
    Returns:
        List of annotation dictionaries with masks
    """
    with open(pkl_path, 'rb') as f:
        compressed_annotations = pickle.load(f)
    
    # Decompress masks
    annotations = []
    for ann in compressed_annotations:
        decompressed_ann = ann.copy()
        if 'mask_rle' in ann:
            # Convert RLE back to mask
            decompressed_ann['mask'] = rle_to_mask(ann['mask_rle'])
            del decompressed_ann['mask_rle']
        annotations.append(decompressed_ann)
    
    return annotations


def save_png_annotations(png_dir, json_dir, frame_idx, annotations):
    """Save annotations in PNG+JSON format
    
    Args:
        png_dir: Directory for PNG files
        json_dir: Directory for JSON files
        frame_idx: Frame index
        annotations: List of annotation dictionaries with masks
    """
    # Prepare data for JSON (metadata without masks)
    json_data = []
    
    if annotations:
        # Get image shape from first annotation mask
        h, w = annotations[0]['mask'].shape[:2]
        
        # Create combined instance mask (each instance has unique ID)
        combined_mask = np.zeros((h, w), dtype=np.uint16)  # uint16 supports up to 65535 instances
        
        for ann in annotations:
            mask_id = ann.get('mask_id', ann.get('instance_id', 0))
            if mask_id > 0 and 'mask' in ann:
                # Add this instance to combined mask
                mask_binary = (ann['mask'] > 0).astype(bool)
                combined_mask[mask_binary] = mask_id
            
            # Save metadata (without mask) and sanitize for JSON
            ann_meta = {k: v for k, v in ann.items() if k not in ['mask', 'mask_rle']}
            json_data.append(ann_meta)
        
        # Save combined mask as PNG (atomic write)
        png_file = png_dir / f'frame_{frame_idx:06d}.png'
        
        # Write to temp file first, then rename (atomic on POSIX)
        temp_png = tempfile.NamedTemporaryFile(
            mode='wb', 
            delete=False, 
            dir=png_dir, 
            prefix=f'.tmp_frame_{frame_idx:06d}_',
            suffix='.png'
        )
        temp_png_path = temp_png.name
        temp_png.close()
        
        try:
            cv2.imwrite(temp_png_path, combined_mask)
            os.replace(temp_png_path, png_file)  # Atomic on POSIX
        except Exception as e:
            # Clean up temp file on error
            if os.path.exists(temp_png_path):
                os.unlink(temp_png_path)
            raise e
    
    # Sanitize JSON data (convert numpy types, handle NaN/Inf)
    try:
        json_data_sanitized = sanitize_for_json(json_data)
    except Exception as e:
        print(f"Error sanitizing JSON data for frame {frame_idx}: {e}")
        # Fallback to basic serialization
        json_data_sanitized = []
        for ann in json_data:
            sanitized_ann = {}
            for k, v in ann.items():
                try:
                    sanitized_ann[k] = sanitize_for_json(v)
                except:
                    print(f"Warning: Could not serialize annotation field '{k}': {type(v)}")
                    sanitized_ann[k] = None
            json_data_sanitized.append(sanitized_ann)
    
    # Save metadata as JSON (atomic write with explicit flush)
    json_file = json_dir / f'frame_{frame_idx:06d}.json'
    
    # Write to temp file first, then rename (atomic on POSIX)
    temp_json = tempfile.NamedTemporaryFile(
        mode='w', 
        delete=False, 
        dir=json_dir, 
        prefix=f'.tmp_frame_{frame_idx:06d}_',
        suffix='.json'
    )
    temp_json_path = temp_json.name
    
    try:
        json.dump(json_data_sanitized, temp_json, indent=2)
        temp_json.flush()  # Ensure data is written
        os.fsync(temp_json.fileno())  # Force write to disk
        temp_json.close()
        os.replace(temp_json_path, json_file)  # Atomic on POSIX
    except Exception as e:
        # Clean up temp file on error
        temp_json.close()
        if os.path.exists(temp_json_path):
            os.unlink(temp_json_path)
        raise e


def convert_pkl_to_png(project_path, force=False, verbose=False):
    """Convert all PKL annotations to PNG+JSON format
    
    Args:
        project_path: Path to project directory
        force: If True, overwrite existing PNG+JSON files
        verbose: If True, print detailed progress
        
    Returns:
        Dictionary with conversion statistics
    """
    project_path = Path(project_path)
    pkl_dir = project_path / 'annotations/pkl'
    
    if not pkl_dir.exists():
        print(f"No PKL annotations directory found: {pkl_dir}")
        return {'total': 0, 'converted': 0, 'skipped': 0, 'errors': 0}
    
    stats = {
        'total': 0,
        'converted': 0,
        'skipped': 0,
        'errors': 0,
        'videos': {}
    }
    
    # Scan all video directories
    for video_dir in sorted(pkl_dir.iterdir()):
        if not video_dir.is_dir():
            continue
        
        video_id = video_dir.name
        video_stats = {'total': 0, 'converted': 0, 'skipped': 0, 'errors': 0}
        
        # Create output directories
        png_out_dir = project_path / 'annotations/png' / video_id
        json_out_dir = project_path / 'annotations/json' / video_id
        png_out_dir.mkdir(parents=True, exist_ok=True)
        json_out_dir.mkdir(parents=True, exist_ok=True)
        
        # Process each PKL file
        pkl_files = sorted(video_dir.glob('frame_*.pkl'))
        for pkl_file in pkl_files:
            stats['total'] += 1
            video_stats['total'] += 1
            
            # Extract frame index
            frame_name = pkl_file.stem  # e.g., 'frame_000123'
            try:
                frame_idx = int(frame_name.split('_')[1])
            except (IndexError, ValueError):
                print(f"Warning: Could not parse frame index from {pkl_file}")
                stats['errors'] += 1
                video_stats['errors'] += 1
                continue
            
            # Check if PNG+JSON already exist
            png_file = png_out_dir / f'frame_{frame_idx:06d}.png'
            json_file = json_out_dir / f'frame_{frame_idx:06d}.json'
            
            if png_file.exists() and json_file.exists() and not force:
                if verbose:
                    print(f"  Skipping {video_id}/frame_{frame_idx:06d} (already exists)")
                stats['skipped'] += 1
                video_stats['skipped'] += 1
                continue
            
            # Convert PKL to PNG+JSON
            try:
                # Load PKL annotations
                annotations = load_pkl_annotations(pkl_file)
                
                # Save as PNG+JSON
                save_png_annotations(png_out_dir, json_out_dir, frame_idx, annotations)
                
                stats['converted'] += 1
                video_stats['converted'] += 1
                
                if verbose:
                    print(f"  Converted {video_id}/frame_{frame_idx:06d} ({len(annotations)} annotations)")
            
            except Exception as e:
                print(f"Error converting {video_id}/frame_{frame_idx:06d}: {e}")
                stats['errors'] += 1
                video_stats['errors'] += 1
        
        # Store video statistics
        if video_stats['total'] > 0:
            stats['videos'][video_id] = video_stats
            
            if not verbose:
                # Print summary per video
                print(f"{video_id}: {video_stats['converted']} converted, "
                      f"{video_stats['skipped']} skipped, "
                      f"{video_stats['errors']} errors "
                      f"(total: {video_stats['total']})")
    
    return stats


def main():
    parser = argparse.ArgumentParser(
        description='Convert PKL annotations to PNG+JSON format',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Convert all PKL files in a project
  python scripts/convert_pkl_to_png.py --project projects/test
  
  # Show detailed progress
  python scripts/convert_pkl_to_png.py --project projects/test --verbose
  
  # Overwrite existing PNG+JSON files
  python scripts/convert_pkl_to_png.py --project projects/test --force
        """
    )
    
    parser.add_argument(
        '--project',
        type=str,
        required=True,
        help='Path to project directory'
    )
    
    parser.add_argument(
        '--force',
        action='store_true',
        help='Overwrite existing PNG+JSON files (default: skip existing)'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Show detailed progress for each file'
    )
    
    args = parser.parse_args()
    
    # Validate project path
    project_path = Path(args.project)
    if not project_path.exists():
        print(f"Error: Project directory does not exist: {project_path}")
        sys.exit(1)
    
    print(f"Converting PKL annotations to PNG+JSON format...")
    print(f"Project: {project_path}")
    print(f"Force overwrite: {args.force}")
    print()
    
    # Run conversion
    stats = convert_pkl_to_png(project_path, force=args.force, verbose=args.verbose)
    
    # Print overall statistics
    print()
    print("=" * 60)
    print("CONVERSION SUMMARY")
    print("=" * 60)
    print(f"Total PKL files found:  {stats['total']}")
    print(f"Converted:              {stats['converted']}")
    print(f"Skipped (already exist):{stats['skipped']}")
    print(f"Errors:                 {stats['errors']}")
    print()
    
    if stats['converted'] > 0:
        print(f"✓ Successfully converted {stats['converted']} PKL files to PNG+JSON format")
    elif stats['skipped'] == stats['total']:
        print("✓ All PKL files already have PNG+JSON conversions")
    elif stats['errors'] > 0:
        print(f"⚠ Completed with {stats['errors']} errors")
        sys.exit(1)
    else:
        print("No PKL files found to convert")


if __name__ == '__main__':
    main()
