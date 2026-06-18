#!/usr/bin/env python3
"""
Deprecate old annotation formats (bbox and pkl)

This script helps clean up old annotation formats that are no longer used:
- annotations/bbox/ - deprecated bbox-only annotations (now computed on-demand from PNG+JSON)
- annotations/pkl/ - deprecated pickle format (now using PNG+JSON format)

The ground truth is now stored only in:
- annotations/png/ - instance masks as PNG files
- annotations/json/ - metadata for each frame

COCO exports are built from PNG+JSON format.
"""

import argparse
import shutil
from pathlib import Path
from datetime import datetime


def deprecate_old_annotations(project_path: Path, dry_run: bool = True):
    """
    Move old annotation formats to _deprecated folders
    
    Args:
        project_path: Path to project directory
        dry_run: If True, only show what would be done without making changes
    """
    project_path = Path(project_path)
    
    print("="*60)
    print("Deprecating Old Annotation Formats")
    print("="*60)
    print(f"Project: {project_path}")
    print(f"Mode: {'DRY RUN (no changes will be made)' if dry_run else 'LIVE (will move files)'}")
    print()
    
    # Timestamp for backup folder
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Check bbox folder
    bbox_dir = project_path / 'annotations/bbox'
    if bbox_dir.exists():
        video_count = len(list(bbox_dir.glob('*')))
        total_files = len(list(bbox_dir.glob('*/*.json')))
        
        print(f"✓ Found bbox annotations:")
        print(f"  Location: {bbox_dir}")
        print(f"  Videos: {video_count}")
        print(f"  Files: {total_files}")
        
        if not dry_run:
            deprecated_bbox = project_path / 'annotations' / f'bbox_deprecated_{timestamp}'
            print(f"  → Moving to: {deprecated_bbox}")
            shutil.move(str(bbox_dir), str(deprecated_bbox))
            print(f"  ✓ Moved successfully")
        else:
            print(f"  [DRY RUN] Would move to: bbox_deprecated_{timestamp}")
    else:
        print("✓ No bbox folder found (already removed)")
    
    print()
    
    # Check pkl folder
    pkl_dir = project_path / 'annotations/pkl'
    if pkl_dir.exists():
        video_count = len(list(pkl_dir.glob('*')))
        total_files = len(list(pkl_dir.glob('*/*.pkl')))
        
        print(f"✓ Found pkl annotations:")
        print(f"  Location: {pkl_dir}")
        print(f"  Videos: {video_count}")
        print(f"  Files: {total_files}")
        
        if not dry_run:
            deprecated_pkl = project_path / 'annotations' / f'pkl_deprecated_{timestamp}'
            print(f"  → Moving to: {deprecated_pkl}")
            shutil.move(str(pkl_dir), str(deprecated_pkl))
            print(f"  ✓ Moved successfully")
        else:
            print(f"  [DRY RUN] Would move to: pkl_deprecated_{timestamp}")
    else:
        print("✓ No pkl folder found (already removed)")
    
    print()
    
    # Check current annotation format
    png_dir = project_path / 'annotations/png'
    json_dir = project_path / 'annotations/json'
    
    if png_dir.exists() and json_dir.exists():
        png_video_count = len(list(png_dir.glob('*')))
        json_video_count = len(list(json_dir.glob('*')))
        png_file_count = len(list(png_dir.glob('*/*.png')))
        json_file_count = len(list(json_dir.glob('*/*.json')))
        
        print(f"✓ Current annotation format (PNG+JSON):")
        print(f"  PNG videos: {png_video_count}, files: {png_file_count}")
        print(f"  JSON videos: {json_video_count}, files: {json_file_count}")
        print(f"  Status: Ready for COCO export")
    else:
        print("⚠ Warning: PNG+JSON annotations not found!")
        print(f"  PNG dir exists: {png_dir.exists()}")
        print(f"  JSON dir exists: {json_dir.exists()}")
    
    print()
    print("="*60)
    
    if dry_run:
        print("DRY RUN complete - no changes were made")
        print("Run with --execute to actually move the folders")
    else:
        print("✓ Deprecation complete!")
        print()
        print("Next steps:")
        print("1. Re-export COCO annotations (will use PNG+JSON format)")
        print("2. Regenerate YOLO training data")
        print("3. You can safely delete the *_deprecated folders after verifying everything works")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Deprecate old annotation formats (bbox and pkl)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Dry run (see what would be done)
  python scripts/deprecate_old_annotations.py --project projects/test
  
  # Actually move the folders
  python scripts/deprecate_old_annotations.py --project projects/test --execute
        """
    )
    parser.add_argument('--project', required=True, help='Path to project directory')
    parser.add_argument('--execute', action='store_true', 
                        help='Actually move folders (default is dry-run)')
    
    args = parser.parse_args()
    
    deprecate_old_annotations(
        project_path=Path(args.project),
        dry_run=not args.execute
    )
