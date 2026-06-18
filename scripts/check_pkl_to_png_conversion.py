#!/usr/bin/env python3
"""
Check if all PKL annotations have corresponding PNG+JSON files

This script scans the project's annotations directory and verifies that
all old PKL format annotations have been converted to the new PNG+JSON format.
"""

import argparse
from pathlib import Path
import sys


def check_pkl_conversion(project_path: Path, verbose: bool = False, show_missing_only: bool = False):
    """
    Check if all PKL annotations have corresponding PNG+JSON files
    
    Args:
        project_path: Path to project directory
        verbose: Whether to print detailed information
        show_missing_only: Only show files that are missing conversions
    """
    project_path = Path(project_path)
    
    # Check if project exists
    if not project_path.exists():
        print(f"Error: Project path does not exist: {project_path}")
        return False
    
    pkl_dir = project_path / 'annotations' / 'pkl'
    png_dir = project_path / 'annotations' / 'png'
    json_dir = project_path / 'annotations' / 'json'
    
    if not pkl_dir.exists():
        print(f"No PKL annotations directory found: {pkl_dir}")
        print("Nothing to check.")
        return True
    
    print(f"Checking PKL annotations in: {project_path}")
    print(f"=" * 70)
    
    # Track statistics
    total_videos = 0
    total_pkl_files = 0
    total_with_png = 0
    total_with_json = 0
    total_complete = 0  # Both PNG and JSON
    videos_with_missing = []
    
    # Scan each video directory
    video_dirs = sorted([d for d in pkl_dir.iterdir() if d.is_dir()])
    
    if not video_dirs:
        print("No video directories found in PKL annotations.")
        return True
    
    for video_dir in video_dirs:
        video_id = video_dir.name
        total_videos += 1
        
        # Get all PKL files for this video
        pkl_files = sorted(video_dir.glob('frame_*.pkl'))
        
        if not pkl_files:
            continue
        
        # Check corresponding PNG and JSON files
        video_pkl_count = len(pkl_files)
        video_has_png = 0
        video_has_json = 0
        video_complete = 0
        missing_conversions = []
        
        for pkl_file in pkl_files:
            frame_name = pkl_file.stem  # e.g., "frame_000123"
            
            # Check for PNG file
            png_file = png_dir / video_id / f"{frame_name}.png"
            has_png = png_file.exists()
            
            # Check for JSON file
            json_file = json_dir / video_id / f"{frame_name}.json"
            has_json = json_file.exists()
            
            if has_png:
                video_has_png += 1
            if has_json:
                video_has_json += 1
            if has_png and has_json:
                video_complete += 1
            else:
                missing_conversions.append({
                    'frame': frame_name,
                    'has_png': has_png,
                    'has_json': has_json
                })
        
        # Update totals
        total_pkl_files += video_pkl_count
        total_with_png += video_has_png
        total_with_json += video_has_json
        total_complete += video_complete
        
        # Report for this video
        if not show_missing_only or missing_conversions:
            print(f"\nVideo: {video_id}")
            print(f"  PKL files: {video_pkl_count}")
            print(f"  With PNG: {video_has_png} ({100*video_has_png/video_pkl_count:.1f}%)")
            print(f"  With JSON: {video_has_json} ({100*video_has_json/video_pkl_count:.1f}%)")
            print(f"  Complete (PNG+JSON): {video_complete} ({100*video_complete/video_pkl_count:.1f}%)")
            
            if missing_conversions:
                videos_with_missing.append(video_id)
                print(f"  ⚠ Missing conversions: {len(missing_conversions)} frames")
                
                if verbose:
                    print(f"  Missing frames:")
                    for missing in missing_conversions[:10]:  # Show first 10
                        status = []
                        if not missing['has_png']:
                            status.append("no PNG")
                        if not missing['has_json']:
                            status.append("no JSON")
                        print(f"    - {missing['frame']}: {', '.join(status)}")
                    
                    if len(missing_conversions) > 10:
                        print(f"    ... and {len(missing_conversions) - 10} more")
    
    # Print summary
    print(f"\n{'=' * 70}")
    print(f"SUMMARY")
    print(f"{'=' * 70}")
    print(f"Total videos scanned: {total_videos}")
    print(f"Total PKL annotation files: {total_pkl_files}")
    print(f"Files with PNG: {total_with_png} ({100*total_with_png/total_pkl_files:.1f}%)")
    print(f"Files with JSON: {total_with_json} ({100*total_with_json/total_pkl_files:.1f}%)")
    print(f"Complete conversions (PNG+JSON): {total_complete} ({100*total_complete/total_pkl_files:.1f}%)")
    print(f"Missing conversions: {total_pkl_files - total_complete} ({100*(total_pkl_files - total_complete)/total_pkl_files:.1f}%)")
    
    if videos_with_missing:
        print(f"\n⚠ Videos with missing conversions ({len(videos_with_missing)}):")
        for vid in videos_with_missing:
            print(f"  - {vid}")
    else:
        print(f"\n✓ All PKL annotations have been converted to PNG+JSON!")
    
    # Return True if all conversions are complete, False otherwise
    all_complete = (total_complete == total_pkl_files)
    return all_complete


def main():
    parser = argparse.ArgumentParser(
        description="Check if all PKL annotations have corresponding PNG+JSON files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Check all conversions in a project
  python check_pkl_to_png_conversion.py --project /path/to/project

  # Verbose output showing missing frames
  python check_pkl_to_png_conversion.py --project /path/to/project --verbose

  # Only show videos with missing conversions
  python check_pkl_to_png_conversion.py --project /path/to/project --missing-only
        """
    )
    
    parser.add_argument(
        '--project',
        type=str,
        required=True,
        help='Path to project directory'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Show detailed information about missing conversions'
    )
    
    parser.add_argument(
        '--missing-only',
        action='store_true',
        help='Only show videos with missing conversions'
    )
    
    args = parser.parse_args()
    
    # Run the check
    all_complete = check_pkl_conversion(
        Path(args.project),
        verbose=args.verbose,
        show_missing_only=args.missing_only
    )
    
    # Exit with non-zero code if conversions are incomplete
    sys.exit(0 if all_complete else 1)


if __name__ == '__main__':
    main()
