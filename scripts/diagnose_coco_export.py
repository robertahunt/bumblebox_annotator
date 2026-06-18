#!/usr/bin/env python3
"""
Diagnose COCO export issues - why we're getting 0 annotations
"""
import json
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from core.annotation import rle_to_mask
from core.project_manager import ProjectManager

def diagnose_coco_export(project_path):
    """
    Check why COCO export might be producing 0 annotations
    
    Args:
        project_path: Path to project directory
    """
    project_path = Path(project_path)
    
    print("="*60)
    print("COCO Export Diagnostics")
    print("="*60)
    
    # Initialize project manager
    pm = ProjectManager(project_path)
    
    # Check 1: Video splits (via project_manager)
    train_videos = pm.get_videos_by_split('train')
    val_videos = pm.get_videos_by_split('val')
    test_videos = pm.get_videos_by_split('test')
    
    print(f"\nVideo splits:")
    print(f"  train: {len(train_videos)} videos")
    print(f"  val: {len(val_videos)} videos")
    print(f"  test: {len(test_videos)} videos")
    
    # Check each video in train split
    if not train_videos:
        print("\n⚠️  No videos in train split!")
        return
    
    print(f"\n{'='*60}")
    print(f"Checking {len(train_videos)} training videos:")
    print(f"{'='*60}")
    
    for video_id in train_videos[:5]:  # Check first 5
        print(f"\nVideo: {video_id}")
        print("-" * 40)
        
        # Check 1: Annotations directories
        png_dir = project_path / 'annotations' / 'png' / video_id
        json_dir = project_path / 'annotations' / 'json' / video_id
        
        if not png_dir.exists():
            print(f"  ✗ No PNG annotations directory: {png_dir.name}")
            continue
        else:
            print(f"  ✓ PNG annotations directory exists")
        
        if not json_dir.exists():
            print(f"  ✗ No JSON annotations directory: {json_dir.name}")
            continue
        else:
            print(f"  ✓ JSON annotations directory exists")
        
        # Check 2: Annotation files
        png_files = list(png_dir.glob('frame_*.png'))
        json_files = list(json_dir.glob('frame_*.json'))
        print(f"  ✓ Found {len(png_files)} PNG files, {len(json_files)} JSON files")
        if len(json_files) == 0:
            print(f"    ⚠️  No annotation files!")
            continue
        
        # Check 3: Video metadata and selected frames
        video_metadata_file = project_path / 'frames' / video_id / 'video_metadata.json'
        if not video_metadata_file.exists():
            print(f"  ✗ No video_metadata.json: {video_metadata_file.name}")
            selected_frames = set()
        else:
            with open(video_metadata_file) as f:
                metadata = json.load(f)
            selected_frames = set(metadata.get('selected_frames', []))
            print(f"  ✓ Video metadata exists")
            print(f"  ✓ Selected frames: {len(selected_frames)}")
            if selected_frames:
                sample_frames = sorted(list(selected_frames))[:5]
                print(f"    Sample: {sample_frames}")
        
        # Check 4: Which annotation files match selected frames
        json_frame_indices = set()
        for json_file in json_files:
            frame_idx = int(json_file.stem.split('_')[1])
            json_frame_indices.add(frame_idx)
        
        if selected_frames:
            matching = json_frame_indices & selected_frames
            print(f"  ✓ Matching frames (annotated AND selected): {len(matching)}")
            if len(matching) == 0:
                print(f"    ⚠️  NO OVERLAP between annotated and selected frames!")
                print(f"       Annotated frames: {sorted(list(json_frame_indices))[:10]}")
                print(f"       Selected frames: {sorted(list(selected_frames))[:10]}")
        else:
            print(f"  ⚠️  No selected frames - will export nothing!")
        
        # Check 5: Load a sample annotation file
        sample_file = sorted(json_files)[0]
        print(f"\n  Checking sample annotation file: {sample_file.name}")
        try:
            with open(sample_file, 'r') as f:
                json_data = json.load(f)
            
            print(f"    ✓ Loaded successfully")
            print(f"    ✓ Contains {len(json_data)} annotations")
            
            if len(json_data) > 0:
                # Check if corresponding PNG exists
                png_file = png_dir / sample_file.name.replace('.json', '.png')
                if png_file.exists():
                    import cv2
                    combined_mask = cv2.imread(str(png_file), cv2.IMREAD_UNCHANGED)
                    if combined_mask is not None:
                        print(f"    ✓ PNG mask loaded: shape {combined_mask.shape}, dtype {combined_mask.dtype}")
                    else:
                        print(f"    ✗ Failed to load PNG mask")
                else:
                    print(f"    ✗ PNG file missing: {png_file.name}")
        except Exception as e:
            print(f"    ✗ Failed to load: {e}")
    
    # Check COCO output
    print(f"\n{'='*60}")
    print("Checking existing COCO exports:")
    print(f"{'='*60}")
    
    coco_train_dir = project_path / 'annotations' / 'coco' / 'train'
    if not coco_train_dir.exists():
        print(f"\n  No train COCO directory yet: {coco_train_dir}")
    else:
        coco_files = list(coco_train_dir.glob('*.json'))
        print(f"\n  Found {len(coco_files)} COCO JSON files")
        
        for coco_file in coco_files[:5]:
            with open(coco_file) as f:
                coco_data = json.load(f)
            
            print(f"\n  {coco_file.name}:")
            print(f"    Images: {len(coco_data['images'])}")
            print(f"    Annotations: {len(coco_data['annotations'])}")
            
            if len(coco_data['annotations']) == 0 and len(coco_data['images']) > 0:
                print(f"    ⚠️  Has images but NO annotations!")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Diagnose COCO export issues')
    parser.add_argument('--project', default='projects/test', help='Project path')
    args = parser.parse_args()
    
    diagnose_coco_export(args.project)
