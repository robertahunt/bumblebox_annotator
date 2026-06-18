#!/usr/bin/env python3
"""
Test COCO export with PNG+JSON annotations
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from training.coco_video_export import export_coco_per_video
from core.project_manager import ProjectManager

def test_export():
    project_path = Path('projects/test')
    
    # Get train videos
    pm = ProjectManager(project_path)
    train_videos = pm.get_videos_by_split('train')
    
    print(f"Found {len(train_videos)} training videos")
    print(f"Testing export for first 3 videos: {train_videos[:3]}")
    
    # Export COCO for first 3 videos as a test
    exported = export_coco_per_video(
        project_path=project_path,
        video_ids=train_videos[:3],
        split_name='train_test',
        class_names=['bee']
    )
    
    print(f"\nExported {len(exported)} files:")
    for file_path in exported:
        import json
        with open(file_path) as f:
            data = json.load(f)
        print(f"  {file_path.name}: {len(data['images'])} images, {len(data['annotations'])} annotations")

if __name__ == '__main__':
    test_export()
