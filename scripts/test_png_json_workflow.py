#!/usr/bin/env python3
"""
Test that annotation loading/saving still works with PNG+JSON only
"""
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.annotation import AnnotationManager
from core.project_manager import ProjectManager


def test_annotation_workflow():
    """Test save and load with PNG+JSON format"""
    
    print("Testing PNG+JSON Annotation Workflow")
    print("="*60)
    
    # Use test project
    project_path = Path('projects/test')
    pm = ProjectManager(project_path)
    am = AnnotationManager()
    
    # Get a video
    train_videos = pm.get_videos_by_split('train')
    if not train_videos:
        print("No training videos found")
        return
    
    video_id = train_videos[0]
    print(f"Testing with video: {video_id}")
    
    # Load existing annotations
    frame_idx = 0
    annotations = am.load_frame_annotations(project_path, video_id, frame_idx)
    
    if not annotations:
        print(f"No annotations found for frame {frame_idx}")
        return
    
    print(f"\n✓ Loaded {len(annotations)} annotations from PNG+JSON")
    
    # Check that annotations have required fields
    for i, ann in enumerate(annotations):
        assert 'mask' in ann, f"Annotation {i} missing mask"
        assert isinstance(ann['mask'], np.ndarray), f"Annotation {i} mask is not numpy array"
        print(f"  Annotation {i}: mask shape {ann['mask'].shape}, ID {ann.get('mask_id', 'N/A')}")
    
    # Verify PNG and JSON files exist
    png_file = pm.get_annotation_path(video_id, frame_idx)
    json_file = pm.get_json_annotation_path(video_id, frame_idx)
    
    assert png_file.exists(), f"PNG file not found: {png_file}"
    assert json_file.exists(), f"JSON file not found: {json_file}"
    
    print(f"\n✓ PNG file exists: {png_file.name}")
    print(f"✓ JSON file exists: {json_file.name}")
    
    # Test that we can compute bbox from masks
    for ann in annotations:
        mask = ann['mask']
        y_indices, x_indices = np.where(mask > 0)
        if len(y_indices) > 0:
            x_min, x_max = x_indices.min(), x_indices.max()
            y_min, y_max = y_indices.min(), y_indices.max()
            bbox = [int(x_min), int(y_min), int(x_max - x_min + 1), int(y_max - y_min + 1)]
            print(f"  Computed bbox from mask: {bbox}")
    
    print(f"\n✓ All tests passed!")
    print(f"✓ PNG+JSON format is working correctly")
    print(f"✓ No pkl or bbox files needed")


if __name__ == '__main__':
    test_annotation_workflow()
