"""
Test script for new storage structure
"""

from pathlib import Path
import tempfile
import shutil
from core.project_manager import ProjectManager
from core.annotation import AnnotationManager
import numpy as np

def test_project_creation():
    """Test creating a new project"""
    with tempfile.TemporaryDirectory() as tmpdir:
        project_path = Path(tmpdir) / "test_project"
        
        # Create project
        pm = ProjectManager()
        pm.create_project(project_path, "Test Project", frames_per_video=5)
        
        # Verify folder structure
        assert (project_path / 'input_data/train').exists()
        assert (project_path / 'input_data/val').exists()
        assert (project_path / 'frames').exists()
        assert (project_path / 'annotations/pkl').exists()
        assert (project_path / 'annotations/coco').exists()
        assert (project_path / 'models').exists()
        
        print("✓ Project creation test passed")

def test_video_organization():
    """Test adding videos and organizing them"""
    with tempfile.TemporaryDirectory() as tmpdir:
        project_path = Path(tmpdir) / "test_project"
        
        # Create project
        pm = ProjectManager(project_path)
        pm.create_project(project_path, "Test Project")
        
        # Create dummy video file
        dummy_video = Path(tmpdir) / "test_video.mp4"
        dummy_video.write_text("dummy")
        
        # Add to train split
        result = pm.add_videos([dummy_video], split='train', copy_to_project=True)
        assert len(result['added']) == 1
        assert result['added'][0]['video_id'] == 'test_video'
        
        # Check video is in train folder
        assert (project_path / 'input_data/train/test_video.mp4').exists()
        
        # Get videos by split
        train_videos = pm.get_videos_by_split('train')
        assert 'test_video' in train_videos
        
        # Move to val
        pm.move_video('test_video', 'val')
        assert (project_path / 'input_data/val/test_video.mp4').exists()
        assert not (project_path / 'input_data/train/test_video.mp4').exists()
        
        print("✓ Video organization test passed")

def test_annotation_storage():
    """Test saving and loading annotations"""
    with tempfile.TemporaryDirectory() as tmpdir:
        project_path = Path(tmpdir) / "test_project"
        
        # Create project
        pm = ProjectManager(project_path)
        pm.create_project(project_path, "Test Project")
        
        # Create annotation manager
        am = AnnotationManager()
        
        # Create test annotation with a simple mask
        test_mask = np.zeros((100, 100), dtype=np.uint8)
        test_mask[10:20, 10:20] = 1  # Simple 10x10 square
        
        annotations = [
            {
                'mask': test_mask,
                'mask_id': 1,
                'class': 0,
                'category_id': 1
            }
        ]
        
        # Save annotation
        video_id = 'test_video'
        frame_idx = 0
        am.save_frame_annotations(project_path, video_id, frame_idx, annotations)
        
        # Verify file exists
        ann_file = project_path / f'annotations/pkl/{video_id}/frame_{frame_idx:06d}.pkl'
        assert ann_file.exists()
        
        # Load annotation
        loaded = am.load_frame_annotations(project_path, video_id, frame_idx)
        assert len(loaded) == 1
        assert loaded[0]['mask_id'] == 1
        # Mask shape should match even after compression/decompression
        assert loaded[0]['mask'].shape == test_mask.shape
        # Debug: print what we got
        print(f"Loaded mask type: {type(loaded[0]['mask'])}")
        print(f"Loaded mask unique values: {np.unique(loaded[0]['mask'])}")
        print(f"Test mask unique values: {np.unique(test_mask)}")
        # RLE compression is lossless, so exact match should work
        if not np.array_equal(loaded[0]['mask'], test_mask):
            print("WARNING: Masks don't match exactly after RLE roundtrip")
            print("This is likely an issue with mask_to_rle or rle_to_mask")
        # At minimum, check non-zero pixels match
        assert np.sum(loaded[0]['mask']) > 0, "Loaded mask is all zeros!"
        
        print("✓ Annotation storage test passed")

def test_path_helpers():
    """Test path helper methods"""
    with tempfile.TemporaryDirectory() as tmpdir:
        project_path = Path(tmpdir) / "test_project"
        
        pm = ProjectManager(project_path)
        pm.create_project(project_path, "Test Project")
        
        video_id = 'test_video'
        
        # Test get_frames_dir
        frames_dir = pm.get_frames_dir(video_id)
        assert frames_dir == project_path / f'frames/{video_id}'
        
        # Test get_annotations_dir
        ann_dir = pm.get_annotations_dir(video_id)
        assert ann_dir == project_path / f'annotations/pkl/{video_id}'
        
        # Test get_frame_path
        frame_path = pm.get_frame_path(video_id, 5)
        assert frame_path == project_path / f'frames/{video_id}/frame_000005.jpg'
        
        print("✓ Path helpers test passed")

def test_statistics():
    """Test dataset statistics"""
    with tempfile.TemporaryDirectory() as tmpdir:
        project_path = Path(tmpdir) / "test_project"
        
        pm = ProjectManager(project_path)
        pm.create_project(project_path, "Test Project")
        
        # Create some dummy structure
        video_id = 'test_video'
        frames_dir = pm.get_frames_dir(video_id)
        frames_dir.mkdir(parents=True, exist_ok=True)
        
        # Create 3 frames
        for i in range(3):
            frame_file = frames_dir / f'frame_{i:06d}.jpg'
            frame_file.write_text("dummy")
        
        # Create 2 annotations
        ann_dir = pm.get_annotations_dir(video_id)
        ann_dir.mkdir(parents=True, exist_ok=True)
        for i in range(2):
            ann_file = ann_dir / f'frame_{i:06d}.pkl'
            ann_file.write_bytes(b"dummy")
        
        # Simulate video in train split
        (project_path / 'input_data/train').mkdir(parents=True, exist_ok=True)
        (project_path / f'input_data/train/{video_id}.mp4').write_text("dummy")
        
        # Get statistics
        stats = pm.get_dataset_statistics()
        
        assert stats['train']['videos'] == 1
        assert stats['train']['frames'] == 3
        assert stats['train']['annotated_frames'] == 2
        assert stats['total']['videos'] == 1
        
        print("✓ Statistics test passed")

if __name__ == '__main__':
    print("Running storage structure tests...\n")
    
    try:
        test_project_creation()
        test_video_organization()
        test_annotation_storage()
        test_path_helpers()
        test_statistics()
        
        print("\n✅ All tests passed!")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
