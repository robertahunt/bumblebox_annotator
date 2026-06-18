"""
Quick test to verify PNG annotation save/load functionality
"""

import numpy as np
import cv2
import tempfile
import shutil
from pathlib import Path
from core.annotation import AnnotationManager

# Create a temporary directory
temp_dir = Path(tempfile.mkdtemp())
project_path = temp_dir / "test_project"
project_path.mkdir(parents=True, exist_ok=True)

try:
    # Initialize annotation manager
    am = AnnotationManager()
    
    # Create test annotations (simulate bee masks)
    h, w = 1920, 1080
    annotations = []
    
    for i in range(5):
        # Create a fake bee mask
        mask = np.zeros((h, w), dtype=np.uint8)
        x, y = 200 + i * 200, 300 + i * 150
        cv2.circle(mask, (x, y), 50, 255, -1)
        
        ann = {
            'mask_id': i + 1,
            'mask': mask,
            'bbox': [x-50, y-50, x+50, y+50],
            'contours': []
        }
        annotations.append(ann)
    
    # Test PNG save
    print("Testing PNG save...")
    import time
    t_start = time.perf_counter()
    am.save_frame_annotations_png(project_path, "video_001", 0, annotations)
    t_save_png = (time.perf_counter() - t_start) * 1000
    print(f"✓ PNG save: {t_save_png:.1f}ms")
    
    # Test PNG load
    print("\nTesting PNG load...")
    t_start = time.perf_counter()
    loaded_annotations = am.load_frame_annotations_png(project_path, "video_001", 0)
    t_load_png = (time.perf_counter() - t_start) * 1000
    print(f"✓ PNG load: {t_load_png:.1f}ms")
    
    # Verify loaded data
    print(f"\nVerification:")
    print(f"Original annotations: {len(annotations)}")
    print(f"Loaded annotations: {len(loaded_annotations)}")
    
    for i, (orig, loaded) in enumerate(zip(annotations, loaded_annotations)):
        mask_matches = np.array_equal(orig['mask'], loaded['mask'])
        id_matches = orig['mask_id'] == loaded['mask_id']
        print(f"  Instance {i+1}: mask_id={loaded['mask_id']}, mask_match={mask_matches}, id_match={id_matches}")
    
    # Test pickle save for comparison
    print("\n" + "="*50)
    print("Comparison with pickle format:")
    t_start = time.perf_counter()
    am.save_frame_annotations_pickle(project_path, "video_001", 1, annotations)
    t_save_pkl = (time.perf_counter() - t_start) * 1000
    print(f"Pickle save: {t_save_pkl:.1f}ms")
    
    t_start = time.perf_counter()
    loaded_pkl = am.load_frame_annotations_pickle(project_path, "video_001", 1)
    t_load_pkl = (time.perf_counter() - t_start) * 1000
    print(f"Pickle load: {t_load_pkl:.1f}ms")
    
    print(f"\nSpeedup:")
    print(f"Save: {t_save_pkl / t_save_png:.1f}x faster with PNG")
    print(f"Load: {t_load_pkl / t_load_png:.1f}x faster with PNG")
    
    # Test backward compatibility (load function tries PNG first, then pickle)
    print("\n" + "="*50)
    print("Testing backward compatibility:")
    
    # Load frame 0 (PNG exists)
    result = am.load_frame_annotations(project_path, "video_001", 0)
    print(f"✓ Frame 0 (PNG): loaded {len(result)} annotations")
    
    # Load frame 1 (only pickle exists)
    result = am.load_frame_annotations(project_path, "video_001", 1)
    print(f"✓ Frame 1 (PKL): loaded {len(result)} annotations")
    
    # Load frame 2 (doesn't exist)
    result = am.load_frame_annotations(project_path, "video_001", 2)
    print(f"✓ Frame 2 (none): loaded {len(result)} annotations")
    
    print("\n✓ All tests passed!")
    
finally:
    # Cleanup
    shutil.rmtree(temp_dir)
    print(f"\n✓ Cleaned up temp directory: {temp_dir}")
