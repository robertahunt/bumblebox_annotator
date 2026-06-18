#!/usr/bin/env python3
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.annotation import AnnotationManager

project_path = Path('projects/test')
video_id = 'worker24-2022-08-18_20-12-01'
frame_idx = 0

ann_manager = AnnotationManager()
annotations = ann_manager.load_frame_annotations(project_path, video_id, frame_idx)

print(f"Loaded {len(annotations)} annotations for frame {frame_idx}")
for i, ann in enumerate(annotations):
    print(f"\nAnnotation {i}:")
    print(f"  Keys: {ann.keys()}")
    if 'bbox' in ann:
        print(f"  BBox: {ann['bbox']}")
    if 'mask' in ann:
        import numpy as np
        print(f"  Mask shape: {ann['mask'].shape}, nonzero pixels: {np.sum(ann['mask'] > 0)}")
