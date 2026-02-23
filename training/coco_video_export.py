"""
Enhanced COCO export with video tracking support for Detectron2 training
"""

import json
import numpy as np
from pathlib import Path
from datetime import datetime
import pickle
from typing import List, Dict
from core.annotation import rle_to_mask, mask_to_polygon


def export_coco_per_video(project_path: Path, video_ids: List[str], 
                          split_name: str, class_names: List[str] = None,
                          image_width: int = 1920, image_height: int = 1080):
    """
    Export annotations as separate COCO JSON files per video
    
    Creates one JSON file per video in annotations/coco/{split_name}/ directory.
    This keeps file sizes manageable for large datasets.
    
    Args:
        project_path: Path to project directory
        video_ids: List of video IDs to export
        split_name: Name of split ('train' or 'val')
        class_names: List of class names (default: ['bee'])
        image_width: Default image width
        image_height: Default image height
        
    Returns:
        List of paths to generated COCO JSON files
    """
    if class_names is None:
        class_names = ['bee']
    
    project_path = Path(project_path)
    
    # Create categories (shared across all files)
    categories = []
    for i, class_name in enumerate(class_names):
        categories.append({
            'id': i + 1,
            'name': class_name,
            'supercategory': 'insect'
        })
    
    # Create output directory for this split
    output_dir = project_path / 'annotations' / 'coco' / split_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    exported_files = []
    
    # Process each video separately
    for video_idx, video_name in enumerate(video_ids):
        video_id_int = video_idx + 1
        
        # Initialize COCO structure for this video
        coco_data = {
            'info': {
                'description': f'{video_name} - {split_name} split',
                'version': '2.0',
                'year': datetime.now().year,
                'contributor': 'BeeWhere Annotator',
                'date_created': datetime.now().isoformat(),
                'video_name': video_name,
                'split': split_name
            },
            'licenses': [{
                'id': 1,
                'name': 'Unknown',
                'url': ''
            }],
            'categories': categories,
            'images': [],
            'annotations': []
        }
        
        # Load annotations for this video
        video_ann_dir = project_path / 'annotations/pkl' / video_name
        if not video_ann_dir.exists():
            continue
        
        # Load frame selection metadata
        video_frames_dir = project_path / 'frames' / video_name
        metadata_file = video_frames_dir / 'video_metadata.json'
        selected_frame_indices = set()
        if metadata_file.exists():
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
                selected_frame_indices = set(metadata.get('selected_frames', []))
        
        # If no frames are selected, skip this video
        if not selected_frame_indices:
            continue
        
        # Process frames
        frame_files = sorted(video_ann_dir.glob('frame_*.pkl'))
        
        image_id = 1
        annotation_id = 1
        
        for ann_file in frame_files:
            # Extract frame index from filename
            frame_idx = int(ann_file.stem.split('_')[1])
            
            # Skip frames not selected for this split
            if frame_idx not in selected_frame_indices:
                continue
            
            # Load annotations
            try:
                with open(ann_file, 'rb') as f:
                    compressed_annotations = pickle.load(f)
            except Exception as e:
                print(f"Warning: Failed to load {ann_file}: {e}")
                continue
            
            # Decompress masks
            annotations = []
            for ann in compressed_annotations:
                decompressed_ann = ann.copy()
                if 'mask_rle' in ann:
                    decompressed_ann['mask'] = rle_to_mask(ann['mask_rle'])
                    del decompressed_ann['mask_rle']
                elif 'mask' not in ann:
                    continue
                annotations.append(decompressed_ann)
            
            # Add image entry
            image_file = f"frames/{video_name}/frame_{frame_idx:06d}.jpg"
            coco_data['images'].append({
                'id': image_id,
                'file_name': image_file,
                'width': image_width,
                'height': image_height,
                'license': 1,
                'date_captured': '',
                'frame_index': frame_idx
            })
            
            # Add annotations for this frame
            for ann in annotations:
                if 'mask' in ann:
                    mask = ann['mask']
                    
                    # Get bounding box from mask
                    coords = np.where(mask > 0)
                    if len(coords[0]) == 0:
                        continue  # Empty mask
                    
                    y_min, y_max = coords[0].min(), coords[0].max()
                    x_min, x_max = coords[1].min(), coords[1].max()
                    bbox_width = int(x_max - x_min + 1)
                    bbox_height = int(y_max - y_min + 1)
                    
                    # Create annotation with polygon segmentation
                    coco_ann = {
                        'id': annotation_id,
                        'image_id': image_id,
                        'category_id': 1,
                        'bbox': [int(x_min), int(y_min), bbox_width, bbox_height],
                        'area': int(np.sum(mask > 0)),
                        'iscrowd': 0
                    }
                    
                    # Add segmentation
                    polygons = mask_to_polygon(mask)
                    if polygons:
                        coco_ann['segmentation'] = polygons
                    else:
                        continue  # Skip if no valid polygons
                    
                    # Add tracking ID
                    if 'mask_id' in ann:
                        coco_ann['track_id'] = ann['mask_id']
                        coco_ann['instance_id'] = ann['mask_id']
                    
                    coco_data['annotations'].append(coco_ann)
                    annotation_id += 1
            
            image_id += 1
        
        # Only save if this video has images
        if len(coco_data['images']) > 0:
            output_path = output_dir / f'{video_name}.json'
            with open(output_path, 'w') as f:
                json.dump(coco_data, f, indent=2)
            
            print(f"Exported: {output_path.name} ({len(coco_data['images'])} images, {len(coco_data['annotations'])} annotations)")
            exported_files.append(output_path)
    
    print(f"\nExported {len(exported_files)} video files for {split_name} split")
    return exported_files


def export_coco_with_tracking(project_path: Path, video_ids: List[str], 
                               split_name: str, class_names: List[str] = None,
                               image_width: int = 1920, image_height: int = 1080):
    """
    Export annotations in enhanced COCO format with video tracking support
    
    This format includes:
    - video_id in images
    - track_id in annotations (for instance tracking)
    - frame_index for temporal ordering
    - video sequences information
    
    Args:
        project_path: Path to project directory
        video_ids: List of video IDs to include
        split_name: Name of split ('train' or 'val')
        class_names: List of class names (default: ['bee'])
        image_width: Default image width
        image_height: Default image height
        
    Returns:
        Path to generated COCO JSON file
    """
    if class_names is None:
        class_names = ['bee']
    
    project_path = Path(project_path)
    
    coco_data = {
        'info': {
            'description': f'Video Instance Segmentation - {split_name} split',
            'version': '2.0',
            'year': datetime.now().year,
            'contributor': 'BeeWhere Annotator',
            'date_created': datetime.now().isoformat(),
            'has_tracking': True  # Flag indicating tracking information is present
        },
        'licenses': [{
            'id': 1,
            'name': 'Unknown',
            'url': ''
        }],
        'categories': [],
        'videos': [],  # New: Video-level metadata
        'images': [],
        'annotations': []
    }
    
    # Add categories
    for i, class_name in enumerate(class_names):
        coco_data['categories'].append({
            'id': i + 1,
            'name': class_name,
            'supercategory': 'insect'
        })
    
    image_id = 1
    annotation_id = 1
    video_id_map = {}  # Map video_name -> video_id_int
    
    # Process each video
    for video_idx, video_name in enumerate(video_ids):
        video_id_int = video_idx + 1
        video_id_map[video_name] = video_id_int
        
        # Add video metadata
        video_ann_dir = project_path / 'annotations/pkl' / video_name
        if not video_ann_dir.exists():
            continue
        
        # Load frame selection metadata
        video_frames_dir = project_path / 'frames' / video_name
        metadata_file = video_frames_dir / 'video_metadata.json'
        selected_frame_indices = set()
        if metadata_file.exists():
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
                selected_frame_indices = set(metadata.get('selected_frames', []))
        
        # If no frames are selected, skip this video
        if not selected_frame_indices:
            continue
        
        # Count frames in this video
        frame_files = sorted(video_ann_dir.glob('frame_*.pkl'))
        
        coco_data['videos'].append({
            'id': video_id_int,
            'name': video_name,
            'num_frames': len(frame_files),
            'width': image_width,
            'height': image_height
        })
        
        # Process each frame in the video
        for ann_file in frame_files:
            # Extract frame index from filename
            frame_idx = int(ann_file.stem.split('_')[1])
            
            # Skip frames that are not selected for this split
            if frame_idx not in selected_frame_indices:
                continue
            
            # Load annotations
            try:
                with open(ann_file, 'rb') as f:
                    compressed_annotations = pickle.load(f)
            except Exception as e:
                print(f"Warning: Failed to load {ann_file}: {e}")
                continue
            
            # Decompress masks
            annotations = []
            for ann in compressed_annotations:
                decompressed_ann = ann.copy()
                if 'mask_rle' in ann:
                    decompressed_ann['mask'] = rle_to_mask(ann['mask_rle'])
                    del decompressed_ann['mask_rle']
                elif 'mask' not in ann:
                    continue  # Skip if no mask
                annotations.append(decompressed_ann)
            
            if not annotations:
                # Still add image entry even if no annotations (for validation)
                image_file = f"frames/{video_name}/frame_{frame_idx:06d}.jpg"
                coco_data['images'].append({
                    'id': image_id,
                    'file_name': image_file,
                    'width': image_width,
                    'height': image_height,
                    'license': 1,
                    'date_captured': '',
                    'video_id': video_id_int,  # Link to video
                    'frame_index': frame_idx   # Temporal ordering
                })
                image_id += 1
                continue
            
            # Add image entry
            image_file = f"frames/{video_name}/frame_{frame_idx:06d}.jpg"
            coco_data['images'].append({
                'id': image_id,
                'file_name': image_file,
                'width': image_width,
                'height': image_height,
                'license': 1,
                'date_captured': '',
                'video_id': video_id_int,  # Link to video
                'frame_index': frame_idx   # Temporal ordering
            })
            
            # Add annotations for this frame
            for ann in annotations:
                if 'mask' in ann:
                    mask = ann['mask']
                    
                    # Get bounding box from mask
                    coords = np.where(mask > 0)
                    if len(coords[0]) == 0:
                        continue  # Empty mask
                    
                    y_min, y_max = coords[0].min(), coords[0].max()
                    x_min, x_max = coords[1].min(), coords[1].max()
                    bbox_width = int(x_max - x_min + 1)
                    bbox_height = int(y_max - y_min + 1)
                    
                    # Create annotation with polygon segmentation
                    coco_ann = {
                        'id': annotation_id,
                        'image_id': image_id,
                        'category_id': 1,  # Default to first category
                        'bbox': [int(x_min), int(y_min), bbox_width, bbox_height],
                        'area': int(np.sum(mask > 0)),
                        'iscrowd': 0
                    }
                    
                    # Add segmentation
                    polygons = mask_to_polygon(mask)
                    if polygons:
                        coco_ann['segmentation'] = polygons
                    else:
                        continue  # Skip if no valid polygons
                    
                    # Add tracking ID (instance ID from annotations)
                    if 'mask_id' in ann:
                        # Use mask_id as track_id for video instance tracking
                        # Format: video_id * 10000 + instance_id for global uniqueness
                        track_id = video_id_int * 10000 + ann['mask_id']
                        coco_ann['track_id'] = track_id
                        coco_ann['instance_id'] = ann['mask_id']  # Local to video
                    
                    coco_data['annotations'].append(coco_ann)
                    annotation_id += 1
            
            image_id += 1
    
    # Save to file
    output_dir = project_path / 'annotations/coco'
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f'instances_{split_name}_tracking.json'
    
    with open(output_path, 'w') as f:
        json.dump(coco_data, f, indent=2)
    
    print(f"Exported COCO with tracking: {output_path}")
    print(f"  Videos: {len(coco_data['videos'])}")
    print(f"  Images: {len(coco_data['images'])}")
    print(f"  Annotations: {len(coco_data['annotations'])}")
    
    return output_path


def get_dataset_statistics(coco_file: Path) -> Dict:
    """Get statistics from COCO dataset"""
    with open(coco_file) as f:
        coco_data = json.load(f)
    
    num_images = len(coco_data['images'])
    num_annotations = len(coco_data['annotations'])
    num_videos = len(coco_data.get('videos', []))
    
    # Count unique tracks
    track_ids = set()
    for ann in coco_data['annotations']:
        if 'track_id' in ann:
            track_ids.add(ann['track_id'])
    
    # Calculate annotations per image
    avg_anns_per_image = num_annotations / max(num_images, 1)
    
    return {
        'num_videos': num_videos,
        'num_images': num_images,
        'num_annotations': num_annotations,
        'num_unique_tracks': len(track_ids),
        'avg_annotations_per_image': avg_anns_per_image
    }
