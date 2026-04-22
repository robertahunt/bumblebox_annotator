"""
Enhanced COCO export with video tracking support for Detectron2 training
"""

import json
import numpy as np
import cv2
from pathlib import Path
from datetime import datetime
import pickle
from typing import List, Dict
from core.annotation import rle_to_mask, mask_to_polygon


def export_coco_per_video(project_path: Path, video_ids: List[str], 
                          split_name: str, class_names: List[str] = None,
                          image_width: int = 1920, image_height: int = 1080,
                          progress_callback=None, cancel_check=None):
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
        progress_callback: Optional callback(current, total, video_name) for progress updates
        cancel_check: Optional callable that returns True if export should be cancelled
        
    Returns:
        List of paths to generated COCO JSON files (or None if cancelled)
    """
    if class_names is None:
        class_names = ['bee', 'hive', 'chamber', 'pollen']
    
    project_path = Path(project_path)
    
    # Create category name to ID mapping
    category_name_to_id = {name: i + 1 for i, name in enumerate(class_names)}
    
    # Create categories (shared across all files)
    supercategories = {'bee': 'insect', 'hive': 'structure', 'chamber': 'structure', 'pollen': 'resource'}
    categories = []
    for i, class_name in enumerate(class_names):
        categories.append({
            'id': i + 1,
            'name': class_name,
            'supercategory': supercategories.get(class_name, 'object')
        })
    
    # Create output directory for this split
    output_dir = project_path / 'annotations' / 'coco' / split_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    exported_files = []
    
    # Process each video separately
    for video_idx, video_name in enumerate(video_ids):
        # Check for cancellation
        if cancel_check and cancel_check():
            print("Export cancelled by user")
            return None
        
        # Report progress
        if progress_callback:
            progress_callback(video_idx, len(video_ids), video_name)
        
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
        
        # Load annotations for this video (PNG+JSON format for masks, bbox folder for bbox-only)
        video_png_dir = project_path / 'annotations/png' / video_name
        video_json_dir = project_path / 'annotations/json' / video_name
        video_bbox_dir = project_path / 'annotations/bbox' / video_name
        
        # Check if annotation directories exist
        if not video_png_dir.exists() or not video_json_dir.exists():
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
        
        # Get actual video frame dimensions from the first saved frame
        actual_frame_width = image_width  # Default fallback
        actual_frame_height = image_height  # Default fallback
        for frame_idx in sorted(selected_frame_indices):
            frame_path = video_frames_dir / f'frame_{frame_idx:06d}.jpg'
            if frame_path.exists():
                try:
                    frame_img = cv2.imread(str(frame_path))
                    if frame_img is not None:
                        actual_frame_height, actual_frame_width = frame_img.shape[:2]
                        break  # Got dimensions, stop checking
                except Exception as e:
                    print(f"Warning: Could not read frame {frame_path}: {e}")
                    continue
        
        # Load video-level annotations (chamber and hive) that apply to all frames
        video_level_annotations = []
        video_annotations_json = video_json_dir / 'video_annotations.json'
        if video_annotations_json.exists():
            try:
                with open(video_annotations_json, 'r') as f:
                    video_json_data = json.load(f)
                
                # Handle both old (list) and new (dict) formats
                if isinstance(video_json_data, dict):
                    # New format: {'annotations': [...], 'aruco_tracking': {...}, 'metadata': {...}}
                    video_annotations_list = video_json_data.get('annotations', [])
                else:
                    # Old format: direct list of annotations
                    video_annotations_list = video_json_data
                
                # Load per-category mask PNGs for video-level annotations
                category_masks = {}
                for category in ('chamber', 'hive'):
                    cat_png = video_png_dir / f'video_annotations_{category}.png'
                    if cat_png.exists():
                        category_masks[category] = cv2.imread(str(cat_png), cv2.IMREAD_UNCHANGED)
                
                # Reconstruct video-level annotations with masks
                for ann_meta in video_annotations_list:
                    mask_id = ann_meta.get('mask_id', 0)
                    category = ann_meta.get('category', 'chamber')
                    
                    # Ensure category_id is correctly set based on category name
                    # This overrides any incorrect category_id that might be in the metadata
                    correct_category_id = category_name_to_id.get(category, 1)
                    
                    if not ann_meta.get('bbox_only', False) and mask_id > 0:
                        cat_mask = category_masks.get(category)
                        if cat_mask is not None:
                            # Extract this instance's mask
                            instance_mask = (cat_mask == mask_id).astype(np.uint8)
                            if np.any(instance_mask > 0):
                                ann = ann_meta.copy()
                                ann['mask'] = instance_mask
                                ann['category_id'] = correct_category_id  # Force correct category_id
                                video_level_annotations.append(ann)
                        else:
                            print(f"Warning: No mask PNG found for {category} annotation {mask_id} in {video_name}")
                    elif ann_meta.get('bbox_only', False):
                        # Include bbox-only video-level annotations
                        ann = ann_meta.copy()
                        ann['category_id'] = correct_category_id  # Force correct category_id
                        video_level_annotations.append(ann)
            except Exception as e:
                print(f"Warning: Failed to load video-level annotations for {video_name}: {e}")
        
        # Process frames - get all JSON annotation files
        frame_files = sorted(video_json_dir.glob('frame_*.json'))
        
        image_id = 1
        annotation_id = 1
        
        for json_file in frame_files:
            # Extract frame index from filename
            frame_idx = int(json_file.stem.split('_')[1])
            
            # Skip frames not selected for this split
            if frame_idx not in selected_frame_indices:
                continue
            
            # Load annotations from PNG+JSON and bbox folder
            png_file = video_png_dir / f'frame_{frame_idx:06d}.png'
            bbox_file = video_bbox_dir / f'frame_{frame_idx:06d}.json' if video_bbox_dir.exists() else None
            
            # Load mask-based annotations
            mask_annotations = []
            if png_file.exists() and json_file.exists():
                try:
                    # Load combined mask
                    combined_mask = cv2.imread(str(png_file), cv2.IMREAD_UNCHANGED)
                    if combined_mask is not None:
                        # Load metadata
                        with open(json_file, 'r') as f:
                            json_data = json.load(f)
                        
                        # Reconstruct annotations with individual masks
                        for ann_meta in json_data:
                            mask_id = ann_meta.get('mask_id', ann_meta.get('instance_id', 0))
                            if mask_id > 0:
                                # Extract this instance's mask
                                instance_mask = (combined_mask == mask_id).astype(np.uint8)
                                
                                # Create full annotation
                                ann = ann_meta.copy()
                                ann['mask'] = instance_mask
                                mask_annotations.append(ann)
                except Exception as e:
                    print(f"Warning: Failed to load mask annotations from {json_file}: {e}")
            
            # Load bbox-only annotations
            bbox_only_annotations = []
            if bbox_file and bbox_file.exists():
                try:
                    with open(bbox_file, 'r') as f:
                        bbox_data = json.load(f)
                    
                    # Filter to only bbox-only annotations (not derived from masks)
                    for ann in bbox_data:
                        if ann.get('bbox_only', False) and not ann.get('from_mask', False):
                            bbox_only_annotations.append(ann)
                except Exception as e:
                    print(f"Warning: Failed to load bbox annotations from {bbox_file}: {e}")
            
            # Combine per-frame annotations (mask-based and bbox-only)
            per_frame_annotations = mask_annotations + bbox_only_annotations
            
            # Skip frames with no per-frame annotations
            # Video-level annotations (chamber/hive) are only added to frames that have bee annotations
            if not per_frame_annotations:
                continue
            
            # Add video-level annotations (chamber/hive) to frames that have bee annotations
            annotations = per_frame_annotations + video_level_annotations
            
            # Use the actual frame dimensions we determined earlier
            frame_width = actual_frame_width
            frame_height = actual_frame_height
            
            # Add image entry
            image_file = f"frames/{video_name}/frame_{frame_idx:06d}.jpg"
            coco_data['images'].append({
                'id': image_id,
                'file_name': image_file,
                'width': frame_width,
                'height': frame_height,
                'license': 1,
                'date_captured': '',
                'frame_index': frame_idx
            })
            
            # Add annotations for this frame
            for ann in annotations:
                # Handle mask-based annotations
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
                    # Get category_id from annotation, or convert category name to ID
                    if 'category_id' in ann:
                        category_id = ann['category_id']
                    elif 'category' in ann:
                        category_id = category_name_to_id.get(ann['category'], 1)
                    else:
                        category_id = 1  # Default to bee
                    coco_ann = {
                        'id': annotation_id,
                        'image_id': image_id,
                        'category_id': category_id,
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
                
                # Handle bbox-only annotations (no mask/segmentation)
                elif ann.get('bbox_only', False) and 'bbox' in ann:
                    bbox = ann['bbox']
                    
                    # bbox format is [x, y, width, height]
                    if len(bbox) == 4 and bbox != [0, 0, 0, 0]:
                        x, y, w, h = bbox
                        
                        # Create annotation without segmentation (detection only)
                        # Get category_id from annotation, or convert category name to ID
                        if 'category_id' in ann:
                            category_id = ann['category_id']
                        elif 'category' in ann:
                            category_id = category_name_to_id.get(ann['category'], 1)
                        else:
                            category_id = 1  # Default to bee
                        coco_ann = {
                            'id': annotation_id,
                            'image_id': image_id,
                            'category_id': category_id,
                            'bbox': [int(x), int(y), int(w), int(h)],
                            'area': int(w * h),
                            'iscrowd': 0
                            # No 'segmentation' field for bbox-only annotations
                        }
                        
                        # Add tracking ID
                        if 'mask_id' in ann or 'instance_id' in ann:
                            track_id = ann.get('mask_id', ann.get('instance_id'))
                            coco_ann['track_id'] = track_id
                            coco_ann['instance_id'] = track_id
                        
                        # Add confidence if available
                        if 'confidence' in ann:
                            coco_ann['confidence'] = float(ann['confidence'])
                        
                        # Add source information
                        if 'source' in ann:
                            coco_ann['source'] = ann['source']
                        
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
    
    # Final progress update
    if progress_callback:
        progress_callback(len(video_ids), len(video_ids), "Complete")
    
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
    
    # Create category name to ID mapping
    category_name_to_id = {name: i + 1 for i, name in enumerate(class_names)}
    
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
        video_png_dir = project_path / 'annotations/png' / video_name
        video_json_dir = project_path / 'annotations/json' / video_name
        video_bbox_dir = project_path / 'annotations/bbox' / video_name
        
        if not video_png_dir.exists() or not video_json_dir.exists():
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
        frame_files = sorted(video_json_dir.glob('frame_*.json'))
        
        # Get actual video frame dimensions from the first saved frame
        video_width = image_width  # Default fallback
        video_height = image_height  # Default fallback
        for frame_idx in sorted(selected_frame_indices):
            frame_path = video_frames_dir / f'frame_{frame_idx:06d}.jpg'
            if frame_path.exists():
                try:
                    frame_img = cv2.imread(str(frame_path))
                    if frame_img is not None:
                        video_height, video_width = frame_img.shape[:2]
                        break  # Got dimensions, stop checking
                except Exception as e:
                    print(f"Warning: Could not read frame {frame_path}: {e}")
                    continue
        
        coco_data['videos'].append({
            'id': video_id_int,
            'name': video_name,
            'num_frames': len(frame_files),
            'width': video_width,
            'height': video_height
        })
        
        # Load video-level annotations (chamber and hive) that apply to all frames
        video_level_annotations = []
        video_annotations_json = video_json_dir / 'video_annotations.json'
        if video_annotations_json.exists():
            try:
                with open(video_annotations_json, 'r') as f:
                    video_json_data = json.load(f)
                
                # Handle both old (list) and new (dict) formats
                if isinstance(video_json_data, dict):
                    # New format: {'annotations': [...], 'aruco_tracking': {...}, 'metadata': {...}}
                    video_annotations_list = video_json_data.get('annotations', [])
                else:
                    # Old format: direct list of annotations
                    video_annotations_list = video_json_data
                
                # Load per-category mask PNGs for video-level annotations
                category_masks = {}
                for category in ('chamber', 'hive'):
                    cat_png = video_png_dir / f'video_annotations_{category}.png'
                    if cat_png.exists():
                        category_masks[category] = cv2.imread(str(cat_png), cv2.IMREAD_UNCHANGED)
                
                # Reconstruct video-level annotations with masks
                for ann_meta in video_annotations_list:
                    mask_id = ann_meta.get('mask_id', 0)
                    category = ann_meta.get('category', 'chamber')
                    
                    # Ensure category_id is correctly set based on category name
                    # This overrides any incorrect category_id that might be in the metadata
                    correct_category_id = category_name_to_id.get(category, 1)
                    
                    if not ann_meta.get('bbox_only', False) and mask_id > 0:
                        cat_mask = category_masks.get(category)
                        if cat_mask is not None:
                            # Extract this instance's mask
                            instance_mask = (cat_mask == mask_id).astype(np.uint8)
                            if np.any(instance_mask > 0):
                                ann = ann_meta.copy()
                                ann['mask'] = instance_mask
                                ann['category_id'] = correct_category_id  # Force correct category_id
                                video_level_annotations.append(ann)
                    elif ann_meta.get('bbox_only', False):
                        # Include bbox-only video-level annotations
                        ann = ann_meta.copy()
                        ann['category_id'] = correct_category_id  # Force correct category_id
                        video_level_annotations.append(ann)
            except Exception as e:
                print(f"Warning: Failed to load video-level annotations for {video_name}: {e}")
        
        # Process each frame in the video
        for json_file in frame_files:
            # Extract frame index from filename
            frame_idx = int(json_file.stem.split('_')[1])
            
            # Skip frames that are not selected for this split
            if frame_idx not in selected_frame_indices:
                continue
            
            # Load annotations from PNG+JSON and bbox folder
            png_file = video_png_dir / f'frame_{frame_idx:06d}.png'
            bbox_file = video_bbox_dir / f'frame_{frame_idx:06d}.json' if video_bbox_dir.exists() else None
            
            # Load mask-based annotations
            mask_annotations = []
            if png_file.exists() and json_file.exists():
                try:
                    # Load combined mask
                    combined_mask = cv2.imread(str(png_file), cv2.IMREAD_UNCHANGED)
                    if combined_mask is not None:
                        # Load metadata
                        with open(json_file, 'r') as f:
                            json_data = json.load(f)
                        
                        # Reconstruct annotations with individual masks
                        for ann_meta in json_data:
                            mask_id = ann_meta.get('mask_id', ann_meta.get('instance_id', 0))
                            if mask_id > 0:
                                # Extract this instance's mask
                                instance_mask = (combined_mask == mask_id).astype(np.uint8)
                                
                                # Create full annotation
                                ann = ann_meta.copy()
                                ann['mask'] = instance_mask
                                mask_annotations.append(ann)
                except Exception as e:
                    print(f"Warning: Failed to load mask annotations from {json_file}: {e}")
            
            # Load bbox-only annotations
            bbox_only_annotations = []
            if bbox_file and bbox_file.exists():
                try:
                    with open(bbox_file, 'r') as f:
                        bbox_data = json.load(f)
                    
                    # Filter to only bbox-only annotations (not derived from masks)
                    for ann in bbox_data:
                        if ann.get('bbox_only', False) and not ann.get('from_mask', False):
                            bbox_only_annotations.append(ann)
                except Exception as e:
                    print(f"Warning: Failed to load bbox annotations from {bbox_file}: {e}")
            
            # Combine per-frame annotations (mask-based and bbox-only)
            per_frame_annotations = mask_annotations + bbox_only_annotations
            
            # Use the actual video dimensions we determined earlier
            frame_width = video_width
            frame_height = video_height
            
            if not per_frame_annotations:
                # Skip frames with no per-frame (bee) annotations
                # Video-level annotations (chamber/hive) are only added to frames that have bee annotations
                continue
            
            # Add video-level annotations (chamber/hive) to frames that have bee annotations
            annotations = per_frame_annotations + video_level_annotations
            
            # Add image entry
            image_file = f"frames/{video_name}/frame_{frame_idx:06d}.jpg"
            coco_data['images'].append({
                'id': image_id,
                'file_name': image_file,
                'width': frame_width,
                'height': frame_height,
                'license': 1,
                'date_captured': '',
                'video_id': video_id_int,  # Link to video
                'frame_index': frame_idx   # Temporal ordering
            })
            
            # Add annotations for this frame
            for ann in annotations:
                # Handle mask-based annotations
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
                    # Get category_id from annotation, or convert category name to ID
                    if 'category_id' in ann:
                        category_id = ann['category_id']
                    elif 'category' in ann:
                        category_id = category_name_to_id.get(ann['category'], 1)
                    else:
                        category_id = 1  # Default to bee
                    coco_ann = {
                        'id': annotation_id,
                        'image_id': image_id,
                        'category_id': category_id,
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
                
                # Handle bbox-only annotations (no mask/segmentation)
                elif ann.get('bbox_only', False) and 'bbox' in ann:
                    bbox = ann['bbox']
                    
                    # bbox format is [x, y, width, height]
                    if len(bbox) == 4 and bbox != [0, 0, 0, 0]:
                        x, y, w, h = bbox
                        
                        # Create annotation without segmentation (detection only)
                        # Get category_id from annotation, or convert category name to ID
                        if 'category_id' in ann:
                            category_id = ann['category_id']
                        elif 'category' in ann:
                            category_id = category_name_to_id.get(ann['category'], 1)
                        else:
                            category_id = 1  # Default to bee
                        coco_ann = {
                            'id': annotation_id,
                            'image_id': image_id,
                            'category_id': category_id,
                            'bbox': [int(x), int(y), int(w), int(h)],
                            'area': int(w * h),
                            'iscrowd': 0
                            # No 'segmentation' field for bbox-only annotations
                        }
                        
                        # Add tracking ID
                        if 'mask_id' in ann or 'instance_id' in ann:
                            instance_id = ann.get('mask_id', ann.get('instance_id'))
                            track_id = video_id_int * 10000 + instance_id
                            coco_ann['track_id'] = track_id
                            coco_ann['instance_id'] = instance_id
                        
                        # Add confidence if available
                        if 'confidence' in ann:
                            coco_ann['confidence'] = float(ann['confidence'])
                        
                        # Add source information
                        if 'source' in ann:
                            coco_ann['source'] = ann['source']
                        
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
