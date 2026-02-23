"""
Annotation data structures and management
"""

import json
import numpy as np
from pathlib import Path
from datetime import datetime
import pickle
from collections import OrderedDict


def mask_to_rle(mask):
    """Convert binary mask to run-length encoding (RLE)
    
    Args:
        mask: 2D numpy array (H, W) with values 0 or 255
        
    Returns:
        dict with 'size', 'counts' (list of run lengths alternating 0s and 1s)
    """
    # Flatten mask and binarize
    pixels = (mask.flatten() > 0).astype(np.uint8)
    
    if len(pixels) == 0:
        return {'size': list(mask.shape), 'counts': []}
    
    # Count consecutive runs
    counts = []
    current_value = 0  # Always start counting 0s
    current_count = 0
    
    for pixel in pixels:
        if pixel == current_value:
            current_count += 1
        else:
            counts.append(current_count)
            current_count = 1
            current_value = 1 - current_value
    
    # Add the last run
    counts.append(current_count)
    
    # If we ended on 0s, add a final 0 for 1s
    # If we ended on 1s, we're good
    if current_value == 0:
        counts.append(0)
    
    return {
        'size': list(mask.shape),
        'counts': counts
    }


def rle_to_mask(rle):
    """Convert run-length encoding to binary mask
    
    Args:
        rle: dict with 'size' and 'counts'
        
    Returns:
        2D numpy array (H, W) with values 0 or 255
    """
    h, w = rle['size']
    counts = rle['counts']
    
    if len(counts) == 0:
        return np.zeros((h, w), dtype=np.uint8)
    
    # Create flat mask
    total_pixels = h * w
    mask = np.zeros(total_pixels, dtype=np.uint8)
    
    # Decode RLE - counts alternate between 0s and 1s
    position = 0
    for i, count in enumerate(counts):
        if i % 2 == 1:  # Odd indices are 1s (foreground)
            end_pos = min(position + count, total_pixels)
            mask[position:end_pos] = 255
        position += count
        if position >= total_pixels:
            break
    
    return mask.reshape((h, w))


def mask_to_polygon(mask):
    """Convert binary mask to polygon coordinates (COCO format)
    
    Args:
        mask: 2D numpy array (H, W) with values 0 or 255
        
    Returns:
        list of polygon coordinates [[x1, y1, x2, y2, ...], ...]
        Each polygon is a list of x, y coordinates in order
    """
    import cv2
    
    # Ensure mask is binary
    binary_mask = (mask > 0).astype(np.uint8)
    
    # Find contours
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    polygons = []
    for contour in contours:
        # Flatten contour and convert to list
        # Contour shape is (n_points, 1, 2) -> flatten to [x1, y1, x2, y2, ...]
        contour = contour.flatten().tolist()
        
        # Only keep polygons with at least 6 coordinates (3 points)
        if len(contour) >= 6:
            polygons.append(contour)
    
    return polygons

class AnnotationManager:
    """Manage annotations for a project with LRU cache"""
    
    def __init__(self, max_cache_size=20):
        self.project_info = {}
        self.frame_annotations = OrderedDict()  # LRU cache: frame_idx -> list of annotations
        self.max_cache_size = max_cache_size  # Maximum frames to keep in memory
        self.class_names = ['bee']  # Default class
        self.unsaved_changes = False
        self.image_width = 0
        self.image_height = 0
        
    def new_project(self, project_info):
        """Create a new project"""
        self.project_info = {
            'name': project_info.get('name', 'Untitled'),
            'created': datetime.now().isoformat(),
            'modified': datetime.now().isoformat(),
            'classes': self.class_names,
            'version': '1.0'
        }
        self.frame_annotations = OrderedDict()
        self.unsaved_changes = True
        
    def load_project(self, project_path):
        """Load project from directory"""
        project_path = Path(project_path)
        annotations_file = project_path / 'annotations' / 'project.json'
        
        if not annotations_file.exists():
            raise FileNotFoundError(f"Project file not found: {annotations_file}")
            
        with open(annotations_file, 'r') as f:
            data = json.load(f)
            
        self.project_info = data.get('project_info', {})
        self.class_names = data.get('classes', ['bee'])
        
        # Load frame annotations into LRU cache (only recent frames)
        self.frame_annotations = OrderedDict()
        annotations_dir = project_path / 'annotations' / 'frames'
        
        if annotations_dir.exists():
            for ann_file in annotations_dir.glob('frame_*.pkl'):
                frame_idx = int(ann_file.stem.split('_')[1])
                with open(ann_file, 'rb') as f:
                    compressed_annotations = pickle.load(f)
                
                # Decompress RLE masks
                annotations = []
                for ann in compressed_annotations:
                    decompressed_ann = ann.copy()
                    if 'mask_rle' in ann:
                        # Convert RLE back to mask
                        decompressed_ann['mask'] = rle_to_mask(ann['mask_rle'])
                        del decompressed_ann['mask_rle']
                    elif 'mask' in ann:
                        # Old format - keep as is
                        decompressed_ann['mask'] = ann['mask']
                    annotations.append(decompressed_ann)
                
                self.frame_annotations[frame_idx] = annotations
                    
        self.unsaved_changes = False
        
    def save_project(self, project_path):
        """Save project to directory"""
        project_path = Path(project_path)
        annotations_dir = project_path / 'annotations'
        annotations_dir.mkdir(parents=True, exist_ok=True)
        
        # Update modification time
        self.project_info['modified'] = datetime.now().isoformat()
        
        # Save project info
        project_file = annotations_dir / 'project.json'
        with open(project_file, 'w') as f:
            json.dump({
                'project_info': self.project_info,
                'classes': self.class_names,
                'num_frames': len(self.frame_annotations),
                'image_width': self.image_width,
                'image_height': self.image_height
            }, f, indent=2)
            
        # Save frame annotations
        frames_dir = annotations_dir / 'frames'
        frames_dir.mkdir(exist_ok=True)
        
        for frame_idx, annotations in self.frame_annotations.items():
            # Convert masks to RLE for efficient storage
            compressed_annotations = []
            for ann in annotations:
                compressed_ann = ann.copy()
                if 'mask' in ann:
                    # Convert mask to RLE
                    compressed_ann['mask_rle'] = mask_to_rle(ann['mask'])
                    del compressed_ann['mask']  # Remove full mask
                compressed_annotations.append(compressed_ann)
            
            frame_file = frames_dir / f'frame_{frame_idx:06d}.pkl'
            with open(frame_file, 'wb') as f:
                pickle.dump(compressed_annotations, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        # Export COCO format
        coco_file = annotations_dir / 'annotations_coco.json'
        try:
            self.export_coco(coco_file)
        except Exception as e:
            print(f"Warning: Could not export COCO format: {e}")
                
        self.unsaved_changes = False
    
    def save_frame_annotations(self, project_path, video_id, frame_idx, annotations):
        """Save annotations for a single frame (video-based structure)
        
        Args:
            project_path: Path to project directory
            video_id: Video identifier
            frame_idx: Frame index within the video
            annotations: List of annotation dicts with masks
        """
        from pathlib import Path
        project_path = Path(project_path)
        
        # Create video-specific annotation directory
        video_ann_dir = project_path / 'annotations/pkl' / video_id
        video_ann_dir.mkdir(parents=True, exist_ok=True)
        
        # Convert masks to RLE for efficient storage
        compressed_annotations = []
        for ann in annotations:
            compressed_ann = ann.copy()
            if 'mask' in ann:
                # Convert mask to RLE
                compressed_ann['mask_rle'] = mask_to_rle(ann['mask'])
                del compressed_ann['mask']  # Remove full mask
            compressed_annotations.append(compressed_ann)
        
        # Save this frame only
        frame_file = video_ann_dir / f'frame_{frame_idx:06d}.pkl'
        with open(frame_file, 'wb') as f:
            pickle.dump(compressed_annotations, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    def load_frame_annotations(self, project_path, video_id, frame_idx):
        """Load annotations for a single frame (video-based structure)
        
        Args:
            project_path: Path to project directory
            video_id: Video identifier
            frame_idx: Frame index within the video
            
        Returns:
            List of annotations with masks
        """
        from pathlib import Path
        project_path = Path(project_path)
        
        frame_file = project_path / 'annotations/pkl' / video_id / f'frame_{frame_idx:06d}.pkl'
        
        if not frame_file.exists():
            return []
        
        with open(frame_file, 'rb') as f:
            compressed_annotations = pickle.load(f)
        
        # Decompress masks
        annotations = []
        for ann in compressed_annotations:
            decompressed_ann = ann.copy()
            if 'mask_rle' in ann:
                # Convert RLE back to mask
                decompressed_ann['mask'] = rle_to_mask(ann['mask_rle'])
                del decompressed_ann['mask_rle']
            annotations.append(decompressed_ann)
        
        return annotations
        
    def get_frame_annotations(self, frame_idx):
        """Get annotations for a frame (LRU access)"""
        if frame_idx in self.frame_annotations:
            # Move to end (most recently used)
            self.frame_annotations.move_to_end(frame_idx)
            return self.frame_annotations[frame_idx]
        return []
        
    def set_frame_annotations(self, frame_idx, annotations):
        """Set annotations for a frame with LRU cache management"""
        import copy
        import numpy as np
        import gc
        
        prev = self.frame_annotations.get(frame_idx, None)
        
        def deep_equal(a, b):
            if type(a) != type(b):
                return False
            if isinstance(a, dict):
                if set(a.keys()) != set(b.keys()):
                    return False
                for k in a:
                    if not deep_equal(a[k], b[k]):
                        return False
                return True
            elif isinstance(a, (list, tuple)):
                if len(a) != len(b):
                    return False
                return all(deep_equal(x, y) for x, y in zip(a, b))
            elif isinstance(a, np.ndarray):
                return np.array_equal(a, b)
            else:
                return a == b

        if prev is None or not deep_equal(prev, annotations):
            self.unsaved_changes = True
        
        # Update existing entry
        if frame_idx in self.frame_annotations:
            self.frame_annotations.move_to_end(frame_idx)
            # Delete old annotations before replacing
            old_anns = self.frame_annotations[frame_idx]
            del old_anns
            self.frame_annotations[frame_idx] = copy.deepcopy(annotations)
            # print(f"Annotation cache: updated frame {frame_idx}")
        else:
            # Evict oldest if over capacity BEFORE adding new one
            if len(self.frame_annotations) >= self.max_cache_size:
                oldest_key = next(iter(self.frame_annotations))
                old_anns = self.frame_annotations[oldest_key]
                del self.frame_annotations[oldest_key]
                # Explicitly delete annotation data (includes large mask arrays)
                del old_anns
                gc.collect()
                print(f"Annotation cache: evicted frame {oldest_key} to make room for frame {frame_idx}")
            
            # Store annotations (deep copy to avoid external modifications)
            self.frame_annotations[frame_idx] = copy.deepcopy(annotations)
            print(f"Annotation cache: cached frame {frame_idx}, cache size={len(self.frame_annotations)}/{self.max_cache_size}")
    
    def clear_annotation_cache(self, keep_recent=10):
        """
        Clear annotation cache to free memory, keeping only recent frames.
        
        Args:
            keep_recent: Number of recent frame annotations to keep in memory
        """
        if len(self.frame_annotations) <= keep_recent:
            return
        
        # Sort by frame index and keep only the most recent ones
        sorted_indices = sorted(self.frame_annotations.keys())
        frames_to_remove = sorted_indices[:-keep_recent]
        
        for frame_idx in frames_to_remove:
            del self.frame_annotations[frame_idx]
        
        # Force garbage collection
        import gc
        gc.collect()
        
        print(f"Cleared annotation cache: removed {len(frames_to_remove)} frames, kept {len(self.frame_annotations)}")
    
    def clear_cache(self):
        """Clear the in-memory annotation cache"""
        self.frame_annotations = OrderedDict()
        self.unsaved_changes = False
        
    def add_annotation(self, frame_idx, annotation):
        """Add an annotation to a frame"""
        if frame_idx not in self.frame_annotations:
            self.frame_annotations[frame_idx] = []
            
        self.frame_annotations[frame_idx].append(annotation)
        self.unsaved_changes = True
        
    def remove_annotation(self, frame_idx, ann_idx):
        """Remove an annotation from a frame"""
        if frame_idx in self.frame_annotations:
            if 0 <= ann_idx < len(self.frame_annotations[frame_idx]):
                self.frame_annotations[frame_idx].pop(ann_idx)
                self.unsaved_changes = True
                
    def update_annotation(self, frame_idx, ann_idx, annotation):
        """Update an annotation"""
        if frame_idx in self.frame_annotations:
            if 0 <= ann_idx < len(self.frame_annotations[frame_idx]):
                self.frame_annotations[frame_idx][ann_idx] = annotation
                self.unsaved_changes = True
                
    def has_unsaved_changes(self):
        """Check if there are unsaved changes"""
        return self.unsaved_changes
        
    def export_coco(self, output_path):
        """Export annotations in COCO format with instance segmentation
        
        Args:
            output_path: Path to output JSON file
        """
        output_path = Path(output_path)
        
        coco_data = {
            'info': {
                'description': self.project_info.get('name', 'Untitled'),
                'version': '1.0',
                'year': datetime.now().year,
                'contributor': 'BeeWhere Annotator',
                'date_created': datetime.now().isoformat()
            },
            'licenses': [{
                'id': 1,
                'name': 'Unknown',
                'url': ''
            }],
            'categories': [],
            'images': [],
            'annotations': []
        }
        
        # Add categories
        for i, class_name in enumerate(self.class_names):
            coco_data['categories'].append({
                'id': i + 1,
                'name': class_name,
                'supercategory': 'insect'
            })
            
        # Add images and annotations
        annotation_id = 1
        
        for frame_idx, annotations in sorted(self.frame_annotations.items()):
            # Add image entry
            image_id = frame_idx + 1
            image_file = f"frame_{frame_idx:06d}.png"
            
            coco_data['images'].append({
                'id': image_id,
                'file_name': image_file,
                'width': self.image_width if self.image_width > 0 else 1920,
                'height': self.image_height if self.image_height > 0 else 1080,
                'license': 1,
                'date_captured': ''
            })
            
            # Add annotations for this frame
            for ann in annotations:
                if 'mask' in ann:
                    mask = ann['mask']
                    
                    # Get bounding box from mask
                    coords = np.where(mask > 0)
                    if len(coords[0]) > 0:
                        y_min, y_max = coords[0].min(), coords[0].max()
                        x_min, x_max = coords[1].min(), coords[1].max()
                        bbox_width = int(x_max - x_min + 1)
                        bbox_height = int(y_max - y_min + 1)
                        
                        # Create annotation with polygon segmentation
                        coco_ann = {
                            'id': annotation_id,
                            'image_id': image_id,
                            'category_id': 1,  # Default to 'bee'
                            'bbox': [int(x_min), int(y_min), bbox_width, bbox_height],
                            'area': int(np.sum(mask > 0)),
                            'iscrowd': 0
                        }
                        
                        # Add segmentation in polygon format (COCO standard)
                        polygons = mask_to_polygon(mask)
                        if polygons:
                            coco_ann['segmentation'] = polygons
                        else:
                            # If no valid polygons, skip this annotation
                            continue
                        
                        # Add mask_id if available
                        if 'mask_id' in ann:
                            coco_ann['mask_id'] = ann['mask_id']
                        
                        coco_data['annotations'].append(coco_ann)
                        annotation_id += 1
                        
        # Save to file
        with open(output_path, 'w') as f:
            json.dump(coco_data, f, indent=2)
            
    def get_statistics(self):
        """Get annotation statistics"""
        total_frames = len(self.frame_annotations)
        total_instances = sum(len(anns) for anns in self.frame_annotations.values())
        
        return {
            'total_frames': total_frames,
            'total_instances': total_instances,
            'avg_instances_per_frame': total_instances / max(total_frames, 1)
        }
    
    def generate_coco_for_split(self, project_path, video_ids: list, 
                                split_name: str = 'train'):
        """
        Generate COCO JSON for a specific split (train or val)
        
        Args:
            project_path: Path to project directory
            video_ids: List of video IDs to include
            split_name: Name of split ('train' or 'val')
            
        Returns:
            Path to generated COCO file
        """
        from pathlib import Path
        project_path = Path(project_path)
        
        coco_data = {
            'info': {
                'description': f'{self.project_info.get("name", "Untitled")} - {split_name} split',
                'version': '1.0',
                'year': datetime.now().year,
                'contributor': 'BeeWhere Annotator',
                'date_created': datetime.now().isoformat()
            },
            'licenses': [{
                'id': 1,
                'name': 'Unknown',
                'url': ''
            }],
            'categories': [],
            'images': [],
            'annotations': []
        }
        
        # Add categories
        for i, class_name in enumerate(self.class_names):
            coco_data['categories'].append({
                'id': i + 1,
                'name': class_name,
                'supercategory': 'insect'
            })
        
        image_id = 1
        annotation_id = 1
        
        # Process each video
        for video_id in video_ids:
            video_ann_dir = project_path / 'annotations/pkl' / video_id
            if not video_ann_dir.exists():
                continue
            
            # Get all annotation files for this video
            for ann_file in sorted(video_ann_dir.glob('frame_*.pkl')):
                # Extract frame index from filename
                frame_idx = int(ann_file.stem.split('_')[1])
                
                # Load annotations
                with open(ann_file, 'rb') as f:
                    compressed_annotations = pickle.load(f)
                
                # Decompress masks
                annotations = []
                for ann in compressed_annotations:
                    decompressed_ann = ann.copy()
                    if 'mask_rle' in ann:
                        decompressed_ann['mask'] = rle_to_mask(ann['mask_rle'])
                        del decompressed_ann['mask_rle']
                    annotations.append(decompressed_ann)
                
                if not annotations:
                    continue  # Skip frames with no annotations
                
                # Add image entry
                image_file = f"frames/{video_id}/frame_{frame_idx:06d}.jpg"
                coco_data['images'].append({
                    'id': image_id,
                    'file_name': image_file,
                    'width': self.image_width if self.image_width > 0 else 1920,
                    'height': self.image_height if self.image_height > 0 else 1080,
                    'license': 1,
                    'date_captured': ''
                })
                
                # Add annotations for this frame
                for ann in annotations:
                    if 'mask' in ann:
                        mask = ann['mask']
                        
                        # Get bounding box from mask
                        coords = np.where(mask > 0)
                        if len(coords[0]) > 0:
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
                            
                            # Add IDs
                            if 'mask_id' in ann:
                                coco_ann['mask_id'] = ann['mask_id']
                                coco_ann['bee_id'] = ann['mask_id']
                            
                            coco_data['annotations'].append(coco_ann)
                            annotation_id += 1
                
                image_id += 1
        
        # Save to file
        output_path = project_path / f'annotations/coco/instances_{split_name}.json'
        with open(output_path, 'w') as f:
            json.dump(coco_data, f, indent=2)
        
        return output_path
