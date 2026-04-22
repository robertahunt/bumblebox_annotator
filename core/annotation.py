"""
Annotation data structures and management
"""

import json
import numpy as np
from pathlib import Path
from datetime import datetime
import pickle
from collections import OrderedDict
import math


def sanitize_for_json(obj):
    """Recursively sanitize data for JSON serialization
    
    Converts numpy types to Python types, handles NaN/Inf, etc.
    
    Args:
        obj: Any Python object
        
    Returns:
        JSON-serializable version of obj
    """
    # Handle None
    if obj is None:
        return None
    
    # Handle numpy types
    if isinstance(obj, (np.integer, np.int64, np.int32, np.int16, np.int8)):
        return int(obj)
    if isinstance(obj, (np.floating, np.float64, np.float32, np.float16)):
        val = float(obj)
        # Handle NaN and Inf
        if math.isnan(val) or math.isinf(val):
            return None
        return val
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        # Don't allow arrays in JSON (should have been removed already)
        raise ValueError(f"Cannot serialize numpy array to JSON: shape={obj.shape}")
    
    # Handle Python floats/ints that might be NaN/Inf
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    
    # Handle sequences
    if isinstance(obj, (list, tuple)):
        return [sanitize_for_json(item) for item in obj]
    
    # Handle dicts
    if isinstance(obj, dict):
        return {str(key): sanitize_for_json(value) for key, value in obj.items()}
    
    # Handle primitives
    if isinstance(obj, (str, int, bool)):
        return obj
    
    # Unknown type - try to convert to string
    try:
        return str(obj)
    except:
        return None


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
        self.class_names = ['bee', 'hive', 'chamber', 'pollen']  # Multi-category support
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
        self.class_names = data.get('classes', ['bee', 'hive', 'chamber', 'pollen'])
        
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
    
    def save_frame_annotations_pickle(self, project_path, video_id, frame_idx, annotations):
        """Save annotations for a single frame using pickle format (legacy)
        
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
    
    def save_frame_annotations_png(self, project_path, video_id, frame_idx, annotations):
        """Save annotations for a single frame using PNG format (faster)
        
        Args:
            project_path: Path to project directory
            video_id: Video identifier
            frame_idx: Frame index within the video
            annotations: List of annotation dicts with masks
        """
        import cv2
        import tempfile
        import os
        from pathlib import Path
        project_path = Path(project_path)
        
        # Create video-specific annotation directories
        png_dir = project_path / 'annotations/png' / video_id
        json_dir = project_path / 'annotations/json' / video_id
        png_dir.mkdir(parents=True, exist_ok=True)
        json_dir.mkdir(parents=True, exist_ok=True)
        
        # Prepare data for JSON (metadata without masks)
        json_data = []
        
        if annotations:
            # Get image shape from first annotation mask
            h, w = annotations[0]['mask'].shape[:2]
            
            # Create combined instance mask (each instance has unique ID)
            combined_mask = np.zeros((h, w), dtype=np.uint16)  # uint16 supports up to 65535 instances
            
            for ann in annotations:
                mask_id = ann.get('mask_id', ann.get('instance_id', 0))
                if mask_id > 0 and 'mask' in ann:
                    if ann['mask'].shape[:2] != (h, w):
                        print(f"Warning: save_frame_annotations_png skipping annotation "
                              f"{mask_id} – shape {ann['mask'].shape[:2]} != {(h, w)}")
                    else:
                        # Add this instance to combined mask
                        mask_binary = (ann['mask'] > 0).astype(bool)
                        combined_mask[mask_binary] = mask_id
                
                # Save metadata (without mask) and sanitize for JSON
                ann_meta = {k: v for k, v in ann.items() if k not in ['mask', 'mask_rle']}
                json_data.append(ann_meta)
            
            # Save combined mask as PNG (atomic write)
            png_file = png_dir / f'frame_{frame_idx:06d}.png'
            
            # Write to temp file first, then rename (atomic on POSIX)
            temp_png = tempfile.NamedTemporaryFile(
                mode='wb', 
                delete=False, 
                dir=png_dir, 
                prefix=f'.tmp_frame_{frame_idx:06d}_',
                suffix='.png'
            )
            temp_png_path = temp_png.name
            temp_png.close()
            
            try:
                cv2.imwrite(temp_png_path, combined_mask)
                os.replace(temp_png_path, png_file)  # Atomic on POSIX
            except Exception as e:
                # Clean up temp file on error
                if os.path.exists(temp_png_path):
                    os.unlink(temp_png_path)
                raise e
        
        # Sanitize JSON data (convert numpy types, handle NaN/Inf)
        try:
            json_data_sanitized = sanitize_for_json(json_data)
        except Exception as e:
            print(f"Error sanitizing JSON data for frame {frame_idx}: {e}")
            # Fallback to basic serialization
            json_data_sanitized = []
            for ann in json_data:
                sanitized_ann = {}
                for k, v in ann.items():
                    try:
                        sanitized_ann[k] = sanitize_for_json(v)
                    except:
                        print(f"Warning: Could not serialize annotation field '{k}': {type(v)}")
                        sanitized_ann[k] = None
                json_data_sanitized.append(sanitized_ann)
        
        # Save metadata as JSON (atomic write with explicit flush)
        json_file = json_dir / f'frame_{frame_idx:06d}.json'
        
        # Write to temp file first, then rename (atomic on POSIX)
        temp_json = tempfile.NamedTemporaryFile(
            mode='w', 
            delete=False, 
            dir=json_dir, 
            prefix=f'.tmp_frame_{frame_idx:06d}_',
            suffix='.json'
        )
        temp_json_path = temp_json.name
        
        try:
            json.dump(json_data_sanitized, temp_json, indent=2)
            temp_json.flush()  # Ensure data is written
            os.fsync(temp_json.fileno())  # Force write to disk
            temp_json.close()
            os.replace(temp_json_path, json_file)  # Atomic on POSIX
        except Exception as e:
            # Clean up temp file on error
            temp_json.close()
            if os.path.exists(temp_json_path):
                os.unlink(temp_json_path)
            raise e
    
    def save_frame_annotations(self, project_path, video_id, frame_idx, annotations):
        """Save annotations for a single frame (uses PNG+JSON format for masks, separate bbox folder for bbox-only)
        
        Args:
            project_path: Path to project directory
            video_id: Video identifier
            frame_idx: Frame index within the video
            annotations: List of annotation dicts (with or without masks)
        """
        # Separate bbox-only annotations from mask-based annotations
        mask_annotations = []
        bbox_only_annotations = []
        
        for ann in annotations:
            if ann.get('bbox_only', False) and 'mask' not in ann:
                # This is a bbox-only annotation
                bbox_only_annotations.append(ann)
            elif 'mask' in ann:
                # This is a mask-based annotation
                mask_annotations.append(ann)
                
                # Also compute and save bbox if not present
                if 'bbox' not in ann or ann['bbox'] == [0, 0, 0, 0]:
                    # Compute bbox from mask
                    mask = ann['mask']
                    coords = np.argwhere(mask > 0)
                    if len(coords) > 0:
                        y_min, x_min = coords.min(axis=0)
                        y_max, x_max = coords.max(axis=0)
                        width = int(x_max - x_min + 1)
                        height = int(y_max - y_min + 1)
                        ann['bbox'] = [int(x_min), int(y_min), width, height]
        
        # Save mask-based annotations to PNG+JSON format
        if mask_annotations:
            self.save_frame_annotations_png(project_path, video_id, frame_idx, mask_annotations)
        else:
            # No masks - create empty PNG and JSON
            self.save_frame_annotations_png(project_path, video_id, frame_idx, [])
        
        # Handle bbox-only annotations
        if bbox_only_annotations or mask_annotations:
            # Save bbox-only annotations separately (or merged with mask bboxes)
            if bbox_only_annotations:
                # Save bbox-only annotations
                bboxes_to_save = bbox_only_annotations
            else:
                bboxes_to_save = []
            
            # Also save bboxes computed from masks to bbox folder for convenience
            # (This allows bbox-only workflow to access all bboxes in one place)
            if mask_annotations:
                mask_bboxes = []
                for ann in mask_annotations:
                    bbox_ann = {k: v for k, v in ann.items() if k != 'mask'}
                    bbox_ann['source'] = bbox_ann.get('source', 'mask')  # Mark as derived from mask
                    bbox_ann['from_mask'] = True  # Flag to indicate this was derived from a mask
                    mask_bboxes.append(bbox_ann)
                
                # Merge with bbox-only annotations
                bboxes_to_save = bbox_only_annotations + mask_bboxes
            
            self.save_bbox_annotations(project_path, video_id, frame_idx, bboxes_to_save)
        else:
            # No annotations at all - delete bbox file if it exists
            from pathlib import Path
            bbox_file = Path(project_path) / 'annotations' / 'bbox' / video_id / f'frame_{frame_idx:06d}.json'
            if bbox_file.exists():
                bbox_file.unlink()
                print(f"Deleted bbox file (no annotations): {bbox_file}")
    
    def load_frame_annotations_pickle(self, project_path, video_id, frame_idx):
        """Load annotations for a single frame from pickle format (legacy)
        
        Args:
            project_path: Path to project directory
            video_id: Video identifier
            frame_idx: Frame index within the video
            
        Returns:
            List of annotations with masks, or None if file doesn't exist
        """
        from pathlib import Path
        project_path = Path(project_path)
        
        frame_file = project_path / 'annotations/pkl' / video_id / f'frame_{frame_idx:06d}.pkl'
        
        if not frame_file.exists():
            return None
        
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
    
    def load_frame_annotations_png(self, project_path, video_id, frame_idx):
        """Load annotations for a single frame from PNG format (faster)
        
        Args:
            project_path: Path to project directory
            video_id: Video identifier
            frame_idx: Frame index within the video
            
        Returns:
            List of annotations with masks, or None if files don't exist
        """
        import cv2
        import os
        from pathlib import Path
        project_path = Path(project_path)
        
        png_file = project_path / 'annotations/png' / video_id / f'frame_{frame_idx:06d}.png'
        json_file = project_path / 'annotations/json' / video_id / f'frame_{frame_idx:06d}.json'
        
        if not png_file.exists() or not json_file.exists():
            return None
        
        # Load combined mask
        combined_mask = cv2.imread(str(png_file), cv2.IMREAD_UNCHANGED)
        if combined_mask is None:
            return None
        
        # Load metadata with error handling
        try:
            with open(json_file, 'r') as f:
                json_data = json.load(f)
        except json.JSONDecodeError as e:
            # JSON is corrupted - provide detailed error
            file_size = os.path.getsize(json_file)
            print(f"ERROR: Corrupted JSON file: {json_file}")
            print(f"  File size: {file_size} bytes")
            print(f"  JSON error: {e}")
            
            # Try to read the file content for debugging
            try:
                with open(json_file, 'r') as f:
                    content = f.read()
                    print(f"  File content length: {len(content)} chars")
                    if len(content) < 2000:
                        print(f"  Full content: {content}")
                    else:
                        print(f"  First 500 chars: {content[:500]}")
                        print(f"  Last 500 chars: {content[-500:]}")
                        # Show content around error position if available
                        if hasattr(e, 'pos'):
                            start = max(0, e.pos - 100)
                            end = min(len(content), e.pos + 100)
                            print(f"  Around error position {e.pos}: {content[start:end]}")
            except Exception as read_err:
                print(f"  Could not read file for debugging: {read_err}")
            
            raise ValueError(f"Corrupted JSON file for frame {frame_idx}: {e}")
        
        # Reconstruct annotations with individual masks
        annotations = []
        for ann_meta in json_data:
            mask_id = ann_meta.get('mask_id', ann_meta.get('instance_id', 0))
            if mask_id > 0:
                # Extract this instance's mask from combined mask
                instance_mask = (combined_mask == mask_id).astype(np.uint8) * 255
                
                # Combine metadata with mask
                ann = ann_meta.copy()
                ann['mask'] = instance_mask
                annotations.append(ann)
        
        return annotations
    
    def load_frame_annotations(self, project_path, video_id, frame_idx):
        """Load annotations for a single frame (PNG+JSON format for masks, bbox folder for bbox-only)
        
        Args:
            project_path: Path to project directory
            video_id: Video identifier
            frame_idx: Frame index within the video
            
        Returns:
            List of annotations (both mask-based and bbox-only), or empty list if not found
        """
        # Load mask-based annotations from PNG+JSON format (current standard)
        mask_annotations = self.load_frame_annotations_png(project_path, video_id, frame_idx)
        
        if mask_annotations is None:
            mask_annotations = []
        
        # Also load bbox-only annotations from bbox folder
        bbox_annotations = self.load_bbox_annotations(project_path, video_id, frame_idx)
        
        if bbox_annotations:
            # Filter out bboxes that were derived from masks (we already have those from PNG+JSON)
            # Only keep bbox-only annotations
            bbox_only = [ann for ann in bbox_annotations if ann.get('bbox_only', False) and not ann.get('from_mask', False)]
            
            # Combine mask annotations with bbox-only annotations
            return mask_annotations + bbox_only
        
        return mask_annotations
    
    def save_bbox_annotations(self, project_path, video_id, frame_idx, bbox_annotations):
        """Save bounding box annotations separately from segmentations
        
        Args:
            project_path: Path to project directory
            video_id: Video identifier
            frame_idx: Frame index within the video
            bbox_annotations: List of bbox annotation dicts (without masks)
        """
        import tempfile
        import os
        from pathlib import Path
        project_path = Path(project_path)
        
        # Create bbox annotation directory
        bbox_dir = project_path / 'annotations/bbox' / video_id
        bbox_dir.mkdir(parents=True, exist_ok=True)
        
        bbox_file = bbox_dir / f'frame_{frame_idx:06d}.json'
        
        # Sanitize data for JSON
        try:
            bbox_data_sanitized = sanitize_for_json(bbox_annotations)
        except Exception as e:
            print(f"Error sanitizing bbox JSON data for frame {frame_idx}: {e}")
            bbox_data_sanitized = []
            for ann in bbox_annotations:
                sanitized_ann = {}
                for k, v in ann.items():
                    try:
                        sanitized_ann[k] = sanitize_for_json(v)
                    except:
                        print(f"Warning: Could not serialize bbox field '{k}': {type(v)}")
                        sanitized_ann[k] = None
                bbox_data_sanitized.append(sanitized_ann)
        
        # Write to temp file first, then rename (atomic on POSIX)
        temp_bbox = tempfile.NamedTemporaryFile(
            mode='w',
            delete=False,
            dir=bbox_dir,
            prefix=f'.tmp_frame_{frame_idx:06d}_',
            suffix='.json'
        )
        temp_bbox_path = temp_bbox.name
        
        try:
            json.dump(bbox_data_sanitized, temp_bbox, indent=2)
            temp_bbox.flush()
            os.fsync(temp_bbox.fileno())
            temp_bbox.close()
            os.replace(temp_bbox_path, bbox_file)
        except Exception as e:
            temp_bbox.close()
            if os.path.exists(temp_bbox_path):
                os.unlink(temp_bbox_path)
            raise e
    
    def load_bbox_annotations(self, project_path, video_id, frame_idx):
        """Load bounding box annotations for a single frame
        
        Args:
            project_path: Path to project directory
            video_id: Video identifier
            frame_idx: Frame index within the video
            
        Returns:
            List of bbox annotations (without masks), or empty list if not found
        """
        from pathlib import Path
        project_path = Path(project_path)
        
        bbox_file = project_path / 'annotations/bbox' / video_id / f'frame_{frame_idx:06d}.json'
        
        if not bbox_file.exists():
            return []
        
        try:
            with open(bbox_file, 'r') as f:
                bbox_annotations = json.load(f)
            return bbox_annotations
        except json.JSONDecodeError as e:
            print(f"Error loading bbox annotations for frame {frame_idx}: {e}")
            return []
        
    def get_frame_annotations(self, frame_idx, video_id=None):
        """Get annotations for a frame (LRU access)
        
        Args:
            frame_idx: Frame index
            video_id: Optional video identifier for cache key. If None, uses frame_idx only (legacy)
            
        Returns:
            List of annotations, or empty list if not cached
        """
        # Use (video_id, frame_idx) as cache key if video_id provided, otherwise just frame_idx
        cache_key = (video_id, frame_idx) if video_id else frame_idx
        
        if cache_key in self.frame_annotations:
            # Move to end (most recently used)
            self.frame_annotations.move_to_end(cache_key)
            return self.frame_annotations[cache_key]
        return []
        
    def set_frame_annotations(self, frame_idx, annotations, video_id=None):
        """Set annotations for a frame with LRU cache management
        
        Args:
            frame_idx: Frame index
            annotations: List of annotation dicts
            video_id: Optional video identifier for cache key. If None, uses frame_idx only (legacy)
        """
        import copy
        import numpy as np
        import gc
        
        # Use (video_id, frame_idx) as cache key if video_id provided, otherwise just frame_idx
        cache_key = (video_id, frame_idx) if video_id else frame_idx
        
        prev = self.frame_annotations.get(cache_key, None)
        
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
        if cache_key in self.frame_annotations:
            self.frame_annotations.move_to_end(cache_key)
            # Delete old annotations before replacing
            old_anns = self.frame_annotations[cache_key]
            del old_anns
            self.frame_annotations[cache_key] = copy.deepcopy(annotations)
            # print(f"Annotation cache: updated frame {cache_key}")
        else:
            # Evict oldest if over capacity BEFORE adding new one
            if len(self.frame_annotations) >= self.max_cache_size:
                oldest_key = next(iter(self.frame_annotations))
                old_anns = self.frame_annotations[oldest_key]
                del self.frame_annotations[oldest_key]
                # Explicitly delete annotation data (includes large mask arrays)
                del old_anns
                gc.collect()
                print(f"Annotation cache: evicted {oldest_key} to make room for {cache_key}")
            
            # Store annotations (deep copy to avoid external modifications)
            self.frame_annotations[cache_key] = copy.deepcopy(annotations)
            print(f"Annotation cache: cached {cache_key}, cache size={len(self.frame_annotations)}/{self.max_cache_size}")
    
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
            # Get actual image dimensions from the first mask
            img_width = self.image_width if self.image_width > 0 else 1920
            img_height = self.image_height if self.image_height > 0 else 1080
            
            if annotations and 'mask' in annotations[0]:
                # Use actual dimensions from mask
                mask_height, mask_width = annotations[0]['mask'].shape
                img_width = mask_width
                img_height = mask_height
            
            # Add image entry
            image_id = frame_idx + 1
            image_file = f"frame_{frame_idx:06d}.png"
            
            coco_data['images'].append({
                'id': image_id,
                'file_name': image_file,
                'width': img_width,
                'height': img_height,
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
                        # Use category_id from annotation, default to 1 (bee) for backward compatibility
                        category_id = ann.get('category_id', 1)
                        coco_ann = {
                            'id': annotation_id,
                            'image_id': image_id,
                            'category_id': category_id,
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
        supercategories = {'bee': 'insect', 'hive': 'structure', 'chamber': 'structure'}
        for i, class_name in enumerate(self.class_names):
            coco_data['categories'].append({
                'id': i + 1,
                'name': class_name,
                'supercategory': supercategories.get(class_name, 'object')
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
                
                # Get actual image dimensions from the first mask
                img_width = self.image_width if self.image_width > 0 else 1920
                img_height = self.image_height if self.image_height > 0 else 1080
                
                if annotations and 'mask' in annotations[0]:
                    # Use actual dimensions from mask
                    mask_height, mask_width = annotations[0]['mask'].shape
                    img_width = mask_width
                    img_height = mask_height
                
                # Add image entry
                image_file = f"frames/{video_id}/frame_{frame_idx:06d}.jpg"
                coco_data['images'].append({
                    'id': image_id,
                    'file_name': image_file,
                    'width': img_width,
                    'height': img_height,
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

    def save_video_annotations(self, project_path, video_id, annotations):
        """Save video-level annotations (chamber, hive, and pollen) shared across all frames.

        Chamber, hive, and pollen are stored in **separate** PNG files so their pixels can
        overlap without one overwriting the other:
            video_annotations_chamber.png  — chamber instance IDs
            video_annotations_hive.png     — hive instance IDs
            video_annotations_pollen.png   — pollen instance IDs
            video_annotations.json         — metadata for all instances

        Args:
            project_path: Path to project directory
            video_id: Video identifier
            annotations: List of annotation dicts (category='chamber', 'hive', or 'pollen')
        """
        import cv2
        import tempfile
        import os
        from pathlib import Path
        project_path = Path(project_path)

        png_dir = project_path / 'annotations/png' / video_id
        json_dir = project_path / 'annotations/json' / video_id
        png_dir.mkdir(parents=True, exist_ok=True)
        json_dir.mkdir(parents=True, exist_ok=True)

        json_file = json_dir / 'video_annotations.json'

        json_data = []

        mask_annotations = [ann for ann in annotations
                            if 'mask' in ann and not ann.get('bbox_only', False)]
        bbox_only_annotations = [ann for ann in annotations
                                  if ann.get('bbox_only', False) or 'mask' not in ann]

        # Write one PNG per category so they never stomp each other's pixels
        for category in ('chamber', 'hive', 'pollen'):
            cat_anns = [ann for ann in mask_annotations
                        if ann.get('category', 'chamber') == category]
            png_file = png_dir / f'video_annotations_{category}.png'

            if cat_anns:
                h, w = cat_anns[0]['mask'].shape[:2]
                cat_mask = np.zeros((h, w), dtype=np.uint16)
                for ann in cat_anns:
                    mask_id = ann.get('mask_id', 0)
                    if mask_id > 0:
                        if ann['mask'].shape[:2] != (h, w):
                            print(f"Warning: save_video_annotations skipping {category} "
                                  f"annotation {mask_id} – shape {ann['mask'].shape[:2]} != {(h, w)}")
                        else:
                            cat_mask[ann['mask'] > 0] = mask_id
                    ann_meta = {k: v for k, v in ann.items() if k not in ['mask', 'mask_rle']}
                    json_data.append(ann_meta)

                tmp = tempfile.NamedTemporaryFile(mode='wb', delete=False, dir=png_dir,
                                                  prefix=f'.tmp_video_ann_{category}_', suffix='.png')
                tmp.close()
                try:
                    cv2.imwrite(tmp.name, cat_mask)
                    os.replace(tmp.name, png_file)
                except Exception as e:
                    if os.path.exists(tmp.name):
                        os.unlink(tmp.name)
                    raise e
            else:
                # No annotations for this category — delete stale PNG
                if png_file.exists():
                    png_file.unlink()

        # Add bbox-only annotations to metadata
        for ann in bbox_only_annotations:
            ann_meta = {k: v for k, v in ann.items() if k not in ['mask', 'mask_rle']}
            json_data.append(ann_meta)

        try:
            json_data_sanitized = sanitize_for_json(json_data)
        except Exception:
            json_data_sanitized = json_data

        # Load existing data to preserve aruco_tracking and metadata
        existing_aruco_tracking = {}
        existing_metadata = {}
        if json_file.exists():
            try:
                with open(json_file, 'r') as f:
                    existing_data = json.load(f)
                    # Handle both legacy (array) and new (object) formats
                    if isinstance(existing_data, dict):
                        existing_aruco_tracking = existing_data.get('aruco_tracking', {})
                        existing_metadata = existing_data.get('metadata', {})
            except Exception:
                pass  # If load fails, start fresh

        # Create new format: object with annotations, aruco_tracking, and metadata
        video_data = {
            'annotations': json_data_sanitized,
            'aruco_tracking': existing_aruco_tracking,
            'metadata': existing_metadata
        }

        tmp = tempfile.NamedTemporaryFile(mode='w', delete=False, dir=json_dir,
                                          prefix='.tmp_video_ann_', suffix='.json')
        tmp_path = tmp.name
        try:
            json.dump(video_data, tmp, indent=2)
            tmp.flush()
            os.fsync(tmp.fileno())
            tmp.close()
            os.replace(tmp_path, json_file)
        except Exception as e:
            tmp.close()
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
            raise e

    def load_video_annotations(self, project_path, video_id):
        """Load video-level annotations (chamber, hive, and pollen) shared across all frames.

        Loads per-category PNGs (video_annotations_chamber.png /
        video_annotations_hive.png / video_annotations_pollen.png).  Falls back to the legacy single
        video_annotations.png if per-category files are absent.

        Args:
            project_path: Path to project directory
            video_id: Video identifier

        Returns:
            Tuple of (annotations_list, aruco_tracking_dict) or ([], {}) if not found
            - annotations_list: List of annotation dicts
            - aruco_tracking_dict: Dict mapping aruco_id (str) -> instance_id (int)
        """
        import cv2
        from pathlib import Path
        project_path = Path(project_path)

        png_dir = project_path / 'annotations/png' / video_id
        json_file = project_path / 'annotations/json' / video_id / 'video_annotations.json'

        if not json_file.exists():
            return [], {}

        try:
            with open(json_file, 'r') as f:
                file_data = json.load(f)
        except Exception as e:
            print(f"Warning: Failed to load video annotations for {video_id}: {e}")
            return [], {}

        # Handle both legacy (array) and new (object) formats
        if isinstance(file_data, list):
            # Legacy format: just an array of annotations
            json_data = file_data
            aruco_tracking = {}
        else:
            # New format: object with annotations, aruco_tracking, metadata
            json_data = file_data.get('annotations', [])
            aruco_tracking = file_data.get('aruco_tracking', {})

        # Load per-category mask PNGs
        category_masks = {}
        for category in ('chamber', 'hive', 'pollen'):
            cat_png = png_dir / f'video_annotations_{category}.png'
            if cat_png.exists():
                category_masks[category] = cv2.imread(str(cat_png), cv2.IMREAD_UNCHANGED)

        # Legacy fallback: single video_annotations.png (no per-category split)
        legacy_mask = None
        legacy_png = png_dir / 'video_annotations.png'
        if not category_masks and legacy_png.exists():
            legacy_mask = cv2.imread(str(legacy_png), cv2.IMREAD_UNCHANGED)

        annotations = []
        for ann_meta in json_data:
            mask_id = ann_meta.get('mask_id', 0)
            category = ann_meta.get('category', 'chamber')
            ann = ann_meta.copy()

            if not ann_meta.get('bbox_only', False) and mask_id > 0:
                # Prefer per-category PNG; fall back to legacy combined PNG
                cat_mask = category_masks.get(category, legacy_mask)
                if cat_mask is not None:
                    ann['mask'] = (cat_mask == mask_id).astype(np.uint8) * 255

            annotations.append(ann)

        return annotations, aruco_tracking

    def update_aruco_tracking(self, project_path, video_id, aruco_id, instance_id):
        """Update video-level ArUco tracking mapping.
        
        Args:
            project_path: Path to project directory
            video_id: Video identifier
            aruco_id: ArUco marker ID (int)
            instance_id: Instance ID to associate (int)
        """
        import tempfile
        import os
        from pathlib import Path
        project_path = Path(project_path)
        
        json_dir = project_path / 'annotations/json' / video_id
        json_file = json_dir / 'video_annotations.json'
        
        if not json_file.exists():
            # No video annotations file yet, create minimal structure
            json_dir.mkdir(parents=True, exist_ok=True)
            video_data = {
                'annotations': [],
                'aruco_tracking': {},
                'metadata': {}
            }
        else:
            # Load existing data
            try:
                with open(json_file, 'r') as f:
                    file_data = json.load(f)
                
                # Handle legacy format
                if isinstance(file_data, list):
                    video_data = {
                        'annotations': file_data,
                        'aruco_tracking': {},
                        'metadata': {}
                    }
                else:
                    video_data = file_data
                    # Ensure all keys exist
                    if 'annotations' not in video_data:
                        video_data['annotations'] = []
                    if 'aruco_tracking' not in video_data:
                        video_data['aruco_tracking'] = {}
                    if 'metadata' not in video_data:
                        video_data['metadata'] = {}
            except Exception as e:
                print(f"Warning: Could not load existing video annotations: {e}")
                video_data = {
                    'annotations': [],
                    'aruco_tracking': {},
                    'metadata': {}
                }
        
        # Update ArUco tracking
        video_data['aruco_tracking'][str(aruco_id)] = instance_id
        
        # Save atomically
        tmp = tempfile.NamedTemporaryFile(mode='w', delete=False, dir=json_dir,
                                          prefix='.tmp_video_ann_', suffix='.json')
        tmp_path = tmp.name
        try:
            json.dump(video_data, tmp, indent=2)
            tmp.flush()
            os.fsync(tmp.fileno())
            tmp.close()
            os.replace(tmp_path, json_file)
        except Exception as e:
            tmp.close()
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
            raise e
    
    def remove_aruco_tracking(self, project_path, video_id, aruco_id):
        """Remove an ArUco tracking entry from video-level tracking.
        
        Args:
            project_path: Path to project directory
            video_id: Video identifier
            aruco_id: ArUco marker ID (int) to remove
        """
        import tempfile
        import os
        from pathlib import Path
        project_path = Path(project_path)
        
        json_dir = project_path / 'annotations/json' / video_id
        json_file = json_dir / 'video_annotations.json'
        
        if not json_file.exists():
            # No file to remove from
            return
        
        try:
            with open(json_file, 'r') as f:
                file_data = json.load(f)
            
            # Handle legacy format
            if isinstance(file_data, list):
                video_data = {
                    'annotations': file_data,
                    'aruco_tracking': {},
                    'metadata': {}
                }
            else:
                video_data = file_data
                # Ensure all keys exist
                if 'annotations' not in video_data:
                    video_data['annotations'] = []
                if 'aruco_tracking' not in video_data:
                    video_data['aruco_tracking'] = {}
                if 'metadata' not in video_data:
                    video_data['metadata'] = {}
        except Exception as e:
            print(f"Warning: Could not load existing video annotations: {e}")
            return
        
        # Remove ArUco tracking entry if it exists
        aruco_str = str(aruco_id)
        if aruco_str in video_data['aruco_tracking']:
            del video_data['aruco_tracking'][aruco_str]
            
            # Save atomically
            tmp = tempfile.NamedTemporaryFile(mode='w', delete=False, dir=json_dir,
                                              prefix='.tmp_video_ann_', suffix='.json')
            tmp_path = tmp.name
            try:
                json.dump(video_data, tmp, indent=2)
                tmp.flush()
                os.fsync(tmp.fileno())
                tmp.close()
                os.replace(tmp_path, json_file)
            except Exception as e:
                tmp.close()
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)
                raise e
    
    def clear_all_aruco_tracking(self, project_path, video_id):
        """Clear all ArUco tracking entries for a video.
        
        Args:
            project_path: Path to project directory
            video_id: Video identifier
            
        Returns:
            int: Number of ArUco tracking entries removed
        """
        import tempfile
        import os
        from pathlib import Path
        project_path = Path(project_path)
        
        json_dir = project_path / 'annotations/json' / video_id
        json_file = json_dir / 'video_annotations.json'
        
        if not json_file.exists():
            # No file to clear from
            return 0
        
        try:
            with open(json_file, 'r') as f:
                file_data = json.load(f)
            
            # Handle legacy format
            if isinstance(file_data, list):
                video_data = {
                    'annotations': file_data,
                    'aruco_tracking': {},
                    'metadata': {}
                }
                count = 0  # No tracking in legacy format
            else:
                video_data = file_data
                # Ensure all keys exist
                if 'annotations' not in video_data:
                    video_data['annotations'] = []
                if 'aruco_tracking' not in video_data:
                    video_data['aruco_tracking'] = {}
                if 'metadata' not in video_data:
                    video_data['metadata'] = {}
                
                # Count how many entries we're removing
                count = len(video_data['aruco_tracking'])
        except Exception as e:
            print(f"Warning: Could not load existing video annotations: {e}")
            return 0
        
        if count == 0:
            # Nothing to clear
            return 0
        
        # Clear all ArUco tracking
        video_data['aruco_tracking'] = {}
        
        # Save atomically
        tmp = tempfile.NamedTemporaryFile(mode='w', delete=False, dir=json_dir,
                                          prefix='.tmp_video_ann_', suffix='.json')
        tmp_path = tmp.name
        try:
            json.dump(video_data, tmp, indent=2)
            tmp.flush()
            os.fsync(tmp.fileno())
            tmp.close()
            os.replace(tmp_path, json_file)
            return count
        except Exception as e:
            tmp.close()
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
            raise e
