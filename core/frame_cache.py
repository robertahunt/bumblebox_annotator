"""
Frame cache with background preloading for fast frame switching
"""

import threading
import queue
import time
import cv2
import numpy as np
from pathlib import Path
from collections import OrderedDict


class FrameCache:
    """LRU cache for frame images with thread-safe access"""
    
    def __init__(self, max_size=10):
        """
        Initialize frame cache
        
        Args:
            max_size: Maximum number of frames to keep in cache
        """
        self.cache = OrderedDict()  # frame_idx -> numpy array (RGB)
        self.max_size = max_size
        self.lock = threading.Lock()
        
    def get(self, frame_idx):
        """
        Get frame from cache (thread-safe)
        
        Args:
            frame_idx: Frame index to retrieve
            
        Returns:
            Numpy array in RGB format, or None if not cached
        """
        with self.lock:
            if frame_idx in self.cache:
                # Move to end (most recently used)
                self.cache.move_to_end(frame_idx)
                return self.cache[frame_idx]
        return None
    
    def put(self, frame_idx, image):
        """
        Add frame to cache (thread-safe)
        
        Args:
            frame_idx: Frame index
            image: Numpy array (grayscale or RGB)
        """
        import gc
        
        with self.lock:
            # If already exists, update it
            if frame_idx in self.cache:
                self.cache.move_to_end(frame_idx)
                # Delete old image before replacing
                old_image = self.cache[frame_idx]
                del old_image
                self.cache[frame_idx] = image
                # print(f"Frame cache: updated frame {frame_idx}, cache size={len(self.cache)}")
            else:
                # Evict oldest if at capacity BEFORE adding new one
                if len(self.cache) >= self.max_size:
                    evicted_idx, evicted_image = self.cache.popitem(last=False)
                    # Explicitly delete evicted image and force garbage collection
                    del evicted_image
                    gc.collect()
                    print(f"Frame cache: evicted frame {evicted_idx} to make room for frame {frame_idx}")
                
                # Add new entry
                self.cache[frame_idx] = image
                print(f"Frame cache: cached frame {frame_idx}, cache size={len(self.cache)}/{self.max_size}")
    
    def clear(self):
        """Clear all cached frames"""
        with self.lock:
            self.cache.clear()
    
    def get_size(self):
        """Get current cache size"""
        with self.lock:
            return len(self.cache)
    
    def is_cached(self, frame_idx):
        """Check if frame is in cache"""
        with self.lock:
            return frame_idx in self.cache


class PreloadWorker(threading.Thread):
    """Background worker thread for preloading frames and annotations"""
    
    def __init__(self, frame_cache, annotation_manager=None):
        """
        Initialize preload worker
        
        Args:
            frame_cache: FrameCache instance to populate
            annotation_manager: AnnotationManager instance for preloading annotations (optional)
        """
        super().__init__(daemon=True)
        self.frame_cache = frame_cache
        self.annotation_manager = annotation_manager
        self.request_queue = queue.Queue()
        self.running = True
        self.current_request_id = 0
        self.lock = threading.Lock()
        self.project_path = None
        self.current_video_id = None
        self.frame_video_ids = []
        
    def request_preload(self, current_idx, frame_paths, selected_indices=None, 
                       project_path=None, video_id=None, frame_video_ids=None):
        """
        Request preloading of frames and annotations around current index
        
        Args:
            current_idx: Current frame index
            frame_paths: List of frame paths
            selected_indices: List of indices that are selected for train/val
            project_path: Path to project (for annotation loading)
            video_id: Current video ID (for annotation loading)
            frame_video_ids: List mapping frame index to video ID
        """
        with self.lock:
            self.current_request_id += 1
            request_id = self.current_request_id
            # Update context for annotation loading
            self.project_path = project_path
            self.current_video_id = video_id
            self.frame_video_ids = frame_video_ids or []
        
        self.request_queue.put({
            'request_id': request_id,
            'current_idx': current_idx,
            'frame_paths': frame_paths,
            'selected_indices': selected_indices or []
        })
    
    def run(self):
        """Main worker loop"""
        while self.running:
            try:
                # Get preload request with timeout
                try:
                    request = self.request_queue.get(timeout=0.1)
                except queue.Empty:
                    continue
                
                # Clear queue of old requests (keep only latest)
                while not self.request_queue.empty():
                    try:
                        request = self.request_queue.get_nowait()
                    except queue.Empty:
                        break
                
                # Process the latest request
                self._process_request(request)
                
            except Exception as e:
                print(f"PreloadWorker error: {e}")
    
    def _process_request(self, request):
        """
        Process a preload request
        
        Args:
            request: Dict with current_idx, frame_paths, selected_indices
        """
        current_idx = request['current_idx']
        frame_paths = request['frame_paths']
        
        if not frame_paths:
            return
        
        # Define preload window: 1 frame behind, 2-3 frames ahead
        # This balances responsiveness with memory usage
        indices_to_preload = []
        
        # Preload backward (1 frame)
        if current_idx > 0:
            indices_to_preload.append(current_idx - 1)
        
        # Preload forward (2-3 frames)
        for offset in range(1, 4):
            if current_idx + offset < len(frame_paths):
                indices_to_preload.append(current_idx + offset)
        
        # Preload frames and annotations
        for idx in indices_to_preload:
            # Check if already cached - skip if so
            if self.frame_cache.is_cached(idx):
                continue
            
            # Check if cache is full - don't cause evictions
            if self.frame_cache.get_size() >= self.frame_cache.max_size:
                break
            
            # Load frame
            frame_image = self._load_frame(frame_paths[idx])
            if frame_image is not None:
                self.frame_cache.put(idx, frame_image)
                # print(f"Preloaded frame {idx}")
            
            # Load annotations if annotation manager is available
            if self.annotation_manager and self.project_path:
                # Get video ID for this frame
                if idx < len(self.frame_video_ids):
                    frame_video_id = self.frame_video_ids[idx]
                else:
                    frame_video_id = self.current_video_id
                
                # Only preload if not already cached
                cached_anns = self.annotation_manager.get_frame_annotations(idx)
                if not cached_anns and frame_video_id:
                    # Extract frame index within video from filename
                    frame_idx_in_video = self._get_frame_idx_in_video(idx, frame_paths)
                    
                    # Load annotations
                    try:
                        annotations = self.annotation_manager.load_frame_annotations(
                            self.project_path, frame_video_id, frame_idx_in_video
                        )
                        if annotations:
                            self.annotation_manager.set_frame_annotations(idx, annotations)
                            # print(f"Preloaded annotations for frame {idx}")
                    except Exception as e:
                        # Silent failure - annotation preloading is best-effort
                        pass
    
    def _get_frame_idx_in_video(self, list_idx, frame_paths):
        """
        Get the frame index within the video for a given list index
        
        Extracts frame number from filename: frame_000001.jpg -> 1
        """
        if list_idx >= len(frame_paths):
            return list_idx
        
        frame = frame_paths[list_idx]
        if isinstance(frame, (Path, str)):
            # Extract frame number from filename
            frame_path = Path(frame)
            if frame_path.stem.startswith('frame_'):
                try:
                    return int(frame_path.stem.split('_')[1])
                except (ValueError, IndexError):
                    pass
        
        # Fallback to list index
        return list_idx
    
    def _load_frame(self, frame_path):
        """
        Load a frame from disk
        
        Args:
            frame_path: Path to frame file or numpy array
            
        Returns:
            Numpy array in RGB format, or None if failed
        """
        # Check if it's already a numpy array
        if isinstance(frame_path, np.ndarray):
            return frame_path
        
        # Load from file
        path = Path(frame_path)
        if not path.exists():
            return None
        
        # Load with OpenCV
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)  # Load as grayscale to save memory
        if image is None or image.size == 0:
            return None
        
        # Return grayscale (H×W) - conversion to RGB done only for display
        return image
    
    def stop(self):
        """Stop the worker thread"""
        self.running = False
    
    def clear_queue(self):
        """Clear all pending requests"""
        while not self.request_queue.empty():
            try:
                self.request_queue.get_nowait()
            except queue.Empty:
                break
