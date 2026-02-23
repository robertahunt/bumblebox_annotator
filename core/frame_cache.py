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
    """Background worker thread for preloading frames"""
    
    def __init__(self, frame_cache):
        """
        Initialize preload worker
        
        Args:
            frame_cache: FrameCache instance to populate
        """
        super().__init__(daemon=True)
        self.frame_cache = frame_cache
        self.request_queue = queue.Queue()
        self.running = True
        self.current_request_id = 0
        self.lock = threading.Lock()
        
    def request_preload(self, current_idx, frame_paths, selected_indices=None):
        """
        Request preloading of frames around current index
        
        Args:
            current_idx: Current frame index
            frame_paths: List of frame paths
            selected_indices: List of indices that are selected for train/val
        """
        with self.lock:
            self.current_request_id += 1
            request_id = self.current_request_id
        
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
        # Disabled: background preloading was causing excessive cache churn and memory issues
        # The cache still works - it just stores frames as they are viewed, not predictively
        return
    
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
