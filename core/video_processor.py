"""
Video processing utilities
"""

import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm


class VideoProcessor:
    """Video processing and frame extraction"""
    
    def __init__(self):
        self.video_path = None
        self.cap = None
        
    def extract_frames(self, video_path, output_dir=None, fps=10, 
                      start_frame=0, end_frame=None, resize=None):
        """
        Extract frames from video
        
        Args:
            video_path: Path to video file
            output_dir: Directory to save frames (if None, returns frames in memory)
            fps: Frames per second to extract
            start_frame: Start frame index
            end_frame: End frame index (None for all)
            resize: Tuple (width, height) to resize frames
            
        Returns:
            List of frame paths (if output_dir) or frame arrays
        """
        self.video_path = Path(video_path)
        
        # Try to open video with different backends for MJPEG support
        self.cap = None
        backends = [cv2.CAP_ANY, cv2.CAP_FFMPEG, cv2.CAP_GSTREAMER]
        
        for backend in backends:
            self.cap = cv2.VideoCapture(str(video_path), backend)
            if self.cap.isOpened():
                # Test if we can actually read a frame
                ret, test_frame = self.cap.read()
                if ret and test_frame is not None:
                    # Reset to beginning
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    break
                else:
                    self.cap.release()
                    self.cap = None
        
        if self.cap is None or not self.cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        print(f"Successfully opened video with backend")
            
        # Get video properties
        total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        video_fps = self.cap.get(cv2.CAP_PROP_FPS)
        
        # Debug output
        print(f"Video properties - Total frames: {total_frames}, FPS: {video_fps}")
        
        # MJPEG files often report 0 frames and 0 FPS - handle this
        if total_frames <= 0:
            print("Warning: Video reports 0 frames (common for MJPEG). Will read until end of stream.")
            total_frames = float('inf')  # Read until end
            
        if video_fps <= 0:
            print(f"Warning: Video reports invalid FPS ({video_fps}). Using default 30 FPS.")
            video_fps = 30.0  # Default FPS for frame extraction
        
        if end_frame is None or end_frame == float('inf'):
            # Will read until stream ends
            end_frame = float('inf')
            use_progress_bar = False
        else:
            use_progress_bar = True
            
        # Calculate frame interval
        frame_interval = max(1, int(video_fps / fps))
        
        frames = []
        frame_idx = 0
        extracted_count = 0
        
        # Create output directory if saving to disk
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
        print(f"Extracting frames from {self.video_path.name}")
        print(f"Video FPS: {video_fps}, Extracting at: {fps} fps")
        print(f"Frame interval: {frame_interval}")
        print(f"Start frame: {start_frame}, End frame: {end_frame}")
        print(f"Output dir: {output_dir}")
        
        # Use progress bar only if we know total frames
        if use_progress_bar:
            pbar = tqdm(total=end_frame - start_frame)
        else:
            pbar = None
            print("Reading frames until end of stream...")
        
        try:
            while True:
                ret, frame = self.cap.read()
                
                if not ret:
                    # End of video
                    print(f"End of video reached at frame_idx {frame_idx}")
                    break
                    
                if frame_idx >= end_frame:
                    print(f"Reached end_frame limit at frame_idx {frame_idx}")
                    break
                
                # Verify frame is valid
                if frame is None or frame.size == 0:
                    print(f"Invalid frame at frame_idx {frame_idx}")
                    frame_idx += 1
                    if pbar:
                        pbar.update(1)
                    continue
                    
                # Check if we should extract this frame
                if (frame_idx >= start_frame and 
                    (frame_idx - start_frame) % frame_interval == 0):
                    
                    # Resize if requested
                    if resize:
                        frame = cv2.resize(frame, resize)
                        
                    if output_dir:
                        # Save to disk
                        frame_path = output_dir / f"frame_{extracted_count:06d}.png"
                        success = cv2.imwrite(str(frame_path), frame)
                        if success:
                            frames.append(frame_path)
                            if extracted_count < 3 or extracted_count % 10 == 0:
                                print(f"Saved frame {extracted_count} to {frame_path.name}")
                        else:
                            print(f"Failed to save frame to {frame_path}")
                    else:
                        # Keep in memory (convert to RGB)
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        frames.append(frame_rgb)
                        if extracted_count < 3 or extracted_count % 10 == 0:
                            print(f"Kept frame {extracted_count} in memory")
                        
                    extracted_count += 1
                    
                frame_idx += 1
                if pbar:
                    pbar.update(1)
        finally:
            if pbar:
                pbar.close()
                
        self.cap.release()
        print(f"Extracted {extracted_count} frames")
        
        return frames
        
    def get_video_info(self, video_path):
        """Get video metadata"""
        # Try different backends for MJPEG support
        cap = None
        backends = [cv2.CAP_ANY, cv2.CAP_FFMPEG, cv2.CAP_GSTREAMER]
        
        for backend in backends:
            cap = cv2.VideoCapture(str(video_path), backend)
            if cap.isOpened():
                # Test if we can read properties
                fps = cap.get(cv2.CAP_PROP_FPS)
                if fps > 0:
                    break
                else:
                    cap.release()
                    cap = None
        
        if cap is None or not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
            
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Handle MJPEG files which may report 0 fps or frame_count
        if fps <= 0:
            print(f"Warning: Video reports invalid FPS ({fps}), using default 30.0")
            fps = 30.0
            
        if frame_count <= 0:
            print(f"Warning: Video reports 0 frames (common for MJPEG). Actual count unknown.")
            # Try to estimate by reading frames
            test_count = 0
            while test_count < 10:  # Test first 10 frames
                ret, _ = cap.read()
                if not ret:
                    break
                test_count += 1
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Reset
            if test_count > 0:
                frame_count = -1  # Unknown but has frames
        
        # Calculate duration carefully
        if frame_count > 0 and fps > 0:
            duration = frame_count / fps
        else:
            duration = 0  # Unknown
        
        info = {
            'fps': fps,
            'frame_count': frame_count,
            'width': width,
            'height': height,
            'duration': duration
        }
        
        cap.release()
        return info
        
    def read_frame(self, frame_idx):
        """Read a specific frame by index"""
        if self.cap is None:
            raise ValueError("No video loaded")
            
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = self.cap.read()
        
        if ret:
            return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return None
