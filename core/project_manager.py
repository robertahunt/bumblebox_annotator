"""
Project management - handles folder structure and video organization
"""

import json
import shutil
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple, Optional
import cv2


class ProjectManager:
    """Manages project structure and video organization"""
    
    def __init__(self, project_path: Optional[Path] = None):
        self.project_path = Path(project_path) if project_path else None
        self.dataset_info = {}
        
    def create_project(self, project_path: Path, project_name: str, 
                      frames_per_video: int = 15) -> Dict:
        """
        Create new project with standard folder structure
        
        Args:
            project_path: Path to project directory
            project_name: Name of the project
            frames_per_video: Number of frames to extract per video
            
        Returns:
            dict with project info
        """
        self.project_path = Path(project_path)
        
        # Create folder structure
        folders = [
            'input_data/train',
            'input_data/val',
            'frames',
            'annotations/pkl',
            'annotations/coco',
            'models'
        ]
        
        for folder in folders:
            (self.project_path / folder).mkdir(parents=True, exist_ok=True)
        
        # Create project info
        project_info = {
            'name': project_name,
            'created': datetime.now().isoformat(),
            'modified': datetime.now().isoformat(),
            'frames_per_video': frames_per_video,
            'version': '2.0'
        }
        
        # Save project.json
        with open(self.project_path / 'annotations/project.json', 'w') as f:
            json.dump(project_info, f, indent=2)
        
        # Create initial dataset_info.json
        self._update_dataset_info()
        
        return project_info
    
    def load_project(self, project_path: Path) -> Dict:
        """Load existing project"""
        self.project_path = Path(project_path)
        
        project_file = self.project_path / 'annotations/project.json'
        if not project_file.exists():
            raise FileNotFoundError(f"Project file not found: {project_file}")
        
        with open(project_file, 'r') as f:
            project_info = json.load(f)
        
        # Refresh dataset info
        self._update_dataset_info()
        
        return project_info
    
    def add_videos(self, video_paths: List[Path], split: str = 'train',
                  copy_to_project: bool = True) -> Dict:
        """
        Add videos to project
        
        Args:
            video_paths: List of video file paths
            split: 'train' or 'val'
            copy_to_project: Whether to copy videos or just reference them
            
        Returns:
            dict with results
        """
        if split not in ['train', 'val']:
            raise ValueError(f"Split must be 'train' or 'val', got: {split}")
        
        dest_folder = self.project_path / 'input_data' / split
        results = {'added': [], 'failed': []}
        
        for video_path in video_paths:
            try:
                video_id = video_path.stem  # filename without extension
                
                # Copy or move video
                if copy_to_project:
                    dest_path = dest_folder / video_path.name
                    shutil.copy2(video_path, dest_path)
                else:
                    # Could create symlink here if needed
                    dest_path = video_path
                
                results['added'].append({
                    'video_id': video_id,
                    'split': split,
                    'path': str(dest_path)
                })
                
            except Exception as e:
                results['failed'].append({
                    'video_path': str(video_path),
                    'error': str(e)
                })
        
        # Update dataset info
        self._update_dataset_info()
        
        return results
    
    def scan_videos(self) -> Dict[str, List[str]]:
        """
        Scan input_data folders to get current video lists
        
        Returns:
            dict with 'train' and 'val' lists of video IDs
        """
        videos = {'train': [], 'val': []}
        
        for split in ['train', 'val']:
            folder = self.project_path / 'input_data' / split
            if folder.exists():
                for video_file in folder.glob('*.mp4'):
                    videos[split].append(video_file.stem)
                # Also check for other video formats
                for ext in ['*.avi', '*.mov', '*.mkv', '*.mjpeg', '*.mjpg']:
                    for video_file in folder.glob(ext):
                        videos[split].append(video_file.stem)
        
        return videos
    
    def get_video_path(self, video_id: str) -> Optional[Path]:
        """Get full path to video file"""
        # Check train folder
        for ext in ['.mp4', '.avi', '.mov', '.mkv', '.mjpeg', '.mjpg']:
            path = self.project_path / 'input_data/train' / f'{video_id}{ext}'
            if path.exists():
                return path
        
        # Check val folder
        for ext in ['.mp4', '.avi', '.mov', '.mkv', '.mjpeg', '.mjpg']:
            path = self.project_path / 'input_data/val' / f'{video_id}{ext}'
            if path.exists():
                return path
        
        return None
    
    def get_video_split(self, video_id: str) -> Optional[str]:
        """Determine which split a video belongs to"""
        # Check train folder
        for ext in ['.mp4', '.avi', '.mov', '.mkv', '.mjpeg', '.mjpg']:
            if (self.project_path / 'input_data/train' / f'{video_id}{ext}').exists():
                return 'train'
        
        # Check val folder
        for ext in ['.mp4', '.avi', '.mov', '.mkv', '.mjpeg', '.mjpg']:
            if (self.project_path / 'input_data/val' / f'{video_id}{ext}').exists():
                return 'val'
        
        return None
    
    def move_video(self, video_id: str, to_split: str) -> bool:
        """Move video between train and val folders"""
        if to_split not in ['train', 'val']:
            raise ValueError(f"Split must be 'train' or 'val', got: {to_split}")
        
        # Find current location
        current_path = self.get_video_path(video_id)
        if not current_path:
            return False
        
        # Determine destination
        dest_folder = self.project_path / 'input_data' / to_split
        dest_path = dest_folder / current_path.name
        
        # Move file
        shutil.move(str(current_path), str(dest_path))
        
        # Update dataset info
        self._update_dataset_info()
        
        return True
    
    def _update_dataset_info(self):
        """Update dataset_info.json by scanning folders"""
        videos = self.scan_videos()
        
        self.dataset_info = {
            'created': datetime.now().isoformat(),
            'last_scanned': datetime.now().isoformat(),
            'train_videos': videos['train'],
            'val_videos': videos['val'],
            'total_videos': len(videos['train']) + len(videos['val']),
            'note': 'Generated from input_data/ folder structure. Move videos between train/val to change splits.'
        }
        
        # Save to file
        info_path = self.project_path / 'annotations/coco/dataset_info.json'
        with open(info_path, 'w') as f:
            json.dump(self.dataset_info, f, indent=2)
    
    def get_frames_dir(self, video_id: str) -> Path:
        """Get directory for video's extracted frames"""
        return self.project_path / 'frames' / video_id
    
    def get_annotations_dir(self, video_id: str) -> Path:
        """Get directory for video's annotations"""
        return self.project_path / 'annotations/pkl' / video_id
    
    def get_frame_path(self, video_id: str, frame_idx: int) -> Path:
        """Get path to specific frame image"""
        return self.get_frames_dir(video_id) / f'frame_{frame_idx:06d}.jpg'
    
    def get_annotation_path(self, video_id: str, frame_idx: int) -> Path:
        """Get path to specific frame annotation"""
        return self.get_annotations_dir(video_id) / f'frame_{frame_idx:06d}.pkl'
    
    def extract_video_frames(self, video_id: str, frame_indices: List[int]) -> Dict:
        """
        Extract specific frames from video
        
        Args:
            video_id: Video identifier
            frame_indices: List of frame indices to extract
            
        Returns:
            dict with extraction results
        """
        video_path = self.get_video_path(video_id)
        if not video_path:
            raise FileNotFoundError(f"Video not found: {video_id}")
        
        # Create frames directory
        frames_dir = self.get_frames_dir(video_id)
        frames_dir.mkdir(parents=True, exist_ok=True)
        
        # Try different backends for MJPEG support
        cap = None
        backends = [cv2.CAP_ANY, cv2.CAP_FFMPEG, cv2.CAP_GSTREAMER]
        
        for backend in backends:
            cap = cv2.VideoCapture(str(video_path), backend)
            if cap.isOpened():
                # Test if we can read a frame
                ret, test_frame = cap.read()
                if ret and test_frame is not None:
                    # Reset to beginning
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    break
                else:
                    cap.release()
                    cap = None
        
        if cap is None or not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"extract_video_frames: total_frames from video = {total_frames}")
        
        # For MJPEG or videos that report 0 frames, we need to read sequentially
        # Sort frame indices to read in order
        sorted_indices = sorted(frame_indices)
        frame_indices_set = set(frame_indices)
        
        extracted = []
        failed = []
        
        if total_frames <= 0:
            # MJPEG or unknown frame count - read sequentially until we have all requested frames
            print(f"Video reports 0 frames (likely MJPEG), reading sequentially for indices: {sorted_indices[:10]}...")
            max_frame_idx = max(frame_indices) if frame_indices else 0
            
            current_idx = 0
            while current_idx <= max_frame_idx:
                ret, frame = cap.read()
                
                if not ret:
                    # Reached end of video
                    print(f"Reached end of video at frame {current_idx}")
                    # Mark remaining requested frames as failed
                    for idx in frame_indices:
                        if idx >= current_idx and idx not in extracted:
                            failed.append(idx)
                    break
                
                # Check if this is a frame we want to extract
                if current_idx in frame_indices_set:
                    frame_path = self.get_frame_path(video_id, current_idx)
                    success = cv2.imwrite(str(frame_path), frame)
                    if success:
                        extracted.append(current_idx)
                        print(f"Extracted frame {current_idx}")
                    else:
                        failed.append(current_idx)
                        print(f"Failed to save frame {current_idx}")
                
                current_idx += 1
        else:
            # Normal video with known frame count - can seek directly
            print(f"Video has {total_frames} frames, seeking to specific frames")
            for frame_idx in frame_indices:
                if frame_idx >= total_frames:
                    failed.append(frame_idx)
                    continue
                
                # Seek to frame
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                
                if ret:
                    # Save frame
                    frame_path = self.get_frame_path(video_id, frame_idx)
                    cv2.imwrite(str(frame_path), frame)
                    extracted.append(frame_idx)
                else:
                    failed.append(frame_idx)
        
        cap.release()
        
        print(f"Extraction complete: {len(extracted)} extracted, {len(failed)} failed")
        
        return {
            'video_id': video_id,
            'extracted': len(extracted),
            'failed': len(failed),
            'frame_indices': extracted
        }
    
    def get_video_metadata(self, video_id: str) -> Dict:
        """Get metadata for a video"""
        video_path = self.get_video_path(video_id)
        if not video_path:
            return {}
        
        # Try different backends for MJPEG support
        cap = None
        backends = [cv2.CAP_ANY, cv2.CAP_FFMPEG, cv2.CAP_GSTREAMER]
        
        for backend in backends:
            cap = cv2.VideoCapture(str(video_path), backend)
            if cap.isOpened():
                # Test if we can read a frame
                ret, test_frame = cap.read()
                if ret and test_frame is not None:
                    # Reset to beginning
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    break
                else:
                    cap.release()
                    cap = None
        
        if cap is None or not cap.isOpened():
            return {}
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Handle MJPEG files which may report 0 fps or frame_count
        if fps <= 0:
            print(f"Warning: Video {video_id} reports invalid FPS ({fps}), using default 30.0")
            fps = 30.0
            
        if total_frames <= 0:
            print(f"Warning: Video {video_id} reports 0 frames (common for MJPEG).")
            # Try to count frames by reading through the video
            print("Counting frames by reading video stream...")
            frame_count = 0
            while True:
                ret, _ = cap.read()
                if not ret:
                    break
                frame_count += 1
                if frame_count % 100 == 0:
                    print(f"  Counted {frame_count} frames so far...")
            total_frames = frame_count
            print(f"Total frames counted: {total_frames}")
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Reset
        
        # Calculate duration
        if total_frames > 0 and fps > 0:
            duration_seconds = total_frames / fps
        else:
            duration_seconds = 0
        
        metadata = {
            'video_id': video_id,
            'fps': fps,
            'total_frames': total_frames,
            'width': width,
            'height': height,
            'duration_seconds': duration_seconds
        }
        
        cap.release()
        return metadata
    
    def select_frames_uniform(self, total_frames: int, n_frames: int) -> List[int]:
        """Select frames uniformly spaced throughout video"""
        if n_frames >= total_frames:
            return list(range(total_frames))
        
        interval = total_frames / n_frames
        return [int(i * interval) for i in range(n_frames)]
    
    def generate_coco_datasets(self, annotation_manager) -> Dict[str, Path]:
        """
        Generate COCO JSON files for train and val splits
        
        Args:
            annotation_manager: AnnotationManager instance
            
        Returns:
            dict with paths to generated COCO files
        """
        if not self.project_path:
            raise ValueError("No project loaded")
        
        # Get videos for each split
        train_videos = self.get_videos_by_split('train')
        val_videos = self.get_videos_by_split('val')
        
        results = {}
        
        # Generate train COCO if there are train videos
        if train_videos:
            train_path = annotation_manager.generate_coco_for_split(
                self.project_path, train_videos, 'train'
            )
            results['train'] = train_path
        
        # Generate val COCO if there are val videos
        if val_videos:
            val_path = annotation_manager.generate_coco_for_split(
                self.project_path, val_videos, 'val'
            )
            results['val'] = val_path
        
        return results
    
    def get_videos_by_split(self, split: str) -> List[str]:
        """Get list of video IDs for a specific split"""
        split_dir = self.project_path / f'input_data/{split}'
        if not split_dir.exists():
            return []
        
        video_ids = []
        for video_file in split_dir.glob('*'):
            if video_file.suffix.lower() in ['.mp4', '.avi', '.mov', '.mkv', '.mjpeg']:
                video_ids.append(video_file.stem)
        
        return sorted(video_ids)
    
    def get_dataset_statistics(self) -> Dict:
        """Get statistics about the dataset"""
        if not self.project_path:
            return {}
        
        train_videos = self.get_videos_by_split('train')
        val_videos = self.get_videos_by_split('val')
        
        # Count frames for each video
        train_frames = 0
        val_frames = 0
        
        for video_id in train_videos:
            frames_dir = self.get_frames_dir(video_id)
            if frames_dir.exists():
                train_frames += len(list(frames_dir.glob('frame_*.jpg')))
        
        for video_id in val_videos:
            frames_dir = self.get_frames_dir(video_id)
            if frames_dir.exists():
                val_frames += len(list(frames_dir.glob('frame_*.jpg')))
        
        # Count annotations
        train_annotations = 0
        val_annotations = 0
        
        for video_id in train_videos:
            ann_dir = self.get_annotations_dir(video_id)
            if ann_dir.exists():
                train_annotations += len(list(ann_dir.glob('frame_*.pkl')))
        
        for video_id in val_videos:
            ann_dir = self.get_annotations_dir(video_id)
            if ann_dir.exists():
                val_annotations += len(list(ann_dir.glob('frame_*.pkl')))
        
        return {
            'train': {
                'videos': len(train_videos),
                'frames': train_frames,
                'annotated_frames': train_annotations
            },
            'val': {
                'videos': len(val_videos),
                'frames': val_frames,
                'annotated_frames': val_annotations
            },
            'total': {
                'videos': len(train_videos) + len(val_videos),
                'frames': train_frames + val_frames,
                'annotated_frames': train_annotations + val_annotations
            }
        }
