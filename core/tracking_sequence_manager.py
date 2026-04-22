"""
Manager for tracking sequences - pairs/groups of consecutive frames used for tracking validation.

A tracking sequence represents consecutive frames where ground truth annotations exist,
used to evaluate tracking algorithm performance.
"""

import json
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime


@dataclass
class TrackingSequence:
    """Represents a sequence of consecutive frames for tracking validation"""
    sequence_id: str  # Unique ID (video_id + timestamp)
    video_id: str
    start_frame: int
    end_frame: int  # Inclusive
    created_date: str
    notes: str = ""
    enabled: bool = True  # Can be disabled without deleting
    
    @property
    def length(self) -> int:
        """Number of frames in sequence"""
        return self.end_frame - self.start_frame + 1
    
    @property
    def frame_range(self) -> List[int]:
        """List of frame indices in sequence"""
        return list(range(self.start_frame, self.end_frame + 1))
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: dict) -> 'TrackingSequence':
        """Create from dictionary"""
        return cls(**data)


class TrackingSequenceManager:
    """Manages tracking sequences for a project"""
    
    SEQUENCES_FILE = "tracking_sequences.json"
    
    def __init__(self, project_path: Path):
        """
        Initialize manager for a project.
        
        Args:
            project_path: Path to project directory
        """
        self.project_path = Path(project_path)
        self.sequences_file = self.project_path / self.SEQUENCES_FILE
        self.sequences: List[TrackingSequence] = []
        self.load()
    
    def load(self):
        """Load sequences from project"""
        if not self.sequences_file.exists():
            self.sequences = []
            return
        
        try:
            with open(self.sequences_file, 'r') as f:
                data = json.load(f)
                self.sequences = [TrackingSequence.from_dict(seq) for seq in data['sequences']]
        except (json.JSONDecodeError, KeyError) as e:
            print(f"Error loading tracking sequences: {e}")
            self.sequences = []
    
    def save(self):
        """Save sequences to project"""
        data = {
            'version': '1.0',
            'sequences': [seq.to_dict() for seq in self.sequences]
        }
        
        with open(self.sequences_file, 'w') as f:
            json.dump(data, f, indent=2)
    
    def add_sequence(self, video_id: str, start_frame: int, end_frame: int, 
                     notes: str = "") -> TrackingSequence:
        """
        Add a new tracking sequence.
        
        Args:
            video_id: Video identifier
            start_frame: Starting frame index
            end_frame: Ending frame index (inclusive)
            notes: Optional notes about this sequence
            
        Returns:
            Created TrackingSequence
        """
        # Generate unique ID
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        sequence_id = f"{video_id}_{start_frame}_{end_frame}_{timestamp}"
        
        sequence = TrackingSequence(
            sequence_id=sequence_id,
            video_id=video_id,
            start_frame=start_frame,
            end_frame=end_frame,
            created_date=datetime.now().isoformat(),
            notes=notes,
            enabled=True
        )
        
        self.sequences.append(sequence)
        self.save()
        return sequence
    
    def remove_sequence(self, sequence_id: str) -> bool:
        """
        Remove a tracking sequence.
        
        Args:
            sequence_id: Sequence ID to remove
            
        Returns:
            True if removed, False if not found
        """
        original_count = len(self.sequences)
        self.sequences = [s for s in self.sequences if s.sequence_id != sequence_id]
        
        if len(self.sequences) < original_count:
            self.save()
            return True
        return False
    
    def update_sequence(self, sequence_id: str, **kwargs) -> bool:
        """
        Update sequence properties.
        
        Args:
            sequence_id: Sequence ID to update
            **kwargs: Properties to update (notes, enabled, start_frame, end_frame, etc.)
            
        Returns:
            True if updated, False if not found
        """
        for seq in self.sequences:
            if seq.sequence_id == sequence_id:
                # Check if frame range is being updated
                frame_range_changed = 'start_frame' in kwargs or 'end_frame' in kwargs
                
                # Update attributes
                for key, value in kwargs.items():
                    if hasattr(seq, key):
                        setattr(seq, key, value)
                
                # If frame range changed, regenerate sequence_id to reflect new range
                if frame_range_changed:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    new_id = f"{seq.video_id}_{seq.start_frame}_{seq.end_frame}_{timestamp}"
                    seq.sequence_id = new_id
                
                self.save()
                return True
        return False
    
    def get_sequence(self, sequence_id: str) -> Optional[TrackingSequence]:
        """Get sequence by ID"""
        for seq in self.sequences:
            if seq.sequence_id == sequence_id:
                return seq
        return None
    
    def get_sequences_for_video(self, video_id: str) -> List[TrackingSequence]:
        """Get all sequences for a specific video"""
        return [s for s in self.sequences if s.video_id == video_id]
    
    def get_sequence_by_frames(self, video_id: str, start_frame: int, end_frame: int) -> Optional[TrackingSequence]:
        """Get sequence by video_id and frame range (useful after editing)"""
        for seq in self.sequences:
            if (seq.video_id == video_id and 
                seq.start_frame == start_frame and 
                seq.end_frame == end_frame):
                return seq
        return None
    
    def get_enabled_sequences(self) -> List[TrackingSequence]:
        """Get all enabled sequences"""
        return [s for s in self.sequences if s.enabled]
    
    def get_all_sequences(self) -> List[TrackingSequence]:
        """Get all sequences"""
        return self.sequences.copy()
    
    def check_sequence_annotations(self, sequence: TrackingSequence, 
                                   annotation_manager) -> Tuple[bool, List[int]]:
        """
        Check if all frames in sequence have annotations.
        
        Args:
            sequence: Tracking sequence to check
            annotation_manager: AnnotationManager instance
            
        Returns:
            Tuple of (all_annotated, list of missing frame indices)
        """
        missing_frames = []
        
        for frame_idx in sequence.frame_range:
            # Check if frame has annotations in either json (mask) or bbox folders
            json_file = self.project_path / "annotations" / "json" / sequence.video_id / f"frame_{frame_idx:06d}.json"
            bbox_file = self.project_path / "annotations" / "bbox" / sequence.video_id / f"frame_{frame_idx:06d}.json"
            
            # Frame is annotated if either file exists and contains annotations
            has_annotations = False
            if json_file.exists():
                # Check if json file has actual annotations (not empty)
                import json
                try:
                    with open(json_file, 'r') as f:
                        data = json.load(f)
                        if data and len(data) > 0:  # Has at least one annotation
                            has_annotations = True
                except:
                    pass
            
            if not has_annotations and bbox_file.exists():
                # Check if bbox file has actual annotations
                import json
                try:
                    with open(bbox_file, 'r') as f:
                        data = json.load(f)
                        if data and len(data) > 0:  # Has at least one annotation
                            has_annotations = True
                except:
                    pass
            
            if not has_annotations:
                missing_frames.append(frame_idx)
        
        return (len(missing_frames) == 0, missing_frames)
    
    def validate_sequences(self, annotation_manager) -> Dict[str, Tuple[bool, List[int]]]:
        """
        Validate all sequences have annotations.
        
        Returns:
            Dictionary mapping sequence_id to (all_annotated, missing_frames)
        """
        results = {}
        for seq in self.sequences:
            results[seq.sequence_id] = self.check_sequence_annotations(seq, annotation_manager)
        return results
