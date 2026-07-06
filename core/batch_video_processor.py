"""
Frame-by-frame video processor for batch inference with tracking and ArUco
"""

import cv2
import gc
import numpy as np
import time
from bisect import bisect_left, bisect_right
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from collections import defaultdict
from scipy.spatial import cKDTree

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from core.instance_tracker import Detection
from core.marker_detector import MarkerDetector
from core.temporal_hive_prior import TemporalHiveOverlap
from utils.validation_metrics import distance_between_masks, mask_to_simplified_polygon, polygon_to_string


@dataclass
class BeeDetectionData:
    """Data for one bee detection in one frame"""
    video_id: str
    chamber_id: int
    frame_number: int
    bee_id: int
    aruco_code: str  # "" if not detected
    identity_segment_id: str
    identity_support_level: str
    frames_since_last_aruco: Optional[int]
    frames_until_next_aruco: Optional[int]
    nearest_aruco_gap_frames: Optional[int]
    bbox_x: float
    bbox_y: float
    bbox_width: float
    bbox_height: float
    confidence: float
    centroid_x: float
    centroid_y: float
    bee_mask_pixels: Optional[int]
    bbox_area_pixels: float
    bee_mask_area_mm2: Optional[float]
    bbox_area_mm2: Optional[float]
    pred_polygon: str
    distance_to_hive_pixels: Optional[float]
    distance_to_hive_mm: Optional[float]
    distance_to_nearest_pollen_pixels: Optional[float]
    distance_to_nearest_pollen_mm: Optional[float]
    pollen_count_in_chamber: Optional[int]
    on_pollen_ball: Optional[bool]
    pollen_overlap_pixels: Optional[int]
    pollen_overlap_fraction: Optional[float]
    on_temporal_hive: Optional[bool]
    temporal_hive_overlap_fraction: Optional[float]
    temporal_hive_mean_probability: Optional[float]
    temporal_hive_known_fraction: Optional[float]
    temporal_hive_overlap_pixels_norm: Optional[int]
    temporal_hive_known_pixels_norm: Optional[int]
    temporal_hive_bee_pixels_norm: Optional[int]
    temporal_hive_prior_weight_mean: Optional[float]
    temporal_hive_prior_weight_sum: Optional[float]
    num_bees_in_chamber: Optional[int]
    avg_distance_to_other_bees_pixels: Optional[float]
    distance_to_nearest_bee_pixels: Optional[float]
    avg_distance_to_nearest_2_bees_pixels: Optional[float]
    avg_distance_to_nearest_3_bees_pixels: Optional[float]
    avg_distance_to_other_bees_mm: Optional[float]
    distance_to_nearest_bee_mm: Optional[float]
    avg_distance_to_nearest_2_bees_mm: Optional[float]
    avg_distance_to_nearest_3_bees_mm: Optional[float]


@dataclass
class BeeInteractionData:
    """Pairwise mask-contact event for one frame."""
    video_id: str
    chamber_id: int
    frame_number: int
    bee_id_1: int
    bee_id_2: int
    aruco_code_1: str
    aruco_code_2: str
    mask_overlap_pixels: int
    mask_contact_pixels: int
    centroid_distance_pixels: float
    centroid_distance_mm: Optional[float]


@dataclass
class ArucoObservationData:
    """One physical ArUco marker observation matched to a tracked bee."""
    video_id: str
    frame_number: int
    tracker_bee_id: int
    bee_id: int
    aruco_code: str
    accepted: bool
    decision: str
    marker_confidence: Optional[float]
    marker_center_x: Optional[float]
    marker_center_y: Optional[float]
    dict_type: str


@dataclass
class BeeIdentityEventData:
    """An identity decision made from ArUco evidence."""
    video_id: str
    frame_number: int
    event_type: str
    aruco_code: str
    source_bee_id: Optional[int]
    target_bee_id: Optional[int]
    accepted: bool
    reason: str
    marker_confidence: Optional[float]


@dataclass
class BeeIdentitySegmentData:
    """Contiguous stretch of a track with the same identity-support level."""
    video_id: str
    identity_segment_id: str
    bee_id: int
    aruco_code: str
    support_level: str
    start_frame: int
    end_frame: int
    duration_frames: int
    detection_count: int
    tag_observation_count: int
    first_aruco_frame: Optional[int]
    last_aruco_frame: Optional[int]
    max_gap_frames: Optional[int]
    median_gap_frames: Optional[float]
    support_reason: str


@dataclass
class ChamberFrameData:
    """Hive pixel data for one chamber in one frame"""
    video_id: str
    chamber_id: int
    frame_number: int
    hive_pixels: Optional[int]


@dataclass
class PollenFrameData:
    """Pollen summary data for one chamber in one frame."""
    video_id: str
    chamber_id: int
    frame_number: int
    pollen_count: int
    pollen_pixels: int
    pollen_area_mm2: Optional[float]


@dataclass
class BeeTrajectory:
    """Track a bee's position across frames for velocity calculation"""
    bee_id: int
    chamber_id: int
    aruco_code: str
    positions: List[Tuple[int, float, float]] = field(default_factory=list)  # (frame, x, y)


class BatchVideoProcessor:
    """Process video frames with detection, tracking, ArUco, and spatial analysis"""
    
    def __init__(self, video_path: Path, video_id: str, bee_model, hive_model, chamber_model,
                 tracker, confidence_threshold: float, nms_iou_threshold: float,
                 pollen_model=None,
                 enable_aruco: bool = True, output_folder: Optional[Path] = None,
                 distance_method: str = 'contour', bee_model_type: str = 'bbox',
                 compute_spatial_metrics: bool = True, store_masks: bool = False,
                 log_callback=None, stop_callback=None, timing_log_interval: int = 1,
                 cleanup_interval: int = 25, verbose_output: bool = False,
                 max_frames: Optional[int] = None, high_quality_masks: bool = False,
                 store_masks_until_frame: Optional[int] = None,
                 streaming_visualization_path: Optional[Path] = None,
                 streaming_visualization_max_frames: Optional[int] = None,
                 streaming_visualization_format: str = "video",
                 streaming_visualization_mode: str = "science",
                 aruco_dicts: Optional[List[str]] = None,
                 aruco_params_bank: Optional[List[Dict]] = None,
                 allowed_tag_ids: Optional[List[int]] = None,
                 excluded_tag_ids: Optional[List[int]] = None,
                 temporal_hive_prior=None,
                 temporal_hive_context_id: str = "default",
                 video_start_time_seconds: Optional[float] = None,
                 pixel_size_mm: Optional[float] = None,
                 exclude_pollen_from_hive: bool = True,
                 prior_only: bool = False):
        """
        Args:
            video_path: Path to video file
            video_id: Unique identifier for this video
            bee_model: YOLO model for bee detection
            hive_model: Optional YOLO model for hive segmentation
            chamber_model: YOLO model for chamber segmentation
            pollen_model: Optional YOLO model for pollen segmentation
            tracker: Tracking algorithm instance
            confidence_threshold: Minimum confidence for detections
            nms_iou_threshold: NMS IoU threshold
            enable_aruco: Whether to detect ArUco markers on bees
            output_folder: Optional output folder for debug files
            distance_method: Method for calculating mask distances ('contour', 'bbox_filter', etc.)
            bee_model_type: Type of bee model ('bbox' or 'segmentation')
            compute_spatial_metrics: Whether to calculate hive/nearest-bee spatial metrics
            store_masks: Whether to store masks for visualization (uses significant memory)
            log_callback: Optional callable for progress/timing messages
            stop_callback: Optional callable returning True when processing should stop
            timing_log_interval: Log per-frame timings every N frames
            cleanup_interval: Clear Python/CUDA caches every N frames
            verbose_output: Whether to print detailed diagnostic/timing output
            max_frames: Optional maximum number of frames to process from the start
            high_quality_masks: Request full-resolution YOLO masks for prettier visualization
            store_masks_until_frame: Optional frame limit for visualization mask storage
            streaming_visualization_path: Optional MP4 path or frame directory
                for frame-by-frame annotation
            streaming_visualization_max_frames: Optional annotated-frame limit without limiting analysis
            streaming_visualization_format: Either "video" or "frames"
            streaming_visualization_mode: Visualization style passed to the annotator
            aruco_dicts: ArUco dictionaries to use for marker detection
            aruco_params_bank: Parameter sets to try for each ArUco frame
            allowed_tag_ids: Optional allowlist of ArUco tag IDs
            excluded_tag_ids: Optional blocklist of ArUco tag IDs to reject
            temporal_hive_prior: Optional chamber-aligned temporal hive prior
            temporal_hive_context_id: Stable context key (for example bumblebox ID)
            video_start_time_seconds: Parsed video start time as Unix seconds, if available
            pixel_size_mm: Optional physical calibration in millimeters per pixel
            exclude_pollen_from_hive: Remove pollen mask pixels from hive masks
                before hive counts, distances, visualization, and prior updates.
            prior_only: Only update the temporal hive prior; skip tracking, ArUco,
                spatial metrics, interactions, and CSV row storage.
        """
        self.video_path = video_path
        self.video_id = video_id
        self.bee_model = bee_model
        self.hive_model = hive_model
        self.chamber_model = chamber_model
        self.pollen_model = pollen_model
        self.tracker = tracker
        self.confidence_threshold = confidence_threshold
        self.nms_iou_threshold = nms_iou_threshold
        self.prior_only = prior_only
        self.enable_aruco = enable_aruco and not prior_only
        self.distance_method = distance_method
        self.bee_model_type = bee_model_type
        self.compute_spatial_metrics = compute_spatial_metrics
        self.store_masks = store_masks
        self.log_callback = log_callback
        self.stop_callback = stop_callback
        self.timing_log_interval = max(1, int(timing_log_interval))
        self.cleanup_interval = max(1, int(cleanup_interval))
        self.verbose_output = verbose_output
        self.max_frames = max_frames
        self.high_quality_masks = high_quality_masks
        self.store_masks_until_frame = (
            int(store_masks_until_frame)
            if store_masks_until_frame is not None and int(store_masks_until_frame) > 0
            else None
        )
        self.streaming_visualization_path = (
            Path(streaming_visualization_path)
            if streaming_visualization_path
            else None
        )
        self.streaming_visualization_max_frames = (
            int(streaming_visualization_max_frames)
            if streaming_visualization_max_frames is not None and int(streaming_visualization_max_frames) > 0
            else None
        )
        self.streaming_visualization_format = (
            "frames" if str(streaming_visualization_format).lower() == "frames" else "video"
        )
        self.streaming_visualization_mode = streaming_visualization_mode or "science"
        self.streaming_visualization_frames_written = 0
        self._streaming_visualizer = None
        self._streaming_video_writer = None
        self._streaming_video_failed = False
        self.was_stopped = False
        self.aruco_dicts = aruco_dicts
        self.aruco_params_bank = aruco_params_bank
        self.allowed_tag_ids = allowed_tag_ids
        self.excluded_tag_ids = excluded_tag_ids
        self.temporal_hive_prior = temporal_hive_prior
        self.temporal_hive_context_id = temporal_hive_context_id
        self.video_start_time_seconds = video_start_time_seconds
        self.video_fps = None
        self.pixel_size_mm = float(pixel_size_mm) if pixel_size_mm and pixel_size_mm > 0 else None
        self.exclude_pollen_from_hive = bool(exclude_pollen_from_hive)
        
        # Initialize ArUco detector for bee ID tracking
        self.marker_detector = None
        if self.enable_aruco:
            self.marker_detector = MarkerDetector(
                aruco_dicts=aruco_dicts or ['4x4_50', '4x4_100', '4x4_250', '4x4_1000'],
                aruco_params=aruco_params_bank,
                allowed_tag_ids=allowed_tag_ids,
                excluded_tag_ids=excluded_tag_ids,
                enable_aruco=True,
                enable_qr=False,  # Disable QR codes for performance
                min_confidence=0.2,
                debug=False,  # Disabled for performance (detect_aruco_in_bee_instances is much faster)
                debug_folder=None
            )
        
        # Data storage
        self.bee_detections: List[BeeDetectionData] = []
        self.bee_interactions: List[BeeInteractionData] = []
        self.aruco_observations: List[ArucoObservationData] = []
        self.bee_identity_events: List[BeeIdentityEventData] = []
        self.bee_identity_segments: List[BeeIdentitySegmentData] = []
        self.chamber_frame_data: List[ChamberFrameData] = []
        self.pollen_frame_data: List[PollenFrameData] = []
        self.bee_trajectories: Dict[int, BeeTrajectory] = {}  # bee_id -> trajectory
        self.bee_to_aruco: Dict[int, str] = {}  # bee_id -> aruco_code (retroactive)
        self.aruco_to_bee: Dict[str, int] = {}  # aruco_code -> bee_id (reverse mapping)
        self.bee_frames: Dict[int, set] = defaultdict(set)  # bee_id -> set of frame_numbers
        
        # Per-frame visualization data
        self.chambers_by_frame: Dict[int, Dict] = {}  # frame_number -> chambers_detected
        self.hive_masks_by_frame: Dict[int, Dict[int, Optional[np.ndarray]]] = {}  # frame -> chamber_id -> mask
        self.bee_masks_by_frame: Dict[int, Dict[int, Optional[np.ndarray]]] = {}  # frame -> bee_id -> mask
        self.pollen_masks_by_frame: Dict[int, Dict[int, Optional[np.ndarray]]] = {}  # frame -> pollen_id -> mask
        self.aruco_markers_by_frame: Dict[int, Dict[int, Dict]] = {}  # frame -> bee_id -> marker corners/metadata
        
        # Chamber management
        self.chamber_mapping: Dict[int, int] = {}  # aruco_id -> chamber_id (left-to-right)
        
        # Frame counter for logging
        self.frame_count = 0
        
        # Performance timing
        self.timings = defaultdict(float)  # operation -> cumulative time
        self.timing_counts = defaultdict(int)  # operation -> count

        # Identity-support scoring thresholds. These are exported as raw gaps too,
        # so downstream analyses can re-threshold without rerunning inference.
        self.identity_high_max_nearest_gap_frames = 10
        self.identity_medium_max_nearest_gap_frames = 20
        
        # Reset tracker at start of new video
        if hasattr(self.tracker, 'reset'):
            self.tracker.reset()

    def _log(self, message: str):
        """Log to the GUI worker when available, otherwise print."""
        if self.log_callback:
            self.log_callback(message)
        else:
            print(message)

    def _log_verbose(self, message: str):
        """Log detailed diagnostics only when verbose output is enabled."""
        if self.verbose_output:
            self._log(message)

    def _should_stop(self) -> bool:
        """Return whether processing has been cancelled."""
        return bool(self.stop_callback and self.stop_callback())

    def _should_store_masks_for_frame(self, frame_number: int) -> bool:
        """Return whether this frame should keep full-resolution visualization masks."""
        if self.store_masks:
            return self.store_masks_until_frame is None or frame_number <= self.store_masks_until_frame
        if self.streaming_visualization_path is not None:
            return (
                self.streaming_visualization_max_frames is None
                or frame_number <= self.streaming_visualization_max_frames
            )
        return False

    def _record_timing(self, name: str, elapsed: float, frame_timings: Dict[str, float]):
        """Record cumulative and per-frame timing for an operation."""
        self.timings[name] += elapsed
        self.timing_counts[name] += 1
        frame_timings[name] = frame_timings.get(name, 0.0) + elapsed

    def _gpu_memory_text(self) -> str:
        """Return a compact CUDA memory string for logs."""
        if TORCH_AVAILABLE and torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1e9
            reserved = torch.cuda.memory_reserved() / 1e9
            peak = torch.cuda.max_memory_allocated() / 1e9
            return f", GPU {allocated:.2f} GB allocated/{reserved:.2f} GB reserved, peak {peak:.2f} GB"
        return ""

    def _sync_cuda(self):
        """Synchronize CUDA so per-step timings are attributed accurately."""
        if TORCH_AVAILABLE and torch.cuda.is_available():
            torch.cuda.synchronize()

    def _streaming_visualization_enabled_for_frame(self, frame_number: int) -> bool:
        if self.streaming_visualization_path is None or self._streaming_video_failed:
            return False
        return (
            self.streaming_visualization_max_frames is None
            or frame_number <= self.streaming_visualization_max_frames
        )

    def _ensure_streaming_visualizer(self):
        if self._streaming_visualizer is not None:
            return self._streaming_visualizer

        from core.visualization_generator import VisualizationGenerator

        output_folder = self.streaming_visualization_path.parent
        video_id = self.streaming_visualization_path.stem
        self._streaming_visualizer = VisualizationGenerator(
            video_path=self.video_path,
            output_folder=output_folder,
            video_id=video_id,
            bee_detections=[],
            chamber_frame_data=[],
            chambers_by_frame={},
            hive_masks_by_frame={},
            bee_masks_by_frame={},
            pollen_masks_by_frame={},
            aruco_markers_by_frame={},
            log_callback=self._log,
            verbose_output=self.verbose_output,
            visualization_mode=self.streaming_visualization_mode,
        )
        return self._streaming_visualizer

    def _write_streaming_visualization_frame(
        self,
        frame: np.ndarray,
        frame_number: int,
    ):
        if not self._streaming_visualization_enabled_for_frame(frame_number):
            return

        try:
            visualizer = self._ensure_streaming_visualizer()
            frame_detections = [
                row for row in self.bee_detections
                if row.frame_number == frame_number
            ]
            frame_chamber_data = [
                row for row in self.chamber_frame_data
                if row.frame_number == frame_number
            ]
            annotated = visualizer.annotate_live_frame(
                frame=frame,
                frame_number=frame_number,
                bee_detections=frame_detections,
                chamber_frame_data=frame_chamber_data,
                chambers=self.chambers_by_frame.get(frame_number, {}),
                hive_masks=self.hive_masks_by_frame.get(frame_number, {}),
                bee_masks=self.bee_masks_by_frame.get(frame_number, {}),
                pollen_masks=self.pollen_masks_by_frame.get(frame_number, {}),
                aruco_markers=self.aruco_markers_by_frame.get(frame_number, {}),
            )

            if self.streaming_visualization_format == "frames":
                self.streaming_visualization_path.mkdir(parents=True, exist_ok=True)
                frame_path = self.streaming_visualization_path / f"frame_{frame_number:06d}.png"
                if cv2.imwrite(str(frame_path), annotated):
                    self.streaming_visualization_frames_written += 1
                else:
                    self._log(f"  ⚠️ Could not write annotated frame: {frame_path}")
                    self._streaming_video_failed = True
                return

            if self._streaming_video_writer is None:
                self.streaming_visualization_path.parent.mkdir(parents=True, exist_ok=True)
                height, width = annotated.shape[:2]
                fps = self.video_fps if self.video_fps and self.video_fps > 0 else 10.0
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                self._streaming_video_writer = cv2.VideoWriter(
                    str(self.streaming_visualization_path),
                    fourcc,
                    fps,
                    (width, height),
                )
                if not self._streaming_video_writer.isOpened():
                    self._log(f"  ⚠️ Could not create annotated video: {self.streaming_visualization_path}")
                    self._streaming_video_failed = True
                    self._streaming_video_writer = None
                    return

            self._streaming_video_writer.write(annotated)
            self.streaming_visualization_frames_written += 1
        except Exception as exc:
            self._log(f"  ⚠️ Streaming visualization failed at frame {frame_number}: {exc}")
            self._streaming_video_failed = True
        finally:
            if self.streaming_visualization_path is not None and not self.store_masks:
                self.chambers_by_frame.pop(frame_number, None)
                self.hive_masks_by_frame.pop(frame_number, None)
                self.bee_masks_by_frame.pop(frame_number, None)
                self.pollen_masks_by_frame.pop(frame_number, None)
                self.aruco_markers_by_frame.pop(frame_number, None)

    def _close_streaming_visualization(self):
        if self._streaming_video_writer is not None:
            self._streaming_video_writer.release()
            self._streaming_video_writer = None
            self._log_verbose(
                f"  Streaming visualization wrote {self.streaming_visualization_frames_written} frame(s) "
                f"to {self.streaming_visualization_path}"
            )
        elif self.streaming_visualization_path is not None and self.streaming_visualization_format == "frames":
            self._log_verbose(
                f"  Streaming visualization wrote {self.streaming_visualization_frames_written} frame image(s) "
                f"to {self.streaming_visualization_path}"
            )

    def _cleanup_memory(self):
        """Run periodic Python and CUDA memory cleanup."""
        gc.collect()
        if TORCH_AVAILABLE and torch.cuda.is_available():
            torch.cuda.empty_cache()
            try:
                torch.cuda.ipc_collect()
            except Exception:
                pass
    
    def process(self) -> bool:
        """
        Process entire video
        
        Returns:
            True if successful, False otherwise
        """
        cap = cv2.VideoCapture(str(self.video_path))
        if not cap.isOpened():
            return False

        if TORCH_AVAILABLE and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        self.video_fps = fps if fps > 0 else None
        if self.max_frames is not None and total_frames > 0:
            total_frames = min(total_frames, self.max_frames)
        frame_number = 0
        
        # For progress reporting
        last_reported_percent = -1
        
        while True:
            if self.max_frames is not None and frame_number >= self.max_frames:
                break

            if self._should_stop():
                self.was_stopped = True
                self._log(f"  Stop requested before frame {frame_number + 1}")
                break

            ret, frame = cap.read()
            if not ret:
                break
            
            frame_number += 1
            self.frame_count = frame_number
            
            # Report progress every 10%
            if total_frames > 0:
                percent_complete = int((frame_number / total_frames) * 100)
                if percent_complete % 10 == 0 and percent_complete != last_reported_percent:
                    self._log(f"  Progress: {frame_number}/{total_frames} frames ({percent_complete}%)")
                    last_reported_percent = percent_complete
            
            # Process this frame
            try:
                self._process_frame(frame, frame_number)
            except RuntimeError as e:
                if "out of memory" in str(e).lower() and TORCH_AVAILABLE and torch.cuda.is_available():
                    self._log(f"  ❌ CUDA out of memory at frame {frame_number}{self._gpu_memory_text()}")
                    self._cleanup_memory()
                raise

            if self._should_stop():
                self.was_stopped = True
                self._log(f"  Stop requested after frame {frame_number}")
                break
        
        cap.release()
        self._close_streaming_visualization()

        if self.was_stopped:
            self._finalize_processing()
            return False
        
        # Finalize data
        self._finalize_processing()
        
        return True
    
    def _process_frame(self, frame: np.ndarray, frame_number: int):
        """Process a single frame"""
        frame_start = time.perf_counter()
        frame_timings = {}

        # 1. Run chamber detection (YOLO) and establish left-to-right ordering
        t0 = time.perf_counter()
        chambers_detected = self._detect_chambers(frame)
        # Only store for visualization if requested (saves memory)
        store_frame_masks = self._should_store_masks_for_frame(frame_number)
        if store_frame_masks:
            self.chambers_by_frame[frame_number] = chambers_detected
        self._record_timing('chamber_detection', time.perf_counter() - t0, frame_timings)
        
        # 2. Run bee detection
        t0 = time.perf_counter()
        # Use inference_mode to prevent autograd graph buildup in GPU memory
        if TORCH_AVAILABLE:
            with torch.inference_mode():
                bee_results = self.bee_model(
                    frame,
                    conf=self.confidence_threshold,
                    iou=self.nms_iou_threshold,
                    retina_masks=self.high_quality_masks,
                    half=torch.cuda.is_available(),
                    verbose=False
                )
        else:
            bee_results = self.bee_model(
                frame,
                conf=self.confidence_threshold,
                iou=self.nms_iou_threshold,
                retina_masks=self.high_quality_masks,
                verbose=False
            )
        self._sync_cuda()
        self._record_timing('bee_detection', time.perf_counter() - t0, frame_timings)
        
        # 3. Run hive detection, if available
        t0 = time.perf_counter()
        hive_results = None
        if self.hive_model is not None:
            # Use inference_mode to prevent autograd graph buildup in GPU memory
            if TORCH_AVAILABLE:
                with torch.inference_mode():
                    hive_results = self.hive_model(
                        frame,
                        conf=self.confidence_threshold,
                        iou=self.nms_iou_threshold,
                        half=torch.cuda.is_available(),
                        verbose=False
                    )
            else:
                hive_results = self.hive_model(
                    frame,
                    conf=self.confidence_threshold,
                    iou=self.nms_iou_threshold,
                    verbose=False
                )
        self._sync_cuda()
        self._record_timing('hive_detection', time.perf_counter() - t0, frame_timings)

        # 4. Run pollen detection, if available. Prior-only replays only need it
        # when pollen pixels should be excluded from temporal hive evidence.
        t0 = time.perf_counter()
        pollen_balls = []
        should_detect_pollen = (
            self.pollen_model is not None
            and (not self.prior_only or self.exclude_pollen_from_hive)
        )
        if should_detect_pollen:
            pollen_balls = self._detect_pollen_balls(frame)
        self._record_timing('pollen_detection', time.perf_counter() - t0, frame_timings)
        
        # 5. Convert YOLO results to Detection objects
        t0 = time.perf_counter()
        bee_detections = self._yolo_to_detections(bee_results[0])
        # Delete YOLO result objects to free GPU memory immediately
        del bee_results
        self._record_timing('yolo_conversion', time.perf_counter() - t0, frame_timings)
        
        # 6. Apply tracking to assign IDs unless this is a temporal-prior replay.
        if not self.prior_only:
            t0 = time.perf_counter()
            bee_detections = self._apply_tracking(bee_detections, frame_number)
            self._record_timing('tracking', time.perf_counter() - t0, frame_timings)
        
        # 7. Extract hive masks per chamber
        t0 = time.perf_counter()
        hive_masks_by_chamber = self._extract_hive_masks(
            hive_results[0] if hive_results is not None else None,
            chambers_detected
        )
        # Delete YOLO result objects to free GPU memory immediately
        if hive_results is not None:
            del hive_results
        self._record_timing('hive_extraction', time.perf_counter() - t0, frame_timings)

        # 8. Assign pollen detections to chambers.
        t0 = time.perf_counter()
        pollen_by_chamber = self._assign_pollen_to_chambers(pollen_balls, chambers_detected)
        if store_frame_masks and pollen_balls:
            self.pollen_masks_by_frame[frame_number] = {
                int(pollen['pollen_id']): pollen.get('mask')
                for pollen in pollen_balls
                if pollen.get('mask') is not None
            }
        self._record_timing('pollen_assignment', time.perf_counter() - t0, frame_timings)

        if self.exclude_pollen_from_hive and pollen_by_chamber:
            t0 = time.perf_counter()
            hive_masks_by_chamber = self._exclude_pollen_from_hive_masks(
                hive_masks_by_chamber,
                pollen_by_chamber,
            )
            self._record_timing('hive_pollen_exclusion', time.perf_counter() - t0, frame_timings)

        # Only store corrected hive masks for visualization if requested.
        if store_frame_masks:
            self.hive_masks_by_frame[frame_number] = hive_masks_by_chamber
        
        frame_time_seconds = self._frame_time_seconds(frame_number)

        if not self.prior_only:
            # 9. Save chamber frame data (hive pixels per chamber)
            for chamber_id, hive_mask in hive_masks_by_chamber.items():
                hive_pixels = int(np.sum(hive_mask > 0)) if hive_mask is not None else None

                self.chamber_frame_data.append(ChamberFrameData(
                    video_id=self.video_id,
                    chamber_id=chamber_id,
                    frame_number=frame_number,
                    hive_pixels=hive_pixels
                ))

            self._record_pollen_frame_data(frame_number, pollen_by_chamber)

            # 10. Assign bees to chambers, optionally calculate spatial metrics, and write bee rows.
            self._process_bee_detections(
                bee_detections,
                frame_number,
                chambers_detected,
                hive_masks_by_chamber,
                pollen_by_chamber,
                frame.shape[:2],
                frame_time_seconds,
                frame_timings
            )

        # Update the prior after current-frame bee rows are scored, so each row
        # reflects only information available before that frame.
        if self.temporal_hive_prior is not None:
            t0 = time.perf_counter()
            self._update_temporal_hive_prior(
                chambers_detected,
                hive_masks_by_chamber,
                bee_detections,
                frame.shape[:2],
                frame_time_seconds,
            )
            self._record_timing('temporal_hive_prior', time.perf_counter() - t0, frame_timings)
        
        # 11. Detect ArUco codes on bees (retroactive tagging)
        aruco_frame_summary = None
        if not self.prior_only and self.enable_aruco and self.marker_detector is not None:
            t0 = time.perf_counter()
            aruco_frame_summary = self._detect_bee_aruco_codes(frame, bee_detections)
            self._record_timing('aruco_detection', time.perf_counter() - t0, frame_timings)

        self._write_streaming_visualization_frame(frame, frame_number)

        frame_elapsed = time.perf_counter() - frame_start
        if self.verbose_output and frame_number % self.timing_log_interval == 0:
            top_steps = sorted(frame_timings.items(), key=lambda item: item[1], reverse=True)
            timing_text = ", ".join(f"{name}={elapsed:.3f}s" for name, elapsed in top_steps)
            aruco_text = ""
            if aruco_frame_summary is not None:
                aruco_text = (
                    f", aruco_tags={aruco_frame_summary['accepted']} accepted/"
                    f"{aruco_frame_summary['matched']} matched"
                )
            self._log(
                f"  Frame {frame_number}: total={frame_elapsed:.3f}s, "
                f"detections={len(bee_detections)}{aruco_text}, "
                f"{timing_text}{self._gpu_memory_text()}"
            )

        # Drop large per-frame references as soon as all per-frame work is complete.
        del bee_detections, hive_masks_by_chamber, pollen_by_chamber, pollen_balls, chambers_detected
        
        # Report timing every 100 frames
        if self.verbose_output and frame_number % 100 == 0:
            self._print_timing_stats(frame_number)
            # Periodic garbage collection and GPU cache clearing to prevent memory buildup
            self._cleanup_memory()

        if frame_number % self.cleanup_interval == 0:
            self._cleanup_memory()
            self._log_verbose(f"  [Frame {frame_number}] Memory cleanup complete{self._gpu_memory_text()}")
        
        # More aggressive cleanup every 500 frames
        if frame_number % 500 == 0:
            self._log_verbose(f"  [Frame {frame_number}] Aggressive memory cleanup...")
            # Clear stored frames if not needed for visualization
            if not self.store_masks and frame_number > 100:
                # Keep only recent frames for trajectory calculation
                frames_to_keep = set(range(frame_number - 50, frame_number + 1))
                
                # Clear old chamber and hive mask data
                old_chamber_frames = [f for f in self.chambers_by_frame.keys() if f not in frames_to_keep]
                for f in old_chamber_frames:
                    del self.chambers_by_frame[f]
                
                old_hive_frames = [f for f in self.hive_masks_by_frame.keys() if f not in frames_to_keep]
                for f in old_hive_frames:
                    del self.hive_masks_by_frame[f]
                
                old_bee_frames = [f for f in self.bee_masks_by_frame.keys() if f not in frames_to_keep]
                for f in old_bee_frames:
                    del self.bee_masks_by_frame[f]
            
            self._cleanup_memory()
            if TORCH_AVAILABLE and torch.cuda.is_available():
                if frame_number % 1000 == 0:  # Log every 1000 frames
                    mem_allocated = torch.cuda.memory_allocated() / 1e9
                    mem_reserved = torch.cuda.memory_reserved() / 1e9
                    self._log_verbose(f"    GPU: {mem_allocated:.2f} GB allocated, {mem_reserved:.2f} GB reserved")
    
    def _detect_chambers(self, frame: np.ndarray) -> Dict[int, Dict]:
        """
        Detect chambers using YOLO chamber model and establish left-to-right ordering
        
        Returns:
            Dict mapping chamber_id -> {'mask': np.ndarray, 'bbox': [x1,y1,x2,y2], 'centroid': (x,y)}
        """
        if self.chamber_model is None:
            # No chamber model - treat entire frame as single chamber
            return {0: {
                'mask': None,
                'bbox': [0, 0, frame.shape[1], frame.shape[0]],
                'centroid': (frame.shape[1] / 2, frame.shape[0] / 2)
            }}
        
        # Run chamber detection
        # Use inference_mode to prevent autograd graph buildup in GPU memory
        if TORCH_AVAILABLE:
            with torch.inference_mode():
                chamber_results = self.chamber_model(
                    frame,
                    conf=self.confidence_threshold,
                    iou=self.nms_iou_threshold,
                    half=torch.cuda.is_available(),
                    verbose=False
                )
        else:
            chamber_results = self.chamber_model(
                frame,
                conf=self.confidence_threshold,
                iou=self.nms_iou_threshold,
                verbose=False
            )
        self._sync_cuda()
        
        if chamber_results[0].masks is None or len(chamber_results[0].masks) == 0:
            del chamber_results
            # No chambers detected - treat entire frame as single chamber
            return {0: {
                'mask': None,
                'bbox': [0, 0, frame.shape[1], frame.shape[0]],
                'centroid': (frame.shape[1] / 2, frame.shape[0] / 2)
            }}
        
        # Extract chamber masks and compute centroids
        chamber_data = []
        for idx in range(len(chamber_results[0].masks)):
            mask = chamber_results[0].masks.data[idx].detach().cpu().numpy()
            # Resize mask to frame size if needed
            if mask.shape[:2] != chamber_results[0].orig_shape[:2]:
                mask = cv2.resize(mask, (chamber_results[0].orig_shape[1], chamber_results[0].orig_shape[0]),
                                interpolation=cv2.INTER_NEAREST)
            mask = (mask > 0.5).astype(np.uint8)
            
            # Get bounding box
            bbox = chamber_results[0].boxes.xyxy[idx].detach().cpu().numpy()
            
            # Calculate centroid from mask
            coords = np.argwhere(mask > 0)
            if len(coords) > 0:
                centroid_y, centroid_x = coords.mean(axis=0)
                centroid = (float(centroid_x), float(centroid_y))
            else:
                # Fall back to bbox center
                centroid = ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)
            
            chamber_data.append({
                'mask': mask,
                'bbox': bbox.tolist(),
                'centroid': centroid
            })
        
        # Delete YOLO result objects to free GPU memory immediately
        del chamber_results
        
        # Sort chambers left-to-right by centroid X-coordinate
        chamber_data_sorted = sorted(chamber_data, key=lambda c: c['centroid'][0])
        
        # Assign chamber IDs (0, 1, 2, ...) from left to right
        chambers = {}
        for chamber_id, data in enumerate(chamber_data_sorted):
            chambers[chamber_id] = data
        
        return chambers
    
    def _assign_bee_to_chamber(self, bee_detection: Detection, chambers_detected: Dict) -> int:
        """
        Assign a bee to a chamber based on centroid location
        
        Args:
            bee_detection: Bee detection with bbox/mask
            chambers_detected: Dict of chamber_id -> chamber data (mask, bbox, centroid)
        
        Returns:
            chamber_id (int) - defaults to 0 if no chambers or no overlap
        """
        if not chambers_detected:
            return 0
        
        # Get bee centroid
        bee_centroid = self._get_centroid(bee_detection.bbox, bee_detection.mask)
        bee_x, bee_y = int(bee_centroid[0]), int(bee_centroid[1])
        
        # Check which chamber mask contains the bee's centroid
        for chamber_id, chamber_info in chambers_detected.items():
            chamber_mask = chamber_info.get('mask')
            
            # If no mask (single chamber case), assign to that chamber
            if chamber_mask is None:
                return chamber_id
            
            # Check if bee centroid falls within chamber mask
            if 0 <= bee_y < chamber_mask.shape[0] and 0 <= bee_x < chamber_mask.shape[1]:
                if chamber_mask[bee_y, bee_x] > 0:
                    return chamber_id
        
        # Fallback: assign to chamber 0 if no overlap found
        return 0
    
    def _yolo_to_detections(self, yolo_result) -> List[Detection]:
        """Convert YOLO results to Detection objects"""
        detections = []
        
        if yolo_result.boxes is None or len(yolo_result.boxes) == 0:
            return detections
        
        # Detach tensors before converting to numpy to break autograd graph
        boxes = yolo_result.boxes.xyxy.detach().cpu().numpy()
        confidences = yolo_result.boxes.conf.detach().cpu().numpy()
        
        # Check for masks
        has_masks = yolo_result.masks is not None
        
        # Log mask availability on first frame
        if self.verbose_output and self.frame_count == 1:
            if has_masks:
                self._log(f"✓ YOLO bee model returned {len(yolo_result.masks)} segmentation masks")
            else:
                self._log(f"⚠ YOLO bee model has NO segmentation masks (detection-only model)")
                if self.enable_aruco:
                    self._log(f"  Creating rectangular masks from bounding boxes for ArUco detection...")
                else:
                    self._log(f"  ArUco disabled; bbox detections will not allocate rectangular masks.")
        
        for idx in range(len(boxes)):
            bbox = boxes[idx]
            conf = float(confidences[idx])
            
            mask = None
            if has_masks:
                mask = yolo_result.masks.data[idx].detach().cpu().numpy()
                # Resize mask to frame size if needed
                if mask.shape[:2] != yolo_result.orig_shape[:2]:
                    # Debug on first frame
                    if self.verbose_output and self.frame_count == 1 and idx == 0:
                        self._log(f"[MASK RESIZE DEBUG]")
                        self._log(f"  Original mask shape: {mask.shape}")
                        self._log(f"  yolo_result.orig_shape: {yolo_result.orig_shape}")
                        self._log(f"  Resizing to: ({yolo_result.orig_shape[1]}, {yolo_result.orig_shape[0]})")
                    mask = cv2.resize(mask, (yolo_result.orig_shape[1], yolo_result.orig_shape[0]), 
                                    interpolation=cv2.INTER_NEAREST)
                    if self.verbose_output and self.frame_count == 1 and idx == 0:
                        self._log(f"  Resized mask shape: {mask.shape}")
                # Convert to binary mask (0 or 255 for marker detector compatibility)
                mask = ((mask > 0.5).astype(np.uint8)) * 255
            elif self.enable_aruco and self.bee_model_type != 'segmentation':
                # Create rectangular mask from bounding box for ArUco detection
                # This allows ArUco detection to work with detection-only models
                frame_height, frame_width = yolo_result.orig_shape[:2]
                mask = np.zeros((frame_height, frame_width), dtype=np.uint8)
                x1, y1, x2, y2 = bbox.astype(int)
                # Clip to frame boundaries
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(frame_width, x2), min(frame_height, y2)
                # Fill rectangular region with 255
                mask[y1:y2, x1:x2] = 255
            
            det = Detection(
                bbox=bbox,
                mask=mask,
                confidence=conf,
                source='yolo',
                instance_id=None
            )
            detections.append(det)

        if self.verbose_output and self.bee_model_type == 'segmentation':
            missing_masks = sum(1 for det in detections if det.mask is None)
            if missing_masks:
                self._log(
                    f"  ⚠ Frame {self.frame_count}: segmentation model returned "
                    f"{missing_masks}/{len(detections)} detections without masks"
                )
        
        return detections
    
    def _apply_tracking(self, detections: List[Detection], frame_number: int) -> List[Detection]:
        """Apply tracking algorithm to assign IDs"""
        if not detections:
            return detections
        
        # Different trackers have different APIs
        if hasattr(self.tracker, 'match_detections_to_tracks'):
            # ByteTrack-style API
            tracker_result = self.tracker.match_detections_to_tracks(detections, frame_number)
            tracked_detections = []
            for tracked_det, track_id in tracker_result:
                tracked_det.instance_id = track_id
                tracked_detections.append(tracked_det)
            return tracked_detections
        else:
            # SimpleIoU/Centroid-style API
            return self.tracker.update(detections)
    
    def _extract_hive_masks(self, hive_result, chambers_detected: Dict) -> Dict[int, Optional[np.ndarray]]:
        """
        Extract hive segmentation masks per chamber
        
        Returns:
            Dict mapping chamber_id -> mask array (or None)
        """
        hive_masks_by_chamber = {}

        if hive_result is None:
            chamber_ids = chambers_detected.keys() if chambers_detected else [0]
            return {chamber_id: None for chamber_id in chamber_ids}
        
        # If no chambers detected, assign all hive to chamber 0
        if not chambers_detected:
            if hive_result.masks is not None and len(hive_result.masks) > 0:
                # Combine all hive masks
                combined_mask = None
                for idx in range(len(hive_result.masks)):
                    mask = hive_result.masks.data[idx].detach().cpu().numpy()
                    if mask.shape[:2] != hive_result.orig_shape[:2]:
                        mask = cv2.resize(mask, (hive_result.orig_shape[1], hive_result.orig_shape[0]),
                                        interpolation=cv2.INTER_NEAREST)
                    mask = (mask > 0.5).astype(np.uint8)
                    
                    if combined_mask is None:
                        combined_mask = mask
                    else:
                        combined_mask = np.maximum(combined_mask, mask)
                
                hive_masks_by_chamber[0] = combined_mask
            else:
                hive_masks_by_chamber[0] = None
            
            return hive_masks_by_chamber
        
        # Extract all hive masks
        hive_masks = []
        if hive_result.masks is not None and len(hive_result.masks) > 0:
            for idx in range(len(hive_result.masks)):
                mask = hive_result.masks.data[idx].detach().cpu().numpy()
                if mask.shape[:2] != hive_result.orig_shape[:2]:
                    mask = cv2.resize(mask, (hive_result.orig_shape[1], hive_result.orig_shape[0]),
                                    interpolation=cv2.INTER_NEAREST)
                mask = (mask > 0.5).astype(np.uint8)
                hive_masks.append(mask)
        
        # Assign each hive mask to the chamber with maximum overlap
        if hive_masks:
            for chamber_id, chamber_info in chambers_detected.items():
                chamber_mask = chamber_info.get('mask')
                
                # Combine hive masks that overlap with this chamber
                chamber_hive_mask = None
                for hive_mask in hive_masks:
                    if chamber_mask is not None:
                        # Calculate overlap
                        overlap = np.logical_and(chamber_mask > 0, hive_mask > 0).sum()
                        if overlap > 0:
                            if chamber_hive_mask is None:
                                chamber_hive_mask = hive_mask.copy()
                            else:
                                chamber_hive_mask = np.maximum(chamber_hive_mask, hive_mask)
                    else:
                        # No chamber mask (single chamber), assign all hive
                        if chamber_hive_mask is None:
                            chamber_hive_mask = hive_mask.copy()
                        else:
                            chamber_hive_mask = np.maximum(chamber_hive_mask, hive_mask)
                
                hive_masks_by_chamber[chamber_id] = chamber_hive_mask
        else:
            # No hive masks detected
            for chamber_id in chambers_detected.keys():
                hive_masks_by_chamber[chamber_id] = None
        
        return hive_masks_by_chamber

    def _detect_pollen_balls(self, frame: np.ndarray) -> List[Dict]:
        """Run YOLO segmentation for pollen balls and return per-instance masks."""
        if self.pollen_model is None:
            return []

        if TORCH_AVAILABLE:
            with torch.inference_mode():
                pollen_results = self.pollen_model(
                    frame,
                    conf=self.confidence_threshold,
                    iou=self.nms_iou_threshold,
                    retina_masks=self.high_quality_masks,
                    half=torch.cuda.is_available(),
                    verbose=False,
                )
        else:
            pollen_results = self.pollen_model(
                frame,
                conf=self.confidence_threshold,
                iou=self.nms_iou_threshold,
                retina_masks=self.high_quality_masks,
                verbose=False,
            )
        self._sync_cuda()

        pollen_balls = []
        result = pollen_results[0] if pollen_results else None
        if result is None or result.boxes is None or result.masks is None or len(result.boxes) == 0:
            del pollen_results
            return pollen_balls

        boxes = result.boxes.xyxy.detach().cpu().numpy()
        confidences = result.boxes.conf.detach().cpu().numpy()
        classes = (
            result.boxes.cls.detach().cpu().numpy()
            if result.boxes.cls is not None
            else np.zeros(len(boxes), dtype=np.float32)
        )

        for idx in range(len(boxes)):
            cls_id = int(classes[idx])
            if cls_id != 0:
                continue

            mask = result.masks.data[idx].detach().cpu().numpy()
            if mask.shape[:2] != result.orig_shape[:2]:
                mask = cv2.resize(
                    mask,
                    (result.orig_shape[1], result.orig_shape[0]),
                    interpolation=cv2.INTER_NEAREST,
                )
            mask = ((mask > 0.5).astype(np.uint8)) * 255
            if not np.any(mask > 0):
                continue

            bbox = boxes[idx]
            centroid = self._mask_centroid(mask, bbox)
            pollen_balls.append({
                'pollen_id': idx + 1,
                'bbox': bbox,
                'confidence': float(confidences[idx]),
                'centroid': centroid,
                'mask': mask,
                'pixels': int(np.sum(mask > 0)),
            })

        del pollen_results
        return pollen_balls

    def _mask_centroid(self, mask: Optional[np.ndarray], bbox=None) -> Tuple[float, float]:
        """Return mask centroid, falling back to bbox center when needed."""
        if mask is not None and np.any(mask > 0):
            coords = np.argwhere(mask > 0)
            centroid_y, centroid_x = coords.mean(axis=0)
            return (float(centroid_x), float(centroid_y))

        if bbox is not None:
            x1, y1, x2, y2 = [float(v) for v in bbox]
            return ((x1 + x2) / 2, (y1 + y2) / 2)

        return (0.0, 0.0)

    def _assign_pollen_to_chambers(self, pollen_balls: List[Dict], chambers_detected: Dict) -> Dict[int, List[Dict]]:
        """Assign pollen instances to the chamber with the largest mask overlap."""
        chamber_ids = list(chambers_detected.keys()) if chambers_detected else [0]
        pollen_by_chamber = {chamber_id: [] for chamber_id in chamber_ids}

        if not pollen_balls:
            return pollen_by_chamber

        if not chambers_detected:
            pollen_by_chamber[0] = list(pollen_balls)
            return pollen_by_chamber

        for pollen in pollen_balls:
            mask = pollen.get('mask')
            centroid = pollen.get('centroid') or self._mask_centroid(mask, pollen.get('bbox'))
            best_chamber_id = chamber_ids[0]
            best_overlap = -1

            for chamber_id, chamber_info in chambers_detected.items():
                chamber_mask = chamber_info.get('mask')
                if chamber_mask is None:
                    best_chamber_id = chamber_id
                    best_overlap = max(best_overlap, 0)
                    continue

                overlap = 0
                if mask is not None and mask.shape[:2] == chamber_mask.shape[:2]:
                    overlap = int(np.logical_and(mask > 0, chamber_mask > 0).sum())
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_chamber_id = chamber_id

            if best_overlap <= 0:
                cx, cy = int(centroid[0]), int(centroid[1])
                for chamber_id, chamber_info in chambers_detected.items():
                    chamber_mask = chamber_info.get('mask')
                    if chamber_mask is None:
                        best_chamber_id = chamber_id
                        break
                    if 0 <= cy < chamber_mask.shape[0] and 0 <= cx < chamber_mask.shape[1]:
                        if chamber_mask[cy, cx] > 0:
                            best_chamber_id = chamber_id
                            break

            pollen_by_chamber.setdefault(best_chamber_id, []).append(pollen)

        return pollen_by_chamber

    def _record_pollen_frame_data(self, frame_number: int, pollen_by_chamber: Dict[int, List[Dict]]):
        """Store pollen count/pixels by chamber for this frame."""
        for chamber_id, pollen_balls in pollen_by_chamber.items():
            pollen_pixels = int(sum(
                int(pollen.get('pixels') or 0)
                for pollen in pollen_balls
            ))
            self.pollen_frame_data.append(PollenFrameData(
                video_id=self.video_id,
                chamber_id=chamber_id,
                frame_number=frame_number,
                pollen_count=len(pollen_balls),
                pollen_pixels=pollen_pixels,
                pollen_area_mm2=self._area_pixels_to_mm2(pollen_pixels),
            ))

    def _combined_pollen_mask(self, pollen_balls: List[Dict], target_shape: Tuple[int, int]) -> Optional[np.ndarray]:
        """Return a binary pollen mask matching target_shape, or None if empty."""
        combined_mask = None
        for pollen in pollen_balls:
            mask = pollen.get('mask')
            if mask is None or not np.any(mask > 0):
                continue
            if mask.shape[:2] != target_shape:
                continue

            binary_mask = (mask > 0).astype(np.uint8, copy=False)
            if combined_mask is None:
                combined_mask = binary_mask.copy()
            else:
                combined_mask = np.maximum(combined_mask, binary_mask)

        return combined_mask

    def _exclude_pollen_from_hive_masks(
        self,
        hive_masks_by_chamber: Dict[int, Optional[np.ndarray]],
        pollen_by_chamber: Dict[int, List[Dict]],
    ) -> Dict[int, Optional[np.ndarray]]:
        """Remove pollen pixels from hive masks chamber-by-chamber."""
        corrected = {}
        for chamber_id, hive_mask in hive_masks_by_chamber.items():
            if hive_mask is None or not np.any(hive_mask > 0):
                corrected[chamber_id] = hive_mask
                continue

            pollen_mask = self._combined_pollen_mask(
                pollen_by_chamber.get(chamber_id, []),
                hive_mask.shape[:2],
            )
            if pollen_mask is None or not np.any(pollen_mask > 0):
                corrected[chamber_id] = hive_mask
                continue

            corrected_mask = hive_mask.copy()
            corrected_mask[pollen_mask > 0] = 0
            corrected[chamber_id] = corrected_mask

        return corrected
    
    def _process_bee_detections(self, bee_detections: List[Detection], frame_number: int,
                                chambers_detected: Dict, hive_masks_by_chamber: Dict[int, Optional[np.ndarray]],
                                pollen_by_chamber: Dict[int, List[Dict]],
                                frame_shape: Tuple[int, int],
                                frame_time_seconds: Optional[float],
                                frame_timings: Optional[Dict[str, float]] = None):
        """Process bee detections: assign to chambers, optionally calculate spatial metrics, save data"""
        if frame_timings is None:
            frame_timings = {}

        # Group bees by chamber
        t_record = time.perf_counter()
        bees_by_chamber: Dict[int, List[Detection]] = defaultdict(list)
        
        for det in bee_detections:
            # Assign bee to chamber based on centroid location
            chamber_id = self._assign_bee_to_chamber(det, chambers_detected)
            bees_by_chamber[chamber_id].append(det)
        self._record_timing('bee_recording', time.perf_counter() - t_record, frame_timings)
        
        # Pre-compute hive coordinates and KDTree for each chamber (OPTIMIZATION)
        hive_kdtrees = {}
        pollen_kdtrees = {}
        pollen_masks = {}
        if self.compute_spatial_metrics:
            t_spatial = time.perf_counter()
            for chamber_id, hive_mask in hive_masks_by_chamber.items():
                if hive_mask is not None and np.sum(hive_mask) > 0:
                    # Extract hive pixel coordinates once per chamber
                    hive_coords = np.argwhere(hive_mask > 0)  # Shape: (N, 2) as (y, x)
                    if len(hive_coords) > 0:
                        # Swap to (x, y) for consistency and build KDTree
                        hive_coords_xy = hive_coords[:, [1, 0]]  # Now (x, y)
                        hive_kdtrees[chamber_id] = cKDTree(hive_coords_xy)
                    else:
                        hive_kdtrees[chamber_id] = None
                else:
                    hive_kdtrees[chamber_id] = None

            for chamber_id, pollen_balls in pollen_by_chamber.items():
                pollen_coords = []
                combined_pollen_mask = None
                for pollen in pollen_balls:
                    mask = pollen.get('mask')
                    if mask is None or not np.any(mask > 0):
                        continue
                    binary_mask = (mask > 0).astype(np.uint8, copy=False)
                    if combined_pollen_mask is None:
                        combined_pollen_mask = binary_mask.copy()
                    else:
                        combined_pollen_mask = np.maximum(combined_pollen_mask, binary_mask)
                    coords = np.argwhere(mask > 0)
                    if len(coords):
                        pollen_coords.append(coords[:, [1, 0]])
                pollen_masks[chamber_id] = combined_pollen_mask
                if pollen_coords:
                    pollen_kdtrees[chamber_id] = cKDTree(np.vstack(pollen_coords))
                else:
                    pollen_kdtrees[chamber_id] = None
            self._record_timing('spatial_metrics', time.perf_counter() - t_spatial, frame_timings)
        
        # Process each chamber
        for chamber_id, bees in bees_by_chamber.items():
            hive_kdtree = hive_kdtrees.get(chamber_id)
            pollen_kdtree = pollen_kdtrees.get(chamber_id)
            pollen_mask = pollen_masks.get(chamber_id)
            chamber_pollen_count = len(pollen_by_chamber.get(chamber_id, []))
            
            # Calculate centroids for all bees in this chamber (vectorize)
            bee_centroids = []
            for bee in bees:
                centroid = self._get_centroid(bee.bbox, bee.mask)
                bee_centroids.append(centroid)

            if self.compute_spatial_metrics:
                t_spatial = time.perf_counter()
                distance_matrix = self._calculate_bee_distance_matrix(bees, bee_centroids)
                self._record_timing('spatial_metrics', time.perf_counter() - t_spatial, frame_timings)
            else:
                distance_matrix = None

            t_contact = time.perf_counter()
            self._record_bee_mask_contacts(
                chamber_id,
                frame_number,
                bees,
                bee_centroids,
            )
            self._record_timing('bee_contacts', time.perf_counter() - t_contact, frame_timings)
            
            # Process each bee
            for bee_idx, bee in enumerate(bees):
                if bee.instance_id is None:
                    continue
                
                centroid = bee_centroids[bee_idx]
                centroid_x, centroid_y = centroid
                temporal_overlap = self._query_temporal_hive_overlap(
                    chamber_id,
                    chambers_detected,
                    bee,
                    frame_shape,
                    frame_time_seconds,
                )
                pollen_overlap = self._pollen_overlap_metrics(
                    bee,
                    pollen_mask,
                    pollen_model_available=self.pollen_model is not None,
                )
                
                if self.compute_spatial_metrics:
                    t_spatial = time.perf_counter()
                    # Calculate distance to hive (using KDTree for speed) when hive data is available.
                    distance_to_hive = (
                        self._calculate_distance_to_hive_fast(centroid, hive_kdtree)
                        if hive_kdtree is not None
                        else None
                    )
                    distance_to_pollen = (
                        self._calculate_distance_to_hive_fast(centroid, pollen_kdtree)
                        if pollen_kdtree is not None
                        else None
                    )
                    num_bees_in_chamber = len(bees)
                    avg_distance_to_bees = self._distance_matrix_avg_to_other_bees(
                        bee_idx, distance_matrix
                    )
                    nearest_distances = self._distance_matrix_nearest_n(
                        bee_idx, distance_matrix, n_values=[1, 2, 3]
                    )
                    self._record_timing('spatial_metrics', time.perf_counter() - t_spatial, frame_timings)
                else:
                    distance_to_hive = None
                    distance_to_pollen = None
                    num_bees_in_chamber = None
                    avg_distance_to_bees = None
                    nearest_distances = {1: None, 2: None, 3: None}
                
                # Get ArUco code (if detected)
                t_record = time.perf_counter()
                aruco_code = self.bee_to_aruco.get(bee.instance_id, "")
                pred_polygon = self._mask_to_polygon_string(bee.mask)
                
                # Save bee detection data
                bbox_x, bbox_y, bbox_x2, bbox_y2 = bee.bbox
                bbox_width = bbox_x2 - bbox_x
                bbox_height = bbox_y2 - bbox_y
                bbox_area_pixels = float(max(0.0, bbox_width) * max(0.0, bbox_height))
                bee_mask_pixels = self._bee_mask_pixels(bee)
                bee_mask_area_mm2 = self._area_pixels_to_mm2(bee_mask_pixels)
                bbox_area_mm2 = self._area_pixels_to_mm2(bbox_area_pixels)
                
                self.bee_detections.append(BeeDetectionData(
                    video_id=self.video_id,
                    chamber_id=chamber_id,
                    frame_number=frame_number,
                    bee_id=bee.instance_id,
                    aruco_code=aruco_code,
                    identity_segment_id="",
                    identity_support_level="low",
                    frames_since_last_aruco=None,
                    frames_until_next_aruco=None,
                    nearest_aruco_gap_frames=None,
                    bbox_x=bbox_x,
                    bbox_y=bbox_y,
                    bbox_width=bbox_width,
                    bbox_height=bbox_height,
                    confidence=bee.confidence,
                    centroid_x=centroid_x,
                    centroid_y=centroid_y,
                    bee_mask_pixels=bee_mask_pixels,
                    bbox_area_pixels=bbox_area_pixels,
                    bee_mask_area_mm2=bee_mask_area_mm2,
                    bbox_area_mm2=bbox_area_mm2,
                    pred_polygon=pred_polygon,
                    distance_to_hive_pixels=distance_to_hive,
                    distance_to_hive_mm=self._pixels_to_mm(distance_to_hive),
                    distance_to_nearest_pollen_pixels=distance_to_pollen,
                    distance_to_nearest_pollen_mm=self._pixels_to_mm(distance_to_pollen),
                    pollen_count_in_chamber=chamber_pollen_count,
                    on_pollen_ball=pollen_overlap['on_pollen_ball'],
                    pollen_overlap_pixels=pollen_overlap['pollen_overlap_pixels'],
                    pollen_overlap_fraction=pollen_overlap['pollen_overlap_fraction'],
                    on_temporal_hive=temporal_overlap.on_hive,
                    temporal_hive_overlap_fraction=temporal_overlap.overlap_fraction,
                    temporal_hive_mean_probability=temporal_overlap.mean_probability,
                    temporal_hive_known_fraction=temporal_overlap.known_fraction,
                    temporal_hive_overlap_pixels_norm=temporal_overlap.overlap_pixels,
                    temporal_hive_known_pixels_norm=temporal_overlap.known_pixels,
                    temporal_hive_bee_pixels_norm=temporal_overlap.bee_pixels,
                    temporal_hive_prior_weight_mean=temporal_overlap.mean_prior_weight,
                    temporal_hive_prior_weight_sum=temporal_overlap.sum_prior_weight,
                    num_bees_in_chamber=num_bees_in_chamber,
                    avg_distance_to_other_bees_pixels=avg_distance_to_bees,
                    distance_to_nearest_bee_pixels=nearest_distances[1],
                    avg_distance_to_nearest_2_bees_pixels=nearest_distances[2],
                    avg_distance_to_nearest_3_bees_pixels=nearest_distances[3],
                    avg_distance_to_other_bees_mm=self._pixels_to_mm(avg_distance_to_bees),
                    distance_to_nearest_bee_mm=self._pixels_to_mm(nearest_distances[1]),
                    avg_distance_to_nearest_2_bees_mm=self._pixels_to_mm(nearest_distances[2]),
                    avg_distance_to_nearest_3_bees_mm=self._pixels_to_mm(nearest_distances[3])
                ))
                self._record_timing('bee_recording', time.perf_counter() - t_record, frame_timings)
                
                # Store bee mask for visualization (only if requested to reduce memory usage)
                if self._should_store_masks_for_frame(frame_number):
                    if frame_number not in self.bee_masks_by_frame:
                        self.bee_masks_by_frame[frame_number] = {}
                    
                    # If current detection has no mask, try to preserve previous frame's mask
                    # This handles cases where YOLO inconsistently produces masks for tracked bees
                    if bee.mask is not None:
                        self.bee_masks_by_frame[frame_number][bee.instance_id] = bee.mask
                    else:
                        # Look for mask in previous frame for same bee_id
                        previous_mask = None
                        for prev_frame in range(frame_number - 1, max(0, frame_number - 10), -1):
                            if prev_frame in self.bee_masks_by_frame:
                                prev_mask = self.bee_masks_by_frame[prev_frame].get(bee.instance_id)
                                if prev_mask is not None:
                                    previous_mask = prev_mask
                                    break
                        
                        # Only store if we found a previous mask, otherwise don't store (or store None)
                        # Storing None explicitly so we don't accidentally retrieve old values
                        if previous_mask is not None:
                            self.bee_masks_by_frame[frame_number][bee.instance_id] = previous_mask
                        else:
                            # No previous mask found - store None (this will fall back to bbox in visualization)
                            self.bee_masks_by_frame[frame_number][bee.instance_id] = None

                if self.verbose_output and self._should_store_masks_for_frame(frame_number) and bee.mask is None:
                    self._log(
                        f"  ⚠ Frame {frame_number}: no segmentation mask stored for "
                        f"bee ID {bee.instance_id}; visualization will use bbox if no previous mask exists"
                    )
                
                # Update trajectory for velocity calculation
                if bee.instance_id not in self.bee_trajectories:
                    self.bee_trajectories[bee.instance_id] = BeeTrajectory(
                        bee_id=bee.instance_id,
                        chamber_id=chamber_id,
                        aruco_code=aruco_code
                    )
                
                self.bee_trajectories[bee.instance_id].positions.append(
                    (frame_number, centroid_x, centroid_y)
                )
                
                # Track which frames this bee appears in
                self.bee_frames[bee.instance_id].add(frame_number)

    def _record_bee_mask_contacts(
        self,
        chamber_id: int,
        frame_number: int,
        bees: List[Detection],
        bee_centroids: List[Tuple[float, float]],
    ):
        """Store pairwise same-chamber contact events for bees with masks."""
        if self.bee_model_type != 'segmentation':
            return

        n_bees = len(bees)
        if n_bees <= 1:
            return

        for i in range(n_bees):
            bee_a = bees[i]
            if bee_a.instance_id is None or bee_a.mask is None:
                continue

            for j in range(i + 1, n_bees):
                bee_b = bees[j]
                if bee_b.instance_id is None or bee_b.mask is None:
                    continue

                touching, overlap_pixels, contact_pixels = self._mask_contact_stats(
                    bee_a.mask,
                    bee_b.mask,
                    bee_a.bbox,
                    bee_b.bbox,
                )
                if not touching:
                    continue

                centroid_distance = float(np.sqrt(
                    (bee_centroids[i][0] - bee_centroids[j][0]) ** 2 +
                    (bee_centroids[i][1] - bee_centroids[j][1]) ** 2
                ))

                self.bee_interactions.append(BeeInteractionData(
                    video_id=self.video_id,
                    chamber_id=chamber_id,
                    frame_number=frame_number,
                    bee_id_1=bee_a.instance_id,
                    bee_id_2=bee_b.instance_id,
                    aruco_code_1=self.bee_to_aruco.get(bee_a.instance_id, ""),
                    aruco_code_2=self.bee_to_aruco.get(bee_b.instance_id, ""),
                    mask_overlap_pixels=overlap_pixels,
                    mask_contact_pixels=contact_pixels,
                    centroid_distance_pixels=centroid_distance,
                    centroid_distance_mm=self._pixels_to_mm(centroid_distance),
                ))

    def _mask_contact_stats(self, mask_a, mask_b, bbox_a, bbox_b) -> Tuple[bool, int, int]:
        """Return whether masks overlap/touch within one pixel, plus overlap/contact pixels."""
        if mask_a is None or mask_b is None:
            return False, 0, 0

        frame_h, frame_w = mask_a.shape[:2]
        ax1, ay1, ax2, ay2 = [float(v) for v in bbox_a]
        bx1, by1, bx2, by2 = [float(v) for v in bbox_b]

        x1 = max(0, int(np.floor(min(ax1, bx1))) - 1)
        y1 = max(0, int(np.floor(min(ay1, by1))) - 1)
        x2 = min(frame_w, int(np.ceil(max(ax2, bx2))) + 1)
        y2 = min(frame_h, int(np.ceil(max(ay2, by2))) + 1)

        if x2 <= x1 or y2 <= y1:
            return False, 0, 0

        crop_a = mask_a[y1:y2, x1:x2] > 0
        crop_b = mask_b[y1:y2, x1:x2] > 0
        if not np.any(crop_a) or not np.any(crop_b):
            return False, 0, 0

        overlap_pixels = int(np.logical_and(crop_a, crop_b).sum())
        dilated_a = cv2.dilate(crop_a.astype(np.uint8), np.ones((3, 3), dtype=np.uint8), iterations=1) > 0
        contact_pixels = int(np.logical_and(dilated_a, crop_b).sum())

        return contact_pixels > 0, overlap_pixels, contact_pixels

    def _bee_mask_pixels(self, bee: Detection) -> Optional[int]:
        """Return true segmentation area when the bee model is segmentation-based."""
        if self.bee_model_type != 'segmentation' or bee.mask is None:
            return None
        return int(np.sum(bee.mask > 0))

    def _pollen_overlap_metrics(
        self,
        bee: Detection,
        pollen_mask: Optional[np.ndarray],
        pollen_model_available: bool,
    ) -> Dict[str, Optional[float]]:
        """Return bee-vs-pollen mask overlap metrics for a single bee."""
        empty = {
            'on_pollen_ball': None,
            'pollen_overlap_pixels': None,
            'pollen_overlap_fraction': None,
        }
        if not pollen_model_available:
            return empty
        if bee.mask is None:
            return empty

        bee_pixels = bee.mask > 0
        bee_pixel_count = int(np.sum(bee_pixels))
        if bee_pixel_count == 0:
            return empty

        if pollen_mask is None or not np.any(pollen_mask > 0):
            return {
                'on_pollen_ball': False,
                'pollen_overlap_pixels': 0,
                'pollen_overlap_fraction': 0.0,
            }

        if pollen_mask.shape[:2] != bee.mask.shape[:2]:
            return empty

        overlap_pixels = int(np.logical_and(bee_pixels, pollen_mask > 0).sum())
        return {
            'on_pollen_ball': overlap_pixels > 0,
            'pollen_overlap_pixels': overlap_pixels,
            'pollen_overlap_fraction': float(overlap_pixels / bee_pixel_count),
        }

    def _pixels_to_mm(self, value) -> Optional[float]:
        if value is None or self.pixel_size_mm is None:
            return None
        return float(value) * self.pixel_size_mm

    def _area_pixels_to_mm2(self, value) -> Optional[float]:
        if value is None or self.pixel_size_mm is None:
            return None
        return float(value) * self.pixel_size_mm * self.pixel_size_mm

    def _frame_time_seconds(self, frame_number: int) -> Optional[float]:
        """Return absolute-ish frame time when the video timestamp was parsed."""
        if self.video_start_time_seconds is None or self.video_fps is None:
            return None
        return self.video_start_time_seconds + max(0, frame_number - 1) / self.video_fps

    def _query_temporal_hive_overlap(
        self,
        chamber_id: int,
        chambers_detected: Dict,
        bee: Detection,
        frame_shape: Tuple[int, int],
        frame_time_seconds: Optional[float],
    ) -> TemporalHiveOverlap:
        if self.temporal_hive_prior is None:
            return TemporalHiveOverlap(None, None, None, None)

        chamber_info = chambers_detected.get(chamber_id, {
            'mask': None,
            'bbox': [0, 0, frame_shape[1], frame_shape[0]],
            'centroid': (frame_shape[1] / 2, frame_shape[0] / 2)
        })
        return self.temporal_hive_prior.query_bee_overlap(
            context_id=self.temporal_hive_context_id,
            chamber_id=chamber_id,
            chamber_info=chamber_info,
            bee_mask=bee.mask,
            bee_bbox=bee.bbox,
            frame_shape=frame_shape,
            observation_time_seconds=frame_time_seconds,
        )

    def _update_temporal_hive_prior(
        self,
        chambers_detected: Dict,
        hive_masks_by_chamber: Dict[int, Optional[np.ndarray]],
        bee_detections: List[Detection],
        frame_shape: Tuple[int, int],
        frame_time_seconds: Optional[float],
    ):
        if self.temporal_hive_prior is None:
            return

        for chamber_id, hive_mask in hive_masks_by_chamber.items():
            chamber_info = chambers_detected.get(chamber_id, {
                'mask': None,
                'bbox': [0, 0, frame_shape[1], frame_shape[0]],
                'centroid': (frame_shape[1] / 2, frame_shape[0] / 2)
            })
            self.temporal_hive_prior.update(
                context_id=self.temporal_hive_context_id,
                chamber_id=chamber_id,
                chamber_info=chamber_info,
                hive_mask=hive_mask,
                bee_detections=bee_detections,
                frame_shape=frame_shape,
                observation_time_seconds=frame_time_seconds,
            )

    def _marker_confidence(self, marker_result) -> Optional[float]:
        if marker_result is None:
            return None
        try:
            return float(marker_result.confidence)
        except (TypeError, ValueError, AttributeError):
            return None

    def _marker_center(self, marker_result) -> Tuple[Optional[float], Optional[float]]:
        if marker_result is None:
            return (None, None)

        center = getattr(marker_result, 'center', None)
        if center is not None:
            try:
                return (float(center[0]), float(center[1]))
            except (TypeError, ValueError, IndexError):
                pass

        corners = getattr(marker_result, 'corners', None)
        if corners is None:
            return (None, None)

        corners = np.asarray(corners, dtype=np.float32).reshape(-1, 2)
        if corners.size == 0:
            return (None, None)
        center = corners.mean(axis=0)
        return (float(center[0]), float(center[1]))

    def _record_aruco_observation(
        self,
        tracker_bee_id: int,
        resolved_bee_id: int,
        aruco_code: str,
        marker_result,
        accepted: bool,
        decision: str,
    ):
        """Store one physical tag observation and how it was used."""
        center_x, center_y = self._marker_center(marker_result)
        self.aruco_observations.append(ArucoObservationData(
            video_id=self.video_id,
            frame_number=self.frame_count,
            tracker_bee_id=int(tracker_bee_id),
            bee_id=int(resolved_bee_id),
            aruco_code=str(aruco_code),
            accepted=bool(accepted),
            decision=str(decision),
            marker_confidence=self._marker_confidence(marker_result),
            marker_center_x=center_x,
            marker_center_y=center_y,
            dict_type=getattr(marker_result, 'dict_type', '') or '',
        ))

    def _record_identity_event(
        self,
        event_type: str,
        aruco_code: str,
        source_bee_id: Optional[int],
        target_bee_id: Optional[int],
        accepted: bool,
        reason: str,
        marker_result=None,
    ):
        """Store an auditable identity decision."""
        self.bee_identity_events.append(BeeIdentityEventData(
            video_id=self.video_id,
            frame_number=self.frame_count,
            event_type=str(event_type),
            aruco_code=str(aruco_code),
            source_bee_id=source_bee_id,
            target_bee_id=target_bee_id,
            accepted=bool(accepted),
            reason=str(reason),
            marker_confidence=self._marker_confidence(marker_result),
        ))
    
    def _detect_bee_aruco_codes(self, frame: np.ndarray, bee_detections: List[Detection]) -> Dict[str, int]:
        """
        Detect ArUco codes on individual bees using efficient full-frame detection.
        
        Strategy:
        1. Detect all ArUco codes in the full frame once
        2. Match them to bee detections based on spatial overlap
        3. Validate assignments with strict rules:
           - If ArUco in multiple boxes in this frame: reject
           - If ArUco already assigned to another bee active in this frame: reject
           - If ArUco not yet assigned and only in one bee's box: assign
           - If ArUco was assigned before but that bee is inactive: re-identify (merge tracks)
        """
        # Diagnostic logging on first frame
        if self.verbose_output and self.frame_count == 1:
            self._log(f"\n=== Frame 1 ArUco Diagnostic ===")
            self._log(f"Total detections: {len(bee_detections)}")
            
            dets_with_id = sum(1 for det in bee_detections if det.instance_id is not None)
            dets_with_mask = sum(1 for det in bee_detections if det.mask is not None)
            dets_with_both = sum(1 for det in bee_detections if det.instance_id is not None and det.mask is not None)
            
            self._log(f"  Detections with instance_id: {dets_with_id}")
            self._log(f"  Detections with mask: {dets_with_mask}")
            self._log(f"  Detections with BOTH (valid for ArUco): {dets_with_both}")
            
            if dets_with_mask > 0:
                sample = next((det for det in bee_detections if det.mask is not None), None)
                if sample:
                    self._log(f"  Sample mask: shape={sample.mask.shape}, dtype={sample.mask.dtype}, range=[{sample.mask.min()}, {sample.mask.max()}]")
                    self._log(f"  Sample has instance_id: {sample.instance_id is not None} (id={sample.instance_id})")
            self._log("=" * 35 + "\n")
        
        # Get active bee IDs in current frame
        frame_summary = {
            'matched': 0,
            'accepted': 0,
            'rejected': 0,
            'unique_ids': 0,
        }
        active_bee_ids = set(det.instance_id for det in bee_detections if det.instance_id is not None)
        
        # Convert Detection objects to annotation format for marker detector
        annotations = []
        for det in bee_detections:
            if det.instance_id is None or det.mask is None:
                continue
            
            # Convert bbox to [x, y, w, h] format
            x1, y1, x2, y2 = det.bbox
            bbox = [x1, y1, x2 - x1, y2 - y1]
            
            # Convert mask from 0/255 to 0/1 format if needed
            mask = det.mask
            if mask.max() > 1:
                mask = (mask > 127).astype(np.uint8)
            
            annotations.append({
                'instance_id': det.instance_id,
                'mask_id': det.instance_id,
                'category': 'bee',
                'bbox': bbox,
                'mask': mask
            })
        
        if not annotations:
            return frame_summary
        
        # Detect all ArUco codes in frame at once and match to bees
        detections = self.marker_detector.detect_aruco_in_bee_instances(
            image=frame,
            annotations=annotations,
            reject_multiple=True  # Reject bees with multiple ArUco codes
        )
        frame_summary['matched'] = len(detections)
        
        # Build reverse mapping: aruco_code -> list of instance_ids in this frame
        aruco_to_instances = {}
        for instance_id, marker_result in detections.items():
            if marker_result.marker_type == 'aruco':
                aruco_code = str(int(marker_result.marker_id))
            else:
                aruco_code = str(marker_result.marker_id)
            
            if aruco_code not in aruco_to_instances:
                aruco_to_instances[aruco_code] = []
            aruco_to_instances[aruco_code].append(instance_id)
        frame_summary['unique_ids'] = len(aruco_to_instances)
        
        # Rule 1: Reject ArUco codes matched to multiple bees in this frame
        ambiguous_codes = set()
        for aruco_code, instance_list in aruco_to_instances.items():
            if len(instance_list) > 1:
                ambiguous_codes.add(aruco_code)
                if self.verbose_output and self.frame_count <= 10:
                    self._log(f"  ⚠ Frame {self.frame_count}: ArUco {aruco_code} in {len(instance_list)} boxes (rejected): {instance_list}")
        
        # Process each detection
        for instance_id, marker_result in detections.items():
            if marker_result.marker_type == 'aruco':
                aruco_code = str(int(marker_result.marker_id))
            else:
                aruco_code = str(marker_result.marker_id)
            
            # Rule 1: Skip if this code is ambiguous in this frame (multiple boxes)
            if aruco_code in ambiguous_codes:
                instance_list = aruco_to_instances.get(aruco_code, [])
                reason = f"tag matched multiple bee boxes in this frame: {instance_list}"
                self._record_aruco_observation(
                    tracker_bee_id=instance_id,
                    resolved_bee_id=instance_id,
                    aruco_code=aruco_code,
                    marker_result=marker_result,
                    accepted=False,
                    decision='ambiguous_tag_rejected',
                )
                self._record_identity_event(
                    event_type='ambiguous_tag_rejected',
                    aruco_code=aruco_code,
                    source_bee_id=instance_id,
                    target_bee_id=None,
                    accepted=False,
                    reason=reason,
                    marker_result=marker_result,
                )
                frame_summary['rejected'] += 1
                continue
            
            # Rule 2: Check if ArUco already assigned to a different bee
            if aruco_code in self.aruco_to_bee:
                assigned_bee_id = self.aruco_to_bee[aruco_code]
                
                # Rule 2a: If the assigned bee is active in this frame, reject new assignment
                if assigned_bee_id in active_bee_ids:
                    if assigned_bee_id != instance_id:
                        # Different bee has this code and is active - conflict!
                        if self.verbose_output and self.frame_count <= 10:
                            self._log(f"  ⚠ Frame {self.frame_count}: ArUco {aruco_code} already on active bee {assigned_bee_id}, rejecting for bee {instance_id}")
                        self._record_aruco_observation(
                            tracker_bee_id=instance_id,
                            resolved_bee_id=instance_id,
                            aruco_code=aruco_code,
                            marker_result=marker_result,
                            accepted=False,
                            decision='tag_conflict_rejected',
                        )
                        self._record_identity_event(
                            event_type='tag_conflict_rejected',
                            aruco_code=aruco_code,
                            source_bee_id=instance_id,
                            target_bee_id=assigned_bee_id,
                            accepted=False,
                            reason=f"tag already assigned to active bee {assigned_bee_id}",
                            marker_result=marker_result,
                        )
                        frame_summary['rejected'] += 1
                        continue
                    # else: same bee, no problem
                else:
                    # Rule 2b: Assigned bee is NOT active - possible re-identification
                    # Check if these two bees ever appeared together
                    if self._bees_coexisted(assigned_bee_id, instance_id):
                        # They appeared together before - they're different bees
                        # This is a conflict - reject this assignment
                        if self.verbose_output and self.frame_count <= 10:
                            self._log(f"  ⚠ Frame {self.frame_count}: ArUco {aruco_code} conflict - bees {assigned_bee_id} and {instance_id} coexisted, rejecting")
                        self._record_aruco_observation(
                            tracker_bee_id=instance_id,
                            resolved_bee_id=instance_id,
                            aruco_code=aruco_code,
                            marker_result=marker_result,
                            accepted=False,
                            decision='tag_conflict_rejected',
                        )
                        self._record_identity_event(
                            event_type='tag_conflict_rejected',
                            aruco_code=aruco_code,
                            source_bee_id=instance_id,
                            target_bee_id=assigned_bee_id,
                            accepted=False,
                            reason=f"candidate bee coexisted with assigned bee {assigned_bee_id}",
                            marker_result=marker_result,
                        )
                        frame_summary['rejected'] += 1
                        continue
                    else:
                        # They never coexisted - this is re-identification
                        # Merge instance_id into assigned_bee_id
                        if self.verbose_output and self.frame_count <= 10:
                            self._log(f"  🔄 Frame {self.frame_count}: Re-identification - merging bee {instance_id} into bee {assigned_bee_id} (ArUco {aruco_code})")
                        self._record_aruco_observation(
                            tracker_bee_id=instance_id,
                            resolved_bee_id=assigned_bee_id,
                            aruco_code=aruco_code,
                            marker_result=marker_result,
                            accepted=True,
                            decision='track_reidentified',
                        )
                        self._record_identity_event(
                            event_type='track_reidentified',
                            aruco_code=aruco_code,
                            source_bee_id=instance_id,
                            target_bee_id=assigned_bee_id,
                            accepted=True,
                            reason=f"tag matched inactive assigned bee {assigned_bee_id}; tracks did not coexist",
                            marker_result=marker_result,
                        )
                        self._merge_bee_tracks(source_id=instance_id, target_id=assigned_bee_id, aruco_code=aruco_code)
                        self._store_aruco_marker_for_visualization(assigned_bee_id, marker_result, aruco_code)
                        frame_summary['accepted'] += 1
                        continue
            
            # Rule 3: New assignment - ArUco not yet assigned
            if instance_id in self.bee_to_aruco:
                # Bee already has a different ArUco code
                if self.bee_to_aruco[instance_id] != aruco_code:
                    if self.verbose_output and self.frame_count <= 10:
                        self._log(f"  ⚠ Frame {self.frame_count}: Bee {instance_id} already has ArUco {self.bee_to_aruco[instance_id]}, rejecting new code {aruco_code}")
                    self._record_aruco_observation(
                        tracker_bee_id=instance_id,
                        resolved_bee_id=instance_id,
                        aruco_code=aruco_code,
                        marker_result=marker_result,
                        accepted=False,
                        decision='tag_conflict_rejected',
                    )
                    self._record_identity_event(
                        event_type='tag_conflict_rejected',
                        aruco_code=aruco_code,
                        source_bee_id=instance_id,
                        target_bee_id=instance_id,
                        accepted=False,
                        reason=f"bee already assigned to ArUco {self.bee_to_aruco[instance_id]}",
                        marker_result=marker_result,
                    )
                    frame_summary['rejected'] += 1
                    continue
                event_type = 'tag_confirmed'
                reason = "same tag observed again on assigned bee"
            else:
                # New assignment
                self.bee_to_aruco[instance_id] = aruco_code
                self.aruco_to_bee[aruco_code] = instance_id
                event_type = 'tag_assigned'
                reason = "first accepted tag observation for this bee"
                if self.verbose_output and self.frame_count <= 10:
                    self._log(f"  ✓ Frame {self.frame_count}: Assigned ArUco {aruco_code} to bee {instance_id}")

            self._record_aruco_observation(
                tracker_bee_id=instance_id,
                resolved_bee_id=instance_id,
                aruco_code=aruco_code,
                marker_result=marker_result,
                accepted=True,
                decision=event_type,
            )
            self._record_identity_event(
                event_type=event_type,
                aruco_code=aruco_code,
                source_bee_id=instance_id,
                target_bee_id=instance_id,
                accepted=True,
                reason=reason,
                marker_result=marker_result,
            )
            self._store_aruco_marker_for_visualization(instance_id, marker_result, aruco_code)
            frame_summary['accepted'] += 1

        return frame_summary

    def _store_aruco_marker_for_visualization(self, bee_id: int, marker_result, aruco_code: str):
        """Keep the actual detected marker corners for annotated frame exports."""
        if not self._should_store_masks_for_frame(self.frame_count):
            return

        if marker_result is None or marker_result.corners is None:
            return

        corners = np.asarray(marker_result.corners, dtype=np.float32).reshape(-1, 2)
        if corners.shape[0] < 4:
            return

        frame_markers = self.aruco_markers_by_frame.setdefault(self.frame_count, {})
        center = self._marker_center(marker_result)
        frame_markers[int(bee_id)] = {
            'aruco_code': str(aruco_code),
            'corners': corners.tolist(),
            'center': center,
            'dict_type': marker_result.dict_type or '',
            'confidence': float(marker_result.confidence),
        }
    
    def _bees_coexisted(self, bee_id_a: int, bee_id_b: int) -> bool:
        """Check if two bees ever appeared in the same frame"""
        frames_a = self.bee_frames.get(bee_id_a, set())
        frames_b = self.bee_frames.get(bee_id_b, set())
        return len(frames_a & frames_b) > 0
    
    def _merge_bee_tracks(self, source_id: int, target_id: int, aruco_code: str):
        """
        Merge source bee track into target bee track (re-identification).
        Updates all past and future detections of source_id to target_id.
        """
        # Update all detections
        for detection in self.bee_detections:
            if detection.bee_id == source_id:
                detection.bee_id = target_id
                detection.aruco_code = aruco_code

        for interaction in self.bee_interactions:
            if interaction.bee_id_1 == source_id:
                interaction.bee_id_1 = target_id
                interaction.aruco_code_1 = aruco_code
            if interaction.bee_id_2 == source_id:
                interaction.bee_id_2 = target_id
                interaction.aruco_code_2 = aruco_code

        for observation in self.aruco_observations:
            if observation.bee_id == source_id:
                observation.bee_id = target_id
        
        # Merge trajectory data
        if source_id in self.bee_trajectories:
            source_traj = self.bee_trajectories[source_id]
            
            if target_id in self.bee_trajectories:
                # Merge positions
                self.bee_trajectories[target_id].positions.extend(source_traj.positions)
                # Sort by frame number
                self.bee_trajectories[target_id].positions.sort(key=lambda x: x[0])
            else:
                # Move entire trajectory
                self.bee_trajectories[target_id] = source_traj
                self.bee_trajectories[target_id].bee_id = target_id
            
            # Update ArUco code
            self.bee_trajectories[target_id].aruco_code = aruco_code
            
            # Remove source trajectory
            del self.bee_trajectories[source_id]
        
        # Merge frame appearances
        if source_id in self.bee_frames:
            self.bee_frames[target_id].update(self.bee_frames[source_id])
            del self.bee_frames[source_id]

        # Move stored visualization masks to the merged ID as well. Without this,
        # retroactively renamed detections can point at target_id while the mask is
        # still keyed under source_id, causing the visualizer to fall back to bbox.
        for frame_masks in self.bee_masks_by_frame.values():
            if source_id not in frame_masks:
                continue

            source_mask = frame_masks.pop(source_id)
            if target_id not in frame_masks or frame_masks[target_id] is None:
                frame_masks[target_id] = source_mask

        for frame_markers in self.aruco_markers_by_frame.values():
            if source_id not in frame_markers:
                continue

            source_marker = frame_markers.pop(source_id)
            if target_id not in frame_markers:
                frame_markers[target_id] = source_marker
        
        # Update bee_to_aruco mapping
        if source_id in self.bee_to_aruco:
            del self.bee_to_aruco[source_id]
        self.bee_to_aruco[target_id] = aruco_code
        
        self._log_verbose(f"    Merged bee {source_id} → {target_id} (ArUco {aruco_code})")
    
    def _get_centroid(self, bbox, mask) -> Tuple[float, float]:
        """Get centroid from bbox or mask"""
        if mask is not None:
            coords = np.argwhere(mask > 0)
            if len(coords) > 0:
                y, x = coords.mean(axis=0)
                return (float(x), float(y))
        
        # Fall back to bbox centroid
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)

    def _mask_to_polygon_string(self, mask: Optional[np.ndarray]) -> str:
        """Convert a bee mask to a simplified polygon string for CSV export."""
        if mask is None:
            return ""

        polygon = mask_to_simplified_polygon((mask > 0).astype(np.uint8), epsilon_percent=2.0)
        return polygon_to_string(polygon)

    def _calculate_centroid_distance_matrix(self, centroids: List[Tuple[float, float]]) -> np.ndarray:
        """Calculate all pairwise centroid distances for one chamber."""
        if not centroids:
            return np.zeros((0, 0), dtype=np.float32)

        coords = np.asarray(centroids, dtype=np.float32)
        deltas = coords[:, None, :] - coords[None, :, :]
        return np.sqrt(np.sum(deltas * deltas, axis=2, dtype=np.float32))

    def _calculate_bee_distance_matrix(self, bees: List[Detection],
                                       centroids: List[Tuple[float, float]]) -> np.ndarray:
        """Calculate all pairwise bee distances using the selected spatial metric."""
        if self.distance_method == 'centroid':
            return self._calculate_centroid_distance_matrix(centroids)

        n_bees = len(bees)
        distance_matrix = np.zeros((n_bees, n_bees), dtype=np.float32)
        if n_bees <= 1:
            return distance_matrix

        centroid_matrix = self._calculate_centroid_distance_matrix(centroids)
        for i in range(n_bees):
            current_bee = bees[i]
            for j in range(i + 1, n_bees):
                other_bee = bees[j]
                if current_bee.mask is not None and other_bee.mask is not None:
                    dist = distance_between_masks(
                        current_bee.mask,
                        other_bee.mask,
                        method=self.distance_method
                    )
                else:
                    dist = centroid_matrix[i, j]

                distance_matrix[i, j] = dist
                distance_matrix[j, i] = dist

        return distance_matrix

    def _distance_matrix_avg_to_other_bees(self, bee_idx: int, distance_matrix: np.ndarray) -> float:
        """Average distance from one bee to every other bee in its chamber."""
        n_bees = distance_matrix.shape[0]
        if n_bees <= 1:
            return 0.0

        distances = np.delete(distance_matrix[bee_idx], bee_idx)
        return float(np.mean(distances)) if len(distances) else 0.0

    def _distance_matrix_nearest_n(self, bee_idx: int, distance_matrix: np.ndarray,
                                   n_values: List[int]) -> Dict[int, float]:
        """Average distance from one bee to its nearest N neighbors."""
        result = {n: 0.0 for n in n_values}
        n_bees = distance_matrix.shape[0]
        if n_bees <= 1:
            return result

        distances = np.delete(distance_matrix[bee_idx], bee_idx)
        if len(distances) == 0:
            return result

        sorted_distances = np.sort(distances)
        for n in n_values:
            count = min(n, len(sorted_distances))
            result[n] = float(np.mean(sorted_distances[:count])) if count > 0 else 0.0

        return result
    
    def _calculate_distance_to_hive_fast(self, bee_centroid: Tuple[float, float], 
                                         hive_kdtree: Optional[cKDTree]) -> float:
        """Calculate Euclidean distance from bee to nearest hive pixel using KDTree (FAST)"""
        if hive_kdtree is None:
            return 0.0
        
        # Query KDTree for nearest neighbor (O(log N) instead of O(N))
        distance, _ = hive_kdtree.query(bee_centroid)
        return float(distance)
    
    def _calculate_distance_to_hive(self, bee_centroid: Tuple[float, float], 
                                    hive_mask: Optional[np.ndarray]) -> float:
        """Calculate Euclidean distance from bee to nearest hive pixel (SLOW - kept for reference)"""
        if hive_mask is None or np.sum(hive_mask) == 0:
            return 0.0
        
        # Find nearest hive pixel
        hive_coords = np.argwhere(hive_mask > 0)  # Shape: (N, 2) as (y, x)
        
        if len(hive_coords) == 0:
            return 0.0
        
        bee_x, bee_y = bee_centroid
        
        # Calculate distances to all hive pixels
        distances = np.sqrt(
            (hive_coords[:, 1] - bee_x) ** 2 + 
            (hive_coords[:, 0] - bee_y) ** 2
        )
        
        return float(np.min(distances))
    
    def _calculate_avg_distance_to_other_bees(self, bee_idx: int, 
                                              bees: List[Detection]) -> float:
        """Calculate average distance to all other bees in the same chamber
        
        Uses mask-based distance when both bees have masks (segmentation model),
        otherwise falls back to centroid distance.
        """
        if len(bees) <= 1:
            return 0.0
        
        current_bee = bees[bee_idx]
        current_has_mask = current_bee.mask is not None
        
        distances = []
        for other_idx, other_bee in enumerate(bees):
            if other_idx == bee_idx:
                continue  # Skip self
            
            # Use mask-based distance if both have masks, otherwise centroid distance
            if current_has_mask and other_bee.mask is not None:
                dist = distance_between_masks(current_bee.mask, other_bee.mask, method=self.distance_method)
            else:
                # Fallback to centroid distance
                current_centroid = self._get_centroid(current_bee.bbox, current_bee.mask)
                other_centroid = self._get_centroid(other_bee.bbox, other_bee.mask)
                dist = np.sqrt(
                    (current_centroid[0] - other_centroid[0])**2 + 
                    (current_centroid[1] - other_centroid[1])**2
                )
            
            distances.append(dist)
        
        return float(np.mean(distances)) if distances else 0.0
    
    def _calculate_nearest_n_bee_distances(self, bee_idx: int, 
                                           bees: List[Detection],
                                           n_values: List[int]) -> Dict[int, float]:
        """Calculate average distance to nearest N bees
        
        Uses mask-based distance when both bees have masks (segmentation model),
        otherwise falls back to centroid distance.
        
        Args:
            bee_idx: Index of current bee
            bees: List of all bee Detection objects in chamber
            n_values: List of N values to calculate (e.g., [1, 2, 3])
            
        Returns:
            Dict mapping N -> average distance to nearest N bees
        """
        result = {n: 0.0 for n in n_values}
        
        if len(bees) <= 1:
            return result
        
        current_bee = bees[bee_idx]
        current_has_mask = current_bee.mask is not None
        
        # Calculate distances to all other bees
        distances = []
        for other_idx, other_bee in enumerate(bees):
            if other_idx == bee_idx:
                continue  # Skip self
            
            # Use mask-based distance if both have masks, otherwise centroid distance
            if current_has_mask and other_bee.mask is not None:
                dist = distance_between_masks(current_bee.mask, other_bee.mask, method=self.distance_method)
            else:
                # Fallback to centroid distance
                current_centroid = self._get_centroid(current_bee.bbox, current_bee.mask)
                other_centroid = self._get_centroid(other_bee.bbox, other_bee.mask)
                dist = np.sqrt(
                    (current_centroid[0] - other_centroid[0])**2 + 
                    (current_centroid[1] - other_centroid[1])**2
                )
            
            distances.append(dist)
        
        if len(distances) == 0:
            return result
        
        # Sort distances to get nearest bees
        sorted_distances = np.sort(distances)
        
        # Calculate for each N value
        for n in n_values:
            if n <= len(sorted_distances):
                # Average of nearest N bees
                result[n] = float(np.mean(sorted_distances[:n]))
            else:
                # Not enough bees - use all available
                result[n] = float(np.mean(sorted_distances)) if len(sorted_distances) > 0 else 0.0
        
        return result

    def _finalize_identity_support(self):
        """Assign per-row identity support and summarize contiguous support segments."""
        self.bee_identity_segments.clear()
        if not self.bee_detections:
            return

        accepted_observations_by_bee = defaultdict(list)
        for observation in self.aruco_observations:
            if not observation.accepted or not observation.aruco_code:
                continue
            accepted_observations_by_bee[(observation.bee_id, observation.aruco_code)].append(
                observation.frame_number
            )

        for key in list(accepted_observations_by_bee.keys()):
            accepted_observations_by_bee[key] = sorted(set(accepted_observations_by_bee[key]))

        detections_by_bee = defaultdict(list)
        for detection in self.bee_detections:
            detections_by_bee[detection.bee_id].append(detection)

        segment_counter = 0
        for bee_id, detections in sorted(detections_by_bee.items()):
            detections.sort(key=lambda det: det.frame_number)
            if not detections:
                continue

            aruco_code = detections[0].aruco_code or self.bee_to_aruco.get(bee_id, "")
            tag_frames = accepted_observations_by_bee.get((bee_id, aruco_code), [])

            row_support = []
            for detection in detections:
                if aruco_code and not detection.aruco_code:
                    detection.aruco_code = aruco_code

                since, until, nearest = self._nearest_aruco_gaps(detection.frame_number, tag_frames)
                support_level, reason = self._identity_support_for_gaps(
                    aruco_code,
                    since,
                    until,
                    nearest,
                )

                detection.frames_since_last_aruco = since
                detection.frames_until_next_aruco = until
                detection.nearest_aruco_gap_frames = nearest
                detection.identity_support_level = support_level
                row_support.append((detection, support_level, reason))

            segment_start = 0
            for idx in range(1, len(row_support) + 1):
                boundary = idx == len(row_support)
                if not boundary:
                    prev_det, prev_level, _ = row_support[idx - 1]
                    current_det, current_level, _ = row_support[idx]
                    boundary = (
                        current_level != prev_level
                        or current_det.frame_number - prev_det.frame_number > 1
                    )

                if boundary:
                    segment_rows = row_support[segment_start:idx]
                    segment_counter += 1
                    segment = self._build_identity_segment(
                        segment_rows,
                        segment_counter,
                        bee_id,
                        aruco_code,
                        tag_frames,
                    )
                    self.bee_identity_segments.append(segment)
                    for detection, _, _ in segment_rows:
                        detection.identity_segment_id = segment.identity_segment_id
                    segment_start = idx

    def _nearest_aruco_gaps(
        self,
        frame_number: int,
        tag_frames: List[int],
    ) -> Tuple[Optional[int], Optional[int], Optional[int]]:
        if not tag_frames:
            return (None, None, None)

        previous_index = bisect_right(tag_frames, frame_number) - 1
        next_index = bisect_left(tag_frames, frame_number)

        since = frame_number - tag_frames[previous_index] if previous_index >= 0 else None
        until = tag_frames[next_index] - frame_number if next_index < len(tag_frames) else None

        gaps = [gap for gap in (since, until) if gap is not None]
        nearest = min(gaps) if gaps else None
        return (since, until, nearest)

    def _identity_support_for_gaps(
        self,
        aruco_code: str,
        since: Optional[int],
        until: Optional[int],
        nearest: Optional[int],
    ) -> Tuple[str, str]:
        if not aruco_code:
            return ("low", "no accepted ArUco identity assigned")
        if nearest is None:
            return ("low", "assigned identity has no accepted tag observation")
        if nearest <= self.identity_high_max_nearest_gap_frames:
            return ("high", "within 10 frames of an accepted same-tag observation")
        if nearest <= self.identity_medium_max_nearest_gap_frames:
            return ("medium", "within 20 frames of an accepted same-tag observation")
        return ("low", "far from accepted same-tag observations")

    def _build_identity_segment(
        self,
        segment_rows,
        segment_counter: int,
        bee_id: int,
        aruco_code: str,
        all_tag_frames: List[int],
    ) -> BeeIdentitySegmentData:
        first_detection = segment_rows[0][0]
        last_detection = segment_rows[-1][0]
        support_level = segment_rows[0][1]
        support_reason = segment_rows[0][2]
        start_frame = first_detection.frame_number
        end_frame = last_detection.frame_number
        tag_frames = [
            frame for frame in all_tag_frames
            if start_frame <= frame <= end_frame
        ]
        tag_count = len(tag_frames)

        if tag_count:
            first_aruco_frame = tag_frames[0]
            last_aruco_frame = tag_frames[-1]
        else:
            first_aruco_frame = None
            last_aruco_frame = None

        if tag_count > 1:
            gaps = np.diff(np.asarray(tag_frames, dtype=np.int32))
            max_gap_frames = int(np.max(gaps))
            median_gap_frames = float(np.median(gaps))
        else:
            max_gap_frames = None
            median_gap_frames = None

        return BeeIdentitySegmentData(
            video_id=self.video_id,
            identity_segment_id=f"{self.video_id}_identity_{segment_counter:05d}",
            bee_id=bee_id,
            aruco_code=aruco_code,
            support_level=support_level,
            start_frame=start_frame,
            end_frame=end_frame,
            duration_frames=end_frame - start_frame + 1,
            detection_count=len(segment_rows),
            tag_observation_count=tag_count,
            first_aruco_frame=first_aruco_frame,
            last_aruco_frame=last_aruco_frame,
            max_gap_frames=max_gap_frames,
            median_gap_frames=median_gap_frames,
            support_reason=support_reason,
        )
    
    def _finalize_processing(self):
        """Finalize processing - update ArUco codes retroactively"""
        # Update all bee detections with final ArUco codes
        for detection in self.bee_detections:
            if detection.bee_id in self.bee_to_aruco:
                detection.aruco_code = self.bee_to_aruco[detection.bee_id]

        for interaction in self.bee_interactions:
            if interaction.bee_id_1 in self.bee_to_aruco:
                interaction.aruco_code_1 = self.bee_to_aruco[interaction.bee_id_1]
            if interaction.bee_id_2 in self.bee_to_aruco:
                interaction.aruco_code_2 = self.bee_to_aruco[interaction.bee_id_2]
        
        # Update trajectories with final ArUco codes
        for bee_id, trajectory in self.bee_trajectories.items():
            if bee_id in self.bee_to_aruco:
                trajectory.aruco_code = self.bee_to_aruco[bee_id]

        self._finalize_identity_support()
        
        # Print ArUco detection summary
        if not self.prior_only:
            total_bees = len(self.bee_trajectories)
            tagged_bees = len(self.bee_to_aruco)
            self._log_verbose(f"\nArUco Detection Summary:")
            self._log_verbose(f"  Total unique bees tracked: {total_bees}")
            self._log_verbose(f"  Bees tagged with ArUco codes: {tagged_bees}")
            if tagged_bees > 0:
                self._log_verbose(f"  ArUco codes detected: {list(self.bee_to_aruco.values())}")
        
        # Print final timing statistics
        if self.verbose_output:
            self._log(f"\n{'='*60}")
            self._log(f"FINAL PERFORMANCE STATISTICS")
            self._log(f"{'='*60}")
            self._print_timing_stats(self.frame_count, final=True)
        
        # Free memory by clearing large data structures that are no longer needed
        # (keep only the essential data: bee_detections, chamber_frame_data, bee_trajectories)
        self._log_verbose(f"\nCleaning up memory...")
        if not self.store_masks:
            # These should already be empty, but ensure they're cleared
            self.chambers_by_frame.clear()
            self.hive_masks_by_frame.clear()
            self.bee_masks_by_frame.clear()
            self.pollen_masks_by_frame.clear()
            self.aruco_markers_by_frame.clear()
        
        # Force garbage collection and GPU cache clearing
        gc.collect()
        if TORCH_AVAILABLE and torch.cuda.is_available():
            torch.cuda.empty_cache()
            self._log_verbose(f"✓ GPU cache cleared")
    
    def _print_timing_stats(self, frame_number: int, final: bool = False):
        """Print timing statistics for performance analysis"""
        if not self.timings:
            return
        
        total_time = sum(self.timings.values())
        
        if final:
            self._log(f"Total frames processed: {frame_number}")
            self._log(f"Total processing time: {total_time:.2f}s")
            self._log(f"Average time per frame: {total_time / max(1, frame_number):.3f}s")
            self._log(f"\nTime breakdown by operation:")
        else:
            self._log(f"\n[Frame {frame_number}] Performance stats (last {frame_number} frames):")
        
        # Sort by time (descending)
        sorted_ops = sorted(self.timings.items(), key=lambda x: x[1], reverse=True)
        
        for op_name, op_time in sorted_ops:
            count = self.timing_counts[op_name]
            avg_time = op_time / count if count > 0 else 0
            percentage = (op_time / total_time * 100) if total_time > 0 else 0
            
            if final:
                self._log(f"  {op_name:20s}: {op_time:7.2f}s ({percentage:5.1f}%) - avg {avg_time*1000:6.1f}ms/frame")
            else:
                self._log(f"  {op_name:20s}: {op_time:7.2f}s ({percentage:5.1f}%) - {avg_time*1000:6.1f}ms avg")
        
        if final:
            self._log(f"{'='*60}\n")
    
    def get_bee_detections(self) -> List[BeeDetectionData]:
        """Get all bee detection data"""
        return self.bee_detections
    
    def get_chamber_frame_data(self) -> List[ChamberFrameData]:
        """Get all chamber frame data"""
        return self.chamber_frame_data

    def get_pollen_frame_data(self) -> List[PollenFrameData]:
        """Get pollen count/pixels by frame and chamber."""
        return self.pollen_frame_data

    def get_bee_interactions(self) -> List[BeeInteractionData]:
        """Get same-frame bee mask-contact events"""
        return self.bee_interactions

    def get_aruco_observations(self) -> List[ArucoObservationData]:
        """Get raw ArUco observation records and their acceptance decisions."""
        return self.aruco_observations

    def get_bee_identity_events(self) -> List[BeeIdentityEventData]:
        """Get ArUco-based identity assignment/rejection events."""
        return self.bee_identity_events

    def get_bee_identity_segments(self) -> List[BeeIdentitySegmentData]:
        """Get contiguous identity-support segments for tracked bees."""
        return self.bee_identity_segments
    
    def get_bee_trajectories(self) -> Dict[int, BeeTrajectory]:
        """Get all bee trajectories for velocity calculation"""
        return self.bee_trajectories    
    def get_chambers_by_frame(self) -> Dict[int, Dict]:
        """Get chamber information per frame"""
        return self.chambers_by_frame
    
    def get_hive_masks_by_frame(self) -> Dict[int, Dict[int, Optional[np.ndarray]]]:
        """Get hive masks per frame per chamber"""
        return self.hive_masks_by_frame
    
    def get_bee_masks_by_frame(self) -> Dict[int, Dict[int, Optional[np.ndarray]]]:
        """Get bee masks per frame per bee_id"""
        return self.bee_masks_by_frame

    def get_pollen_masks_by_frame(self) -> Dict[int, Dict[int, Optional[np.ndarray]]]:
        """Get pollen masks per frame per pollen_id."""
        return self.pollen_masks_by_frame

    def get_aruco_markers_by_frame(self) -> Dict[int, Dict[int, Dict]]:
        """Get detected ArUco marker corners per frame per bee_id."""
        return self.aruco_markers_by_frame
