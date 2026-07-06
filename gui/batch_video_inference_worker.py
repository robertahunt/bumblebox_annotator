"""
Worker thread for batch video inference with tracking and ArUco detection
"""

import cv2
import csv
import gc
import hashlib
import json
import numpy as np
import random
import re
import time
import torch
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional, Set
from PyQt6.QtCore import QThread, pyqtSignal
from ultralytics import YOLO

from core.aruco_parameter_optimizer import (
    load_tag_ids,
    normalize_dictionary_name,
    optimize_aruco_parameter_bank,
)
from core.batch_video_processor import BatchVideoProcessor
from core.temporal_hive_prior import TemporalHivePrior
from core.video_inference_exporter import VideoInferenceExporter
from core.visualization_generator import VisualizationGenerator


class BatchVideoInferenceWorker(QThread):
    """Worker thread for batch video inference with tracking"""
    STATUS_CSV = 'batch_video_status.csv'
    VISUALIZATION_SIGNATURE_KEYS = {
        'save_visualizations',
        'visualization_format',
        'visualization_interval',
        'visualization_max_frames',
    }
    STREAMING_CSVS = (
        'bee_detections.csv',
        'bee_interactions.csv',
        'aruco_observations.csv',
        'bee_identity_events.csv',
        'bee_identity_segments.csv',
        'bee_velocity.csv',
        'pollen_detections.csv',
        'hive_detections.csv',
        'chamber_detections.csv',
        'temporal_hive_priors.csv',
    )
    
    # Signals
    status_updated = pyqtSignal(str)
    progress_updated = pyqtSignal(int, int)  # current, total
    log_message = pyqtSignal(str)
    inference_complete = pyqtSignal()
    inference_stopped = pyqtSignal()
    inference_failed = pyqtSignal(str)
    
    def __init__(self, config: Dict):
        super().__init__()
        self.config = config
        self.should_stop = False
        
        # Data storage across all videos
        self.all_bee_detections = []
        self.all_bee_interactions = []
        self.all_aruco_observations = []
        self.all_bee_identity_events = []
        self.all_bee_identity_segments = []
        self.all_chamber_frame_data = []
        self.all_pollen_frame_data = []
        # Use composite keys (video_id, bee_id) to avoid collisions across videos
        self.all_bee_trajectories = {}  # {(video_id, bee_id): BeeTrajectory}
        
        # Accumulated masks for averaging
        # Format: {(video_id, chamber_id): {'accumulated_mask': np.ndarray, 'frame_count': int, 'shape': tuple}}
        self.accumulated_hive_masks = {}
        # Format: {(video_id, chamber_id): {'accumulated_mask': np.ndarray, 'frame_count': int, 'shape': tuple}}
        self.accumulated_chamber_masks = {}
        
        # Memory management: Track size of accumulated data
        self.accumulated_data_size_mb = 0
        self.verbose_output = bool(self.config.get('verbose_output', False))
        self.temporal_hive_prior = None
        self.total_bee_detections = 0
        self.total_bee_interactions = 0
        self.total_aruco_observations = 0
        self.total_bee_identity_events = 0
        self.total_bee_identity_segments = 0
        self.total_chamber_frame_records = 0
        self.total_pollen_frame_records = 0
        self.total_bee_trajectories = 0
        self.config_signature = self._config_signature()
        self.analysis_config_signature = self._config_signature(ignore_visualization=True)
        self._tag_map_cache = {}

    def _log_verbose(self, message: str):
        """Emit detailed diagnostic output only when verbose mode is enabled."""
        if self.verbose_output:
            self.log_message.emit(message)

    def _config_signature(
        self,
        ignore_visualization: bool = False,
        visualization_overrides: Optional[Dict] = None,
        drop_keys: Optional[set] = None,
    ) -> str:
        """Hash output-affecting settings so resume skips only compatible rows."""
        ignored_keys = {
            'output_folder',
            'video_source',
            'folder_mode',
            'preserve_file_order',
            'resume_completed_videos',
            'resume_ignore_config_mismatch',
            'skip_completed_visualizations',
            'verbose_output',
        }
        if ignore_visualization:
            ignored_keys.update(self.VISUALIZATION_SIGNATURE_KEYS)
        drop_keys = set(drop_keys or ())
        config_items = dict(self.config)
        if visualization_overrides:
            config_items.update(visualization_overrides)

        def normalize(value):
            if isinstance(value, Path):
                return str(value)
            if isinstance(value, dict):
                return {
                    str(k): normalize(v)
                    for k, v in sorted(value.items())
                    if str(k) not in ignored_keys
                }
            if isinstance(value, (list, tuple)):
                return [normalize(item) for item in value]
            return value

        normalized = {
            str(k): normalize(v)
            for k, v in sorted(config_items.items())
            if str(k) not in ignored_keys and str(k) not in drop_keys
        }
        payload = json.dumps(normalized, sort_keys=True, default=str)
        return hashlib.sha256(payload.encode('utf-8')).hexdigest()[:16]

    def _legacy_visualization_signature_candidates(self) -> set:
        """Signatures used by older manifests before visualization controls were split out."""
        cached = getattr(self, '_legacy_visualization_signatures', None)
        if cached is not None:
            return cached

        candidates = {self.config_signature}
        intervals = {1, 5, 10, 25, 50, 100, self._visualization_interval()}
        formats = {'video', 'frames'}

        for save_visualizations in (False, True):
            for visualization_format in formats:
                for interval in intervals:
                    candidates.add(self._config_signature(
                        visualization_overrides={
                            'save_visualizations': save_visualizations,
                            'visualization_format': visualization_format,
                            'visualization_interval': interval,
                        }
                    ))

            candidates.add(self._config_signature(
                visualization_overrides={'save_visualizations': save_visualizations},
                drop_keys={'visualization_format', 'visualization_interval'}
            ))

        candidates.add(self._config_signature(
            drop_keys=self.VISUALIZATION_SIGNATURE_KEYS
        ))
        self._legacy_visualization_signatures = candidates
        return candidates

    def _resume_ignores_visualization_settings(self) -> bool:
        """Whether completed analysis rows remain compatible despite visualization-only changes."""
        return (
            not self.config.get('save_visualizations', False)
            or bool(self.config.get('skip_completed_visualizations', True))
        )

    def _completed_row_config_match(self, row: Dict) -> tuple:
        """Return (matches, mode) for a completed-row compatibility check."""
        row_config_signature = row.get('config_signature') or ''
        if row_config_signature == self.config_signature:
            return True, 'exact'

        if self.config.get('resume_ignore_config_mismatch', False):
            return True, 'manual_override'

        if not self._resume_ignores_visualization_settings():
            return False, None

        row_analysis_signature = row.get('analysis_config_signature') or ''
        if row_analysis_signature and row_analysis_signature == self.analysis_config_signature:
            return True, 'analysis'

        if not row_analysis_signature and row_config_signature in self._legacy_visualization_signature_candidates():
            return True, 'legacy_visualization'

        return False, None
    
    def stop(self):
        """Request worker to stop"""
        self.should_stop = True
    
    def run(self):
        """Main execution"""
        try:
            self.log_message.emit("=== Batch Video Inference with Tracking ===")
            self.log_message.emit(f"Model type: {self.config.get('bee_model_type', 'bbox')}")
            self.log_message.emit(f"Distance method: {self.config.get('distance_method', 'contour')}")
            self.log_message.emit(f"Spatial metrics: {'Enabled' if self.config.get('compute_spatial_metrics', True) else 'Disabled'}")
            pixel_size_mm = self.config.get('pixel_size_mm')
            self.log_message.emit(
                f"Pixel scale: {pixel_size_mm:.4f} mm/px" if pixel_size_mm else "Pixel scale: not set"
            )
            self.log_message.emit(
                "Temporal hive prior: "
                f"{'Enabled' if self.config.get('use_temporal_hive_prior', False) else 'Disabled'}"
            )
            if self.config.get('preserve_file_order', False) and not self.config.get('folder_mode', False):
                self.log_message.emit("Video ordering: selected file/list order will be preserved")
                if (
                    self.config.get('use_temporal_hive_prior', False)
                    and self.config.get('temporal_context_include_date', False)
                ):
                    self.log_message.emit(
                        "Temporal prior context: includes MC/date folders for ordered-list balancing"
                    )
            self.log_message.emit(
                "Resume completed videos: "
                f"{'Enabled' if self.config.get('resume_completed_videos', False) else 'Disabled'}"
            )
            if self.config.get('resume_completed_videos', False) and self.config.get('resume_ignore_config_mismatch', False):
                self.log_message.emit(
                    "Resume config override: Enabled (completed videos may be skipped despite configuration changes)"
                )
            self.log_message.emit(f"Tracking algorithm: {self.config['tracking_config']['algorithm']}")
            self.log_message.emit(f"ArUco detection: {'Enabled' if self.config['enable_aruco'] else 'Disabled'}")
            if self.config['enable_aruco']:
                opt_cfg = self.config.get('aruco_optimization', {})
                if opt_cfg.get('enabled', False):
                    self.log_message.emit(
                        "ArUco parameter bank optimization: "
                        f"Enabled ({opt_cfg.get('sample_frames', 12)} sampled frames, "
                        f"bank size {opt_cfg.get('bank_size', 5)}, "
                        f"{opt_cfg.get('workers', 1)} workers)"
                    )
                else:
                    self.log_message.emit("ArUco parameter bank optimization: Disabled")
            if self.config.get('save_visualizations', False):
                viz_format = self.config.get('visualization_format', 'video')
                viz_format_text = "MP4 videos" if viz_format == 'video' else "frame images"
                frame_limit = self._visualization_frame_limit()
                frame_limit_text = (
                    "all frames" if frame_limit is None else f"first {frame_limit} frame(s)"
                )
                selected_indices = self.config.get('visualization_video_indices') or []
                if selected_indices:
                    indices_text = ", ".join(str(idx) for idx in selected_indices[:12])
                    if len(selected_indices) > 12:
                        indices_text += f", ... ({len(selected_indices)} total)"
                    self.log_message.emit(
                        f"Annotated visualizations: {viz_format_text}, "
                        f"selected video positions [{indices_text}], {frame_limit_text}"
                    )
                else:
                    self.log_message.emit(
                        f"Annotated visualizations: {viz_format_text}, "
                        f"every {self._visualization_interval()} video(s), {frame_limit_text}"
                    )
                if self.config.get('resume_completed_videos', False):
                    if self.config.get('skip_completed_visualizations', True):
                        self.log_message.emit(
                            "Completed-video visualization backfill: skipped in resume mode"
                        )
                    else:
                        self.log_message.emit(
                            "Completed-video visualization backfill: allowed if visualization settings changed"
                        )
            else:
                self.log_message.emit("Annotated visualizations: Disabled")
            if self.config.get('pollen_model_path') and self.config.get('exclude_pollen_from_hive', True):
                self.log_message.emit(
                    "Hive/pollen overlap handling: pollen pixels excluded from hive masks"
                )
            self.log_message.emit(f"Verbose output: {'Enabled' if self.verbose_output else 'Disabled'}")
            self.log_message.emit(f"Output folder: {self.config['output_folder']}")
            self.log_message.emit("")
            
            # Discover video files
            self.status_updated.emit("Discovering videos...")
            video_files = self._discover_videos()
            
            if not video_files:
                self.log_message.emit("❌ No video files found!")
                self.log_message.emit(f"  Looking in: {self.config['video_source']}")
                self.log_message.emit(f"  Folder mode: {self.config['folder_mode']}")
                self.inference_failed.emit("No video files found")
                return
            
            order_text = self._video_order_description()
            self.log_message.emit(f"Found {len(video_files)} video(s) to process ({order_text})")
            if self.verbose_output:
                for i, vf in enumerate(video_files[:5], 1):  # Log first 5 videos
                    self.log_message.emit(f"  {i}. {vf.name}")
                if len(video_files) > 5:
                    self.log_message.emit(f"  ... and {len(video_files) - 5} more")
                self.log_message.emit("  CSV rows will be flushed after each video to keep memory bounded")
            self.log_message.emit("")
            completed_video_paths = self._prepare_resume_state(video_files)
            if completed_video_paths and len(completed_video_paths) == len(video_files):
                temporal_summary_missing = (
                    self.config.get('use_temporal_hive_prior', False)
                    and not (Path(self.config['output_folder']) / 'temporal_hive_priors.csv').exists()
                )
                if not temporal_summary_missing:
                    self.status_updated.emit("All selected videos are already complete")
                    self.progress_updated.emit(len(video_files), len(video_files))
                    self.log_message.emit("Resume mode found all selected videos completed.")
                    self.log_message.emit("Existing CSVs were left unchanged.")
                    self.inference_complete.emit()
                    return

                self.log_message.emit(
                    "All selected videos are marked complete, but temporal_hive_priors.csv "
                    "is missing. Replaying completed videos in prior-only mode to rebuild it."
                )
            
            # Load models
            self.status_updated.emit("Loading models...")
            self.log_message.emit("Loading detection models...")
            
            bee_model = YOLO(self.config['bee_model_path'])
            self.log_message.emit(f"✓ Loaded bee model: {Path(self.config['bee_model_path']).name}")
            
            hive_model = None
            if self.config.get('hive_model_path'):
                hive_model = YOLO(self.config['hive_model_path'])
                self.log_message.emit(f"✓ Loaded hive model: {Path(self.config['hive_model_path']).name}")
            else:
                self.log_message.emit("  (No hive model - hive distance metrics will be blank)")

            pollen_model = None
            if self.config.get('pollen_model_path'):
                pollen_model = YOLO(self.config['pollen_model_path'])
                self.log_message.emit(f"✓ Loaded pollen model: {Path(self.config['pollen_model_path']).name}")
            else:
                self.log_message.emit("  (No pollen model - pollen metrics will be blank)")

            if self.config.get('use_temporal_hive_prior', False) and hive_model is not None:
                window_hours = float(self.config.get('temporal_hive_window_hours', 8.0))
                resolution = int(self.config.get('temporal_hive_resolution', 256))
                self.temporal_hive_prior = TemporalHivePrior(
                    window_seconds=window_hours * 60 * 60,
                    resolution=(resolution, resolution),
                )
                self.log_message.emit(
                    f"✓ Temporal hive prior enabled ({window_hours:.1f}h window, "
                    f"{resolution}x{resolution} chamber map)"
                )
            elif self.config.get('use_temporal_hive_prior', False):
                self.log_message.emit("  (Temporal hive prior disabled because no hive model was provided)")
            
            # Chamber model is optional
            chamber_model = None
            if self.config.get('chamber_model_path'):
                chamber_model = YOLO(self.config['chamber_model_path'])
                self.log_message.emit(f"✓ Loaded chamber model: {Path(self.config['chamber_model_path']).name}")
            else:
                self.log_message.emit("  (No chamber model - treating video as single chamber)")
            
            # Set models to eval mode to disable gradient tracking (memory optimization)
            if hasattr(bee_model, 'model') and hasattr(bee_model.model, 'eval'):
                bee_model.model.eval()
            if hive_model and hasattr(hive_model, 'model') and hasattr(hive_model.model, 'eval'):
                hive_model.model.eval()
            if pollen_model and hasattr(pollen_model, 'model') and hasattr(pollen_model.model, 'eval'):
                pollen_model.model.eval()
            if chamber_model and hasattr(chamber_model, 'model') and hasattr(chamber_model.model, 'eval'):
                chamber_model.model.eval()
            
            self.log_message.emit("")
            
            # Initialize tracking algorithm
            tracker = self._initialize_tracker()
            
            # Log ArUco status
            if self.config['enable_aruco']:
                self.log_message.emit("✓ ArUco detection enabled (for bee ID tracking)")
            
            # Process each video
            total_videos = len(video_files)
            for video_idx, video_path in enumerate(video_files, 1):
                if self.should_stop:
                    self.log_message.emit("\n⚠️ Processing stopped by user")
                    return

                video_key = self._video_manifest_path(video_path)
                if video_key in completed_video_paths:
                    if self.temporal_hive_prior is not None:
                        self.status_updated.emit(
                            f"Replaying temporal prior {video_idx}/{total_videos}: {video_path.name}"
                        )
                        self.log_message.emit(
                            f"=== Video {video_idx}/{total_videos}: {video_path.name} "
                            "(already complete; temporal prior replay) ==="
                        )
                        replay_success = self._replay_temporal_hive_prior(
                            video_path,
                            bee_model,
                            hive_model,
                            pollen_model,
                            chamber_model,
                            video_idx,
                            total_videos,
                        )
                        if not replay_success:
                            if self.should_stop:
                                break
                            raise RuntimeError(
                                f"Temporal prior replay failed for completed video: {video_path.name}"
                            )
                    else:
                        self.status_updated.emit(
                            f"Skipping completed video {video_idx}/{total_videos}: {video_path.name}"
                        )
                        self.log_message.emit(
                            f"=== Video {video_idx}/{total_videos}: {video_path.name} "
                            "(already complete; skipped) ==="
                        )

                    self.progress_updated.emit(video_idx, total_videos)
                    if self.should_stop:
                        break
                    continue
                
                self.status_updated.emit(f"Processing video {video_idx}/{total_videos}: {video_path.name}")
                self.log_message.emit(f"=== Video {video_idx}/{total_videos}: {video_path.name} ===")

                video_wall_start = time.perf_counter()
                aruco_runtime_config = self._prepare_aruco_runtime_config(video_path, video_idx, total_videos)
                if self.should_stop:
                    break
                
                # Process this video
                video_result = self._process_video(
                    video_path, 
                    bee_model, 
                    hive_model,
                    pollen_model,
                    chamber_model,
                    tracker,
                    video_idx,
                    aruco_runtime_config
                )
                video_wall_elapsed = time.perf_counter() - video_wall_start

                if video_result:
                    optimization_elapsed = float(
                        aruco_runtime_config.get('optimization_elapsed_seconds') or 0.0
                    )
                    video_result['aruco_optimization_seconds'] = optimization_elapsed
                    video_result['total_video_wall_seconds'] = video_wall_elapsed
                    if optimization_elapsed > 0:
                        self.log_message.emit(
                            "  Video timing: "
                            f"processing={float(video_result.get('elapsed_seconds') or 0.0):.2f}s, "
                            f"ArUco optimization={optimization_elapsed:.2f}s, "
                            f"total incl. optimization={video_wall_elapsed:.2f}s"
                        )

                if not self._export_csvs(intermediate=True, append=True, clear_after=True):
                    raise RuntimeError(f"CSV export failed for video: {video_path.name}")

                if video_result and video_result.get('completed', False):
                    self._write_completed_video_status(video_path, video_result)

                if self.should_stop:
                    break
                
                # Update progress after processing is complete
                self.progress_updated.emit(video_idx, total_videos)
                
                # Periodic checkpoint cleanup; CSV rows are already flushed after every video.
                if video_idx == 1 or video_idx % 10 == 0:
                    self.status_updated.emit(f"Checkpoint cleanup ({video_idx}/{total_videos})...")
                    # Aggressive GPU memory cleanup at checkpoint
                    self._log_verbose(f"  Running aggressive GPU memory cleanup at checkpoint...")
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        torch.cuda.synchronize()  # Wait for all operations to complete
                        mem_allocated = torch.cuda.memory_allocated() / 1e9
                        mem_reserved = torch.cuda.memory_reserved() / 1e9
                        self._log_verbose(f"    GPU memory after checkpoint cleanup: {mem_allocated:.2f} GB allocated, {mem_reserved:.2f} GB reserved")
                    
                    self.log_message.emit("")
                
                if self.should_stop:
                    break
            
            if self.should_stop:
                self.status_updated.emit("Exporting partial results...")
                self.log_message.emit("\n⚠️ Processing stopped by user")
                self.log_message.emit("=== Partial Export ===")
                self._finalize_csv_exports()
                self.log_message.emit(f"Partial results saved to: {self.config['output_folder']}")
                self.inference_stopped.emit()
                return
            
            # Final export of all data to CSVs
            self.status_updated.emit("Exporting final results...")
            self.log_message.emit("\n=== Final Export ===")
            if not self._finalize_csv_exports():
                raise RuntimeError("Final CSV export failed")
            
            self.log_message.emit("\n✓ Batch inference complete!")
            self.log_message.emit(f"Results saved to: {self.config['output_folder']}")
            self.inference_complete.emit()
            
        except Exception as e:
            import traceback
            error_msg = f"Inference failed: {str(e)}"
            self.log_message.emit(f"\n❌ {error_msg}")
            self._log_verbose(traceback.format_exc())
            self.inference_failed.emit(error_msg)
    
    def _discover_videos(self) -> List[Path]:
        """Discover video files from folder or file list."""
        video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.mjpeg', '.mjpg'}
        
        if self.config['folder_mode']:
            # Folder mode - discover recursively
            folder = Path(self.config['video_source'])
            video_files = []
            for ext in video_extensions:
                video_files.extend(folder.rglob(f"*{ext}"))
            video_files = [
                path for path in video_files
                if not self._should_exclude_discovered_video(path, folder)
            ]
            return self._order_videos(video_files)
        else:
            # File mode - use provided list
            video_files = [Path(f) for f in self.config['video_source']]
            return self._order_videos(video_files)

    def _path_is_relative_to(self, path: Path, root: Path) -> bool:
        try:
            path.expanduser().resolve().relative_to(root.expanduser().resolve())
            return True
        except ValueError:
            return False

    def _should_exclude_discovered_video(self, video_path: Path, input_folder: Path) -> bool:
        """Avoid re-ingesting videos generated by this tool during recursive discovery."""
        stem = video_path.stem.lower()
        if stem.endswith(('_annotated', '_partial_annotated', '_tracking_visualization')):
            return True

        output_folder = Path(self.config.get('output_folder', '')).expanduser().resolve()
        input_folder = input_folder.expanduser().resolve()
        video_path = video_path.expanduser().resolve()

        generated_dirs = [
            output_folder / 'annotated_videos',
            output_folder / 'visualizations',
            output_folder / 'aruco_optimization',
        ]
        if any(self._path_is_relative_to(video_path, generated_dir) for generated_dir in generated_dirs):
            return True

        output_inside_input = self._path_is_relative_to(output_folder, input_folder)
        if (
            output_inside_input
            and output_folder != input_folder
            and self._path_is_relative_to(video_path, output_folder)
        ):
            return True

        return False

    def _video_order_description(self) -> str:
        if self.config.get('preserve_file_order', False) and not self.config.get('folder_mode', False):
            return "selected file/list order"
        if self.config.get('use_temporal_hive_prior', False):
            return "chronological order"
        return "randomized order"

    def _order_videos(self, video_files: List[Path]) -> List[Path]:
        """Use selected order when requested; otherwise sort for priors or randomize."""
        if self.config.get('preserve_file_order', False) and not self.config.get('folder_mode', False):
            self.log_message.emit(
                f"Selected-file ordering: preserved ({len(video_files)} entries)"
            )
            return list(video_files)

        if self.config.get('use_temporal_hive_prior', False):
            ordered = sorted(video_files, key=self._video_sort_key)
            parsed = sum(1 for path in ordered if self._extract_video_datetime(path) is not None)
            self.log_message.emit(
                f"Temporal prior ordering: chronological filename sort "
                f"({parsed}/{len(ordered)} timestamps parsed)"
            )
            return ordered

        random.shuffle(video_files)
        return video_files

    def _manifest_path(self) -> Path:
        return Path(self.config['output_folder']) / self.STATUS_CSV

    def _video_manifest_path(self, video_path: Path) -> str:
        return str(Path(video_path).expanduser().resolve())

    def _existing_streaming_outputs(self) -> List[str]:
        output_folder = Path(self.config['output_folder'])
        filenames = list(self.STREAMING_CSVS) + [self.STATUS_CSV]
        return [name for name in filenames if (output_folder / name).exists()]

    def _missing_required_resume_outputs(self) -> List[str]:
        output_folder = Path(self.config['output_folder'])
        required = [
            'bee_detections.csv',
            'bee_interactions.csv',
            'aruco_observations.csv',
            'bee_identity_events.csv',
            'bee_identity_segments.csv',
            'bee_velocity.csv',
            'chamber_detections.csv',
            self.STATUS_CSV,
        ]
        if self.config.get('hive_model_path'):
            required.append('hive_detections.csv')
        if self.config.get('pollen_model_path'):
            required.append('pollen_detections.csv')
        return [name for name in required if not (output_folder / name).exists()]

    def _load_completed_video_paths(self, video_files: List[Path]) -> set:
        """Read compatible completed-video rows from the resume manifest."""
        manifest_path = self._manifest_path()
        if not manifest_path.exists():
            return set()

        candidate_paths = {self._video_manifest_path(path) for path in video_files}
        completed_paths = set()
        ignored_signature = 0
        ignored_missing = 0
        exact_matches = 0
        analysis_matches = 0
        legacy_visualization_matches = 0
        manual_override_matches = 0

        try:
            with open(manifest_path, 'r', newline='') as handle:
                reader = csv.DictReader(handle)
                for row in reader:
                    if row.get('status') != 'complete':
                        continue

                    matches_config, match_mode = self._completed_row_config_match(row)
                    if not matches_config:
                        ignored_signature += 1
                        continue

                    video_path = row.get('video_path') or ''
                    if video_path not in candidate_paths:
                        ignored_missing += 1
                        continue

                    completed_paths.add(video_path)
                    if match_mode == 'exact':
                        exact_matches += 1
                    elif match_mode == 'analysis':
                        analysis_matches += 1
                    elif match_mode == 'legacy_visualization':
                        legacy_visualization_matches += 1
                    elif match_mode == 'manual_override':
                        manual_override_matches += 1
        except Exception as exc:
            self.log_message.emit(f"  ⚠ Could not read resume manifest: {exc}")
            return set()

        if ignored_signature:
            self.log_message.emit(
                f"  Resume manifest ignored {ignored_signature} completed row(s) "
                "from a different batch configuration."
            )
        if ignored_missing and self.verbose_output:
            self.log_message.emit(
                f"  Resume manifest ignored {ignored_missing} row(s) outside this video selection."
            )
        if analysis_matches or legacy_visualization_matches:
            self.log_message.emit(
                "  Resume manifest accepted "
                f"{analysis_matches + legacy_visualization_matches} completed row(s) "
                "where only visualization settings differed."
            )
        if manual_override_matches:
            self.log_message.emit(
                "  Resume manifest accepted "
                f"{manual_override_matches} completed row(s) despite configuration differences "
                "(manual override enabled)."
            )
        if exact_matches and self.verbose_output:
            self.log_message.emit(
                f"  Resume manifest accepted {exact_matches} exact configuration match(es)."
            )

        return completed_paths

    def _infer_completed_video_paths_from_bee_csv(self, video_files: List[Path]) -> set:
        """Infer completed videos from an older bee_detections.csv with no status manifest."""
        detections_path = Path(self.config['output_folder']) / 'bee_detections.csv'
        if not detections_path.exists():
            return set()

        paths_by_video_id = {}
        for video_path in video_files:
            paths_by_video_id.setdefault(video_path.stem, []).append(self._video_manifest_path(video_path))

        completed_paths = set()
        ambiguous_ids = set()
        rows_read = 0

        try:
            with open(detections_path, 'r', newline='') as handle:
                reader = csv.DictReader(handle)
                if 'video_id' not in (reader.fieldnames or []):
                    self.log_message.emit(
                        "  Resume fallback could not use bee_detections.csv because it has no video_id column."
                    )
                    return set()

                for row in reader:
                    rows_read += 1
                    video_id = row.get('video_id') or ''
                    candidate_paths = paths_by_video_id.get(video_id, [])
                    if len(candidate_paths) == 1:
                        completed_paths.add(candidate_paths[0])
                    elif len(candidate_paths) > 1:
                        ambiguous_ids.add(video_id)
        except Exception as exc:
            self.log_message.emit(f"  ⚠ Could not infer completed videos from bee_detections.csv: {exc}")
            return set()

        if ambiguous_ids:
            self.log_message.emit(
                "  Resume fallback ignored ambiguous video_id(s) that match multiple selected files: "
                f"{', '.join(sorted(ambiguous_ids)[:5])}"
            )
        if completed_paths:
            self.log_message.emit(
                "  Resume fallback inferred "
                f"{len(completed_paths)} completed video(s) from bee_detections.csv "
                f"({rows_read:,} row(s) scanned)."
            )

        return completed_paths

    def _write_inferred_completed_video_status(self, completed_paths: set):
        """Create status rows for videos inferred from legacy CSV outputs."""
        if not completed_paths:
            return

        manifest_path = self._manifest_path()
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = self._status_fieldnames()
        self._ensure_manifest_schema(manifest_path, fieldnames)
        write_header = not manifest_path.exists() or manifest_path.stat().st_size == 0

        with open(manifest_path, 'a', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            if write_header:
                writer.writeheader()

            for video_path_text in sorted(completed_paths):
                video_path = Path(video_path_text)
                writer.writerow({
                    'status': 'complete',
                    'video_id': video_path.stem,
                    'video_path': video_path_text,
                    'completed_at': datetime.now().isoformat(timespec='seconds'),
                    'frame_count': '',
                    'bee_detections': '',
                    'bee_interactions': '',
                    'aruco_observations': '',
                    'bee_identity_events': '',
                    'bee_identity_segments': '',
                    'bee_trajectories': '',
                    'chamber_frame_records': '',
                    'pollen_frame_records': '',
                    'elapsed_seconds': '',
                    'config_signature': self.config_signature,
                    'analysis_config_signature': self.analysis_config_signature,
                })

        self.log_message.emit(
            f"  ✓ Created {self.STATUS_CSV} entries for inferred completed videos."
        )

    def _prepare_resume_state(self, video_files: List[Path]) -> set:
        """Decide whether to reset outputs or append to compatible completed rows."""
        resume_enabled = bool(self.config.get('resume_completed_videos', False))

        if not resume_enabled:
            self._reset_streaming_csvs(include_manifest=True)
            return set()

        missing_outputs = self._missing_required_resume_outputs()
        if missing_outputs:
            if missing_outputs == [self.STATUS_CSV]:
                inferred_completed_paths = self._infer_completed_video_paths_from_bee_csv(video_files)
                if inferred_completed_paths:
                    self.log_message.emit(
                        f"Resume requested, but {self.STATUS_CSV} was missing. "
                        "Using existing CSV rows to seed the resume manifest."
                    )
                    self._write_inferred_completed_video_status(inferred_completed_paths)
                    pending_count = len(video_files) - len(inferred_completed_paths)
                    self.log_message.emit(
                        f"Resume mode: {len(inferred_completed_paths)} completed video(s) will be skipped; "
                        f"{pending_count} video(s) remain."
                    )
                    self.log_message.emit("Existing CSVs will be preserved and new rows will be appended.")
                    return inferred_completed_paths

            existing_outputs = self._existing_streaming_outputs()
            if existing_outputs:
                self.log_message.emit(
                    "Resume requested, but existing outputs are not resumable "
                    f"(missing: {', '.join(missing_outputs)})."
                )
                self.log_message.emit("Starting a fresh batch export.")
            else:
                self.log_message.emit("Resume requested, but no previous batch outputs were found.")
            self._reset_streaming_csvs(include_manifest=True)
            return set()

        completed_paths = self._load_completed_video_paths(video_files)
        if not completed_paths:
            self.log_message.emit(
                "Resume requested, but no compatible completed videos were found. "
                "Starting a fresh batch export."
            )
            self._reset_streaming_csvs(include_manifest=True)
            return set()

        pending_count = len(video_files) - len(completed_paths)
        self.log_message.emit(
            f"Resume mode: {len(completed_paths)} completed video(s) will be skipped; "
            f"{pending_count} video(s) remain."
        )
        self.log_message.emit("Existing CSVs will be preserved and new rows will be appended.")
        return completed_paths

    def _status_fieldnames(self) -> List[str]:
        return [
            'status',
            'video_id',
            'video_path',
            'completed_at',
            'frame_count',
            'bee_detections',
            'bee_interactions',
            'aruco_observations',
            'bee_identity_events',
            'bee_identity_segments',
            'bee_trajectories',
            'chamber_frame_records',
            'pollen_frame_records',
            'elapsed_seconds',
            'aruco_optimization_seconds',
            'total_video_wall_seconds',
            'config_signature',
            'analysis_config_signature',
        ]

    def _ensure_manifest_schema(self, manifest_path: Path, fieldnames: List[str]):
        """Upgrade older status manifests before appending rows with new columns."""
        if not manifest_path.exists() or manifest_path.stat().st_size == 0:
            return

        try:
            with open(manifest_path, 'r', newline='') as handle:
                reader = csv.DictReader(handle)
                existing_fieldnames = reader.fieldnames or []
                if all(name in existing_fieldnames for name in fieldnames):
                    return
                rows = list(reader)

            with open(manifest_path, 'w', newline='') as handle:
                writer = csv.DictWriter(handle, fieldnames=fieldnames)
                writer.writeheader()
                for row in rows:
                    writer.writerow({name: row.get(name, '') for name in fieldnames})

            self._log_verbose(f"  ✓ Upgraded {self.STATUS_CSV} schema")
        except Exception as exc:
            self.log_message.emit(f"  ⚠ Could not upgrade resume manifest schema: {exc}")

    def _write_completed_video_status(self, video_path: Path, video_result: Dict):
        """Append a completed-video row after that video's CSV rows are safely flushed."""
        manifest_path = self._manifest_path()
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = self._status_fieldnames()
        self._ensure_manifest_schema(manifest_path, fieldnames)
        write_header = not manifest_path.exists() or manifest_path.stat().st_size == 0

        with open(manifest_path, 'a', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            if write_header:
                writer.writeheader()
            writer.writerow({
                'status': 'complete',
                'video_id': video_path.stem,
                'video_path': self._video_manifest_path(video_path),
                'completed_at': datetime.now().isoformat(timespec='seconds'),
                'frame_count': int(video_result.get('frame_count') or 0),
                'bee_detections': int(video_result.get('bee_detections') or 0),
                'bee_interactions': int(video_result.get('bee_interactions') or 0),
                'aruco_observations': int(video_result.get('aruco_observations') or 0),
                'bee_identity_events': int(video_result.get('bee_identity_events') or 0),
                'bee_identity_segments': int(video_result.get('bee_identity_segments') or 0),
                'bee_trajectories': int(video_result.get('bee_trajectories') or 0),
                'chamber_frame_records': int(video_result.get('chamber_frame_records') or 0),
                'pollen_frame_records': int(video_result.get('pollen_frame_records') or 0),
                'elapsed_seconds': f"{float(video_result.get('elapsed_seconds') or 0.0):.2f}",
                'aruco_optimization_seconds': f"{float(video_result.get('aruco_optimization_seconds') or 0.0):.2f}",
                'total_video_wall_seconds': f"{float(video_result.get('total_video_wall_seconds') or 0.0):.2f}",
                'config_signature': self.config_signature,
                'analysis_config_signature': self.analysis_config_signature,
            })

        self._log_verbose(f"  ✓ Marked complete in {self.STATUS_CSV}: {video_path.name}")

    def _video_sort_key(self, video_path: Path):
        video_dt = self._extract_video_datetime(video_path)
        context_id = self._temporal_context_id(video_path)
        if video_dt is None:
            return (1, context_id, video_path.name.lower())
        return (0, video_dt, context_id, video_path.name.lower())

    def _extract_video_datetime(self, video_path: Path):
        match = re.search(
            r'(\d{4})-(\d{2})-(\d{2})[_-](\d{2})[_:](\d{2})[_:](\d{2})',
            video_path.stem
        )
        if not match:
            return None

        try:
            year, month, day, hour, minute, second = [int(part) for part in match.groups()]
            return datetime(year, month, day, hour, minute, second)
        except ValueError:
            return None

    def _video_start_time_seconds(self, video_path: Path):
        video_dt = self._extract_video_datetime(video_path)
        return video_dt.timestamp() if video_dt is not None else None

    def _temporal_context_id(self, video_path: Path) -> str:
        match = re.search(r'(bumblebox[-_]\d+)', video_path.stem, flags=re.IGNORECASE)
        if match:
            context_id = match.group(1).lower().replace('_', '-')
        else:
            context_id = video_path.parent.name or "default"

        if self.config.get('temporal_context_include_date', False):
            context_parts = [context_id]
            for parent in (video_path.parent.parent.name, video_path.parent.name):
                if re.match(r'mcs?[-_ ]+\d+[-_ ]+and[-_ ]+\d+', parent, flags=re.IGNORECASE):
                    context_parts.append(parent.lower().replace(" ", "-").replace("_", "-"))
                    break

            video_dt = self._extract_video_datetime(video_path)
            if video_dt is not None:
                context_parts.append(video_dt.date().isoformat())
            context_id = ":".join(context_parts)

        return context_id
    
    def _initialize_tracker(self):
        """Initialize tracking algorithm based on config"""
        tracking_config = self.config['tracking_config']
        algo = tracking_config['algorithm']
        
        if algo == 'bytetrack':
            from core.instance_tracker import InstanceTracker
            
            tracker_config = {
                'high_conf_threshold': tracking_config['high_conf_threshold'],
                'high_iou_threshold': tracking_config['high_iou_threshold'],
                'low_iou_threshold': tracking_config['low_iou_threshold'],
                'max_frames_lost': tracking_config['max_frames_lost'],
                'use_mask_iou': tracking_config['use_mask_iou']
            }
            tracker = InstanceTracker(config=tracker_config)
            self.log_message.emit(f"✓ Initialized ByteTrack tracker")
            
        elif algo == 'simple_iou':
            # Import from tracking_validation_worker where these are defined
            from gui.tracking_validation_worker import SimpleIoUTracker
            
            tracker = SimpleIoUTracker(
                iou_threshold=tracking_config['iou_threshold'],
                use_mask_iou=tracking_config['use_mask_iou']
            )
            self.log_message.emit(f"✓ Initialized SimpleIoU tracker")
            
        elif algo == 'centroid':
            from gui.tracking_validation_worker import CentroidTracker
            
            tracker = CentroidTracker(
                max_distance=tracking_config['max_distance'],
                max_frames_missing=tracking_config['max_frames_missing']
            )
            self.log_message.emit(f"✓ Initialized Centroid tracker")
        
        else:
            raise ValueError(f"Unknown tracking algorithm: {algo}")
        
        return tracker

    def _parse_tag_id_tokens(self, raw_value) -> Set[int]:
        tag_ids = set()
        if raw_value is None:
            return tag_ids

        if isinstance(raw_value, (list, tuple, set)):
            values = raw_value
        else:
            values = re.findall(r'-?\d+', str(raw_value))

        for raw_id in values:
            try:
                tag_ids.add(int(raw_id))
            except (TypeError, ValueError):
                continue
        return tag_ids

    def _normalize_microcolony_key(self, raw_value) -> Optional[str]:
        text = str(raw_value or "").strip().lower()
        if not text:
            return None

        numbers = [int(item) for item in re.findall(r'\d+', text)]
        if len(numbers) >= 2:
            return f"mcs-{numbers[0]}-and-{numbers[1]}"
        if len(numbers) == 1:
            return f"mc-{numbers[0]}"

        normalized = re.sub(r'[^a-z0-9]+', '-', text).strip('-')
        return normalized or None

    def _microcolony_keys_for_video(self, video_path: Path) -> List[str]:
        keys = []
        for parent in [video_path.parent.name, video_path.parent.parent.name, *[p.name for p in video_path.parents]]:
            text = str(parent or "")
            if not re.search(r'\bmcs?\b', text, flags=re.IGNORECASE):
                continue
            numbers = [int(item) for item in re.findall(r'\d+', text)]
            if len(numbers) < 2:
                continue
            pair_key = f"mcs-{numbers[0]}-and-{numbers[1]}"
            keys.extend([pair_key, f"mc-{numbers[0]}", f"mc-{numbers[1]}"])
            break
        return keys

    def _load_tag_id_map(self, path_key: str) -> Optional[Dict[str, Set[int]]]:
        tag_list_path = self.config.get(path_key)
        if not tag_list_path:
            return None

        tag_path = Path(tag_list_path).expanduser()
        cache_key = str(tag_path)
        if cache_key in self._tag_map_cache:
            return self._tag_map_cache[cache_key]

        if tag_path.suffix.lower() != ".csv" or not tag_path.exists():
            self._tag_map_cache[cache_key] = None
            return None

        pair_columns = (
            "microcolony_pair",
            "mc_pair",
            "microcolony_ids",
            "microcolony_id",
            "microcolony",
            "mc",
            "mcs",
        )
        tag_columns = (
            "tag_id",
            "tag_ids",
            "aruco_id",
            "aruco_ids",
            "marker_id",
            "marker_ids",
            "id",
            "ids",
            "tag",
            "tags",
        )

        with open(tag_path, "r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            raw_fieldnames = reader.fieldnames or []
            field_lookup = {
                re.sub(r'[^a-z0-9]+', '_', name.strip().lower()).strip('_'): name
                for name in raw_fieldnames
            }
            pair_column = next((field_lookup[name] for name in pair_columns if name in field_lookup), None)
            tag_column = next((field_lookup[name] for name in tag_columns if name in field_lookup), None)
            if not pair_column or not tag_column:
                self._tag_map_cache[cache_key] = None
                return None

            tag_map: Dict[str, Set[int]] = {}
            for row in reader:
                pair_key = self._normalize_microcolony_key(row.get(pair_column))
                tag_ids = self._parse_tag_id_tokens(row.get(tag_column))
                if not pair_key or not tag_ids:
                    continue
                tag_map.setdefault(pair_key, set()).update(tag_ids)

        self._tag_map_cache[cache_key] = tag_map or None
        return self._tag_map_cache[cache_key]

    def _mapped_tag_ids_for_video(self, path_key: str, video_path: Path) -> Optional[Set[int]]:
        tag_map = self._load_tag_id_map(path_key)
        if not tag_map:
            return None

        tag_ids = set()
        for key in self._microcolony_keys_for_video(video_path):
            tag_ids.update(tag_map.get(key, set()))
        return tag_ids

    def _load_tag_ids_from_config(self, inline_key: str, path_key: str, video_path: Optional[Path] = None) -> Optional[List[int]]:
        tag_ids = set()
        tag_ids.update(self._parse_tag_id_tokens(self.config.get(inline_key, []) or []))

        tag_list_path = self.config.get(path_key)
        mapped_ids = self._mapped_tag_ids_for_video(path_key, video_path) if video_path else None
        if mapped_ids is not None:
            tag_ids.update(mapped_ids)
            if not mapped_ids:
                self._log_verbose(
                    f"  ArUco tag map had no match for {video_path}; using inline tags only"
                )
        elif tag_list_path:
            tag_ids.update(load_tag_ids(tag_list_path))

        return sorted(tag_ids) if tag_ids else None

    def _load_allowed_tag_ids(self, video_path: Path) -> Optional[List[int]]:
        return self._load_tag_ids_from_config('allowed_tag_ids', 'tag_list_path', video_path)

    def _load_excluded_tag_ids(self, video_path: Path) -> Optional[List[int]]:
        return self._load_tag_ids_from_config('excluded_tag_ids', 'exclude_tag_list_path', video_path)

    def _prepare_aruco_runtime_config(self, video_path: Path, video_idx: int, total_videos: int) -> Dict:
        if not self.config.get('enable_aruco', True):
            return {}

        allowed_tag_ids = self._load_allowed_tag_ids(video_path)
        excluded_tag_ids = self._load_excluded_tag_ids(video_path)
        opt_cfg = self.config.get('aruco_optimization', {}) or {}
        dictionary = normalize_dictionary_name(opt_cfg.get('dictionary', self.config.get('aruco_dictionary', '4x4_100')))
        runtime_config = {
            'aruco_dicts': [dictionary],
            'aruco_params_bank': None,
            'allowed_tag_ids': allowed_tag_ids,
            'excluded_tag_ids': excluded_tag_ids,
            'optimization_elapsed_seconds': 0.0,
        }

        if allowed_tag_ids:
            self._log_verbose(f"  ArUco allowed tags for this video: {allowed_tag_ids}")
        if excluded_tag_ids:
            self._log_verbose(f"  ArUco excluded tags: {excluded_tag_ids}")

        if not opt_cfg.get('enabled', False):
            if self.config.get('aruco_dictionary_mode', 'auto_4x4') == 'auto_4x4':
                runtime_config['aruco_dicts'] = None
            return runtime_config

        self.status_updated.emit(f"Optimizing ArUco parameters {video_idx}/{total_videos}: {video_path.name}")
        self.log_message.emit(f"  Optimizing ArUco parameter bank for {video_path.name}...")

        def progress(done: int, total: int, latest: Dict):
            self.log_message.emit(
                "    "
                f"ArUco sweep {done}/{total}: "
                f"accepted_ids/frame={float(latest.get('mean_detected', 0.0)):.2f}, "
                f"decoded/frame={float(latest.get('mean_decoded', 0.0)):.2f}, "
                f"stability={float(latest.get('stability', 0.0)):.2f}, "
                f"score={float(latest.get('score', 0.0)):.3f}"
            )

        optimization_start = time.perf_counter()
        result = optimize_aruco_parameter_bank(
            video_path=video_path,
            output_dir=Path(self.config['output_folder']) / "aruco_optimization",
            dictionary_name=dictionary,
            profile=opt_cfg.get('profile', 'daily'),
            sample_frames=int(opt_cfg.get('sample_frames', 12)),
            max_combinations=int(opt_cfg.get('max_combinations', 750)),
            bank_size=int(opt_cfg.get('bank_size', 5)),
            expected_tags=opt_cfg.get('expected_tags'),
            allowed_tag_ids=allowed_tag_ids,
            excluded_tag_ids=excluded_tag_ids,
            sweep_overrides=opt_cfg.get('sweep_overrides') or None,
            workers=int(opt_cfg.get('workers', 1)),
            progress_callback=progress,
            stop_requested=lambda: self.should_stop,
        )
        optimization_elapsed = time.perf_counter() - optimization_start

        runtime_config['aruco_params_bank'] = result.parameter_bank
        runtime_config['optimization_elapsed_seconds'] = optimization_elapsed
        self.log_message.emit(
            f"  ✓ Selected {len(result.parameter_bank)} ArUco parameter set(s); "
            f"workers used: {result.workers}; optimization={optimization_elapsed:.2f}s; "
            f"summary: {result.summary_json_path}"
        )
        requested_bank_size = int(opt_cfg.get('bank_size', 5))
        if len(result.parameter_bank) < requested_bank_size:
            self.log_message.emit(
                f"  ⚠️ Selected only {len(result.parameter_bank)}/{requested_bank_size} requested "
                "ArUco parameter set(s); remaining candidates added no accepted-ID coverage."
            )
        if result.selected_candidates:
            best = result.selected_candidates[0]
            self.log_message.emit(
                f"    Primary candidate: accepted_ids/frame={best.mean_detected:.2f}, "
                f"decoded/frame={best.mean_decoded:.2f}, "
                f"unique_ids={best.unique_ids}, "
                f"coverage={best.coverage_frames}/{len(result.sampled_frame_indices)}, "
                f"stability={best.stability:.2f}, "
                f"score={best.score:.3f}"
            )
            if best.mean_detected <= 0 and best.mean_decoded > 0:
                self.log_message.emit(
                    "    ⚠️ ArUco decoded markers, but none survived allow/exclude/perimeter filtering. "
                    "Check the per-video allowlist, exclude list, and perimeter sweep range."
                )
        return runtime_config

    def _log_processor_timing(self, processor, wall_elapsed: float):
        """Log a timing breakdown from the batch video processor."""
        if not self.verbose_output:
            return

        timings = getattr(processor, 'timings', {})
        timing_counts = getattr(processor, 'timing_counts', {})

        if not timings:
            self.log_message.emit("  Timing breakdown unavailable")
            return

        measured_total = sum(timings.values())
        other_time = max(0.0, wall_elapsed - measured_total)
        frame_count = max(1, getattr(processor, 'frame_count', 0))

        self.log_message.emit(f"  Timing breakdown:")
        self.log_message.emit(f"    - Wall-clock total: {wall_elapsed:.2f}s ({wall_elapsed / frame_count:.3f}s/frame)")
        self.log_message.emit(f"    - Measured processing: {measured_total:.2f}s")

        sorted_ops = sorted(timings.items(), key=lambda item: item[1], reverse=True)
        for op_name, op_time in sorted_ops:
            count = timing_counts.get(op_name, 0)
            avg_ms = (op_time / count * 1000) if count else 0.0
            pct = (op_time / wall_elapsed * 100) if wall_elapsed > 0 else 0.0
            pretty_name = op_name.replace('_', ' ')
            self.log_message.emit(f"    - {pretty_name}: {op_time:.2f}s ({pct:.1f}%), {avg_ms:.1f} ms/call")

        if other_time > 0.01:
            pct = (other_time / wall_elapsed * 100) if wall_elapsed > 0 else 0.0
            self.log_message.emit(f"    - other/frame IO/finalization: {other_time:.2f}s ({pct:.1f}%)")

        if torch.cuda.is_available():
            self.log_message.emit("    Note: GPU timings can shift a little because CUDA work may be asynchronous.")

    def _visualization_interval(self) -> int:
        try:
            return max(1, int(self.config.get('visualization_interval', 1)))
        except (TypeError, ValueError):
            return 1

    def _visualization_frame_limit(self) -> Optional[int]:
        try:
            limit = int(self.config.get('visualization_max_frames', 25))
        except (TypeError, ValueError):
            return 25
        return limit if limit > 0 else None

    def _should_generate_visualization(self, video_idx: int) -> bool:
        if not self.config.get('save_visualizations', False):
            return False

        selected_indices = self.config.get('visualization_video_indices') or []
        if selected_indices:
            try:
                return int(video_idx) in {int(idx) for idx in selected_indices}
            except (TypeError, ValueError):
                return False

        interval = self._visualization_interval()
        return interval <= 1 or video_idx % interval == 0

    def _replay_temporal_hive_prior(
        self,
        video_path: Path,
        bee_model,
        hive_model,
        pollen_model,
        chamber_model,
        video_idx: int,
        total_videos: int,
    ) -> bool:
        """Replay a completed video only to rebuild the rolling temporal hive prior."""
        if self.temporal_hive_prior is None or hive_model is None:
            return True

        temporal_context_id = self._temporal_context_id(video_path)
        video_start_time_seconds = self._video_start_time_seconds(video_path)
        timestamp_text = (
            datetime.fromtimestamp(video_start_time_seconds).isoformat(sep=' ')
            if video_start_time_seconds is not None
            else "unparsed timestamp"
        )
        self._log_verbose(
            f"  Prior replay context: {temporal_context_id}, start: {timestamp_text}"
        )

        processor = BatchVideoProcessor(
            video_path=video_path,
            video_id=video_path.stem,
            bee_model=bee_model,
            hive_model=hive_model,
            chamber_model=chamber_model,
            pollen_model=(
                pollen_model
                if self.config.get('exclude_pollen_from_hive', True)
                else None
            ),
            tracker=None,
            confidence_threshold=self.config['confidence_threshold'],
            nms_iou_threshold=self.config['nms_iou_threshold'],
            enable_aruco=False,
            output_folder=Path(self.config['output_folder']),
            distance_method='centroid',
            bee_model_type=self.config.get('bee_model_type', 'bbox'),
            compute_spatial_metrics=False,
            store_masks=False,
            log_callback=self.log_message.emit,
            stop_callback=lambda: self.should_stop,
            timing_log_interval=1,
            cleanup_interval=25,
            verbose_output=self.verbose_output,
            max_frames=self.config.get('max_frames'),
            temporal_hive_prior=self.temporal_hive_prior,
            temporal_hive_context_id=temporal_context_id,
            video_start_time_seconds=video_start_time_seconds,
            pixel_size_mm=self.config.get('pixel_size_mm'),
            exclude_pollen_from_hive=self.config.get('exclude_pollen_from_hive', True),
            prior_only=True,
        )

        start = time.perf_counter()
        success = processor.process()
        elapsed = time.perf_counter() - start

        if success:
            self.log_message.emit(
                f"  ✓ Replayed temporal prior from {processor.frame_count} frame(s) "
                f"for completed video {video_idx}/{total_videos}"
            )
            self._log_processor_timing(processor, elapsed)
        elif self.should_stop or getattr(processor, 'was_stopped', False):
            self.log_message.emit(f"  ⚠️ Stopped during temporal prior replay: {video_path.name}")
        else:
            self.log_message.emit(f"  ❌ Temporal prior replay failed: {video_path.name}")

        del processor
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return success
    
    def _process_video(self, video_path, bee_model, hive_model, pollen_model, chamber_model, tracker, video_idx, aruco_runtime_config=None):
        """Process a single video file and return completion stats when successful."""
        video_id = video_path.stem  # Use filename without extension as video_id
        self.log_message.emit(f"  Processing: {video_path.name}")
        aruco_runtime_config = aruco_runtime_config or {}
        temporal_context_id = self._temporal_context_id(video_path)
        video_start_time_seconds = self._video_start_time_seconds(video_path)
        if self.temporal_hive_prior is not None:
            timestamp_text = (
                datetime.fromtimestamp(video_start_time_seconds).isoformat(sep=' ')
                if video_start_time_seconds is not None
                else "unparsed timestamp"
            )
            self._log_verbose(
                f"  Temporal hive prior context: {temporal_context_id}, start: {timestamp_text}"
            )
        
        # Check if visualization is enabled for this batch index before storing masks.
        visualization_enabled = self._should_generate_visualization(video_idx)
        visualization_frame_limit = self._visualization_frame_limit() if visualization_enabled else None
        visualization_format = self.config.get('visualization_format', 'video')
        streaming_visualization_enabled = (
            visualization_enabled and visualization_format in {'video', 'frames'}
        )
        streaming_visualization_path = None
        if streaming_visualization_enabled:
            if visualization_format == 'frames':
                streaming_visualization_path = (
                    Path(self.config['output_folder']) / 'visualizations' / video_id
                )
            else:
                suffix = "_partial_annotated.mp4" if self.should_stop else "_annotated.mp4"
                streaming_visualization_path = (
                    Path(self.config['output_folder']) / 'annotated_videos' / f"{video_id}{suffix}"
                )
        
        # Log memory optimization setting
        if not visualization_enabled:
            if self.config.get('save_visualizations', False):
                self._log_verbose(
                    f"  Visualization interval: skipping annotated output for video {video_idx}"
                )
            self._log_verbose(f"  Memory optimization: Masks will NOT be stored")
        else:
            limit_text = (
                "all frames" if visualization_frame_limit is None
                else f"first {visualization_frame_limit} frame(s)"
            )
            if streaming_visualization_enabled:
                if visualization_format == 'frames':
                    self._log_verbose(f"  Streaming annotated frame images during processing ({limit_text})")
                else:
                    self._log_verbose(f"  Streaming annotated MP4 during processing ({limit_text})")
            else:
                self._log_verbose(f"  Memory usage: Storing masks for visualization ({limit_text})")
        
        # Create video processor
        processor = BatchVideoProcessor(
            video_path=video_path,
            video_id=video_id,
            bee_model=bee_model,
            hive_model=hive_model,
            chamber_model=chamber_model,
            pollen_model=pollen_model,
            tracker=tracker,
            confidence_threshold=self.config['confidence_threshold'],
            nms_iou_threshold=self.config['nms_iou_threshold'],
            enable_aruco=self.config['enable_aruco'],
            output_folder=Path(self.config['output_folder']),
            distance_method=self.config.get('distance_method', 'contour'),
            bee_model_type=self.config.get('bee_model_type', 'bbox'),
            compute_spatial_metrics=self.config.get('compute_spatial_metrics', True),
            store_masks=visualization_enabled and not streaming_visualization_enabled,
            log_callback=self.log_message.emit,
            stop_callback=lambda: self.should_stop,
            timing_log_interval=1,
            cleanup_interval=25,
            verbose_output=self.verbose_output,
            max_frames=self.config.get('max_frames'),
            store_masks_until_frame=(
                visualization_frame_limit
                if visualization_enabled and not streaming_visualization_enabled
                else None
            ),
            streaming_visualization_path=streaming_visualization_path,
            streaming_visualization_max_frames=(
                visualization_frame_limit if streaming_visualization_enabled else None
            ),
            streaming_visualization_format=visualization_format,
            aruco_dicts=aruco_runtime_config.get('aruco_dicts'),
            aruco_params_bank=aruco_runtime_config.get('aruco_params_bank'),
            allowed_tag_ids=aruco_runtime_config.get('allowed_tag_ids'),
            excluded_tag_ids=aruco_runtime_config.get('excluded_tag_ids'),
            temporal_hive_prior=self.temporal_hive_prior,
            temporal_hive_context_id=temporal_context_id,
            video_start_time_seconds=video_start_time_seconds,
            pixel_size_mm=self.config.get('pixel_size_mm'),
            exclude_pollen_from_hive=self.config.get('exclude_pollen_from_hive', True)
        )
        
        # Process video
        process_start = time.perf_counter()
        success = processor.process()
        process_elapsed = time.perf_counter() - process_start
        
        if not success:
            if self.should_stop or getattr(processor, 'was_stopped', False):
                self.log_message.emit(f"  ⚠️ Stopped while processing video: {video_path.name}")
                # Keep the partial data collected before the stop.
                self.all_bee_detections.extend(processor.get_bee_detections())
                self.all_bee_interactions.extend(processor.get_bee_interactions())
                self.all_aruco_observations.extend(processor.get_aruco_observations())
                self.all_bee_identity_events.extend(processor.get_bee_identity_events())
                self.all_bee_identity_segments.extend(processor.get_bee_identity_segments())
                self.all_chamber_frame_data.extend(processor.get_chamber_frame_data())
                self.all_pollen_frame_data.extend(processor.get_pollen_frame_data())
                for bee_id, trajectory in processor.get_bee_trajectories().items():
                    self.all_bee_trajectories[(video_id, bee_id)] = trajectory
                if streaming_visualization_enabled:
                    self._log_streaming_visualization_result(processor, streaming_visualization_path, partial=True)
                elif visualization_enabled:
                    self._generate_visualizations(
                        video_path, video_id, processor, partial=True,
                        frame_limit=visualization_frame_limit,
                    )
                    self._accumulate_masks(video_id, processor.get_hive_masks_by_frame(), processor.get_chambers_by_frame())
                del processor
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                return None

            self.log_message.emit(f"  ❌ Failed to process video: {video_path}")
            self.log_message.emit(f"     Check that video file is valid and models are compatible")
            return None
        
        # Log processing statistics
        num_detections = len(processor.get_bee_detections())
        num_interactions = len(processor.get_bee_interactions())
        num_aruco_observations = len(processor.get_aruco_observations())
        num_identity_events = len(processor.get_bee_identity_events())
        num_identity_segments = len(processor.get_bee_identity_segments())
        num_trajectories = len(processor.get_bee_trajectories())
        num_chamber_data = len(processor.get_chamber_frame_data())
        num_pollen_data = len(processor.get_pollen_frame_data())
        processor_frame_count = processor.frame_count
        
        self.log_message.emit(f"  ✓ Processed successfully:")
        self.log_message.emit(f"    - {num_detections} bee detections")
        self.log_message.emit(f"    - {num_interactions} bee contact events")
        self.log_message.emit(f"    - {num_aruco_observations} ArUco observations")
        self.log_message.emit(f"    - {num_identity_segments} identity support segments")
        self.log_message.emit(f"    - {num_trajectories} unique tracked bees")
        self.log_message.emit(f"    - {num_chamber_data} chamber frame records")
        self.log_message.emit(f"    - {num_pollen_data} pollen frame records")
        self._log_processor_timing(processor, process_elapsed)
        
        # Collect data BEFORE generating visualizations
        self.all_bee_detections.extend(processor.get_bee_detections())
        self.all_bee_interactions.extend(processor.get_bee_interactions())
        self.all_aruco_observations.extend(processor.get_aruco_observations())
        self.all_bee_identity_events.extend(processor.get_bee_identity_events())
        self.all_bee_identity_segments.extend(processor.get_bee_identity_segments())
        self.all_chamber_frame_data.extend(processor.get_chamber_frame_data())
        self.all_pollen_frame_data.extend(processor.get_pollen_frame_data())
        
        # Collect trajectories with composite keys (video_id, bee_id) to avoid collisions
        for bee_id, trajectory in processor.get_bee_trajectories().items():
            composite_key = (video_id, bee_id)
            self.all_bee_trajectories[composite_key] = trajectory
        
        # Accumulate masks for averaging (only if visualization disabled)
        # If visualization enabled, we'll handle masks during viz generation
        visualization_enabled = self._should_generate_visualization(video_idx)
        visualization_frame_limit = self._visualization_frame_limit() if visualization_enabled else None
        visualization_format = self.config.get('visualization_format', 'video')
        streaming_visualization_enabled = (
            visualization_enabled and visualization_format in {'video', 'frames'}
        )
        if not visualization_enabled or streaming_visualization_enabled:
            self._accumulate_masks(video_id, processor.get_hive_masks_by_frame(), processor.get_chambers_by_frame())
        
        # Generate visualization if requested (BEFORE deleting processor)
        if streaming_visualization_enabled and not self.should_stop:
            self._log_streaming_visualization_result(processor, streaming_visualization_path, partial=False)
        elif visualization_enabled and not self.should_stop:
            self._generate_visualizations(
                video_path, video_id, processor, partial=False,
                frame_limit=visualization_frame_limit,
            )
            # Accumulate masks after visualization (so we still have averaged masks for CSV)
            self._accumulate_masks(video_id, processor.get_hive_masks_by_frame(), processor.get_chambers_by_frame())
        else:
            if not self.config.get('save_visualizations', False):
                reason = "checkbox not enabled"
            elif not visualization_enabled:
                reason = f"interval skip for video {video_idx}"
            else:
                reason = "processing stopped"
            self._log_verbose(f"  Skipping visualizations ({reason})")
        
        # Reset tracker state to prevent memory buildup across videos
        if hasattr(tracker, 'reset'):
            tracker.reset()
        
        # Explicitly delete processor to free memory (especially mask storage)
        # CRITICAL: Must happen AFTER visualization generation
        self._log_verbose(f"  Freeing processor memory...")
        del processor
        
        # Clear GPU cache and run garbage collection to free memory
        self._log_verbose(f"  Cleaning up GPU memory...")
        gc.collect()
        
        if torch.cuda.is_available():
            # Log GPU memory before cleanup
            mem_allocated = torch.cuda.memory_allocated() / 1e9  # GB
            mem_reserved = torch.cuda.memory_reserved() / 1e9  # GB
            self._log_verbose(f"    Before cleanup: {mem_allocated:.2f} GB allocated, {mem_reserved:.2f} GB reserved")
            
            torch.cuda.empty_cache()
            
            # Log GPU memory after cleanup
            mem_allocated_after = torch.cuda.memory_allocated() / 1e9  # GB
            mem_reserved_after = torch.cuda.memory_reserved() / 1e9  # GB
            freed = mem_reserved - mem_reserved_after
            self._log_verbose(f"    After cleanup: {mem_allocated_after:.2f} GB allocated, {mem_reserved_after:.2f} GB reserved (freed {freed:.2f} GB)")

        return {
            'completed': True,
            'frame_count': processor_frame_count,
            'bee_detections': num_detections,
            'bee_interactions': num_interactions,
            'aruco_observations': num_aruco_observations,
            'bee_identity_events': num_identity_events,
            'bee_identity_segments': num_identity_segments,
            'bee_trajectories': num_trajectories,
            'chamber_frame_records': num_chamber_data,
            'pollen_frame_records': num_pollen_data,
            'elapsed_seconds': process_elapsed,
        }

    def _log_streaming_visualization_result(self, processor, output_path, partial=False):
        """Report the annotated visualization written during frame processing."""
        if output_path is None:
            return

        output_path = Path(output_path)
        frames_written = int(getattr(processor, 'streaming_visualization_frames_written', 0) or 0)
        output_format = self.config.get('visualization_format', 'video')
        if output_format == 'frames':
            label = "partial annotated frames" if partial else "annotated frames"
            if frames_written > 0 and output_path.exists():
                rel_path = output_path.relative_to(Path(self.config['output_folder']))
                self.log_message.emit(
                    f"  ✓ Saved {label} to: {rel_path}/ ({frames_written} frame(s))"
                )
                self.log_message.emit(f"    Output folder: {output_path}")
            else:
                self.log_message.emit(f"  ⚠️ Streaming annotated frames wrote no frames: {output_path}")
            return

        label = "partial annotated video" if partial else "annotated video"
        if frames_written > 0 and output_path.exists():
            self.log_message.emit(
                f"  ✓ Saved {label} to: annotated_videos/{output_path.name} "
                f"({frames_written} frame(s))"
            )
            self.log_message.emit(f"    Output video: {output_path}")
        else:
            self.log_message.emit(f"  ⚠️ Streaming annotated video wrote no frames: {output_path}")

    def _generate_visualizations(self, video_path, video_id, processor, partial=False, frame_limit=None):
        """Generate annotated visualization output for a completed or partial video."""
        output_format = self.config.get('visualization_format', 'video')
        if output_format not in {'video', 'frames'}:
            output_format = 'video'

        label = "partial visualization" if partial else "visualization"
        self.log_message.emit(f"  Generating {label}...")

        output_folder = Path(self.config['output_folder'])
        viz_base_folder = output_folder / 'visualizations'

        try:
            max_frame = processor.frame_count if partial else None
            if frame_limit is not None:
                max_frame = min(processor.frame_count, int(frame_limit))

            viz_gen = VisualizationGenerator(
                video_path=video_path,
                output_folder=viz_base_folder,
                video_id=video_id,
                bee_detections=processor.get_bee_detections(),
                chamber_frame_data=processor.get_chamber_frame_data(),
                chambers_by_frame=processor.get_chambers_by_frame(),
                hive_masks_by_frame=processor.get_hive_masks_by_frame(),
                bee_masks_by_frame=processor.get_bee_masks_by_frame(),
                pollen_masks_by_frame=processor.get_pollen_masks_by_frame(),
                aruco_markers_by_frame=processor.get_aruco_markers_by_frame(),
                max_frame=max_frame,
                log_callback=self.log_message.emit,
                verbose_output=self.verbose_output
            )

            if output_format == 'video':
                video_folder = output_folder / 'annotated_videos'
                suffix = "_partial_annotated.mp4" if partial else "_annotated.mp4"
                video_output_path = video_folder / f"{video_id}{suffix}"
                success = viz_gen.generate_video(video_output_path)
                if success:
                    self.log_message.emit(f"  ✓ Saved annotated video to: annotated_videos/{video_output_path.name}")
                    self.log_message.emit(f"    Output video: {video_output_path}")
                else:
                    self.log_message.emit(f"  ⚠️ Annotated video generation returned False")
            else:
                success = viz_gen.generate()
                frames_output_folder = viz_base_folder / video_id
                if success:
                    self.log_message.emit(f"  ✓ Saved annotated frames to: visualizations/{video_id}/")
                    self.log_message.emit(f"    Output folder: {frames_output_folder}")
                else:
                    self.log_message.emit(f"  ⚠️ Visualization generation returned False")

            del viz_gen

        except Exception as e:
            self.log_message.emit(f"  ⚠️ Visualization error: {str(e)}")
            import traceback
            self._log_verbose(traceback.format_exc())
    
    def _accumulate_masks(self, video_id: str, hive_masks_by_frame: Dict, chambers_by_frame: Dict):
        """
        Accumulate hive and chamber masks across frames for averaging
        
        Args:
            video_id: Video identifier
            hive_masks_by_frame: Dict[frame_number -> Dict[chamber_id -> mask]]
            chambers_by_frame: Dict[frame_number -> Dict[chamber_id -> chamber_info]]
        
        Note: If visualization is disabled, these dictionaries will be empty (store_masks=False)
        and this function will do nothing, which is the intended behavior for memory efficiency.
        """
        # Skip if no data (visualization disabled)
        if not hive_masks_by_frame and not chambers_by_frame:
            return
        
        # Accumulate hive masks
        for frame_number, hive_masks in hive_masks_by_frame.items():
            for chamber_id, mask in hive_masks.items():
                if mask is None:
                    continue

                mask_counts = (mask > 0).astype(np.uint16, copy=False)
                
                key = (video_id, chamber_id)
                
                if key not in self.accumulated_hive_masks:
                    # Initialize with zeros
                    self.accumulated_hive_masks[key] = {
                        'accumulated_mask': np.zeros_like(mask_counts, dtype=np.uint16),
                        'frame_count': 0,
                        'shape': mask.shape
                    }
                    # Track memory usage (estimate)
                    mask_size_mb = self.accumulated_hive_masks[key]['accumulated_mask'].nbytes / (1024 * 1024)
                    self.accumulated_data_size_mb += mask_size_mb
                
                # Add this frame's mask
                self.accumulated_hive_masks[key]['accumulated_mask'] += mask_counts
                self.accumulated_hive_masks[key]['frame_count'] += 1
        
        # Accumulate chamber masks
        for frame_number, chambers in chambers_by_frame.items():
            for chamber_id, chamber_info in chambers.items():
                mask = chamber_info.get('mask')
                centroid = chamber_info.get('centroid')
                
                if mask is None:
                    continue

                mask_counts = (mask > 0).astype(np.uint16, copy=False)
                
                key = (video_id, chamber_id)
                
                if key not in self.accumulated_chamber_masks:
                    # Initialize with zeros
                    self.accumulated_chamber_masks[key] = {
                        'accumulated_mask': np.zeros_like(mask_counts, dtype=np.uint16),
                        'frame_count': 0,
                        'shape': mask.shape,
                        'accumulated_centroid': np.array([0.0, 0.0], dtype=np.float32)
                    }
                    # Track memory usage (estimate)
                    mask_size_mb = self.accumulated_chamber_masks[key]['accumulated_mask'].nbytes / (1024 * 1024)
                    self.accumulated_data_size_mb += mask_size_mb
                
                # Add this frame's mask
                self.accumulated_chamber_masks[key]['accumulated_mask'] += mask_counts
                self.accumulated_chamber_masks[key]['frame_count'] += 1
                
                # Accumulate centroid
                if centroid is not None:
                    self.accumulated_chamber_masks[key]['accumulated_centroid'] += np.array(centroid, dtype=np.float32)
        
        # Log memory usage if it's getting large
        if self.accumulated_data_size_mb > 100:  # Over 100 MB
            self._log_verbose(f"  ⚠️  Accumulated mask data: ~{self.accumulated_data_size_mb:.1f} MB in memory")

    def _reset_streaming_csvs(self, include_manifest: bool = False):
        """Remove stale CSVs before a streaming batch export starts."""
        output_folder = Path(self.config['output_folder'])
        output_folder.mkdir(parents=True, exist_ok=True)
        filenames = list(self.STREAMING_CSVS)
        if include_manifest:
            filenames.append(self.STATUS_CSV)

        for filename in filenames:
            path = output_folder / filename
            if path.exists():
                path.unlink()

    def _finalize_csv_exports(self):
        """Flush any pending rows and write small final summary CSVs."""
        if self._has_pending_export_data():
            if not self._export_csvs(intermediate=True, append=True, clear_after=True):
                return False

        output_folder = Path(self.config['output_folder'])
        exporter = VideoInferenceExporter(output_folder)
        if self.temporal_hive_prior is not None:
            try:
                csv_path = exporter.export_temporal_hive_priors(self.temporal_hive_prior.summaries())
                file_size = csv_path.stat().st_size if csv_path.exists() else 0
                self.log_message.emit(f"  ✓ temporal_hive_priors.csv created ({file_size:,} bytes)")
            except Exception as e:
                self.log_message.emit(f"  ❌ Temporal hive prior export failed: {str(e)}")
                import traceback
                self._log_verbose(traceback.format_exc())
                return False

        self.log_message.emit(f"\n=== Summary ===")
        self.log_message.emit(f"Total bee detections: {self.total_bee_detections}")
        self.log_message.emit(f"Bee contact events: {self.total_bee_interactions}")
        self.log_message.emit(f"ArUco observations: {self.total_aruco_observations}")
        self.log_message.emit(f"Identity events: {self.total_bee_identity_events}")
        self.log_message.emit(f"Identity support segments: {self.total_bee_identity_segments}")
        self.log_message.emit(f"Unique bee trajectories: {self.total_bee_trajectories}")
        self.log_message.emit(f"Chamber frame records: {self.total_chamber_frame_records}")
        self.log_message.emit(f"Pollen frame records: {self.total_pollen_frame_records}")

        for filename in list(self.STREAMING_CSVS) + [self.STATUS_CSV]:
            path = output_folder / filename
            if path.exists():
                self.log_message.emit(f"  ✓ {filename} ({path.stat().st_size:,} bytes)")

        return True

    def _clear_pending_export_buffers(self):
        """Release per-video rows and mask accumulators after they are safely on disk."""
        self.all_bee_detections.clear()
        self.all_bee_interactions.clear()
        self.all_aruco_observations.clear()
        self.all_bee_identity_events.clear()
        self.all_bee_identity_segments.clear()
        self.all_chamber_frame_data.clear()
        self.all_pollen_frame_data.clear()
        self.all_bee_trajectories.clear()
        self.accumulated_hive_masks.clear()
        self.accumulated_chamber_masks.clear()
        self.accumulated_data_size_mb = 0
        gc.collect()

    def _has_pending_export_data(self) -> bool:
        """Return True when there is any per-video data waiting to be written."""
        return any((
            self.all_bee_detections,
            self.all_bee_interactions,
            self.all_aruco_observations,
            self.all_bee_identity_events,
            self.all_bee_identity_segments,
            self.all_chamber_frame_data,
            self.all_pollen_frame_data,
            self.all_bee_trajectories,
            self.accumulated_hive_masks,
            self.accumulated_chamber_masks,
        ))

    def _export_csvs(self, intermediate: bool = False, append: bool = False, clear_after: bool = False):
        """Export all CSV files
        
        Args:
            intermediate: If True, this is an intermediate export during processing
            append: Append rows to existing CSV files instead of rewriting them
            clear_after: Clear pending in-memory rows after a successful export
        """
        output_folder = Path(self.config['output_folder'])
        
        if intermediate:
            self.log_message.emit(f"  Exporting intermediate CSVs to: {output_folder}")
        else:
            self.log_message.emit(f"  Exporting final CSVs to: {output_folder}")
        
        # Check if we have any data to export
        if not self._has_pending_export_data():
            self._log_verbose("  No pending CSV rows to export")
            return True
        
        # Create exporter
        exporter = VideoInferenceExporter(output_folder)
        
        # Export all CSVs
        try:
            csv_paths = exporter.export_all(
                self.all_bee_detections,
                self.all_bee_interactions,
                self.all_aruco_observations,
                self.all_bee_identity_events,
                self.all_bee_identity_segments,
                self.all_bee_trajectories,
                self.all_chamber_frame_data,
                self.all_pollen_frame_data,
                self.accumulated_hive_masks,
                self.accumulated_chamber_masks,
                export_hive_detections=self.config.get('hive_model_path') is not None,
                export_pollen_detections=self.config.get('pollen_model_path') is not None,
                temporal_hive_prior_summaries=(
                    self.temporal_hive_prior.summaries()
                    if self.temporal_hive_prior is not None and not append
                    else None
                ),
                append=append
            )

            self.total_bee_detections += len(self.all_bee_detections)
            self.total_bee_interactions += len(self.all_bee_interactions)
            self.total_aruco_observations += len(self.all_aruco_observations)
            self.total_bee_identity_events += len(self.all_bee_identity_events)
            self.total_bee_identity_segments += len(self.all_bee_identity_segments)
            self.total_chamber_frame_records += len(self.all_chamber_frame_data)
            self.total_pollen_frame_records += len(self.all_pollen_frame_data)
            self.total_bee_trajectories += len(self.all_bee_trajectories)
            
            # Log results
            if not intermediate:
                for csv_name, csv_path in csv_paths.items():
                    file_size = csv_path.stat().st_size if csv_path.exists() else 0
                    self.log_message.emit(f"  ✓ {csv_name}.csv created ({file_size:,} bytes)")
            elif append:
                self._log_verbose(
                    f"  ✓ Flushed {len(self.all_bee_detections)} detections, "
                    f"{len(self.all_bee_interactions)} contact events, "
                    f"{len(self.all_aruco_observations)} ArUco observations, "
                    f"{len(self.all_pollen_frame_data)} pollen frame records to CSV"
                )

            if clear_after:
                self._clear_pending_export_buffers()
            return True
        except Exception as e:
            self.log_message.emit(f"  ❌ CSV export failed: {str(e)}")
            import traceback
            self._log_verbose(traceback.format_exc())
            return False
