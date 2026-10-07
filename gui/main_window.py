"""
Main application window
"""

import numpy as np
import cv2
import torch
from PyQt6.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QSplitter, QFileDialog, QMessageBox, QStatusBar,
                             QDockWidget, QListWidget, QToolBar, QLabel, 
                             QApplication, QDialog, QPushButton, QScrollArea,
                             QInputDialog, QSlider, QProgressDialog, QCheckBox)
from PyQt6.QtCore import Qt, pyqtSignal, QSettings, QThread, QTimer, QEvent
from PyQt6.QtGui import QAction, QKeySequence, QActionGroup
from pathlib import Path
from datetime import datetime
import queue
import time
import json

from .canvas import ImageCanvas
from .toolbar import AnnotationToolbar
from .yolo_toolbar import YOLOToolbar
from .yolo_instance_focused_toolbar import YOLOInstanceFocusedToolbar
from .yolo_bbox_toolbar import YOLOBBoxToolbar
from .hive_chamber_toolbar import HiveChamberToolbar
from .sam2_toolbar import SAM2Toolbar
from .sam2_training_dialog import SAM2TrainingConfigDialog
from .dialogs import VideoImportDialog, ProjectDialog, ProjectVideoImportDialog, TrainingMaskCopyDialog
from .training_dialog import TrainingConfigDialog, TrainingProgressDialog
from .validation_dialog import ValidationConfigDialog, ValidationProgressDialog
from .validation_worker import ValidationWorker
from .frame_level_validation_dialog import FrameLevelValidationConfigDialog, FrameLevelValidationProgressDialog
from .frame_level_validation_worker import FrameLevelValidationWorker
from .batch_inference_dialog import BatchInferenceConfigDialog, BatchInferenceProgressDialog
from .batch_inference_worker import BatchInferenceWorker
from .batch_video_inference_dialog import BatchVideoInferenceConfigDialog, BatchVideoInferenceProgressDialog
from .batch_video_inference_worker import BatchVideoInferenceWorker
from .tracking_visualization_dialog import TrackingVisualizationConfigDialog, TrackingVisualizationWorker
from .tracking_sequences_panel import TrackingSequencesPanel
from .project_sync_dialog import (ProjectSyncDialog, SyncProgressDialog, choose_contributor,
                                  setting_json, save_setting)
from .project_import_dialog import ProjectImportDialog
from core.project_sync import sync_project
from core.contributors import utc_now, provenance_records
from .tracking_validation_dialog import TrackingValidationConfigDialog, TrackingValidationProgressDialog
from .tracking_validation_worker import TrackingValidationWorker
from core.video_processor import VideoProcessor
from core.annotation import AnnotationManager
from core.categories import CATEGORIES, BROOD_CATEGORIES, BROOD_SHORT_LABELS, BROOD_MODEL_LABEL, category_label, training_categories
from gui.category_menu import add_category_menu_actions
from core.annotation_scope import split_annotations, merge_annotations, frame_categories
from core.annotation_copy import training_copy_targets, selected_copy_masks, merge_copied_masks
from core.project_manager import ProjectManager
from core.instance_tracker import InstanceTracker, Detection, Track
from core.frame_cache import FrameCache, PreloadWorker
from core.marker_detector import MarkerDetector
from core.tracking_sequence_manager import TrackingSequenceManager
from core.validation_review import (
    frame_has_bee_mask_annotation,
    video_has_selected_bee_masks,
)
from training.coco_video_export import export_coco_per_video
from training.yolo_trainer import YOLOTrainingWorker
from training.yolo_trainer_instance_focused import YOLOTrainingWorkerInstanceFocused
from training.yolo_trainer_bbox import YOLOTrainingWorkerBBox
from training.sam2_trainer import SAM2TrainingWorker


class SaveWorker(QThread):
    """Background worker thread for saving annotations"""
    
    error_occurred = pyqtSignal(str)
    save_started = pyqtSignal(int)  # Emits frame_idx when save starts
    save_completed = pyqtSignal(int)  # Emits frame_idx when save completes
    attribution_saved = pyqtSignal(str, str, int, object)
    
    def __init__(self, annotation_manager):
        super().__init__()
        self.annotation_manager = annotation_manager
        self.save_queue = queue.Queue()
        self.running = True
        self.failed_tasks = {}
        
    def add_save_task(self, project_path, video_id, frame_idx, annotations):
        """Queue a save task"""
        session = getattr(self.annotation_manager, 'contributor_session', None)
        self.save_queue.put((project_path, video_id, frame_idx, annotations,
                             dict(session) if session else None))
        
    def run(self):
        """Process save tasks in background"""
        while self.running:
            try:
                # Wait for save task with timeout
                try:
                    project_path, video_id, frame_idx, annotations, session = self.save_queue.get(timeout=0.1)
                except queue.Empty:
                    continue
                    
                # Perform the save
                try:
                    self.save_started.emit(frame_idx)
                    saved = self.annotation_manager.save_frame_annotations(
                        project_path, video_id, frame_idx, annotations, contributor=session
                    )
                    if session:
                        self.attribution_saved.emit(str(project_path), video_id, frame_idx,
                                                    provenance_records(saved))
                    self.failed_tasks.pop((str(project_path), video_id, frame_idx), None)
                    self.save_completed.emit(frame_idx)
                except Exception as e:
                    self.failed_tasks[(str(project_path), video_id, frame_idx)] = str(e)
                    self.error_occurred.emit(f"Save failed for frame {frame_idx}: {str(e)}")
                finally:
                    self.save_queue.task_done()
                    
            except Exception as e:
                self.error_occurred.emit(f"Worker error: {str(e)}")
                
    def wait_until_idle(self, raise_errors=True):
        """Finish queued and in-flight writes before synchronous annotation copies."""
        if self.save_queue.unfinished_tasks and (not self.running or not self.isRunning()):
            raise RuntimeError('The annotation save worker is not running.')
        self.save_queue.join()
        if raise_errors and self.failed_tasks:
            raise OSError('Some annotation saves failed: ' + '; '.join(self.failed_tasks.values()))

    def stop(self):
        """Stop the worker thread"""
        self.running = False


class MainWindow(QMainWindow):
    """Main application window"""
    
    def __init__(self, sam2_checkpoint=None, coarse_yolo_checkpoint=None, bbox_checkpoint=None, instance_focused_checkpoint=None):
        super().__init__()
        
        self.video_processor = VideoProcessor()
        self.annotation_manager = AnnotationManager(max_cache_size=2)  # Very small cache - annotations with masks are huge (~100MB/frame)
        self.project_manager = ProjectManager()
        self.tracking_sequence_manager = None  # Initialized when project is opened
        
        # Initialize ArUco/QR marker detector with debug mode
        # Debug images will be saved to project_path/marker_debug/ when available
        self.marker_detector = MarkerDetector(
            debug=True,  # Enable debug output and image saving
            debug_folder=None  # Will be set when project is opened
        )
        self.trainer = None  # Initialized when training starts
        self.current_video_id = None  # Track current video being annotated
        
        # Initialize background save worker
        self.save_worker = SaveWorker(self.annotation_manager)
        self.save_worker.error_occurred.connect(self.on_save_error)
        self.save_worker.save_started.connect(self.on_save_started)
        self.save_worker.save_completed.connect(self.on_save_completed)
        self.save_worker.attribution_saved.connect(self._receive_saved_attribution)
        self.save_worker.start()
        
        # Initialize frame cache with background preloading
        self.frame_cache = FrameCache(max_size=10)  # Cache up to 10 frames
        self.preload_worker = PreloadWorker(self.frame_cache, self.annotation_manager)
        self.preload_worker.start()
        
        # SAM2 will be initialized when user loads checkpoint via SAM2Toolbar
        self.sam2 = None
        
        # Store checkpoint paths for use in init_ui
        self.sam2_checkpoint = sam2_checkpoint
        self.coarse_yolo_checkpoint = coarse_yolo_checkpoint
        self.bbox_checkpoint = bbox_checkpoint
        self.instance_focused_checkpoint = instance_focused_checkpoint
        
        self.current_frame_idx = 0
        self.frames = []
        self.frame_video_ids = []  # Track which video each frame belongs to
        self.frame_splits = []  # Track split (train/val) for each frame
        self.frame_selected = []  # Track if frame is selected for train/val
        self.frame_list_to_frames_map = []  # Map list row to actual frame index
        self.project_path = None
        self.recent_projects = []
        self.split_filter = 'all'  # 'all', 'train', 'val', 'test', or 'inference'
        self.video_split_filter = 'all'
        self.video_next_mask_id = {}  # Track next_mask_id per video for unique IDs
        self.video_mask_colors = {}  # Track mask colors per video: {video_id: {mask_id: (r,g,b)}}
        self.video_physical_scales = {}  # {video_id: pixels_per_cm or None}
        self.video_scale_known_lengths_cm = {}  # Last calibration length per video
        self.imaging_setup_profiles = {}  # Project-level named calibration profiles
        self.video_imaging_setups = {}  # {video_id: setup_name}
        self._imaging_setups_loaded = False
        self.box_inference_mode = False  # Track if box inference mode is active
        self.annotation_mode = 'segmentation'  # 'segmentation' or ' bbox' - which annotation type to display/edit
        
        # Track unsaved changes to avoid unnecessary saves
        self.current_frame_modified = False
        self.dirty_frame_annotation_keys = set()
        self.dirty_video_annotation_ids = set()
        self.coco_export_dirty = False
        
        # Throttled instance list update (performance optimization)
        self._instance_list_update_timer = QTimer()
        self._instance_list_update_timer.setSingleShot(True)
        self._instance_list_update_timer.setInterval(100)  # Update at most every 100ms
        self._instance_list_update_timer.timeout.connect(self._do_update_instance_list)
        self._instance_list_update_pending = False
        
        # Play timer for automatic frame playback
        self.play_timer = QTimer()
        self.play_timer.timeout.connect(self._play_next_frame)
        self.is_playing = False
        
        # Initialize instance tracker for ID matching across frames
        self.tracker = InstanceTracker()
        self.video_trackers = {}  # Separate tracker state per video: {video_id: InstanceTracker}
        self.tracking_enabled = True  # Enable/disable tracking
        
        # Create default projects directory
        self.default_projects_dir = Path(__file__).parent.parent / 'projects'
        self.default_projects_dir.mkdir(exist_ok=True)
        
        self.init_ui()
        self.setup_shortcuts()
        self.load_settings()
        self.contributor_session = None
        self._sync_in_progress = False
    
    def on_save_error(self, error_msg):
        """Handle save errors from background thread"""
        print(f"Background save error: {error_msg}")
    
    def on_save_started(self, frame_idx):
        """Handle save started event"""
        self.status_label.setText(f"💾 Saving frame {frame_idx}...")
        self.status_label.setStyleSheet("color: orange; font-weight: bold;")
    
    def on_save_completed(self, frame_idx):
        """Handle save completed event"""
        # Show saved message briefly, then clear
        self.status_label.setText(f"✓ Frame {frame_idx} saved")
        self.status_label.setStyleSheet("color: green; font-weight: bold;")
        self._schedule_done_counter_update(saved_counts_dirty=True)
        # Auto-clear after 2 seconds
        from PyQt6.QtCore import QTimer
        QTimer.singleShot(2000, self._clear_save_status)
    
    def _clear_save_status(self):
        """Clear save status message and reset style"""
        if self.status_label.text().startswith("✓"):
            self.status_label.setText("Ready")
            self.status_label.setStyleSheet("")
        
    def init_ui(self):
        """Initialize the user interface"""
        self.setWindowTitle("BumbleBox Annotator")
        self.setGeometry(100, 100, 1400, 900)
        
        # Central widget with canvas
        self.canvas = ImageCanvas(self)
        self.canvas.point_clicked.connect(self.on_canvas_point_clicked)
        self.canvas.box_drawn.connect(self.on_canvas_box_drawn)
        self.canvas.masks_visibility_changed.connect(self.on_masks_visibility_changed)
        self.canvas.bboxes_visibility_changed.connect(self.on_bboxes_visibility_changed)
        self.canvas.instance_visibility_changed.connect(self.on_instance_visibility_shortcut_changed)
        self.canvas.instance_visibility_toggle_blocked.connect(self.on_instance_visibility_shortcut_blocked)
        self.canvas.instance_selected.connect(self.on_canvas_instance_selected)
        self.canvas.brush_size_step_requested.connect(self.on_brush_size_step_requested)
        self.canvas.brush_size_unit_step_requested.connect(self.on_brush_size_unit_step_requested)
        self.canvas.brush_eraser_toggle_requested.connect(self.on_brush_eraser_toggle_requested)
        self.canvas.instance_switch_tap_progress.connect(self.on_instance_switch_tap_progress)
        self.canvas.new_instance_requested.connect(self.new_instance)
        self.canvas.measurement_line_completed.connect(self.on_measurement_line_completed)
        self.canvas.annotation_changed.connect(self.on_annotation_changed)
        self.canvas.annotation_changed.connect(self._schedule_instance_list_update)
        self.canvas.setToolTip(
            "Space toggles masks and boxes | "
            "Shift+Space toggles selected instance | Ctrl+/- zooms | +/- changes brush size | "
            "Middle-click toggles Brush/Eraser | Triple-tap another instance to switch while brushing | "
            "No Draw Zone: click vertices, double-click/Enter closes, Backspace/right-click undoes | "
            "Measure: drag a ruler line, then set its known length in cm | "
            "Press F to fit image to window"
        )
        
        # Create annotation toolbar
        self.toolbar = AnnotationToolbar(self)
        self.toolbar.tool_changed.connect(self.on_tool_changed)
        self.toolbar.brush_size_changed.connect(self.canvas.set_brush_size)
        self.toolbar.brush_cursor_preview_changed.connect(self.canvas.set_brush_cursor_preview_enabled)
        self.toolbar.mask_opacity_changed.connect(self.canvas.set_mask_opacity)
        self.toolbar.clear_instance_requested.connect(self.clear_selected_instance)
        self.toolbar.clear_no_draw_zones_requested.connect(self.on_clear_no_draw_zones)
        self.toolbar.set_measurement_scale_requested.connect(self.on_set_measurement_scale)
        self.toolbar.clear_measurement_scale_requested.connect(self.on_clear_measurement_scale)
        self.toolbar.clear_measurement_line_requested.connect(self.canvas.clear_measurement_line)
        self.toolbar.imaging_setup_changed.connect(self.on_imaging_setup_changed)
        self.toolbar.apply_imaging_setup_requested.connect(
            self.on_imaging_setup_changed
        )
        self.toolbar.new_imaging_setup_requested.connect(self.on_new_imaging_setup)
        self.toolbar.new_instance_requested.connect(self.new_instance)
        self.toolbar.detect_aruco_requested.connect(self.on_detect_aruco_in_bees)
        self.toolbar.clear_all_aruco_requested.connect(self.on_clear_all_aruco_tracking)
        self.toolbar.delete_all_requested.connect(self.delete_all_instances)
        self.toolbar.show_segmentations_changed.connect(self.on_show_segmentations_changed)
        self.toolbar.show_bboxes_changed.connect(self.on_show_bboxes_changed)
        self.toolbar.annotation_type_changed.connect(self.on_annotation_type_changed)
        self.toolbar.annotation_type_visibility_changed.connect(self.on_annotation_type_visibility_changed)
        self.toolbar.annotation_group_visibility_changed.connect(self.on_annotation_group_visibility_changed)
        
        # Synchronize canvas visibility with toolbar checkbox initial states
        # (in case signals fired before connections were made)
        self.canvas.show_segmentations = self.toolbar.segmentation_checkbox.isChecked()
        self.canvas.masks_visible = self.canvas.show_segmentations
        self.canvas.show_bboxes = self.toolbar.bbox_checkbox.isChecked()
        self.canvas.set_annotation_type_visibility('bee', self.toolbar.show_bees_checkbox.isChecked(), rebuild=False)
        self.canvas.set_annotation_type_visibility('hive', self.toolbar.show_hives_checkbox.isChecked(), rebuild=False)
        self.canvas.set_annotation_type_visibility('chamber', self.toolbar.show_chambers_checkbox.isChecked(), rebuild=False)
        self.canvas.set_annotation_type_visibility('pollen', self.toolbar.show_pollen_checkbox.isChecked(), rebuild=False)
        self.canvas.set_annotation_type_visibility('nectar', self.toolbar.show_nectar_checkbox.isChecked(), rebuild=False)
        self.canvas.set_brush_cursor_preview_enabled(self.toolbar.brush_cursor_checkbox.isChecked())
        self.toolbar.set_measurement_scale(None)
        
        # Create SAM2 toolbar
        if self.sam2_checkpoint:
            print(f"Loading SAM2 checkpoint from command line: {self.sam2_checkpoint}")
        self.sam2_toolbar = SAM2Toolbar(self, checkpoint_path=self.sam2_checkpoint)
        self.sam2_toolbar.tool_changed.connect(self.on_tool_changed)
        self.sam2_toolbar.clear_prompts_requested.connect(self.on_clear_prompts_requested)
        self.sam2_toolbar.propagate_requested.connect(self.propagate_to_next_frame)
        self.sam2_toolbar.propagate_to_selected_requested.connect(self.propagate_to_next_selected)
        self.sam2_toolbar.unload_requested.connect(self.on_sam2_unload)
        self.sam2_toolbar.sam2_loaded.connect(self.on_sam2_loaded)
        self.sam2_toolbar.finetune_requested.connect(self.on_sam2_finetune_requested)
        self.sam2_toolbar.run_on_bbox_requested.connect(self.on_sam2_run_on_bbox)
        self.sam2_toolbar.hide_checked_toolbars_requested.connect(self.hide_checked_visible_top_toolbars)
        
        # Create YOLO toolbar
        if self.coarse_yolo_checkpoint:
            print(f"Loading coarse YOLO checkpoint from command line: {self.coarse_yolo_checkpoint}")
        self.yolo_toolbar = YOLOToolbar(self, checkpoint_path=self.coarse_yolo_checkpoint)
        self.yolo_toolbar.inference_requested.connect(self.run_yolo_inference)
        self.yolo_toolbar.track_from_last_requested.connect(self.track_from_last_frame)
        self.yolo_toolbar.propagate_requested.connect(self.propagate_yolo)
        self.yolo_toolbar.process_video_requested.connect(self.process_entire_video_yolo)
        
        # Create YOLO instance-focused toolbar
        if self.instance_focused_checkpoint:
            print(f"Loading instance-focused YOLO checkpoint from command line: {self.instance_focused_checkpoint}")
        self.yolo_instance_focused_toolbar = YOLOInstanceFocusedToolbar(self, checkpoint_path=self.instance_focused_checkpoint)
        self.yolo_instance_focused_toolbar.refine_requested.connect(self.refine_selected_instance_focused)
        self.yolo_instance_focused_toolbar.refine_all_requested.connect(self.refine_all_instances_focused)
        
        # Create YOLO BBox toolbar
        if self.bbox_checkpoint:
            print(f"Loading YOLO BBox checkpoint from command line: {self.bbox_checkpoint}")
        self.yolo_bbox_toolbar = YOLOBBoxToolbar(self, checkpoint_path=self.bbox_checkpoint)
        self.yolo_bbox_toolbar.inference_requested.connect(self.run_yolo_bbox_inference)
        self.yolo_bbox_toolbar.track_from_last_requested.connect(self.track_from_last_frame)
        self.yolo_bbox_toolbar.propagate_requested.connect(self.propagate_yolo_bbox)
        
        # Create Hive & Chamber toolbar
        self.hive_chamber_toolbar = HiveChamberToolbar(self)
        self.hive_chamber_toolbar.hive_inference_requested.connect(self.run_hive_inference)
        self.hive_chamber_toolbar.chamber_inference_requested.connect(self.run_chamber_inference)
        self.hive_chamber_toolbar.pollen_inference_requested.connect(self.run_pollen_inference)
        self.hive_chamber_toolbar.nectar_inference_requested.connect(self.run_nectar_inference)
        self.hive_chamber_toolbar.both_inference_requested.connect(self.run_hive_chamber_both)
        
        # Create layout
        central_widget = QWidget()
        layout = QVBoxLayout(central_widget)
        layout.addWidget(self.toolbar)
        layout.addWidget(self.sam2_toolbar)
        layout.addWidget(self.yolo_toolbar)
        layout.addWidget(self.yolo_instance_focused_toolbar)
        layout.addWidget(self.yolo_bbox_toolbar)
        layout.addWidget(self.hive_chamber_toolbar)
        layout.addWidget(self.canvas)
        
        # Create frame navigation controls at the bottom
        nav_layout = QHBoxLayout()

        self.prev_frame_button = QPushButton("◀ Previous")
        self.prev_frame_button.setMaximumWidth(110)
        self.prev_frame_button.setEnabled(False)
        self.prev_frame_button.clicked.connect(self.prev_frame)
        self.prev_frame_button.setToolTip("Go to previous frame (A or Up)")
        nav_layout.addWidget(self.prev_frame_button)
        
        # Play/Pause button
        self.play_button = QPushButton("▶ Play")
        self.play_button.setMaximumWidth(100)
        self.play_button.clicked.connect(self.toggle_play)
        self.play_button.setToolTip("Play through frames automatically")
        nav_layout.addWidget(self.play_button)

        self.next_frame_button = QPushButton("Next ▶")
        self.next_frame_button.setMaximumWidth(110)
        self.next_frame_button.setEnabled(False)
        self.next_frame_button.clicked.connect(self.next_frame)
        self.next_frame_button.setToolTip("Go to next frame (D or Down)")
        nav_layout.addWidget(self.next_frame_button)
        
        # Frame slider
        self.frame_slider = QSlider(Qt.Orientation.Horizontal)
        self.frame_slider.setMinimum(0)
        self.frame_slider.setMaximum(0)
        self.frame_slider.setValue(0)
        self.frame_slider.setEnabled(False)
        self.frame_slider.valueChanged.connect(self.on_slider_changed)
        self.frame_slider.setToolTip("Scroll through frames")
        nav_layout.addWidget(self.frame_slider)
        
        layout.addLayout(nav_layout)
        self.setCentralWidget(central_widget)
        
        # Hide YOLO toolbars by default
        self.yolo_toolbar.hide()
        self.yolo_instance_focused_toolbar.hide()
        self.yolo_bbox_toolbar.hide()
        self.hive_chamber_toolbar.hide()
        
        # Create menu bar
        self.create_menus()
        
        # Create video list dock (sidebar)
        self.create_video_list_dock()
        
        # Create frame list dock
        self.create_frame_list_dock()
        
        # Create instance list dock
        self.create_instance_list_dock()
        
        # Create tracking sequences dock
        self.create_tracking_sequences_dock()
        
        # Status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_label = QLabel("Ready")
        self.status_bar.addPermanentWidget(self.status_label)
        self.contributor_label = QLabel('Contributor: not selected')
        self.contributor_label.setTextFormat(Qt.TextFormat.PlainText)
        self.contributor_label.setMaximumWidth(280)
        self.status_bar.addWidget(self.contributor_label)
        self._init_done_counter_overlay()

    def _init_done_counter_overlay(self):
        """Create a small automatic instance counter over the canvas viewport."""
        self.done_counter_categories = CATEGORIES
        self.done_counter_saved_counts = self._zero_annotation_counts()
        self.done_counter_session_baseline_counts = self._zero_annotation_counts()
        self.done_counter_saved_counts_dirty = True

        self.done_counter_overlay = QWidget(self.canvas.viewport())
        self.done_counter_overlay.setObjectName("doneCounterOverlay")
        self.done_counter_overlay.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.done_counter_overlay.setFocusPolicy(Qt.FocusPolicy.NoFocus)

        overlay_layout = QVBoxLayout(self.done_counter_overlay)
        overlay_layout.setContentsMargins(10, 8, 10, 8)
        overlay_layout.setSpacing(2)

        self.done_counter_title_label = QLabel("Instances made")
        self.done_counter_title_label.setObjectName("doneCounterTitle")
        self.done_counter_summary_label = QLabel()
        self.done_counter_category_labels = {}

        for label in (self.done_counter_title_label, self.done_counter_summary_label):
            label.setFocusPolicy(Qt.FocusPolicy.NoFocus)
            overlay_layout.addWidget(label)

        for category in self.done_counter_categories:
            label = QLabel()
            label.setFocusPolicy(Qt.FocusPolicy.NoFocus)
            self.done_counter_category_labels[category] = label
            overlay_layout.addWidget(label)

        self.done_counter_overlay.setStyleSheet("""
            QWidget#doneCounterOverlay {
                background-color: rgba(24, 28, 34, 200);
                border: 1px solid rgba(255, 255, 255, 80);
                border-radius: 8px;
            }
            QLabel {
                color: rgb(245, 247, 250);
                font-size: 12px;
                font-weight: 600;
            }
            QLabel#doneCounterTitle {
                color: rgb(255, 246, 170);
            }
        """)

        self._refresh_done_counter_labels()
        self.done_counter_overlay.adjustSize()
        self.done_counter_overlay.show()
        self.canvas.viewport().installEventFilter(self)
        self._position_done_counter_overlay()

        self.done_counter_update_timer = QTimer(self)
        self.done_counter_update_timer.setSingleShot(True)
        self.done_counter_update_timer.setInterval(300)
        self.done_counter_update_timer.timeout.connect(self._refresh_done_counter_labels)

    def _position_done_counter_overlay(self):
        if not hasattr(self, 'done_counter_overlay') or self.done_counter_overlay is None:
            return
        self.done_counter_overlay.adjustSize()
        viewport_rect = self.canvas.viewport().rect()
        x = viewport_rect.width() - self.done_counter_overlay.width() - 18
        y = 18
        x = max(8, x)
        y = max(8, y)
        self.done_counter_overlay.move(x, y)
        self.done_counter_overlay.raise_()

    def _zero_annotation_counts(self):
        categories = getattr(self, 'done_counter_categories', CATEGORIES)
        return {category: 0 for category in categories}

    def _add_annotation_counts(self, base_counts, delta_counts, sign=1):
        counts = self._zero_annotation_counts()
        for category in counts:
            counts[category] = (
                int(base_counts.get(category, 0)) +
                sign * int(delta_counts.get(category, 0))
            )
        return counts

    def _read_annotation_records(self, json_path):
        if not json_path or not Path(json_path).exists():
            return []
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError) as e:
            print(f"Warning: failed to read annotation counts from {json_path}: {e}")
            return []

        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            return data.get('annotations', [])
        return []

    def _count_annotation_records(self, records, bbox_only=False):
        counts = self._zero_annotation_counts()
        seen = set()

        for index, annotation in enumerate(records or []):
            if not isinstance(annotation, dict):
                continue
            if bbox_only and (
                not annotation.get('bbox_only', False) or
                annotation.get('from_mask', False)
            ):
                continue

            category = annotation.get('category', 'bee')
            if category not in counts:
                continue

            instance_id = annotation.get(
                'mask_id',
                annotation.get('instance_id', annotation.get('id'))
            )
            key = (category, instance_id) if instance_id is not None else (category, index)
            if key in seen:
                continue

            seen.add(key)
            counts[category] += 1

        return counts

    def _saved_frame_annotation_counts(self, video_id, frame_idx):
        counts = self._zero_annotation_counts()
        if not self.project_path or video_id is None or frame_idx is None:
            return counts

        json_dir = Path(self.project_path) / 'annotations' / 'json' / str(video_id)
        bbox_dir = Path(self.project_path) / 'annotations' / 'bbox' / str(video_id)
        frame_name = f'frame_{int(frame_idx):06d}.json'

        counts = self._add_annotation_counts(
            counts,
            self._count_annotation_records(self._read_annotation_records(json_dir / frame_name))
        )
        counts = self._add_annotation_counts(
            counts,
            self._count_annotation_records(
                self._read_annotation_records(bbox_dir / frame_name),
                bbox_only=True
            )
        )
        return counts

    def _saved_video_annotation_counts(self, video_id):
        if not self.project_path or not video_id:
            return self._zero_annotation_counts()

        json_path = (
            Path(self.project_path) / 'annotations' / 'json' /
            str(video_id) / 'video_annotations.json'
        )
        return self._count_annotation_records(self._read_annotation_records(json_path))

    def _scan_saved_annotation_counts(self):
        counts = self._zero_annotation_counts()
        if not self.project_path:
            return counts

        project_path = Path(self.project_path)
        json_root = project_path / 'annotations' / 'json'
        bbox_root = project_path / 'annotations' / 'bbox'

        if json_root.exists():
            for video_dir in sorted(path for path in json_root.iterdir() if path.is_dir()):
                for frame_json in sorted(video_dir.glob('frame_*.json')):
                    counts = self._add_annotation_counts(
                        counts,
                        self._count_annotation_records(self._read_annotation_records(frame_json))
                    )
                video_json = video_dir / 'video_annotations.json'
                counts = self._add_annotation_counts(
                    counts,
                    self._count_annotation_records(self._read_annotation_records(video_json))
                )

        if bbox_root.exists():
            for video_dir in sorted(path for path in bbox_root.iterdir() if path.is_dir()):
                for frame_json in sorted(video_dir.glob('frame_*.json')):
                    counts = self._add_annotation_counts(
                        counts,
                        self._count_annotation_records(
                            self._read_annotation_records(frame_json),
                            bbox_only=True
                        )
                    )

        return counts

    def _live_canvas_annotation_counts(self):
        counts = self._zero_annotation_counts()
        if not hasattr(self, 'canvas') or self.canvas is None:
            return counts

        try:
            entries = self.canvas.get_instance_entries()
        except Exception as e:
            print(f"Warning: failed to count live canvas instances: {e}")
            return counts

        for entry in entries:
            category = entry.get('category', 'bee')
            if category in counts:
                counts[category] += 1
        return counts

    def _current_annotation_counts(self):
        if getattr(self, 'done_counter_saved_counts_dirty', True):
            self.done_counter_saved_counts = self._scan_saved_annotation_counts()
            self.done_counter_saved_counts_dirty = False

        counts = dict(getattr(self, 'done_counter_saved_counts', self._zero_annotation_counts()))

        if self.project_path and self.current_video_id:
            frame_idx = None
            if self.current_frame_idx is not None and self.current_frame_idx >= 0:
                try:
                    frame_idx = self._get_frame_idx_in_video(self.current_frame_idx)
                except Exception:
                    frame_idx = self.current_frame_idx

            if frame_idx is not None:
                counts = self._add_annotation_counts(
                    counts,
                    self._saved_frame_annotation_counts(self.current_video_id, frame_idx),
                    sign=-1
                )

            counts = self._add_annotation_counts(
                counts,
                self._saved_video_annotation_counts(self.current_video_id),
                sign=-1
            )
            counts = self._add_annotation_counts(
                counts,
                self._live_canvas_annotation_counts()
            )

        return {
            category: max(0, int(counts.get(category, 0)))
            for category in self._zero_annotation_counts()
        }

    def _category_display_name(self, category):
        return {
            'bee': 'Bees',
            'hive': 'Hives',
            'chamber': 'Chambers',
            'pollen': 'Pollen',
        }.get(category, category_label(category))

    def _reset_done_counter_baseline(self):
        self.done_counter_saved_counts_dirty = True
        current_counts = self._current_annotation_counts()
        self.done_counter_session_baseline_counts = dict(current_counts)
        self._refresh_done_counter_labels()

    def _schedule_done_counter_update(self, saved_counts_dirty=False):
        if saved_counts_dirty:
            self.done_counter_saved_counts_dirty = True

        timer = getattr(self, 'done_counter_update_timer', None)
        if timer is not None:
            timer.start()
        else:
            self._refresh_done_counter_labels()

    def _refresh_done_counter_labels(self):
        counts = self._current_annotation_counts()
        baseline_counts = getattr(
            self,
            'done_counter_session_baseline_counts',
            self._zero_annotation_counts()
        )
        session_counts = {
            category: max(0, counts.get(category, 0) - baseline_counts.get(category, 0))
            for category in self._zero_annotation_counts()
        }

        project_total = sum(counts.values())
        session_total = sum(session_counts.values())

        if hasattr(self, 'done_counter_summary_label'):
            self.done_counter_summary_label.setText(
                f"Project: {project_total} (+{session_total} session)"
            )

        if hasattr(self, 'done_counter_category_labels'):
            for category, label in self.done_counter_category_labels.items():
                label.setVisible(category not in BROOD_CATEGORIES or counts.get(category, 0) > 0)
                label.setText(
                    f"{self._category_display_name(category)}: "
                    f"{counts.get(category, 0)} (+{session_counts.get(category, 0)})"
                )

        if hasattr(self, 'done_counter_overlay'):
            self._position_done_counter_overlay()

    def eventFilter(self, obj, event):
        if (
            hasattr(self, 'canvas') and
            obj is self.canvas.viewport() and
            event.type() in (QEvent.Type.Resize, QEvent.Type.Show)
        ):
            self._position_done_counter_overlay()
        return super().eventFilter(obj, event)

    def create_menus(self):
        """Create menu bar"""
        menubar = self.menuBar()
        
        # File menu
        file_menu = menubar.addMenu("&File")
        
        new_project_action = QAction("&New Project...", self)
        new_project_action.setShortcut(QKeySequence.StandardKey.New)
        new_project_action.triggered.connect(self.new_project)
        file_menu.addAction(new_project_action)
        
        open_project_action = QAction("&Open Project...", self)
        open_project_action.setShortcut(QKeySequence.StandardKey.Open)
        open_project_action.triggered.connect(self.open_project)
        file_menu.addAction(open_project_action)

        self.recent_projects_menu = file_menu.addMenu("Recent Projects")
        self.update_recent_projects_menu()
        
        file_menu.addSeparator()
        
        add_video_action = QAction("Add &Video to Project...", self)
        add_video_action.setShortcut("Ctrl+I")
        add_video_action.triggered.connect(self.show_add_video_dialog)
        file_menu.addAction(add_video_action)

        import_project_action = QAction('Import From Project...', self)
        import_project_action.triggered.connect(self.import_project_data)
        file_menu.addAction(import_project_action)
        
        file_menu.addSeparator()
        
        save_action = QAction("&Save", self)
        save_action.setShortcut(QKeySequence.StandardKey.Save)
        save_action.triggered.connect(self.save_annotations)
        file_menu.addAction(save_action)

        self.sync_project_action = QAction('Sync Project Now...', self)
        self.sync_project_action.triggered.connect(self.sync_current_project)
        file_menu.addAction(self.sync_project_action)
        
        export_coco_action = QAction("Export &COCO Format...", self)
        export_coco_action.triggered.connect(self.export_coco_format)
        file_menu.addAction(export_coco_action)
        
        export_video_action = QAction("Export Annotations as &Video...", self)
        export_video_action.triggered.connect(self.export_annotations_as_video)
        file_menu.addAction(export_video_action)
        
        export_all_action = QAction("Export / &Visualize All Annotations...", self)
        export_all_action.triggered.connect(self.export_visualize_all_annotations)
        file_menu.addAction(export_all_action)
        
        file_menu.addSeparator()
        
        exit_action = QAction("E&xit", self)
        exit_action.setShortcut(QKeySequence.StandardKey.Quit)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        # Edit menu
        edit_menu = menubar.addMenu("&Edit")
        
        undo_action = QAction("&Undo", self)
        undo_action.setShortcut(QKeySequence.StandardKey.Undo)
        undo_action.triggered.connect(self.undo)
        edit_menu.addAction(undo_action)
        
        redo_action = QAction("&Redo", self)
        redo_action.setShortcut(QKeySequence.StandardKey.Redo)
        redo_action.triggered.connect(self.redo)
        edit_menu.addAction(redo_action)
        
        edit_menu.addSeparator()
        
        detect_markers_action = QAction("Detect &Markers (ArUco/QR)", self)
        detect_markers_action.setShortcut("Ctrl+M")
        detect_markers_action.setToolTip("Detect ArUco and QR code markers on annotated bees")
        detect_markers_action.triggered.connect(self.on_detect_markers_manually)
        edit_menu.addAction(detect_markers_action)
        
        # Model menu
        model_menu = menubar.addMenu("&Model")
        
        sam2_prompt_action = QAction("SAM2 &Prompt Mode", self)
        sam2_prompt_action.setShortcut("Ctrl+P")
        sam2_prompt_action.triggered.connect(self.enable_sam2_prompt_mode)
        model_menu.addAction(sam2_prompt_action)
        
        model_menu.addSeparator()
        
        # YOLO training
        train_yolo_action = QAction("Train YOLO Model...", self)
        train_yolo_action.setShortcut("Ctrl+T")
        train_yolo_action.triggered.connect(self.train_yolo_model)
        model_menu.addAction(train_yolo_action)
        
        train_yolo_instance_focused_action = QAction("Train Instance-Focused Model...", self)
        train_yolo_instance_focused_action.triggered.connect(self.train_yolo_model_instance_focused)
        model_menu.addAction(train_yolo_instance_focused_action)
        
        train_yolo_bbox_action = QAction("Train BBox Detection Model...", self)
        train_yolo_bbox_action.triggered.connect(self.train_yolo_bbox_model)
        model_menu.addAction(train_yolo_bbox_action)
        
        predict_action = QAction("Create Tracking &Visualization Video...", self)
        predict_action.setShortcut("Ctrl+R")
        predict_action.triggered.connect(self.run_inference)
        model_menu.addAction(predict_action)
        
        model_menu.addSeparator()
        
        # Tracking toggle
        self.tracking_action = QAction("Enable Instance &Tracking", self)
        self.tracking_action.setCheckable(True)
        self.tracking_action.setChecked(True)  # Enabled by default
        self.tracking_action.setToolTip("Use ID tracking to match detections across frames")
        self.tracking_action.triggered.connect(self.toggle_tracking)
        model_menu.addAction(self.tracking_action)
        
        model_menu.addSeparator()
        
        # Frame-level validation analysis
        frame_level_validation_action = QAction("Analyze Frame Level &Validation...", self)
        frame_level_validation_action.setToolTip("Analyze predictions vs ground truth for validation frames")
        frame_level_validation_action.triggered.connect(self.analyze_frame_level_validation)
        model_menu.addAction(frame_level_validation_action)
        
        # Batch image inference
        batch_inference_action = QAction("Batch Image &Inference...", self)
        batch_inference_action.setToolTip("Run inference on arbitrary PNG images from folders")
        batch_inference_action.triggered.connect(self.batch_image_inference)
        model_menu.addAction(batch_inference_action)
        
        # Batch video inference with tracking
        batch_video_inference_action = QAction("Batch &Video Inference with Tracking...", self)
        batch_video_inference_action.setToolTip("Process videos with bee tracking, ArUco detection, and spatial analysis")
        batch_video_inference_action.triggered.connect(self.batch_video_inference)
        model_menu.addAction(batch_video_inference_action)
        
        # Tracking algorithm validation
        tracking_validation_action = QAction("Validate &Tracking Algorithms...", self)
        tracking_validation_action.setToolTip("Test and compare tracking algorithms on annotated sequences")
        tracking_validation_action.triggered.connect(self.validate_tracking_algorithms)
        model_menu.addAction(tracking_validation_action)
        
        # View menu
        view_menu = menubar.addMenu("&View")
        
        # Toggle video sidebar
        self.toggle_video_sidebar_action = QAction("Hide Video &List", self)
        self.toggle_video_sidebar_action.setCheckable(True)
        self.toggle_video_sidebar_action.setChecked(True)
        self.toggle_video_sidebar_action.setShortcut("Ctrl+L")
        self.toggle_video_sidebar_action.triggered.connect(self.toggle_video_sidebar)
        view_menu.addAction(self.toggle_video_sidebar_action)
        
        view_menu.addSeparator()
        
        # Annotation view toggles (non-mutually exclusive)
        self.segmentation_mode_action = QAction("Show &Segmentations", self)
        self.segmentation_mode_action.setCheckable(True)
        self.segmentation_mode_action.setChecked(True)  # Segmentation visible by default
        self.segmentation_mode_action.setShortcut("Ctrl+Shift+S")
        self.segmentation_mode_action.setToolTip("Show/hide segmentation masks")
        self.segmentation_mode_action.triggered.connect(self.on_menu_show_segmentations_changed)
        view_menu.addAction(self.segmentation_mode_action)
        
        self.bbox_mode_action = QAction("Show &Bounding Boxes", self)
        self.bbox_mode_action.setCheckable(True)
        self.bbox_mode_action.setChecked(True)
        self.bbox_mode_action.setShortcut("Ctrl+Shift+B")
        self.bbox_mode_action.setToolTip("Show/hide bounding box annotations")
        self.bbox_mode_action.triggered.connect(self.on_menu_show_bboxes_changed)
        view_menu.addAction(self.bbox_mode_action)
        
        view_menu.addSeparator()
        
        zoom_in_action = QAction("Zoom &In", self)
        zoom_in_action.setShortcut(QKeySequence.StandardKey.ZoomIn)
        zoom_in_action.triggered.connect(self.canvas.zoom_in)
        view_menu.addAction(zoom_in_action)
        
        zoom_out_action = QAction("Zoom &Out", self)
        zoom_out_action.setShortcut(QKeySequence.StandardKey.ZoomOut)
        zoom_out_action.triggered.connect(self.canvas.zoom_out)
        view_menu.addAction(zoom_out_action)
        
        fit_action = QAction("&Fit to Window", self)
        fit_action.setShortcut("Ctrl+0")
        fit_action.triggered.connect(self.canvas.fit_to_window)
        view_menu.addAction(fit_action)
        
        # Toolbars menu
        toolbars_menu = menubar.addMenu("&Toolbars")
        
        # Annotation Toolbar toggle
        self.annotation_toolbar_action = QAction("&Annotation Toolbar", self)
        self.annotation_toolbar_action.setCheckable(True)
        self.annotation_toolbar_action.setChecked(True)
        self.annotation_toolbar_action.triggered.connect(self.toggle_annotation_toolbar)
        toolbars_menu.addAction(self.annotation_toolbar_action)
        
        # SAM2 Toolbar toggle
        self.sam2_toolbar_action = QAction("&SAM2 Toolbar", self)
        self.sam2_toolbar_action.setCheckable(True)
        self.sam2_toolbar_action.setChecked(True)
        self.sam2_toolbar_action.triggered.connect(self.toggle_sam2_toolbar)
        toolbars_menu.addAction(self.sam2_toolbar_action)
        
        # YOLO Coarse Toolbar toggle
        self.yolo_toolbar_action = QAction("YOLO &Coarse Toolbar", self)
        self.yolo_toolbar_action.setCheckable(True)
        self.yolo_toolbar_action.setChecked(False)  # Hidden by default
        self.yolo_toolbar_action.triggered.connect(self.toggle_yolo_toolbar)
        toolbars_menu.addAction(self.yolo_toolbar_action)
        
        # YOLO Instance-Focused Toolbar toggle
        self.yolo_instance_focused_toolbar_action = QAction("YOLO &Instance-Focused Toolbar", self)
        self.yolo_instance_focused_toolbar_action.setCheckable(True)
        self.yolo_instance_focused_toolbar_action.setChecked(False)  # Hidden by default
        self.yolo_instance_focused_toolbar_action.triggered.connect(self.toggle_yolo_instance_focused_toolbar)
        toolbars_menu.addAction(self.yolo_instance_focused_toolbar_action)
        
        # YOLO BBox Toolbar toggle
        self.yolo_bbox_toolbar_action = QAction("YOLO B&Box Toolbar", self)
        self.yolo_bbox_toolbar_action.setCheckable(True)
        self.yolo_bbox_toolbar_action.setChecked(False)  # Hidden by default
        self.yolo_bbox_toolbar_action.triggered.connect(self.toggle_yolo_bbox_toolbar)
        toolbars_menu.addAction(self.yolo_bbox_toolbar_action)
        
        # Hive & Chamber Toolbar toggle
        self.hive_chamber_toolbar_action = QAction("&Hive, Chamber, Pollen && Nectar Toolbar", self)
        self.hive_chamber_toolbar_action.setCheckable(True)
        self.hive_chamber_toolbar_action.setChecked(False)  # Hidden by default
        self.hive_chamber_toolbar_action.triggered.connect(self.toggle_hive_chamber_toolbar)
        toolbars_menu.addAction(self.hive_chamber_toolbar_action)

        toolbars_menu.addSeparator()

        hide_visible_toolbars_action = QAction("&Hide Checked Visible Toolbars", self)
        hide_visible_toolbars_action.triggered.connect(self.hide_checked_visible_top_toolbars)
        toolbars_menu.addAction(hide_visible_toolbars_action)

        settings_menu = menubar.addMenu('&Settings')
        contributor_action = QAction('Change Contributor...', self)
        contributor_action.triggered.connect(self.change_contributor)
        settings_menu.addAction(contributor_action)
        sync_action = QAction('Project Sync...', self)
        sync_action.triggered.connect(self.configure_project_sync)
        settings_menu.addAction(sync_action)

    def set_contributor(self, session):
        self.contributor_session = dict(session)
        self.annotation_manager.contributor_session = dict(session)
        self.contributor_label.setText('Contributor: ' + session['name'])
        self.contributor_label.setToolTip('Active contributor: ' + session['name'])

    def _receive_saved_attribution(self, project, video, frame, records):
        if (str(self.project_path) != project or self.current_video_id != video or
                (frame >= 0 and frame != self._get_frame_idx_in_video(self.current_frame_idx))):
            return
        for record in records:
            metadata = self.canvas.annotation_metadata.get(record['mask_id'])
            if metadata is not None and metadata.get('category', 'bee') == record['category']:
                metadata['provenance'] = record['provenance']

    def change_contributor(self):
        try:
            # Finish the previous person's edits before changing the session identity.
            self.save_worker.wait_until_idle(raise_errors=False)
            if self.project_path:
                self._save_annotation_sources()
            self.save_worker.wait_until_idle()
        except Exception as exc:
            QMessageBox.warning(self, 'Save Before Switching Contributor', str(exc))
            return
        session = choose_contributor(self)
        if session:
            self.set_contributor(session)

    def offer_sync_setup(self):
        settings = QSettings()
        if settings.value('project_sync/setup_offered', False, type=bool):
            return
        question = QMessageBox(self)
        question.setWindowTitle('Optional Project Sync')
        question.setText('Keep a separate copy of your annotation projects in an external or shared folder?')
        setup = question.addButton('Set Up Sync', QMessageBox.ButtonRole.AcceptRole)
        question.addButton('Not Now', QMessageBox.ButtonRole.RejectRole)
        question.exec()
        settings.setValue('project_sync/setup_offered', True)
        if question.clickedButton() is setup:
            self.configure_project_sync()

    def configure_project_sync(self):
        dialog = ProjectSyncDialog(self.project_path, self)
        if dialog.exec() == QDialog.DialogCode.Accepted and dialog.sync_requested:
            self.sync_current_project()

    def sync_current_project(self):
        if self._sync_in_progress:
            return
        if not self.project_path:
            QMessageBox.information(self, 'No Project', 'Open a project before syncing.')
            return
        if not self.contributor_session:
            self.change_contributor()
            if not self.contributor_session:
                return
        source = str(Path(self.project_path).resolve())
        projects = setting_json('project_sync/projects', {})
        configuration = setting_json('project_sync/destination', {})
        if not projects.get(source, {}).get('enabled') or not configuration.get('root_id'):
            dialog = ProjectSyncDialog(self.project_path, self)
            if dialog.exec() != QDialog.DialogCode.Accepted:
                return
            projects = setting_json('project_sync/projects', {})
            configuration = setting_json('project_sync/destination', {})
            if not projects.get(source, {}).get('enabled'):
                return
        if self.is_playing:
            self.toggle_play()
        self._sync_in_progress = True
        try:
            self.save_worker.wait_until_idle(raise_errors=False)
            self._save_annotation_sources(force=True, refresh_next_id=True)
            self.save_worker.wait_until_idle()
            session = dict(self.contributor_session)
            result = SyncProgressDialog(
                'Sync Project',
                lambda progress, cancelled: sync_project(
                    source, configuration, session, progress, cancelled), self).run()
            projects[source].update(last_success=utc_now(), last_error='', last_path=result['path'])
            self.status_label.setText(f"Sync complete: {result['copied_files']} changed files")
            QMessageBox.information(self, 'Sync Complete',
                                    'Verified project copy:\n' + result['path'])
        except Exception as exc:
            projects.setdefault(source, {})['last_error'] = str(exc)
            self.status_label.setText('Sync failed; local annotations remain saved')
            QMessageBox.warning(self, 'Sync Not Completed', str(exc))
        finally:
            save_setting('project_sync/projects', projects)
            self._sync_in_progress = False
    
    def toggle_annotation_toolbar(self):
        """Toggle visibility of annotation toolbar"""
        if self.annotation_toolbar_action.isChecked():
            self.toolbar.show()
        else:
            self.toolbar.hide()
    
    def toggle_sam2_toolbar(self):
        """Toggle visibility of SAM2 toolbar"""
        if self.sam2_toolbar_action.isChecked():
            self.sam2_toolbar.show()
        else:
            self.sam2_toolbar.hide()
    
    def toggle_yolo_toolbar(self):
        """Toggle visibility of YOLO coarse toolbar"""
        if self.yolo_toolbar_action.isChecked():
            self.yolo_toolbar.show()
        else:
            self.yolo_toolbar.hide()
    
    def toggle_yolo_instance_focused_toolbar(self):
        """Toggle visibility of YOLO instance-focused toolbar"""
        if self.yolo_instance_focused_toolbar_action.isChecked():
            self.yolo_instance_focused_toolbar.show()
        else:
            self.yolo_instance_focused_toolbar.hide()
    
    def toggle_yolo_bbox_toolbar(self):
        """Toggle visibility of YOLO BBox toolbar"""
        if self.yolo_bbox_toolbar_action.isChecked():
            self.yolo_bbox_toolbar.show()
        else:
            self.yolo_bbox_toolbar.hide()
    
    def toggle_hive_chamber_toolbar(self):
        """Toggle visibility of Hive & Chamber toolbar"""
        if self.hive_chamber_toolbar_action.isChecked():
            self.hive_chamber_toolbar.show()
        else:
            self.hive_chamber_toolbar.hide()

    def _optional_top_toolbar_entries(self):
        """Return optional top toolbars controlled by the Toolbars menu."""
        return [
            ("SAM2", self.sam2_toolbar, self.sam2_toolbar_action),
            ("YOLO coarse", self.yolo_toolbar, self.yolo_toolbar_action),
            (
                "YOLO instance-focused",
                self.yolo_instance_focused_toolbar,
                self.yolo_instance_focused_toolbar_action
            ),
            ("YOLO bbox", self.yolo_bbox_toolbar, self.yolo_bbox_toolbar_action),
            ("Hive/Chamber/Pollen/Nectar", self.hive_chamber_toolbar, self.hive_chamber_toolbar_action),
        ]

    def hide_checked_visible_top_toolbars(self):
        """Hide optional top toolbars that are both checked and currently visible."""
        hidden_labels = []
        for label, toolbar, action in self._optional_top_toolbar_entries():
            if action.isChecked() and toolbar.isVisible():
                action.setChecked(False)
                toolbar.hide()
                hidden_labels.append(label)

        if hidden_labels:
            self.status_label.setText(f"Hidden toolbars: {', '.join(hidden_labels)}")
        else:
            self.status_label.setText("No checked optional toolbars are currently visible")
        self.canvas.setFocus()
    
    def toggle_video_sidebar(self):
        """Toggle visibility of video sidebar"""
        if self.video_dock.isVisible():
            self.video_dock.hide()
            self.toggle_video_sidebar_action.setText("Show Video &List")
        else:
            self.video_dock.show()
            self.toggle_video_sidebar_action.setText("Hide Video &List")
    
    def on_video_dock_visibility_changed(self, visible):
        """Update menu action when dock visibility changes"""
        if visible:
            self.toggle_video_sidebar_action.setText("Hide Video &List")
            self.toggle_video_sidebar_action.setChecked(True)
        else:
            self.toggle_video_sidebar_action.setText("Show Video &List")
            self.toggle_video_sidebar_action.setChecked(False)
    
    def toggle_tracking(self):
        """Toggle instance tracking on/off"""
        self.tracking_enabled = self.tracking_action.isChecked()
        status = "enabled" if self.tracking_enabled else "disabled"
        self.status_label.setText(f"Instance tracking {status}")
        print(f"Instance tracking {status}")
    
    def set_annotation_mode(self, mode):
        """DEPRECATED: Annotation view is now controlled by show_segmentations and show_bboxes flags.
        
        This method is kept for backward compatibility but no longer changes the view mode.
        Use segmentation_mode_action and bbox_mode_action menu items or toolbar checkboxes instead.
        
        Args:
            mode: 'segmentation' or 'bbox' (for legacy purposes)
        """
        if mode not in ['segmentation', 'bbox']:
            print(f"Warning: Invalid annotation mode '{mode}'")
            return
        
        # Update the internal mode for backward compatibility with new_instance() etc
        self.annotation_mode = mode
        print(f"Note: set_annotation_mode is deprecated. Use view toggles instead.")
    
    def _get_or_create_tracker(self, video_id):
        """Get or create tracker for a specific video"""
        if video_id not in self.video_trackers:
            tracker = InstanceTracker()
            # Set next track ID from existing annotations for this video
            if video_id in self.video_next_mask_id:
                tracker.set_next_track_id(self.video_next_mask_id[video_id])
            self.video_trackers[video_id] = tracker
        return self.video_trackers[video_id]

    def _get_current_bee_bbox_annotations(self):
        """Get bbox-only bee annotations currently shown on the canvas."""
        annotations = self.canvas.get_annotations()
        bbox_annotations = []
        for ann in annotations:
            if ann.get('category', 'bee') != 'bee':
                continue
            if not ann.get('bbox_only', False):
                continue
            bbox = ann.get('bbox')
            if not bbox or bbox == [0, 0, 0, 0]:
                continue
            bbox_annotations.append(ann)
        return bbox_annotations

    def _box_iou_xyxy(self, box_a, box_b):
        """Calculate IoU between two xyxy boxes."""
        ax1, ay1, ax2, ay2 = box_a
        bx1, by1, bx2, by2 = box_b

        inter_x1 = max(ax1, bx1)
        inter_y1 = max(ay1, by1)
        inter_x2 = min(ax2, bx2)
        inter_y2 = min(ay2, by2)

        inter_w = max(0, inter_x2 - inter_x1)
        inter_h = max(0, inter_y2 - inter_y1)
        inter_area = inter_w * inter_h

        area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
        area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
        union = area_a + area_b - inter_area
        return inter_area / union if union > 0 else 0.0

    def _match_bbox_detections_to_existing_annotations(self, detections, existing_annotations, min_iou=0.1):
        """Preserve existing IDs when rerunning bbox detection on the same frame."""
        existing_boxes = []
        max_existing_id = 0

        for ann in existing_annotations:
            instance_id = int(ann.get('mask_id', ann.get('instance_id', 0)))
            bbox = ann.get('bbox')
            if instance_id <= 0 or not bbox:
                continue
            x, y, w, h = bbox
            existing_boxes.append((instance_id, [x, y, x + w, y + h]))
            max_existing_id = max(max_existing_id, instance_id)

        candidates = []
        for det_idx, detection in enumerate(detections):
            for existing_idx, (_, existing_box) in enumerate(existing_boxes):
                iou = self._box_iou_xyxy(detection.bbox, existing_box)
                if iou >= min_iou:
                    candidates.append((iou, det_idx, existing_idx))

        candidates.sort(reverse=True, key=lambda item: item[0])
        assigned_detections = {}
        used_existing = set()

        for _, det_idx, existing_idx in candidates:
            if det_idx in assigned_detections or existing_idx in used_existing:
                continue
            assigned_detections[det_idx] = existing_boxes[existing_idx][0]
            used_existing.add(existing_idx)

        next_id = max_existing_id + 1
        matched = []
        for det_idx, detection in enumerate(detections):
            if det_idx in assigned_detections:
                instance_id = assigned_detections[det_idx]
            else:
                instance_id = next_id
                next_id += 1
            matched.append((detection, instance_id))

        return matched, next_id
    
    def _yolo_results_to_detections(self, yolo_result, model):
        """Convert YOLO results to Detection objects"""
        detections = []
        if yolo_result.masks is not None:
            masks = yolo_result.masks.data.cpu().numpy()
            boxes = yolo_result.boxes.xyxy.cpu().numpy()
            confidences = yolo_result.boxes.conf.cpu().numpy()
            class_ids = yolo_result.boxes.cls.cpu().numpy() if yolo_result.boxes.cls is not None else None
            
            # Get original image shape from result
            orig_shape = yolo_result.orig_shape  # (height, width)
            
            for i in range(len(masks)):
                # With retina_masks=True, masks are already at original image size
                mask = masks[i]
                
                # Validate shape (should already match with retina_masks=True)
                if mask.shape != orig_shape:
                    print(f"WARNING: Mask shape {mask.shape} != orig_shape {orig_shape}, using as-is")
                
                # Convert to uint8
                mask = (mask * 255).astype(np.uint8)
                
                # Get bbox
                bbox = boxes[i]
                # Create Detection
                detection = Detection(
                    bbox=bbox.tolist(),
                    mask=mask,
                    confidence=float(confidences[i]),
                    source='yolo',
                    class_id=int(class_ids[i]) if class_ids is not None else 0
                )
                detections.append(detection)
        return detections
    
    def _clean_detections_contours(self, detections, overlap_threshold=0.15):
        """Remove overlapping contours from individual detections BEFORE merging
        
        This handles cases where a detection has picked up contours from nearby detections.
        We clean individual detection masks first, then merge duplicates.
        
        Only removes smaller disconnected contours, never the main (largest) contour.
        
        Args:
            detections: List of Detection objects with individual masks
            overlap_threshold: Threshold for mask overlap (intersection / contour_area).
            
        Returns:
            List of cleaned Detection objects
        """
        import cv2
        
        if len(detections) <= 1:
            return detections
        
        print(f"Cleaning contours from {len(detections)} individual detections...")
        
        # Extract contours and metadata for each detection
        detection_data = []
        for det in detections:
            mask = det.mask
            if not isinstance(mask, np.ndarray) or not np.any(mask > 0):
                detection_data.append({'contours': [], 'main_idx': None, 'main_center': None, 'bbox': det.bbox})
                continue
            
            # Find all contours for this detection
            binary_mask = (mask > 0).astype(np.uint8)
            contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Get bbox
            y_coords, x_coords = np.where(mask > 0)
            bbox = [x_coords.min(), y_coords.min(), x_coords.max(), y_coords.max()]
            
            # Store contours with their areas and centers
            contours_with_data = []
            max_area = 0
            max_area_idx = 0
            
            for idx, contour in enumerate(contours):
                area = cv2.contourArea(contour)
                if area > 10:  # Filter out very small noise contours
                    # Compute contour center
                    M = cv2.moments(contour)
                    if M['m00'] != 0:
                        cx = M['m10'] / M['m00']
                        cy = M['m01'] / M['m00']
                    else:
                        cx, cy = contour[0][0]
                    
                    contours_with_data.append((contour, area, (cx, cy)))
                    if area > max_area:
                        max_area = area
                        max_area_idx = len(contours_with_data) - 1
            
            # Store main contour center
            main_center = contours_with_data[max_area_idx][2] if contours_with_data else None
            
            detection_data.append({
                'contours': contours_with_data,
                'main_idx': max_area_idx if contours_with_data else None,
                'main_center': main_center,
                'bbox': bbox
            })
        
        # Track which contours to remove from which detections
        contours_to_remove = {i: [] for i in range(len(detections))}
        total_removed = 0
        
        # Compare contours across different detections
        for i in range(len(detections)):
            data_i = detection_data[i]
            if not data_i['contours']:
                continue
            
            main_idx_i = data_i['main_idx']
            if main_idx_i is None:
                continue
            
            main_area_i = data_i['contours'][main_idx_i][1]
            main_center_i = data_i['main_center']
            
            for contour_idx, (contour_i, area_i, center_i) in enumerate(data_i['contours']):
                if contour_idx in contours_to_remove[i]:
                    continue
                
                # NEVER remove the main contour
                if contour_idx == main_idx_i:
                    continue
                
                # Only consider removing contours significantly smaller than main
                if area_i > 0.5 * main_area_i:
                    continue
                
                # Calculate distance from main contour
                if main_center_i:
                    dist_from_main = np.sqrt((center_i[0] - main_center_i[0])**2 + 
                                            (center_i[1] - main_center_i[1])**2)
                else:
                    dist_from_main = 0
                
                # Check against all other detections
                should_remove = False
                removal_reason = ""
                overlapping_detection = -1
                
                for j in range(len(detections)):
                    if i == j:
                        continue
                    
                    data_j = detection_data[j]
                    mask_j = detections[j].mask
                    binary_mask_j = (mask_j > 0).astype(np.uint8)
                    bbox_j = data_j['bbox']
                    center_j = data_j['main_center']
                    
                    # Create temp mask for this contour
                    h, w = detections[i].mask.shape
                    contour_mask = np.zeros((h, w), dtype=np.uint8)
                    cv2.drawContours(contour_mask, [contour_i], 0, 255, -1)
                    
                    # Check 1: Mask overlap
                    intersection = np.logical_and(contour_mask > 0, binary_mask_j > 0).sum()
                    if intersection > 0:
                        overlap_ratio = intersection / area_i
                        if overlap_ratio > overlap_threshold:
                            should_remove = True
                            removal_reason = f"mask overlap {overlap_ratio:.1%}"
                            overlapping_detection = j
                            break
                    
                    # Check 2: Bbox containment
                    if bbox_j:
                        x1_j, y1_j, x2_j, y2_j = bbox_j
                        in_bbox_j = np.logical_and(
                            contour_mask > 0,
                            np.logical_and(
                                np.logical_and(np.arange(w)[None, :] >= x1_j, np.arange(w)[None, :] <= x2_j),
                                np.logical_and(np.arange(h)[:, None] >= y1_j, np.arange(h)[:, None] <= y2_j)
                            )
                        ).sum()
                        bbox_containment = in_bbox_j / area_i if area_i > 0 else 0
                        
                        if bbox_containment > 0.7:  # 70% in another detection's bbox
                            should_remove = True
                            removal_reason = f"bbox containment {bbox_containment:.1%}"
                            overlapping_detection = j
                            break
                    
                    # Check 3: Spatial proximity
                    if center_j and dist_from_main > 50:
                        dist_to_j = np.sqrt((center_i[0] - center_j[0])**2 + 
                                          (center_i[1] - center_j[1])**2)
                        if dist_to_j < dist_from_main * 0.5 and area_i < 0.2 * main_area_i:
                            should_remove = True
                            removal_reason = f"spatial proximity (dist to j={dist_to_j:.0f}, dist from main={dist_from_main:.0f})"
                            overlapping_detection = j
                            break
                
                if should_remove:
                    contours_to_remove[i].append(contour_idx)
                    total_removed += 1
                    print(f"  Removing contour {contour_idx} from detection {i}: {removal_reason} (overlaps detection {overlapping_detection})")
        
        # Rebuild masks for detections that had contours removed
        cleaned_detections = []
        for i, det in enumerate(detections):
            if not contours_to_remove[i]:
                # No contours to remove, keep original
                cleaned_detections.append(det)
            else:
                # Rebuild mask without the removed contours
                h, w = det.mask.shape
                new_mask = np.zeros((h, w), dtype=np.uint8)
                
                kept_contours = 0
                for contour_idx, (contour, area, center) in enumerate(detection_data[i]['contours']):
                    if contour_idx not in contours_to_remove[i]:
                        cv2.drawContours(new_mask, [contour], 0, 255, -1)
                        kept_contours += 1
                
                # Only keep the detection if it still has contours
                if kept_contours > 0:
                    # Create new detection with cleaned mask
                    cleaned_det = Detection(
                        bbox=det.bbox,
                        mask=new_mask,
                        confidence=det.confidence,
                        source=det.source,
                        class_id=det.class_id
                    )
                    cleaned_detections.append(cleaned_det)
                    print(f"  Detection {i}: kept {kept_contours}/{len(detection_data[i]['contours'])} contours")
                else:
                    print(f"  Detection {i}: removed entirely (no contours left)")
        
        if total_removed > 0:
            print(f"✓ Removed {total_removed} overlapping contours from individual detections")
            print(f"✓ {len(detections)} detections → {len(cleaned_detections)} after cleaning")
        else:
            print(f"✓ No overlapping contours found in individual detections")
        
        return cleaned_detections
    
    def _remove_duplicate_detections(self, detections, iou_threshold=0.5):
        """Merge duplicate detections based on IoU of masks
        
        Args:
            detections: List of Detection objects
            iou_threshold: IoU threshold for considering detections as duplicates
            
        Returns:
            List of Detection objects with duplicates merged
        """
        if len(detections) <= 1:
            return detections
        
        # Sort by confidence (highest first) to process in order of confidence
        detections = sorted(detections, key=lambda d: d.confidence if d.confidence is not None else 0.0, reverse=True)
        
        # Track which detections have been merged (to skip them later)
        merged_indices = set()
        merged_detections = []
        
        for i, det_i in enumerate(detections):
            if i in merged_indices:
                continue  # Already merged into another detection
            
            # Start with the current detection
            merged_mask = det_i.mask.copy() > 0
            max_confidence = det_i.confidence if det_i.confidence is not None else 0.0
            merged_with = [i]
            
            # Check for duplicates to merge
            for j, det_j in enumerate(detections[i+1:], start=i+1):
                if j in merged_indices:
                    continue
                
                # Calculate IoU between masks
                intersection = np.logical_and(det_i.mask > 0, det_j.mask > 0).sum()
                union = np.logical_or(det_i.mask > 0, det_j.mask > 0).sum()
                
                if union > 0:
                    iou = intersection / union
                    if iou > iou_threshold:
                        # Merge this detection
                        merged_mask = np.logical_or(merged_mask, det_j.mask > 0)
                        if det_j.confidence is not None:
                            max_confidence = max(max_confidence, det_j.confidence)
                        merged_with.append(j)
                        merged_indices.add(j)
            
            # Create merged detection
            merged_mask_uint8 = (merged_mask * 255).astype(np.uint8)
            
            # Compute bbox from merged mask
            if np.any(merged_mask):
                y_indices, x_indices = np.where(merged_mask)
                bbox = np.array([
                    float(x_indices.min()),
                    float(y_indices.min()),
                    float(x_indices.max()),
                    float(y_indices.max())
                ])
            else:
                # Use original bbox if mask is somehow empty
                bbox = det_i.bbox
            
            # Create the merged detection object
            merged_det = Detection(
                bbox=bbox,
                mask=merged_mask_uint8,
                confidence=max_confidence,
                source=det_i.source,
                class_id=det_i.class_id
            )
            merged_detections.append(merged_det)
            
            # Log if we merged multiple detections
            if len(merged_with) > 1:
                print(f"Merged {len(merged_with)} overlapping detections (IoU > {iou_threshold})")
        
        return merged_detections
    
    def _clean_duplicate_contours(self, annotations, overlap_threshold=0.15):
        """Remove duplicate contours from instances where a contour overlaps heavily with another instance
        
        This handles cases where an instance has picked up contours from other bees.
        Uses multiple criteria: mask overlap, bbox containment, and spatial distance.
        
        Only removes smaller disconnected contours, never the main (largest) contour.
        
        Args:
            annotations: List of annotation dicts with 'mask' key
            overlap_threshold: Threshold for mask overlap (intersection / contour_area). 
                             If a contour from instance A has overlap > threshold with instance B,
                             remove it from A. Default 0.15 (15% overlap).
            
        Returns:
            List of cleaned annotations
        """
        import cv2
        
        if len(annotations) <= 1:
            return annotations
        
        print(f"Cleaning duplicate contours from {len(annotations)} instances...")
        
        # Extract contours, compute centers and bboxes for each instance
        instance_contours = []
        main_contour_indices = []
        main_contour_centers = []
        instance_bboxes = []
        
        for ann in annotations:
            mask = ann['mask']
            if not isinstance(mask, np.ndarray):
                instance_contours.append([])
                main_contour_indices.append(None)
                main_contour_centers.append(None)
                instance_bboxes.append(None)
                continue
            
            # Find all contours for this instance
            binary_mask = (mask > 0).astype(np.uint8)
            contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Compute bbox from mask
            y_coords, x_coords = np.where(mask > 0)
            if len(y_coords) > 0:
                bbox = [x_coords.min(), y_coords.min(), x_coords.max(), y_coords.max()]
            else:
                bbox = [0, 0, 0, 0]
            instance_bboxes.append(bbox)
            
            # Store contours with their areas and centers
            contours_with_areas = []
            max_area = 0
            max_area_idx = 0
            
            for idx, contour in enumerate(contours):
                area = cv2.contourArea(contour)
                if area > 10:  # Filter out very small noise contours
                    # Compute contour center
                    M = cv2.moments(contour)
                    if M['m00'] != 0:
                        cx = M['m10'] / M['m00']
                        cy = M['m01'] / M['m00']
                    else:
                        cx, cy = contour[0][0]
                    
                    contours_with_areas.append((contour, area, (cx, cy)))
                    if area > max_area:
                        max_area = area
                        max_area_idx = len(contours_with_areas) - 1
            
            instance_contours.append(contours_with_areas)
            main_contour_indices.append(max_area_idx if contours_with_areas else None)
            
            # Store main contour center
            if contours_with_areas and max_area_idx < len(contours_with_areas):
                main_contour_centers.append(contours_with_areas[max_area_idx][2])
            else:
                main_contour_centers.append(None)
        
        # Track which contours to remove from which instances
        contours_to_remove = {i: [] for i in range(len(annotations))}
        
        total_removed = 0
        
        # Compare contours across different instances
        for i in range(len(annotations)):
            if not instance_contours[i]:
                continue
            
            main_idx = main_contour_indices[i]
            if main_idx is None:
                continue
            
            main_area = instance_contours[i][main_idx][1]
            main_center_i = main_contour_centers[i]
            
            for contour_idx, (contour_i, area_i, center_i) in enumerate(instance_contours[i]):
                if contour_idx in contours_to_remove[i]:
                    continue
                
                # NEVER remove the main contour
                if contour_idx == main_idx:
                    continue
                
                # Only consider removing contours that are significantly smaller than main
                if area_i > 0.5 * main_area:
                    continue
                
                # Calculate distance from main contour
                if main_center_i:
                    dist_from_main = np.sqrt((center_i[0] - main_center_i[0])**2 + 
                                            (center_i[1] - main_center_i[1])**2)
                else:
                    dist_from_main = 0
                
                # Check against all other instances
                should_remove = False
                removal_reason = ""
                overlapping_instance = -1
                
                for j in range(len(annotations)):
                    if i == j:
                        continue
                    
                    # Get instance j's data
                    mask_j = annotations[j]['mask']
                    binary_mask_j = (mask_j > 0).astype(np.uint8)
                    bbox_j = instance_bboxes[j]
                    center_j = main_contour_centers[j]
                    
                    # Create temp mask for this contour
                    h, w = annotations[i]['mask'].shape
                    contour_mask = np.zeros((h, w), dtype=np.uint8)
                    cv2.drawContours(contour_mask, [contour_i], 0, 255, -1)
                    
                    # Check 1: Mask overlap
                    intersection = np.logical_and(contour_mask > 0, binary_mask_j > 0).sum()
                    if intersection > 0:
                        overlap_ratio = intersection / area_i
                        if overlap_ratio > overlap_threshold:
                            should_remove = True
                            removal_reason = f"mask overlap {overlap_ratio:.1%}"
                            overlapping_instance = j
                            break
                    
                    # Check 2: Bbox containment - is this contour mostly in instance j's bbox?
                    if bbox_j:
                        x1_j, y1_j, x2_j, y2_j = bbox_j
                        # Check what fraction of contour pixels are in bbox j
                        in_bbox_j = np.logical_and(
                            contour_mask > 0,
                            np.logical_and(
                                np.logical_and(np.arange(w)[None, :] >= x1_j, np.arange(w)[None, :] <= x2_j),
                                np.logical_and(np.arange(h)[:, None] >= y1_j, np.arange(h)[:, None] <= y2_j)
                            )
                        ).sum()
                        bbox_containment = in_bbox_j / area_i if area_i > 0 else 0
                        
                        if bbox_containment > 0.7:  # 70% in another instance's bbox
                            should_remove = True
                            removal_reason = f"bbox containment {bbox_containment:.1%}"
                            overlapping_instance = j
                            break
                    
                    # Check 3: Spatial proximity - is contour center much closer to instance j?
                    if center_j and dist_from_main > 50:  # Only if reasonably far from own main contour
                        dist_to_j = np.sqrt((center_i[0] - center_j[0])**2 + 
                                          (center_i[1] - center_j[1])**2)
                        # If contour is 2x closer to j than to own main contour
                        if dist_to_j < dist_from_main * 0.5 and area_i < 0.2 * main_area:
                            should_remove = True
                            removal_reason = f"spatial proximity (dist to j={dist_to_j:.0f}, dist from main={dist_from_main:.0f})"
                            overlapping_instance = j
                            break
                
                if should_remove:
                    contours_to_remove[i].append(contour_idx)
                    total_removed += 1
                    print(f"  Removing contour {contour_idx} from instance {i}: {removal_reason} (area={area_i:.0f}px, {area_i/main_area:.1%} of main, to instance {overlapping_instance})")
        
        # Rebuild masks for instances that had contours removed
        cleaned_annotations = []
        for i, ann in enumerate(annotations):
            if not contours_to_remove[i]:
                # No contours to remove, keep original
                cleaned_annotations.append(ann)
            else:
                # Rebuild mask without the removed contours
                h, w = ann['mask'].shape
                new_mask = np.zeros((h, w), dtype=np.uint8)
                
                kept_contours = 0
                for contour_idx, (contour, area, center) in enumerate(instance_contours[i]):
                    if contour_idx not in contours_to_remove[i]:
                        cv2.drawContours(new_mask, [contour], 0, 255, -1)
                        kept_contours += 1
                
                # Only keep the instance if it still has contours
                if kept_contours > 0:
                    # Create new annotation with cleaned mask
                    cleaned_ann = ann.copy()
                    cleaned_ann['mask'] = new_mask
                    cleaned_ann['area'] = int(np.sum(new_mask > 0))
                    cleaned_annotations.append(cleaned_ann)
                    print(f"  Instance {i}: kept {kept_contours}/{len(instance_contours[i])} contours")
                else:
                    print(f"  Instance {i}: removed entirely (no contours left)")
        
        if total_removed > 0:
            print(f"✓ Removed {total_removed} duplicate contours, {len(annotations) - len(cleaned_annotations)} instances removed entirely")
        else:
            print(f"✓ No duplicate contours found")
        
        return cleaned_annotations
    
    def _sahi_results_to_detections(self, sahi_result, frame_path):
        """Convert SAHI results to Detection objects"""
        import cv2
        detections = []
        
        # Load original image to get dimensions
        img = cv2.imread(str(frame_path))
        if img is None:
            return detections
        
        img_h, img_w = img.shape[:2]
        
        # Process each object prediction
        for obj_pred in sahi_result.object_prediction_list:
            # Get bbox (in SAHI format: [x1, y1, x2, y2])
            bbox = obj_pred.bbox.to_xyxy()
            
            # Get mask if available
            if hasattr(obj_pred, 'mask') and obj_pred.mask is not None:
                # SAHI provides masks as boolean arrays
                mask_bool = obj_pred.mask.bool_mask
                
                # Convert to uint8 mask
                mask = (mask_bool * 255).astype(np.uint8)
                
                # Resize mask to match image dimensions if needed
                if mask.shape != (img_h, img_w):
                    mask = cv2.resize(mask, (img_w, img_h), interpolation=cv2.INTER_NEAREST)
            else:
                # If no mask, create one from bbox
                mask = np.zeros((img_h, img_w), dtype=np.uint8)
                x1, y1, x2, y2 = [int(coord) for coord in bbox]
                mask[y1:y2, x1:x2] = 255
            
            # Create Detection object
            detection = Detection(
                bbox=[float(x) for x in bbox],
                mask=mask,
                confidence=float(obj_pred.score.value),
                source='sahi',
                class_id=int(obj_pred.category.id)
            )
            detections.append(detection)
        
        return detections
    
    def _annotations_to_detections(self, annotations):
        """Convert annotations (from canvas or annotation manager) to Detection objects"""
        detections = []
        for ann in annotations:
            # Handle both mask-based and bbox-only annotations
            mask = ann.get('mask')
            bbox_data = ann.get('bbox')
            
            # Need either mask or bbox
            if mask is None and bbox_data is None:
                continue
            
            # Compute or get bbox
            if mask is not None and np.any(mask > 0):
                # Compute bbox from mask
                y_indices, x_indices = np.where(mask > 0)
                bbox = np.array([
                    float(x_indices.min()),
                    float(y_indices.min()),
                    float(x_indices.max()),
                    float(y_indices.max())
                ])
            elif bbox_data is not None:
                # Use provided bbox (convert from [x, y, w, h] to [x1, y1, x2, y2] if needed)
                if len(bbox_data) == 4:
                    x, y, w, h = bbox_data
                    bbox = np.array([float(x), float(y), float(x + w), float(y + h)])
                else:
                    # Already in [x1, y1, x2, y2] format
                    bbox = np.array(bbox_data, dtype=float)
            else:
                # Empty mask and no bbox, skip
                continue
            
            # Create Detection
            detection = Detection(
                bbox=bbox,
                mask=mask,  # May be None for bbox-only
                confidence=ann.get('confidence'),  # May be None for manual annotations
                source=ann.get('source', 'manual'),
                class_id=ann.get('class_id', 0)
            )
            detections.append(detection)
        return detections
    
    def _annotations_to_tracks(self, annotations):
        """Convert existing annotations to Track objects"""
        tracks = []
        for ann in annotations:
            track = Track(
                track_id=ann['id'],
                bbox=ann.get('bbox', [0, 0, 0, 0]),
                mask=ann['segmentation'],
                last_seen_frame=self.current_frame_idx,
                source_history=[ann.get('source', 'manual')],
                confidence_history=[ann.get('confidence', 1.0)],
                frames_lost=0
            )
            tracks.append(track)
        return tracks
    
    def _selected_video_ids(self):
        """Return selected sidebar video IDs in their visible order."""
        selected_ids = {
            item.data(Qt.ItemDataRole.UserRole)
            for item in self.video_list.selectedItems()
            if item.data(Qt.ItemDataRole.UserRole)
        }
        ordered_ids = []
        for row in range(self.video_list.count()):
            video_id = self.video_list.item(row).data(Qt.ItemDataRole.UserRole)
            if video_id in selected_ids:
                ordered_ids.append(video_id)
        if not ordered_ids and self.current_video_id:
            ordered_ids.append(self.current_video_id)
        return ordered_ids

    def _load_video_list_item(self, item):
        """Load one sidebar video without changing the batch selection."""
        if item is None:
            return
        video_id = item.data(Qt.ItemDataRole.UserRole)
        if not video_id:
            return
        if video_id == self.current_video_id:
            self.canvas.setFocus()
            return

        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            self.load_video_frames(video_id)
        finally:
            QApplication.restoreOverrideCursor()
        self.canvas.setFocus()

    def on_video_item_clicked(self, item):
        """Load plain clicks, while Ctrl/Shift clicks only modify selection."""
        modifiers = QApplication.keyboardModifiers()
        selection_modifiers = (
            Qt.KeyboardModifier.ControlModifier |
            Qt.KeyboardModifier.ShiftModifier
        )
        if modifiers & selection_modifiers:
            selected_count = len(self._selected_video_ids())
            self.status_label.setText(
                f"Selected {selected_count} video"
                f"{'s' if selected_count != 1 else ''} for batch assignment."
            )
            return
        self._load_video_list_item(item)

    def on_video_item_activated(self, item):
        """Allow Enter or double-click to explicitly load a selected video."""
        self._load_video_list_item(item)

    def on_video_selected(self, row):
        """Compatibility entry point for code that loads a sidebar row."""
        if 0 <= row < self.video_list.count():
            self._load_video_list_item(self.video_list.item(row))
    
    def on_tracking_filter_changed(self, state):
        """Handle tracking filter checkbox state change"""
        self.update_video_list()
    
    def _get_videos_with_tracking_sequences(self):
        """Get set of video IDs that have tracking sequences"""
        if not hasattr(self, 'tracking_sequence_manager'):
            return set()
        
        videos = set()
        for sequence in self.tracking_sequence_manager.get_all_sequences():
            videos.add(sequence.video_id)
        return videos
    
    def show_video_context_menu(self, position):
        """Show context menu for video list"""
        item = self.video_list.itemAt(position)
        if not item:
            return
        
        video_id = item.data(Qt.ItemDataRole.UserRole)
        if not video_id:
            return
        
        from PyQt6.QtWidgets import QMenu
        menu = QMenu()
        
        delete_annotations_action = menu.addAction("Delete All Annotations...")
        menu.addSeparator()
        delete_video_action = menu.addAction("Delete Video...")
        
        action = menu.exec(self.video_list.mapToGlobal(position))
        
        if action == delete_annotations_action:
            self.delete_video_annotations(video_id)
        elif action == delete_video_action:
            self.delete_video(video_id)
    
    def delete_video_annotations(self, video_id):
        """Delete all annotations for a video (keep frames and video file)"""
        if not self.project_path or not video_id:
            return
        
        # Confirmation dialog
        reply = QMessageBox.question(
            self, 
            "Delete All Annotations",
            f"Are you sure you want to delete ALL annotations for '{video_id}'?\n\n"
            f"This will permanently delete:\n"
            f"• All annotation files (PNG/JSON/BBox/PKL)\n"
            f"• Cached annotation data\n\n"
            f"The video frames will be kept.\n"
            f"This action cannot be undone.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No
        )
        
        if reply != QMessageBox.StandardButton.Yes:
            return
        
        try:
            import shutil
            
            deleted_count = 0
            
            # Delete PNG annotations directory
            png_annotations_dir = self.project_path / 'annotations' / 'png' / video_id
            if png_annotations_dir.exists():
                file_count = len(list(png_annotations_dir.glob('frame_*.png')))
                shutil.rmtree(png_annotations_dir)
                print(f"Deleted PNG annotations: {png_annotations_dir} ({file_count} files)")
                deleted_count += file_count
            
            # Delete JSON annotations directory
            json_annotations_dir = self.project_path / 'annotations' / 'json' / video_id
            if json_annotations_dir.exists():
                file_count = len(list(json_annotations_dir.glob('frame_*.json')))
                shutil.rmtree(json_annotations_dir)
                print(f"Deleted JSON annotations: {json_annotations_dir} ({file_count} files)")
                deleted_count += file_count
            
            # Delete PKL annotations directory (old format)
            pkl_annotations_dir = self.project_path / 'annotations' / 'pkl' / video_id
            if pkl_annotations_dir.exists():
                file_count = len(list(pkl_annotations_dir.glob('frame_*.pkl')))
                shutil.rmtree(pkl_annotations_dir)
                print(f"Deleted PKL annotations: {pkl_annotations_dir} ({file_count} files)")
                deleted_count += file_count
            
            # Delete BBox annotations directory (bbox-only annotations)
            bbox_annotations_dir = self.project_path / 'annotations' / 'bbox' / video_id
            if bbox_annotations_dir.exists():
                file_count = len(list(bbox_annotations_dir.glob('frame_*.json')))
                shutil.rmtree(bbox_annotations_dir)
                print(f"Deleted BBox annotations: {bbox_annotations_dir} ({file_count} files)")
                deleted_count += file_count
            
            # Clear annotation cache for this video
            # Cache keys are (video_id, frame_idx) tuples
            keys_to_remove = [key for key in self.annotation_manager.frame_annotations.keys() 
                             if isinstance(key, tuple) and key[0] == video_id]
            for key in keys_to_remove:
                del self.annotation_manager.frame_annotations[key]
            print(f"Cleared {len(keys_to_remove)} cached annotations for video {video_id}")
            
            # Reset max_mask_id tracking for this video
            # This ensures new annotations start from ID 1
            self._save_max_mask_id_to_metadata(video_id, 0)
            if video_id in self.video_next_mask_id:
                del self.video_next_mask_id[video_id]
            print(f"Reset max_mask_id tracking for video {video_id}")
            
            # If we're currently viewing this video, also reset canvas counter
            if self.current_video_id == video_id:
                self.canvas.next_mask_id = 1
                print(f"Reset canvas next_mask_id to 1")
            
            # If we're currently viewing this video, refresh the display
            if self.current_video_id == video_id and self.frames:
                # Reload the current frame (now without annotations)
                current_idx = self.current_frame_idx
                if 0 <= current_idx < len(self.frames):
                    self.load_frame(current_idx)
            
            self.status_label.setText(f"✓ Deleted annotations for {video_id}")
            QMessageBox.information(
                self, 
                "Annotations Deleted",
                f"Successfully deleted all annotations for '{video_id}'.\n\n"
                f"Total annotation files deleted: {deleted_count}"
            )
            
        except Exception as e:
            error_msg = f"Failed to delete annotations:\n\n{str(e)}"
            self.status_label.setText("✗ Delete failed")
            QMessageBox.critical(self, "Delete Error", error_msg)
            import traceback
            traceback.print_exc()
    
    def delete_video(self, video_id):
        """Delete a video and all its associated data"""
        if not self.project_path or not video_id:
            return
        
        # Confirmation dialog
        reply = QMessageBox.question(
            self, 
            "Delete Video",
            f"Are you sure you want to delete '{video_id}'?\n\n"
            f"This will permanently delete:\n"
            f"• All frames for this video\n"
            f"• All annotations for this video\n"
            f"• The video file from input_data\n\n"
            f"This action cannot be undone.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No
        )
        
        if reply != QMessageBox.StandardButton.Yes:
            return
        
        try:
            import shutil
            
            # If we're currently viewing this video, clear the view
            if self.current_video_id == video_id:
                self.frames = []
                self.frame_video_ids = []
                self.frame_splits = []
                self.frame_selected = []
                self.canvas.clear_image()
                self.update_frame_list()
                self.current_video_id = None
            
            # Delete frames directory
            frames_dir = self.project_manager.get_frames_dir(video_id)
            if frames_dir.exists():
                shutil.rmtree(frames_dir)
                print(f"Deleted frames: {frames_dir}")
            
            # Delete all annotation directories
            # PKL annotations (old format)
            pkl_annotations_dir = self.project_path / 'annotations' / 'pkl' / video_id
            if pkl_annotations_dir.exists():
                shutil.rmtree(pkl_annotations_dir)
                print(f"Deleted PKL annotations: {pkl_annotations_dir}")
            
            # PNG annotations
            png_annotations_dir = self.project_path / 'annotations' / 'png' / video_id
            if png_annotations_dir.exists():
                shutil.rmtree(png_annotations_dir)
                print(f"Deleted PNG annotations: {png_annotations_dir}")
            
            # JSON annotations
            json_annotations_dir = self.project_path / 'annotations' / 'json' / video_id
            if json_annotations_dir.exists():
                shutil.rmtree(json_annotations_dir)
                print(f"Deleted JSON annotations: {json_annotations_dir}")
            
            # BBox annotations
            bbox_annotations_dir = self.project_path / 'annotations' / 'bbox' / video_id
            if bbox_annotations_dir.exists():
                shutil.rmtree(bbox_annotations_dir)
                print(f"Deleted BBox annotations: {bbox_annotations_dir}")
            
            # Delete video file from input_data (check all splits)
            deleted_video = False
            
            # Try to find the video in all splits if split detection fails
            for split in ['train', 'val', 'test', 'inference']:
                input_dir = self.project_path / 'input_data' / split
                if not input_dir.exists():
                    continue
                
                # Look for any file matching the video_id
                for video_path in input_dir.iterdir():
                    if video_path.stem == video_id or video_path.name == video_id:
                        try:
                            if video_path.is_file():
                                video_path.unlink()
                                print(f"Deleted video: {video_path}")
                                deleted_video = True
                            elif video_path.is_dir():
                                shutil.rmtree(video_path)
                                print(f"Deleted video directory: {video_path}")
                                deleted_video = True
                        except Exception as e:
                            print(f"Error deleting {video_path}: {e}")
            
            if not deleted_video:
                print(f"Warning: Could not find video file for {video_id} (might have failed to import)")

            self._load_imaging_setup_config()
            if self.video_imaging_setups.pop(video_id, None) is not None:
                self._save_imaging_setup_config()
            self.video_physical_scales.pop(video_id, None)
            self.video_scale_known_lengths_cm.pop(video_id, None)
            
            # Update video list
            self.update_video_list()
            
            self.status_label.setText(f"✓ Deleted {video_id}")
            QMessageBox.information(
                self, 
                "Video Deleted",
                f"Successfully deleted '{video_id}' and all associated data."
            )
            
        except Exception as e:
            error_msg = f"Failed to delete video:\n\n{str(e)}"
            self.status_label.setText("✗ Delete failed")
            QMessageBox.critical(self, "Delete Error", error_msg)
            import traceback
            traceback.print_exc()
    
    def update_video_list(self):
        """Update the video list with all videos in the project"""
        selected_video_ids = set()
        current_list_video_id = None
        if hasattr(self, 'video_list'):
            selected_video_ids = {
                item.data(Qt.ItemDataRole.UserRole)
                for item in self.video_list.selectedItems()
                if item.data(Qt.ItemDataRole.UserRole)
            }
            current_item = self.video_list.currentItem()
            if current_item is not None:
                current_list_video_id = current_item.data(Qt.ItemDataRole.UserRole)
        self.video_list.clear()
        
        if not self.project_path:
            return

        self._load_imaging_setup_config()

        def setup_sort_key(video_id):
            setup_name = self.video_imaging_setups.get(video_id)
            return (setup_name is None, (setup_name or "").casefold(), video_id.casefold())

        def video_item_text(split_label, video_id):
            setup_name = self.video_imaging_setups.get(video_id)
            setup_suffix = f"  •  {setup_name}" if setup_name else "  •  unassigned"
            return f"[{split_label}] {video_id}{setup_suffix}"
        
        # Get videos with tracking sequences if filter is enabled
        videos_with_tracking = None
        if hasattr(self, 'tracking_filter_checkbox') and self.tracking_filter_checkbox.isChecked():
            videos_with_tracking = self._get_videos_with_tracking_sequences()
        
        # Get all videos from train, val, test, and inference splits
        train_videos = self.project_manager.get_videos_by_split('train')
        val_videos = self.project_manager.get_videos_by_split('val')
        test_videos = self.project_manager.get_videos_by_split('test')
        inference_videos = self.project_manager.get_videos_by_split('inference')

        # Restrict the sidebar by split. The mask-only choice follows the
        # selected-frame metadata used by the annotation export workflow.
        if self.video_split_filter == 'train':
            val_videos = []
            test_videos = []
            inference_videos = []
        elif self.video_split_filter == 'val':
            train_videos = []
            test_videos = []
            inference_videos = []
        elif self.video_split_filter == 'val_masks':
            train_videos = []
            val_videos = [
                video_id for video_id in val_videos
                if video_has_selected_bee_masks(self.project_path, video_id)
            ]
            test_videos = []
            inference_videos = []
        elif self.video_split_filter == 'test':
            train_videos = []
            val_videos = []
            inference_videos = []
        elif self.video_split_filter == 'inference':
            train_videos = []
            val_videos = []
            test_videos = []
        
        # Apply filter if enabled
        if videos_with_tracking is not None:
            train_videos = [v for v in train_videos if v in videos_with_tracking]
            val_videos = [v for v in val_videos if v in videos_with_tracking]
            test_videos = [v for v in test_videos if v in videos_with_tracking]
            inference_videos = [v for v in inference_videos if v in videos_with_tracking]
        
        # Add train videos
        for video_id in sorted(train_videos, key=setup_sort_key):
            item_text = video_item_text("TRAIN", video_id)
            from PyQt6.QtWidgets import QListWidgetItem
            item = QListWidgetItem(item_text)
            item.setData(Qt.ItemDataRole.UserRole, video_id)  # Store video_id
            item.setToolTip(item_text)
            
            # Highlight current video
            if video_id == self.current_video_id:
                from PyQt6.QtGui import QFont
                font = QFont()
                font.setBold(True)
                item.setFont(font)
            
            self.video_list.addItem(item)

        # Add val videos
        for video_id in sorted(val_videos, key=setup_sort_key):
            item_text = video_item_text("VAL", video_id)
            from PyQt6.QtWidgets import QListWidgetItem
            item = QListWidgetItem(item_text)
            item.setData(Qt.ItemDataRole.UserRole, video_id)  # Store video_id
            item.setToolTip(item_text)
            
            # Highlight current video
            if video_id == self.current_video_id:
                from PyQt6.QtGui import QFont
                font = QFont()
                font.setBold(True)
                item.setFont(font)
            
            self.video_list.addItem(item)

        # Add test videos
        for video_id in sorted(test_videos, key=setup_sort_key):
            item_text = video_item_text("TEST", video_id)
            from PyQt6.QtWidgets import QListWidgetItem
            item = QListWidgetItem(item_text)
            item.setData(Qt.ItemDataRole.UserRole, video_id)  # Store video_id
            item.setToolTip(item_text)
            
            # Highlight current video
            if video_id == self.current_video_id:
                from PyQt6.QtGui import QFont
                font = QFont()
                font.setBold(True)
                item.setFont(font)
            
            self.video_list.addItem(item)
        
        # Add inference videos
        for video_id in sorted(inference_videos, key=setup_sort_key):
            item_text = video_item_text("INFERENCE", video_id)
            from PyQt6.QtWidgets import QListWidgetItem
            item = QListWidgetItem(item_text)
            item.setData(Qt.ItemDataRole.UserRole, video_id)  # Store video_id
            item.setToolTip(item_text)
            
            # Highlight current video
            if video_id == self.current_video_id:
                from PyQt6.QtGui import QFont
                font = QFont()
                font.setBold(True)
                item.setFont(font)
            
            self.video_list.addItem(item)

        # Preserve batch selections when labels/sorting are refreshed. The
        # current video remains the keyboard anchor, but only previously
        # selected videos are reselected.
        restore_current_id = current_list_video_id or self.current_video_id
        self.video_list.blockSignals(True)
        try:
            current_item_to_restore = None
            for index in range(self.video_list.count()):
                item = self.video_list.item(index)
                video_id = item.data(Qt.ItemDataRole.UserRole)
                if video_id == restore_current_id:
                    current_item_to_restore = item
                if video_id in selected_video_ids:
                    item.setSelected(True)
            if current_item_to_restore is not None:
                self.video_list.setCurrentItem(current_item_to_restore)
                for index in range(self.video_list.count()):
                    item = self.video_list.item(index)
                    item.setSelected(
                        item.data(Qt.ItemDataRole.UserRole) in selected_video_ids
                    )
        finally:
            self.video_list.blockSignals(False)

    def create_video_list_dock(self):
        """Create dock widget for video list sidebar"""
        self.video_dock = QDockWidget("Project Videos", self)
        self.video_dock.setAllowedAreas(Qt.DockWidgetArea.LeftDockWidgetArea | 
                                        Qt.DockWidgetArea.RightDockWidgetArea)
        
        # Create container widget with filter controls
        from PyQt6.QtWidgets import QCheckBox, QComboBox
        container = QWidget()
        layout = QVBoxLayout()
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(5)

        split_filter_layout = QHBoxLayout()
        split_filter_layout.addWidget(QLabel("Show:"))
        self.video_split_filter_combo = QComboBox()
        self.video_split_filter_combo.addItems([
            'All Videos',
            'Training Videos',
            'Validation Videos',
            'Validation Videos with Masks',
            'Test Videos',
            'Inference Videos',
        ])
        self.video_split_filter_combo.currentTextChanged.connect(
            self.on_video_split_filter_changed
        )
        split_filter_layout.addWidget(self.video_split_filter_combo)
        layout.addLayout(split_filter_layout)
        
        # Tracking filter checkbox
        self.tracking_filter_checkbox = QCheckBox("Show only videos with tracking sequences")
        self.tracking_filter_checkbox.stateChanged.connect(self.on_tracking_filter_changed)
        layout.addWidget(self.tracking_filter_checkbox)
        
        # Create video list widget
        self.video_list = QListWidget()
        self.video_list.setSelectionMode(QListWidget.SelectionMode.ExtendedSelection)
        self.video_list.itemClicked.connect(self.on_video_item_clicked)
        self.video_list.itemActivated.connect(self.on_video_item_activated)
        self.video_list.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.video_list.customContextMenuRequested.connect(self.show_video_context_menu)
        layout.addWidget(self.video_list)

        batch_help = QLabel(
            "Ctrl-click selects individual videos; Shift-click selects a range. "
            "Ctrl+A selects all visible videos. Choosing an imaging setup applies "
            "it to all selected videos."
        )
        batch_help.setWordWrap(True)
        layout.addWidget(batch_help)
        
        container.setLayout(layout)
        self.video_dock.setWidget(container)
        self.video_dock.visibilityChanged.connect(self.on_video_dock_visibility_changed)
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, self.video_dock)

    def on_video_split_filter_changed(self, text):
        """Filter the project video sidebar by split and mask availability."""
        filter_map = {
            'All Videos': 'all',
            'Training Videos': 'train',
            'Validation Videos': 'val',
            'Validation Videos with Masks': 'val_masks',
            'Test Videos': 'test',
            'Inference Videos': 'inference',
        }
        self.video_split_filter = filter_map.get(text, 'all')

        if self.video_split_filter == 'val_masks' and hasattr(self, 'split_filter_combo'):
            self.split_filter_combo.setCurrentText('Validation Masks Only')

        self.update_video_list()
    
    def create_frame_list_dock(self):
        """Create dock widget for frame list"""
        dock = QDockWidget("Frames", self)
        dock.setAllowedAreas(Qt.DockWidgetArea.LeftDockWidgetArea | 
                            Qt.DockWidgetArea.RightDockWidgetArea)
        
        # Create container widget with filter controls
        from PyQt6.QtWidgets import QComboBox
        container = QWidget()
        layout = QVBoxLayout()
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(5)
        
        # Split filter dropdown
        filter_layout = QHBoxLayout()
        filter_layout.addWidget(QLabel("Show:"))
        self.split_filter_combo = QComboBox()
        self.split_filter_combo.addItems([
            'All Frames',
            'Training Only',
            'Validation Only',
            'Validation Masks Only',
            'Test Only',
            'Inference Only',
        ])
        self.split_filter_combo.currentTextChanged.connect(self.on_split_filter_changed)
        filter_layout.addWidget(self.split_filter_combo)
        layout.addLayout(filter_layout)
        
        # Frame list
        self.frame_list = QListWidget()
        self.frame_list.currentRowChanged.connect(self.on_frame_changed)
        self.frame_list.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        layout.addWidget(self.frame_list)
        
        container.setLayout(layout)
        dock.setWidget(container)
        
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, dock)
        
    def create_instance_list_dock(self):
        """Create dock widget for instance list"""
        dock = QDockWidget("Instances", self)
        dock.setAllowedAreas(Qt.DockWidgetArea.LeftDockWidgetArea | 
                            Qt.DockWidgetArea.RightDockWidgetArea)
        
        # Create container widget with layout
        from PyQt6.QtWidgets import QCheckBox
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(5)
        
        # Add checkbox to show/hide instance numbers
        self.show_labels_checkbox = QCheckBox("Show Instance Numbers")
        self.show_labels_checkbox.setChecked(True)
        self.show_labels_checkbox.stateChanged.connect(self.on_show_labels_changed)
        layout.addWidget(self.show_labels_checkbox)

        self.show_aruco_labels_checkbox = QCheckBox("Show ArUco Tags")
        self.show_aruco_labels_checkbox.setChecked(False)
        self.show_aruco_labels_checkbox.stateChanged.connect(self.on_show_aruco_labels_changed)
        layout.addWidget(self.show_aruco_labels_checkbox)
        
        # Add instance list
        self._instance_list_updating = False
        self.instance_list = QListWidget()
        self.instance_list.setSelectionMode(QListWidget.SelectionMode.ExtendedSelection)  # Enable multi-selection with Shift/Ctrl
        self.instance_list.currentRowChanged.connect(self.on_instance_changed)
        self.instance_list.itemClicked.connect(self.on_instance_clicked)  # Also handle re-clicks on same item
        self.instance_list.itemChanged.connect(self.on_instance_bbox_visibility_changed)
        self.instance_list.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.instance_list.customContextMenuRequested.connect(self.show_instance_context_menu)
        layout.addWidget(self.instance_list)
        
        dock.setWidget(container)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, dock)
    
    def create_tracking_sequences_dock(self):
        """Create dock widget for tracking sequences"""
        dock = QDockWidget("Tracking Sequences", self)
        dock.setAllowedAreas(Qt.DockWidgetArea.LeftDockWidgetArea | 
                            Qt.DockWidgetArea.RightDockWidgetArea)
        
        # Create tracking sequences panel
        self.tracking_sequences_panel = TrackingSequencesPanel(self)
        
        # Connect signals
        self.tracking_sequences_panel.sequence_created.connect(self.on_sequence_created)
        self.tracking_sequences_panel.sequence_deleted.connect(self.on_sequence_deleted)
        self.tracking_sequences_panel.sequence_selected.connect(self.on_sequence_selected)
        self.tracking_sequences_panel.validation_requested.connect(self.on_validation_requested)
        
        dock.setWidget(self.tracking_sequences_panel)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, dock)
        
    def setup_shortcuts(self):
        """Setup keyboard shortcuts"""
        # Navigation
        from PyQt6.QtGui import QShortcut
        
        next_frame = QShortcut(QKeySequence(Qt.Key.Key_D), self)
        next_frame.activated.connect(self.next_frame)
        
        prev_frame = QShortcut(QKeySequence(Qt.Key.Key_A), self)
        prev_frame.activated.connect(self.prev_frame)
        
        # Arrow key navigation
        next_frame_arrow = QShortcut(QKeySequence(Qt.Key.Key_Down), self)
        next_frame_arrow.activated.connect(self.next_frame)
        
        prev_frame_arrow = QShortcut(QKeySequence(Qt.Key.Key_Up), self)
        prev_frame_arrow.activated.connect(self.prev_frame)
        
        # Delete instance
        delete_instance = QShortcut(QKeySequence(Qt.Key.Key_Delete), self)
        delete_instance.activated.connect(self.delete_selected_instance)
        
        backspace_instance = QShortcut(QKeySequence(Qt.Key.Key_Backspace), self)
        backspace_instance.activated.connect(self.delete_selected_instance)
        
        # Clear instance mask and points
        clear_instance = QShortcut(QKeySequence(Qt.Key.Key_C), self)
        clear_instance.activated.connect(self.clear_selected_instance)

        toggle_overlays = QShortcut(QKeySequence(Qt.Key.Key_Space), self)
        toggle_overlays.setContext(Qt.ShortcutContext.WindowShortcut)
        toggle_overlays.activated.connect(self.on_space_toggle_annotation_overlays)

        toggle_selected_instance = QShortcut(QKeySequence("Shift+Space"), self)
        toggle_selected_instance.setContext(Qt.ShortcutContext.WindowShortcut)
        toggle_selected_instance.activated.connect(self.on_shift_space_toggle_selected_instance)
    
    def _load_max_mask_id_from_metadata(self, video_id):
        """
        Load cached max_mask_id for a video from metadata (backward compatible)
        
        Returns:
            int or None: The cached max_mask_id, or None if not cached
        """
        if not self.project_path or not video_id:
            return None
        
        metadata_file = self.project_manager.get_frames_dir(video_id) / 'video_metadata.json'
        if metadata_file.exists():
            try:
                import json
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                    return metadata.get('max_mask_id')
            except (json.JSONDecodeError, ValueError) as e:
                print(f"Warning: Corrupted metadata file, ignoring max_mask_id: {e}")
                return None
            except Exception as e:
                print(f"Warning: Could not load max_mask_id from metadata: {e}")
                return None
        
        return None

    def _imaging_setups_path(self):
        """Return the project-level imaging-setup calibration file."""
        if not self.project_path:
            return None
        return Path(self.project_path) / 'annotations' / 'imaging_setups.json'

    @staticmethod
    def _valid_pixels_per_cm(value):
        """Return a finite positive scale, or None for an invalid value."""
        try:
            value = float(value)
        except (TypeError, ValueError):
            return None
        return value if np.isfinite(value) and value > 0 else None

    def _load_imaging_setup_config(self, force=False):
        """Load named setup profiles and video assignments for this project."""
        if self._imaging_setups_loaded and not force:
            return

        self.imaging_setup_profiles = {}
        self.video_imaging_setups = {}
        config_path = self._imaging_setups_path()
        if config_path and config_path.exists():
            try:
                with open(config_path, 'r') as f:
                    config = json.load(f)
                raw_setups = config.get('setups', {})
                raw_assignments = config.get('video_assignments', {})
                if not isinstance(raw_setups, dict) or not isinstance(raw_assignments, dict):
                    raise ValueError("setups and video_assignments must be JSON objects")

                for raw_name, raw_profile in raw_setups.items():
                    name = str(raw_name).strip()
                    if not name or not isinstance(raw_profile, dict):
                        continue
                    profile = dict(raw_profile)
                    profile['pixels_per_cm'] = self._valid_pixels_per_cm(
                        profile.get('pixels_per_cm')
                    )
                    self.imaging_setup_profiles[name] = profile

                self.video_imaging_setups = {
                    str(video_id): str(setup_name)
                    for video_id, setup_name in raw_assignments.items()
                    if str(setup_name) in self.imaging_setup_profiles
                }
            except (json.JSONDecodeError, ValueError, TypeError) as e:
                print(f"Warning: Could not load imaging setup calibrations: {e}")

        self._imaging_setups_loaded = True

    def _save_imaging_setup_config(self):
        """Persist named setup profiles and video assignments atomically."""
        config_path = self._imaging_setups_path()
        if config_path is None:
            return False
        try:
            config = {
                'version': 1,
                'updated_at': datetime.now().isoformat(),
                'setups': self.imaging_setup_profiles,
                'video_assignments': self.video_imaging_setups,
            }
            config_path.parent.mkdir(parents=True, exist_ok=True)
            temp_file = config_path.parent / f'.{config_path.name}.tmp'
            try:
                with open(temp_file, 'w') as f:
                    json.dump(config, f, indent=2, sort_keys=True)
                temp_file.replace(config_path)
            except Exception:
                if temp_file.exists():
                    temp_file.unlink()
                raise
            return True
        except Exception as e:
            print(f"Warning: Could not save imaging setup calibrations: {e}")
            return False

    def _reset_imaging_setup_cache(self):
        """Clear setup state when opening or creating another project."""
        self.video_physical_scales.clear()
        self.video_scale_known_lengths_cm.clear()
        self.imaging_setup_profiles = {}
        self.video_imaging_setups = {}
        self._imaging_setups_loaded = False
        if hasattr(self, 'toolbar'):
            self.toolbar.set_imaging_setups([], None)
            self.toolbar.set_measurement_scale(None)

    def _current_imaging_setup(self, video_id=None):
        """Return the named setup assigned to a video, if any."""
        self._load_imaging_setup_config()
        return self.video_imaging_setups.get(video_id or self.current_video_id)

    def _profile_calibration_values(self, setup_name):
        """Return normalized calibration values for a named setup."""
        profile = self.imaging_setup_profiles.get(setup_name, {})
        return {
            'pixels_per_cm': self._valid_pixels_per_cm(profile.get('pixels_per_cm')),
            'calibration_line_pixels': profile.get('calibration_line_pixels'),
            'calibration_distance_cm': profile.get('calibration_distance_cm'),
            'image_width_pixels': profile.get('image_width_pixels'),
            'image_height_pixels': profile.get('image_height_pixels'),
        }

    def _read_video_physical_scale_record(self, video_id):
        """Read a video's complete legacy/mirrored calibration record."""
        if not self.project_path or not video_id:
            return {}
        metadata_file = self.project_manager.get_frames_dir(video_id) / 'video_metadata.json'
        if not metadata_file.exists():
            return {}
        try:
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
            physical_scale = metadata.get('physical_scale', {})
            if isinstance(physical_scale, dict):
                record = dict(physical_scale)
            else:
                record = {}
            if record.get('pixels_per_cm') is None:
                record['pixels_per_cm'] = metadata.get('pixels_per_cm')
            return record
        except (json.JSONDecodeError, ValueError, TypeError) as e:
            print(f"Warning: Could not read physical scale for {video_id}: {e}")
            return {}

    def _load_video_physical_scale(self, video_id, use_cache=True):
        """Load a video's pixels-per-centimeter calibration from metadata."""
        if not video_id:
            return None
        if use_cache and video_id in self.video_physical_scales:
            return self.video_physical_scales[video_id]

        pixels_per_cm = None
        known_length_cm = None
        if self.project_path:
            physical_scale = self._read_video_physical_scale_record(video_id)
            pixels_per_cm = physical_scale.get('pixels_per_cm')
            known_length_cm = physical_scale.get('calibration_distance_cm')

        try:
            pixels_per_cm = float(pixels_per_cm)
            if not np.isfinite(pixels_per_cm) or pixels_per_cm <= 0:
                pixels_per_cm = None
        except (TypeError, ValueError):
            pixels_per_cm = None

        try:
            known_length_cm = float(known_length_cm)
            if np.isfinite(known_length_cm) and known_length_cm > 0:
                self.video_scale_known_lengths_cm[video_id] = known_length_cm
        except (TypeError, ValueError):
            pass

        self.video_physical_scales[video_id] = pixels_per_cm
        return pixels_per_cm

    def _save_video_physical_scale(self, video_id, pixels_per_cm,
                                   calibration_line_pixels=None,
                                   calibration_distance_cm=None,
                                   imaging_setup=None,
                                   image_width_pixels=None,
                                   image_height_pixels=None):
        """Persist or remove video-level physical calibration atomically."""
        if not self.project_path or not video_id:
            return False

        metadata_file = self.project_manager.get_frames_dir(video_id) / 'video_metadata.json'
        try:
            metadata = {}
            if metadata_file.exists():
                try:
                    with open(metadata_file, 'r') as f:
                        metadata = json.load(f)
                except (json.JSONDecodeError, ValueError) as e:
                    print(f"Warning: Corrupted metadata file, recreating: {e}")

            if imaging_setup:
                metadata['imaging_setup'] = str(imaging_setup)
            else:
                metadata.pop('imaging_setup', None)

            if pixels_per_cm is None:
                metadata.pop('physical_scale', None)
                metadata.pop('pixels_per_cm', None)
            else:
                if (
                    (image_width_pixels is None or image_height_pixels is None) and
                    self.canvas.current_image is not None
                ):
                    current_height, current_width = self.canvas.current_image.shape[:2]
                    image_width_pixels = image_width_pixels or current_width
                    image_height_pixels = image_height_pixels or current_height
                metadata['physical_scale'] = {
                    'pixels_per_cm': float(pixels_per_cm),
                    'cm_per_pixel': 1.0 / float(pixels_per_cm),
                    'calibration_line_pixels': (
                        float(calibration_line_pixels)
                        if calibration_line_pixels is not None else None
                    ),
                    'calibration_distance_cm': (
                        float(calibration_distance_cm)
                        if calibration_distance_cm is not None else None
                    ),
                    'image_width_pixels': image_width_pixels,
                    'image_height_pixels': image_height_pixels,
                    'source': 'imaging_setup' if imaging_setup else 'video_specific',
                    'imaging_setup': str(imaging_setup) if imaging_setup else None,
                }

            metadata_file.parent.mkdir(parents=True, exist_ok=True)
            temp_file = metadata_file.parent / f'.{metadata_file.name}.tmp'
            try:
                with open(temp_file, 'w') as f:
                    json.dump(metadata, f, indent=2)
                temp_file.replace(metadata_file)
            except Exception:
                if temp_file.exists():
                    temp_file.unlink()
                raise
            return True
        except Exception as e:
            print(f"Warning: Could not save physical scale for {video_id}: {e}")
            return False

    def _apply_video_physical_scale(self, video_id):
        """Apply the current video's persisted scale to canvas and toolbar."""
        self._load_imaging_setup_config()
        setup_name = self.video_imaging_setups.get(video_id)
        if setup_name:
            pixels_per_cm = self._profile_calibration_values(setup_name)[
                'pixels_per_cm'
            ]
            compatible, mismatch_reason = self._setup_matches_current_resolution(
                setup_name
            )
            if not compatible:
                pixels_per_cm = None
            self.video_physical_scales[video_id] = pixels_per_cm
        else:
            mismatch_reason = None
            pixels_per_cm = self._load_video_physical_scale(video_id)

        self.canvas.set_pixels_per_cm(pixels_per_cm)
        self.toolbar.set_imaging_setups(
            self.imaging_setup_profiles.keys(), setup_name
        )
        self.toolbar.set_measurement_scale(pixels_per_cm, setup_name)
        if mismatch_reason:
            self.toolbar.measurement_scale_label.setText(
                f'Setup "{setup_name}": resolution mismatch — pixels only'
            )
            self.toolbar.measurement_scale_label.setToolTip(mismatch_reason)

    def _setup_matches_current_resolution(self, setup_name):
        """Check that a calibrated setup uses the current frame dimensions."""
        if not self.current_video_id:
            return True, None
        return self._setup_matches_video_resolution(
            setup_name, self.current_video_id
        )

    def _video_frame_dimensions(self, video_id):
        """Read a video's native saved-frame dimensions without loading it."""
        if video_id == self.current_video_id and self.canvas.current_image is not None:
            image_height, image_width = self.canvas.current_image.shape[:2]
            return image_width, image_height

        frames_dir = self.project_manager.get_frames_dir(video_id)
        frame_files = []
        for pattern in ('frame_*.jpg', 'frame_*.jpeg', 'frame_*.png'):
            frame_files.extend(frames_dir.glob(pattern))
        if not frame_files:
            return None
        image = cv2.imread(str(sorted(frame_files)[0]), cv2.IMREAD_GRAYSCALE)
        if image is None:
            return None
        image_height, image_width = image.shape[:2]
        return image_width, image_height

    def _setup_matches_video_resolution(self, setup_name, video_id):
        """Check one video against the resolution stored for a setup."""
        values = self._profile_calibration_values(setup_name)
        expected_width = values['image_width_pixels']
        expected_height = values['image_height_pixels']
        if expected_width is None or expected_height is None:
            return True, None
        dimensions = self._video_frame_dimensions(video_id)
        if dimensions is None:
            return True, None
        current_width, current_height = dimensions
        if (int(expected_width), int(expected_height)) == dimensions:
            return True, None
        return False, (
            f'Setup "{setup_name}" was calibrated at '
            f'{expected_width} × {expected_height} pixels, but {video_id} is '
            f'{current_width} × {current_height} pixels. The same pixels/cm value '
            "cannot safely be reused at a different resolution."
        )

    def _mirror_setup_calibration_to_assigned_videos(self, setup_name):
        """Write an auditable copy of a setup scale into assigned video metadata."""
        values = self._profile_calibration_values(setup_name)
        assigned_videos = [
            video_id
            for video_id, assigned_setup in self.video_imaging_setups.items()
            if assigned_setup == setup_name
        ]
        saved_count = 0
        for video_id in assigned_videos:
            if self._save_video_physical_scale(
                video_id,
                values['pixels_per_cm'],
                calibration_line_pixels=values['calibration_line_pixels'],
                calibration_distance_cm=values['calibration_distance_cm'],
                imaging_setup=setup_name,
                image_width_pixels=values['image_width_pixels'],
                image_height_pixels=values['image_height_pixels'],
            ):
                saved_count += 1
            self.video_physical_scales[video_id] = values['pixels_per_cm']
            known_length = values['calibration_distance_cm']
            if known_length is not None:
                self.video_scale_known_lengths_cm[video_id] = float(known_length)
        return saved_count, len(assigned_videos)

    def _assign_videos_to_imaging_setup(self, setup_name, video_ids):
        """Assign a setup to compatible videos and return assigned/skipped IDs."""
        values = self._profile_calibration_values(setup_name)
        assigned_ids = []
        skipped = []
        for video_id in video_ids:
            compatible, reason = self._setup_matches_video_resolution(
                setup_name, video_id
            )
            if not compatible:
                skipped.append((video_id, reason))
                continue

            self.video_imaging_setups[video_id] = setup_name
            self._save_video_physical_scale(
                video_id,
                values['pixels_per_cm'],
                calibration_line_pixels=values['calibration_line_pixels'],
                calibration_distance_cm=values['calibration_distance_cm'],
                imaging_setup=setup_name,
                image_width_pixels=values['image_width_pixels'],
                image_height_pixels=values['image_height_pixels'],
            )
            self.video_physical_scales[video_id] = values['pixels_per_cm']
            known_length = values['calibration_distance_cm']
            if known_length is not None:
                self.video_scale_known_lengths_cm[video_id] = float(known_length)
            else:
                self.video_scale_known_lengths_cm.pop(video_id, None)
            assigned_ids.append(video_id)
        return assigned_ids, skipped

    def _show_batch_resolution_warning(self, setup_name, skipped):
        """Summarize videos omitted from a batch due to resolution mismatch."""
        if not skipped:
            return
        video_lines = "\n".join(f"• {video_id}" for video_id, _ in skipped)
        first_reason = skipped[0][1]
        QMessageBox.warning(
            self,
            "Some Videos Were Skipped",
            f'The following videos were not assigned to "{setup_name}" because '
            f"their saved-frame resolution does not match the setup:\n\n"
            f"{video_lines}\n\n{first_reason}"
        )

    def on_new_imaging_setup(self):
        """Create a named setup and assign all selected videos to it."""
        if not self.project_path or not self.current_video_id:
            QMessageBox.information(
                self,
                "Open a Project Video",
                "Open a project video before creating an imaging setup."
            )
            return

        self._load_imaging_setup_config()
        setup_name, accepted = QInputDialog.getText(
            self,
            "New Imaging Setup",
            "Setup name (for example, D2024 paired microcolonies):"
        )
        setup_name = setup_name.strip()
        if not accepted or not setup_name:
            self._apply_video_physical_scale(self.current_video_id)
            return

        existing_names = {
            name.casefold(): name for name in self.imaging_setup_profiles
        }
        if setup_name.casefold() in existing_names:
            existing_name = existing_names[setup_name.casefold()]
            QMessageBox.information(
                self,
                "Setup Already Exists",
                f'An imaging setup named "{existing_name}" already exists. '
                "Select it from the setup list instead."
            )
            self._apply_video_physical_scale(self.current_video_id)
            return

        target_video_ids = self._selected_video_ids()

        previous_setup = self.video_imaging_setups.get(self.current_video_id)
        inherited_scale = (
            self.canvas.pixels_per_cm if previous_setup is None else None
        )
        inherited_record = (
            self._read_video_physical_scale_record(self.current_video_id)
            if inherited_scale is not None else {}
        )
        inherited_known_length = inherited_record.get('calibration_distance_cm')
        inherited_line_length = inherited_record.get('calibration_line_pixels')
        image_height = None
        image_width = None
        if self.canvas.current_image is not None:
            image_height, image_width = self.canvas.current_image.shape[:2]
        timestamp = datetime.now().isoformat()
        self.imaging_setup_profiles[setup_name] = {
            'pixels_per_cm': inherited_scale,
            'cm_per_pixel': (
                1.0 / inherited_scale if inherited_scale is not None else None
            ),
            'created_at': timestamp,
            'updated_at': timestamp,
            'calibration_video_id': (
                self.current_video_id if inherited_scale is not None else None
            ),
            'calibration_line_pixels': inherited_line_length,
            'calibration_distance_cm': inherited_known_length,
            'image_width_pixels': (
                inherited_record.get('image_width_pixels', image_width)
                if inherited_scale is not None else None
            ),
            'image_height_pixels': (
                inherited_record.get('image_height_pixels', image_height)
                if inherited_scale is not None else None
            ),
            'calibration_source': (
                'migrated_video_specific' if inherited_scale is not None else None
            ),
        }
        assigned_ids, skipped = self._assign_videos_to_imaging_setup(
            setup_name, target_video_ids
        )
        self._save_imaging_setup_config()
        self._apply_video_physical_scale(self.current_video_id)
        self.update_video_list()
        assignment_note = (
            f"assigned {len(assigned_ids)} selected video"
            f"{'s' if len(assigned_ids) != 1 else ''}"
        )
        if inherited_scale is not None:
            self.status_label.setText(
                f'Created setup "{setup_name}", {assignment_note}, '
                f"and retained its existing {inherited_scale:.3f} px/cm calibration."
            )
        else:
            self.status_label.setText(
                f'Created setup "{setup_name}" and {assignment_note}. '
                "Draw a known-length line and choose Set Scale from Line."
            )
        self._show_batch_resolution_warning(setup_name, skipped)

    def on_imaging_setup_changed(self, setup_name):
        """Assign all selected videos to a setup selected in the toolbar."""
        if not self.project_path or not self.current_video_id:
            self._apply_video_physical_scale(self.current_video_id)
            return

        self._load_imaging_setup_config()
        setup_name = str(setup_name).strip()
        target_video_ids = self._selected_video_ids()
        if setup_name and setup_name not in self.imaging_setup_profiles:
            self._apply_video_physical_scale(self.current_video_id)
            return

        if setup_name:
            if all(
                self.video_imaging_setups.get(video_id) == setup_name
                for video_id in target_video_ids
            ):
                return
            assigned_ids, skipped = self._assign_videos_to_imaging_setup(
                setup_name, target_video_ids
            )
        else:
            assigned_ids = []
            skipped = []
            for video_id in target_video_ids:
                previous_setup = self.video_imaging_setups.get(video_id)
                if not previous_setup:
                    continue
                previous_values = self._profile_calibration_values(previous_setup)
                self.video_imaging_setups.pop(video_id, None)
                retained_scale = previous_values.get('pixels_per_cm')
                self._save_video_physical_scale(
                    video_id,
                    retained_scale,
                    calibration_line_pixels=previous_values.get(
                        'calibration_line_pixels'
                    ),
                    calibration_distance_cm=previous_values.get(
                        'calibration_distance_cm'
                    ),
                    image_width_pixels=previous_values.get('image_width_pixels'),
                    image_height_pixels=previous_values.get('image_height_pixels'),
                )
                self.video_physical_scales[video_id] = retained_scale
                retained_known_length = previous_values.get(
                    'calibration_distance_cm'
                )
                if retained_known_length is not None:
                    self.video_scale_known_lengths_cm[video_id] = float(
                        retained_known_length
                    )
                else:
                    self.video_scale_known_lengths_cm.pop(video_id, None)
                assigned_ids.append(video_id)

            if not assigned_ids:
                return

        self._save_imaging_setup_config()
        self._apply_video_physical_scale(self.current_video_id)
        self.update_video_list()
        if setup_name:
            pixels_per_cm = self._profile_calibration_values(setup_name)[
                'pixels_per_cm'
            ]
            if pixels_per_cm:
                scale_note = f" using {pixels_per_cm:.3f} px/cm"
            else:
                scale_note = "; this setup still needs calibration"
            plural_suffix = 's' if len(assigned_ids) != 1 else ''
            self.status_label.setText(
                f'Assigned {len(assigned_ids)} selected video'
                f'{plural_suffix} to '
                f'"{setup_name}"{scale_note}.'
            )
            self._show_batch_resolution_warning(setup_name, skipped)
        else:
            self.status_label.setText(
                f"Removed the imaging-setup assignment for {len(assigned_ids)} "
                f"selected video{'s' if len(assigned_ids) != 1 else ''}. "
                "Any inherited scale values were retained as video-specific "
                "calibrations."
            )

    def on_measurement_line_completed(self, length_pixels):
        """Report a completed temporary ruler measurement."""
        if self.canvas.pixels_per_cm:
            length_cm = length_pixels / self.canvas.pixels_per_cm
            self.status_label.setText(
                f"Measured line: {length_pixels:.1f} px = {length_cm:.3f} cm"
            )
        else:
            self.status_label.setText(
                f"Measured line: {length_pixels:.1f} px. "
                "Choose Set Scale from Line to assign its known length in cm."
            )

    def on_set_measurement_scale(self):
        """Calibrate the current video's setup, or the video itself if unassigned."""
        length_pixels = self.canvas.get_measurement_pixel_length()
        if length_pixels <= 0:
            QMessageBox.information(
                self,
                "Draw a Measurement Line",
                "Choose the Measure tool and drag a line over an object of known "
                "length before setting the scale."
            )
            return

        setup_name = self._current_imaging_setup()
        setup_values = (
            self._profile_calibration_values(setup_name) if setup_name else {}
        )
        default_cm = setup_values.get('calibration_distance_cm')
        if default_cm is None:
            default_cm = self.video_scale_known_lengths_cm.get(
                self.current_video_id, 1.0
            )
        known_length_cm, accepted = QInputDialog.getDouble(
            self,
            "Set Physical Scale",
            f"The current line is {length_pixels:.1f} pixels long.\n"
            "Enter the physical length represented by this line (cm):",
            default_cm,
            0.000001,
            1000000.0,
            6
        )
        if not accepted:
            return

        pixels_per_cm = length_pixels / known_length_cm
        if setup_name:
            assigned_count = sum(
                assigned_setup == setup_name
                for assigned_setup in self.video_imaging_setups.values()
            )
            old_scale = setup_values.get('pixels_per_cm')
            if old_scale is not None and assigned_count > 1:
                reply = QMessageBox.question(
                    self,
                    "Update Shared Setup Scale?",
                    f'This will update setup "{setup_name}" from '
                    f'{old_scale:.3f} to {pixels_per_cm:.3f} px/cm for '
                    f"all {assigned_count} assigned videos. Continue?",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                    QMessageBox.StandardButton.No
                )
                if reply != QMessageBox.StandardButton.Yes:
                    return

        self.canvas.set_pixels_per_cm(pixels_per_cm)
        self.toolbar.set_measurement_scale(pixels_per_cm, setup_name)
        if setup_name:
            image_height = None
            image_width = None
            if self.canvas.current_image is not None:
                image_height, image_width = self.canvas.current_image.shape[:2]
            profile = self.imaging_setup_profiles[setup_name]
            profile.update({
                'pixels_per_cm': float(pixels_per_cm),
                'cm_per_pixel': 1.0 / float(pixels_per_cm),
                'calibration_video_id': self.current_video_id,
                'calibration_line_pixels': float(length_pixels),
                'calibration_distance_cm': float(known_length_cm),
                'image_width_pixels': image_width,
                'image_height_pixels': image_height,
                'calibration_source': 'ruler_line',
                'updated_at': datetime.now().isoformat(),
            })
            config_saved = self._save_imaging_setup_config()
            saved_count, assigned_count = (
                self._mirror_setup_calibration_to_assigned_videos(setup_name)
            )
            self.status_label.setText(
                f'Calibrated setup "{setup_name}": {pixels_per_cm:.3f} px/cm; '
                f"applied to {saved_count}/{assigned_count} assigned videos"
                f"{' and saved project setup data' if config_saved else ''}."
            )
        else:
            if self.current_video_id:
                self.video_physical_scales[self.current_video_id] = pixels_per_cm
                self.video_scale_known_lengths_cm[self.current_video_id] = known_length_cm
            saved = self._save_video_physical_scale(
                self.current_video_id,
                pixels_per_cm,
                calibration_line_pixels=length_pixels,
                calibration_distance_cm=known_length_cm
            )
            save_note = "saved for this video" if saved else "active for this session"
            self.status_label.setText(
                f"Video-specific physical scale {save_note}: "
                f"{pixels_per_cm:.3f} px/cm "
                f"(1 px = {1.0 / pixels_per_cm:.6f} cm)"
            )

    def on_clear_measurement_scale(self):
        """Clear a shared setup calibration or a video-specific calibration."""
        setup_name = self._current_imaging_setup()
        if setup_name:
            assigned_count = sum(
                assigned_setup == setup_name
                for assigned_setup in self.video_imaging_setups.values()
            )
            reply = QMessageBox.question(
                self,
                "Clear Shared Setup Scale?",
                f'This will clear the calibration for setup "{setup_name}" and '
                f"all {assigned_count} assigned video"
                f"{'s' if assigned_count != 1 else ''}. The setup assignments "
                "will be retained. Continue?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No
            )
            if reply != QMessageBox.StandardButton.Yes:
                return
            profile = self.imaging_setup_profiles[setup_name]
            profile.update({
                'pixels_per_cm': None,
                'cm_per_pixel': None,
                'calibration_video_id': None,
                'calibration_line_pixels': None,
                'calibration_distance_cm': None,
                'image_width_pixels': None,
                'image_height_pixels': None,
                'calibration_source': None,
                'updated_at': datetime.now().isoformat(),
            })
            self._save_imaging_setup_config()
            saved_count, _ = self._mirror_setup_calibration_to_assigned_videos(
                setup_name
            )
            self.canvas.set_pixels_per_cm(None)
            self.toolbar.set_measurement_scale(None, setup_name)
            self.status_label.setText(
                f'Cleared the scale for setup "{setup_name}" in '
                f"{saved_count} assigned video metadata files."
            )
            return

        self.canvas.set_pixels_per_cm(None)
        self.toolbar.set_measurement_scale(None)
        if self.current_video_id:
            self.video_physical_scales[self.current_video_id] = None
            self.video_scale_known_lengths_cm.pop(self.current_video_id, None)
        removed = self._save_video_physical_scale(self.current_video_id, None)
        if removed:
            self.status_label.setText(
                "Cleared the saved physical scale for this video; measurements show pixels only."
            )
        else:
            self.status_label.setText("Cleared the physical scale for this session.")
    
    def _save_max_mask_id_to_metadata(self, video_id, max_mask_id):
        """
        Save max_mask_id to video metadata for faster loading next time
        
        Args:
            video_id: Video identifier
            max_mask_id: The maximum mask_id found in this video's annotations
        """
        if not self.project_path or not video_id:
            return
        
        metadata_file = self.project_manager.get_frames_dir(video_id) / 'video_metadata.json'
        
        try:
            import json
            # Load existing metadata if present
            metadata = {}
            if metadata_file.exists():
                try:
                    with open(metadata_file, 'r') as f:
                        metadata = json.load(f)
                except (json.JSONDecodeError, ValueError) as e:
                    # If metadata is corrupted, start fresh
                    print(f"Warning: Corrupted metadata file, recreating: {e}")
                    metadata = {}
            
            # Update max_mask_id
            metadata['max_mask_id'] = max_mask_id
            
            # Save back atomically (write to temp file, then rename)
            import tempfile
            temp_file = metadata_file.parent / f'.{metadata_file.name}.tmp'
            try:
                with open(temp_file, 'w') as f:
                    json.dump(metadata, f, indent=2)
                # Atomic rename
                temp_file.replace(metadata_file)
            except Exception as e:
                # Clean up temp file if it exists
                if temp_file.exists():
                    temp_file.unlink()
                raise
        except Exception as e:
            print(f"Warning: Could not save max_mask_id to metadata: {e}")

    def _extract_max_mask_id_from_json_data(self, data):
        """Return the highest bee instance ID in a frame/bbox annotation JSON payload."""
        max_id = 0
        if isinstance(data, dict):
            annotations = data.get('annotations', [])
        elif isinstance(data, list):
            annotations = data
        else:
            annotations = []

        for ann in annotations:
            if not isinstance(ann, dict):
                continue
            if ann.get('category', 'bee') != 'bee':
                continue
            try:
                instance_id = int(ann.get('mask_id', ann.get('instance_id', 0)) or 0)
            except (TypeError, ValueError):
                instance_id = 0
            max_id = max(max_id, instance_id)

        return max_id

    def _recompute_max_mask_id_for_video(self, video_id):
        """Scan saved annotation files and cache to find the real max bee ID."""
        if not self.project_path or not video_id:
            return 0

        import json
        import pickle

        max_id = 0
        project_path = Path(self.project_path)

        # Current in-memory cache may contain unsaved or recently changed frames.
        for cache_key, annotations in self.annotation_manager.frame_annotations.items():
            key_video = cache_key[0] if isinstance(cache_key, tuple) else None
            if key_video != video_id:
                continue
            for ann in annotations:
                if ann.get('category', 'bee') != 'bee':
                    continue
                try:
                    instance_id = int(ann.get('mask_id', ann.get('instance_id', 0)) or 0)
                except (TypeError, ValueError):
                    instance_id = 0
                max_id = max(max_id, instance_id)

        # Frame metadata and bbox-only files are the current primary formats.
        for subdir in ('json', 'bbox'):
            annotations_dir = project_path / 'annotations' / subdir / video_id
            if not annotations_dir.exists():
                continue
            for ann_file in annotations_dir.glob('frame_*.json'):
                try:
                    with open(ann_file, 'r') as f:
                        data = json.load(f)
                    max_id = max(max_id, self._extract_max_mask_id_from_json_data(data))
                except Exception as e:
                    print(f"Warning: Could not scan {ann_file} for max_mask_id: {e}")

        # Legacy pickle annotations may still exist in older projects.
        pkl_dir = project_path / 'annotations' / 'pkl' / video_id
        if pkl_dir.exists():
            for ann_file in pkl_dir.glob('frame_*.pkl'):
                try:
                    with open(ann_file, 'rb') as f:
                        annotations = pickle.load(f)
                    max_id = max(max_id, self._extract_max_mask_id_from_json_data(annotations))
                except Exception as e:
                    print(f"Warning: Could not scan {ann_file} for max_mask_id: {e}")

        return max_id

    def _refresh_video_next_mask_id(self, video_id):
        """Refresh next instance ID from saved annotations after deletions."""
        max_id = self._recompute_max_mask_id_for_video(video_id)
        next_id = max_id + 1
        self.video_next_mask_id[video_id] = next_id
        self._save_max_mask_id_to_metadata(video_id, max_id)
        if self.current_video_id == video_id:
            self.canvas.next_mask_id = next_id
        print(f"Refreshed max_mask_id for video {video_id}: {max_id} (next={next_id})")
        return next_id
    
    def _get_frame_idx_in_video(self, list_idx):
        """
        Get the frame index within the video for a given list index
        
        In the new structure, frames are named frame_XXXXXX.jpg where XXXXXX
        is the frame index within the video. We extract this from the frame path.
        """
        if list_idx >= len(self.frames):
            return list_idx
        
        frame = self.frames[list_idx]
        if isinstance(frame, (Path, str)):
            # Extract frame number from filename: frame_000001.jpg -> 1
            frame_path = Path(frame)
            if frame_path.stem.startswith('frame_'):
                try:
                    return int(frame_path.stem.split('_')[1])
                except (ValueError, IndexError):
                    pass
        
        # Fallback to list index
        return list_idx

    def _frame_annotation_key(self, video_id=None, list_idx=None):
        """Return the dirty-tracking key for a frame annotation file."""
        target_video_id = video_id or self.current_video_id
        if not target_video_id:
            return None

        target_list_idx = self.current_frame_idx if list_idx is None else list_idx
        if target_list_idx is None:
            return None

        return (target_video_id, self._get_frame_idx_in_video(target_list_idx))

    def _split_source_annotations(self, annotations):
        """Split canvas annotations into per-frame and video-level sources."""
        return split_annotations(annotations, self.annotation_manager.project_info)

    def _mark_current_annotations_dirty(self, include_frame=True, include_video=True):
        """Mark the current annotation source files as needing a save."""
        self.current_frame_modified = True
        self.coco_export_dirty = True

        if include_frame:
            frame_key = self._frame_annotation_key()
            if frame_key is not None:
                self.dirty_frame_annotation_keys.add(frame_key)

        if include_video and self.current_video_id:
            self.dirty_video_annotation_ids.add(self.current_video_id)

    def _commit_canvas_edit_if_needed(self):
        """Commit an active canvas edit and mark it dirty if it contains pixels."""
        if not hasattr(self, 'canvas') or self.canvas.editing_instance_id <= 0:
            return False

        had_edit_pixels = (
            self.canvas.editing_mask is not None
            and np.any(self.canvas.editing_mask > 0)
        )
        self.canvas.commit_editing()
        if had_edit_pixels:
            self._mark_current_annotations_dirty()
        return had_edit_pixels

    def _save_annotation_sources(self, video_id=None, list_idx=None, annotations=None,
                                 force=False, use_background=False, refresh_next_id=False):
        """Save dirty source annotation files without regenerating derived COCO files."""
        if not self.project_path:
            return {'saved_any': False, 'reason': 'no_project'}

        if not use_background:
            self.save_worker.wait_until_idle(raise_errors=False)

        target_video_id = video_id or self.current_video_id
        if not target_video_id:
            return {'saved_any': False, 'reason': 'no_video'}

        target_list_idx = self.current_frame_idx if list_idx is None else list_idx
        frame_key = self._frame_annotation_key(target_video_id, target_list_idx)
        frame_idx_in_video = frame_key[1] if frame_key is not None else target_list_idx

        if annotations is None:
            self._commit_canvas_edit_if_needed()
            annotations = self.canvas.get_annotations()

        save_frame = (
            force
            or self.current_frame_modified
            or frame_key in self.dirty_frame_annotation_keys
        )
        save_video = (
            force
            or self.current_frame_modified
            or target_video_id in self.dirty_video_annotation_ids
        )

        if not save_frame and not save_video:
            return {
                'saved_any': False,
                'reason': 'clean',
                'frame_idx': frame_idx_in_video,
                'video_id': target_video_id,
            }

        frame_annotations, video_level_annotations = self._split_source_annotations(annotations)
        saved_files = []

        if save_frame:
            self.annotation_manager.set_frame_annotations(
                target_list_idx, frame_annotations, video_id=target_video_id
            )
            if use_background:
                self.save_worker.add_save_task(
                    self.project_path, target_video_id,
                    frame_idx_in_video, frame_annotations
                )
            else:
                saved = self.annotation_manager.save_frame_annotations(
                    self.project_path, target_video_id,
                    frame_idx_in_video, frame_annotations
                )
                self._receive_saved_attribution(str(self.project_path), target_video_id,
                                                frame_idx_in_video, provenance_records(saved))
                self.save_worker.failed_tasks.pop(
                    (str(self.project_path), target_video_id, frame_idx_in_video), None)
            saved_files.append('frame')
            if frame_key is not None:
                self.dirty_frame_annotation_keys.discard(frame_key)

        if save_video:
            saved = self.annotation_manager.save_video_annotations(
                self.project_path, target_video_id, video_level_annotations
            )
            self._receive_saved_attribution(str(self.project_path), target_video_id,
                                            -1, provenance_records(saved))
            saved_files.append('video-level')
            self.dirty_video_annotation_ids.discard(target_video_id)

        if refresh_next_id:
            self._refresh_video_next_mask_id(target_video_id)

        if saved_files:
            self.coco_export_dirty = True
            self._schedule_done_counter_update(saved_counts_dirty=True)

        current_key = self._frame_annotation_key()
        if target_video_id == self.current_video_id and frame_key == current_key:
            self.current_frame_modified = False

        return {
            'saved_any': bool(saved_files),
            'saved_files': saved_files,
            'frame_idx': frame_idx_in_video,
            'video_id': target_video_id,
            'bee_count': sum(ann.get('category', 'bee') == 'bee' for ann in frame_annotations),
            'frame_count': len(frame_annotations),
            'video_level_count': len(video_level_annotations),
        }
    
    def _queue_tracked_bee_annotations(self, list_idx, annotations):
        """Replace tracked bees without erasing other frame-local labels."""
        frame_idx = self._get_frame_idx_in_video(list_idx)
        key = (self.current_video_id, list_idx)
        if key in self.annotation_manager.frame_annotations:
            existing = self.annotation_manager.get_frame_annotations(list_idx, video_id=self.current_video_id)
        else:
            existing = self.annotation_manager.load_frame_annotations(
                self.project_path, self.current_video_id, frame_idx
            ) or []
        existing_frame, _ = self._split_source_annotations(existing)
        preserved = [ann for ann in existing_frame if ann.get('category', 'bee') != 'bee']
        reserved = {ann.get('mask_id', ann.get('instance_id')) for ann in preserved}
        if any(ann.get('mask_id', ann.get('instance_id')) in reserved for ann in annotations):
            raise ValueError('Tracked bee IDs conflict with existing frame-local nest IDs; '
                             'the frame annotations have not been overwritten.')
        combined = preserved + annotations
        self.annotation_manager.set_frame_annotations(list_idx, combined, video_id=self.current_video_id)
        self.save_worker.add_save_task(self.project_path, self.current_video_id, frame_idx, combined)

    def _save_video_level_annotations(self, video_id=None):
        """Save annotations whose project scope is shared across the video.

        Args:
            video_id: The video ID to save for. Defaults to self.current_video_id.
        """
        if not self.project_path:
            return
        target_video_id = video_id or self.current_video_id
        if not target_video_id:
            return
        try:
            all_annotations = self.canvas.get_annotations()
            _, video_anns = self._split_source_annotations(all_annotations)
            self.annotation_manager.save_video_annotations(
                self.project_path, target_video_id, video_anns
            )
        except Exception as e:
            print(f"Warning: Failed to save video-level annotations: {e}")

    def _convert_segmentations_to_bboxes(self, segmentation_annotations):
        """Convert segmentation annotations to bbox-only annotations
        
        Args:
            segmentation_annotations: List of annotation dicts with masks
            
        Returns:
            List of bbox-only annotation dicts (no masks)
        """
        bbox_annotations = []
        for ann in segmentation_annotations:
            # Create bbox-only annotation (remove masks)
            bbox_ann = {k: v for k, v in ann.items() if k not in ['mask', 'mask_rle']}
            
            # Compute bbox from mask if not present or invalid
            if 'bbox' not in bbox_ann or bbox_ann.get('bbox') == [0, 0, 0, 0]:
                if 'mask' in ann:
                    # Compute bbox from mask
                    mask = ann['mask']
                    y_indices, x_indices = np.where(mask > 0)
                    if len(y_indices) > 0:
                        x_min = int(x_indices.min())
                        x_max = int(x_indices.max())
                        y_min = int(y_indices.min())
                        y_max = int(y_indices.max())
                        bbox_ann['bbox'] = [x_min, y_min, x_max - x_min + 1, y_max - y_min + 1]
                    else:
                        bbox_ann['bbox'] = [0, 0, 0, 0]
            
            bbox_annotations.append(bbox_ann)
        
        return bbox_annotations
    
    def _get_next_frame_index(self):
        """
        Get the next frame index based on current filter
        
        If filter is active, returns next frame in filtered list.
        Otherwise, returns next sequential frame.
        
        Returns None if at last frame.
        """
        try:
            # Find current frame in the mapping
            current_position = self.frame_list_to_frames_map.index(self.current_frame_idx)
            
            # Check if there's a next frame in filtered list
            if current_position < len(self.frame_list_to_frames_map) - 1:
                return self.frame_list_to_frames_map[current_position + 1]
            else:
                return None
                
        except (ValueError, IndexError):
            # Current frame not in filtered view, or error occurred
            # Fall back to sequential next frame if no filter
            if self.split_filter == 'all' and self.current_frame_idx < len(self.frames) - 1:
                return self.current_frame_idx + 1
            return None
    
    def _get_prev_frame_index(self):
        """
        Get the previous frame index based on current filter
        
        If filter is active, returns previous frame in filtered list.
        Otherwise, returns previous sequential frame.
        
        Returns None if at first frame.
        """
        try:
            # Find current frame in the mapping
            current_position = self.frame_list_to_frames_map.index(self.current_frame_idx)
            
            # Check if there's a previous frame in filtered list
            if current_position > 0:
                return self.frame_list_to_frames_map[current_position - 1]
            else:
                return None
                
        except (ValueError, IndexError):
            # Current frame not in filtered view, or error occurred
            # Fall back to sequential previous frame if no filter
            if self.split_filter == 'all' and self.current_frame_idx > 0:
                return self.current_frame_idx - 1
            return None
    
    def on_split_filter_changed(self, text):
        """Handle split filter change"""
        if text == 'All Frames':
            self.split_filter = 'all'
        elif text == 'Training Only':
            self.split_filter = 'train'
        elif text == 'Validation Only':
            self.split_filter = 'val'
        elif text == 'Validation Masks Only':
            self.split_filter = 'val_masks'
        elif text == 'Test Only':
            self.split_filter = 'test'
        elif text == 'Inference Only':
            self.split_filter = 'inference'
        
        # Refresh frame list with new filter
        self.update_frame_list()
        
    def new_project(self):
        """Create a new project"""
        dialog = ProjectDialog(self, mode='new', default_dir=self.default_projects_dir)
        if dialog.exec():
            project_info = dialog.get_project_info()
            project_name = project_info['name']
            self._reset_imaging_setup_cache()
            
            # Create project in a subfolder with the project name
            base_path = Path(project_info['path'])
            self.project_path = base_path / project_name
            
            # Use ProjectManager to create new project structure
            self.project_manager.create_project(
                self.project_path, 
                project_name,
                frames_per_video=15,  # Default, user can change per video
                hive_annotation_scope=project_info['hive_annotation_scope']
            )
            
            # Set marker detector debug folder
            debug_folder = self.project_path / 'marker_debug'
            self.marker_detector.set_debug_folder(str(debug_folder))
            
            self.annotation_manager.new_project(project_info)
            self.status_label.setText(
                f"Project created: {self.project_path.name} "
                f"(ready to add videos)"
            )
            self._reset_done_counter_baseline()
            self.add_recent_project(self.project_path)
            
    def open_project(self):
        """Open an existing project"""
        path = QFileDialog.getExistingDirectory(
            self, "Open Project", 
            str(self.default_projects_dir)
        )
        if path:
            self.load_project(path)

    def update_recent_projects_menu(self):
        """Refresh the File > Recent Projects submenu."""
        if not hasattr(self, 'recent_projects_menu'):
            return

        self.recent_projects_menu.clear()

        existing_projects = []
        for project in self.recent_projects:
            project_path = Path(project)
            if project_path.exists():
                existing_projects.append(str(project_path))

        if existing_projects != self.recent_projects:
            self.recent_projects = existing_projects
            self._save_recent_projects()

        if not self.recent_projects:
            empty_action = QAction("No Recent Projects", self)
            empty_action.setEnabled(False)
            self.recent_projects_menu.addAction(empty_action)
            return

        for project in self.recent_projects:
            project_path = Path(project)
            action = QAction(project_path.name, self)
            action.setToolTip(str(project_path))
            action.triggered.connect(
                lambda checked=False, path=str(project_path): self.open_recent_project(path)
            )
            self.recent_projects_menu.addAction(action)

        self.recent_projects_menu.addSeparator()

        clear_action = QAction("Clear Recent Projects", self)
        clear_action.triggered.connect(self.clear_recent_projects)
        self.recent_projects_menu.addAction(clear_action)

    def add_recent_project(self, path):
        """Add a project path to the recent projects list."""
        project_path = str(Path(path))
        self.recent_projects = [
            recent for recent in self.recent_projects
            if str(Path(recent)) != project_path
        ]
        self.recent_projects.insert(0, project_path)
        self.recent_projects = self.recent_projects[:10]
        self._save_recent_projects()
        self.update_recent_projects_menu()

    def open_recent_project(self, path):
        """Open a project from the recent projects menu."""
        project_path = Path(path)
        if not project_path.exists():
            QMessageBox.warning(
                self,
                "Project Not Found",
                f"Recent project no longer exists:\n{project_path}"
            )
            self.recent_projects = [
                recent for recent in self.recent_projects
                if str(Path(recent)) != str(project_path)
            ]
            self._save_recent_projects()
            self.update_recent_projects_menu()
            return

        self.load_project(project_path)

    def clear_recent_projects(self):
        """Clear the recent projects menu."""
        self.recent_projects = []
        self._save_recent_projects()
        self.update_recent_projects_menu()

    def _save_recent_projects(self):
        """Persist recent projects to application settings."""
        settings = QSettings()
        settings.setValue('recent_projects', self.recent_projects)
            
    def load_project(self, path):
        """Load a project from path"""
        self.project_path = Path(path)
        self._reset_imaging_setup_cache()
        
        # Set project path in ProjectManager
        self.project_manager.project_path = self.project_path
        
        # Initialize tracking sequence manager for this project
        self.tracking_sequence_manager = TrackingSequenceManager(self.project_path)
        
        # Set managers in tracking sequences panel
        if hasattr(self, 'tracking_sequences_panel'):
            self.tracking_sequences_panel.set_managers(
                self.tracking_sequence_manager,
                self.annotation_manager
            )
        
        # Set marker detector debug folder
        debug_folder = self.project_path / 'marker_debug'
        self.marker_detector.set_debug_folder(str(debug_folder))
        
        try:
            # Load project annotations
            self.annotation_manager.load_project(self.project_path)
            
            # Update video list
            self.update_video_list()
            
            # Load frames from project
            self.load_frames_from_project()
            
            # Count loaded annotations
            num_annotated_frames = len([f for f in self.annotation_manager.frame_annotations.values() if f])
            
            self.status_label.setText(
                f"Project loaded: {self.project_path.name} "
                f"({len(self.frames)} frames, {num_annotated_frames} annotated)"
            )
            self._reset_done_counter_baseline()
            self.add_recent_project(self.project_path)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load project: {str(e)}")
            import traceback
            traceback.print_exc()
            
    def import_project_data(self):
        """Import saved copies, then discard stale canvas/cache data before reload."""
        if not self.project_path:
            QMessageBox.information(self, 'No Project', 'Open the destination project before importing.')
            return
        if self.is_playing:
            self.toggle_play()
        try:
            self.save_worker.wait_until_idle(raise_errors=False)
            self._save_annotation_sources(refresh_next_id=True)
            self.save_worker.wait_until_idle()
        except Exception as exc:
            QMessageBox.warning(self, 'Save Failed', str(exc))
            return
        # A background annotation read must not repopulate old metadata after import.
        self.preload_worker.stop()
        self.preload_worker.join()
        try:
            dialog = ProjectImportDialog(self.project_path, self.contributor_session, self)
            if dialog.exec() != QDialog.DialogCode.Accepted or not dialog.result:
                return
            self.current_frame_modified = False
            self.dirty_frame_annotation_keys.clear()
            self.dirty_video_annotation_ids.clear()
            self.current_video_id = None
            self.current_frame_idx = -1
            self.canvas.clear_image()
            self.annotation_manager.clear_cache()
            self.annotation_manager.attributor.baselines.clear()
            self.frame_cache.clear()
            self.video_next_mask_id.clear()
            self.load_project(self.project_path)
            self.status_label.setText('Project data imported')
            QMessageBox.information(self, 'Import Complete',
                                    'Copied frames and annotations are ready for review.\n'
                                    'Re-export COCO before training.\n\n'
                                    'Import record and originals:\n' + dialog.result['backup'])
        finally:
            self.preload_worker = PreloadWorker(self.frame_cache, self.annotation_manager)
            self.preload_worker.start()

    def import_video(self):
        """Import video and extract frames"""
        video_path, _ = QFileDialog.getOpenFileName(
            self, "Import Video", "",
            "Video Files (*.mp4 *.avi *.mov *.mkv *.mjpeg *.mjpg);;All Files (*)"
        )
        
        if video_path:
            self.load_video(video_path)
    
    def show_add_video_dialog(self):
        """Show dialog to add video to current project"""
        if not self.project_path:
            QMessageBox.warning(
                self, "No Project",
                "Please create or open a project first before adding videos."
            )
            return
        
        # Select video file
        video_path, _ = QFileDialog.getOpenFileName(
            self, "Select Video to Add", "",
            "Video Files (*.mp4 *.avi *.mov *.mkv *.mjpeg *.mjpg);;All Files (*)"
        )
        
        if not video_path:
            return
        
        dialog = ProjectVideoImportDialog(self)
        if dialog.exec():
            self.add_video_to_project(video_path, **dialog.get_settings())
            
    def load_video(self, video_path):
        """Load video and extract frames, or load existing project if available"""
        video_path_obj = Path(video_path)
        project_name = video_path_obj.stem  # Video filename without extension
        
        # Project directory is next to the video file with same basename
        project_dir = video_path_obj.parent / f"{project_name}_project"
        
        # Check if project already exists
        if project_dir.exists() and (project_dir / 'frames').exists():
            # Project exists - ask user if they want to load it
            frames_dir = project_dir / 'frames'
            num_frames = len(list(frames_dir.glob('*.png')) + list(frames_dir.glob('*.jpg')))
            
            reply = QMessageBox.question(
                self,
                "Existing Project Found",
                f"Found existing project for this video:\n\n"
                f"Location: {project_dir}\n"
                f"Frames: {num_frames}\n\n"
                f"Would you like to load the existing project?\n\n"
                f"Click 'Yes' to load existing project and annotations.\n"
                f"Click 'No' to re-extract frames (existing data will be overwritten).",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No | QMessageBox.StandardButton.Cancel,
                QMessageBox.StandardButton.Yes
            )
            
            if reply == QMessageBox.StandardButton.Cancel:
                return
            elif reply == QMessageBox.StandardButton.Yes:
                # Load existing project
                try:
                    self.project_path = project_dir
                    self._reset_imaging_setup_cache()
                    self.annotation_manager.load_project(project_dir)
                    self.load_frames_from_project()
                    
                    # Count annotated frames
                    annotated_count = len([f for f in self.annotation_manager.frame_annotations.values() if f])
                    self.status_label.setText(
                        f"✓ Loaded project: {num_frames} frames, {annotated_count} annotated"
                    )
                    self._reset_done_counter_baseline()
                    return
                except Exception as e:
                    QMessageBox.warning(
                        self, 
                        "Load Failed", 
                        f"Failed to load existing project:\n{str(e)}\n\nWill re-extract frames."
                    )
                    # Fall through to re-extract
        
        # Either no project exists, or user chose to re-extract
        dialog = VideoImportDialog(self, video_path)
        if dialog.exec():
            import_settings = dialog.get_settings()
            
            # Create project structure
            project_dir.mkdir(parents=True, exist_ok=True)
            (project_dir / 'frames').mkdir(exist_ok=True)
            (project_dir / 'annotations').mkdir(exist_ok=True)
            
            self.project_path = project_dir
            self._reset_imaging_setup_cache()
            
            # Initialize annotation manager with new project
            self.annotation_manager.new_project({
                'name': project_name,
                'video_path': str(video_path)
            })
            
            self.status_label.setText(f"Extracting frames to {project_dir.name}...")
            QApplication.processEvents()
            
            try:
                self.frames = self.video_processor.extract_frames(
                    video_path,
                    output_dir=self.project_path / 'frames',
                    **import_settings
                )
                
                # Save initial project structure
                self.annotation_manager.save_project(self.project_path)
                
                self.update_frame_list()
                if self.frames:
                    self.load_frame(0)
                    
                self.status_label.setText(f"✓ Created project: {len(self.frames)} frames extracted to {project_dir.name}")
                self._reset_done_counter_baseline()
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to extract frames: {str(e)}")
                self.status_label.setText("Ready")
    
    def add_video_to_project(self, video_path, split='train', n_selected=15,
                             extract_all=True):
        """
        Copy a video to the project and extract selected frames (or all frames).
        
        Args:
            video_path: Path to video file
            split: 'train', 'val', 'test', or 'inference'
            n_selected: Number of frames to mark for the chosen split
            extract_all: Also extract frames outside the selected subset
        """
        if not self.project_path:
            QMessageBox.warning(self, "No Project", "Please create or open a project first")
            return
        
        try:
            video_path = Path(video_path)
            
            # Copy or move video to appropriate split folder
            result = self.project_manager.add_videos(
                [video_path],
                split=split,
                copy_to_project=True  # Copy instead of move to be safe
            )
            
            if not result['added']:
                error_msg = '\n'.join([f"{Path(f['video_path']).stem}: {f['error']}" 
                                      for f in result['failed']])
                QMessageBox.warning(
                    self, "Add Failed",
                    f"Could not add video:\n{error_msg}"
                )
                return
            
            video_id = result['added'][0]['video_id']
            
            # Extract frames
            self.status_label.setText(f"Extracting frames from {video_id}...")
            QApplication.processEvents()
            
            # Get video metadata to determine total frames
            metadata = self.project_manager.get_video_metadata(video_id)
            if not metadata:
                raise ValueError(f"Could not read video metadata for {video_id}")
            
            total_frames = metadata['total_frames']
            if total_frames <= 0:
                raise ValueError(f"No readable frames found in {video_id}")
            selected_indices = self.project_manager.select_frames_uniform(
                total_frames, n_selected
            )
            frame_indices = list(range(total_frames)) if extract_all else selected_indices
            
            extract_result = self.project_manager.extract_video_frames(
                video_id, frame_indices
            )
            
            extracted_indices = set(extract_result['frame_indices'])
            selected_indices = [idx for idx in selected_indices if idx in extracted_indices]
            
            # Save metadata about which frames are selected for training/validation
            video_metadata_file = self.project_manager.get_frames_dir(video_id) / 'video_metadata.json'
            import json
            with open(video_metadata_file, 'w') as f:
                json.dump({
                    'split': split,
                    'total_frames': total_frames,
                    'selected_frames': selected_indices,
                    'n_selected': len(selected_indices),
                    'extraction_mode': 'all' if extract_all else 'selected',
                    'extracted_frame_count': extract_result['extracted'],
                }, f, indent=2)
            
            self.status_label.setText(
                f"✓ Added {video_id} to {split}: "
                f"{extract_result['extracted']} frames extracted, "
                f"{len(selected_indices)} marked for {split}"
            )

            if extract_result['failed']:
                QMessageBox.warning(
                    self, "Incomplete Frame Extraction",
                    f"Extracted {extract_result['extracted']} of {len(frame_indices)} requested frames.\n"
                    f"{extract_result['failed']} frames could not be read or saved.\n"
                    "Only successfully extracted frames were selected. The video copy was kept."
                )
            
            # Update video list to show newly added video
            self.update_video_list()
            
            # Load this video's frames for annotation
            if extract_result['extracted']:
                self.load_video_frames(video_id)
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to add video: {str(e)}")
            import traceback
            traceback.print_exc()
    
    def load_video_frames(self, video_id):
        """Load frames from a specific video for annotation"""
        if not self.project_path:
            return
        
        # Create progress dialog for loading feedback
        from PyQt6.QtWidgets import QProgressDialog
        progress = QProgressDialog(f"Loading video {video_id}...", None, 0, 0, self)
        progress.setWindowTitle("Loading Video")
        progress.setWindowModality(Qt.WindowModality.WindowModal)
        progress.setMinimumDuration(200)  # Show after 200ms if still loading
        progress.setValue(0)
        QApplication.processEvents()
        
        try:
            # Save current frame annotations before switching videos
            if self.current_video_id and self.current_video_id != video_id:
                try:
                    self._commit_canvas_edit_if_needed()
                    save_summary = self._save_annotation_sources(
                        video_id=self.current_video_id,
                        list_idx=self.current_frame_idx,
                        annotations=self.canvas.get_annotations(),
                        use_background=False,
                        refresh_next_id=False
                    )
                    if save_summary.get('saved_any'):
                        print(
                            f"Saved {save_summary['video_id']} frame "
                            f"{save_summary['frame_idx']} before switching videos"
                        )
                except Exception as e:
                    print(f"Warning: Failed to save before switching videos: {e}")
                
                # Clear annotation cache before loading new video
                self.annotation_manager.clear_cache()
                
                # Clear canvas to remove masks from previous video
                self.canvas.clear_image()
                
                # Clear frame cache when switching videos
                self.frame_cache.clear()
            
            self.current_video_id = video_id
            
            # Save project state when switching videos
            self._save_project_state()
            
            # Get frame paths
            frames_dir = self.project_manager.get_frames_dir(video_id)
            if not frames_dir.exists():
                QMessageBox.warning(
                    self, "No Frames",
                    f"No frames found for {video_id}.\nPlease extract frames first."
                )
                return
            
            # Load all frame paths
            frame_files = sorted(frames_dir.glob('frame_*.jpg'))
            split = self.project_manager.get_video_split(video_id)
            
            # Load metadata to see which frames are selected
            import json
            metadata_file = frames_dir / 'video_metadata.json'
            selected_indices = set()
            if metadata_file.exists():
                try:
                    with open(metadata_file) as f:
                        metadata = json.load(f)
                        selected_indices = set(metadata.get('selected_frames', []))
                except (json.JSONDecodeError, ValueError) as e:
                    print(f"Warning: Corrupted metadata file for {video_id}, ignoring: {e}")
                    # Continue with empty selected_indices
            
            self.frames = frame_files
            self.frame_video_ids = [video_id] * len(frame_files)
            self.frame_splits = [split] * len(frame_files)
            # Mark which frames are selected for training/validation
            self.frame_selected = [
                self._get_frame_idx_in_video(i) in selected_indices
                for i in range(len(frame_files))
            ]
            
            progress.setLabelText(f"Loading video {video_id}... (updating UI)")
            QApplication.processEvents()
            
            # Update UI
            self.update_video_list()  # Refresh video list to update current video highlighting
            self.update_frame_list()
            
            # Update tracking sequences panel for new video
            if hasattr(self, 'tracking_sequences_panel'):
                if not self.tracking_sequence_manager:
                    # Manager not initialized yet - skip update
                    pass
                else:
                    # Ensure managers are set on panel (defensive programming)
                    self.tracking_sequences_panel.set_managers(
                        self.tracking_sequence_manager,
                        self.annotation_manager
                    )
                    # Update panel with new video
                    self.tracking_sequences_panel.set_video(video_id)
                    # Initialize with first frame (will be updated when load_frame(0) is called)
                    if self.frames:
                        first_frame_idx = self._get_frame_idx_in_video(0)
                        self.tracking_sequences_panel.set_current_frame(first_frame_idx)
            
            progress.setLabelText(f"Loading video {video_id}... (loading first frame)")
            QApplication.processEvents()
            
            if self.frame_list_to_frames_map:
                self.load_frame(self.frame_list_to_frames_map[0])
            elif self.frames:
                self.load_frame(0)
            
            self.status_label.setText(
                f"Loaded {video_id} ({split}): {len(self.frames)} frames"
            )
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load video frames: {str(e)}")
            import traceback
            traceback.print_exc()
        finally:
            progress.close()
    
    def preload_first_frames(self):
        """Preload first frame from each video for faster video switching"""
        if not self.project_path:
            return
        
        try:
            frames_dir = self.project_path / 'frames'
            if not frames_dir.exists():
                return
            
            # Get all video subdirectories
            video_dirs = [d for d in frames_dir.iterdir() if d.is_dir()]
            
            if not video_dirs:
                return
            
            # Preload first frame from each video in background
            for video_dir in video_dirs:
                # Get first frame
                frame_files = sorted(video_dir.glob('frame_*.jpg'))
                if frame_files:
                    first_frame_path = frame_files[0]
                    # Try to load and cache it
                    try:
                        import cv2
                        image = cv2.imread(str(first_frame_path), cv2.IMREAD_GRAYSCALE)
                        if image is not None and image.size > 0:
                            # Loaded as grayscale to save memory
                            # Preloading happens in background worker
                            pass
                    except Exception as e:
                        print(f"Failed to preload first frame from {video_dir.name}: {e}")
        except Exception as e:
            print(f"Error preloading first frames: {e}")
                
    def update_frame_list(self):
        """Update the frame list widget with split indicators"""
        self.frame_list.clear()
        self.frame_list_to_frames_map = []  # Reset mapping
        
        for i, frame_path in enumerate(self.frames):
            # Get split and selection status for this frame
            split = self.frame_splits[i] if i < len(self.frame_splits) else 'unknown'
            is_selected = self.frame_selected[i] if i < len(self.frame_selected) else False
            
            # Apply filter (only affects selected frames)
            if self.split_filter == 'train' and (not is_selected or split != 'train'):
                continue
            elif self.split_filter == 'val' and (not is_selected or split != 'val'):
                continue
            elif self.split_filter == 'val_masks':
                video_id = (
                    self.frame_video_ids[i]
                    if i < len(self.frame_video_ids)
                    else self.current_video_id
                )
                try:
                    frame_idx_in_video = int(Path(frame_path).stem.split('_')[1])
                except (ValueError, IndexError):
                    frame_idx_in_video = i
                if (
                    not is_selected
                    or split != 'val'
                    or not video_id
                    or not frame_has_bee_mask_annotation(
                        self.project_path, video_id, frame_idx_in_video
                    )
                ):
                    continue
            elif self.split_filter == 'test' and (not is_selected or split != 'test'):
                continue
            elif self.split_filter == 'inference' and (not is_selected or split != 'inference'):
                continue
            
            # Create label with split indicator (only for selected frames)
            if is_selected and split != 'unknown':
                split_tag = f"[{split.upper()}] "
            else:
                split_tag = ""
            label = f"{split_tag}Frame {i:04d}"
            self.frame_list.addItem(label)
            
            # Store mapping from list row to actual frame index
            self.frame_list_to_frames_map.append(i)
        
        # Update slider range to match the filtered frame list
        if len(self.frame_list_to_frames_map) > 0:
            self.frame_slider.setMaximum(len(self.frame_list_to_frames_map) - 1)
            self.frame_slider.setEnabled(True)
        else:
            self.frame_slider.setMaximum(0)
            self.frame_slider.setEnabled(False)
        self._update_frame_navigation_buttons()

    def _update_frame_navigation_buttons(self):
        """Enable frame navigation buttons based on the current filtered view."""
        if not hasattr(self, 'prev_frame_button') or not hasattr(self, 'next_frame_button'):
            return

        has_frames = bool(self.frame_list_to_frames_map)
        self.prev_frame_button.setEnabled(has_frames and self._get_prev_frame_index() is not None)
        self.next_frame_button.setEnabled(has_frames and self._get_next_frame_index() is not None)
            
    def load_frame(self, idx):
        """Load a specific frame"""
        if 0 <= idx < len(self.frames):
            # IMPORTANT: Capture the video ID for the current frame BEFORE any changes
            # This must be done first because self.current_video_id and self.frame_video_ids
            # may be updated when switching videos
            old_video_id = None
            if self.current_frame_idx < len(self.frame_video_ids):
                old_video_id = self.frame_video_ids[self.current_frame_idx]
            if old_video_id is None:
                old_video_id = self.current_video_id
            target_video_id = self.frame_video_ids[idx] if idx < len(self.frame_video_ids) else self.current_video_id
            preserve_canvas_view = (
                self.canvas.current_image is not None
                and self.current_frame_idx != idx
                and old_video_id == target_video_id
            )
            
            # Always commit any pending edits before navigating away, regardless of
            # current_frame_modified.  SAM2 predictions go directly into editing_mask
            # without emitting annotation_changed, so current_frame_modified can be
            # False even though there is unsaved work in the editing mask.
            if self.current_frame_idx != idx and self.canvas.editing_instance_id > 0:
                had_edit_pixels = (
                    self.canvas.editing_mask is not None
                    and np.any(self.canvas.editing_mask > 0)
                )
                self.canvas.commit_editing()
                if had_edit_pixels:
                    self._mark_current_annotations_dirty()

            # Auto-save current frame annotations before loading new frame (only if modified)
            if self.current_frame_idx != idx and self.project_path and self.current_frame_modified:
                try:
                    if old_video_id:  # Only save if we know the video
                        # Update video next_mask_id tracking
                        if old_video_id in self.video_next_mask_id:
                            self.video_next_mask_id[old_video_id] = self.canvas.next_mask_id

                        self._save_annotation_sources(
                            video_id=old_video_id,
                            list_idx=self.current_frame_idx,
                            annotations=self.canvas.get_annotations(),
                            use_background=True,
                            refresh_next_id=False
                        )
                except Exception as e:
                    print(f"Warning: Failed to queue save: {e}")
            
            self.current_frame_idx = idx
            
            # Update current video ID from frame_video_ids list
            if idx < len(self.frame_video_ids):
                prev_video_id = self.current_video_id
                self.current_video_id = self.frame_video_ids[idx]
                
                # Get actual frame number within video (not list index)
                frame_idx_in_video = self._get_frame_idx_in_video(idx)
                
                # Update tracking sequences panel if video changed or this is first video load
                if hasattr(self, 'tracking_sequences_panel') and self.tracking_sequence_manager:
                    if prev_video_id != self.current_video_id:
                        # Video changed or initial load - update panel
                        self.tracking_sequences_panel.set_video(self.current_video_id)
                        self.tracking_sequences_panel.set_current_frame(frame_idx_in_video)
                    else:
                        # Same video, just update current frame
                        self.tracking_sequences_panel.set_current_frame(frame_idx_in_video)
            
            # Request preloading of nearby frames in background
            selected_indices = [i for i, selected in enumerate(self.frame_selected) if selected] if self.frame_selected else []
            self.preload_worker.request_preload(
                current_idx=idx,
                frame_paths=self.frames,
                selected_indices=selected_indices,
                project_path=self.project_path,
                video_id=self.current_video_id,
                frame_video_ids=self.frame_video_ids
            )
            
            # Try to get frame from cache first
            cached_image = self.frame_cache.get(idx)
            if cached_image is not None:
                # Use cached image (already in RGB format)
                image_to_load = cached_image
            else:
                # Not in cache - will need to load from disk
                frame = self.frames[idx]
                
                # Check if frame is a file path or numpy array
                if isinstance(frame, Path) or isinstance(frame, str):
                    # It's a file path
                    if not Path(frame).exists():
                        QMessageBox.warning(self, "Error", f"Frame file not found: {frame}")
                        return
                
                image_to_load = frame
            
            try:
                # Pass frame to canvas (can be path or array)
                self.canvas.load_image(image_to_load, preserve_view=preserve_canvas_view)
                self._apply_video_physical_scale(self.current_video_id)
                
                # Force garbage collection after loading to free memory from old frame
                import gc
                gc.collect()
                
                # Debug: Print memory info
                if self.canvas.current_image is not None:
                    img_shape = self.canvas.current_image.shape
                    img_size_mb = self.canvas.current_image.nbytes / (1024 * 1024)
                    cache_status = "cached" if cached_image is not None else "from disk"
                    print(f"Loaded frame {idx} ({cache_status}): {img_size_mb:.1f}MB, cache={self.frame_cache.get_size()}/{self.frame_cache.max_size}")
                
                # If we loaded from disk, add to cache for future use
                if cached_image is None and self.canvas.current_image is not None:
                    self.frame_cache.put(idx, self.canvas.current_image.copy())
                
                # Give focus to canvas for keyboard shortcuts (spacebar toggles masks)
                self.canvas.setFocus()
                
                # Store image dimensions in annotation manager (for COCO export)
                if self.canvas.current_image is not None:
                    h, w = self.canvas.current_image.shape[:2]
                    self.annotation_manager.image_height = h
                    self.annotation_manager.image_width = w
                
                # Restore next_mask_id for this video to maintain unique IDs
                if self.current_video_id:
                    if self.current_video_id not in self.video_next_mask_id:
                        # Recompute once per video per session so stale metadata from deleted
                        # annotation files cannot keep IDs artificially high.
                        import time
                        t_start = time.perf_counter()
                        max_id = self._recompute_max_mask_id_for_video(self.current_video_id)
                        t_elapsed = (time.perf_counter() - t_start) * 1000
                        cached_max_id = self._load_max_mask_id_from_metadata(self.current_video_id)
                        if cached_max_id != max_id:
                            print(
                                f"  Updated max_mask_id cache for {self.current_video_id}: "
                                f"{cached_max_id} -> {max_id} (scan took {t_elapsed:.0f}ms)"
                            )
                        self._save_max_mask_id_to_metadata(self.current_video_id, max_id)
                        
                        self.video_next_mask_id[self.current_video_id] = max_id + 1
                    
                    # Set canvas next_mask_id from video tracking
                    self.canvas.next_mask_id = self.video_next_mask_id[self.current_video_id]
                
                # Load annotations for this frame
                # First check cache, then load from disk if not present
                import time
                t_load_start = time.perf_counter()
                
                # Always load full annotations (canvas will handle display based on view flags)
                annotations = self.annotation_manager.get_frame_annotations(idx, video_id=self.current_video_id)
                annotation_source = "cached" if annotations else "disk"
                
                if not annotations and self.project_path and self.current_video_id:
                    # Not in cache - load from disk
                    frame_idx_in_video = self._get_frame_idx_in_video(idx)
                    
                    # Load annotations from PNG+JSON format
                    annotations = self.annotation_manager.load_frame_annotations(
                        self.project_path, self.current_video_id, frame_idx_in_video
                    )
                    
                    # Update cache
                    if annotations:
                        self.annotation_manager.set_frame_annotations(idx, annotations, video_id=self.current_video_id)

                # Load video-level chamber/hive/pollen annotations and merge
                if self.project_path and self.current_video_id:
                    video_level_anns, aruco_tracking = self.annotation_manager.load_video_annotations(
                        self.project_path, self.current_video_id
                    )
                    if video_level_anns:
                        annotations = merge_annotations(
                            annotations or [], video_level_anns,
                            self.annotation_manager.project_info,
                        )
                        annotation_source += "+video-level"
                        
                t_load = (time.perf_counter() - t_load_start) * 1000
                
                # Debug: print annotation info
                if annotations:
                    num_instances = len(annotations)
                    has_masks = any('mask' in ann for ann in annotations)
                    if has_masks:
                        total_mask_mb = sum(ann.get('mask', np.array([])).nbytes / (1024*1024) for ann in annotations)
                        print(f"  Annotations ({annotation_source}): {num_instances} instances, {total_mask_mb:.1f}MB, loaded in {t_load:.0f}ms")
                    else:
                        print(f"  Bbox-only Annotations ({annotation_source}): {num_instances} instances, loaded in {t_load:.0f}ms")
                else:
                    print(f"  No annotations (checked in {t_load:.0f}ms)")
                
                # Get color mapping for current video
                mask_colors = None
                if self.current_video_id and self.current_video_id in self.video_mask_colors:
                    mask_colors = self.video_mask_colors[self.current_video_id]
                
                self.canvas.set_annotations(annotations, mask_colors)
                
                # Register any new colors that were generated
                self._register_canvas_colors()
                
                self.update_instance_list_from_canvas()
                
                # Reset modification flag AFTER loading annotations from disk
                # This prevents the set_annotations call from marking the frame as modified
                self.current_frame_modified = False
                
                # Find the list row for this frame index
                try:
                    list_row = self.frame_list_to_frames_map.index(idx)
                    previous_block_state = self.frame_list.blockSignals(True)
                    try:
                        self.frame_list.setCurrentRow(list_row)
                    finally:
                        self.frame_list.blockSignals(previous_block_state)
                    # Update slider to match
                    self.frame_slider.blockSignals(True)
                    self.frame_slider.setValue(list_row)
                    self.frame_slider.blockSignals(False)
                except ValueError:
                    # Frame not in current filter view
                    pass
                
                # Save current project state for restoration
                self._save_project_state()
                self._update_frame_navigation_buttons()
                
            except Exception as e:
                QMessageBox.warning(self, "Error", f"Failed to load frame: {str(e)}")
                import traceback
                traceback.print_exc()
            
    def on_frame_changed(self, idx):
        """Handle frame selection change"""
        if idx >= 0 and idx < len(self.frame_list_to_frames_map):
            # Map list row to actual frame index
            actual_frame_idx = self.frame_list_to_frames_map[idx]
            self.load_frame(actual_frame_idx)
            
            # Set focus to canvas so keyboard shortcuts work immediately
            self.canvas.setFocus()
            
    def next_frame(self):
        """Navigate to next frame (respects filter)"""
        next_idx = self._get_next_frame_index()
        if next_idx is not None:
            self.load_frame(next_idx)
        else:
            self.status_label.setText("Already at last frame in current view")
            
    def prev_frame(self):
        """Navigate to previous frame (respects filter)"""
        prev_idx = self._get_prev_frame_index()
        if prev_idx is not None:
            self.load_frame(prev_idx)
        else:
            self.status_label.setText("Already at first frame in current view")
    
    def on_slider_changed(self, value):
        """Handle frame slider value change"""
        if value >= 0 and value < len(self.frame_list_to_frames_map):
            actual_frame_idx = self.frame_list_to_frames_map[value]
            if actual_frame_idx != self.current_frame_idx:
                self.load_frame(actual_frame_idx)
    
    def toggle_play(self):
        """Toggle automatic playback through frames"""
        if self.is_playing:
            # Stop playing
            self.play_timer.stop()
            self.is_playing = False
            self.play_button.setText("\u25b6 Play")
        else:
            # Start playing
            self.is_playing = True
            self.play_button.setText("\u23f8 Pause")
            # Default playback speed: 10 FPS (100ms per frame)
            self.play_timer.start(100)
    
    def _play_next_frame(self):
        """Advance to next frame during playback"""
        next_idx = self._get_next_frame_index()
        if next_idx is not None:
            self.load_frame(next_idx)
        else:
            # Reached end of frames, stop playing
            self.toggle_play()
            
    def on_tool_changed(self, tool_name):
        """Handle tool change"""
        self.canvas.set_tool(tool_name)
        
        # When switching to brush/eraser with an instance selected, start editing mode
        # But only if the instance has segmentation (not bbox-only)
        if tool_name in ['brush', 'eraser'] and self.canvas.selected_mask_idx > 0:
            # Check if instance has segmentation
            selected_category = self.canvas.selected_instance_category
            mask_array, _ = self.canvas._get_mask_array_for_instance(
                self.canvas.selected_mask_idx,
                selected_category
            )
            has_segmentation = (
                mask_array is not None and
                np.any(mask_array == self.canvas.selected_mask_idx)
            )
            if has_segmentation:
                self.canvas.start_editing_instance(
                    self.canvas.selected_mask_idx,
                    category=selected_category
                )
        
        # When switching away from SAM2 tools, uncheck SAM2 toolbar buttons
        if tool_name not in ['sam2_prompt', 'sam2_box']:
            self.sam2_toolbar.uncheck_tools()
        
        # When switching to SAM2 tools, uncheck annotation toolbar buttons
        if tool_name in ['sam2_prompt', 'sam2_box']:
            self.toolbar.uncheck_all_tools()

        if tool_name == 'no_draw_zone':
            self.status_label.setText(
                "No Draw Zone: click polygon vertices; double-click or Enter to close; "
                "Backspace/right-click undoes a vertex; Esc cancels the open polygon"
            )
        elif tool_name == 'measure':
            if self.canvas.pixels_per_cm:
                scale_text = f"{self.canvas.pixels_per_cm:.3f} px/cm"
            else:
                scale_text = "not calibrated; pixels only"
            self.status_label.setText(
                f"Measure: click and drag a line ({scale_text}). "
                "Use Set Scale from Line for a known distance."
            )

    def on_clear_no_draw_zones(self):
        """Remove current-frame brush protection guides."""
        removed_count = self.canvas.clear_no_draw_zones()
        if removed_count:
            self.status_label.setText(
                f"Cleared {removed_count} No Draw Zone"
                f"{'s' if removed_count != 1 else ''}"
            )
        else:
            self.status_label.setText("No completed No Draw Zones to clear")

    def on_brush_size_step_requested(self, direction):
        """Handle +/- keyboard shortcuts for brush and eraser size."""
        current_size = max(1, int(self.canvas.brush_size))

        if direction > 0:
            new_size = current_size + 1 if current_size < 10 else round(current_size * 1.15)
        else:
            new_size = current_size - 1 if current_size <= 10 else round(current_size / 1.15)

        if new_size == current_size:
            new_size = current_size + (1 if direction > 0 else -1)

        new_size = max(1, min(1000, int(new_size)))
        self.toolbar.set_brush_size(new_size, emit=True)
        self.status_label.setText(f"Brush size: {new_size}")
        self.canvas.setFocus()

    def on_brush_size_unit_step_requested(self, direction):
        """Handle floating +/- buttons for exact one-pixel brush-size changes."""
        current_size = max(1, int(self.canvas.brush_size))
        new_size = max(1, min(1000, current_size + (1 if direction > 0 else -1)))
        self.toolbar.set_brush_size(new_size, emit=True)
        self.status_label.setText(f"Brush size: {new_size}")
        self.canvas.setFocus()

    def on_brush_eraser_toggle_requested(self):
        """Handle middle-click toggling between brush and eraser."""
        if self.canvas.current_tool == 'brush':
            next_tool = 'eraser'
        elif self.canvas.current_tool == 'eraser':
            next_tool = 'brush'
        else:
            return

        self.toolbar.set_tool(next_tool)
        self.status_label.setText(f"{next_tool.capitalize()} selected")
        self.canvas.setFocus()

    def on_instance_switch_tap_progress(self, instance_id, category, remaining):
        """Show progress for triple-tap instance switching while brushing."""
        label = category or "instance"
        if remaining > 0:
            plural = "s" if remaining != 1 else ""
            self.status_label.setText(
                f"Tap {remaining} more time{plural} to switch to {label} {instance_id}"
            )
        else:
            self.status_label.setText(f"Editing {label} {instance_id}")
        self.canvas.setFocus()
    
    def on_show_segmentations_changed(self, show):
        """Handle show segmentations checkbox change from toolbar"""
        self.canvas.set_show_segmentations(show)
        self._sync_segmentation_visibility_controls(show)
        self.canvas.setFocus()
    
    def on_show_bboxes_changed(self, show):
        """Handle show bboxes checkbox change from toolbar"""
        self.canvas.set_show_bboxes(show)
        self._sync_bbox_visibility_controls(show)
        self.canvas.setFocus()
    
    def on_menu_show_segmentations_changed(self, checked):
        """Handle show segmentations menu action change"""
        self.canvas.set_show_segmentations(checked)
        self._sync_segmentation_visibility_controls(checked)
        self.canvas.setFocus()

    def on_space_toggle_annotation_overlays(self):
        """Toggle segmentation and bbox layers without changing class switches."""
        self.canvas.toggle_annotation_overlays()
        self._sync_segmentation_visibility_controls(self.canvas.show_segmentations)
        self._sync_bbox_visibility_controls(self.canvas.show_bboxes)

        visible_layers = []
        if self.canvas.show_segmentations:
            visible_layers.append("masks")
        if self.canvas.show_bboxes:
            visible_layers.append("boxes")

        if visible_layers:
            self.status_label.setText(f"Annotation overlays visible: {', '.join(visible_layers)}")
        else:
            self.status_label.setText("Annotation overlays hidden (press Spacebar to restore)")
        self.canvas.setFocus()

    def on_shift_space_toggle_selected_instance(self):
        """Toggle the selected instance from a window-level shortcut."""
        if not self.canvas.toggle_selected_instance_visibility():
            self.status_label.setText("No selected instance to toggle")
        self.canvas.setFocus()

    def _sync_segmentation_visibility_controls(self, visible):
        """Keep all segmentation visibility controls showing the same state."""
        visible = bool(visible)

        if hasattr(self, 'segmentation_mode_action'):
            self.segmentation_mode_action.blockSignals(True)
            self.segmentation_mode_action.setChecked(visible)
            self.segmentation_mode_action.blockSignals(False)

        if hasattr(self, 'toolbar') and hasattr(self.toolbar, 'segmentation_checkbox'):
            self.toolbar.segmentation_checkbox.blockSignals(True)
            self.toolbar.segmentation_checkbox.setChecked(visible)
            self.toolbar.segmentation_checkbox.blockSignals(False)
    
    def on_menu_show_bboxes_changed(self, checked):
        """Handle show bboxes menu action change"""
        self.canvas.set_show_bboxes(checked)
        self._sync_bbox_visibility_controls(checked)
        self.canvas.setFocus()

    def _sync_bbox_visibility_controls(self, visible):
        """Keep all bounding-box visibility controls showing the same state."""
        visible = bool(visible)

        if hasattr(self, 'bbox_mode_action'):
            self.bbox_mode_action.blockSignals(True)
            self.bbox_mode_action.setChecked(visible)
            self.bbox_mode_action.blockSignals(False)

        if hasattr(self, 'toolbar') and hasattr(self.toolbar, 'bbox_checkbox'):
            self.toolbar.bbox_checkbox.blockSignals(True)
            self.toolbar.bbox_checkbox.setChecked(visible)
            self.toolbar.bbox_checkbox.blockSignals(False)
    
    def on_annotation_type_changed(self, annotation_type):
        """Handle annotation type selection change from toolbar"""
        self.canvas.set_annotation_type(annotation_type)
        print(f"Annotation type changed to: {annotation_type}")
        
        # Annotation type chooses what new edits create; class visibility is
        # controlled independently by the show/hide checkboxes.
        self.update_instance_list_from_canvas()
    
    def on_annotation_group_visibility_changed(self, categories, visible):
        """Apply a group switch without redrawing full-frame masks for every child."""
        for category in categories:
            self.canvas.set_annotation_type_visibility(category, visible, rebuild=False)
        self.canvas.rebuild_visualizations()
        self.update_instance_list_from_canvas()
        self.canvas.setFocus()

    def on_annotation_type_visibility_changed(self, annotation_type, visible):
        """Handle annotation type visibility checkbox change from toolbar"""
        self.canvas.set_annotation_type_visibility(annotation_type, visible)
        print(f"Annotation type '{annotation_type}' visibility set to: {visible}")
        # Update instance list to reflect visibility changes
        self.update_instance_list_from_canvas()
        self.canvas.setFocus()
    
    def on_sam2_loaded(self, sam2_integrator):
        """Called when SAM2 model is loaded via the SAM2Toolbar"""
        self.sam2 = sam2_integrator
        print(f"SAM2 loaded successfully via toolbar! Type: {type(sam2_integrator)}, self.sam2 is None: {self.sam2 is None}")
        # Only update status label if it exists (may not exist during init)
        if hasattr(self, 'status_label'):
            self.status_label.setText("✓ SAM2 model loaded and ready")
    
    def on_sam2_unload(self):
        """Called when SAM2 model is unloaded via the SAM2Toolbar"""
        import gc
        import torch
        
        print(f"on_sam2_unload called. self.sam2 before unload: {type(self.sam2) if self.sam2 else 'None'}")
        
        # Clear SAM2 instance
        self.sam2 = None
        
        print(f"on_sam2_unload: self.sam2 set to None")
        
        # Force garbage collection
        gc.collect()
        
        # Clear CUDA cache if available
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print("CUDA cache cleared")
        
        print("SAM2 unloaded and memory freed")
        
        # Update status
        if hasattr(self, 'status_label'):
            self.status_label.setText("SAM2 unloaded")
        
        # Switch to pan tool
        self.canvas.set_tool('pan')
        
    def on_sam2_finetune_requested(self):
        """Handle SAM2 fine-tuning request"""
        # Check if SAM2 is loaded
        if not hasattr(self, 'sam2_toolbar') or not self.sam2_toolbar.is_checkpoint_loaded():
            QMessageBox.warning(
                self, 
                "No Model Loaded", 
                "Please load a SAM2 checkpoint first before fine-tuning."
            )
            return
        
        # Check if project is loaded
        if not self.project_path:
            QMessageBox.warning(
                self, 
                "No Project", 
                "Please open or create a project first."
            )
            return
        
        # Check if we have any annotations
        train_dir = self.project_path / 'annotations/coco/train'
        val_dir = self.project_path / 'annotations/coco/val'
        
        # Check for existing annotations if not exporting
        if not train_dir.exists() or not list(train_dir.glob('*.json')):
            reply = QMessageBox.question(
                self,
                "Export Annotations",
                "No training annotations found. Would you like to export COCO annotations now?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )
            
            if reply == QMessageBox.StandardButton.Yes:
                self.export_coco_format()
                # Check again after export
                if not train_dir.exists() or not list(train_dir.glob('*.json')):
                    QMessageBox.warning(self, "Export Failed", "Failed to export annotations.")
                    return
            else:
                return
        
        # Get current checkpoint info from toolbar
        checkpoint_path = self.sam2_toolbar.checkpoint_path
        config_name = self.sam2_toolbar.config_name
        
        # Show configuration dialog
        config_dialog = SAM2TrainingConfigDialog(
            self, 
            checkpoint_path=str(checkpoint_path),
            config_name=config_name
        )
        
        if config_dialog.exec():
            config = config_dialog.get_config()
            
            # Export COCO annotations if requested
            if config.get('export_coco', True):
                self.export_coco_format()
                # Check annotations exist after export
                train_jsons = list(train_dir.glob('*.json')) if train_dir.exists() else []
                if not train_jsons:
                    QMessageBox.warning(self, "Export Failed", "Failed to export training annotations.")
                    return
            
            # Create progress dialog
            progress_dialog = TrainingProgressDialog(self)
            progress_dialog.setWindowTitle("SAM2 Fine-tuning Progress")
            
            # Create training worker
            worker = SAM2TrainingWorker(self.project_path, config)
            progress_dialog.set_worker(worker)
            
            # Start training
            worker.start()
            
            # Show progress dialog (blocks until closed)
            progress_dialog.exec()
            
            # Handle completion
            if progress_dialog.training_completed:
                # Training completed successfully
                model_path = progress_dialog.model_path
                final_metrics = progress_dialog.final_metrics
                
                # Build completion message
                msg = "SAM2 fine-tuning completed successfully!\n\n"
                msg += f"Model saved to:\n{model_path}\n\n"
                
                if final_metrics:
                    msg += "Training Steps:\n"
                    for key, value in final_metrics.items():
                        if isinstance(value, (int, float)):
                            msg += f"  {key}: {value}\n"
                    msg += "\n"
                
                msg += "Would you like to load the fine-tuned model into the SAM2 toolbar?"
                
                reply = QMessageBox.question(
                    self,
                    "Fine-tuning Complete",
                    msg,
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                    QMessageBox.StandardButton.Yes
                )
                
                if reply == QMessageBox.StandardButton.Yes:
                    # Load the new model
                    from pathlib import Path
                    if Path(model_path).exists():
                        self.sam2_toolbar._load_checkpoint_from_path(model_path, show_dialogs=False)
                        self.status_label.setText(f"✓ Loaded fine-tuned SAM2 model: {Path(model_path).name}")
                    else:
                        QMessageBox.warning(
                            self,
                            "Model Not Found",
                            f"Could not find trained model at:\n{model_path}"
                        )
            elif progress_dialog.training_failed:
                # Show error message
                QMessageBox.critical(
                    self,
                    "Training Failed",
                    "SAM2 fine-tuning failed. Check the training log for details."
                )
    
    def on_canvas_point_clicked(self, x, y, is_positive):
        """Handle point click on canvas for SAM2 prompting"""
        print(f"Point clicked: ({x}, {y}), positive={is_positive}")
        print(f"SAM2 available: {self.sam2 is not None}")
        print(f"Current tool: {self.canvas.current_tool}")
        
        if self.sam2 and self.canvas.current_tool == 'sam2_prompt':
            try:
                if self.canvas.selected_mask_idx <= 0 and self.canvas.active_sam2_mask_idx <= 0:
                    self.canvas.clear_prompt_points()
                    self.status_label.setText("Create or select an instance before using SAM2 prompts")
                    self.canvas.setFocus()
                    return

                # Get all accumulated prompt points
                prompts = self.canvas.get_prompt_points()
                positive_points = prompts['positive']
                negative_points = prompts['negative']
                
                print(f"Running SAM2 with {len(positive_points)} positive and {len(negative_points)} negative points")
                
                # Convert image to RGB for SAM2 if grayscale
                import cv2
                image_for_sam2 = self.canvas.current_image
                if len(image_for_sam2.shape) == 2:
                    image_for_sam2 = cv2.cvtColor(image_for_sam2, cv2.COLOR_GRAY2RGB)
                
                # Call SAM2 with all accumulated points
                mask = self.sam2.predict_with_points(
                    image_for_sam2,
                    positive_points,
                    negative_points
                )
                print(f"Mask shape: {mask.shape}, unique values: {np.unique(mask)}")
                
                # Check if we're editing an existing selected instance
                if self.canvas.selected_mask_idx > 0:
                    # Start editing mode if not already active
                    if (self.canvas.editing_instance_id != self.canvas.selected_mask_idx or
                        self.canvas.editing_instance_category != self.canvas.selected_instance_category):
                        self.canvas.start_editing_instance(
                            self.canvas.selected_mask_idx,
                            category=self.canvas.selected_instance_category
                        )
                    # Update the editing mask directly
                    self.canvas.editing_mask = (mask > 0).astype(np.uint8) * 255
                    self.canvas._update_editing_visualization()
                    self.canvas.active_sam2_mask_idx = self.canvas.selected_mask_idx
                # Check if we're editing an active SAM2 mask
                elif self.canvas.active_sam2_mask_idx > 0:
                    # Start editing mode if not already active
                    if self.canvas.editing_instance_id != self.canvas.active_sam2_mask_idx:
                        self.canvas.start_editing_instance(
                            self.canvas.active_sam2_mask_idx,
                            category=self.canvas.selected_instance_category
                        )
                    # Update the editing mask
                    self.canvas.editing_mask = (mask > 0).astype(np.uint8) * 255
                    self.canvas._update_editing_visualization()
                else:
                    self.status_label.setText("Create or select an instance before using SAM2 prompts")
                    self.canvas.setFocus()
                    return
                
                # Register color for this mask
                self._register_canvas_colors()
                
                # Update instance list
                self.update_instance_list_from_canvas()
                    
                self.status_label.setText(f"SAM2 prediction updated ({len(positive_points)} pos, {len(negative_points)} neg)")
            except Exception as e:
                import traceback
                traceback.print_exc()
                QMessageBox.warning(self, "Error", f"SAM2 prediction failed: {str(e)}")
        else:
            if not self.sam2:
                self.status_label.setText("SAM2 not loaded - download checkpoint first")
            else:
                self.status_label.setText(f"Switch to SAM2 Point tool (current: {self.canvas.current_tool})")
                
    def on_canvas_box_drawn(self, x1, y1, x2, y2):
        """Handle box drawn on canvas for SAM2 prompting or box inference mode"""
        print(f"Box drawn: ({x1}, {y1}) to ({x2}, {y2})")
        print(f"SAM2 available: {self.sam2 is not None}")
        print(f"Current tool: {self.canvas.current_tool}")
        print(f"Box inference mode: {self.box_inference_mode}")
        
        # Normalize box coordinates
        x1, x2 = min(x1, x2), max(x1, x2)
        y1, y2 = min(y1, y2), max(y1, y2)
        
        # Handle box inference mode (inference_box tool)
        if self.canvas.current_tool == 'inference_box' and self.box_inference_mode:
            # Box is drawn and stored in canvas - notify toolbars
            self.yolo_sahi_toolbar.set_box_drawn(True)
            self.yolo_beehavesque_toolbar.set_box_drawn(True)
            self.status_label.setText("Box drawn. Adjust if needed, then click 'Run Inference on Box'")
            return
        
        # Handle SAM2 box prediction
        if self.sam2 and self.canvas.current_tool == 'sam2_box':
            try:
                if self.canvas.selected_mask_idx <= 0 and self.canvas.active_sam2_mask_idx <= 0:
                    self.status_label.setText("Create or select an instance before using SAM2 Box")
                    self.canvas.setFocus()
                    return

                print("Calling SAM2 box prediction...")
                
                # Convert image to RGB for SAM2 if grayscale
                import cv2
                image_for_sam2 = self.canvas.current_image
                if len(image_for_sam2.shape) == 2:
                    image_for_sam2 = cv2.cvtColor(image_for_sam2, cv2.COLOR_GRAY2RGB)
                
                mask = self.sam2.predict_with_box(
                    image_for_sam2,
                    x1, y1, x2, y2
                )
                print(f"Mask shape: {mask.shape}, unique values: {np.unique(mask)}")
                
                # Similar logic to sam2_prompt tool:
                # If a mask is selected and user creates a box, update that mask
                if self.canvas.selected_mask_idx > 0:
                    self.canvas.active_sam2_mask_idx = self.canvas.selected_mask_idx
                # Check if we're editing an active SAM2 mask
                if self.canvas.active_sam2_mask_idx > 0:
                    # Start editing mode if not already active
                    if self.canvas.editing_instance_id != self.canvas.active_sam2_mask_idx:
                        self.canvas.start_editing_instance(
                            self.canvas.active_sam2_mask_idx,
                            category=self.canvas.selected_instance_category
                        )
                    # Update the editing mask
                    self.canvas.editing_mask = (mask > 0).astype(np.uint8) * 255
                    self.canvas._update_editing_visualization()
                else:
                    self.status_label.setText("Create or select an instance before using SAM2 Box")
                    self.canvas.setFocus()
                    return
                
                # Register color for this mask
                self._register_canvas_colors()
                
                self.update_instance_list_from_canvas()
                self.status_label.setText("SAM2 box prediction created in editing mode (click away to confirm)")
            except Exception as e:
                import traceback
                traceback.print_exc()
                QMessageBox.warning(self, "Error", f"SAM2 prediction failed: {str(e)}")
        else:
            if not self.sam2:
                self.status_label.setText("SAM2 not loaded - download checkpoint first")
            else:
                self.status_label.setText(f"Switch to SAM2 Box tool (current: {self.canvas.current_tool})")
    
    def on_masks_visibility_changed(self, visible):
        """Handle mask visibility changes"""
        self._sync_segmentation_visibility_controls(visible)

        # Flush pending instance list update to avoid lag
        if self._instance_list_update_timer.isActive():
            self._instance_list_update_timer.stop()
            self._do_update_instance_list()
        
        if visible:
            self.status_label.setText("Segmentation masks visible")
        else:
            self.status_label.setText("⚠ Segmentation masks hidden (press Spacebar to show)")

    def on_bboxes_visibility_changed(self, visible):
        """Handle bbox visibility changes from canvas shortcuts."""
        self._sync_bbox_visibility_controls(visible)

    def on_instance_visibility_shortcut_changed(self, instance_id, category, visible):
        """Handle Shift+Space toggling the selected instance."""
        self.update_instance_list_from_canvas()
        state = "shown" if visible else "hidden"
        self.status_label.setText(f"{category.capitalize()} instance {instance_id} {state}")
        self.canvas.setFocus()

    def on_instance_visibility_shortcut_blocked(self, instance_id, category):
        """Handle selected-instance shortcut when the parent class is hidden."""
        self.status_label.setText(
            f"Turn on {category} visibility before showing instance {instance_id}"
        )
        self.canvas.setFocus()
    
    def on_annotation_changed(self):
        """Handle annotation changes - mark frame as modified"""
        self._mark_current_annotations_dirty()
        self._schedule_done_counter_update()
    
    def on_show_labels_changed(self, state):
        """Handle show instance labels checkbox change"""
        from PyQt6.QtCore import Qt
        visible = (state == Qt.CheckState.Checked.value)
        self.canvas.set_labels_visible(visible)

    def on_show_aruco_labels_changed(self, state):
        """Handle show ArUco labels checkbox change"""
        from PyQt6.QtCore import Qt
        visible = (state == Qt.CheckState.Checked.value)
        self._sync_canvas_aruco_labels()
        self.canvas.set_aruco_labels_visible(visible)
                
    def update_instance_list(self):
        """Update the instance list widget from annotation manager"""
        self.instance_list.clear()
        annotations = self.annotation_manager.get_frame_annotations(self.current_frame_idx, video_id=self.current_video_id)
        for i, ann in enumerate(annotations):
            label = ann.get('label', 'Unknown')
            self.instance_list.addItem(f"{label} #{i+1}")
    
    def _schedule_instance_list_update(self):
        """Schedule a throttled instance list update (performance optimization)"""
        if not self._instance_list_update_timer.isActive():
            self._instance_list_update_timer.start()
    
    def _do_update_instance_list(self):
        """Actually perform the instance list update (called by timer)"""
        self.update_instance_list_from_canvas()
            
    def update_instance_list_from_canvas(self):
        """Update the instance list widget from current canvas masks"""
        from PyQt6.QtWidgets import QListWidgetItem

        # An immediate refresh also satisfies any queued annotation-change refresh.
        self._instance_list_update_timer.stop()
        
        self._instance_list_updating = True
        self.instance_list.clear()
        
        # Preserve selection for the active/editing instance
        active_instance_id = (
            self.canvas.editing_instance_id
            if self.canvas.editing_instance_id > 0
            else self.canvas.selected_mask_idx
        )
        active_instance_category = (
            self.canvas.editing_instance_category
            if self.canvas.editing_instance_id > 0
            else self.canvas.selected_instance_category
        )
        
        # Get ArUco tracking from video-level annotations
        instance_to_aruco = self._get_instance_to_aruco_map()
        if instance_to_aruco != self.canvas.aruco_instance_labels:
            self.canvas.set_aruco_instance_labels(instance_to_aruco)
        
        instance_entries = self.canvas.get_instance_entries()
        
        # Track which row to select
        row_to_select = -1
        current_row = 0
        
        # Add all instances (bee, chamber, hive, and pollen). Class visibility
        # controls canvas display, not whether the row exists in this list.
        for entry in instance_entries:
            instance_id = entry['id']
            category = entry['category']
            metadata = self.canvas.annotation_metadata.get(instance_id, {})
            
            # Determine category prefix
            category_prefix_map = {'bee': 'B', 'hive': 'H', 'chamber': 'C', 'pollen': 'P', 'nectar': 'N', **BROOD_SHORT_LABELS}
            prefix = category_prefix_map.get(category, category_label(category))
            class_visible = self.canvas.annotation_type_visibility.get(category, True)
            
            geometry = self.canvas.get_instance_geometry(instance_id, category)
            editing = (instance_id == self.canvas.editing_instance_id and
                       category == self.canvas.editing_instance_category and
                       self.canvas.editing_mask is not None)
            if editing or geometry.area:
                area = geometry.area
            else:
                # Bbox-only: calculate area from bbox (width * height)
                bbox = metadata.get('bbox', [0, 0, 0, 0])
                area = bbox[2] * bbox[3]  # width * height
            
            # Build display text with category prefix
            display_text = f"{prefix} {instance_id}"
            
            # Add ArUco info if available (only for bees)
            if category == 'bee' and instance_id in instance_to_aruco:
                aruco_id = instance_to_aruco[instance_id]
                display_text += f" [ArUco: {aruco_id}]"
            
            display_text += f" (area: {area})"
            if not class_visible:
                display_text += " [class hidden]"
            
            item = QListWidgetItem(display_text)
            item.setData(Qt.ItemDataRole.UserRole, {'type': category, 'id': instance_id})
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(
                Qt.CheckState.Checked
                if self.canvas.is_instance_enabled(instance_id, category)
                else Qt.CheckState.Unchecked
            )
            item.setToolTip(
                f"{category_label(category)} {instance_id}\nChecked: show this instance. Unchecked: hide this instance."
                if class_visible else
                f"Turn on {category_label(category)} visibility first; the class switch overrides this row."
            )
            provenance = metadata.get('provenance', {})
            if provenance:
                creator = (provenance.get('created_by') or {}).get('name', 'Unknown')
                editor = (provenance.get('last_modified_by') or {}).get('name', 'Unknown')
                item.setToolTip(item.toolTip() + f'\nCreated by: {creator}\nLast edited by: {editor}')
            self.instance_list.addItem(item)
            
            if instance_id == active_instance_id and category == active_instance_category:
                row_to_select = current_row
            current_row += 1
        
        # Restore selection
        if row_to_select >= 0:
            self.instance_list.blockSignals(True)
            self.instance_list.setCurrentRow(row_to_select)
            self.instance_list.blockSignals(False)

        self._instance_list_updating = False
        self._schedule_done_counter_update()

    def _get_instance_to_aruco_map(self):
        """Build reverse map: instance ID -> ArUco ID for the current video."""
        instance_to_aruco = {}
        if self.current_video_id and self.project_path:
            _, aruco_tracking = self.annotation_manager.load_video_annotations(
                self.project_path, self.current_video_id
            )
            for aruco_str, instance_id in aruco_tracking.items():
                try:
                    instance_to_aruco[int(instance_id)] = int(aruco_str)
                except (TypeError, ValueError):
                    continue
        return instance_to_aruco

    def _sync_canvas_aruco_labels(self):
        """Refresh canvas ArUco labels from video-level tracking."""
        self.canvas.set_aruco_instance_labels(self._get_instance_to_aruco_map(), rebuild=False)
    
    def detect_and_update_markers(self):
        """Detect ArUco/QR markers in current frame annotations and update annotation data"""
        if not hasattr(self.canvas, 'current_image') or self.canvas.current_image is None:
            return
        
        annotations = self.canvas.get_annotations()
        if not annotations:
            return
        
        # Detect markers in all annotations
        detections = self.marker_detector.detect_in_annotations(
            self.canvas.current_image, annotations
        )
        
        # Update annotations with marker data
        markers_found = 0
        for ann in annotations:
            instance_id = ann.get('mask_id', ann.get('instance_id', 0))
            if instance_id in detections:
                detection = detections[instance_id]
                ann['marker'] = {
                    'type': detection.marker_type,
                    'id': detection.marker_id,
                    'confidence': detection.confidence,
                    'corners': detection.corners.tolist() if detection.corners is not None else None,
                    'center': detection.center
                }
                markers_found += 1
        
        # Update canvas with marked annotations
        if markers_found > 0:
            # Get current mask colors to preserve them
            mask_colors_dict = None
            if self.current_video_id and self.current_video_id in self.video_mask_colors:
                mask_colors_dict = self.video_mask_colors[self.current_video_id]
            
            self.canvas.set_annotations(annotations, mask_colors_dict)
            print(f"Detected {markers_found} marker(s) in current frame")
        
        return markers_found
    
    def on_detect_markers_manually(self):
        """Handle manual marker detection trigger from menu"""
        if not hasattr(self.canvas, 'current_image') or self.canvas.current_image is None:
            QMessageBox.information(
                self, "No Frame",
                "No frame is currently loaded."
            )
            return
        
        annotations = self.canvas.get_annotations()
        if not annotations:
            QMessageBox.information(
                self, "No Annotations",
                "No annotations to analyze. Create some annotations first."
            )
            return
        
        # Run detection
        markers_found = self.detect_and_update_markers()
        
        # Update the instance list to show markers
        self.update_instance_list_from_canvas()
        
        # Show result
        if markers_found and markers_found > 0:
            QMessageBox.information(
                self, "Markers Detected",
                f"Found {markers_found} ArUco/QR code marker(s) on {len(annotations)} bee(s).\n\n"
                f"The markers are now shown in the instance list and will be saved with the annotations."
            )
        else:
            QMessageBox.information(
                self, "No Markers Found",
                f"No ArUco or QR code markers were detected on the {len(annotations)} annotated bee(s).\n\n"
                f"Make sure the markers are clearly visible in the segmentation masks."
            )
    
    def on_detect_aruco_in_bees(self):
        """Handle ArUco detection in bee instances only (toolbar button)"""
        if not hasattr(self.canvas, 'current_image') or self.canvas.current_image is None:
            QMessageBox.information(
                self, "No Frame",
                "No frame is currently loaded."
            )
            return
        
        if not self.current_video_id:
            QMessageBox.information(
                self, "No Video",
                "No video is currently loaded. ArUco tracking requires a video context."
            )
            return
        
        annotations = self.canvas.get_annotations()
        if not annotations:
            QMessageBox.information(
                self, "No Annotations",
                "No annotations to analyze. Create some annotations first."
            )
            return
        
        # Count bee instances
        bee_count = sum(1 for ann in annotations if ann.get('category', 'bee') == 'bee')
        if bee_count == 0:
            QMessageBox.information(
                self, "No Bee Instances",
                "No bee instances found in the current frame.\n\n"
                "This tool only processes bee annotations."
            )
            return
        
        # Load video-level ArUco tracking
        _, aruco_tracking = self.annotation_manager.load_video_annotations(
            self.project_path, self.current_video_id
        )
        
        # Build reverse mapping: instance_id -> aruco_id (video-wide)
        # This helps us detect when an instance already has a different ArUco code
        instance_to_aruco = {}  # instance_id -> aruco_id
        for aruco_str, instance_id in aruco_tracking.items():
            aruco_int = int(aruco_str)
            # If instance already has a different ArUco, there's a conflict in the tracking data
            if instance_id in instance_to_aruco and instance_to_aruco[instance_id] != aruco_int:
                print(f"[ArUco Tracking] WARNING: Instance {instance_id} has multiple ArUco codes in tracking: {instance_to_aruco[instance_id]} and {aruco_int}")
            instance_to_aruco[instance_id] = aruco_int
        
        # Run ArUco detection on bee instances only
        detections = self.marker_detector.detect_aruco_in_bee_instances(
            self.canvas.current_image,
            annotations,
            reject_multiple=True
        )
        
        # Process detections and handle ID reassignment
        markers_found = 0
        reassignments = []  # Track (old_id, new_id, aruco_id) for reporting
        new_trackings = []  # Track new ArUco codes
        preservations = []  # Track (preserved_id, new_id) when moving existing instances
        tracking_updates = []  # Track ArUco tracking updates for preserved instances
        skipped_conflicts = []  # Track skipped detections due to conflicts
        
        # Build reverse map of video-level tracking: instance_id -> aruco_id
        instance_to_aruco = {}  # instance_id -> aruco_id (video-wide)
        for aruco_str, inst_id in aruco_tracking.items():
            instance_to_aruco[inst_id] = int(aruco_str)
        
        # Build a map of ArUco codes currently assigned in this frame
        # (only from instances that exist in current frame)
        current_frame_aruco_map = {}  # aruco_id -> instance_id (for current frame only)
        current_instance_ids = set(ann.get('mask_id', ann.get('instance_id', 0)) 
                                   for ann in annotations if ann.get('category', 'bee') == 'bee')
        for inst_id in current_instance_ids:
            if inst_id in instance_to_aruco:
                aruco_id = instance_to_aruco[inst_id]
                current_frame_aruco_map[aruco_id] = inst_id
        
        for ann in annotations:
            if ann.get('category', 'bee') != 'bee':
                continue
                
            instance_id = ann.get('mask_id', ann.get('instance_id', 0))
            
            if instance_id in detections:
                detection = detections[instance_id]
                aruco_id = detection.marker_id
                aruco_id_str = str(aruco_id)
                
                # Check if this instance already has a different ArUco code assigned (video-level)
                if instance_id in instance_to_aruco and instance_to_aruco[instance_id] != aruco_id:
                    existing_aruco = instance_to_aruco[instance_id]
                    print(f"[ArUco Tracking] ✗ Skipping: Instance {instance_id} already has ArUco {existing_aruco}, cannot assign ArUco {aruco_id}")
                    skipped_conflicts.append((instance_id, aruco_id, f"already has {existing_aruco}"))
                    continue  # Skip this detection to maintain 1-1 mapping
                
                # Check if this ArUco code is already assigned to a DIFFERENT instance in this frame
                if aruco_id in current_frame_aruco_map and current_frame_aruco_map[aruco_id] != instance_id:
                    existing_instance = current_frame_aruco_map[aruco_id]
                    print(f"[ArUco Tracking] ✗ Skipping: ArUco {aruco_id} already assigned to Instance {existing_instance} in this frame, cannot assign to Instance {instance_id}")
                    skipped_conflicts.append((instance_id, aruco_id, f"already on instance {existing_instance}"))
                    continue  # Skip to maintain 1-1 mapping
                
                # Check if this ArUco code is already tracked
                if aruco_id_str in aruco_tracking:
                    tracked_id = aruco_tracking[aruco_id_str]
                    if tracked_id != instance_id:
                        # Check if the tracked ID already exists in this frame
                        existing_ids = self.canvas.get_instance_ids()
                        if tracked_id in existing_ids:
                            # The tracked ID already has an annotation in this frame
                            # Check if it has its own ArUco code that needs to be preserved (video-level)
                            preserved_aruco_id = instance_to_aruco.get(tracked_id)
                            
                            # Move that existing annotation to a new ID to preserve it
                            new_preserved_id = self.canvas.next_mask_id
                            print(f"[ArUco Tracking] Tracked ID {tracked_id} already exists in frame, preserving as instance {new_preserved_id}")
                            if self.canvas.reassign_instance_id(tracked_id, new_preserved_id):
                                preservations.append((tracked_id, new_preserved_id))
                                print(f"  Preserved: instance {tracked_id} -> {new_preserved_id}")
                                
                                # Update ArUco tracking if the preserved instance had its own ArUco code
                                if preserved_aruco_id is not None:
                                    preserved_aruco_str = str(preserved_aruco_id)
                                    print(f"  Updating tracking: ArUco {preserved_aruco_id} now points to instance {new_preserved_id}")
                                    aruco_tracking[preserved_aruco_str] = new_preserved_id
                                    instance_to_aruco[new_preserved_id] = preserved_aruco_id  # Update reverse map
                                    # Remove old reverse mapping
                                    if tracked_id in instance_to_aruco and instance_to_aruco[tracked_id] == preserved_aruco_id:
                                        del instance_to_aruco[tracked_id]
                                    tracking_updates.append((preserved_aruco_id, new_preserved_id))
                                    self.annotation_manager.update_aruco_tracking(
                                        self.project_path, self.current_video_id, preserved_aruco_id, new_preserved_id
                                    )
                            else:
                                print(f"  Warning: Failed to preserve existing instance {tracked_id}")
                                continue  # Skip this reassignment if we couldn't preserve
                        
                        # Now reassign the ArUco-coded annotation to the tracked ID
                        print(f"[ArUco Tracking] ArUco {aruco_id} already tracked to instance {tracked_id}, reassigning {instance_id} -> {tracked_id}")
                        if self.canvas.reassign_instance_id(instance_id, tracked_id):
                            reassignments.append((instance_id, tracked_id, aruco_id))
                            # Update reverse map: the tracked_id now has this ArUco
                            instance_to_aruco[tracked_id] = aruco_id
                            # Remove old instance from reverse map
                            if instance_id in instance_to_aruco and instance_id != tracked_id:
                                del instance_to_aruco[instance_id]
                            instance_id = tracked_id  # Use new ID for metadata
                        else:
                            print(f"  Warning: Failed to reassign instance {instance_id} to {tracked_id}")
                else:
                    # New ArUco code - check if this instance already has a different ArUco tracked
                    if instance_id in instance_to_aruco:
                        old_aruco = instance_to_aruco[instance_id]
                        if old_aruco != aruco_id:
                            # Instance was previously tracked with a different ArUco code
                            # Remove the old tracking entry to maintain 1-1 mapping
                            old_aruco_str = str(old_aruco)
                            if old_aruco_str in aruco_tracking:
                                print(f"[ArUco Tracking] Removing old tracking: ArUco {old_aruco} → Instance {instance_id}")
                                del aruco_tracking[old_aruco_str]
                                # Save the removal
                                self.annotation_manager.remove_aruco_tracking(
                                    self.project_path, self.current_video_id, old_aruco
                                )
                    
                    # Add new ArUco code to tracking
                    print(f"[ArUco Tracking] New ArUco {aruco_id} -> instance {instance_id}")
                    aruco_tracking[aruco_id_str] = instance_id
                    instance_to_aruco[instance_id] = aruco_id  # Update reverse map
                    new_trackings.append((aruco_id, instance_id))
                    # Save immediately to ensure tracking is persisted
                    self.annotation_manager.update_aruco_tracking(
                        self.project_path, self.current_video_id, aruco_id, instance_id
                    )
                
                # Update current frame ArUco map to track this assignment
                current_frame_aruco_map[aruco_id] = instance_id
                
                markers_found += 1
        
        # Mark frame as modified so it gets auto-saved
        if markers_found > 0 or reassignments or preservations:
            self._mark_current_annotations_dirty()
        
        # Update the instance list to show markers
        self.update_instance_list_from_canvas()
        
        # Show result dialog with detailed info
        if markers_found > 0:
            msg_parts = [f"Found {markers_found} ArUco marker(s) on {bee_count} bee instance(s)."]
            
            if new_trackings:
                msg_parts.append(f"\n\n✓ New ArUco codes tracked:")
                for aruco_id, inst_id in new_trackings:
                    msg_parts.append(f"  • ArUco {aruco_id} → Instance {inst_id}")
            
            if reassignments:
                msg_parts.append(f"\n\n✓ Instances reassigned based on existing tracking:")
                for old_id, new_id, aruco_id in reassignments:
                    msg_parts.append(f"  • Instance {old_id} → {new_id} (ArUco {aruco_id})")
            
            if preservations:
                msg_parts.append(f"\n\n✓ Existing instances preserved (moved to new IDs):")
                for old_id, new_id in preservations:
                    msg_parts.append(f"  • Instance {old_id} → {new_id}")
            
            if tracking_updates:
                msg_parts.append(f"\n\n✓ ArUco tracking updated for preserved instances:")
                for aruco_id, inst_id in tracking_updates:
                    msg_parts.append(f"  • ArUco {aruco_id} → Instance {inst_id}")
            
            if skipped_conflicts:
                msg_parts.append(f"\n\n⚠ Skipped {len(skipped_conflicts)} conflicting detection(s):")
                for inst_id, aruco, reason in skipped_conflicts:
                    msg_parts.append(f"  • Instance {inst_id}: ArUco {aruco} ({reason})")
            
            msg_parts.append("\n\nArUco IDs are now tracked across the entire video.")
            msg_parts.append("Instances with 0 or multiple codes were skipped.")
            
            QMessageBox.information(self, "ArUco Detection Complete", "".join(msg_parts))
        else:
            # Build message for no markers case
            msg_parts = [f"No ArUco markers were detected on the {bee_count} bee instance(s)."]
            
            if skipped_conflicts:
                msg_parts.append(f"\n\n⚠ {len(skipped_conflicts)} detection(s) skipped due to conflicts:")
                for inst_id, aruco, reason in skipped_conflicts:
                    msg_parts.append(f"  • Instance {inst_id}: ArUco {aruco} ({reason})")
            
            msg_parts.append(f"\n\nPossible reasons:")
            msg_parts.append(f"• ArUco codes are not visible or too small")
            msg_parts.append(f"• Multiple codes detected in the same bee (rejected)")
            msg_parts.append(f"• ArUco codes are outside the annotation regions")
            
            QMessageBox.information(self, "No ArUco Markers Found", "".join(msg_parts))
    
    def on_clear_all_aruco_tracking(self):
        """Clear all ArUco tracking for the current video"""
        if not self.current_video_id:
            QMessageBox.information(
                self, "No Video",
                "No video is currently loaded."
            )
            return
        
        # Confirmation dialog
        reply = QMessageBox.question(
            self,
            "Clear All ArUco Tracking",
            f"Are you sure you want to clear ALL ArUco tracking for video '{self.current_video_id}'?\n\n"
            f"This will:\n"
            f"• Remove all ArUco code → instance ID associations\n"
            f"• Remove ArUco marker metadata from all frames in this video\n"
            f"• Allow re-detection and new associations\n\n"
            f"This action cannot be undone.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No
        )
        
        if reply != QMessageBox.StandardButton.Yes:
            return
        
        try:
            # Clear all ArUco tracking for this video
            count = self.annotation_manager.clear_all_aruco_tracking(
                self.project_path, self.current_video_id
            )
            
            # Update instance list to reflect removal
            if count > 0:
                self.update_instance_list_from_canvas()
            
            if count > 0:
                QMessageBox.information(
                    self, "ArUco Tracking Cleared",
                    f"Cleared ArUco tracking for video '{self.current_video_id}'.\n\n"
                    f"• Removed {count} video-level tracking association(s)\n\n"
                    f"You can now re-detect and reassign ArUco codes to instances."
                )
                print(f"[ArUco Tracking] Cleared {count} tracking entries for video {self.current_video_id}")
            else:
                QMessageBox.information(
                    self, "No Tracking Found",
                    f"No ArUco tracking associations found for video '{self.current_video_id}'."
                )
        except Exception as e:
            QMessageBox.critical(
                self, "Error",
                f"Failed to clear ArUco tracking: {str(e)}"
            )
            print(f"Error clearing ArUco tracking: {e}")
    
    def clear_instance_aruco_association(self, instance_id):
        """Clear ArUco tracking association for a specific instance
        
        Args:
            instance_id: The instance ID to clear ArUco association for
        """
        if not self.current_video_id or not self.project_path:
            return
        
        # Get the ArUco ID for this instance from video-level tracking
        _, aruco_tracking = self.annotation_manager.load_video_annotations(
            self.project_path, self.current_video_id
        )
        
        # Build reverse map: instance_id -> aruco_id
        aruco_id = None
        for aruco_str, inst_id in aruco_tracking.items():
            if inst_id == instance_id:
                aruco_id = int(aruco_str)
                break
        
        if aruco_id is None:
            QMessageBox.information(
                self, "No ArUco Marker",
                f"Instance {instance_id} does not have an ArUco marker association."
            )
            return
        
        # Confirmation dialog
        reply = QMessageBox.question(
            self,
            "Remove ArUco Association",
            f"Remove ArUco code {aruco_id} association from Instance {instance_id}?\n\n"
            f"This will:\n"
            f"• Remove the ArUco {aruco_id} → Instance {instance_id} tracking for this video\n"
            f"• Keep the ArUco marker visible on this annotation in this frame\n"
            f"• Allow this ArUco code to be assigned to a different instance\n\n"
            f"This action cannot be undone.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No
        )
        
        if reply != QMessageBox.StandardButton.Yes:
            return
        
        try:
            # Remove the ArUco tracking for this specific code
            self.annotation_manager.remove_aruco_tracking(
                self.project_path, self.current_video_id, aruco_id
            )
            
            # Update the instance list to reflect the removal
            self.update_instance_list_from_canvas()
            
            QMessageBox.information(
                self, "ArUco Association Removed",
                f"Removed ArUco {aruco_id} → Instance {instance_id} tracking association.\n\n"
                f"The ArUco code can now be assigned to a different instance."
            )
            print(f"[ArUco Tracking] Removed tracking: ArUco {aruco_id} → Instance {instance_id}")
            
        except Exception as e:
            QMessageBox.critical(
                self, "Error",
                f"Failed to remove ArUco association: {str(e)}"
            )
            print(f"Error removing ArUco association: {e}")
    
    def _sync_annotation_type_control(self, category, preserve_active_edit=False):
        """Keep the canvas active category in sync with the selected instance."""
        if category not in CATEGORIES:
            return

        if preserve_active_edit:
            self.canvas.current_annotation_type = category
        else:
            self.canvas.set_annotation_type(category, rebuild=False)

    def on_instance_changed(self, idx):
        """Handle instance selection change
        
        Args:
            idx: Row index in the instance list widget
        """
        if idx < 0:
            return
        
        # Don't update canvas selection during multi-selection
        # Only update when exactly one item is selected
        selected_items = self.instance_list.selectedItems()
        if len(selected_items) != 1:
            return
        
        # Get the selected item and check its type
        item = self.instance_list.item(idx)
        if item is None:
            return
        
        item_data = item.data(Qt.ItemDataRole.UserRole)
        if item_data is None:
            return
        
        item_type = item_data.get('type')

        # Commit any active editing before switching instance/type
        instance_id = item_data.get('id')
        will_edit_target = (
            item_type in CATEGORIES and instance_id is not None and
            self.canvas.current_tool in {'brush', 'eraser'} and
            self.canvas.get_instance_geometry(instance_id, item_type).area > 0)
        if (self.canvas.editing_instance_id > 0 and
            (self.canvas.editing_instance_id != instance_id or
             self.canvas.editing_instance_category != item_type) and not will_edit_target):
            self.canvas.commit_editing()

        # Handle chamber/hive/pollen instance selection
        if item_type in CATEGORIES and item_type != 'bee':
            # Switch to that annotation mode
            self._sync_annotation_type_control(item_type, preserve_active_edit=will_edit_target)
            
            # Select the instance
            instance_id = item_data.get('id')
            if instance_id is not None:
                self.canvas.set_selected_instance(instance_id, category=item_type, zoom=False)
                self.canvas.highlight_instance(instance_id, category=item_type)
            
            self.status_label.setText(f"Switched to {item_type.capitalize()} mode - selected instance {instance_id}")
            # Don't rebuild the list - we're responding to a click on it!
            # The list is already correct and rebuilding it disrupts the selection
            return
        
        # Handle regular bee instance selection
        if item_type == 'bee':
            instance_id = item_data.get('id')
            if instance_id is not None:
                self._sync_annotation_type_control('bee', preserve_active_edit=will_edit_target)
                self.canvas.set_selected_instance(instance_id, category='bee', zoom=False)
                self.canvas.highlight_instance(instance_id, category='bee')
                self.status_label.setText(f"Selected instance {instance_id} - Use Brush/Eraser to edit")
    
    def on_instance_clicked(self, item):
        """Handle click on instance list item (including re-clicks on already selected item)
        
        Args:
            item: QListWidgetItem that was clicked
        """
        if item is None:
            return
        
        # Get the row of the clicked item
        idx = self.instance_list.row(item)
        
        # Call the same handler as row change - this ensures clicks on already-selected items work
        self.on_instance_changed(idx)
        
        # Set focus to canvas so keyboard shortcuts (spacebar, etc.) work immediately
        self.canvas.setFocus()

    def on_canvas_instance_selected(self, instance_id, category):
        """Select the matching row when an instance is clicked on the canvas."""
        if getattr(self, '_instance_list_updating', False):
            return

        self._sync_annotation_type_control(category, preserve_active_edit=True)

        matching_row = -1
        for row in range(self.instance_list.count()):
            item = self.instance_list.item(row)
            item_data = item.data(Qt.ItemDataRole.UserRole) if item else None
            if not item_data:
                continue
            if item_data.get('id') == instance_id and item_data.get('type') == category:
                matching_row = row
                break

        if matching_row < 0:
            self.update_instance_list_from_canvas()
            for row in range(self.instance_list.count()):
                item = self.instance_list.item(row)
                item_data = item.data(Qt.ItemDataRole.UserRole) if item else None
                if item_data and item_data.get('id') == instance_id and item_data.get('type') == category:
                    matching_row = row
                    break

        if matching_row < 0:
            return

        self.instance_list.blockSignals(True)
        self.instance_list.clearSelection()
        self.instance_list.setCurrentRow(matching_row)
        selected_item = self.instance_list.item(matching_row)
        if selected_item is not None:
            selected_item.setSelected(True)
            self.instance_list.scrollToItem(selected_item)
        self.instance_list.blockSignals(False)
    
    def on_instance_double_clicked(self, item):
        """Handle double-click on instance to edit ID"""
        idx = self.instance_list.currentRow()
        
        if self.canvas.combined_mask is None:
            return
        
        # Use cached instance IDs for performance
        instance_ids = self.canvas.get_instance_ids()
        
        if idx >= 0 and idx < len(instance_ids):
            current_id = int(instance_ids[idx])
            
            # Check if there are subsequent frames in the current video
            has_next_frames = False
            if self.current_video_id and self.frames:
                # Get frames for current video
                video_frames = [f for f in self.frames if self.current_video_id in str(f)]
                if video_frames:
                    current_frame_path = self.frames[self.current_frame_idx]
                    current_idx_in_video = video_frames.index(current_frame_path) if current_frame_path in video_frames else -1
                    has_next_frames = current_idx_in_video >= 0 and current_idx_in_video < len(video_frames) - 1
            
            # Show custom dialog
            from gui.dialogs import EditBeeIdDialog
            dialog = EditBeeIdDialog(
                self,
                current_id=current_id,
                existing_ids=list(instance_ids),
                has_next_frames=has_next_frames
            )
            
            if dialog.exec() == QDialog.DialogCode.Accepted:
                new_id = dialog.get_new_id()
                
                if new_id != current_id:
                    # Check if ID already exists
                    if new_id in instance_ids and new_id != current_id:
                        QMessageBox.warning(
                            self,
                            "Duplicate ID",
                            f"Bee ID {new_id} already exists. Please choose a different ID."
                        )
                        return
                    
                    # Update ID in current frame
                    self._update_instance_id_in_frame(current_id, new_id)
                    
                    # If user wants to propagate, update all subsequent frames
                    if dialog.should_propagate() and has_next_frames:
                        self._propagate_id_change_through_video(current_id, new_id)
    
    def _update_instance_id_in_frame(self, old_id, new_id):
        """Update an instance ID in the current frame"""
        # Update ID in combined mask (replace all pixels with old_id to new_id)
        if self.canvas.combined_mask is not None:
            self.canvas.combined_mask[self.canvas.combined_mask == old_id] = new_id
        
        # Update editing_instance_id if the instance being edited is the one being renamed
        if self.canvas.editing_instance_id == old_id:
            self.canvas.editing_instance_id = new_id
        
        # Update selected_mask_idx if the selected instance is the one being renamed
        if self.canvas.selected_mask_idx == old_id:
            self.canvas.selected_mask_idx = new_id
        
        # Update annotation_metadata (for bbox-only instances or metadata associated with masks)
        if old_id in self.canvas.annotation_metadata:
            metadata = self.canvas.annotation_metadata[old_id]
            metadata['mask_id'] = new_id  # Update the mask_id field
            self.canvas.annotation_metadata[new_id] = metadata
            del self.canvas.annotation_metadata[old_id]
        
        # Invalidate cached instance IDs since we modified annotation_metadata
        self.canvas._cached_instance_ids = None
        
        # Update bbox_items_map (if instance has a bbox visual element)
        if old_id in self.canvas.bbox_items_map:
            self.canvas.bbox_items_map[new_id] = self.canvas.bbox_items_map[old_id]
            del self.canvas.bbox_items_map[old_id]
        
        # Update color mapping
        if old_id in self.canvas.mask_colors:
            self.canvas.mask_colors[new_id] = self.canvas.mask_colors[old_id]
            del self.canvas.mask_colors[old_id]
        
        # Update video color mapping if tracking is enabled
        if self.current_video_id and old_id in self.video_mask_colors.get(self.current_video_id, {}):
            self.video_mask_colors[self.current_video_id][new_id] = self.video_mask_colors[self.current_video_id][old_id]
            del self.video_mask_colors[self.current_video_id][old_id]
        
        # Update next_mask_id if necessary
        if new_id >= self.canvas.next_mask_id:
            self.canvas.next_mask_id = new_id + 1
        
        # Mark frame as modified so it will be saved
        self._mark_current_annotations_dirty()
        
        # Rebuild visualization
        self.canvas.rebuild_visualizations()
        
        # Refresh the list
        self.update_instance_list_from_canvas()
        self.status_label.setText(f"Updated bee ID from {old_id} to {new_id}")
    
    def _propagate_id_change_through_video(self, old_id, new_id):
        """Propagate an ID change through all subsequent frames in the current video"""
        if not self.current_video_id or not self.frames or not self.project_path:
            return
        
        # Save current frame first (blocking save to disk)
        annotations = self.canvas.get_annotations()
        if annotations:
            try:
                save_summary = self._save_annotation_sources(
                    annotations=annotations,
                    force=True,
                    refresh_next_id=False
                )
                self.annotation_manager.unsaved_changes = False  # Clear unsaved flag since we just saved
                print(
                    f"✓ Saved current frame (frame {save_summary['frame_idx']}) "
                    f"to disk before propagation"
                )
            except Exception as e:
                print(f"Warning: Failed to save current frame before propagation: {e}")
                import traceback
                traceback.print_exc()
        
        # Get all frames for current video
        video_frames = [f for f in self.frames if self.current_video_id in str(f)]
        if not video_frames:
            return
        
        current_frame_path = self.frames[self.current_frame_idx]
        if current_frame_path not in video_frames:
            return
        
        current_idx_in_video = video_frames.index(current_frame_path)
        subsequent_frames = video_frames[current_idx_in_video + 1:]
        
        if not subsequent_frames:
            return
        
        # Show progress dialog
        from PyQt6.QtWidgets import QProgressDialog
        progress = QProgressDialog(
            f"Propagating bee ID change from {old_id} to {new_id}...",
            "Cancel",
            0,
            len(subsequent_frames),
            self
        )
        progress.setWindowModality(Qt.WindowModality.WindowModal)
        progress.setMinimumDuration(0)
        progress.setValue(0)
        
        frames_updated = 0
        
        # Process each subsequent frame
        for i, frame_path in enumerate(subsequent_frames):
            if progress.wasCanceled():
                break
            
            progress.setLabelText(
                f"Propagating bee ID change from {old_id} to {new_id}...\n"
                f"Frame {i + 1} of {len(subsequent_frames)}"
            )
            progress.setValue(i)
            QApplication.processEvents()
            
            # Get frame index in global frame list
            global_frame_idx = self.frames.index(frame_path)
            
            # Get frame index within the video
            frame_idx_in_video = self._get_frame_idx_in_video(global_frame_idx)
            
            # Load annotations for this frame
            annotations = self.annotation_manager.load_frame_annotations(
                self.project_path, self.current_video_id, frame_idx_in_video
            )
            
            if annotations:
                # Check if old_id exists in this frame
                modified = False
                for ann in annotations:
                    if ann.get('mask_id') == old_id or ann.get('instance_id') == old_id:
                        # Update the ID
                        if 'mask_id' in ann:
                            ann['mask_id'] = new_id
                        if 'instance_id' in ann:
                            ann['instance_id'] = new_id
                        modified = True
                
                # Save if modified
                if modified:
                    try:
                        # Save to disk (this writes the file immediately)
                        self.annotation_manager.save_frame_annotations(
                            self.project_path, self.current_video_id, frame_idx_in_video, annotations
                        )
                        # Clear from cache so it will reload from disk when needed
                        # (avoids marking as "unsaved" which could trigger unwanted background saves)
                        if global_frame_idx in self.annotation_manager.frame_annotations:
                            del self.annotation_manager.frame_annotations[global_frame_idx]
                        frames_updated += 1
                        print(f"  ✓ Saved frame {frame_idx_in_video} to disk with updated ID")
                    except Exception as e:
                        print(f"  ✗ Error saving frame {frame_idx_in_video}: {e}")
                        import traceback
                        traceback.print_exc()
        
        progress.setValue(len(subsequent_frames))
        
        # Show completion message
        if frames_updated > 0:
            QMessageBox.information(
                self,
                "Propagation Complete",
                f"Successfully updated bee ID from {old_id} to {new_id} in {frames_updated} subsequent frame(s)."
            )
            self.status_label.setText(f"Propagated ID change to {frames_updated} frame(s)")
        else:
            QMessageBox.information(
                self,
                "No Changes",
                f"Bee ID {old_id} was not found in any subsequent frames."
            )
            self.status_label.setText("No frames needed updating")
    
    def show_instance_context_menu(self, position):
        """Show context menu for instance list"""
        selected_items = self.instance_list.selectedItems()
        if not selected_items:
            return
        selected_item_data = [
            dict(item.data(Qt.ItemDataRole.UserRole) or {})
            for item in selected_items
        ]
        first_item_data = selected_item_data[0] if selected_item_data else {}
        
        from PyQt6.QtWidgets import QMenu
        menu = QMenu()
        
        # For single selection, show edit option
        if len(selected_items) == 1:
            edit_id_action = menu.addAction("Edit Bee ID...")
            
            # Check if instance has ArUco marker
            item_data = first_item_data
            if item_data and item_data.get('type') == 'bee':
                instance_id = item_data.get('id')
                
                # Check if instance has ArUco from video-level tracking
                has_aruco = False
                if self.current_video_id and self.project_path:
                    _, aruco_tracking = self.annotation_manager.load_video_annotations(
                        self.project_path, self.current_video_id
                    )
                    # Check if any ArUco code points to this instance
                    for aruco_str, inst_id in aruco_tracking.items():
                        if inst_id == instance_id:
                            has_aruco = True
                            break
                
                # Add option to clear ArUco tracking if instance has a marker
                if has_aruco:
                    clear_aruco_action = menu.addAction("Remove ArUco Association...")
                else:
                    clear_aruco_action = None
            else:
                clear_aruco_action = None
            
            # Add category change submenu
            if item_data:
                instance_id = item_data.get('id')
                current_category = item_data.get('type', 'bee')
                
                change_category_menu = menu.addMenu(f"Change Category")
                category_actions = {
                    action: category for category, action in
                    add_category_menu_actions(change_category_menu, current_category).items()
                    if category != current_category
                }
            else:
                category_actions = {}
            
            delete_action = menu.addAction("Delete Instance")
        else:
            # For multiple selection, only show delete
            delete_action = menu.addAction(f"Delete {len(selected_items)} Instances")
            edit_id_action = None
            clear_aruco_action = None
            category_actions = {}
        
        # Copy destinations are independent of the frame-list display filter.
        propagate_action = None
        propagate_through_action = None
        if self._get_training_copy_targets():
            menu.addSeparator()
            if len(selected_items) == 1:
                propagate_action = menu.addAction("Copy to Next Training Frame...")
                propagate_through_action = menu.addAction("Copy Through Training Frames...")
            else:
                propagate_through_action = menu.addAction(f"Copy {len(selected_items)} Instances Through Training Frames...")
        
        action = menu.exec(self.instance_list.mapToGlobal(position))
        
        if edit_id_action and action == edit_id_action:
            # Trigger the same edit dialog as double-click
            idx = self.instance_list.currentRow()
            item = self.instance_list.item(idx)
            if item:
                self.on_instance_double_clicked(item)
        elif clear_aruco_action and action == clear_aruco_action:
            # Clear ArUco association for this instance
            instance_id = first_item_data.get('id')
            if instance_id is not None:
                self.clear_instance_aruco_association(instance_id)
        elif len(selected_items) == 1 and action in category_actions:
            # Handle category change
            new_category = category_actions[action]
            instance_id = first_item_data.get('id')
            current_category = first_item_data.get('type', 'bee')
            if instance_id is not None:
                self.canvas.change_instance_category(
                    instance_id,
                    new_category,
                    old_category=current_category
                )
                # Update the instance list to reflect the change
                self._schedule_instance_list_update()
        elif action == delete_action:
            if len(selected_items) == 1:
                self.delete_selected_instance()
            else:
                self.delete_selected_instances()
        elif propagate_action and action == propagate_action:
            self.propagate_selected_instance_to_next_frame()
        elif propagate_through_action and action == propagate_through_action:
            if len(selected_items) == 1:
                self.propagate_selected_instance_through_frames()
            else:
                self.propagate_selected_instances_through_frames()
    
    def _get_training_copy_targets(self):
        return [index for index in training_copy_targets(
            self.current_frame_idx, self.current_video_id,
            self.frame_video_ids, self.frame_splits, self.frame_selected,
        ) if index < len(self.frames)]

    def _training_copy_existing_annotations(self, list_idx):
        key = (self.current_video_id, list_idx)
        if key in self.annotation_manager.frame_annotations:
            annotations = self.annotation_manager.get_frame_annotations(
                list_idx, video_id=self.current_video_id)
        else:
            annotations = self.annotation_manager.load_frame_annotations(
                self.project_path, self.current_video_id,
                self._get_frame_idx_in_video(list_idx)) or []
        frame_annotations, _ = self._split_source_annotations(annotations)
        return frame_annotations

    def _copy_masks_to_training_frame(self, list_idx, sources, replace, shared):
        """Write one eligible frame; never navigate the canvas during batch copies."""
        if list_idx not in self._get_training_copy_targets():
            raise ValueError('Destination is no longer a selected training frame in this video.')
        frame = self.frames[list_idx]
        image = frame if isinstance(frame, np.ndarray) else cv2.imread(str(frame), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError(f'Cannot read destination image: {frame}')
        existing = self._training_copy_existing_annotations(list_idx)
        updated, copied, skipped = merge_copied_masks(
            existing, sources, image.shape, shared=shared, replace=replace)
        if copied:
            frame_number = self._get_frame_idx_in_video(list_idx)
            # Persist before caching so eviction, cancellation, and closing do not
            # lose completed copies. Shared video annotation files are untouched.
            self.annotation_manager.save_frame_annotations(
                self.project_path, self.current_video_id, frame_number, updated)
            self.annotation_manager.set_frame_annotations(
                list_idx, updated, video_id=self.current_video_id)
            self.dirty_frame_annotation_keys.discard((self.current_video_id, frame_number))
            self.coco_export_dirty = True
            self._schedule_done_counter_update(saved_counts_dirty=True)
        return copied, skipped

    def _copy_selected_training_masks(self, next_only=False):
        if not self.project_path or not self.current_video_id:
            return
        targets = self._get_training_copy_targets()
        if not targets:
            QMessageBox.information(
                self, "No Training Frames",
                "There are no subsequent selected training frames in this video. "
                "Validation videos and unselected frames are not copy destinations.")
            return
        try:
            # Sidebar rows are category-sorted and are not numeric instance IDs.
            keys = self._get_selected_instance_keys()
            sources = selected_copy_masks(
                self.canvas.get_annotations(), keys, self.annotation_manager.project_info)
        except ValueError as exc:
            QMessageBox.warning(self, "Cannot Copy Masks", str(exc))
            return

        source_context = (self.project_path, self.current_video_id, self.current_frame_idx)
        if next_only:
            targets = targets[:1]
        dialog = TrainingMaskCopyDialog(
            self, [(index, self._get_frame_idx_in_video(index)) for index in targets],
            len(sources))
        if next_only:
            dialog.setWindowTitle("Copy to Next Training Frame")
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        targets = targets[:dialog.endpoint.currentIndex() + 1]
        replace = dialog.replace_check.isChecked()
        if source_context != (self.project_path, self.current_video_id, self.current_frame_idx):
            QMessageBox.warning(self, "Copy Canceled", "The source frame changed. Please try again.")
            return

        progress = QProgressDialog("Saving source annotations...", "Cancel", 0, len(targets), self)
        progress.setWindowTitle("Copy Training Masks")
        progress.setWindowModality(Qt.WindowModality.WindowModal)
        progress.setMinimumDuration(0)
        progress.setAutoClose(False)
        progress.setAutoReset(False)
        progress.setValue(0)
        completed, skipped_count, failures = [], 0, []
        canceled = False
        try:
            # A pending autosave must not overwrite a newly copied destination.
            self.save_worker.wait_until_idle()
            self._commit_canvas_edit_if_needed()
            self._save_annotation_sources(use_background=False)
            shared, _ = self.annotation_manager.load_video_annotations(
                self.project_path, self.current_video_id)
            shared = merge_annotations([], shared, self.annotation_manager.project_info)
            for position, index in enumerate(targets):
                frame_number = self._get_frame_idx_in_video(index)
                progress.setLabelText(f"Copying to training frame {frame_number:06d}...")
                QApplication.processEvents()
                if progress.wasCanceled():
                    canceled = True
                    break
                if source_context != (self.project_path, self.current_video_id, self.current_frame_idx):
                    raise ValueError('Source frame changed; copying stopped.')
                try:
                    copied, skipped = self._copy_masks_to_training_frame(index, sources, replace, shared)
                except ValueError as exc:
                    failures.append(f"Frame {frame_number:06d}: {exc}")
                else:
                    skipped_count += len(skipped)
                    if copied:
                        completed.append((index, copied))
                progress.setValue(position + 1)
        except Exception as exc:
            failures.append(f"Copy stopped: {exc}")
        finally:
            progress.close()

        summary = (f"Saved copies to {len(completed)} training frame(s). "
                   f"Skipped {skipped_count} existing matching instance(s).")
        if canceled:
            summary += " Canceled; completed copies were kept."
        if failures:
            summary += f" {len(failures)} issue(s); see details."
        self.status_label.setText(summary)

        if next_only and completed and not failures and not canceled:
            index, copied_keys = completed[0]
            self.load_frame(index)
            if self.current_frame_idx == index:
                instance_id, category = copied_keys[0]
                self.canvas.current_annotation_type = category
                self.canvas.set_instance_visible(instance_id, category, True, rebuild=False)
                self.toolbar.set_tool('brush')
                self.canvas.set_selected_instance(instance_id, category=category, zoom=False)
                self.update_instance_list_from_canvas()
                self.status_label.setText(
                    f"Copied {category} {instance_id} to training frame "
                    f"{self._get_frame_idx_in_video(index):06d}. Review where bees moved.")
        else:
            message = QMessageBox(self)
            message.setWindowTitle("Training Mask Copy")
            message.setText(summary)
            if completed:
                message.setInformativeText(
                    "Review each copied mask where bees have moved or nest visibility has changed.")
            if failures:
                message.setIcon(QMessageBox.Icon.Warning)
                message.setDetailedText("\n".join(failures))
            message.exec()

    def propagate_selected_instance_through_frames(self):
        self._copy_selected_training_masks()

    def propagate_selected_instances_through_frames(self):
        self._copy_selected_training_masks()

    def propagate_selected_instance_to_next_frame(self):
        self._copy_selected_training_masks(next_only=True)

    def delete_selected_instance(self):
        """Delete the currently selected instance"""
        selected_instances = self._get_selected_instance_keys()
        if len(selected_instances) > 1:
            self.delete_selected_instances()
            return

        if selected_instances:
            instance_id, category = selected_instances[0]
        elif self.canvas.selected_mask_idx > 0:
            instance_id = self.canvas.selected_mask_idx
            category = (
                self.canvas.selected_instance_category or
                self.canvas.annotation_metadata.get(instance_id, {}).get('category')
            )
        else:
            self.status_label.setText("No instance selected")
            return

        if self.canvas.delete_instance(instance_id, category=category):
            self.update_instance_list_from_canvas()
            self.status_label.setText(f"{category.capitalize() if category else 'Instance'} {instance_id} deleted")
        else:
            self.status_label.setText("Selected instance was not found")

    def _get_selected_instance_keys(self):
        """Return selected sidebar instances as unique (id, category) pairs."""
        selected_instances = []
        seen = set()
        for item in self.instance_list.selectedItems():
            item_data = item.data(Qt.ItemDataRole.UserRole) if item else None
            if not item_data:
                continue
            instance_id = item_data.get('id')
            category = item_data.get('type', 'bee')
            if instance_id is None:
                continue
            key = (int(instance_id), category)
            if key not in seen:
                selected_instances.append(key)
                seen.add(key)
        return selected_instances
    
    def delete_selected_instances(self):
        """Delete multiple selected instances"""
        selected_instances = self._get_selected_instance_keys()
        if not selected_instances:
            QMessageBox.warning(
                self, "No Instances Selected",
                "Please select one or more instances to delete."
            )
            return

        display_ids = [
            f"{category} {instance_id}"
            for instance_id, category in selected_instances
        ]
        
        # Confirm deletion
        reply = QMessageBox.question(
            self,
            "Confirm Delete",
            f"Delete {len(selected_instances)} instance(s)?\n"
            f"Instances: {', '.join(display_ids)}",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No
        )
        
        if reply != QMessageBox.StandardButton.Yes:
            return
        
        deleted_count = 0
        for instance_id, category in selected_instances:
            if self.canvas.delete_instance(
                instance_id,
                category=category,
                rebuild=False,
                emit_changed=False
            ):
                deleted_count += 1

        self.canvas.rebuild_visualizations()
        if deleted_count > 0:
            self.canvas.annotation_changed.emit()
        self.update_instance_list_from_canvas()
        self.status_label.setText(f"{deleted_count} instance(s) deleted")
    
    def delete_all_instances(self):
        """Delete all bee instances in the current frame and remove annotation files"""
        # Check if there are any annotations to delete
        annotations = self.canvas.get_annotations()
        if not annotations:
            QMessageBox.information(
                self, "No Instances",
                "There are no instances to delete in the current frame."
            )
            return
        
        # Filter to only bee instances (don't delete hive/chamber/pollen)
        bee_annotations = [a for a in annotations if a.get('category', 'bee') == 'bee']
        other_annotations = [a for a in annotations if a.get('category', 'bee') != 'bee']
        
        if not bee_annotations:
            QMessageBox.information(
                self, "No Bee Instances",
                "There are no bee instances to delete in the current frame.\n\n"
                f"(Frame has {len(other_annotations)} hive/chamber annotations which are preserved.)"
            )
            return
        
        # Show confirmation dialog
        num_instances = len(bee_annotations)
        confirm_msg = f"Are you sure you want to delete ALL {num_instances} bee instance(s) in this frame?\n\n"
        confirm_msg += f"This will remove the bee annotation files from disk.\n\n"
        if other_annotations:
            confirm_msg += f"Note: {len(other_annotations)} hive/chamber annotations will be preserved.\n\n"
        confirm_msg += f"This action cannot be undone."
        
        reply = QMessageBox.question(
            self, "Delete All Bee Instances",
            confirm_msg,
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No
        )
        
        if reply == QMessageBox.StandardButton.Yes:
            # Keep only non-bee annotations (hive/chamber/pollen)
            self.canvas.set_annotations(other_annotations)
            self.update_instance_list_from_canvas()
            
            # Delete annotation files from disk if we have a project and video
            if self.project_path and self.current_video_id is not None:
                try:
                    from pathlib import Path
                    project_path = Path(self.project_path)
                    video_id = self.current_video_id
                    
                    # Get frame index within video (not global frame index)
                    frame_idx_in_video = self._get_frame_idx_in_video(self.current_frame_idx)
                    
                    # Frame files can also contain nest labels. Replace their
                    # contents instead of deleting the entire annotation source.
                    remaining_frame, _ = self._split_source_annotations(other_annotations)
                    self.annotation_manager.save_frame_annotations(
                        project_path, video_id, frame_idx_in_video, remaining_frame
                    )
                    
                    # Delete pickle file if it exists (legacy format)
                    pkl_file = project_path / 'annotations' / 'pkl' / video_id / f'frame_{frame_idx_in_video:06d}.pkl'
                    if pkl_file.exists():
                        pkl_file.unlink()
                        print(f"Deleted pickle annotation: {pkl_file}")
                    
                    # Clear the annotation cache for this frame
                    if self.current_frame_idx in self.annotation_manager.frame_annotations:
                        del self.annotation_manager.frame_annotations[self.current_frame_idx]
                        print(f"Cleared cache for frame {self.current_frame_idx}")

                    cache_key = (video_id, self.current_frame_idx)
                    if cache_key in self.annotation_manager.frame_annotations:
                        del self.annotation_manager.frame_annotations[cache_key]
                        print(f"Cleared video cache for frame {self.current_frame_idx}")

                    next_id = self._refresh_video_next_mask_id(video_id)
                    
                    self.status_label.setText(
                        f"✓ Deleted all {num_instances} bee instance(s); next ID is {next_id}"
                    )
                    self.dirty_frame_annotation_keys.discard((video_id, frame_idx_in_video))
                    self.coco_export_dirty = True
                    self.current_frame_modified = False  # Mark as not modified since we explicitly deleted
                    
                except Exception as e:
                    import traceback
                    error_msg = f"Error deleting annotation files: {str(e)}\n\n{traceback.format_exc()}"
                    print(error_msg)
                    QMessageBox.warning(
                        self, "File Deletion Error",
                        f"Instances were cleared from canvas, but there was an error deleting files:\n{str(e)}"
                    )
                    self.status_label.setText(f"⚠ Deleted instances but file deletion failed")
            else:
                self.status_label.setText(f"✓ Deleted all {num_instances} bee instance(s)")
                self._mark_current_annotations_dirty()
    
    def clear_selected_instance(self):
        """Clear the selected instance's mask and prompt points"""
        if self.canvas.selected_mask_idx >= 0:
            # Clear the mask of the selected instance and prompts
            self.canvas.clear_selected_instance()
            # Update UI
            self.update_instance_list_from_canvas()
            self.status_label.setText("Instance cleared - ready to re-annotate")
        else:
            self.status_label.setText("No instance selected - select one first")
            
    def on_clear_prompts_requested(self):
        """Handle clear prompts request"""
        self.canvas.clear_prompt_points()
        self.update_instance_list_from_canvas()
        self.status_label.setText("Prompts cleared")

    def on_instance_bbox_visibility_changed(self, item):
        """Toggle one instance's mask/bbox display from the instance-list checkbox."""
        if getattr(self, '_instance_list_updating', False) or item is None:
            return

        item_data = item.data(Qt.ItemDataRole.UserRole)
        if not item_data:
            return

        instance_id = item_data.get('id')
        category = item_data.get('type', 'bee')
        if instance_id is None:
            return

        visible = item.checkState() == Qt.CheckState.Checked

        if not self.canvas.annotation_type_visibility.get(category, True):
            self._instance_list_updating = True
            item.setCheckState(
                Qt.CheckState.Checked
                if self.canvas.is_instance_enabled(instance_id, category)
                else Qt.CheckState.Unchecked
            )
            self._instance_list_updating = False
            self.status_label.setText(f"Turn on {category} visibility before changing instance {instance_id}")
            self.canvas.setFocus()
            return

        self.canvas.set_instance_visible(instance_id, category, visible)
        state = "shown" if visible else "hidden"
        self.status_label.setText(f"{category.capitalize()} instance {instance_id} {state}")
        self.canvas.setFocus()
    
    def on_sam2_run_on_bbox(self):
        """Run SAM2 on the selected instance's bounding box"""
        # Check if SAM2 is loaded
        if not self.sam2:
            QMessageBox.warning(self, "SAM2 Not Loaded", "Please load a SAM2 checkpoint first.")
            return
        
        # Check if an instance is selected
        if self.canvas.selected_mask_idx <= 0:
            QMessageBox.warning(self, "No Instance Selected", "Please select an instance from the list first.")
            return
        
        selected_id = self.canvas.selected_mask_idx
        bbox = None
        
        # First, try to get bbox from annotation metadata (works for both bbox-only and masked instances)
        if selected_id in self.canvas.annotation_metadata:
            metadata = self.canvas.annotation_metadata[selected_id]
            if 'bbox' in metadata:
                bbox = metadata['bbox']  # [x, y, width, height]
                x, y, w, h = bbox
                x1, y1, x2, y2 = x, y, x + w, y + h
                print(f"Using bbox from metadata: ({x1}, {y1}, {x2}, {y2})")
        
        # If no bbox in metadata, try to extract from mask
        if bbox is None:
            if self.canvas.combined_mask is None:
                QMessageBox.warning(self, "No Bbox Data", "Selected instance has no bounding box or mask data.")
                return
            
            # Extract the mask for the selected instance
            instance_mask = (self.canvas.combined_mask == selected_id).astype(np.uint8) * 255
            
            # Check if mask is empty
            if not np.any(instance_mask > 0):
                QMessageBox.warning(self, "No Data", "Selected instance has no bounding box or mask data.")
                return
            
            # Get bounding box from the mask
            bbox = self.canvas.get_mask_bbox(instance_mask)
            if bbox is None:
                QMessageBox.warning(self, "No Bounding Box", "Could not compute bounding box for selected instance.")
                return
            
            x, y, w, h = bbox
            x1, y1, x2, y2 = x, y, x + w, y + h
            print(f"Extracted bbox from mask: ({x1}, {y1}, {x2}, {y2})")
        
        try:
            print(f"Running SAM2 on instance {selected_id} bbox: ({x1}, {y1}, {x2}, {y2})")
            
            # Convert image to RGB for SAM2 if grayscale
            import cv2
            image_for_sam2 = self.canvas.current_image
            if len(image_for_sam2.shape) == 2:
                image_for_sam2 = cv2.cvtColor(image_for_sam2, cv2.COLOR_GRAY2RGB)
            
            # Run SAM2 prediction with the bounding box
            mask = self.sam2.predict_with_box(
                image_for_sam2,
                x1, y1, x2, y2
            )
            print(f"SAM2 mask shape: {mask.shape}, unique values: {np.unique(mask)}")
            
            # Start editing mode for this instance
            selected_category = self.canvas.selected_instance_category
            if (self.canvas.editing_instance_id != selected_id or
                self.canvas.editing_instance_category != selected_category):
                self.canvas.start_editing_instance(selected_id, category=selected_category)
            
            # Update the editing mask with SAM2 prediction
            self.canvas.editing_mask = (mask > 0).astype(np.uint8) * 255
            self.canvas._update_editing_visualization()
            
            # Update instance list
            self.update_instance_list_from_canvas()
            
            self.status_label.setText(f"SAM2 refined instance {selected_id} using bbox (in editing mode - click away to confirm)")
            print(f"SAM2 refinement complete for instance {selected_id}")
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            QMessageBox.warning(self, "SAM2 Error", f"SAM2 prediction failed: {str(e)}")
            self.status_label.setText(f"SAM2 refinement failed: {str(e)}")
            
    def new_instance(self, category=None):
        """Start annotating a new instance"""
        # Commit any pending edits
        if self.canvas.editing_instance_id > 0:
            self.canvas.commit_editing()
        
        # Deselect any current bbox
        self.canvas._deselect_bbox()
        
        # Create a new instance ID
        new_instance_id = self.canvas.next_mask_id
        self.canvas.next_mask_id += 1
        
        # Set as selected instance
        self.canvas.selected_mask_idx = new_instance_id
        
        # Use the type chosen from the New Instance menu.
        current_category = category or 'bee'
        if current_category not in CATEGORIES:
            current_category = 'bee'
        self.canvas.current_annotation_type = current_category
        self.canvas.selected_instance_category = current_category
        
        # Add to annotation metadata to make it appear in the list
        if new_instance_id not in self.canvas.annotation_metadata:
            self.canvas.annotation_metadata[new_instance_id] = {
                'bbox': [0, 0, 0, 0],  # Placeholder bbox
                'bbox_only': True,
                'category': current_category,
                'source': 'manual',
            }
        
        # Invalidate cached instance IDs so new instance appears in list
        self.canvas._cached_instance_ids = None
        
        # Generate category-specific color for new instance. Only bees get
        # per-instance colors; chamber/hive/pollen use fixed category colors.
        self.canvas.ensure_instance_color(new_instance_id, current_category)

        if current_category in BROOD_CATEGORIES:
            self.toolbar.brood_visibility_actions[current_category].setChecked(True)
        elif current_category == 'nectar':
            self.toolbar.show_nectar_checkbox.setChecked(True)
        self.canvas.set_instance_visible(new_instance_id, current_category, True, rebuild=False)

        self.toolbar.set_tool('brush')
        self.canvas.start_editing_instance(new_instance_id, category=current_category)
        
        # Update list to show new instance
        self.update_instance_list_from_canvas()
        
        # Find and select the new instance in the list
        for i in range(self.instance_list.count()):
            item = self.instance_list.item(i)
            item_data = item.data(Qt.ItemDataRole.UserRole) if item else None
            if item_data and item_data.get('id') == new_instance_id and item_data.get('type') == current_category:
                self.instance_list.setCurrentRow(i)
                break
        
        self.status_label.setText(f"Instance {new_instance_id} created - Brush selected")
    
    def _register_canvas_colors(self):
        """Register current canvas mask colors for the current video"""
        if not self.current_video_id:
            return
        
        # Initialize color mapping for this video if needed
        if self.current_video_id not in self.video_mask_colors:
            self.video_mask_colors[self.current_video_id] = {}
        
        # Register colors from canvas (mask_colors is dict {instance_id: (r,g,b)}).
        # Fixed-color categories overwrite stale cached colors; bees keep their
        # existing per-instance cached color when present.
        for mask_id, color in self.canvas.mask_colors.items():
            category = self.canvas.annotation_metadata.get(mask_id, {}).get('category', 'bee')
            color = self.canvas.ensure_instance_color(
                mask_id,
                category,
                preferred_color=color
            )
            if category != 'bee' or mask_id not in self.video_mask_colors[self.current_video_id]:
                self.video_mask_colors[self.current_video_id][mask_id] = color

    def propagate_to_next_selected(self):
        """Propagate annotations through all frames until the next selected (train/val) frame"""
        from PyQt6.QtWidgets import QProgressDialog
        
        # Check if we have annotations to propagate
        current_annotations = self.canvas.get_annotations()
        if not current_annotations:
            QMessageBox.warning(self, "No Annotations", "No annotations to propagate. Annotate some instances first.")
            return
        
        # Check if SAM2 is loaded
        print(f"Propagate to selected: Checking SAM2. self.sam2 is None: {self.sam2 is None}, type: {type(self.sam2) if self.sam2 else 'None'}")
        if not self.sam2:
            QMessageBox.warning(self, "SAM2 Not Loaded", "SAM2 model must be loaded for propagation.")
            return
        
        # Find the next selected frame
        current_idx = self.current_frame_idx
        next_selected_idx = None
        
        for i in range(current_idx + 1, len(self.frames)):
            if i < len(self.frame_selected) and self.frame_selected[i]:
                next_selected_idx = i
                break
        
        if next_selected_idx is None:
            QMessageBox.information(self, "No More Frames", "No more selected training/validation frames found.")
            return
        
        num_frames = next_selected_idx - current_idx
        
        # Create progress dialog
        progress = QProgressDialog(
            f"Propagating through {num_frames} frames...",
            "Cancel",
            0,
            num_frames,
            self
        )
        progress.setWindowTitle("SAM2 Propagation")
        progress.setWindowModality(Qt.WindowModality.WindowModal)
        progress.setMinimumDuration(0)
        progress.setValue(0)
        
        # Propagate frame by frame
        successful_propagations = 0
        for step in range(num_frames):
            # Check if user cancelled
            if progress.wasCanceled():
                self.status_label.setText(f"Propagation cancelled at frame {self.current_frame_idx}")
                break
            
            # Update progress
            progress.setValue(step)
            progress.setLabelText(
                f"Propagating frame {self.current_frame_idx} → {self.current_frame_idx + 1}\n"
                f"({step + 1} of {num_frames})"
            )
            QApplication.processEvents()
            
            # Propagate to next frame
            try:
                self.propagate_to_next_frame()
                successful_propagations += 1
            except Exception as e:
                print(f"Error propagating frame {self.current_frame_idx}: {e}")
                import traceback
                traceback.print_exc()
                
                reply = QMessageBox.question(
                    self,
                    "Propagation Error",
                    f"Error at frame {self.current_frame_idx}:\n{str(e)}\n\nContinue propagating?",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
                )
                
                if reply == QMessageBox.StandardButton.No:
                    break
        
        progress.setValue(num_frames)
        progress.close()
        
        # Show completion message
        if not progress.wasCanceled():
            QMessageBox.information(
                self,
                "Propagation Complete",
                f"Successfully propagated through {successful_propagations} frames.\n"
                f"Now at frame {self.current_frame_idx} (selected for {self.frame_splits[self.current_frame_idx] if self.current_frame_idx < len(self.frame_splits) else 'unknown'})."
            )
            self.status_label.setText(f"✓ Propagated through {successful_propagations} frames")
    
    def propagate_to_next_frame(self):
        """Propagate current annotations to next frame using SAM2"""
        # Find next frame based on filter
        next_idx = self._get_next_frame_index()
        
        if next_idx is None:
            self.status_label.setText("Already at last frame in current view")
            return
            
        # Save current frame annotations (including manual edits)
        current_annotations = self.canvas.get_annotations()
        if not current_annotations:
            QMessageBox.warning(self, "No Annotations", "No annotations to propagate. Annotate some instances first.")
            return
            
        print(f"Propagate to next: Checking SAM2. self.sam2 is None: {self.sam2 is None}, type: {type(self.sam2) if self.sam2 else 'None'}")
            
        # Save to annotation manager
        self.annotation_manager.set_frame_annotations(
            self.current_frame_idx, current_annotations, video_id=self.current_video_id
        )
        
        # Get current masks and their IDs
        current_masks = [ann['mask'] for ann in current_annotations if 'mask' in ann]
        current_mask_ids = [ann.get('mask_id', i+1) for i, ann in enumerate(current_annotations) if 'mask' in ann]
        
        if not current_masks:
            QMessageBox.warning(self, "No Masks", "No masks to propagate.")
            return
        
        # Get next frame
        next_frame = self.frames[next_idx]
        
        # Load current frame image
        current_frame = self.frames[self.current_frame_idx]
        if isinstance(current_frame, Path) or isinstance(current_frame, str):
            import cv2
            # Try cache first (grayscale)
            current_frame_image = self.frame_cache.get(self.current_frame_idx)
            if current_frame_image is None:
                current_frame_image = cv2.imread(str(current_frame), cv2.IMREAD_GRAYSCALE)
            if current_frame_image is None:
                QMessageBox.warning(self, "Error", f"Failed to load current frame {self.current_frame_idx}")
                return
            # Convert to RGB for SAM2 if grayscale
            if len(current_frame_image.shape) == 2:
                current_frame_image = cv2.cvtColor(current_frame_image, cv2.COLOR_GRAY2RGB)
        else:
            current_frame_image = current_frame
        
        # Load next frame image
        if isinstance(next_frame, Path) or isinstance(next_frame, str):
            import cv2
            # Try cache first (grayscale)
            next_frame_image = self.frame_cache.get(next_idx)
            if next_frame_image is None:
                next_frame_image = cv2.imread(str(next_frame), cv2.IMREAD_GRAYSCALE)
            if next_frame_image is None:
                QMessageBox.warning(self, "Error", f"Failed to load frame {next_idx}")
                return
            # Convert to RGB for SAM2 if grayscale
            if len(next_frame_image.shape) == 2:
                next_frame_image = cv2.cvtColor(next_frame_image, cv2.COLOR_GRAY2RGB)
        else:
            next_frame_image = next_frame
        
        # Use SAM2 to propagate masks if available
        propagated_masks = []
        if self.sam2 and current_masks:
            try:
                self.status_label.setText("Propagating with SAM2 video predictor...")
                QApplication.processEvents()
                
                # Use video propagation for more accurate tracking
                propagated_masks = self.sam2.propagate_masks_to_frame_video(
                    current_frame_image,
                    current_masks,
                    next_frame_image,
                    mask_threshold=0.0
                )
                
                # Free memory immediately after propagation
                import gc
                del current_frame_image, next_frame_image
                gc.collect()
                
                # Clear CUDA cache if available
                try:
                    import torch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except:
                    pass
                
                # Filter out None or empty masks
                valid_propagated = []
                valid_mask_ids = []
                for i, m in enumerate(propagated_masks):
                    if m is not None and np.any(m > 0):
                        valid_propagated.append(m)
                        # Preserve mask ID from current frame
                        if i < len(current_mask_ids):
                            valid_mask_ids.append(current_mask_ids[i])
                        else:
                            valid_mask_ids.append(i+1)
                
                propagated_masks = valid_propagated
                
                print(f"SAM2 propagated {len(propagated_masks)} masks successfully")
                
                if len(propagated_masks) == 0:
                    QMessageBox.warning(self, "Propagation Failed", 
                                      "SAM2 failed to propagate masks. Try adjusting current masks or annotate next frame manually.")
                    # Free memory before returning
                    import gc
                    gc.collect()
                    # Still load the frame, but without masks
                    self.load_frame(next_idx)
                    self.update_instance_list_from_canvas()
                    self.status_label.setText(f"Frame {next_idx} loaded - propagation failed, annotate manually")
                    return
                    
            except Exception as e:
                print(f"Error: SAM2 propagation failed: {e}")
                import traceback
                traceback.print_exc()
                # Free memory after error
                import gc
                try:
                    del current_frame_image, next_frame_image
                except:
                    pass
                gc.collect()
                try:
                    import torch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except:
                    pass
                QMessageBox.warning(self, "Propagation Error", 
                                  f"SAM2 propagation failed: {str(e)}\n\nLoad frame without masks?",
                                  QMessageBox.StandardButton.Ok)
                # Load frame without masks
                self.load_frame(next_idx)
                self.update_instance_list_from_canvas()
                self.status_label.setText(f"Frame {next_idx} loaded - propagation failed")
                return
        else:
            # No SAM2 available
            if not self.sam2:
                QMessageBox.information(self, "No SAM2", 
                                      "SAM2 not loaded. Masks will not be propagated.\n\n"
                                      "Load a SAM2 checkpoint via the SAM2 toolbar to enable propagation.")
            # Load frame without propagating
            self.load_frame(next_idx)
            self.update_instance_list_from_canvas()
            self.status_label.setText(f"Frame {next_idx} loaded - no propagation (SAM2 not loaded)")
            return
        
        # Load the next frame first
        self.load_frame(next_idx)
        
        # Clear any existing annotations (including any loaded from disk)
        self.canvas.set_annotations([])
        
        # Get or create color mapping for this video
        if self.current_video_id not in self.video_mask_colors:
            self.video_mask_colors[self.current_video_id] = {}
        video_colors = self.video_mask_colors[self.current_video_id]
        
        # Add successfully propagated masks to canvas with their original IDs and colors
        # Use rebuild_viz=False for all but the last mask to avoid redundant rebuilds
        for i, mask in enumerate(propagated_masks):
            # Use the corresponding mask_id from valid_mask_ids (after filtering)
            if i < len(valid_mask_ids):
                mask_id = valid_mask_ids[i]
            else:
                mask_id = None  # Let canvas assign a new ID
            
            # Look up existing color for this mask_id
            color = video_colors.get(mask_id) if mask_id else None
            
            # Only rebuild visualization on the last mask
            rebuild_viz = (i == len(propagated_masks) - 1)
            # Use current category for propagated masks
            category = self.canvas.current_annotation_type
            self.canvas.add_mask(mask, mask_id, color, rebuild_viz=rebuild_viz, category=category)
            propagated_id = mask_id if mask_id is not None else self.canvas.next_mask_id - 1
            self.canvas.annotation_metadata[propagated_id]['source'] = 'sam2_propagated'
        
        # Register any new colors that were generated
        self._register_canvas_colors()
        
        # Update tracker with propagated instances
        if self.tracking_enabled and self.current_video_id:
            tracker = self._get_or_create_tracker(self.current_video_id)
            
            # Update or create tracks for propagated instances
            for i, mask in enumerate(propagated_masks):
                if i < len(valid_mask_ids):
                    mask_id = valid_mask_ids[i]
                    
                    # Compute bbox from mask
                    if np.any(mask > 0):
                        y_indices, x_indices = np.where(mask > 0)
                        bbox = np.array([
                            float(x_indices.min()),
                            float(y_indices.min()),
                            float(x_indices.max()),
                            float(y_indices.max())
                        ])
                    else:
                        bbox = np.array([0.0, 0.0, 1.0, 1.0])
                    
                    # Update track in tracker
                    if mask_id in tracker.active_tracks:
                        track = tracker.active_tracks[mask_id]
                        track.bbox = bbox
                        track.mask = mask
                        track.last_seen_frame = next_idx
                        track.frames_lost = 0
                        track.source_history.append('propagated')
                    else:
                        # Create new track
                        track = Track(
                            track_id=mask_id,
                            bbox=bbox,
                            mask=mask,
                            last_seen_frame=next_idx
                        )
                        track.source_history.append('propagated')
                        tracker.active_tracks[mask_id] = track
        
        # Update video next_mask_id tracking after adding masks
        if self.current_video_id:
            self.video_next_mask_id[self.current_video_id] = self.canvas.next_mask_id
        
        # Mark frame as modified so annotations will be saved
        self._mark_current_annotations_dirty()
        
        self.update_instance_list_from_canvas()
        
        # Clear annotation cache to free memory (keep last 10 frames)
        self.annotation_manager.clear_annotation_cache(keep_recent=10)
        
        # Force garbage collection after propagation
        import gc
        gc.collect()
        
        # Report memory usage if available
        try:
            import torch
            if torch.cuda.is_available():
                memory_gb = torch.cuda.memory_allocated() / 1024**3
                print(f"Post-propagation GPU memory: {memory_gb:.2f} GB")
        except:
            pass
        
        self.status_label.setText(f"✓ Propagated {len(propagated_masks)} instance(s) to frame {next_idx}")
            
    def enable_sam2_prompt_mode(self):
        """Enable SAM2 prompt mode"""
        if self.sam2:
            self.sam2_toolbar.point_btn.setChecked(True)
            self.on_tool_changed('sam2_prompt')
            self.status_label.setText("SAM2 Prompt Mode: Left click for positive points, Right click for negative")
        else:
            QMessageBox.warning(self, "SAM2 Not Available", 
                              "SAM2 model is not loaded.\n\n"
                              "Please load a SAM2 checkpoint using the SAM2 toolbar.")
            
    def start_training(self):
        """Training with Detectron2 has been removed - use YOLO training notebook"""
        QMessageBox.information(
            self, "Training Removed",
            "Detectron2 training has been removed from the GUI.\n\n"
            "Please use the YOLO training notebook instead:\n"
            "bee_annotator/training_yolo_demo.ipynb\n\n"
            "The notebook provides:\n"
            "• COCO to YOLO format conversion\n"
            "• YOLOv8 segmentation training\n"
            "• Automatic validation and best model saving\n"
            "• Easy export for deployment"
        )
        return
    
    def _run_training(self, train_coco, val_coco, config, dialog):
        """Execute training in background worker thread - DISABLED (Detectron2 removed)"""
        # This method is no longer used - see training_yolo_demo.ipynb
        pass
    
    def _on_validation_visualization_ready(self, validation_results):
        """Store validation results for viewing - DISABLED (Detectron2 removed)"""
        # This method is no longer used
        pass
        
    def _show_validation_viewer(self):
        """Show validation predictions viewer - DISABLED (Detectron2 removed)"""
        # This method is no longer used
        pass
    
    def _handle_training_completion(self, dialog, success, metrics):
        """Handle training completion - DISABLED (Detectron2 removed)"""
        # This method is no longer used
        pass
    
    def _show_inference_preview(self, image_path):
        """Show inference preview in a dialog - DISABLED (Detectron2 removed)"""
        # This method is no longer used
        pass
    
    def _on_training_completed(self, success, final_metrics):
        """Handle training completion - DISABLED (Detectron2 removed)"""
        # This method is no longer used
        pass
    
    def _on_training_error(self, error_message):
        """Handle training error - DISABLED (Detectron2 removed)"""
        # This method is no longer used
        pass

    def _validate_coco_categories_for_model(self, coco_jsons, model_type):
        """Return whether all COCO files define the selected training category."""
        missing_files = []
        for json_file in coco_jsons:
            try:
                with open(json_file, 'r') as f:
                    coco_data = json.load(f)
            except (OSError, json.JSONDecodeError):
                missing_files.append(Path(json_file).name)
                continue

            category_names = {
                category.get('name')
                for category in coco_data.get('categories', [])
                if category.get('name')
            }
            if not set(training_categories(model_type)).issubset(category_names):
                missing_files.append(Path(json_file).name)

        if not missing_files:
            return True

        shown_files = '\n'.join(missing_files[:5])
        if len(missing_files) > 5:
            shown_files += f"\n...and {len(missing_files) - 5} more"

        QMessageBox.warning(
            self,
            "Stale COCO Annotations",
            f"Some COCO files do not define the '{model_type}' category:\n\n"
            f"{shown_files}\n\n"
            "Restart the app if you have not since the latest code change, then "
            "run training again with Export COCO Annotations checked."
        )
        return False
    
    def train_yolo_model(self):
        """Train YOLO coarse detection model"""
        if not self.project_path:
            QMessageBox.warning(self, "No Project", "Please open a project first.")
            return
        
        # Check if COCO annotations exist
        train_dir = self.project_path / 'annotations/coco/train'
        val_dir = self.project_path / 'annotations/coco/val'
        train_jsons = list(train_dir.glob('*.json')) if train_dir.exists() else []
        val_jsons = list(val_dir.glob('*.json')) if val_dir.exists() else []
        
        if not train_jsons or not val_jsons:
            reply = QMessageBox.question(
                self,
                "Export Annotations",
                "COCO format annotations not found. Export annotations now?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )
            
            if reply == QMessageBox.StandardButton.Yes:
                self.export_coco_format()
                # Check again
                train_jsons = list(train_dir.glob('*.json')) if train_dir.exists() else []
                val_jsons = list(val_dir.glob('*.json')) if val_dir.exists() else []
                if not train_jsons or not val_jsons:
                    QMessageBox.warning(self, "Export Failed", "Failed to export annotations.")
                    return
            else:
                return
        
        # Get current coarse YOLO model path
        current_model_path = None
        if hasattr(self, 'yolo_toolbar') and hasattr(self.yolo_toolbar, 'model_path'):
            current_model_path = self.yolo_toolbar.model_path
        
        # Show configuration dialog
        config_dialog = TrainingConfigDialog(self, current_model_path)
        if config_dialog.exec():
            config = config_dialog.get_config()
            
            # Export COCO annotations if requested
            if config.get('export_coco', True):
                self.export_coco_format()
                # Check annotations exist after export
                train_jsons = list(train_dir.glob('*.json')) if train_dir.exists() else []
                val_jsons = list(val_dir.glob('*.json')) if val_dir.exists() else []
                if not train_jsons or not val_jsons:
                    QMessageBox.warning(self, "Export Failed", "Failed to export annotations.")
                    return

            model_type = config.get('model_type', 'bee')
            if not self._validate_coco_categories_for_model(
                train_jsons + val_jsons,
                model_type
            ):
                return
            
            # Create progress dialog
            progress_dialog = TrainingProgressDialog(self)
            
            # Create training worker
            worker = YOLOTrainingWorker(self.project_path, config)
            progress_dialog.set_worker(worker)
            
            # Start training
            worker.start()
            
            # Show progress dialog (blocks until closed)
            progress_dialog.exec()
            
            # Handle completion
            if progress_dialog.training_completed:
                # Training completed successfully
                model_path = progress_dialog.model_path
                final_metrics = progress_dialog.final_metrics
                
                # Build completion message
                msg = "Training completed successfully!\n\n"
                msg += f"Model saved to:\n{model_path}\n\n"
                
                if final_metrics:
                    msg += "Final Metrics:\n"
                    for key, value in final_metrics.items():
                        msg += f"  {key}: {value:.4f}\n"
                    msg += "\n"
                
                model_type = config.get('model_type', 'bee')
                if model_type == 'brood':
                    QMessageBox.information(self, 'Brood Training Complete', msg +
                        'Use this checkpoint in the optional Brood Model slot in Batch Video Inference.')
                    return
                target_labels = {
                    'bee': 'coarse YOLO toolbar',
                    'hive': 'Hive model slot',
                    'chamber': 'Chamber model slot',
                    'pollen': 'Pollen model slot',
                    'nectar': 'Nectar model slot',
                }
                target_label = target_labels.get(model_type, 'coarse YOLO toolbar')
                msg += f"Would you like to load the new {model_type} model into the {target_label}?"
                
                reply = QMessageBox.question(
                    self,
                    "Training Complete",
                    msg,
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                    QMessageBox.StandardButton.Yes
                )
                
                if reply == QMessageBox.StandardButton.Yes:
                    # Load the new model
                    from pathlib import Path
                    if Path(model_path).exists():
                        if model_type == 'hive':
                            self.hive_chamber_toolbar._load_hive_checkpoint_from_path(model_path, show_dialogs=False)
                            self.hive_chamber_toolbar.show()
                            self.hive_chamber_toolbar_action.setChecked(True)
                        elif model_type == 'chamber':
                            self.hive_chamber_toolbar._load_chamber_checkpoint_from_path(model_path, show_dialogs=False)
                            self.hive_chamber_toolbar.show()
                            self.hive_chamber_toolbar_action.setChecked(True)
                        elif model_type == 'pollen':
                            self.hive_chamber_toolbar._load_pollen_checkpoint_from_path(model_path, show_dialogs=False)
                            self.hive_chamber_toolbar.show()
                            self.hive_chamber_toolbar_action.setChecked(True)
                        elif model_type == 'nectar':
                            if not self.hive_chamber_toolbar._load_nectar_checkpoint_from_path(model_path, show_dialogs=True):
                                return
                            self.hive_chamber_toolbar.show()
                            self.hive_chamber_toolbar_action.setChecked(True)
                        else:
                            self.yolo_toolbar._load_checkpoint_from_path(model_path, show_dialogs=False)
                            self.yolo_toolbar.show()
                            self.yolo_toolbar_action.setChecked(True)
                        self.status_label.setText(f"✓ Loaded trained {model_type} model: {Path(model_path).name}")
                    else:
                        QMessageBox.warning(self, "Model Not Found", f"Model file not found: {model_path}")
            elif progress_dialog.training_failed:
                # Training failed
                QMessageBox.critical(
                    self,
                    "Training Failed",
                    "Training failed. Check the training log for details.\n\n"
                    "The backup model has been preserved."
                )
            # else: training was stopped/cancelled, no action needed
    
    def train_yolo_model_instance_focused(self):
        """Train YOLO instance-focused refinement model (single-instance crops)"""
        if not self.project_path:
            QMessageBox.warning(self, "No Project", "Please open a project first.")
            return
        
        # Check if COCO annotations exist
        train_dir = self.project_path / 'annotations/coco/train'
        val_dir = self.project_path / 'annotations/coco/val'
        train_jsons = list(train_dir.glob('*.json')) if train_dir.exists() else []
        val_jsons = list(val_dir.glob('*.json')) if val_dir.exists() else []
        
        if not train_jsons or not val_jsons:
            reply = QMessageBox.question(
                self,
                "Export Annotations",
                "COCO format annotations not found. Export annotations now?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )
            
            if reply == QMessageBox.StandardButton.Yes:
                self.export_coco_format()
                # Check again
                train_jsons = list(train_dir.glob('*.json')) if train_dir.exists() else []
                val_jsons = list(val_dir.glob('*.json')) if val_dir.exists() else []
                if not train_jsons or not val_jsons:
                    QMessageBox.warning(self, "Export Failed", "Failed to export annotations.")
                    return
            else:
                return
        
        # Get current fine YOLO model path
        current_model_path = None
        if hasattr(self, 'yolo_refine_toolbar') and hasattr(self.yolo_refine_toolbar, 'model_path'):
            current_model_path = self.yolo_refine_toolbar.model_path
        
        # Show configuration dialog with instance-focused specific parameters
        config_dialog = TrainingConfigDialog(self, current_model_path, instance_focused=True)
        if config_dialog.exec():
            config = config_dialog.get_config()
            
            # Export COCO annotations if requested
            if config.get('export_coco', True):
                self.export_coco_format()
                # Check annotations exist after export
                train_jsons = list(train_dir.glob('*.json')) if train_dir.exists() else []
                val_jsons = list(val_dir.glob('*.json')) if val_dir.exists() else []
                if not train_jsons or not val_jsons:
                    QMessageBox.warning(self, "Export Failed", "Failed to export annotations.")
                    return
            
            # Create progress dialog for instance-focused training (uses IoU metrics)
            progress_dialog = TrainingProgressDialog(self, instance_focused=True)
            
            # Create training worker for instance-focused
            worker = YOLOTrainingWorkerInstanceFocused(self.project_path, config)
            progress_dialog.set_worker(worker)
            
            # Start training
            worker.start()
            
            # Show progress dialog (blocks until closed)
            progress_dialog.exec()
            
            # Handle completion
            if progress_dialog.training_completed:
                # Training completed successfully
                model_path = progress_dialog.model_path
                final_metrics = progress_dialog.final_metrics
                
                # Build completion message
                msg = "Instance-focused training completed successfully!\n\n"
                msg += f"Model saved to:\n{model_path}\n\n"
                
                if final_metrics:
                    msg += "Final Metrics:\n"
                    for key, value in final_metrics.items():
                        msg += f"  {key}: {value:.4f}\n"
                    msg += "\n"
                
                msg += "Would you like to load the new model into the fine YOLO toolbar?"
                
                reply = QMessageBox.question(
                    self,
                    "Training Complete",
                    msg,
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                    QMessageBox.StandardButton.Yes
                )
                
                if reply == QMessageBox.StandardButton.Yes:
                    # Load the new model
                    from pathlib import Path
                    if Path(model_path).exists():
                        self.yolo_refine_toolbar._load_checkpoint_from_path(model_path, show_dialogs=False)
                        self.status_label.setText(f"✓ Loaded trained instance-focused model: {Path(model_path).name}")
                    else:
                        QMessageBox.warning(self, "Model Not Found", f"Model file not found: {model_path}")
            elif progress_dialog.training_failed:
                # Training failed
                QMessageBox.critical(
                    self,
                    "Training Failed",
                    "Instance-focused training failed. Check the training log for details.\n\n"
                    "The backup model has been preserved."
                )
            # else: training was stopped/cancelled, no action needed
    def run_inference(self):
        """Create a tracked single-video MP4 visualization."""
        config_dialog = TrackingVisualizationConfigDialog(self)
        if not config_dialog.exec():
            return

        progress_dialog = BatchVideoInferenceProgressDialog(self)
        progress_dialog.setWindowTitle("Tracking Visualization Video")
        progress_dialog.current_video_label.setText("Single video")

        worker = TrackingVisualizationWorker(config_dialog.config)

        worker.status_updated.connect(progress_dialog.update_status)
        worker.progress_updated.connect(progress_dialog.update_progress)
        worker.log_message.connect(progress_dialog.append_log)
        worker.visualization_complete.connect(
            lambda output_path: self._on_tracking_visualization_complete(progress_dialog, output_path)
        )
        worker.visualization_stopped.connect(progress_dialog.processing_stopped)
        worker.visualization_failed.connect(progress_dialog.processing_failed)
        progress_dialog.stop_btn.clicked.connect(worker.stop)

        worker.start()
        progress_dialog.exec()

        if worker.isRunning():
            worker.stop()
            worker.wait(5000)

    def _on_tracking_visualization_complete(self, progress_dialog, output_path: str):
        """Handle completion of single-video tracking visualization."""
        progress_dialog.processing_complete()
        self.status_label.setText(f"✓ Tracking visualization saved: {output_path}")
        
    def save_annotations(self):
        """Save current annotation source files without regenerating COCO datasets."""
        if not self.project_path:
            QMessageBox.warning(self, "No Project", "Please create or open a project first")
            return

        try:
            save_summary = self._save_annotation_sources(refresh_next_id=True)

            if save_summary.get('saved_any'):
                saved_parts = ", ".join(save_summary.get('saved_files', []))
                self.status_label.setText(
                    f"✓ Saved {saved_parts} annotations for "
                    f"{save_summary['video_id']} frame {save_summary['frame_idx']} "
                    f"(COCO updates on Export/Train)"
                )
            elif self.coco_export_dirty:
                self.status_label.setText(
                    "✓ Annotation files already saved; COCO updates on Export/Train"
                )
            else:
                self.status_label.setText("✓ No unsaved annotation changes")

        except Exception as e:
            import traceback
            error_msg = f"Failed to save annotations:\n\n{str(e)}\n\n{traceback.format_exc()}"
            self.status_label.setText("✗ Save failed")
            QMessageBox.critical(self, "Save Error", error_msg)
    
    def train_yolo_bbox_model(self):
        """Train YOLO bounding box detection model"""
        if not self.project_path:
            QMessageBox.warning(self, "No Project", "Please open a project first.")
            return
        
        # Check if COCO annotations exist
        train_dir = self.project_path / 'annotations/coco/train'
        val_dir = self.project_path / 'annotations/coco/val'
        train_jsons = list(train_dir.glob('*.json')) if train_dir.exists() else []
        val_jsons = list(val_dir.glob('*.json')) if val_dir.exists() else []
        
        if not train_jsons or not val_jsons:
            reply = QMessageBox.question(
                self,
                "Export Annotations",
                "COCO format annotations not found. Export annotations now?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )
            
            if reply == QMessageBox.StandardButton.Yes:
                self.export_coco_format()
                # Check again
                train_jsons = list(train_dir.glob('*.json')) if train_dir.exists() else []
                val_jsons = list(val_dir.glob('*.json')) if val_dir.exists() else []
                if not train_jsons or not val_jsons:
                    QMessageBox.warning(self, "Export Failed", "Failed to export annotations.")
                    return
            else:
                return
        
        # Get current bbox YOLO model path
        current_model_path = None
        if hasattr(self, 'yolo_bbox_toolbar') and hasattr(self.yolo_bbox_toolbar, 'checkpoint_path'):
            current_model_path = self.yolo_bbox_toolbar.checkpoint_path
        
        # Show configuration dialog
        config_dialog = TrainingConfigDialog(self, current_model_path)
        config_dialog.model_type_combo.removeItem(
            config_dialog.model_type_combo.findText(BROOD_MODEL_LABEL))
        config_dialog.model_type_combo.removeItem(
            config_dialog.model_type_combo.findText('Nectar source'))
        if config_dialog.exec():
            config = config_dialog.get_config()
            
            # Export COCO annotations if requested
            if config.get('export_coco', True):
                self.export_coco_format()
                # Check annotations exist after export
                train_jsons = list(train_dir.glob('*.json')) if train_dir.exists() else []
                val_jsons = list(val_dir.glob('*.json')) if val_dir.exists() else []
                if not train_jsons or not val_jsons:
                    QMessageBox.warning(self, "Export Failed", "Failed to export annotations.")
                    return
            
            # Create progress dialog
            progress_dialog = TrainingProgressDialog(self)
            
            # Create training worker (bbox specific)
            worker = YOLOTrainingWorkerBBox(self.project_path, config)
            progress_dialog.set_worker(worker)
            
            # Start training
            worker.start()
            
            # Show progress dialog (blocks until closed)
            progress_dialog.exec()
            
            # Handle completion
            if progress_dialog.training_completed:
                # Training completed successfully
                model_path = progress_dialog.model_path
                final_metrics = progress_dialog.final_metrics
                
                # Build completion message
                msg = "BBox detection training completed successfully!\n\n"
                msg += f"Model saved to:\n{model_path}\n\n"
                
                if final_metrics:
                    msg += "Final Metrics:\n"
                    for key, value in final_metrics.items():
                        msg += f"  {key}: {value:.4f}\n"
                    msg += "\n"
                
                msg += "Would you like to load the new model into the YOLO BBox toolbar?"
                
                reply = QMessageBox.question(
                    self,
                    "Training Complete",
                    msg,
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                    QMessageBox.StandardButton.Yes
                )
                
                if reply == QMessageBox.StandardButton.Yes:
                    # Load the new model
                    from pathlib import Path
                    if Path(model_path).exists():
                        self.yolo_bbox_toolbar._load_checkpoint_from_path(model_path, show_dialogs=False)
                        self.status_label.setText(f"✓ Loaded trained bbox model: {Path(model_path).name}")
                    else:
                        QMessageBox.warning(self, "Model Not Found", f"Model file not found: {model_path}")
            elif progress_dialog.training_failed:
                # Training failed
                QMessageBox.critical(
                    self,
                    "Training Failed",
                    "BBox training failed. Check the training log for details.\n\n"
                    "The backup model has been preserved."
                )
            # else: training was stopped/cancelled, no action needed
    
    def analyze_frame_level_validation(self):
        """Analyze predictions vs ground truth for validation frames"""
        if not self.project_path:
            QMessageBox.warning(self, "No Project", "Please open a project first.")
            return
        
        # Check if validation frames exist
        val_videos = self.project_manager.get_videos_by_split('val')
        if not val_videos:
            QMessageBox.warning(
                self,
                "No Validation Videos",
                "No videos found in validation split.\n\n"
                "Please add videos to the 'val' split first."
            )
            return
        
        # Ask if user wants to re-export COCO annotations
        reply = QMessageBox.question(
            self,
            "Export COCO Annotations?",
            "Frame-level validation uses COCO export format.\n\n"
            "Do you want to export/update COCO annotations before running validation?\n\n"
            "(Recommended if you have made recent changes to annotations)",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.Yes
        )
        
        if reply == QMessageBox.StandardButton.Yes:
            # Export COCO annotations
            self.export_coco_format()
        
        # Show configuration dialog
        config_dialog = FrameLevelValidationConfigDialog(self)
        if config_dialog.exec():
            config = config_dialog.get_config()
            
            # Create progress dialog
            progress_dialog = FrameLevelValidationProgressDialog(self)
            
            # Create worker
            worker = FrameLevelValidationWorker(self, config)
            
            # Connect signals
            worker.status_updated.connect(progress_dialog.update_status)
            worker.stats_updated.connect(progress_dialog.update_stats)
            worker.match_stats_updated.connect(progress_dialog.update_match_stats)
            worker.time_remaining_updated.connect(progress_dialog.update_time_remaining)
            worker.log_message.connect(progress_dialog.log)
            worker.analysis_complete.connect(progress_dialog.on_complete)
            worker.analysis_failed.connect(progress_dialog.on_error)
            
            # Connect cancel button
            progress_dialog.cancel_btn.clicked.connect(worker.stop)
            
            # Start worker
            worker.start()
            
            # Show progress dialog (blocks until closed)
            progress_dialog.exec()
            
            # Wait for worker to finish
            if worker.isRunning():
                worker.stop()
                worker.wait(5000)  # Wait up to 5 seconds
    
    def batch_image_inference(self):
        """Run inference on arbitrary PNG images from folders"""
        if not self.project_path:
            QMessageBox.warning(self, "No Project", "Please open a project first.")
            return
        
        # Show configuration dialog
        config_dialog = BatchInferenceConfigDialog(self)
        if config_dialog.exec():
            config = config_dialog.get_config()
            
            # Create progress dialog
            progress_dialog = BatchInferenceProgressDialog(self)
            
            # Create worker
            worker = BatchInferenceWorker(config, self.project_path)
            
            # Connect signals
            worker.status_updated.connect(progress_dialog.update_status)
            worker.progress_updated.connect(progress_dialog.update_progress)
            worker.time_remaining_updated.connect(progress_dialog.update_time_remaining)
            worker.stats_updated.connect(progress_dialog.update_stats)
            worker.log_message.connect(progress_dialog.append_log)
            worker.finished.connect(progress_dialog.on_complete)
            
            # Connect cancel button
            progress_dialog.cancel_btn.clicked.connect(lambda: setattr(worker, 'should_stop', True))
            
            # Start worker
            worker.start()
            
            # Show progress dialog (blocks until closed)
            progress_dialog.exec()
            
            # Wait for worker to finish
            if worker.isRunning():
                worker.should_stop = True
                worker.wait(5000)  # Wait up to 5 seconds
    
    def batch_video_inference(self):
        """Run batch video inference with tracking and ArUco detection"""
        # Show configuration dialog
        config_dialog = BatchVideoInferenceConfigDialog(self)
        if config_dialog.exec():
            config = config_dialog.config
            
            # Create progress dialog
            progress_dialog = BatchVideoInferenceProgressDialog(self)
            
            # Create worker
            worker = BatchVideoInferenceWorker(config)
            
            # Connect signals
            worker.status_updated.connect(progress_dialog.update_status)
            worker.progress_updated.connect(progress_dialog.update_progress)
            worker.log_message.connect(progress_dialog.append_log)
            worker.inference_complete.connect(progress_dialog.processing_complete)
            worker.inference_stopped.connect(progress_dialog.processing_stopped)
            worker.inference_failed.connect(progress_dialog.processing_failed)
            
            # Connect stop button
            progress_dialog.stop_btn.clicked.connect(worker.stop)
            
            # Start worker
            worker.start()
            
            # Show progress dialog (blocks until closed)
            progress_dialog.exec()
            
            # Wait for worker to finish
            if worker.isRunning():
                worker.stop()
                worker.wait(5000)  # Wait up to 5 seconds
    
    def validate_tracking_algorithms(self):
        """Open tracking algorithm validation dialog"""
        if not self.project_path:
            QMessageBox.warning(self, "No Project", "Please open a project first.")
            return
        
        if not self.tracking_sequence_manager:
            QMessageBox.warning(self, "Error", "Tracking sequence manager not initialized.")
            return
        
        # Get enabled sequences
        sequences = self.tracking_sequence_manager.get_enabled_sequences()
        
        if not sequences:
            reply = QMessageBox.question(
                self,
                "No Sequences",
                "No tracking sequences found.\n\n"
                "Would you like to create a tracking sequence now?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )
            if reply == QMessageBox.StandardButton.Yes:
                # Open tracking sequences panel
                pass
            return
        
        # Show configuration dialog
        config_dialog = TrackingValidationConfigDialog(sequences, self)
        if config_dialog.exec() == QDialog.DialogCode.Accepted:
            config = config_dialog.config
            
            # Create progress dialog
            progress_dialog = TrackingValidationProgressDialog(self)
            
            # Create worker
            worker = TrackingValidationWorker(self, config)
            
            # Connect signals
            worker.status_updated.connect(progress_dialog.update_status)
            worker.progress_updated.connect(progress_dialog.update_progress)
            worker.log_message.connect(progress_dialog.append_log)
            worker.metrics_updated.connect(progress_dialog.update_metrics)
            worker.validation_complete.connect(self.on_tracking_validation_complete)
            worker.validation_complete.connect(progress_dialog.validation_finished)
            worker.validation_failed.connect(progress_dialog.validation_error)
            
            # Connect stop button
            progress_dialog.stop_requested.connect(worker.stop)
            
            # Start worker
            worker.start()
            
            # Show progress dialog (blocks until closed)
            progress_dialog.exec()
            
            # Ensure worker finishes
            if worker.isRunning():
                worker.stop()
                worker.wait(5000)
    
    def on_tracking_validation_complete(self, results_path):
        """Handle tracking validation completion"""
        QMessageBox.information(
            self,
            "Validation Complete",
            f"Tracking validation completed successfully!\n\n"
            f"Results saved to:\n{results_path}"
        )
    
    def on_sequence_created(self, sequence_id):
        """Handle tracking sequence creation"""
        self.status_label.setText(f"Tracking sequence created: {sequence_id}")
    
    def on_sequence_deleted(self, sequence_id):
        """Handle tracking sequence deletion"""
        self.status_label.setText(f"Tracking sequence deleted: {sequence_id}")
    
    def on_sequence_selected(self, sequence_id, frame_idx):
        """Navigate to tracking sequence"""
        # Find frame in frames list
        for i, (frame_path, video_id) in enumerate(zip(self.frames, self.frame_video_ids)):
            frame_num = int(Path(frame_path).stem.split('_')[1])
            if frame_num == frame_idx:
                self.load_frame(i)
                self.frame_list.setCurrentRow(i)
                break
    
    def on_validation_requested(self, sequence_ids):
        """Handle validation request from tracking sequences panel"""
        if not sequence_ids:
            return
        
        # Get sequence objects
        sequences = []
        for seq_id in sequence_ids:
            seq = self.tracking_sequence_manager.get_sequence(seq_id)
            if seq:
                sequences.append(seq)
        
        if not sequences:
            return
        
        # Show configuration dialog
        config_dialog = TrackingValidationConfigDialog(sequences, self)
        if config_dialog.exec() == QDialog.DialogCode.Accepted:
            config = config_dialog.config
            
            # Create progress dialog
            progress_dialog = TrackingValidationProgressDialog(self)
            
            # Create worker
            worker = TrackingValidationWorker(self, config)
            
            # Connect signals
            worker.status_updated.connect(progress_dialog.update_status)
            worker.progress_updated.connect(progress_dialog.update_progress)
            worker.log_message.connect(progress_dialog.append_log)
            worker.metrics_updated.connect(progress_dialog.update_metrics)
            worker.validation_complete.connect(self.on_tracking_validation_complete)
            worker.validation_complete.connect(progress_dialog.validation_finished)
            worker.validation_failed.connect(progress_dialog.validation_error)
            
            # Connect stop button
            progress_dialog.stop_requested.connect(worker.stop)
            
            # Start worker
            worker.start()
            
            # Show progress dialog (blocks until closed)
            progress_dialog.exec()
            
            # Ensure worker finishes
            if worker.isRunning():
                worker.stop()
                worker.wait(5000)
    
    def export_coco_format(self):
        """Export annotations in COCO format (train, val, and test splits only - inference split is excluded)"""
        if not self.project_path:
            QMessageBox.warning(self, "No Project", "Please create or open a project first")
            return
        
        try:
            # Save dirty source annotations first because COCO export reads from disk.
            if self.current_frame_idx is not None:
                self._save_annotation_sources(refresh_next_id=True)
            
            # Get train, validation, and test videos (inference videos are excluded from export)
            train_videos = self.project_manager.get_videos_by_split('train')
            val_videos = self.project_manager.get_videos_by_split('val')
            test_videos = self.project_manager.get_videos_by_split('test')
            
            if not train_videos and not val_videos and not test_videos:
                QMessageBox.warning(
                    self,
                    "No Videos",
                    "No videos found in train, validation, or test splits.\n\n"
                    "Please import videos and assign them to splits first."
                )
                return
            
            # Calculate total videos to process
            total_videos = len(train_videos) + len(val_videos) + len(test_videos)
            
            # Create progress dialog
            from PyQt6.QtWidgets import QProgressDialog
            from PyQt6.QtCore import Qt
            progress = QProgressDialog(
                "Exporting COCO annotations...",
                "Cancel",
                0,
                total_videos,
                self
            )
            progress.setWindowTitle("Export COCO Format")
            progress.setWindowModality(Qt.WindowModality.WindowModal)
            progress.setMinimumDuration(0)
            progress.setValue(0)
            
            # Track overall progress
            videos_processed = 0
            
            def update_progress(current, total, video_name):
                """Update progress dialog"""
                nonlocal videos_processed
                if current > 0:
                    videos_processed += 1
                progress.setValue(videos_processed)
                progress.setLabelText(f"Exporting {video_name}...\n({videos_processed}/{total_videos} videos)")
                QApplication.processEvents()
            
            def check_cancelled():
                """Check if user cancelled"""
                return progress.wasCanceled()
            
            coco_files_generated = []
            
            # Export train split if videos exist
            if train_videos:
                train_paths = export_coco_per_video(
                    self.project_path,
                    train_videos,
                    'train',
                    class_names=self.annotation_manager.class_names,
                    image_width=self.annotation_manager.image_width,
                    image_height=self.annotation_manager.image_height,
                    progress_callback=update_progress,
                    cancel_check=check_cancelled
                )
                if train_paths is None:  # Cancelled
                    self.status_label.setText("Export cancelled")
                    return
                coco_files_generated.append(f"Training: {len(train_paths)} videos")
            
            # Export val split if videos exist
            if val_videos:
                val_paths = export_coco_per_video(
                    self.project_path,
                    val_videos,
                    'val',
                    class_names=self.annotation_manager.class_names,
                    image_width=self.annotation_manager.image_width,
                    image_height=self.annotation_manager.image_height,
                    progress_callback=update_progress,
                    cancel_check=check_cancelled
                )
                if val_paths is None:  # Cancelled
                    self.status_label.setText("Export cancelled")
                    return
                coco_files_generated.append(f"Validation: {len(val_paths)} videos")
            
            # Export test split if videos exist
            if test_videos:
                test_paths = export_coco_per_video(
                    self.project_path,
                    test_videos,
                    'test',
                    class_names=self.annotation_manager.class_names,
                    image_width=self.annotation_manager.image_width,
                    image_height=self.annotation_manager.image_height,
                    progress_callback=update_progress,
                    cancel_check=check_cancelled
                )
                if test_paths is None:  # Cancelled
                    self.status_label.setText("Export cancelled")
                    return
                coco_files_generated.append(f"Test: {len(test_paths)} videos")
            
            # Close progress dialog
            progress.setValue(total_videos)
            
            # Show success message
            files_list = '\n'.join(coco_files_generated)
            QMessageBox.information(
                self,
                "Export Successful",
                f"COCO format annotations exported successfully!\n\n"
                f"{files_list}\n\n"
                f"Location: {self.project_path / 'annotations/coco'}"
            )
            self.coco_export_dirty = False
            self.status_label.setText(f"✓ Exported COCO format ({files_list})")
            
        except Exception as e:
            error_msg = f"Failed to export COCO format:\n\n{str(e)}"
            self.status_label.setText("✗ Export failed")
            QMessageBox.critical(self, "Export Error", error_msg)
            import traceback
            traceback.print_exc()
    
    def export_annotations_as_video(self):
        """Export annotations as a video file with instance IDs and green outlines"""
        if not self.project_path:
            QMessageBox.warning(self, "No Project", "Please create or open a project first")
            return
        
        if not self.frames:
            QMessageBox.warning(self, "No Video", "No video frames loaded")
            return
        
        if not self.current_video_id:
            QMessageBox.warning(self, "No Video", "No video is currently loaded")
            return
        
        try:
            # Collect all frames from CURRENT VIDEO that have annotations
            frames_with_annotations = []
            for idx in range(len(self.frames)):
                # Only process frames from the current video
                if idx < len(self.frame_video_ids) and self.frame_video_ids[idx] != self.current_video_id:
                    continue
                
                # Check if this frame has annotations - first check cache
                annotations = self.annotation_manager.get_frame_annotations(idx, video_id=self.current_video_id)
                
                # If not in cache, load from disk
                if not annotations and self.project_path and self.current_video_id:
                    frame_idx_in_video = self._get_frame_idx_in_video(idx)
                    annotations = self.annotation_manager.load_frame_annotations(
                        self.project_path, self.current_video_id, frame_idx_in_video
                    )
                
                if annotations and len(annotations) > 0:
                    # Check if any annotation has a mask
                    has_mask = any('mask' in ann or 'mask_rle' in ann for ann in annotations)
                    if has_mask:
                        frames_with_annotations.append(idx)
            
            print(f"Found {len(frames_with_annotations)} frames with mask annotations out of {len(self.frames)} total frames")
            
            if not frames_with_annotations:
                QMessageBox.warning(
                    self,
                    "No Annotations",
                    f"No frames with mask annotations found for video '{self.current_video_id}'."
                )
                return
            
            # Ask user for output video path (use video name instead of project name)
            default_name = f"{self.current_video_id}_annotations.mp4"
            output_path, _ = QFileDialog.getSaveFileName(
                self,
                "Export Annotations Video",
                str(self.project_path / default_name),
                "Video Files (*.mp4 *.avi);;All Files (*)"
            )
            
            if not output_path:
                return  # User cancelled
            
            output_path = Path(output_path)
            
            # Create progress dialog
            from PyQt6.QtWidgets import QProgressDialog
            progress = QProgressDialog(
                f"Exporting {len(frames_with_annotations)} annotated frames...",
                "Cancel",
                0,
                len(frames_with_annotations),
                self
            )
            progress.setWindowTitle("Export Annotations Video")
            progress.setWindowModality(Qt.WindowModality.WindowModal)
            progress.setMinimumDuration(0)
            progress.setValue(0)
            
            # Get first frame to determine video properties
            first_frame_path = self.frames[frames_with_annotations[0]]
            first_img = cv2.imread(str(first_frame_path))
            if first_img is None:
                QMessageBox.critical(self, "Error", f"Failed to read first frame: {first_frame_path}")
                return
            
            height, width = first_img.shape[:2]
            fps = 10.0  # Default FPS for annotation video
            
            # Create video writer
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # MP4 codec
            video_writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
            
            if not video_writer.isOpened():
                QMessageBox.critical(self, "Error", "Failed to create video writer")
                return
            
            try:
                # Process each annotated frame
                for i, frame_idx in enumerate(frames_with_annotations):
                    # Check if user cancelled
                    if progress.wasCanceled():
                        self.status_label.setText("Export cancelled")
                        break
                    
                    # Update progress
                    progress.setValue(i)
                    progress.setLabelText(
                        f"Processing frame {frame_idx}...\n"
                        f"({i + 1} / {len(frames_with_annotations)})"
                    )
                    QApplication.processEvents()
                    
                    # Load frame image
                    frame_path = self.frames[frame_idx]
                    img = cv2.imread(str(frame_path))
                    if img is None:
                        print(f"Warning: Failed to read frame {frame_idx}: {frame_path}")
                        continue
                    
                    # Get annotations for this frame - first check cache
                    annotations = self.annotation_manager.get_frame_annotations(frame_idx, video_id=self.current_video_id)
                    
                    # If not in cache, load from disk
                    if not annotations and self.project_path and self.current_video_id:
                        frame_idx_in_video = self._get_frame_idx_in_video(frame_idx)
                        annotations = self.annotation_manager.load_frame_annotations(
                            self.project_path, self.current_video_id, frame_idx_in_video
                        )
                    
                    # Create overlay for semi-transparent masks
                    overlay = img.copy()
                    
                    # Get or initialize color mapping for this video
                    if self.current_video_id not in self.video_mask_colors:
                        self.video_mask_colors[self.current_video_id] = {}
                    
                    # Draw each annotation
                    for ann in annotations:
                        if 'mask' not in ann and 'mask_rle' not in ann:
                            continue
                        
                        # Get mask
                        if 'mask' in ann:
                            mask = ann['mask']
                        else:
                            # Decode RLE mask
                            from core.annotation import AnnotationManager
                            mask = AnnotationManager._decode_rle(
                                ann['mask_rle'],
                                ann.get('height', height),
                                ann.get('width', width)
                            )
                        
                        # Get instance ID
                        instance_id = ann.get('instance_id', ann.get('mask_id', 0))
                        
                        # Get or generate consistent color for this instance
                        if 'color' in ann:
                            # Use color from annotation
                            color = ann['color']
                        elif instance_id in self.video_mask_colors[self.current_video_id]:
                            # Use previously assigned color
                            color = self.video_mask_colors[self.current_video_id][instance_id]
                        else:
                            # Generate consistent color based on instance ID
                            # Use a pseudo-random but deterministic color based on ID
                            np.random.seed(instance_id)
                            color = tuple(np.random.randint(50, 255, 3).tolist())
                            self.video_mask_colors[self.current_video_id][instance_id] = color
                            np.random.seed()  # Reset seed
                        
                        color_bgr = (int(color[2]), int(color[1]), int(color[0]))  # RGB to BGR
                        
                        # Draw semi-transparent mask fill with instance-specific color
                        overlay[mask > 0] = color_bgr
                        
                        # Find contours for outline
                        mask_uint8 = (mask > 0).astype(np.uint8) * 255
                        contours, _ = cv2.findContours(
                            mask_uint8,
                            cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE
                        )
                        
                        # Draw green outline
                        green = (0, 255, 0)  # Light green in BGR
                        cv2.drawContours(img, contours, -1, green, 2)
                        
                        # Calculate centroid for text position
                        if len(contours) > 0:
                            # Use largest contour
                            largest_contour = max(contours, key=cv2.contourArea)
                            M = cv2.moments(largest_contour)
                            if M['m00'] != 0:
                                cx = int(M['m10'] / M['m00'])
                                cy = int(M['m01'] / M['m00'])
                            else:
                                # Fallback to mask center
                                coords = np.where(mask > 0)
                                if len(coords[0]) > 0:
                                    cy = int(coords[0].mean())
                                    cx = int(coords[1].mean())
                                else:
                                    cx, cy = 0, 0
                            
                            # Draw instance ID text with background
                            text = str(instance_id)
                            font = cv2.FONT_HERSHEY_SIMPLEX
                            font_scale = 1.0
                            thickness = 2
                            
                            # Get text size for background rectangle
                            (text_width, text_height), baseline = cv2.getTextSize(
                                text, font, font_scale, thickness
                            )
                            
                            # Draw background rectangle (dark with transparency)
                            text_x = cx - text_width // 2
                            text_y = cy + text_height // 2
                            
                            cv2.rectangle(
                                img,
                                (text_x - 5, text_y - text_height - 5),
                                (text_x + text_width + 5, text_y + baseline + 5),
                                (0, 0, 0),
                                -1  # Filled
                            )
                            
                            # Draw text in white
                            cv2.putText(
                                img,
                                text,
                                (text_x, text_y),
                                font,
                                font_scale,
                                (255, 255, 255),  # White
                                thickness,
                                cv2.LINE_AA
                            )
                    
                    # Blend overlay with original image for semi-transparency (30% opacity)
                    alpha = 0.3
                    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
                    
                    # Write frame to video
                    video_writer.write(img)
                
                # Update final progress
                progress.setValue(len(frames_with_annotations))
                
            finally:
                # Release video writer
                video_writer.release()
            
            # Show completion message
            if not progress.wasCanceled():
                QMessageBox.information(
                    self,
                    "Export Complete",
                    f"Annotations video exported successfully!\n\n"
                    f"Video: {self.current_video_id}\n"
                    f"Frames processed: {len(frames_with_annotations)}\n"
                    f"Output: {output_path}\n\n"
                    f"Video properties:\n"
                    f"  Resolution: {width}x{height}\n"
                    f"  FPS: {fps}"
                )
                self.status_label.setText(f"✓ Exported video: {len(frames_with_annotations)} frames from {self.current_video_id}")
            
        except Exception as e:
            error_msg = f"Failed to export annotations video:\n\n{str(e)}"
            self.status_label.setText("✗ Video export failed")
            QMessageBox.critical(self, "Export Error", error_msg)
            import traceback
            traceback.print_exc()
    
    def export_visualize_all_annotations(self):
        """Export/visualize all annotations organized by type (hive, chamber, bee)"""
        if not self.project_path:
            QMessageBox.warning(self, "No Project", "Please create or open a project first")
            return
        
        # Prompt user for output directory
        output_dir = QFileDialog.getExistingDirectory(
            self, 
            "Select output directory for annotation visualizations",
            str(self.project_path)
        )
        
        if not output_dir:
            return  # User cancelled
        
        output_dir = Path(output_dir)
        
        try:
            # Create subdirectories
            hive_dir = output_dir / "hive_annotations"
            chamber_dir = output_dir / "chamber_annotations"
            bee_dir = output_dir / "bee_annotations"
            
            hive_dir.mkdir(parents=True, exist_ok=True)
            chamber_dir.mkdir(parents=True, exist_ok=True)
            bee_dir.mkdir(parents=True, exist_ok=True)
            
            # Create progress dialog
            progress = QProgressDialog(
                "Exporting annotation visualizations...",
                "Cancel",
                0,
                100,
                self
            )
            progress.setWindowTitle("Export All Annotations")
            progress.setWindowModality(Qt.WindowModality.WindowModal)
            progress.setMinimumDuration(0)
            progress.setValue(0)
            
            stats = {
                'hive': 0,
                'chamber': 0,
                'bee': 0
            }
            
            # Get all videos from project
            frames_dir = self.project_path / 'frames'
            if not frames_dir.exists():
                QMessageBox.warning(self, "No Frames", "No frames directory found in project")
                return
            
            video_dirs = [d for d in frames_dir.iterdir() if d.is_dir()]
            
            if not video_dirs:
                QMessageBox.warning(self, "No Videos", "No video frames found in project")
                return
            
            # Process hive annotations - pick one random frame from each video with hive annotations
            progress.setLabelText("Processing hive annotations...")
            progress.setValue(10)
            QApplication.processEvents()
            
            for video_dir in video_dirs:
                if progress.wasCanceled():
                    break
                    
                video_id = video_dir.name

                if 'hive' in frame_categories(self.annotation_manager.project_info):
                    for frame_path in sorted(video_dir.glob('frame_*.jpg')):
                        if progress.wasCanceled():
                            break
                        frame_idx = int(frame_path.stem.split('_')[1])
                        frame_anns = self.annotation_manager.load_frame_annotations(
                            self.project_path, video_id, frame_idx
                        ) or []
                        hive_anns = [ann for ann in frame_anns if ann.get('category') == 'hive']
                        if hive_anns:
                            img = cv2.imread(str(frame_path))
                            if img is not None:
                                img = self._draw_annotations_on_image(img, hive_anns, video_id)
                                cv2.imwrite(str(hive_dir / f'{video_id}_{frame_path.stem}_hive.jpg'), img)
                                stats['hive'] += 1
                    continue

                # Load video-level annotations (hive and chamber)
                video_anns, _ = self.annotation_manager.load_video_annotations(
                    self.project_path, video_id
                )
                
                # Check if there are hive annotations
                hive_anns = [ann for ann in video_anns if ann.get('category', '') == 'hive']
                
                if hive_anns:
                    # Get a random frame from this video
                    frame_files = sorted(video_dir.glob('frame_*.jpg'))
                    if frame_files:
                        import random
                        random_frame = random.choice(frame_files)
                        
                        # Load and visualize
                        img = cv2.imread(str(random_frame))
                        if img is not None:
                            img = self._draw_annotations_on_image(img, hive_anns, video_id)
                            output_path = hive_dir / f"{video_id}_hive.jpg"
                            cv2.imwrite(str(output_path), img)
                            stats['hive'] += 1
            
            # Process chamber annotations - one random frame per video
            progress.setLabelText("Processing chamber annotations...")
            progress.setValue(30)
            QApplication.processEvents()
            
            for video_dir in video_dirs:
                if progress.wasCanceled():
                    break
                    
                video_id = video_dir.name
                
                # Load video-level annotations
                video_anns, _ = self.annotation_manager.load_video_annotations(
                    self.project_path, video_id
                )
                
                # Check if there are chamber annotations
                chamber_anns = [ann for ann in video_anns if ann.get('category', '') == 'chamber']
                
                if chamber_anns:
                    # Get a random frame from this video
                    frame_files = sorted(video_dir.glob('frame_*.jpg'))
                    if frame_files:
                        import random
                        random_frame = random.choice(frame_files)
                        
                        # Load and visualize
                        img = cv2.imread(str(random_frame))
                        if img is not None:
                            img = self._draw_annotations_on_image(img, chamber_anns, video_id)
                            output_path = chamber_dir / f"{video_id}_chamber.jpg"
                            cv2.imwrite(str(output_path), img)
                            stats['chamber'] += 1
            
            # Process bee annotations - all selected training/validation frames from all videos
            progress.setLabelText("Processing bee annotations...")
            progress.setValue(50)
            QApplication.processEvents()
            
            bee_count = 0
            total_frames_checked = 0
            
            # Iterate through all videos in the project
            for video_idx, video_dir in enumerate(video_dirs):
                if progress.wasCanceled():
                    break
                
                video_id = video_dir.name
                
                # Load video metadata to get selected frames
                import json
                metadata_file = video_dir / 'video_metadata.json'
                selected_indices = set()
                if metadata_file.exists():
                    with open(metadata_file) as f:
                        metadata = json.load(f)
                        selected_indices = set(metadata.get('selected_frames', []))
                
                # Get all frame files for this video
                frame_files = sorted(video_dir.glob('frame_*.jpg'))
                
                # Process each frame
                for frame_path in frame_files:
                    if progress.wasCanceled():
                        break
                    
                    # Extract frame index from filename (frame_000123.jpg -> 123)
                    frame_idx_in_video = int(frame_path.stem.split('_')[1])
                    
                    # Only export selected frames (training or validation)
                    if frame_idx_in_video not in selected_indices:
                        continue
                    
                    total_frames_checked += 1
                    
                    # Load frame annotations
                    annotations = self.annotation_manager.load_frame_annotations(
                        self.project_path, video_id, frame_idx_in_video
                    )
                    
                    if not annotations:
                        continue
                    
                    # Filter for bee annotations only
                    bee_anns = [ann for ann in annotations if ann.get('category', 'bee') == 'bee']
                    
                    if not bee_anns:
                        continue  # Skip if no bee annotations
                    
                    # Load frame image
                    img = cv2.imread(str(frame_path))
                    if img is None:
                        continue
                    
                    # Draw annotations (will use segmentations if available, otherwise bboxes)
                    img = self._draw_annotations_on_image(img, bee_anns, video_id)
                    
                    # Save with descriptive filename
                    output_path = bee_dir / f"{video_id}_frame_{frame_idx_in_video:06d}.jpg"
                    cv2.imwrite(str(output_path), img)
                    stats['bee'] += 1
                    bee_count += 1
                    
                    # Update progress
                    if total_frames_checked % 10 == 0:
                        progress_val = 50 + int((video_idx / len(video_dirs)) * 50)
                        progress.setValue(min(progress_val, 99))
                        progress.setLabelText(f"Processing bee annotations... ({bee_count} exported)")
                        QApplication.processEvents()
            
            progress.setValue(100)
            
            # Show completion message
            if not progress.wasCanceled():
                QMessageBox.information(
                    self,
                    "Export Complete",
                    f"Annotation visualizations exported successfully!\n\n"
                    f"Output directory: {output_dir}\n\n"
                    f"Exported:\n"
                    f"  Hive annotations: {stats['hive']} frames\n"
                    f"  Chamber annotations: {stats['chamber']} frames\n"
                    f"  Bee annotations: {stats['bee']} frames"
                )
                self.status_label.setText(
                    f"✓ Exported visualizations: {stats['hive']} hive, "
                    f"{stats['chamber']} chamber, {stats['bee']} bee"
                )
            else:
                self.status_label.setText("Export cancelled")
                
        except Exception as e:
            error_msg = f"Failed to export annotation visualizations:\n\n{str(e)}"
            self.status_label.setText("✗ Export failed")
            QMessageBox.critical(self, "Export Error", error_msg)
            import traceback
            traceback.print_exc()
    
    def _draw_annotations_on_image(self, img, annotations, video_id):
        """Draw annotations on an image with masks/bboxes and labels
        
        Args:
            img: OpenCV image (BGR)
            annotations: List of annotation dicts
            video_id: Video identifier for color consistency
            
        Returns:
            Image with annotations drawn
        """
        overlay = img.copy()
        
        # Get or initialize color mapping for this video
        if video_id not in self.video_mask_colors:
            self.video_mask_colors[video_id] = {}
        
        for ann in annotations:
            instance_id = ann.get('instance_id', ann.get('mask_id', 0))
            category = ann.get('category', 'bee')
            
            # Get or generate consistent color for this instance
            if instance_id in self.video_mask_colors[video_id]:
                color = self.video_mask_colors[video_id][instance_id]
            else:
                # Generate consistent color based on instance ID
                np.random.seed(instance_id)
                color = tuple(np.random.randint(50, 255, 3).tolist())
                self.video_mask_colors[video_id][instance_id] = color
                np.random.seed()  # Reset seed
            
            color_bgr = (int(color[2]), int(color[1]), int(color[0]))  # RGB to BGR
            
            # Draw segmentation mask if available
            if 'mask' in ann:
                mask = ann['mask']
                
                # Validate mask shape matches image shape
                img_h, img_w = img.shape[:2]
                mask_h, mask_w = mask.shape[:2]
                
                if mask_h != img_h or mask_w != img_w:
                    # Resize mask to match image dimensions
                    print(f"Warning: Resizing mask from {mask.shape} to {img.shape[:2]} for instance {instance_id}")
                    mask = cv2.resize(mask, (img_w, img_h), interpolation=cv2.INTER_NEAREST)
                
                # Draw semi-transparent mask fill
                overlay[mask > 0] = color_bgr
                
                # Find contours for outline
                contours, _ = cv2.findContours(
                    mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )
                
                # Draw green outline (thicker for visibility)
                cv2.drawContours(img, contours, -1, (0, 255, 0), 2)
                
                # Get centroid for label
                coords = np.argwhere(mask > 0)
                if len(coords) > 0:
                    cy, cx = coords.mean(axis=0).astype(int)
                    
                    # Draw label
                    label = f"{category} {instance_id}"
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = 0.6
                    thickness = 2
                    
                    # Get text size for background
                    (text_w, text_h), baseline = cv2.getTextSize(
                        label, font, font_scale, thickness
                    )
                    
                    # Draw background rectangle
                    cv2.rectangle(
                        img,
                        (cx - 5, cy - text_h - 5),
                        (cx + text_w + 5, cy + 5),
                        (0, 0, 0),
                        -1
                    )
                    
                    # Draw text
                    cv2.putText(
                        img, label, (cx, cy),
                        font, font_scale, (255, 255, 255),
                        thickness, cv2.LINE_AA
                    )
            
            # Draw bounding box if no mask or if bbox_only
            elif 'bbox' in ann or ann.get('bbox_only', False):
                bbox = ann.get('bbox', [0, 0, 0, 0])
                if len(bbox) == 4 and bbox != [0, 0, 0, 0]:
                    x, y, w, h = [int(v) for v in bbox]
                    
                    # Draw rectangle
                    cv2.rectangle(img, (x, y), (x + w, y + h), color_bgr, 2)
                    
                    # Draw label
                    label = f"{category} {instance_id}"
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = 0.6
                    thickness = 2
                    
                    # Get text size for background
                    (text_w, text_h), baseline = cv2.getTextSize(
                        label, font, font_scale, thickness
                    )
                    
                    # Draw background rectangle
                    cv2.rectangle(
                        img,
                        (x, y - text_h - 10),
                        (x + text_w + 10, y),
                        (0, 0, 0),
                        -1
                    )
                    
                    # Draw text
                    cv2.putText(
                        img, label, (x + 5, y - 5),
                        font, font_scale, (255, 255, 255),
                        thickness, cv2.LINE_AA
                    )
        
        # Blend overlay with original image for semi-transparency (30% opacity)
        # Only blend if any annotation had a mask
        has_mask = any('mask' in ann for ann in annotations)
        if has_mask:
            alpha = 0.3
            cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
        
        return img
            
    def undo(self):
        """Undo last action"""
        self.canvas.undo()
        
    def redo(self):
        """Redo last undone action"""
        self.canvas.redo()
        
    def load_frames_from_project(self):
        """Load frames from all videos in project"""
        frames_dir = self.project_path / 'frames'
        if not frames_dir.exists():
            return
        
        # Load frames from all video folders
        all_frames = []
        all_video_ids = []
        all_splits = []
        all_selected = []
        
        # Get all video subdirectories
        video_dirs = [d for d in frames_dir.iterdir() if d.is_dir()]
        
        for video_dir in sorted(video_dirs):
            video_id = video_dir.name
            split = self.project_manager.get_video_split(video_id)
            
            # Get frames for this video
            frame_files = sorted(video_dir.glob('frame_*.jpg'))
            
            # Load metadata to see which frames are selected
            import json
            metadata_file = video_dir / 'video_metadata.json'
            selected_indices = set()
            if metadata_file.exists():
                with open(metadata_file) as f:
                    metadata = json.load(f)
                    selected_indices = set(metadata.get('selected_frames', []))
            
            all_frames.extend(frame_files)
            all_video_ids.extend([video_id] * len(frame_files))
            all_splits.extend([split] * len(frame_files))
            all_selected.extend([
                int(frame_path.stem.split('_')[1]) in selected_indices
                for frame_path in frame_files
            ])
        
        self.frames = all_frames
        self.frame_video_ids = all_video_ids
        self.frame_splits = all_splits
        self.frame_selected = all_selected
        
        self.update_frame_list()
        
        # Restore last video and frame if reopening the same project
        if (hasattr(self, 'last_project_path') and self.last_project_path and
            str(self.project_path) == self.last_project_path and
            hasattr(self, 'last_video_id') and self.last_video_id):
            # Try to select the last video in the video list
            print(f"Restoring last viewed video: {self.last_video_id}")
            for i in range(self.video_list.count()):
                item = self.video_list.item(i)
                video_id = item.data(Qt.ItemDataRole.UserRole)
                if video_id == self.last_video_id:
                    # Found the video. Programmatic selection does not emit the
                    # click signal, so load it explicitly.
                    self.video_list.setCurrentRow(i)
                    self.load_video_frames(video_id)
                    # After video loads, find and load the correct frame within the video
                    if hasattr(self, 'last_frame_index_in_video'):
                        # Find the frame in self.frames that matches the saved frame index
                        for frame_idx, frame_path in enumerate(self.frames):
                            if self._get_frame_idx_in_video(frame_idx) == self.last_frame_index_in_video:
                                print(f"Restoring last viewed frame: frame_{self.last_frame_index_in_video:06d}")
                                self.load_frame(frame_idx)
                                return
                    # Fallback to first frame if we couldn't find the exact frame
                    if self.frames:
                        self.load_frame(0)
                    return
        # No saved state or video not found - just load first frame
        if self.frames:
            self.load_frame(0)
                
    def load_settings(self):
        """Load application settings"""
        settings = QSettings()
        geometry = settings.value('geometry')
        if geometry:
            self.restoreGeometry(geometry)
        
        # Load last project, video, and frame
        self.last_project_path = settings.value('last_project_path')
        self.last_video_id = settings.value('last_video_id')
        self.last_frame_index = settings.value('last_frame_index', 0, type=int)
        self.last_frame_index_in_video = settings.value('last_frame_index_in_video', 0, type=int)

        recent_projects = settings.value('recent_projects', [])
        if recent_projects is None:
            recent_projects = []
        if isinstance(recent_projects, str):
            recent_projects = [recent_projects]
        self.recent_projects = [str(Path(path)) for path in recent_projects if path]
        self.update_recent_projects_menu()
    
    def _save_project_state(self):
        """Save current project path, video ID, and frame index for restoration"""
        if hasattr(self, 'project_path') and self.project_path:
            settings = QSettings()
            settings.setValue('last_project_path', str(self.project_path))
            if hasattr(self, 'current_video_id') and self.current_video_id:
                settings.setValue('last_video_id', self.current_video_id)
            if hasattr(self, 'current_frame_idx'):
                # Save both global frame index and frame index within video
                settings.setValue('last_frame_index', self.current_frame_idx)
                # Get frame index within current video (from filename)
                frame_idx_in_video = self._get_frame_idx_in_video(self.current_frame_idx)
                settings.setValue('last_frame_index_in_video', frame_idx_in_video)
    
    def run_yolo_inference(self):
        """Run YOLO inference on the current frame"""
        if not self.yolo_toolbar.is_model_loaded():
            QMessageBox.warning(
                self, "No Model",
                "Please load a YOLO checkpoint first."
            )
            return
        
        if not self.frames or self.current_frame_idx >= len(self.frames):
            QMessageBox.warning(
                self, "No Frame",
                "No frame is currently loaded."
            )
            return
        
        try:
            # Get current frame path
            frame_path = self.frames[self.current_frame_idx]
            
            # Run inference
            model = self.yolo_toolbar.get_model()
            self.status_label.setText("Running YOLO inference...")
            QApplication.processEvents()  # Update UI
            
            # Run prediction with verbose=False to reduce output
            results = model.predict(
                source=str(frame_path),
                conf=0.5,  # Confidence threshold
                iou=0.5,   # NMS IoU threshold
                retina_masks=True,  # Return masks at original image size
                verbose=False
            )
            
            if not results or len(results) == 0:
                self.status_label.setText("No detections found")
                QMessageBox.information(
                    self, "No Detections",
                    "YOLO did not detect any instances in the current frame."
                )
                return
            
            # Get the first result (single image)
            result = results[0]
            
            # Check if there are any detections
            if result.masks is None or len(result.masks) == 0:
                self.status_label.setText("No detections found")
                QMessageBox.information(
                    self, "No Detections",
                    "YOLO did not detect any instances in the current frame."
                )
                return
            
            # Convert YOLO results to Detection objects
            detections = self._yolo_results_to_detections(result, model)
            
            if not detections:
                self.status_label.setText("No detections found")
                QMessageBox.information(
                    self, "No Detections",
                    "YOLO did not detect any instances in the current frame."
                )
                return
            
            # Use tracker to match detections to existing annotations
            if self.tracking_enabled and self.current_video_id:
                tracker = self._get_or_create_tracker(self.current_video_id)
                
                # Note: We don't initialize tracker from current canvas because:
                # - Tracker maintains state across frames (frame N-1 -> frame N)
                # - Current canvas might be empty (new frame) or have frame N's old annotations
                # - Tracker's active_tracks already contain the previous frame's detections
                # The only exceptions are when manually adding the first annotations to a fresh video
                
                # Match detections to tracks
                matched_detections = tracker.match_detections_to_tracks(
                    detections,
                    self.current_frame_idx
                )
                
                # Count how many were matched vs new
                existing_track_ids = set(tracker.active_tracks.keys())
                matched_ids = [track_id for _, track_id in matched_detections]
                num_matched = sum(1 for tid in matched_ids if tid in existing_track_ids or tid in tracker.lost_tracks)
                num_new = len(matched_detections) - num_matched
                
                print(f"Tracking: {num_matched} matched to existing tracks, {num_new} new tracks created")
                print(f"  Previous tracks: {list(existing_track_ids)}")
                print(f"  Assigned IDs: {matched_ids}")
                
                # Clear existing annotations and add matched detections
                self.canvas.set_annotations([])
                
                # Add matched detections with their assigned IDs
                # Use rebuild_viz=False for all but the last mask to avoid redundant rebuilds
                current_category = self.canvas.current_annotation_type
                for i, (detection, track_id) in enumerate(matched_detections):
                    # Get or use existing color for this track
                    if self.current_video_id not in self.video_mask_colors:
                        self.video_mask_colors[self.current_video_id] = {}
                    
                    color = self.video_mask_colors[self.current_video_id].get(track_id)
                    rebuild_viz = (i == len(matched_detections) - 1)
                    self.canvas.add_mask(detection.mask, mask_id=track_id, color=color, rebuild_viz=rebuild_viz, category=current_category)
                    self.canvas.annotation_metadata[track_id]['source'] = 'yolo_tracked'
                
                # Register any new colors
                self._register_canvas_colors()
                
                # Update next_mask_id
                self.video_next_mask_id[self.current_video_id] = tracker.next_track_id
                
                # Mark frame as modified so annotations will be saved
                self._mark_current_annotations_dirty()
                
                self.status_label.setText(
                    f"✓ YOLO inference complete: {len(matched_detections)} instances detected (with tracking)"
                )
            else:
                # No tracking - just add detections with sequential IDs
                # Get highest existing ID from canvas
                canvas_annotations = self.canvas.get_annotations()
                next_id = max([ann.get('instance_id', ann.get('mask_id', 0)) for ann in canvas_annotations], default=0) + 1
                
                # Clear and add new detections
                self.canvas.set_annotations([])
                
                # Use rebuild_viz=False for all but the last mask to avoid redundant rebuilds
                current_category = self.canvas.current_annotation_type
                for i, detection in enumerate(detections):
                    mask_id = next_id + i
                    rebuild_viz = (i == len(detections) - 1)
                    self.canvas.add_mask(detection.mask, mask_id=mask_id, rebuild_viz=rebuild_viz, category=current_category)
                    self.canvas.annotation_metadata[mask_id]['source'] = 'yolo'
                
                self._register_canvas_colors()
                
                # Mark frame as modified so annotations will be saved
                self._mark_current_annotations_dirty()
                
                self.status_label.setText(
                    f"✓ YOLO inference complete: {len(detections)} instances detected"
                )
            
            # Detect markers (ArUco/QR codes) in new annotations
            self.detect_and_update_markers()
            
            self.update_instance_list_from_canvas()
            
            # Show summary
            num_detected = len(detections)
            tracking_msg = " (with ID tracking)" if self.tracking_enabled and self.current_video_id else ""
            QMessageBox.information(
                self, "Inference Complete",
                f"YOLO detected {num_detected} instance(s) in the current frame{tracking_msg}.\n\n"
                f"The detections have been added as new instances.\n"
                f"You can now edit, delete, or propagate them as needed."
            )
            
        except ImportError as e:
            QMessageBox.critical(
                self, "Import Error",
                f"Could not import required libraries:\n{str(e)}\n\n"
                "Make sure ultralytics and opencv-python are installed."
            )
        except Exception as e:
            import traceback
            error_msg = traceback.format_exc()
            QMessageBox.critical(
                self, "Inference Error",
                f"Error running YOLO inference:\n{str(e)}\n\n{error_msg}"
            )
            self.status_label.setText("Inference failed")
        finally:
            # Return focus to canvas so spacebar can toggle masks
            self.canvas.setFocus()
    
    def run_yolo_bbox_inference(self):
        """Run YOLO bounding box inference on the current frame"""
        if not self.yolo_bbox_toolbar.is_model_loaded():
            QMessageBox.warning(
                self, "No Model",
                "Please load a YOLO checkpoint first."
            )
            return
        
        if not self.frames or self.current_frame_idx >= len(self.frames):
            QMessageBox.warning(
                self, "No Frame",
                "No frame is currently loaded."
            )
            return
        
        # Check if bee instance segmentations already exist
        # (Allow bbox detection even if hive/chamber masks exist)
        if self.canvas.bee_mask is not None and np.any(self.canvas.bee_mask > 0):
            QMessageBox.warning(
                self, "Bee Segmentations Exist",
                "Cannot run YOLO bbox inference on frames that already have bee segmentation masks.\n\n"
                "Bounding boxes should only be used for frames without bee segmentation masks.\n"
                "Note: Hive and chamber masks are allowed."
            )
            return

        existing_bbox_annotations = self._get_current_bee_bbox_annotations()
        replacing_existing_bboxes = False
        if existing_bbox_annotations:
            reply = QMessageBox.warning(
                self,
                "Replace Existing Bounding Boxes?",
                f"This frame already has {len(existing_bbox_annotations)} bee bounding box annotation(s).\n\n"
                "Running YOLO BBox detection again will replace the current boxes on this frame. "
                "If you only want to show hidden boxes, use the BBoxes checkbox or the per-instance "
                "checkboxes in the Instances panel instead.\n\n"
                "Replace the existing boxes with a fresh detection pass?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No
            )
            if reply != QMessageBox.StandardButton.Yes:
                self.status_label.setText("BBox detection cancelled; existing boxes kept")
                return
            replacing_existing_bboxes = True
        
        try:
            # Get current frame path
            frame_path = self.frames[self.current_frame_idx]
            
            # Run inference
            model = self.yolo_bbox_toolbar.get_model()
            self.status_label.setText("Running YOLO BBox inference...")
            QApplication.processEvents()  # Update UI
            
            # Run prediction with verbose=False to reduce output
            results = model.predict(
                source=str(frame_path),
                conf=0.5,  # Confidence threshold
                iou=0.5,   # NMS IoU threshold
                verbose=False
            )
            
            if not results or len(results) == 0:
                self.status_label.setText("No detections found")
                QMessageBox.information(
                    self, "No Detections",
                    "YOLO did not detect any instances in the current frame."
                )
                return
            
            # Get the first result (single image)
            result = results[0]
            
            # Check if there are any detections
            if result.boxes is None or len(result.boxes) == 0:
                self.status_label.setText("No detections found")
                QMessageBox.information(
                    self, "No Detections",
                    "YOLO did not detect any instances in the current frame."
                )
                return
            
            # Convert YOLO bboxes to Detection objects (bbox-only, no masks)
            boxes = result.boxes.xyxy.cpu().numpy()
            confidences = result.boxes.conf.cpu().numpy()
            class_ids = result.boxes.cls.cpu().numpy() if result.boxes.cls is not None else None
            orig_shape = result.orig_shape  # (height, width)
            
            detections = []
            for i in range(len(boxes)):
                bbox = boxes[i]
                x1, y1, x2, y2 = map(int, bbox)
                
                # Don't create mask - bbox-only detection
                # Bboxes can be converted to segmentations later using SAM2
                detection = Detection(
                    bbox=bbox.tolist(),
                    mask=None,  # No mask for bbox-only detections
                    confidence=float(confidences[i]),
                    source='yolo_bbox',
                    class_id=int(class_ids[i]) if class_ids is not None else 0
                )
                detections.append(detection)
            
            if not detections:
                self.status_label.setText("No detections found")
                QMessageBox.information(
                    self, "No Detections",
                    "YOLO did not detect any instances in the current frame."
                )
                return
            
            if replacing_existing_bboxes:
                matched_detections, next_id = self._match_bbox_detections_to_existing_annotations(
                    detections,
                    existing_bbox_annotations
                )

                # Clear existing annotations and add replacement detections while
                # preserving current IDs where boxes overlap.
                self.canvas.set_annotations([])

                bbox_annotations = []
                for detection, track_id in matched_detections:
                    if self.current_video_id not in self.video_mask_colors:
                        self.video_mask_colors[self.current_video_id] = {}

                    color = self.video_mask_colors[self.current_video_id].get(track_id)
                    if color is None:
                        color = tuple(np.random.randint(0, 255, 3).tolist())
                        self.video_mask_colors[self.current_video_id][track_id] = color

                    x1, y1, x2, y2 = detection.bbox
                    bbox_xywh = [x1, y1, x2 - x1, y2 - y1]

                    bbox_annotations.append({
                        'mask_id': track_id,
                        'instance_id': track_id,
                        'bbox': bbox_xywh,
                        'confidence': detection.confidence,
                        'source': 'yolo_bbox',
                        'bbox_only': True
                    })

                self.canvas.set_annotations(
                    bbox_annotations,
                    mask_colors=self.video_mask_colors.get(self.current_video_id, {})
                )
                self._register_canvas_colors()

                if self.current_video_id:
                    self.video_next_mask_id[self.current_video_id] = max(
                        self.video_next_mask_id.get(self.current_video_id, next_id),
                        next_id
                    )

                self._mark_current_annotations_dirty()
                self.status_label.setText(
                    f"✓ YOLO BBox replacement complete: {len(matched_detections)} instances detected"
                )

            # Use tracker to match detections to existing annotations
            elif self.tracking_enabled and self.current_video_id:
                tracker = self._get_or_create_tracker(self.current_video_id)
                
                # Match detections to tracks
                matched_detections = tracker.match_detections_to_tracks(
                    detections,
                    self.current_frame_idx
                )
                
                # Count how many were matched vs new
                existing_track_ids = set(tracker.active_tracks.keys())
                matched_ids = [track_id for _, track_id in matched_detections]
                num_matched = sum(1 for tid in matched_ids if tid in existing_track_ids or tid in tracker.lost_tracks)
                num_new = len(matched_detections) - num_matched
                
                print(f"BBox Tracking: {num_matched} matched to existing tracks, {num_new} new tracks created")
                print(f"  Previous tracks: {list(existing_track_ids)}")
                print(f"  Assigned IDs: {matched_ids}")
                
                # Clear existing annotations and add matched detections
                self.canvas.set_annotations([])
                
                # Build bbox-only annotations
                bbox_annotations = []
                for detection, track_id in matched_detections:
                    # Get or use existing color for this track
                    if self.current_video_id not in self.video_mask_colors:
                        self.video_mask_colors[self.current_video_id] = {}
                    
                    color = self.video_mask_colors[self.current_video_id].get(track_id)
                    if color is None:
                        color = tuple(np.random.randint(0, 255, 3).tolist())
                        self.video_mask_colors[self.current_video_id][track_id] = color
                    
                    # Convert bbox from [x1, y1, x2, y2] to [x, y, w, h]
                    x1, y1, x2, y2 = detection.bbox
                    bbox_xywh = [x1, y1, x2 - x1, y2 - y1]
                    
                    bbox_annotations.append({
                        'mask_id': track_id,
                        'instance_id': track_id,
                        'bbox': bbox_xywh,
                        'confidence': detection.confidence,
                        'source': 'yolo_bbox',
                        'bbox_only': True  # Flag to indicate this is bbox-only
                    })
                
                # Set bbox annotations on canvas
                self.canvas.set_annotations(bbox_annotations, mask_colors=self.video_mask_colors.get(self.current_video_id, {}))
                
                # Register any new colors
                self._register_canvas_colors()
                
                # Update next_mask_id
                self.video_next_mask_id[self.current_video_id] = tracker.next_track_id
                
                # Mark frame as modified so annotations will be saved
                self._mark_current_annotations_dirty()
                
                self.status_label.setText(
                    f"✓ YOLO BBox inference complete: {len(matched_detections)} instances detected (with tracking)"
                )
            else:
                # No tracking - just add detections with sequential IDs
                # Get highest existing ID from canvas
                canvas_annotations = self.canvas.get_annotations()
                next_id = max([ann.get('instance_id', ann.get('mask_id', 0)) for ann in canvas_annotations], default=0) + 1
                
                # Clear and add new detections
                self.canvas.set_annotations([])
                
                # Build bbox-only annotations
                bbox_annotations = []
                for i, detection in enumerate(detections):
                    mask_id = next_id + i
                    
                    # Convert bbox from [x1, y1, x2, y2] to [x, y, w, h]
                    x1, y1, x2, y2 = detection.bbox
                    bbox_xywh = [x1, y1, x2 - x1, y2 - y1]
                    
                    bbox_annotations.append({
                        'mask_id': mask_id,
                        'instance_id': mask_id,
                        'bbox': bbox_xywh,
                        'confidence': detection.confidence,
                        'source': 'yolo_bbox',
                        'bbox_only': True  # Flag to indicate this is bbox-only
                    })
                
                # Set bbox annotations on canvas
                self.canvas.set_annotations(bbox_annotations)
                self._register_canvas_colors()
                
                # Mark frame as modified so annotations will be saved
                self._mark_current_annotations_dirty()
                
                self.status_label.setText(
                    f"✓ YOLO BBox inference complete: {len(detections)} instances detected"
                )
            
            # Detect markers (ArUco/QR codes) in new annotations
            self.detect_and_update_markers()
            
            self.update_instance_list_from_canvas()
            
            # Show summary
            num_detected = len(detections)
            tracking_msg = " (with ID tracking)" if self.tracking_enabled and self.current_video_id else ""
            QMessageBox.information(
                self, "Inference Complete",
                f"YOLO BBox detected {num_detected} instance(s) in the current frame{tracking_msg}.\n\n"
                f"The detections have been added as bounding box instances.\n"
                f"You can now edit, delete, or propagate them as needed."
            )
            
        except ImportError as e:
            QMessageBox.critical(
                self, "Import Error",
                f"Could not import required libraries:\n{str(e)}\n\n"
                "Make sure ultralytics and opencv-python are installed."
            )
        except Exception as e:
            import traceback
            error_msg = traceback.format_exc()
            QMessageBox.critical(
                self, "Inference Error",
                f"Error running YOLO BBox inference:\n{str(e)}\n\n{error_msg}"
            )
            self.status_label.setText("Inference failed")
        finally:
            # Return focus to canvas so spacebar can toggle masks
            self.canvas.setFocus()
    
    def run_hive_inference(self):
        """Run YOLO hive detection on the current frame"""
        model = self.hive_chamber_toolbar.get_hive_model()
        if model is None:
            QMessageBox.warning(
                self, "No Model",
                "Please load a Hive detection model first."
            )
            return
        
        self._run_hive_chamber_inference_impl(model, 'hive', "Hive")
    
    def run_chamber_inference(self):
        """Run YOLO chamber detection on the current frame"""
        model = self.hive_chamber_toolbar.get_chamber_model()
        if model is None:
            QMessageBox.warning(
                self, "No Model",
                "Please load a Chamber detection model first."
            )
            return
        
        self._run_hive_chamber_inference_impl(model, 'chamber', "Chamber")
    
    def run_pollen_inference(self):
        """Run YOLO pollen detection on the current frame"""
        model = self.hive_chamber_toolbar.get_pollen_model()
        if model is None:
            QMessageBox.warning(
                self, "No Model",
                "Please load a Pollen detection model first."
            )
            return
        
        self._run_hive_chamber_inference_impl(model, 'pollen', "Pollen")
    
    def run_nectar_inference(self):
        model = self.hive_chamber_toolbar.get_nectar_model()
        if model is None:
            QMessageBox.warning(self, "Model Not Loaded", "Please load a Nectar model first.")
            return
        self.toolbar.show_nectar_checkbox.setChecked(True)
        self._run_hive_chamber_inference_impl(model, 'nectar', "Nectar source")

    def run_hive_chamber_both(self):
        """Run the loaded structure and resource models on the current frame."""
        hive_model = self.hive_chamber_toolbar.get_hive_model()
        chamber_model = self.hive_chamber_toolbar.get_chamber_model()
        pollen_model = self.hive_chamber_toolbar.get_pollen_model()
        nectar_model = self.hive_chamber_toolbar.get_nectar_model()
        
        if all(model is None for model in (hive_model, chamber_model, pollen_model, nectar_model)):
            QMessageBox.warning(
                self, "Models Not Loaded",
                "Please load at least one model (Hive, Chamber, Pollen, or Nectar) first."
            )
            return
        
        # Run all loaded models
        if hive_model is not None:
            self._run_hive_chamber_inference_impl(hive_model, 'hive', "Hive", show_dialog=False)
        if chamber_model is not None:
            self._run_hive_chamber_inference_impl(chamber_model, 'chamber', "Chamber", show_dialog=False)
        if pollen_model is not None:
            self._run_hive_chamber_inference_impl(pollen_model, 'pollen', "Pollen", show_dialog=True)
        if nectar_model is not None:
            self.run_nectar_inference()
    
    def _run_hive_chamber_inference_impl(self, model, category, display_name, show_dialog=True):
        """Implementation of hive/chamber/pollen inference
        
        Args:
            model: The YOLO model to use
            category: 'hive', 'chamber', or 'pollen'
            display_name: Display name for messages ("Hive", "Chamber", or "Pollen")
            show_dialog: Whether to show completion dialog
        """
        if not self.frames or self.current_frame_idx >= len(self.frames):
            QMessageBox.warning(
                self, "No Frame",
                "No frame is currently loaded."
            )
            return
        
        try:
            # Get current frame path
            frame_path = self.frames[self.current_frame_idx]
            
            # Run inference
            self.status_label.setText(f"Running {display_name} detection...")
            QApplication.processEvents()  # Update UI
            
            # Run prediction with verbose=False to reduce output
            results = model.predict(
                source=str(frame_path),
                conf=0.5,  # Confidence threshold
                iou=0.5,   # NMS IoU threshold
                retina_masks=True,  # Return masks at original image size
                verbose=False
            )
            
            if not results or len(results) == 0:
                self.status_label.setText(f"No {category} detections found")
                if show_dialog:
                    QMessageBox.information(
                        self, "No Detections",
                        f"YOLO did not detect any {category} instances in the current frame."
                    )
                return
            
            # Get the first result (single image)
            result = results[0]
            
            # Check if there are any detections
            if result.masks is None or len(result.masks) == 0:
                self.status_label.setText(f"No {category} detections found")
                if show_dialog:
                    QMessageBox.information(
                        self, "No Detections",
                        f"YOLO did not detect any {category} instances in the current frame."
                    )
                return
            
            # Convert YOLO results to Detection objects
            detections = self._yolo_results_to_detections(result, model)
            
            if not detections:
                self.status_label.setText(f"No {category} detections found")
                if show_dialog:
                    QMessageBox.information(
                        self, "No Detections",
                        f"YOLO did not detect any {category} instances in the current frame."
                    )
                return
            
            # Get existing annotations
            existing_annotations = self.canvas.get_annotations()
            
            # Filter out any existing annotations of this category
            # (We'll replace them with the new detections)
            other_category_annotations = [ann for ann in existing_annotations 
                                         if ann.get('category', 'bee') != category]
            
            # Get the highest existing ID across ALL categories (not just this category)
            # This is important because all instances (bee, hive, chamber) share the same ID space
            # and we need to avoid ID conflicts
            if self.current_video_id:
                # Always find the max ID across ALL categories and ALL stored next_ids
                # to ensure we never have ID conflicts
                max_id = 0
                
                # Check existing annotations
                for ann in existing_annotations:
                    mask_id = ann.get('instance_id', ann.get('mask_id', 0))
                    max_id = max(max_id, mask_id)
                
                # Also check all stored next_id values for this video
                for key, stored_next_id in self.video_next_mask_id.items():
                    if key.startswith(f"{self.current_video_id}_"):
                        max_id = max(max_id, stored_next_id - 1)
                
                next_id = max_id + 1
                
                # Store the updated next_id for this category
                category_key = f"{self.current_video_id}_{category}"
                self.video_next_mask_id[category_key] = next_id
            else:
                # No video loaded, use simple sequential IDs
                # Find max ID across ALL categories to avoid conflicts
                next_id = max([ann.get('instance_id', ann.get('mask_id', 0)) 
                              for ann in existing_annotations], 
                             default=0) + 1
            
            # Add new detections with the appropriate category
            for i, detection in enumerate(detections):
                mask_id = next_id + i
                # Don't rebuild viz until the last mask for performance
                rebuild_viz = (i == len(detections) - 1)
                self.canvas.add_mask(detection.mask, mask_id=mask_id, rebuild_viz=rebuild_viz, 
                                   category=category)
                self.canvas.annotation_metadata[mask_id]['source'] = 'yolo'
            
            # Update next ID for all categories of this video to avoid future conflicts
            if self.current_video_id:
                new_next_id = next_id + len(detections)
                for cat in CATEGORIES:
                    cat_key = f"{self.current_video_id}_{cat}"
                    # Update to the max of current stored value or new value
                    if cat_key in self.video_next_mask_id:
                        self.video_next_mask_id[cat_key] = max(self.video_next_mask_id[cat_key], new_next_id)
                    else:
                        self.video_next_mask_id[cat_key] = new_next_id
            
            self._register_canvas_colors()
            
            # Mark frame as modified so annotations will be saved
            self._mark_current_annotations_dirty()
            
            self.status_label.setText(
                f"✓ {display_name} detection complete: {len(detections)} instance(s) detected"
            )
            
            # Update instance list
            self.update_instance_list_from_canvas()
            
            # Show summary dialog if requested
            if show_dialog:
                scope_note = (
                    'These annotations apply only to the current frame.'
                    if category in frame_categories(self.annotation_manager.project_info)
                    else 'These annotations are shared across all frames of this video.'
                )
                QMessageBox.information(
                    self, "Detection Complete",
                    f"{display_name} detection found {len(detections)} instance(s) in the current frame.\n\n"
                    f"The detections have been added as {category} instances.\n"
                    + scope_note
                )
            
        except ImportError as e:
            QMessageBox.critical(
                self, "Import Error",
                f"Could not import required libraries:\n{str(e)}\n\n"
                "Make sure ultralytics and opencv-python are installed."
            )
        except Exception as e:
            import traceback
            error_msg = traceback.format_exc()
            QMessageBox.critical(
                self, "Inference Error",
                f"Error running {display_name} detection:\n{str(e)}\n\n{error_msg}"
            )
            self.status_label.setText(f"{display_name} detection failed")
        finally:
            # Return focus to canvas so spacebar can toggle masks
            self.canvas.setFocus()
    
    def track_from_last_frame(self):
        """Track instances from the last annotated frame to match IDs with current frame's instances"""
        if not self.frames or self.current_frame_idx >= len(self.frames):
            QMessageBox.warning(
                self, "No Frame",
                "No frame is currently loaded."
            )
            return
        
        if not self.current_video_id:
            QMessageBox.warning(
                self, "No Video",
                "No video is currently selected."
            )
            return
        
        # Check if current frame has annotations
        current_annotations = self.canvas.get_annotations()
        if not current_annotations or len(current_annotations) == 0:
            QMessageBox.warning(
                self, "No Current Annotations",
                "The current frame has no instances.\n\n"
                "Please run YOLO inference or annotate instances first before tracking."
            )
            return
        
        try:
            # Find the last annotated frame before the current one
            last_annotated_idx = None
            for idx in range(self.current_frame_idx - 1, -1, -1):
                # Check if frame belongs to same video
                if idx < len(self.frame_video_ids) and self.frame_video_ids[idx] == self.current_video_id:
                    # Check if frame has annotations
                    frame_idx_in_video = self._get_frame_idx_in_video(idx)
                    annotations = self.annotation_manager.load_frame_annotations(
                        self.project_path, self.current_video_id, frame_idx_in_video
                    )
                    if annotations and len(annotations) > 0:
                        last_annotated_idx = idx
                        break
            
            if last_annotated_idx is None:
                QMessageBox.information(
                    self, "No Annotations",
                    "No annotated frames found before the current frame in this video."
                )
                return
            
            # Get the last annotated frame's annotations
            frame_idx_in_video = self._get_frame_idx_in_video(last_annotated_idx)
            last_annotations = self.annotation_manager.load_frame_annotations(
                self.project_path, self.current_video_id, frame_idx_in_video
            )
            
            self.status_label.setText(f"Tracking from frame {last_annotated_idx}...")
            QApplication.processEvents()
            
            # Build a simple IoU-based matching without creating new tracks
            # Convert current frame's annotations to Detection objects for matching
            current_detections = self._annotations_to_detections(current_annotations)
            
            # Build last frame's tracks for matching
            last_frame_tracks = []
            last_frame_track_ids = []
            
            for ann in last_annotations:
                # Handle both mask-based and bbox-only annotations
                mask = ann.get('mask')
                bbox_data = ann.get('bbox')
                mask_id = ann.get('mask_id', ann.get('instance_id', 1))
                
                # Need either mask or bbox
                if mask is None and bbox_data is None:
                    continue
                
                # Compute or get bbox
                if mask is not None and np.any(mask > 0):
                    # Compute bbox from mask
                    y_indices, x_indices = np.where(mask > 0)
                    bbox = np.array([
                        float(x_indices.min()),
                        float(y_indices.min()),
                        float(x_indices.max()),
                        float(y_indices.max())
                    ])
                elif bbox_data is not None:
                    # Use provided bbox (convert from [x, y, w, h] to [x1, y1, x2, y2] if needed)
                    if len(bbox_data) == 4:
                        x, y, w, h = bbox_data
                        bbox = np.array([float(x), float(y), float(x + w), float(y + h)])
                    else:
                        # Already in [x1, y1, x2, y2] format
                        bbox = np.array(bbox_data, dtype=float)
                else:
                    continue  # Skip if neither valid mask nor bbox
                
                track = Track(
                    track_id=mask_id,
                    bbox=bbox,
                    mask=mask,  # May be None for bbox-only
                    last_seen_frame=last_annotated_idx
                )
                last_frame_tracks.append(track)
                last_frame_track_ids.append(mask_id)
            
            if not last_frame_tracks:
                QMessageBox.warning(
                    self, "No Valid Tracks",
                    "No valid tracks found in the last annotated frame."
                )
                return
            
            # Compute IoU matrix between current detections and last frame tracks
            from core.instance_tracker import InstanceTracker
            tracker = self._get_or_create_tracker(self.current_video_id)
            iou_matrix = tracker._compute_iou_matrix(
                current_detections, 
                last_frame_tracks, 
                use_mask_iou=True
            )
            
            # Perform Hungarian matching with IoU threshold
            from scipy.optimize import linear_sum_assignment
            iou_threshold = 0.3  # Minimum IoU for matching
            
            # Convert to cost matrix (maximize IoU = minimize -IoU)
            cost_matrix = 1.0 - iou_matrix
            
            # Hungarian algorithm
            row_indices, col_indices = linear_sum_assignment(cost_matrix)
            
            # Build assignment mapping: detection_idx -> track_id (only for valid matches)
            detection_to_track_id = {}
            num_matched = 0
            
            for det_idx, track_idx in zip(row_indices, col_indices):
                if iou_matrix[det_idx, track_idx] >= iou_threshold:
                    detection_to_track_id[det_idx] = last_frame_track_ids[track_idx]
                    num_matched += 1
            
            num_unmatched = len(current_detections) - num_matched
            
            print(f"Tracking from frame {last_annotated_idx}:")
            print(f"  {num_matched} matched to existing track IDs, {num_unmatched} unmatched (keeping original IDs)")
            print(f"  Previous track IDs: {last_frame_track_ids}")
            print(f"  Matched pairs: {[(i, detection_to_track_id[i]) for i in detection_to_track_id]}")
            
            # Update canvas annotations with new IDs (only for matched ones)
            # First, collect all IDs that will be assigned
            assigned_ids = set()
            id_assignments = {}  # det_idx -> new_id
            
            # First pass: assign matched IDs and detect conflicts
            for i in range(len(current_detections)):
                if i in detection_to_track_id:
                    # This detection matched - use the matched track ID
                    matched_id = detection_to_track_id[i]
                    id_assignments[i] = matched_id
                    assigned_ids.add(matched_id)
                else:
                    # Unmatched - keep original ID
                    original_id = current_annotations[i].get('mask_id', current_annotations[i].get('instance_id', i + 1))
                    id_assignments[i] = original_id
                    assigned_ids.add(original_id)
            
            # Second pass: resolve ID conflicts
            # If multiple detections were assigned the same ID, reassign the unmatched ones
            id_counts = {}
            for i, assigned_id in id_assignments.items():
                id_counts[assigned_id] = id_counts.get(assigned_id, 0) + 1
            
            # Find conflicting IDs and reassign
            next_available_id = max(assigned_ids) + 1 if assigned_ids else 1
            final_assignments = {}
            id_usage = {}  # Track which ID is used by which detection index
            
            # Process matched detections first (they have priority)
            for i in range(len(current_detections)):
                if i in detection_to_track_id:
                    matched_id = id_assignments[i]
                    if matched_id not in id_usage:
                        # This matched ID is not yet used - assign it
                        final_assignments[i] = matched_id
                        id_usage[matched_id] = i
                    else:
                        # This matched ID is already used by another detection
                        # This shouldn't happen with Hungarian matching, but check anyway
                        print(f"Warning: Matched ID {matched_id} already assigned to detection {id_usage[matched_id]}, reassigning detection {i}")
                        while next_available_id in assigned_ids or next_available_id in id_usage:
                            next_available_id += 1
                        final_assignments[i] = next_available_id
                        id_usage[next_available_id] = i
                        next_available_id += 1
            
            # Process unmatched detections
            for i in range(len(current_detections)):
                if i not in detection_to_track_id:
                    original_id = id_assignments[i]
                    if original_id not in id_usage:
                        # Original ID is free - keep it
                        final_assignments[i] = original_id
                        id_usage[original_id] = i
                    else:
                        # Original ID conflicts with a matched detection - assign new ID
                        print(f"Warning: Detection {i} wanted ID {original_id} but it's taken by matched detection {id_usage[original_id]}, reassigning")
                        while next_available_id in id_usage:
                            next_available_id += 1
                        final_assignments[i] = next_available_id
                        id_usage[next_available_id] = i
                        next_available_id += 1
            
            # Rebuild the canvas with updated IDs
            updated_annotations = []
            
            for i, annotation in enumerate(current_annotations):
                new_mask_id = final_assignments[i]
                
                # Get or preserve color
                if self.current_video_id not in self.video_mask_colors:
                    self.video_mask_colors[self.current_video_id] = {}
                
                color = self.video_mask_colors[self.current_video_id].get(new_mask_id)
                if color is None:
                    color = tuple(np.random.randint(0, 255, 3).tolist())
                    self.video_mask_colors[self.current_video_id][new_mask_id] = color
                
                # Build updated annotation preserving format (bbox-only vs mask-based)
                updated_ann = annotation.copy()
                updated_ann['mask_id'] = new_mask_id
                updated_ann['instance_id'] = new_mask_id
                
                updated_annotations.append(updated_ann)
            
            # Set updated annotations on canvas
            self.canvas.set_annotations(updated_annotations, mask_colors=self.video_mask_colors.get(self.current_video_id, {}))
            
            # Register any new colors that were generated
            self._register_canvas_colors()
            
            # Mark frame as modified so annotations will be saved
            self._mark_current_annotations_dirty()
            
            self.update_instance_list_from_canvas()
            
            self.status_label.setText(
                f"✓ Tracked from frame {last_annotated_idx}: "
                f"{num_matched} matched, {num_unmatched} kept original IDs"
            )
            
            QMessageBox.information(
                self, "Tracking Complete",
                f"Reassigned IDs for {num_matched} instance(s) from frame {last_annotated_idx}.\n\n"
                f"Matched to existing IDs: {num_matched}\n"
                f"Kept original IDs: {num_unmatched}\n\n"
                f"Instance IDs and colors have been preserved for matched instances."
            )
            
        except Exception as e:
            import traceback
            error_msg = traceback.format_exc()
            QMessageBox.critical(
                self, "Tracking Error",
                f"Error tracking from last frame:\n{str(e)}\n\n{error_msg}"
            )
            self.status_label.setText("Tracking failed")

    def propagate_yolo_bbox(self):
        """Propagate YOLO bbox detections through video frames using Ultralytics ByteTrack"""
        # Check if model is loaded
        if not self.yolo_bbox_toolbar.is_model_loaded():
            QMessageBox.warning(
                self, "No Model",
                "Please load a YOLO checkpoint first."
            )
            return
        
        # Check if we have frames
        if not self.frames:
            QMessageBox.warning(
                self, "No Video",
                "No video is currently loaded."
            )
            return
        
        # Ask user for target frame
        from PyQt6.QtWidgets import QInputDialog
        current_idx = self.current_frame_idx
        max_idx = len(self.frames) - 1
        
        target_idx, ok = QInputDialog.getInt(
            self,
            "Propagate YOLO BBox Detections",
            f"Propagate from current frame ({current_idx}) to which frame?\n\n"
            f"Enter target frame index (current: {current_idx}, max: {max_idx}):",
            value=min(current_idx + 100, max_idx),  # Default to 100 frames ahead
            min=current_idx + 1,
            max=max_idx,
            step=1
        )
        
        if not ok:
            return
        
        # Calculate number of frames to propagate through
        num_frames = target_idx - current_idx + 1  # +1 to include current frame
        start_idx = current_idx
        
        # Confirm with user
        reply = QMessageBox.question(
            self,
            "Confirm Propagation",
            f"This will run YOLO bbox detection with ByteTrack on {num_frames} frames:\n"
            f"  From: Frame {current_idx}\n"
            f"  To: Frame {target_idx}\n\n"
            f"ByteTrack will maintain instance IDs across frames.\n\n"
            f"Continue?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        
        if reply == QMessageBox.StandardButton.No:
            return
        
        # Create progress dialog
        from PyQt6.QtWidgets import QProgressDialog
        progress = QProgressDialog(
            f"Propagating YOLO bbox detections through {num_frames} frames...",
            "Cancel",
            0,
            num_frames,
            self
        )
        progress.setWindowTitle("YOLO BBox Propagation")
        progress.setWindowModality(Qt.WindowModality.WindowModal)
        progress.setMinimumDuration(0)
        progress.setValue(0)
        
        # Track success/failure counts
        successful_frames = 0
        failed_frames = 0
        frames_with_detections = 0
        frames_without_detections = 0
        
        # Create custom ByteTrack config
        import tempfile
        import yaml
        
        bytetrack_config = {
            'tracker_type': 'bytetrack',
            'track_high_thresh': 0.1,
            'track_low_thresh': 0.1,
            'new_track_thresh': 0.75,
            'track_buffer': 20,
            'match_thresh': 0.9,
            'fuse_score': True,
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(bytetrack_config, f)
            tracker_config_path = f.name
        
        print(f"Starting Ultralytics ByteTrack on {num_frames} frames...")
        print(f"  Frames: {current_idx} to {target_idx}")
        print(f"  Config: {tracker_config_path}")
        
        try:
            # Get YOLO model
            model = self.yolo_bbox_toolbar.get_model()
            
            # Process frame by frame using model.track() with persist=True
            for step in range(num_frames):
                # Check if user cancelled
                if progress.wasCanceled():
                    self.status_label.setText(f"Propagation cancelled at frame {start_idx + step}")
                    break
                
                next_frame_idx = start_idx + step
                
                # Update progress every N frames
                if step % 5 == 0 or step == num_frames - 1:
                    progress.setValue(step)
                    progress.setLabelText(
                        f"Processing frame {next_frame_idx} of {target_idx}\n"
                        f"({step + 1} / {num_frames})\n\n"
                        f"Detections: {frames_with_detections} frames\n"
                        f"No detections: {frames_without_detections} frames"
                    )
                    QApplication.processEvents()
                
                try:
                    # Run YOLO tracking on this frame
                    frame_path = str(self.frames[next_frame_idx])
                    
                    results = model.track(
                        source=frame_path,
                        conf=0.5,
                        iou=0.5,
                        verbose=False,
                        persist=True,
                        tracker=tracker_config_path
                    )
                    
                    # Get the result (single frame)
                    result = results[0] if results else None
                    
                    # Clear GPU cache periodically
                    if step % 20 == 0:
                        import torch
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                    
                    # Check if there are detections
                    if not result or result.boxes is None or len(result.boxes) == 0:
                        frames_without_detections += 1
                        successful_frames += 1
                        # Save empty annotations
                        if self.project_path and self.current_video_id:
                            self._queue_tracked_bee_annotations(next_frame_idx, [])
                        continue
                    
                    frames_with_detections += 1
                    
                    # Extract bboxes and track IDs from Ultralytics tracker
                    boxes = result.boxes.xyxy.cpu().numpy()
                    confidences = result.boxes.conf.cpu().numpy()
                    
                    # Get track IDs (Ultralytics assigns these)
                    if hasattr(result.boxes, 'id') and result.boxes.id is not None:
                        track_ids = result.boxes.id.cpu().numpy().astype(int)
                    else:
                        # Fallback: use sequential IDs if tracking failed
                        track_ids = list(range(1, len(boxes) + 1))
                    
                    # Build bbox annotations directly
                    bbox_annotations = []
                    if self.current_video_id not in self.video_mask_colors:
                        self.video_mask_colors[self.current_video_id] = {}
                    
                    for i in range(len(boxes)):
                        # Get track ID from Ultralytics
                        track_id = int(track_ids[i])
                        
                        # Ensure we have a color for this track
                        if track_id not in self.video_mask_colors[self.current_video_id]:
                            self.video_mask_colors[self.current_video_id][track_id] = tuple(np.random.randint(0, 255, 3).tolist())
                        
                        # Convert bbox from xyxy to xywh
                        x1, y1, x2, y2 = boxes[i]
                        bbox_xywh = [int(x1), int(y1), int(x2 - x1), int(y2 - y1)]
                        
                        # Build annotation dict
                        annotation = {
                            'mask_id': track_id,
                            'instance_id': track_id,
                            'bbox': bbox_xywh,
                            'confidence': float(confidences[i]),
                            'source': 'yolo_bbox_tracked',
                            'bbox_only': True,
                            'color': self.video_mask_colors[self.current_video_id][track_id]
                        }
                        bbox_annotations.append(annotation)
                    
                    # Update next_mask_id
                    if len(track_ids) > 0:
                        max_track_id = max(track_ids)
                        self.video_next_mask_id[self.current_video_id] = max_track_id + 1
                    
                    # Save annotations directly (no canvas interaction during batch)
                    if self.project_path and self.current_video_id:
                        self._queue_tracked_bee_annotations(next_frame_idx, bbox_annotations)
                    
                    successful_frames += 1
                    
                except Exception as e:
                    print(f"Error processing frame {next_frame_idx}: {e}")
                    import traceback
                    traceback.print_exc()
                    failed_frames += 1
                    
                    # Ask user if they want to continue
                    reply = QMessageBox.question(
                        self,
                        "Propagation Error",
                        f"Error at frame {next_frame_idx}:\n{str(e)}\n\nContinue propagating?",
                        QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
                    )
                    
                    if reply == QMessageBox.StandardButton.No:
                        break
            
            # Update final progress
            progress.setValue(num_frames)
            
            # Load the final frame to show results
            if successful_frames > 0:
                self.load_frame(target_idx)
                QApplication.processEvents()
            
        finally:
            progress.close()
            
            # Clean up temporary tracker config file
            import os
            try:
                os.unlink(tracker_config_path)
                print(f"Cleaned up temporary tracker config: {tracker_config_path}")
            except Exception as e:
                print(f"Warning: Failed to delete temp tracker config: {e}")
        
        # Show completion message
        if not progress.wasCanceled():
            QMessageBox.information(
                self,
                "Propagation Complete",
                f"YOLO BBox propagation complete with ByteTrack!\n\n"
                f"Successfully processed: {successful_frames} frames\n"
                f"Failed: {failed_frames} frames\n\n"
                f"Frames with detections: {frames_with_detections}\n"
                f"Frames without detections: {frames_without_detections}\n\n"
                f"Current frame: {self.current_frame_idx}"
            )
            self.status_label.setText(
                f"✓ Propagated through {successful_frames} frames ({frames_with_detections} with detections)"
            )
        
        # Update instance list
        self.update_instance_list_from_canvas()
    
    def propagate_yolo(self):
        """Propagate YOLO segmentation detections through video frames"""
        # Check if model is loaded
        if not self.yolo_toolbar.is_model_loaded():
            QMessageBox.warning(
                self, "No Model",
                "Please load a YOLO checkpoint first."
            )
            return
        
        # Check if we have frames
        if not self.frames:
            QMessageBox.warning(
                self, "No Video",
                "No video is currently loaded."
            )
            return
        
        # Note: With Ultralytics ByteTrack, we don't need initial annotations
        # ByteTrack will maintain consistent IDs automatically
        current_annotations = self.canvas.get_annotations()
        
        # Ask user for target frame
        from PyQt6.QtWidgets import QInputDialog
        current_idx = self.current_frame_idx
        max_idx = len(self.frames) - 1
        
        target_idx, ok = QInputDialog.getInt(
            self,
            "Propagate YOLO Detections",
            f"Propagate from current frame ({current_idx}) to which frame?\n\n"
            f"Enter target frame index (current: {current_idx}, max: {max_idx}):",
            value=min(current_idx + 100, max_idx),  # Default to 100 frames ahead
            min=current_idx + 1,
            max=max_idx,
            step=1
        )
        
        if not ok:
            return
        
        # Calculate number of frames to propagate through
        num_frames = target_idx - current_idx
        
        # Confirm with user
        reply = QMessageBox.question(
            self,
            "Confirm Propagation",
            f"This will run YOLO segmentation on {num_frames} frames:\n"
            f"  From: Frame {current_idx + 1}\n"
            f"  To: Frame {target_idx}\n\n"
            f"Using Ultralytics built-in ByteTrack for consistent IDs.\n\n"
            f"Continue?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        
        if reply == QMessageBox.StandardButton.No:
            return
        
        # Create progress dialog
        from PyQt6.QtWidgets import QProgressDialog
        progress = QProgressDialog(
            f"Propagating YOLO detections through {num_frames} frames...",
            "Cancel",
            0,
            num_frames,
            self
        )
        progress.setWindowTitle("YOLO Propagation")
        progress.setWindowModality(Qt.WindowModality.WindowModal)
        progress.setMinimumDuration(0)
        progress.setValue(0)
        
        # Store original frame index to restore if cancelled
        original_frame_idx = current_idx
        
        # Track success/failure counts
        successful_frames = 0
        failed_frames = 0
        frames_with_detections = 0
        frames_without_detections = 0
        
        try:
            # Get YOLO model
            model = self.yolo_toolbar.get_model()
            
            print(f"Starting Ultralytics built-in tracker on {num_frames} frames...")
            print(f"  Frames: {current_idx + 1} to {target_idx}")
            
            # Process frame by frame using model.track() with persist=True
            # This maintains tracking state while processing one frame at a time (memory-efficient)
            for step in range(num_frames):
                # Check if user cancelled
                if progress.wasCanceled():
                    self.status_label.setText(f"Propagation cancelled at frame {self.current_frame_idx}")
                    break
                
                next_frame_idx = current_idx + step + 1
                
                # Update progress every N frames to reduce UI overhead
                if step % 5 == 0 or step == num_frames - 1:
                    progress.setValue(step)
                    progress.setLabelText(
                        f"Processing frame {next_frame_idx} of {target_idx}\n"
                        f"({step + 1} / {num_frames})\n\n"
                        f"Detections: {frames_with_detections} frames\n"
                        f"No detections: {frames_without_detections} frames"
                    )
                    QApplication.processEvents()
                
                    QApplication.processEvents()
                
                try:
                    # Run YOLO tracking on this frame
                    # persist=True maintains tracker state across calls
                    frame_path = str(self.frames[next_frame_idx])
                    results = model.track(
                        source=frame_path,
                        conf=0.5,
                        iou=0.5,
                        retina_masks=True,
                        verbose=False,
                        persist=True,  # Persist tracker state across frames
                        tracker='bytetrack.yaml'
                    )
                    
                    # Get the result (single frame)
                    result = results[0] if results else None
                    
                    # Clear GPU cache periodically to prevent memory buildup
                    if step % 20 == 0:
                        import torch
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                    
                    # Check if there are detections
                    if not result or result.masks is None or len(result.masks) == 0:
                        frames_without_detections += 1
                        successful_frames += 1
                        # Save empty annotations directly (no canvas interaction)
                        if self.project_path and self.current_video_id:
                            self._queue_tracked_bee_annotations(next_frame_idx, [])
                        continue
                    
                    frames_with_detections += 1
                    
                    # Extract masks and track IDs from Ultralytics tracker
                    masks = result.masks.data.cpu().numpy()
                    boxes = result.boxes.xyxy.cpu().numpy()
                    confidences = result.boxes.conf.cpu().numpy()
                    
                    # Get track IDs (Ultralytics assigns these)
                    if hasattr(result.boxes, 'id') and result.boxes.id is not None:
                        track_ids = result.boxes.id.cpu().numpy().astype(int)
                    else:
                        # Fallback: use sequential IDs if tracking failed
                        track_ids = list(range(1, len(masks) + 1))
                    
                    # Get original image shape
                    orig_shape = result.orig_shape  # (height, width)
                    
                    # Build annotations directly without using canvas (faster, no UI overhead)
                    frame_annotations = []
                    if self.current_video_id not in self.video_mask_colors:
                        self.video_mask_colors[self.current_video_id] = {}
                    
                    for i in range(len(masks)):
                        # Convert mask to uint8
                        mask = masks[i]
                        if mask.shape != orig_shape:
                            print(f"WARNING: Mask shape {mask.shape} != orig_shape {orig_shape}")
                        mask = (mask * 255).astype(np.uint8)
                        
                        # Get track ID from Ultralytics
                        track_id = int(track_ids[i])
                        
                        # Ensure we have a color for this track
                        if track_id not in self.video_mask_colors[self.current_video_id]:
                            self.video_mask_colors[self.current_video_id][track_id] = tuple(np.random.randint(0, 255, 3).tolist())
                        
                        # Calculate bbox from mask
                        coords = np.where(mask > 0)
                        if len(coords[0]) > 0:
                            y_min, y_max = coords[0].min(), coords[0].max()
                            x_min, x_max = coords[1].min(), coords[1].max()
                            bbox = [int(x_min), int(y_min), int(x_max - x_min + 1), int(y_max - y_min + 1)]
                        else:
                            bbox = [0, 0, 0, 0]
                        
                        # Build annotation dict
                        annotation = {
                            'mask': mask,
                            'mask_id': track_id,
                            'instance_id': track_id,
                            'bbox': bbox,
                            'confidence': float(confidences[i]),
                            'source': 'yolo_tracked',
                            'color': self.video_mask_colors[self.current_video_id][track_id]
                        }
                        frame_annotations.append(annotation)
                    
                    # Update next_mask_id
                    if len(track_ids) > 0:
                        max_track_id = max(track_ids)
                        self.video_next_mask_id[self.current_video_id] = max_track_id + 1
                    
                    # Save annotations directly (no canvas interaction)
                    if self.project_path and self.current_video_id:
                        self._queue_tracked_bee_annotations(next_frame_idx, frame_annotations)
                    
                    successful_frames += 1
                    
                except Exception as e:
                    print(f"Error processing frame {next_frame_idx}: {e}")
                    import traceback
                    traceback.print_exc()
                    failed_frames += 1
                    
                    # Ask user if they want to continue
                    reply = QMessageBox.question(
                        self,
                        "Propagation Error",
                        f"Error at frame {next_frame_idx}:\n{str(e)}\n\nContinue propagating?",
                        QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
                    )
                    
                    if reply == QMessageBox.StandardButton.No:
                        break
            
            # Update final progress
            progress.setValue(num_frames)
            
            # Load the final frame to show the user the end result
            if successful_frames > 0:
                self.load_frame(target_idx)
                QApplication.processEvents()
            
        finally:
            progress.close()
        
        # Show completion message
        if not progress.wasCanceled():
            QMessageBox.information(
                self,
                "Propagation Complete",
                f"YOLO propagation complete with Ultralytics ByteTrack!\n\n"
                f"Successfully processed: {successful_frames} frames\n"
                f"Failed: {failed_frames} frames\n\n"
                f"Frames with detections: {frames_with_detections}\n"
                f"Frames without detections: {frames_without_detections}\n\n"
                f"Current frame: {self.current_frame_idx}"
            )
            self.status_label.setText(
                f"✓ Propagated through {successful_frames} frames ({frames_with_detections} with detections)"
            )
        
        # Update instance list
        self.update_instance_list_from_canvas()
    
    def process_entire_video_yolo(self):
        """Process entire video from frame 0 to end with YOLO segmentation"""
        # Check if model is loaded
        if not self.yolo_toolbar.is_model_loaded():
            QMessageBox.warning(
                self, "No Model",
                "Please load a YOLO checkpoint first."
            )
            return
        
        # Check if we have frames
        if not self.frames:
            QMessageBox.warning(
                self, "No Video",
                "No video is currently loaded."
            )
            return
        
        # Always process entire video from frame 0
        start_idx = 0
        max_idx = len(self.frames) - 1
        num_frames = len(self.frames)
        
        # Show tracker configuration dialog
        from PyQt6.QtWidgets import QDialog, QVBoxLayout, QHBoxLayout, QLabel, QComboBox, QCheckBox, QPushButton, QDialogButtonBox
        
        dialog = QDialog(self)
        dialog.setWindowTitle("Process Entire Video")
        layout = QVBoxLayout(dialog)
        
        # Info label
        info_label = QLabel(
            f"This will run YOLO segmentation on the entire video:\n"
            f"  From: Frame 0\n"
            f"  To: Frame {max_idx}\n"
            f"  Total frames: {num_frames}\n\n"
            f"Configure tracking options below:"
        )
        layout.addWidget(info_label)
        
        # Tracker selection
        tracker_layout = QHBoxLayout()
        tracker_layout.addWidget(QLabel("Tracker:"))
        tracker_combo = QComboBox()
        tracker_combo.addItems(["ByteTrack", "BotSort"])
        tracker_combo.setCurrentIndex(0)  # Default to ByteTrack
        tracker_combo.setToolTip(
            "ByteTrack: Fast, motion-based tracking\n"
            "BotSort: More accurate, supports ReID"
        )
        tracker_layout.addWidget(tracker_combo)
        tracker_layout.addStretch()
        layout.addLayout(tracker_layout)
        
        # ReID checkbox (only for BotSort)
        reid_checkbox = QCheckBox("Enable ReID (Re-Identification)")
        reid_checkbox.setChecked(False)
        reid_checkbox.setToolTip("Use appearance features for better tracking (slower, only BotSort)")
        reid_checkbox.setEnabled(False)  # Initially disabled
        layout.addWidget(reid_checkbox)
        
        # Enable/disable ReID based on tracker selection
        def on_tracker_changed(index):
            is_botsort = tracker_combo.currentText() == "BotSort"
            reid_checkbox.setEnabled(is_botsort)
            if not is_botsort:
                reid_checkbox.setChecked(False)
        tracker_combo.currentIndexChanged.connect(on_tracker_changed)
        
        # Buttons
        button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        button_box.accepted.connect(dialog.accept)
        button_box.rejected.connect(dialog.reject)
        layout.addWidget(button_box)
        
        # Show dialog
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        
        # Get selected options
        tracker_name = tracker_combo.currentText()
        use_reid = reid_checkbox.isChecked()
        
        # Create custom tracker config if needed
        import tempfile
        import yaml
        
        tracker_config_path = None
        if use_reid and tracker_name == "BotSort":
            # Create custom BotSort config with ReID enabled
            botsort_config = {
                'tracker_type': 'botsort',
                'track_high_thresh': 0.25,
                'track_low_thresh': 0.1,
                'new_track_thresh': 0.25,
                'track_buffer': 30,
                'match_thresh': 0.8,
                'fuse_score': True,
                'gmc_method': 'sparseOptFlow',
                'proximity_thresh': 0.5,
                'appearance_thresh': 0.8,
                'with_reid': True,
                'model': 'auto'
            }
            # Create temp file for custom config
            with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
                yaml.dump(botsort_config, f)
                tracker_config_path = f.name
            print(f"Created custom BotSort config with ReID at: {tracker_config_path}")
        elif tracker_name == "BotSort":
            # Use default BotSort but ensure consistent thresholds
            botsort_config = {
                'tracker_type': 'botsort',
                'track_high_thresh': 0.25,
                'track_low_thresh': 0.1,
                'new_track_thresh': 0.25,
                'track_buffer': 30,
                'match_thresh': 0.8,
                'fuse_score': True,
                'gmc_method': 'sparseOptFlow',
                'proximity_thresh': 0.5,
                'appearance_thresh': 0.8,
                'with_reid': False,
            }
            with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
                yaml.dump(botsort_config, f)
                tracker_config_path = f.name
            print(f"Created custom BotSort config at: {tracker_config_path}")
        else:
            # ByteTrack - use custom config with matching thresholds
            bytetrack_config = {
                'tracker_type': 'bytetrack',
                'track_high_thresh': 0.2,
                'track_low_thresh': 0.1,
                'new_track_thresh': 0.5,
                'track_buffer': 30,
                'match_thresh': 0.8,
                'fuse_score': True,
            }
            with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
                yaml.dump(bytetrack_config, f)
                tracker_config_path = f.name
            print(f"Created custom ByteTrack config at: {tracker_config_path}")
        
        # Create progress dialog
        from PyQt6.QtWidgets import QProgressDialog
        progress = QProgressDialog(
            f"Processing entire video ({num_frames} frames)...",
            "Cancel",
            0,
            num_frames,
            self
        )
        progress.setWindowTitle("Process Entire Video")
        progress.setWindowModality(Qt.WindowModality.WindowModal)
        progress.setMinimumDuration(0)
        progress.setValue(0)
        
        # Track success/failure counts
        successful_frames = 0
        failed_frames = 0
        frames_with_detections = 0
        frames_without_detections = 0
        
        try:
            # Get YOLO model
            model = self.yolo_toolbar.get_model()
            
            # Initialize video_next_mask_id if not present (e.g., after deleting annotations)
            if self.current_video_id not in self.video_next_mask_id:
                # Will start from 1 for a fresh video with no annotations
                self.video_next_mask_id[self.current_video_id] = 1
            
            print(f"Starting Ultralytics built-in tracker on {num_frames} frames...")
            print(f"  Frames: 0 to {max_idx}")
            print(f"  Tracker: {tracker_name}")
            print(f"  ReID: {use_reid}")
            print(f"  Config: {tracker_config_path}")
            
            # Process frame by frame using model.track() with persist=True
            # This maintains tracking state while processing one frame at a time (memory-efficient)
            for step in range(num_frames):
                # Check if user cancelled
                if progress.wasCanceled():
                    self.status_label.setText(f"Processing cancelled at frame {step}")
                    break
                
                next_frame_idx = start_idx + step
                
                # Update progress every N frames to reduce UI overhead
                if step % 5 == 0 or step == num_frames - 1:
                    progress.setValue(step)
                    progress.setLabelText(
                        f"Processing frame {next_frame_idx} of {max_idx}\n"
                        f"({step + 1} / {num_frames})\n\n"
                        f"Detections: {frames_with_detections} frames\n"
                        f"No detections: {frames_without_detections} frames"
                    )
                    QApplication.processEvents()
                
                try:
                    # Run YOLO tracking on this frame
                    # persist=True maintains tracker state across calls
                    frame_path = str(self.frames[next_frame_idx])
                    
                    results = model.track(
                        source=frame_path,
                        conf=0.5,
                        iou=0.5,
                        retina_masks=True,
                        verbose=False,
                        persist=True,
                        tracker=tracker_config_path
                    )
                    
                    # Debug: print detection count on first frame
                    if step == 0:
                        result = results[0] if results else None
                        num_detections = len(result.masks) if result and result.masks is not None else 0
                        print(f"  First frame detections: {num_detections}")
                    
                    # Get the result (single frame)
                    result = results[0] if results else None
                    
                    # Clear GPU cache periodically to prevent memory buildup
                    if step % 20 == 0:
                        import torch
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                    
                    # Check if there are detections
                    if not result or result.masks is None or len(result.masks) == 0:
                        frames_without_detections += 1
                        successful_frames += 1
                        # Save empty annotations directly (no canvas interaction)
                        if self.project_path and self.current_video_id:
                            self._queue_tracked_bee_annotations(next_frame_idx, [])
                        continue
                    
                    frames_with_detections += 1
                    
                    # Extract masks and track IDs from Ultralytics tracker
                    masks = result.masks.data.cpu().numpy()
                    boxes = result.boxes.xyxy.cpu().numpy()
                    confidences = result.boxes.conf.cpu().numpy()
                    
                    # Get track IDs (Ultralytics assigns these)
                    if hasattr(result.boxes, 'id') and result.boxes.id is not None:
                        track_ids = result.boxes.id.cpu().numpy().astype(int)
                    else:
                        # Fallback: use sequential IDs if tracking failed
                        track_ids = list(range(1, len(masks) + 1))
                    
                    # Get original image shape
                    orig_shape = result.orig_shape  # (height, width)
                    
                    # Build annotations directly without using canvas (faster, no UI overhead)
                    frame_annotations = []
                    if self.current_video_id not in self.video_mask_colors:
                        self.video_mask_colors[self.current_video_id] = {}
                    
                    for i in range(len(masks)):
                        # Convert mask to uint8
                        mask = masks[i]
                        if mask.shape != orig_shape:
                            print(f"WARNING: Mask shape {mask.shape} != orig_shape {orig_shape}")
                        mask = (mask * 255).astype(np.uint8)
                        
                        # Get track ID from Ultralytics
                        track_id = int(track_ids[i])
                        
                        # Ensure we have a color for this track
                        if track_id not in self.video_mask_colors[self.current_video_id]:
                            self.video_mask_colors[self.current_video_id][track_id] = tuple(np.random.randint(0, 255, 3).tolist())
                        
                        # Calculate bbox from mask
                        coords = np.where(mask > 0)
                        if len(coords[0]) > 0:
                            y_min, y_max = coords[0].min(), coords[0].max()
                            x_min, x_max = coords[1].min(), coords[1].max()
                            bbox = [int(x_min), int(y_min), int(x_max - x_min + 1), int(y_max - y_min + 1)]
                        else:
                            bbox = [0, 0, 0, 0]
                        
                        # Build annotation dict
                        annotation = {
                            'mask': mask,
                            'mask_id': track_id,
                            'instance_id': track_id,
                            'bbox': bbox,
                            'confidence': float(confidences[i]),
                            'source': 'yolo_tracked',
                            'color': self.video_mask_colors[self.current_video_id][track_id]
                        }
                        frame_annotations.append(annotation)
                    
                    # Update next_mask_id
                    if len(track_ids) > 0:
                        max_track_id = max(track_ids)
                        self.video_next_mask_id[self.current_video_id] = max_track_id + 1
                    
                    # Save annotations directly (no canvas interaction)
                    if self.project_path and self.current_video_id:
                        self._queue_tracked_bee_annotations(next_frame_idx, frame_annotations)
                    
                    successful_frames += 1
                    
                except Exception as e:
                    print(f"Error processing frame {next_frame_idx}: {e}")
                    import traceback
                    traceback.print_exc()
                    failed_frames += 1
                    
                    # Ask user if they want to continue
                    reply = QMessageBox.question(
                        self,
                        "Processing Error",
                        f"Error at frame {next_frame_idx}:\n{str(e)}\n\nContinue processing?",
                        QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
                    )
                    
                    if reply == QMessageBox.StandardButton.No:
                        break
            
            # Update final progress
            progress.setValue(num_frames)
            
            # Save max_mask_id to metadata for faster loading next time
            if self.current_video_id and self.current_video_id in self.video_next_mask_id:
                max_id = self.video_next_mask_id[self.current_video_id] - 1
                self._save_max_mask_id_to_metadata(self.current_video_id, max_id)
                print(f"Saved max_mask_id={max_id} to metadata")
            
            # Load the final frame to show the user the end result
            if successful_frames > 0:
                self.load_frame(max_idx)
                QApplication.processEvents()
            
        finally:
            progress.close()
            
            # Clean up temporary tracker config file
            if tracker_config_path:
                import os
                try:
                    os.unlink(tracker_config_path)
                    print(f"Cleaned up temporary tracker config: {tracker_config_path}")
                except Exception as e:
                    print(f"Warning: Failed to delete temp tracker config: {e}")
        
        # Show completion message
        if not progress.wasCanceled():
            reid_status = " (with ReID)" if use_reid else ""
            QMessageBox.information(
                self,
                "Processing Complete",
                f"Entire video processed with {tracker_name}{reid_status}!\n\n"
                f"Successfully processed: {successful_frames} frames\n"
                f"Failed: {failed_frames} frames\n\n"
                f"Frames with detections: {frames_with_detections}\n"
                f"Frames without detections: {frames_without_detections}\n\n"
                f"Current frame: {self.current_frame_idx}"
            )
            self.status_label.setText(
                f"✓ Processed entire video: {successful_frames} frames ({frames_with_detections} with detections)"
            )
        
        # Update instance list
        self.update_instance_list_from_canvas()
    
    def refine_selected_instance_focused(self):
        """Refine the currently selected instance using instance-focused YOLO model"""
        if not self.yolo_instance_focused_toolbar.is_model_loaded():
            QMessageBox.warning(
                self, "No Model",
                "Please load an instance-focused YOLO checkpoint first."
            )
            return
        
        if self.canvas.selected_mask_idx < 0:
            QMessageBox.warning(
                self, "No Instance Selected",
                "Please select an instance to refine.\n\n"
                "Click on an instance in the canvas or select one from the instance list."
            )
            return
        
        if not self.frames or self.current_frame_idx >= len(self.frames):
            QMessageBox.warning(
                self, "No Frame",
                "No frame is currently loaded."
            )
            return
        
        try:
            selected_idx = self.canvas.selected_mask_idx
            
            self.status_label.setText("Running instance-focused YOLO refinement...")
            QApplication.processEvents()
            
            success, message, stats = self._refine_instance_focused_by_index(selected_idx)
            
            if success:
                # Update visualization
                self.canvas.update_mask_visualization(selected_idx)
                self.canvas.annotation_changed.emit()
                
                # Show success message
                source_hint = " (from bbox)" if stats.get('from_bbox', False) else ""
                self.status_label.setText(
                    f"✓ Instance refined{source_hint} (confidence: {stats['confidence']:.3f}, "
                    f"area: {stats['refined_area']}px, Δ{stats['area_diff']:+d}px)"
                )
            else:
                # Show failure message
                self.status_label.setText(f"Refinement failed: {message}")
                
                if "No detections" in message:
                    QMessageBox.information(
                        self, "No Detections",
                        "Instance-focused YOLO did not detect the bee in the crop.\n\n"
                        "Try adjusting the crop padding or using a different model."
                    )
                else:
                    QMessageBox.warning(
                        self, "Refinement Failed",
                        f"Failed to refine instance:\n{message}"
                    )
            
        except ImportError as e:
            QMessageBox.critical(
                self, "OpenCV Required",
                f"OpenCV (cv2) is required for YOLO refinement but not installed:\n{str(e)}"
            )
        except Exception as e:
            import traceback
            error_msg = traceback.format_exc()
            QMessageBox.critical(
                self, "Error",
                f"An error occurred during refinement:\n{str(e)}\n\n{error_msg}"
            )
            self.status_label.setText("Refinement failed")
        finally:
            # Return focus to canvas so spacebar can toggle masks
            self.canvas.setFocus()
    
    def _get_instance_focused_prediction(self, mask_idx):
        """
        Get YOLO prediction for an instance without applying it (for overlap resolution).
        
        Args:
            mask_idx: Instance ID (mask_id) to predict
            
        Returns:
            Tuple of (success: bool, message: str, pred_data: dict)
            pred_data contains:
                - mask_probabilities: np.array of per-pixel probabilities (H x W)
                - detection_confidence: float, overall detection confidence
                - from_bbox: bool, whether created from bbox-only annotation
                - original_area: int, pixel count of original mask
        """
        import cv2
        
        try:
            if mask_idx <= 0:
                return False, "Invalid mask ID", {}
            
            # Check if instance has a segmentation mask or is bbox-only
            has_mask = (
                self.canvas.combined_mask is not None and 
                np.any(self.canvas.combined_mask == mask_idx)
            )
            
            is_bbox_only = (
                mask_idx in self.canvas.annotation_metadata and 
                self.canvas.annotation_metadata[mask_idx].get('bbox_only', False)
            )
            
            # Determine the bounding box to use for cropping
            if has_mask:
                # Instance has segmentation - use mask bbox
                # Check if we're currently editing this instance
                if self.canvas.editing_instance_id == mask_idx and self.canvas.editing_mask is not None:
                    original_mask = self.canvas.editing_mask.copy()
                else:
                    # Extract mask from combined_mask
                    original_mask = (self.canvas.combined_mask == mask_idx).astype(np.uint8) * 255
                
                if np.sum(original_mask > 0) == 0:
                    return False, "Empty mask", {}
                
                # Get bounding box of the mask
                coords = np.where(original_mask > 0)
                if len(coords[0]) == 0:
                    return False, "Empty mask", {}
                
                y_min, y_max = coords[0].min(), coords[0].max()
                x_min, x_max = coords[1].min(), coords[1].max()
                
            elif is_bbox_only:
                # Instance is bbox-only - use bbox from metadata
                bbox = self.canvas.annotation_metadata[mask_idx].get('bbox')
                if not bbox or bbox == [0, 0, 0, 0]:
                    return False, "No valid bbox available", {}
                
                # bbox format is [x, y, width, height]
                x_min, y_min, bbox_w, bbox_h = bbox
                x_max = x_min + bbox_w
                y_max = y_min + bbox_h
                
                # Create empty mask for comparison
                h, w = self.canvas.current_image.shape[:2]
                original_mask = np.zeros((h, w), dtype=np.uint8)
                
            else:
                # Check if there's a bbox in metadata (computed from mask)
                # This handles cases where segmentation exists but isn't loaded in combined_mask
                if mask_idx in self.canvas.annotation_metadata:
                    bbox = self.canvas.annotation_metadata[mask_idx].get('bbox')
                    if bbox and bbox != [0, 0, 0, 0]:
                        # bbox format is [x, y, width, height]
                        x_min, y_min, bbox_w, bbox_h = bbox
                        x_max = x_min + bbox_w
                        y_max = y_min + bbox_h
                        
                        # Create empty mask
                        h, w = self.canvas.current_image.shape[:2]
                        original_mask = np.zeros((h, w), dtype=np.uint8)
                    else:
                        return False, "No valid bbox available", {}
                else:
                    return False, "No mask or bbox found for instance", {}
            
            # Get padding
            padding = self.yolo_instance_focused_toolbar.get_padding()
            
            # Add padding to bounding box
            h, w = self.canvas.current_image.shape[:2]
            crop_y1 = max(0, int(y_min) - padding)
            crop_y2 = min(h, int(y_max) + padding + 1)
            crop_x1 = max(0, int(x_min) - padding)
            crop_x2 = min(w, int(x_max) + padding + 1)
            
            # Crop the image
            crop = self.canvas.current_image[crop_y1:crop_y2, crop_x1:crop_x2]
            
            if crop.size == 0:
                return False, "Invalid crop", {}
            
            # Store original crop size
            orig_crop_h, orig_crop_w = crop.shape[:2]
            
            # Resize crop to standard size (640x640)
            target_size = 640
            crop_resized = cv2.resize(crop, (target_size, target_size), interpolation=cv2.INTER_LINEAR)
            
            # Run YOLO on the resized crop
            model = self.yolo_instance_focused_toolbar.get_model()
            
            # Convert crop for YOLO (it expects BGR or 3-channel)
            if len(crop_resized.shape) == 2:
                crop_bgr = cv2.cvtColor(crop_resized, cv2.COLOR_GRAY2BGR)
            else:
                crop_bgr = cv2.cvtColor(crop_resized, cv2.COLOR_RGB2BGR)
            
            results = model.predict(
                source=crop_bgr,
                conf=0.5,
                iou=0.45,
                augment=False,
                retina_masks=True,
                verbose=False,
            )
            
            if not results or len(results) == 0 or results[0].masks is None or len(results[0].masks) == 0:
                return False, "No detections", {}
            
            result = results[0]
            
            # Take the first (highest confidence) prediction
            if len(result.masks) > 0:
                best_pred_idx = 0
                detection_confidence = float(result.boxes[0].conf[0]) if result.boxes is not None else 0.0
                
                # Get the predicted mask probabilities (BEFORE thresholding at 0.5)
                # This is the raw probability map from YOLO
                pred_mask_probs = result.masks[best_pred_idx].data[0].cpu().numpy()
                
                # Resize mask probabilities back to original crop size
                pred_mask_probs_resized = cv2.resize(
                    pred_mask_probs,
                    (orig_crop_w, orig_crop_h),
                    interpolation=cv2.INTER_LINEAR  # Use LINEAR for probabilities
                )
                
                # Extract largest contour at 0.5 threshold for cleaning
                pred_mask_binary = (pred_mask_probs_resized > 0.5).astype('uint8') * 255
                contours, _ = cv2.findContours(pred_mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                if len(contours) > 0:
                    # Find the largest contour
                    largest_contour = max(contours, key=cv2.contourArea)
                    
                    # Create mask of largest contour
                    contour_mask = np.zeros_like(pred_mask_binary)
                    cv2.drawContours(contour_mask, [largest_contour], 0, 255, -1)
                    
                    # Apply contour mask to probabilities (zero out probabilities outside largest contour)
                    pred_mask_probs_clean = pred_mask_probs_resized * (contour_mask > 0)
                else:
                    pred_mask_probs_clean = pred_mask_probs_resized
                
                # Create full-size probability map
                h, w = self.canvas.current_image.shape[:2]
                full_mask_probs = np.zeros((h, w), dtype=np.float32)
                full_mask_probs[crop_y1:crop_y2, crop_x1:crop_x2] = pred_mask_probs_clean
                
                # Calculate original area
                original_area = np.sum(original_mask > 0)
                
                pred_data = {
                    'mask_probabilities': full_mask_probs,
                    'detection_confidence': detection_confidence,
                    'from_bbox': is_bbox_only,
                    'original_area': original_area
                }
                
                return True, "Success", pred_data
            else:
                return False, "No predictions", {}
                
        except Exception as e:
            import traceback
            error_msg = traceback.format_exc()
            print(f"Error in _get_instance_focused_prediction: {error_msg}")
            return False, f"Error: {str(e)}", {}
    
    def _refine_instance_focused_by_index(self, mask_idx):
        """
        Refine a mask using instance-focused YOLO (resized crop, single instance).
        Can work with either existing segmentation masks or bbox-only annotations.
        
        Args:
            mask_idx: Instance ID (mask_id) of the mask to refine
            
        Returns:
            Tuple of (success: bool, message: str, stats: dict)
        """
        import cv2
        
        try:
            if mask_idx <= 0:
                return False, "Invalid mask ID", {}
            
            # Check if instance has a segmentation mask or is bbox-only
            has_mask = (
                self.canvas.combined_mask is not None and 
                np.any(self.canvas.combined_mask == mask_idx)
            )
            
            is_bbox_only = (
                mask_idx in self.canvas.annotation_metadata and 
                self.canvas.annotation_metadata[mask_idx].get('bbox_only', False)
            )
            
            # Determine the bounding box to use for cropping
            if has_mask:
                # Instance has segmentation - use mask bbox
                # Check if we're currently editing this instance
                if self.canvas.editing_instance_id == mask_idx and self.canvas.editing_mask is not None:
                    # Get mask from editing layer
                    original_mask = self.canvas.editing_mask.copy()
                else:
                    # Not currently editing - need to start editing mode first
                    self.canvas.start_editing_instance(mask_idx)
                    # Now get from editing mask
                    original_mask = self.canvas.editing_mask.copy()
                
                if np.sum(original_mask > 0) == 0:
                    return False, "Empty mask", {}
                
                # Get bounding box of the mask
                coords = np.where(original_mask > 0)
                if len(coords[0]) == 0:
                    return False, "Empty mask", {}
                
                y_min, y_max = coords[0].min(), coords[0].max()
                x_min, x_max = coords[1].min(), coords[1].max()
                
                print(f"\n=== Instance-Focused YOLO Refinement (from mask) ===")
                
            elif is_bbox_only:
                # Instance is bbox-only - use bbox from metadata
                bbox = self.canvas.annotation_metadata[mask_idx].get('bbox')
                if not bbox or bbox == [0, 0, 0, 0]:
                    return False, "No valid bbox available", {}
                
                # bbox format is [x, y, width, height]
                x_min, y_min, bbox_w, bbox_h = bbox
                x_max = x_min + bbox_w
                y_max = y_min + bbox_h
                
                # Create empty mask for comparison (will be filled with prediction)
                h, w = self.canvas.current_image.shape[:2]
                original_mask = np.zeros((h, w), dtype=np.uint8)
                
                # Start editing mode for this instance (creates editing_mask)
                self.canvas.start_editing_instance(mask_idx)
                
                print(f"\n=== Instance-Focused YOLO Refinement (from bbox) ===")
                
            else:
                # Check if there's a bbox in metadata (computed from mask)
                # This handles cases where segmentation exists but isn't loaded in combined_mask
                if mask_idx in self.canvas.annotation_metadata:
                    bbox = self.canvas.annotation_metadata[mask_idx].get('bbox')
                    if bbox and bbox != [0, 0, 0, 0]:
                        # bbox format is [x, y, width, height]
                        x_min, y_min, bbox_w, bbox_h = bbox
                        x_max = x_min + bbox_w
                        y_max = y_min + bbox_h
                        
                        # Create empty mask for comparison
                        h, w = self.canvas.current_image.shape[:2]
                        original_mask = np.zeros((h, w), dtype=np.uint8)
                        
                        # Start editing mode for this instance
                        self.canvas.start_editing_instance(mask_idx)
                        
                        print(f"\n=== Instance-Focused YOLO Refinement (from stored bbox) ===")
                    else:
                        return False, "No valid bbox available", {}
                else:
                    return False, "No mask or bbox found for instance", {}
            
            # Get padding
            padding = self.yolo_instance_focused_toolbar.get_padding()
            
            # Add padding to bounding box
            h, w = self.canvas.current_image.shape[:2]
            crop_y1 = max(0, int(y_min) - padding)
            crop_y2 = min(h, int(y_max) + padding + 1)
            crop_x1 = max(0, int(x_min) - padding)
            crop_x2 = min(w, int(x_max) + padding + 1)
            
            # Crop the image
            crop = self.canvas.current_image[crop_y1:crop_y2, crop_x1:crop_x2]
            
            if crop.size == 0:
                return False, "Invalid crop", {}
            
            # Store original crop size
            orig_crop_h, orig_crop_w = crop.shape[:2]
            
            # Resize crop to standard size (640x640) - this is the key difference from Stage2
            target_size = 640
            crop_resized = cv2.resize(crop, (target_size, target_size), interpolation=cv2.INTER_LINEAR)
            
            print(f"Source: {'mask' if has_mask else 'bbox'}")
            print(f"Bbox: ({int(x_min)}, {int(y_min)}) to ({int(x_max)}, {int(y_max)})")
            print(f"Crop region: ({crop_x1}, {crop_y1}) to ({crop_x2}, {crop_y2})")
            print(f"Original crop size: {orig_crop_w}x{orig_crop_h} (WxH)")
            print(f"Resized to: {target_size}x{target_size}")
            print(f"Padding used: {padding}px")
            
            # Run YOLO on the resized crop
            model = self.yolo_instance_focused_toolbar.get_model()
            
            # Convert crop for YOLO (it expects BGR or 3-channel)
            if len(crop_resized.shape) == 2:
                # Grayscale: convert to BGR
                crop_bgr = cv2.cvtColor(crop_resized, cv2.COLOR_GRAY2BGR)
            else:
                # RGB: convert to BGR
                crop_bgr = cv2.cvtColor(crop_resized, cv2.COLOR_RGB2BGR)
            
            results = model.predict(
                source=crop_bgr,
                conf=0.5,  # Confidence threshold
                iou=0.45,
                augment=False,
                retina_masks=True,
                verbose=False,
            )
            
            if not results or len(results) == 0 or results[0].masks is None or len(results[0].masks) == 0:
                print(f"No YOLO detections in resized crop")
                return False, "No detections", {}
            
            result = results[0]
            print(f"YOLO found {len(result.masks)} predictions in crop")
            
            # Take the first (highest confidence) prediction since we expect only one bee
            if len(result.masks) > 0:
                best_pred_idx = 0  # Highest confidence prediction
                confidence = float(result.boxes[0].conf[0]) if result.boxes is not None else 0.0
                
                print(f"Using prediction 0 (confidence: {confidence:.3f})")
                
                # Get the predicted mask (at target_size x target_size)
                pred_mask = result.masks[best_pred_idx].data[0].cpu().numpy()
                
                # Binarize
                pred_mask_binary = (pred_mask > 0.5).astype('uint8') * 255
                
                # Resize mask back to original crop size
                pred_mask_resized = cv2.resize(
                    pred_mask_binary,
                    (orig_crop_w, orig_crop_h),
                    interpolation=cv2.INTER_NEAREST
                )
                
                print(f"Resized prediction from {pred_mask.shape} back to ({orig_crop_h}, {orig_crop_w})")
                
                # Extract only the largest contour to clean up the mask
                contours, _ = cv2.findContours(pred_mask_resized, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if len(contours) > 0:
                    # Find the largest contour
                    largest_contour = max(contours, key=cv2.contourArea)
                    largest_area = cv2.contourArea(largest_contour)
                    
                    print(f"Found {len(contours)} contour(s), keeping largest (area: {largest_area:.0f}px)")
                    
                    # Create clean mask with only the largest contour
                    pred_mask_clean = np.zeros_like(pred_mask_resized)
                    cv2.drawContours(pred_mask_clean, [largest_contour], 0, 255, -1)
                else:
                    print("Warning: No contours found in predicted mask")
                    pred_mask_clean = pred_mask_resized
                
                # Create full-size mask
                h, w = self.canvas.current_image.shape[:2]
                refined_mask = np.zeros((h, w), dtype=np.uint8)
                refined_mask[crop_y1:crop_y2, crop_x1:crop_x2] = pred_mask_clean
                
                # Update the editing mask directly
                self.canvas.editing_mask = refined_mask
                self.canvas._update_editing_visualization()
                
                # If this was a bbox-only annotation, mark it as no longer bbox-only
                if is_bbox_only and mask_idx in self.canvas.annotation_metadata:
                    self.canvas.annotation_metadata[mask_idx]['bbox_only'] = False
                
                # Calculate stats
                original_area = np.sum(original_mask > 0)
                refined_area = np.sum(refined_mask > 0)
                area_diff = refined_area - original_area
                
                stats = {
                    "confidence": confidence,
                    "original_area": original_area,
                    "refined_area": refined_area,
                    "area_diff": area_diff,
                    "from_bbox": is_bbox_only
                }
                
                return True, "Success", stats
            else:
                return False, "No predictions", {}
                
        except Exception as e:
            import traceback
            error_msg = traceback.format_exc()
            print(f"Error in _refine_instance_focused_by_index: {error_msg}")
            return False, f"Error: {str(e)}", {}
    
    def refine_all_instances_focused(self):
        """Refine all bee instances in current frame using instance-focused YOLO with overlap resolution"""
        if not self.yolo_instance_focused_toolbar.is_model_loaded():
            QMessageBox.warning(
                self, "No Model",
                "Please load an instance-focused YOLO checkpoint first."
            )
            return
        
        if not self.frames or self.current_frame_idx >= len(self.frames):
            QMessageBox.warning(
                self, "No Frame",
                "No frame is currently loaded."
            )
            return
        
        # Collect all instance IDs (both from masks and bbox-only annotations)
        instance_ids = set()
        
        # Get IDs from segmentation masks
        if self.canvas.combined_mask is not None:
            unique_ids = np.unique(self.canvas.combined_mask)
            instance_ids.update([int(id) for id in unique_ids if id > 0])
        
        # Get IDs from bbox-only annotations
        for instance_id, metadata in self.canvas.annotation_metadata.items():
            if metadata.get('bbox_only', False):
                # Check for valid bbox
                bbox = metadata.get('bbox')
                if bbox and bbox != [0, 0, 0, 0]:
                    instance_ids.add(int(instance_id))
        
        # Convert to sorted list
        all_mask_ids = sorted(list(instance_ids))
        
        # Filter to only bee instances (skip hive and chamber)
        mask_ids = []
        skipped_non_bee = 0
        for mask_id in all_mask_ids:
            # Check category in metadata
            category = 'bee'  # default
            if mask_id in self.canvas.annotation_metadata:
                category = self.canvas.annotation_metadata[mask_id].get('category', 'bee')
            
            # Only include bee instances
            if category == 'bee':
                mask_ids.append(mask_id)
            else:
                skipped_non_bee += 1
        
        if len(mask_ids) == 0:
            if skipped_non_bee > 0:
                QMessageBox.information(
                    self, "No Bee Instances",
                    f"No bee instances found in current frame.\n\n"
                    f"Skipped {skipped_non_bee} non-bee annotations (hive/chamber)."
                )
            else:
                QMessageBox.information(
                    self, "No Instances",
                    "No instances found in current frame.\n\n"
                    "You need either segmentation masks or bounding box annotations."
                )
            return
        
        # Ask for confirmation
        confirm_msg = f"Refine all {len(mask_ids)} bee instances in this frame?\n\n"
        confirm_msg += f"This will predict masks for all bee instances, then resolve overlaps\n"
        confirm_msg += f"by choosing the prediction with highest confidence per pixel.\n\n"
        confirm_msg += f"Works with both segmentation masks and bbox-only annotations."
        if skipped_non_bee > 0:
            confirm_msg += f"\n\nNote: Skipping {skipped_non_bee} non-bee annotations (hive/chamber)."
        
        reply = QMessageBox.question(
            self,
            "Refine All Bee Instances",
            confirm_msg,
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        
        if reply != QMessageBox.StandardButton.Yes:
            return
        
        try:
            # Initialize arrays for overlap resolution
            h, w = self.canvas.current_image.shape[:2]
            confidence_map = np.zeros((h, w), dtype=np.float32)  # Best probability per pixel
            instance_map = np.zeros((h, w), dtype=np.uint16)     # Which instance won each pixel
            
            # Collect predictions for all instances
            predictions = {}  # mask_id -> (mask_probabilities, detection_confidence, from_bbox, original_area)
            successes = 0
            failures = 0
            from_bbox_count = 0
            
            print(f"\n{'='*60}")
            print(f"Refining {len(mask_ids)} bee instances with overlap resolution")
            if skipped_non_bee > 0:
                print(f"(Skipped {skipped_non_bee} non-bee annotations)")
            print(f"{'='*60}")
            
            for i, mask_id in enumerate(mask_ids):
                # Update UI less frequently (every 5 instances or at end)
                if i % 5 == 0 or i == len(mask_ids) - 1:
                    self.status_label.setText(f"Predicting instance {i+1}/{len(mask_ids)}...")
                    QApplication.processEvents()
                
                # Get prediction without applying it
                success, message, pred_data = self._get_instance_focused_prediction(mask_id)
                
                if success:
                    predictions[mask_id] = pred_data
                    successes += 1
                    if pred_data['from_bbox']:
                        from_bbox_count += 1
                    print(f"Instance {mask_id}: confidence={pred_data['detection_confidence']:.3f}, "
                          f"pixels={pred_data['mask_probabilities'].sum():.0f}")
                else:
                    failures += 1
                    print(f"Instance {mask_id}: FAILED - {message}")
            
            if len(predictions) == 0:
                QMessageBox.warning(
                    self, "No Predictions",
                    "No successful predictions. All instances failed."
                )
                return
            
            # Resolve overlaps based on per-pixel probabilities
            self.status_label.setText(f"Resolving overlaps ({len(predictions)} predictions)...")
            QApplication.processEvents()
            
            print(f"\nResolving overlaps for {len(predictions)} predictions...")
            overlap_pixels = 0
            
            for mask_id, pred_data in predictions.items():
                mask_probs = pred_data['mask_probabilities']
                
                # Find pixels where this prediction is positive (>0.5)
                mask_pixels = mask_probs > 0.5
                
                # Find pixels where this prediction has higher probability than current best
                better_pixels = mask_pixels & (mask_probs > confidence_map)
                
                # Count overlaps (pixels where we're overwriting another instance)
                overlap_count = np.sum(better_pixels & (instance_map > 0))
                if overlap_count > 0:
                    overlap_pixels += overlap_count
                
                # Update maps where this instance wins
                confidence_map[better_pixels] = mask_probs[better_pixels]
                instance_map[better_pixels] = mask_id
            
            print(f"Resolved {overlap_pixels} overlapping pixels based on probability")
            
            # Apply all predictions to canvas
            self.status_label.setText("Applying refined masks...")
            QApplication.processEvents()
            
            print(f"\nApplying {len(predictions)} refined masks...")
            
            # OPTIMIZED: Direct batch update of separate mask arrays instead of per-instance editing
            # This is much faster than calling start_editing_instance/commit_editing for each instance
            
            # Initialize mask arrays if needed
            if self.canvas.bee_mask is None:
                self.canvas.bee_mask = np.zeros((h, w), dtype=np.int32)
            if self.canvas.chamber_mask is None:
                self.canvas.chamber_mask = np.zeros((h, w), dtype=np.int32)
            if self.canvas.hive_mask is None:
                self.canvas.hive_mask = np.zeros((h, w), dtype=np.int32)
            
            # Clear existing masks for instances that were refined and apply new masks
            for mask_id, pred_data in predictions.items():
                # Determine category for this instance
                category = self.canvas.annotation_metadata.get(mask_id, {}).get('category', 'bee')
                
                # Get the appropriate mask array
                if category == 'chamber':
                    mask_array = self.canvas.chamber_mask
                elif category == 'hive':
                    mask_array = self.canvas.hive_mask
                else:  # Default to bee
                    mask_array = self.canvas.bee_mask
                
                # Clear old mask for this instance
                mask_array[mask_array == mask_id] = 0
                
                # Apply new mask from instance_map
                new_mask_pixels = instance_map == mask_id
                mask_array[new_mask_pixels] = mask_id
            
            # Mark combined mask cache as dirty so it gets regenerated
            self.canvas._combined_mask_dirty = True
            
            # Clear cached instance data
            self.canvas._cached_instance_ids = None
            self.canvas._geometry_cache.invalidate()
            
            # Update metadata and calculate statistics
            total_area_diff = 0
            for mask_id, pred_data in predictions.items():
                # Ensure metadata exists
                if mask_id not in self.canvas.annotation_metadata:
                    self.canvas.annotation_metadata[mask_id] = {}
                
                # Ensure category is set (important for subsequent operations)
                if 'category' not in self.canvas.annotation_metadata[mask_id]:
                    self.canvas.annotation_metadata[mask_id]['category'] = 'bee'
                
                # If this was a bbox-only annotation, mark it as no longer bbox-only
                if pred_data['from_bbox']:
                    self.canvas.annotation_metadata[mask_id]['bbox_only'] = False
                
                # Calculate area difference
                refined_area = np.sum(instance_map == mask_id)
                area_diff = refined_area - pred_data['original_area']
                total_area_diff += area_diff
                
                print(f"Instance {mask_id}: area {pred_data['original_area']} → {refined_area} (Δ{area_diff:+d})")
            
            # Rebuild visualizations ONCE for all changes
            self.canvas.rebuild_visualizations()
            
            # Emit annotation changed ONCE
            self.canvas.annotation_changed.emit()
            
            # Show summary
            summary = f"Refined {successes}/{len(mask_ids)} instances\n"
            if from_bbox_count > 0:
                summary += f"{from_bbox_count} created from bboxes\n"
            if failures > 0:
                summary += f"{failures} failed\n"
            if overlap_pixels > 0:
                summary += f"{overlap_pixels} overlapping pixels resolved\n"
            summary += f"Total area change: {total_area_diff:+d} pixels"
            
            self.status_label.setText(
                f"✓ Refined {successes}/{len(mask_ids)} instances "
                f"({overlap_pixels} overlaps resolved, Δ{total_area_diff:+d}px)"
            )
            
            print(f"\n{'='*60}")
            print(f"Refinement complete!")
            print(f"  Success: {successes}/{len(mask_ids)}")
            print(f"  From bbox: {from_bbox_count}")
            print(f"  Overlaps resolved: {overlap_pixels} pixels")
            print(f"  Total area change: {total_area_diff:+d} pixels")
            print(f"{'='*60}\n")
            
            QMessageBox.information(
                self, "Refinement Complete",
                summary
            )
            
        except Exception as e:
            import traceback
            error_msg = traceback.format_exc()
            QMessageBox.critical(
                self, "Refinement Error",
                f"Error refining instances:\n{str(e)}\n\n{error_msg}"
            )
            self.status_label.setText("Refinement failed")
        finally:
            # Return focus to canvas so spacebar can toggle masks
            self.canvas.setFocus()
            
    def closeEvent(self, event):
        """Handle window close event"""
        should_save_current_frame = False
        has_pending_canvas_edit = (
            hasattr(self, 'canvas')
            and self.canvas.editing_instance_id > 0
            and self.canvas.editing_mask is not None
            and np.any(self.canvas.editing_mask > 0)
        )
        has_unsaved_annotation_changes = (
            getattr(self, 'current_frame_modified', False)
            or bool(getattr(self, 'dirty_frame_annotation_keys', set()))
            or bool(getattr(self, 'dirty_video_annotation_ids', set()))
            or has_pending_canvas_edit
        )

        if has_unsaved_annotation_changes:
            reply = QMessageBox.question(
                self,
                "Save Changes Before Closing?",
                "The current frame has unsaved annotation changes.\n\n"
                "Save these changes before closing?",
                QMessageBox.StandardButton.Save |
                QMessageBox.StandardButton.Discard |
                QMessageBox.StandardButton.Cancel,
                QMessageBox.StandardButton.Save
            )

            if reply == QMessageBox.StandardButton.Cancel:
                event.ignore()
                return
            should_save_current_frame = reply == QMessageBox.StandardButton.Save
        else:
            reply = QMessageBox.question(
                self, "Close Application",
                "Are you sure you want to close the application?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )

            if reply == QMessageBox.StandardButton.No:
                event.ignore()
                return

        # Save current frame before closing only when requested. This keeps an
        # accidental inference pass from being written just because the app exits.
        if (should_save_current_frame and hasattr(self, 'project_path') and self.project_path
                and hasattr(self, 'current_video_id') and self.current_video_id):
            try:
                save_summary = self._save_annotation_sources(refresh_next_id=True)
                if save_summary.get('saved_any'):
                    print(
                        f"Saved {save_summary['video_id']} frame "
                        f"{save_summary['frame_idx']} before closing"
                    )
            except Exception as e:
                print(f"Warning: Failed to save current frame on close: {e}")
        
        # Stop and wait for save worker to finish
        if hasattr(self, 'save_worker'):
            self.save_worker.stop()
            self.save_worker.wait(2000)  # Wait up to 2 seconds
        
        # Stop preload worker
        if hasattr(self, 'preload_worker'):
            self.preload_worker.stop()
            self.preload_worker.join(timeout=1.0)  # Wait up to 1 second
        
        # Save settings
        settings = QSettings()
        settings.setValue('geometry', self.saveGeometry())
        
        # Save last project, video, and frame
        if hasattr(self, 'project_path') and self.project_path:
            settings.setValue('last_project_path', str(self.project_path))
            if hasattr(self, 'current_video_id') and self.current_video_id:
                settings.setValue('last_video_id', self.current_video_id)
            if hasattr(self, 'current_frame_idx'):
                settings.setValue('last_frame_index', self.current_frame_idx)
                # Also save frame index within video
                frame_idx_in_video = self._get_frame_idx_in_video(self.current_frame_idx)
                settings.setValue('last_frame_index_in_video', frame_idx_in_video)
                
        event.accept()
