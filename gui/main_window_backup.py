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
                             QInputDialog, QSlider)
from PyQt6.QtCore import Qt, pyqtSignal, QSettings, QThread, QTimer
from PyQt6.QtGui import QAction, QKeySequence, QActionGroup
from pathlib import Path
import queue
import time

from .canvas import ImageCanvas
from .toolbar import AnnotationToolbar
from .yolo_toolbar import YOLOToolbar
from .yolo_refine_toolbar import YOLORefinementToolbar
from .yolo_instance_focused_toolbar import YOLOInstanceFocusedToolbar
from .yolo_sahi_toolbar import YOLOSAHIToolbar
from .yolo_beehavesque_toolbar import YOLOBeehavesqueToolbar
from .yolo_bbox_toolbar import YOLOBBoxToolbar
from .sam2_toolbar import SAM2Toolbar
from .sam2_training_dialog import SAM2TrainingConfigDialog
from .dialogs import VideoImportDialog, ProjectDialog
from .training_dialog import TrainingConfigDialog, TrainingProgressDialog
from .validation_dialog import ValidationConfigDialog, ValidationProgressDialog
from .validation_worker import ValidationWorker
from core.video_processor import VideoProcessor
from core.annotation import AnnotationManager
from core.project_manager import ProjectManager
from core.instance_tracker import InstanceTracker, Detection, Track
from core.frame_cache import FrameCache, PreloadWorker
from core.marker_detector import MarkerDetector
from training.coco_video_export import export_coco_per_video
from training.yolo_trainer import YOLOTrainingWorker
from training.yolo_trainer_stage2 import YOLOTrainingWorkerStage2
from training.yolo_trainer_instance_focused import YOLOTrainingWorkerInstanceFocused
from training.yolo_trainer_sahi import YOLOTrainingWorkerSAHI
from training.yolo_trainer_beehavesque import YOLOTrainingWorkerBeehavesque
from training.yolo_trainer_bbox import YOLOTrainingWorkerBBox
from training.sam2_trainer import SAM2TrainingWorker


class SaveWorker(QThread):
    """Background worker thread for saving annotations"""
    
    error_occurred = pyqtSignal(str)
    save_started = pyqtSignal(int)  # Emits frame_idx when save starts
    save_completed = pyqtSignal(int)  # Emits frame_idx when save completes
    
    def __init__(self, annotation_manager):
        super().__init__()
        self.annotation_manager = annotation_manager
        self.save_queue = queue.Queue()
        self.running = True
        
    def add_save_task(self, project_path, video_id, frame_idx, annotations):
        """Queue a save task"""
        self.save_queue.put((project_path, video_id, frame_idx, annotations))
        
    def run(self):
        """Process save tasks in background"""
        while self.running:
            try:
                # Wait for save task with timeout
                try:
                    project_path, video_id, frame_idx, annotations = self.save_queue.get(timeout=0.1)
                except queue.Empty:
                    continue
                    
                # Perform the save
                try:
                    self.save_started.emit(frame_idx)
                    self.annotation_manager.save_frame_annotations(
                        project_path, video_id, frame_idx, annotations
                    )
                    self.save_completed.emit(frame_idx)
                except Exception as e:
                    self.error_occurred.emit(f"Save failed for frame {frame_idx}: {str(e)}")
                    
            except Exception as e:
                self.error_occurred.emit(f"Worker error: {str(e)}")
                
    def stop(self):
        """Stop the worker thread"""
        self.running = False


class MainWindow(QMainWindow):
    """Main application window"""
    
    def __init__(self, sam2_checkpoint=None, coarse_yolo_checkpoint=None, fine_yolo_checkpoint=None, beehavesque_checkpoint=None, bbox_checkpoint=None, instance_focused_checkpoint=None):
        super().__init__()
        
        self.video_processor = VideoProcessor()
        self.annotation_manager = AnnotationManager(max_cache_size=2)  # Very small cache - annotations with masks are huge (~100MB/frame)
        self.project_manager = ProjectManager()
        
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
        self.fine_yolo_checkpoint = fine_yolo_checkpoint
        self.beehavesque_checkpoint = beehavesque_checkpoint
        self.bbox_checkpoint = bbox_checkpoint
        self.instance_focused_checkpoint = instance_focused_checkpoint
        
        self.current_frame_idx = 0
        self.frames = []
        self.frame_video_ids = []  # Track which video each frame belongs to
        self.frame_splits = []  # Track split (train/val) for each frame
        self.frame_selected = []  # Track if frame is selected for train/val
        self.frame_list_to_frames_map = []  # Map list row to actual frame index
        self.project_path = None
        self.split_filter = 'all'  # 'all', 'train', 'val', 'test', or 'inference'
        self.video_next_mask_id = {}  # Track next_mask_id per video for unique IDs
        self.video_mask_colors = {}  # Track mask colors per video: {video_id: {mask_id: (r,g,b)}}
        self.box_inference_mode = False  # Track if box inference mode is active
        self.annotation_mode = 'segmentation'  # 'segmentation' or ' bbox' - which annotation type to display/edit
        
        # Track unsaved changes to avoid unnecessary saves
        self.current_frame_modified = False
        
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
        self.canvas.annotation_changed.connect(self.on_annotation_changed)
        self.canvas.annotation_changed.connect(self._schedule_instance_list_update)
        self.canvas.setToolTip("Hold Spacebar to temporarily hide masks & numbers | Press F to fit image to window")
        
        # Create annotation toolbar
        self.toolbar = AnnotationToolbar(self)
        self.toolbar.tool_changed.connect(self.on_tool_changed)
        self.toolbar.brush_size_changed.connect(self.canvas.set_brush_size)
        self.toolbar.mask_opacity_changed.connect(self.canvas.set_mask_opacity)
        self.toolbar.clear_instance_requested.connect(self.clear_selected_instance)
        self.toolbar.new_instance_requested.connect(self.new_instance)
        self.toolbar.resolve_overlaps_requested.connect(self.resolve_overlaps)
        self.toolbar.delete_all_requested.connect(self.delete_all_instances)
        self.toolbar.show_segmentations_changed.connect(self.on_show_segmentations_changed)
        self.toolbar.show_bboxes_changed.connect(self.on_show_bboxes_changed)
        
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
        
        # Create YOLO toolbar
        if self.coarse_yolo_checkpoint:
            print(f"Loading coarse YOLO checkpoint from command line: {self.coarse_yolo_checkpoint}")
        self.yolo_toolbar = YOLOToolbar(self, checkpoint_path=self.coarse_yolo_checkpoint)
        self.yolo_toolbar.inference_requested.connect(self.run_yolo_inference)
        self.yolo_toolbar.track_from_last_requested.connect(self.track_from_last_frame)
        self.yolo_toolbar.propagate_requested.connect(self.propagate_yolo)
        self.yolo_toolbar.process_video_requested.connect(self.process_entire_video_yolo)
        
        # Create YOLO refinement toolbar
        if self.fine_yolo_checkpoint:
            print(f"Loading fine YOLO checkpoint from command line: {self.fine_yolo_checkpoint}")
        self.yolo_refine_toolbar = YOLORefinementToolbar(self, checkpoint_path=self.fine_yolo_checkpoint)
        self.yolo_refine_toolbar.refine_requested.connect(self.refine_selected_mask)
        self.yolo_refine_toolbar.refine_all_requested.connect(self.refine_all_masks)
        
        # Create YOLO instance-focused toolbar
        if self.instance_focused_checkpoint:
            print(f"Loading instance-focused YOLO checkpoint from command line: {self.instance_focused_checkpoint}")
        self.yolo_instance_focused_toolbar = YOLOInstanceFocusedToolbar(self, checkpoint_path=self.instance_focused_checkpoint)
        self.yolo_instance_focused_toolbar.refine_requested.connect(self.refine_selected_instance_focused)
        self.yolo_instance_focused_toolbar.refine_all_requested.connect(self.refine_all_instances_focused)
        
        # Create YOLO SAHI toolbar
        self.yolo_sahi_toolbar = YOLOSAHIToolbar(self, checkpoint_path=None)
        self.yolo_sahi_toolbar.inference_requested.connect(self.run_sahi_inference)
        self.yolo_sahi_toolbar.box_inference_mode_requested.connect(self.on_box_inference_mode_requested)
        self.yolo_sahi_toolbar.box_inference_requested.connect(self.on_box_inference_requested)
        self.yolo_sahi_toolbar.soho_inference_requested.connect(self.run_soho_inference)
        self.yolo_sahi_toolbar.propagate_soho_requested.connect(self.propagate_soho_to_next_frame)
        self.yolo_sahi_toolbar.propagate_soho_to_selected_requested.connect(self.propagate_soho_to_selected)
        self.yolo_sahi_toolbar.propagate_soho_through_video_requested.connect(self.propagate_soho_through_video)
        self.yolo_sahi_toolbar.track_from_last_frame_requested.connect(self.track_from_last_frame)
        
        # Create YOLO Beehavesque toolbar
        if self.beehavesque_checkpoint:
            print(f"Loading BeeHaveSquE checkpoint from command line: {self.beehavesque_checkpoint}")
        self.yolo_beehavesque_toolbar = YOLOBeehavesqueToolbar(self, checkpoint_path=self.beehavesque_checkpoint)
        self.yolo_beehavesque_toolbar.inference_requested.connect(self.run_beehavesque_inference)
        self.yolo_beehavesque_toolbar.box_inference_mode_requested.connect(self.on_box_inference_mode_requested)
        self.yolo_beehavesque_toolbar.box_inference_requested.connect(self.on_beehavesque_box_inference_requested)
        self.yolo_beehavesque_toolbar.soho_inference_requested.connect(self.run_beehavesque_soho_inference)
        self.yolo_beehavesque_toolbar.propagate_soho_requested.connect(self.propagate_beehavesque_to_next_frame)
        self.yolo_beehavesque_toolbar.propagate_soho_to_selected_requested.connect(self.propagate_beehavesque_to_selected)
        self.yolo_beehavesque_toolbar.propagate_soho_through_video_requested.connect(self.propagate_beehavesque_through_video)
        self.yolo_beehavesque_toolbar.track_from_last_frame_requested.connect(self.track_from_last_frame)
        self.yolo_beehavesque_toolbar.validation_requested.connect(self.run_beehavesque_validation)
        
        # Create YOLO BBox toolbar
        if self.bbox_checkpoint:
            print(f"Loading YOLO BBox checkpoint from command line: {self.bbox_checkpoint}")
        self.yolo_bbox_toolbar = YOLOBBoxToolbar(self, checkpoint_path=self.bbox_checkpoint)
        self.yolo_bbox_toolbar.inference_requested.connect(self.run_yolo_bbox_inference)
        self.yolo_bbox_toolbar.track_from_last_requested.connect(self.track_from_last_frame)
        self.yolo_bbox_toolbar.propagate_requested.connect(self.propagate_yolo_bbox)
        
        # Create layout
        central_widget = QWidget()
        layout = QVBoxLayout(central_widget)
        layout.addWidget(self.toolbar)
        layout.addWidget(self.sam2_toolbar)
        layout.addWidget(self.yolo_toolbar)
        layout.addWidget(self.yolo_refine_toolbar)
        layout.addWidget(self.yolo_instance_focused_toolbar)
        layout.addWidget(self.yolo_sahi_toolbar)
        layout.addWidget(self.yolo_beehavesque_toolbar)
        layout.addWidget(self.yolo_bbox_toolbar)
        layout.addWidget(self.canvas)
        
        # Create frame navigation controls at the bottom
        nav_layout = QHBoxLayout()
        
        # Play/Pause button
        self.play_button = QPushButton("▶ Play")
        self.play_button.setMaximumWidth(100)
        self.play_button.clicked.connect(self.toggle_play)
        self.play_button.setToolTip("Play through frames automatically")
        nav_layout.addWidget(self.play_button)
        
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
        
        # Hide coarse and fine YOLO toolbars by default
        self.yolo_toolbar.hide()
        self.yolo_refine_toolbar.hide()
        self.yolo_instance_focused_toolbar.hide()
        # Show BeeHaveSquE toolbar, hide SAHI toolbar by default
        self.yolo_sahi_toolbar.hide()
        # Hide bbox toolbar by default
        self.yolo_bbox_toolbar.hide()
        
        # Create menu bar
        self.create_menus()
        
        # Create video list dock (sidebar)
        self.create_video_list_dock()
        
        # Create frame list dock
        self.create_frame_list_dock()
        
        # Create instance list dock
        self.create_instance_list_dock()
        
        # Status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_label = QLabel("Ready")
        self.status_bar.addPermanentWidget(self.status_label)
        
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
        
        file_menu.addSeparator()
        
        add_video_action = QAction("Add &Video to Project...", self)
        add_video_action.setShortcut("Ctrl+I")
        add_video_action.triggered.connect(self.show_add_video_dialog)
        file_menu.addAction(add_video_action)
        
        file_menu.addSeparator()
        
        save_action = QAction("&Save", self)
        save_action.setShortcut(QKeySequence.StandardKey.Save)
        save_action.triggered.connect(self.save_annotations)
        file_menu.addAction(save_action)
        
        export_coco_action = QAction("Export &COCO Format...", self)
        export_coco_action.triggered.connect(self.export_coco_format)
        file_menu.addAction(export_coco_action)
        
        export_video_action = QAction("Export Annotations as &Video...", self)
        export_video_action.triggered.connect(self.export_annotations_as_video)
        file_menu.addAction(export_video_action)
        
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
        
        resolve_overlaps_action = QAction("Resolve &Overlaps", self)
        resolve_overlaps_action.setShortcut("Ctrl+R")
        resolve_overlaps_action.triggered.connect(self.resolve_overlaps)
        edit_menu.addAction(resolve_overlaps_action)
        
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
        
        train_yolo_stage2_action = QAction("Train Refinement Model...", self)
        train_yolo_stage2_action.setShortcut("Ctrl+Shift+T")
        train_yolo_stage2_action.triggered.connect(self.train_yolo_model_stage2)
        model_menu.addAction(train_yolo_stage2_action)
        
        train_yolo_instance_focused_action = QAction("Train Instance-Focused Model...", self)
        train_yolo_instance_focused_action.triggered.connect(self.train_yolo_model_instance_focused)
        model_menu.addAction(train_yolo_instance_focused_action)
        
        train_yolo_bbox_action = QAction("Train BBox Detection Model...", self)
        train_yolo_bbox_action.triggered.connect(self.train_yolo_bbox_model)
        model_menu.addAction(train_yolo_bbox_action)
        
        predict_action = QAction("Run &Inference", self)
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
        self.bbox_mode_action.setChecked(False)
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
        
        # YOLO Refinement Toolbar toggle
        self.yolo_refine_toolbar_action = QAction("YOLO &Fine Toolbar", self)
        self.yolo_refine_toolbar_action.setCheckable(True)
        self.yolo_refine_toolbar_action.setChecked(False)  # Hidden by default
        self.yolo_refine_toolbar_action.triggered.connect(self.toggle_yolo_refine_toolbar)
        toolbars_menu.addAction(self.yolo_refine_toolbar_action)
        
        # YOLO Instance-Focused Toolbar toggle
        self.yolo_instance_focused_toolbar_action = QAction("YOLO &Instance-Focused Toolbar", self)
        self.yolo_instance_focused_toolbar_action.setCheckable(True)
        self.yolo_instance_focused_toolbar_action.setChecked(False)  # Hidden by default
        self.yolo_instance_focused_toolbar_action.triggered.connect(self.toggle_yolo_instance_focused_toolbar)
        toolbars_menu.addAction(self.yolo_instance_focused_toolbar_action)
        
        # YOLO SAHI Toolbar toggle
        self.yolo_sahi_toolbar_action = QAction("YOLO SA&HI Toolbar", self)
        self.yolo_sahi_toolbar_action.setCheckable(True)
        self.yolo_sahi_toolbar_action.setChecked(False)  # Hidden by default
        self.yolo_sahi_toolbar_action.triggered.connect(self.toggle_yolo_sahi_toolbar)
        toolbars_menu.addAction(self.yolo_sahi_toolbar_action)
        
        # YOLO Beehavesque Toolbar toggle
        self.yolo_beehavesque_toolbar_action = QAction("YOLO &Beehavesque Toolbar", self)
        self.yolo_beehavesque_toolbar_action.setCheckable(True)
        self.yolo_beehavesque_toolbar_action.setChecked(True)  # Visible by default
        self.yolo_beehavesque_toolbar_action.triggered.connect(self.toggle_yolo_beehavesque_toolbar)
        toolbars_menu.addAction(self.yolo_beehavesque_toolbar_action)
        
        # YOLO BBox Toolbar toggle
        self.yolo_bbox_toolbar_action = QAction("YOLO B&Box Toolbar", self)
        self.yolo_bbox_toolbar_action.setCheckable(True)
        self.yolo_bbox_toolbar_action.setChecked(False)  # Hidden by default
        self.yolo_bbox_toolbar_action.triggered.connect(self.toggle_yolo_bbox_toolbar)
        toolbars_menu.addAction(self.yolo_bbox_toolbar_action)
    
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
    
    def toggle_yolo_refine_toolbar(self):
        """Toggle visibility of YOLO refinement toolbar"""
        if self.yolo_refine_toolbar_action.isChecked():
            self.yolo_refine_toolbar.show()
        else:
            self.yolo_refine_toolbar.hide()
    
    def toggle_yolo_instance_focused_toolbar(self):
        """Toggle visibility of YOLO instance-focused toolbar"""
        if self.yolo_instance_focused_toolbar_action.isChecked():
            self.yolo_instance_focused_toolbar.show()
        else:
            self.yolo_instance_focused_toolbar.hide()
    
    def toggle_yolo_sahi_toolbar(self):
        """Toggle visibility of YOLO SAHI toolbar"""
        if self.yolo_sahi_toolbar_action.isChecked():
            self.yolo_sahi_toolbar.show()
        else:
            self.yolo_sahi_toolbar.hide()
    
    def toggle_yolo_beehavesque_toolbar(self):
        """Toggle visibility of YOLO Beehavesque toolbar"""
        if self.yolo_beehavesque_toolbar_action.isChecked():
            self.yolo_beehavesque_toolbar.show()
        else:
            self.yolo_beehavesque_toolbar.hide()
    
    def toggle_yolo_bbox_toolbar(self):
        """Toggle visibility of YOLO BBox toolbar"""
        if self.yolo_bbox_toolbar_action.isChecked():
            self.yolo_bbox_toolbar.show()
        else:
            self.yolo_bbox_toolbar.hide()
    
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
            if 'mask' not in ann:
                continue
            
            mask = ann['mask']
            
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
                # Empty mask, skip
                continue
            
            # Create Detection
            detection = Detection(
                bbox=bbox,
                mask=mask,
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
    
    def on_video_selected(self, row):
        """Handle video selection from video list"""
        if row < 0 or row >= self.video_list.count():
            return
        
        item = self.video_list.item(row)
        video_id = item.data(Qt.ItemDataRole.UserRole)  # Store video_id in item data
        
        if video_id:
            # Show loading cursor
            QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
            try:
                # Load this video's frames
                self.load_video_frames(video_id)
            finally:
                # Restore normal cursor
                QApplication.restoreOverrideCursor()
    
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
        self.video_list.clear()
        
        if not self.project_path:
            return
        
        # Get all videos from train, val, test, and inference splits
        train_videos = self.project_manager.get_videos_by_split('train')
        val_videos = self.project_manager.get_videos_by_split('val')
        test_videos = self.project_manager.get_videos_by_split('test')
        inference_videos = self.project_manager.get_videos_by_split('inference')
        
        # Add train videos
        for video_id in train_videos:
            item_text = f"[TRAIN] {video_id}"
            from PyQt6.QtWidgets import QListWidgetItem
            item = QListWidgetItem(item_text)
            item.setData(Qt.ItemDataRole.UserRole, video_id)  # Store video_id
            
            # Highlight current video
            if video_id == self.current_video_id:
                from PyQt6.QtGui import QFont
                font = QFont()
                font.setBold(True)
                item.setFont(font)
            
            self.video_list.addItem(item)
        
        # Add val videos
        for video_id in val_videos:
            item_text = f"[VAL] {video_id}"
            from PyQt6.QtWidgets import QListWidgetItem
            item = QListWidgetItem(item_text)
            item.setData(Qt.ItemDataRole.UserRole, video_id)  # Store video_id
            
            # Highlight current video
            if video_id == self.current_video_id:
                from PyQt6.QtGui import QFont
                font = QFont()
                font.setBold(True)
                item.setFont(font)
            
            self.video_list.addItem(item)
        
        # Add test videos
        for video_id in test_videos:
            item_text = f"[TEST] {video_id}"
            from PyQt6.QtWidgets import QListWidgetItem
            item = QListWidgetItem(item_text)
            item.setData(Qt.ItemDataRole.UserRole, video_id)  # Store video_id
            
            # Highlight current video
            if video_id == self.current_video_id:
                from PyQt6.QtGui import QFont
                font = QFont()
                font.setBold(True)
                item.setFont(font)
            
            self.video_list.addItem(item)
        
        # Add inference videos
        for video_id in inference_videos:
            item_text = f"[INFERENCE] {video_id}"
            from PyQt6.QtWidgets import QListWidgetItem
            item = QListWidgetItem(item_text)
            item.setData(Qt.ItemDataRole.UserRole, video_id)  # Store video_id
            
            # Highlight current video
            if video_id == self.current_video_id:
                from PyQt6.QtGui import QFont
                font = QFont()
                font.setBold(True)
                item.setFont(font)
            
            self.video_list.addItem(item)
        
    def create_video_list_dock(self):
        """Create dock widget for video list sidebar"""
        self.video_dock = QDockWidget("Project Videos", self)
        self.video_dock.setAllowedAreas(Qt.DockWidgetArea.LeftDockWidgetArea | 
                                        Qt.DockWidgetArea.RightDockWidgetArea)
        
        # Create video list widget
        self.video_list = QListWidget()
        self.video_list.currentRowChanged.connect(self.on_video_selected)
        self.video_list.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.video_list.customContextMenuRequested.connect(self.show_video_context_menu)
        
        self.video_dock.setWidget(self.video_list)
        self.video_dock.visibilityChanged.connect(self.on_video_dock_visibility_changed)
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, self.video_dock)
    
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
        self.split_filter_combo.addItems(['All Frames', 'Training Only', 'Validation Only', 'Test Only', 'Inference Only'])
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
        self.show_labels_checkbox.setChecked(False)
        self.show_labels_checkbox.stateChanged.connect(self.on_show_labels_changed)
        layout.addWidget(self.show_labels_checkbox)
        
        # Add instance list
        self.instance_list = QListWidget()
        self.instance_list.currentRowChanged.connect(self.on_instance_changed)
        self.instance_list.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.instance_list.customContextMenuRequested.connect(self.show_instance_context_menu)
        layout.addWidget(self.instance_list)
        
        dock.setWidget(container)
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
            
            # Create project in a subfolder with the project name
            base_path = Path(project_info['path'])
            self.project_path = base_path / project_name
            
            # Use ProjectManager to create new project structure
            self.project_manager.create_project(
                self.project_path, 
                project_name,
                frames_per_video=15  # Default, user can change per video
            )
            
            # Set marker detector debug folder
            debug_folder = self.project_path / 'marker_debug'
            self.marker_detector.set_debug_folder(str(debug_folder))
            
            self.annotation_manager.new_project(project_info)
            self.status_label.setText(
                f"Project created: {self.project_path.name} "
                f"(ready to add videos)"
            )
            
    def open_project(self):
        """Open an existing project"""
        path = QFileDialog.getExistingDirectory(
            self, "Open Project", 
            str(self.default_projects_dir)
        )
        if path:
            self.load_project(path)
            
    def load_project(self, path):
        """Load a project from path"""
        self.project_path = Path(path)
        
        # Set project path in ProjectManager
        self.project_manager.project_path = self.project_path
        
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
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load project: {str(e)}")
            import traceback
            traceback.print_exc()
            
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
        
        from PyQt6.QtWidgets import QDialog, QVBoxLayout, QComboBox, QPushButton, QSpinBox
        
        # Select video file
        video_path, _ = QFileDialog.getOpenFileName(
            self, "Select Video to Add", "",
            "Video Files (*.mp4 *.avi *.mov *.mkv *.mjpeg *.mjpg);;All Files (*)"
        )
        
        if not video_path:
            return
        
        # Create dialog for settings
        dialog = QDialog(self)
        dialog.setWindowTitle("Add Video to Project")
        layout = QVBoxLayout()
        
        # Split selector
        from PyQt6.QtWidgets import QLabel
        layout.addWidget(QLabel("Select split:"))
        split_combo = QComboBox()
        split_combo.addItems(['train', 'val', 'test', 'inference'])
        layout.addWidget(split_combo)
        
        # Frames for training/validation/test/inference selector
        layout.addWidget(QLabel("Number of frames to select:"))
        frames_spin = QSpinBox()
        frames_spin.setMinimum(1)
        frames_spin.setMaximum(1000)
        frames_spin.setValue(15)
        layout.addWidget(frames_spin)
        
        # Info text
        info_label = QLabel("All frames will be extracted, but only selected frames\nwill be marked for the chosen split.")
        info_label.setStyleSheet("color: gray; font-style: italic;")
        layout.addWidget(info_label)
        
        # Buttons
        from PyQt6.QtWidgets import QDialogButtonBox
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | 
            QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)
        
        dialog.setLayout(layout)
        
        if dialog.exec():
            split = split_combo.currentText()
            n_selected = frames_spin.value()
            self.add_video_to_project(video_path, split, n_selected)
            
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
                    self.annotation_manager.load_project(project_dir)
                    self.load_frames_from_project()
                    
                    # Count annotated frames
                    annotated_count = len([f for f in self.annotation_manager.frame_annotations.values() if f])
                    self.status_label.setText(
                        f"✓ Loaded project: {num_frames} frames, {annotated_count} annotated"
                    )
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
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to extract frames: {str(e)}")
                self.status_label.setText("Ready")
    
    def add_video_to_project(self, video_path, split='train', n_selected=15):
        """
        Add a video to project and extract all frames
        
        Args:
            video_path: Path to video file
            split: 'train', 'val', or 'test'
            n_selected: Number of frames to mark for the chosen split
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
            
            # Extract ALL frames
            total_frames = metadata['total_frames']
            frame_indices = list(range(total_frames))
            
            extract_result = self.project_manager.extract_video_frames(
                video_id, frame_indices
            )
            
            # Select subset for training/validation
            selected_indices = self.project_manager.select_frames_uniform(
                total_frames, n_selected
            )
            
            # Save metadata about which frames are selected for training/validation
            video_metadata_file = self.project_manager.get_frames_dir(video_id) / 'video_metadata.json'
            import json
            with open(video_metadata_file, 'w') as f:
                json.dump({
                    'split': split,
                    'total_frames': total_frames,
                    'selected_frames': selected_indices,
                    'n_selected': n_selected
                }, f, indent=2)
            
            self.status_label.setText(
                f"✓ Added {video_id} to {split}: "
                f"{extract_result['extracted']} frames extracted, "
                f"{n_selected} marked for {split}"
            )
            
            # Update video list to show newly added video
            self.update_video_list()
            
            # Load this video's frames for annotation
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
                    annotations = self.canvas.get_annotations()
                    if annotations:
                        frame_idx_in_video = self._get_frame_idx_in_video(self.current_frame_idx)
                        # Do a blocking save for current frame
                        self.annotation_manager.save_frame_annotations(
                            self.project_path, self.current_video_id,
                            frame_idx_in_video, annotations
                        )
                        # Mark frame as saved to prevent duplicate saves
                        self.current_frame_modified = False
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
            self.frame_selected = [i in selected_indices for i in range(len(frame_files))]
            
            progress.setLabelText(f"Loading video {video_id}... (updating UI)")
            QApplication.processEvents()
            
            # Update UI
            self.update_video_list()  # Refresh video list to update current video highlighting
            self.update_frame_list()
            
            progress.setLabelText(f"Loading video {video_id}... (loading first frame)")
            QApplication.processEvents()
            
            if self.frames:
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
            elif self.split_filter == 'test' and (not is_selected or split != 'test'):
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
            
            # Auto-save current frame annotations before loading new frame (only if modified)
            if self.current_frame_idx != idx and self.project_path and self.current_frame_modified:
                try:
                    # Commit any pending edits before saving current frame
                    if self.canvas.editing_instance_id > 0:
                        self.canvas.commit_editing()
                    
                    annotations = self.canvas.get_annotations()
                    if annotations and old_video_id:  # Only save if there are annotations and we know the video
                        # Update video next_mask_id tracking
                        if old_video_id in self.video_next_mask_id:
                            self.video_next_mask_id[old_video_id] = self.canvas.next_mask_id
                        
                        # Update in-memory cache
                        self.annotation_manager.set_frame_annotations(
                            self.current_frame_idx, annotations, video_id=old_video_id
                        )
                        # Get actual frame index within video (not the list index)
                        frame_idx_in_video = self._get_frame_idx_in_video(self.current_frame_idx)
                        # Queue background save - non-blocking!
                        self.save_worker.add_save_task(
                            self.project_path, old_video_id, 
                            frame_idx_in_video, annotations
                        )
                except Exception as e:
                    print(f"Warning: Failed to queue save: {e}")
            
            self.current_frame_idx = idx
            
            # Update current video ID from frame_video_ids list
            if idx < len(self.frame_video_ids):
                self.current_video_id = self.frame_video_ids[idx]
            
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
                self.canvas.load_image(image_to_load)
                
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
                
                # Give focus to canvas for keyboard shortcuts (spacebar to hide masks)
                self.canvas.setFocus()
                
                # Store image dimensions in annotation manager (for COCO export)
                if self.canvas.current_image is not None:
                    h, w = self.canvas.current_image.shape[:2]
                    self.annotation_manager.image_height = h
                    self.annotation_manager.image_width = w
                
                # Restore next_mask_id for this video to maintain unique IDs
                if self.current_video_id:
                    if self.current_video_id not in self.video_next_mask_id:
                        # Try to load cached max_mask_id from metadata (backward compatible)
                        max_id = self._load_max_mask_id_from_metadata(self.current_video_id)
                        
                        if max_id is None:
                            # No cached value - compute by scanning all frames (slow, first time only)
                            import time
                            t_start = time.perf_counter()
                            max_id = 0
                            file_count = 0
                            if self.project_path:
                                annotations_dir = self.project_path / 'annotations' / 'pkl' / self.current_video_id
                                if annotations_dir.exists():
                                    for ann_file in annotations_dir.glob('frame_*.pkl'):
                                        file_count += 1
                                        try:
                                            import pickle
                                            with open(ann_file, 'rb') as f:
                                                anns = pickle.load(f)
                                                for ann in anns:
                                                    if 'mask_rle' in ann or 'mask' in ann:
                                                        mask_id = ann.get('mask_id', 0)
                                                        max_id = max(max_id, mask_id)
                                        except Exception:
                                            pass
                            t_elapsed = (time.perf_counter() - t_start) * 1000
                            if file_count > 0:
                                print(f"  ⚠ Scanned {file_count} annotation files to find max_mask_id (took {t_elapsed:.0f}ms, caching...)")
                            
                            # Cache for future use
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
                    self.frame_list.setCurrentRow(list_row)
                    # Update slider to match
                    self.frame_slider.blockSignals(True)
                    self.frame_slider.setValue(list_row)
                    self.frame_slider.blockSignals(False)
                except ValueError:
                    # Frame not in current filter view
                    pass
                
                # Save current project state for restoration
                self._save_project_state()
                
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
            has_segmentation = (
                self.canvas.combined_mask is not None and 
                np.any(self.canvas.combined_mask == self.canvas.selected_mask_idx)
            )
            if has_segmentation:
                self.canvas.start_editing_instance(self.canvas.selected_mask_idx)
        
        # When switching away from SAM2 tools, uncheck SAM2 toolbar buttons
        if tool_name not in ['sam2_prompt', 'sam2_box']:
            self.sam2_toolbar.uncheck_tools()
        
        # When switching to SAM2 tools, uncheck annotation toolbar buttons
        if tool_name in ['sam2_prompt', 'sam2_box']:
            self.toolbar.uncheck_all_tools()
    
    def on_show_segmentations_changed(self, show):
        """Handle show segmentations checkbox change from toolbar"""
        self.canvas.set_show_segmentations(show)
        # Sync with menu action
        self.segmentation_mode_action.setChecked(show)
    
    def on_show_bboxes_changed(self, show):
        """Handle show bboxes checkbox change from toolbar"""
        self.canvas.set_show_bboxes(show)
        # Sync with menu action
        self.bbox_mode_action.setChecked(show)
    
    def on_menu_show_segmentations_changed(self, checked):
        """Handle show segmentations menu action change"""
        self.canvas.set_show_segmentations(checked)
        # Sync with toolbar checkbox
        self.toolbar.segmentation_checkbox.setChecked(checked)
    
    def on_menu_show_bboxes_changed(self, checked):
        """Handle show bboxes menu action change"""
        self.canvas.set_show_bboxes(checked)
        # Sync with toolbar checkbox
        self.toolbar.bbox_checkbox.setChecked(checked)
    
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
                    if self.canvas.editing_instance_id != self.canvas.selected_mask_idx:
                        self.canvas.start_editing_instance(self.canvas.selected_mask_idx)
                    # Update the editing mask directly
                    self.canvas.editing_mask = (mask > 0).astype(np.uint8) * 255
                    self.canvas._update_editing_visualization()
                    self.canvas.active_sam2_mask_idx = self.canvas.selected_mask_idx
                # Check if we're editing an active SAM2 mask
                elif self.canvas.active_sam2_mask_idx > 0:
                    # Start editing mode if not already active
                    if self.canvas.editing_instance_id != self.canvas.active_sam2_mask_idx:
                        self.canvas.start_editing_instance(self.canvas.active_sam2_mask_idx)
                    # Update the editing mask
                    self.canvas.editing_mask = (mask > 0).astype(np.uint8) * 255
                    self.canvas._update_editing_visualization()
                else:
                    # Create new mask with new ID and start editing it
                    mask_id = self.canvas.next_mask_id
                    self.canvas.selected_mask_idx = mask_id
                    self.canvas.start_editing_instance(mask_id)
                    self.canvas.editing_mask = (mask > 0).astype(np.uint8) * 255
                    self.canvas._update_editing_visualization()
                    self.canvas.active_sam2_mask_idx = mask_id
                    # Increment next_mask_id since we used this one
                    self.canvas.next_mask_id += 1
                
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
                        self.canvas.start_editing_instance(self.canvas.active_sam2_mask_idx)
                    # Update the editing mask
                    self.canvas.editing_mask = (mask > 0).astype(np.uint8) * 255
                    self.canvas._update_editing_visualization()
                else:
                    # Create new mask with new ID and start editing it
                    mask_id = self.canvas.next_mask_id
                    self.canvas.selected_mask_idx = mask_id
                    self.canvas.start_editing_instance(mask_id)
                    self.canvas.editing_mask = (mask > 0).astype(np.uint8) * 255
                    self.canvas._update_editing_visualization()
                    self.canvas.active_sam2_mask_idx = mask_id
                    # Increment next_mask_id since we used this one
                    self.canvas.next_mask_id += 1
                
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
        # Flush pending instance list update to avoid lag
        if self._instance_list_update_timer.isActive():
            self._instance_list_update_timer.stop()
            self._do_update_instance_list()
        
        if visible:
            self.status_label.setText("Masks visible")
        else:
            self.status_label.setText("⚠ Annotations hidden (release Spacebar to show)")
    
    def on_annotation_changed(self):
        """Handle annotation changes - mark frame as modified"""
        self.current_frame_modified = True
    
    def on_show_labels_changed(self, state):
        """Handle show instance labels checkbox change"""
        from PyQt6.QtCore import Qt
        visible = (state == Qt.CheckState.Checked.value)
        self.canvas.set_labels_visible(visible)
                
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
        self.instance_list.clear()
        
        if self.canvas.combined_mask is None:
            return

        # Preserve selection for the active/editing instance
        active_instance_id = (
            self.canvas.editing_instance_id
            if self.canvas.editing_instance_id > 0
            else self.canvas.selected_mask_idx
        )
        
        # Get marker info from annotation_metadata
        marker_info = {
            inst_id: meta.get('marker') 
            for inst_id, meta in self.canvas.annotation_metadata.items() 
            if meta.get('marker')
        }
        
        # Use cached instance IDs for performance (avoids expensive np.unique call)
        instance_ids = self.canvas.get_instance_ids()
        
        for instance_id in instance_ids:
            # Determine if instance has segmentation or is bbox-only
            has_segmentation = (
                (instance_id == self.canvas.editing_instance_id and self.canvas.editing_mask is not None) or
                np.any(self.canvas.combined_mask == instance_id)
            )
            
            # Calculate area from appropriate source
            if instance_id == self.canvas.editing_instance_id and self.canvas.editing_mask is not None:
                area = np.sum(self.canvas.editing_mask > 0)
            elif has_segmentation:
                area = np.sum(self.canvas.combined_mask == instance_id)
            else:
                # Bbox-only: calculate area from bbox (width * height)
                meta = self.canvas.annotation_metadata.get(instance_id, {})
                bbox = meta.get('bbox', [0, 0, 0, 0])
                area = bbox[2] * bbox[3]  # width * height
            
            # Build display text with marker info if available
            display_text = f"Instance ID: {instance_id}"
            if instance_id in marker_info:
                marker = marker_info[instance_id]
                marker_type = marker.get('type', 'unknown')
                marker_id = marker.get('id', 'N/A')
                if marker_type == 'aruco':
                    display_text += f" [ArUco: {marker_id}]"
                elif marker_type == 'qr':
                    display_text += f" [QR: {marker_id}]"
            display_text += f" (area: {area})"
            
            self.instance_list.addItem(display_text)
        
        # Update instance labels on canvas if they are currently visible
        if self.canvas.labels_visible:
            self.canvas.update_instance_labels()

        if active_instance_id in instance_ids:
            active_index = instance_ids.index(active_instance_id)
            self.instance_list.blockSignals(True)
            self.instance_list.setCurrentRow(active_index)
            self.instance_list.blockSignals(False)
    
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
    
    def on_instance_changed(self, idx):
        """Handle instance selection change
        
        Args:
            idx: Row index in the instance list widget
        """
        if idx < 0:
            return
        
        if self.canvas.combined_mask is None:
            return
        
        # Use cached instance IDs for performance (avoids expensive np.unique call)
        instance_ids = self.canvas.get_instance_ids()
        
        if idx < len(instance_ids):
            instance_id = instance_ids[idx]
            self.canvas.set_selected_instance(instance_id)
            self.canvas.highlight_instance(instance_id)
            self.status_label.setText(f"Selected instance {instance_id} - Use Brush/Eraser to edit")
    
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
        self.current_frame_modified = True
        
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
            frame_idx_in_video = self._get_frame_idx_in_video(self.current_frame_idx)
            try:
                self.annotation_manager.save_frame_annotations(
                    self.project_path, self.current_video_id,
                    frame_idx_in_video, annotations
                )
                # Update cache and clear unsaved flag (we just saved!)
                self.annotation_manager.set_frame_annotations(
                    self.current_frame_idx, annotations, video_id=self.current_video_id
                )
                self.annotation_manager.unsaved_changes = False  # Clear unsaved flag since we just saved
                self.current_frame_modified = False
                print(f"✓ Saved current frame (frame {frame_idx_in_video}) to disk before propagation")
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
        idx = self.instance_list.currentRow()
        if idx < 0:
            return
        
        from PyQt6.QtWidgets import QMenu
        menu = QMenu()
        
        edit_id_action = menu.addAction("Edit Bee ID...")
        delete_action = menu.addAction("Delete Instance")
        
        # Add propagate options if not at last frame
        propagate_action = None
        propagate_through_action = None
        next_idx = self._get_next_frame_index()
        if next_idx is not None:
            menu.addSeparator()
            propagate_action = menu.addAction("Copy to Next Frame...")
            propagate_through_action = menu.addAction("Copy Through Frames...")
        
        action = menu.exec(self.instance_list.mapToGlobal(position))
        
        if action == edit_id_action:
            # Trigger the same edit dialog as double-click
            item = self.instance_list.item(idx)
            if item:
                self.on_instance_double_clicked(item)
        elif action == delete_action:
            self.delete_selected_instance()
        elif propagate_action and action == propagate_action:
            self.propagate_selected_instance_to_next_frame()
        elif propagate_through_action and action == propagate_through_action:
            self.propagate_selected_instance_through_frames()
    
    def propagate_selected_instance_through_frames(self):
        """Copy the selected instance mask through multiple frames until a specified frame"""
        from PyQt6.QtWidgets import QInputDialog
        
        # Get selected instance
        idx = self.instance_list.currentRow()
        if idx < 0:
            QMessageBox.warning(
                self, "No Instance Selected",
                "Please select an instance to copy."
            )
            return
        
        # Get the instance ID and mask
        if self.canvas.combined_mask is None:
            QMessageBox.warning(
                self, "No Annotations",
                "No annotations found on current frame."
            )
            return
        
        # Get unique instance IDs
        instance_ids = np.unique(self.canvas.combined_mask)
        instance_ids = instance_ids[instance_ids > 0].tolist()
        
        # Add editing instance if present
        if self.canvas.editing_instance_id > 0 and self.canvas.editing_mask is not None:
            if self.canvas.editing_instance_id not in instance_ids:
                instance_ids.append(self.canvas.editing_instance_id)
        
        instance_ids = sorted(instance_ids)
        
        if idx >= len(instance_ids):
            QMessageBox.warning(
                self, "Invalid Selection",
                "Selected instance index is out of range."
            )
            return
        
        instance_id = instance_ids[idx]
        
        # Get the mask for this instance
        if instance_id == self.canvas.editing_instance_id and self.canvas.editing_mask is not None:
            instance_mask = self.canvas.editing_mask.copy()
        else:
            instance_mask = (self.canvas.combined_mask == instance_id).astype(np.uint8) * 255
        
        if not np.any(instance_mask > 0):
            QMessageBox.warning(
                self, "Empty Mask",
                "The selected instance has an empty mask."
            )
            return
        
        # Find the range of available frames
        last_frame_idx = len(self.frames) - 1
        
        # Ask user for the target frame
        target_frame, ok = QInputDialog.getInt(
            self,
            "Copy Through Frames",
            f"Copy Instance ID {instance_id} through frames until:\n"
            f"(Current frame: {self.current_frame_idx}, Last frame: {last_frame_idx})",
            value=last_frame_idx,  # Default to last frame
            min=self.current_frame_idx + 1,
            max=last_frame_idx,
            step=1
        )
        
        if not ok:
            return
        
        # Calculate number of frames to copy to
        num_frames = target_frame - self.current_frame_idx
        
        # Confirm the operation
        reply = QMessageBox.question(
            self,
            "Confirm Copy",
            f"Copy Instance ID {instance_id} to {num_frames} frame(s)\n"
            f"(from frame {self.current_frame_idx + 1} to {target_frame})?\n\n"
            f"This will overwrite any existing annotations for this instance ID on those frames.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No
        )
        
        if reply != QMessageBox.StandardButton.Yes:
            return
        
        try:
            # Save current frame first
            current_annotations = self.canvas.get_annotations()
            self.annotation_manager.set_frame_annotations(
                self.current_frame_idx, current_annotations, video_id=self.current_video_id
            )
            
            original_frame_idx = self.current_frame_idx
            frames_copied = 0
            
            self.status_label.setText(f"Copying Instance ID {instance_id} through frames...")
            QApplication.processEvents()
            
            # Loop through frames
            for frame_offset in range(1, num_frames + 1):
                target_idx = original_frame_idx + frame_offset
                
                self.status_label.setText(
                    f"Copying Instance ID {instance_id} to frame {target_idx} ({frame_offset}/{num_frames})..."
                )
                QApplication.processEvents()
                
                # Navigate to target frame
                self.load_frame(target_idx)
                
                # Get existing annotations on target frame
                target_annotations = self.canvas.get_annotations()
                
                # Check if this instance ID already exists
                existing_ids = [ann.get('mask_id', ann.get('instance_id', 0)) 
                               for ann in target_annotations]
                
                # Preserve mask ID and color if tracking is enabled
                if self.tracking_enabled and self.current_video_id:
                    mask_id = instance_id
                    # Get color from video colors
                    if self.current_video_id not in self.video_mask_colors:
                        self.video_mask_colors[self.current_video_id] = {}
                    color = self.video_mask_colors[self.current_video_id].get(mask_id)
                else:
                    mask_id = instance_id
                    color = None
                
                # If instance ID already exists, remove the old one
                if mask_id in existing_ids:
                    # Remove old instance with this ID
                    target_annotations = [ann for ann in target_annotations 
                                         if ann.get('mask_id', ann.get('instance_id', 0)) != mask_id]
                    # Rebuild canvas with remaining annotations
                    mask_colors_dict = {ann['mask_id']: self.canvas.mask_colors.get(ann['mask_id']) 
                                       for ann in target_annotations if ann.get('mask_id')}
                    self.canvas.set_annotations(target_annotations, mask_colors_dict)
                
                # Add the copied mask to canvas
                self.canvas.add_mask(instance_mask, mask_id=mask_id, color=color, rebuild_viz=True)
                self._register_canvas_colors()
                
                # Save this frame
                updated_annotations = self.canvas.get_annotations()
                self.annotation_manager.set_frame_annotations(
                    target_idx, updated_annotations, video_id=self.current_video_id
                )
                
                frames_copied += 1
            
            # Return to original frame
            self.load_frame(original_frame_idx)
            
            self.status_label.setText(
                f"✓ Copied Instance ID {instance_id} to {frames_copied} frame(s)"
            )
            
            QMessageBox.information(
                self, "Copy Complete",
                f"Instance ID {instance_id} has been copied to {frames_copied} frame(s)\n"
                f"(frames {original_frame_idx + 1} through {target_frame}).\n\n"
                f"All frames have been saved."
            )
            
        except Exception as e:
            import traceback
            error_msg = traceback.format_exc()
            QMessageBox.critical(
                self, "Copy Error",
                f"Error copying instance through frames:\n{str(e)}\n\n{error_msg}"
            )
            self.status_label.setText("Copy failed")
            # Try to return to original frame
            try:
                self.load_frame(original_frame_idx)
            except:
                pass
    
    def propagate_selected_instance_to_next_frame(self):
        """Copy the selected instance mask to the next frame and enter edit mode"""
        # Get selected instance
        idx = self.instance_list.currentRow()
        if idx < 0:
            QMessageBox.warning(
                self, "No Instance Selected",
                "Please select an instance to copy."
            )
            return
        
        # Get the instance ID
        if self.canvas.combined_mask is None:
            QMessageBox.warning(
                self, "No Annotations",
                "No annotations found on current frame."
            )
            return
        
        # Get unique instance IDs
        instance_ids = np.unique(self.canvas.combined_mask)
        instance_ids = instance_ids[instance_ids > 0].tolist()
        
        # Add editing instance if present
        if self.canvas.editing_instance_id > 0 and self.canvas.editing_mask is not None:
            if self.canvas.editing_instance_id not in instance_ids:
                instance_ids.append(self.canvas.editing_instance_id)
        
        instance_ids = sorted(instance_ids)
        
        if idx >= len(instance_ids):
            QMessageBox.warning(
                self, "Invalid Selection",
                "Selected instance index is out of range."
            )
            return
        
        instance_id = instance_ids[idx]
        
        # Get the mask for this instance
        if instance_id == self.canvas.editing_instance_id and self.canvas.editing_mask is not None:
            instance_mask = self.canvas.editing_mask.copy()
        else:
            instance_mask = (self.canvas.combined_mask == instance_id).astype(np.uint8) * 255
        
        if not np.any(instance_mask > 0):
            QMessageBox.warning(
                self, "Empty Mask",
                "The selected instance has an empty mask."
            )
            return
        
        # Find next frame
        next_idx = self._get_next_frame_index()
        if next_idx is None:
            QMessageBox.information(
                self, "Last Frame",
                "Already at the last frame in current view."
            )
            return
        
        try:
            # Save current frame annotations first
            current_annotations = self.canvas.get_annotations()
            self.annotation_manager.set_frame_annotations(
                self.current_frame_idx, current_annotations, video_id=self.current_video_id
            )
            
            self.status_label.setText(f"Copying Instance ID {instance_id} to next frame...")
            QApplication.processEvents()
            
            # Navigate to next frame
            self.load_frame(next_idx)
            
            # Get existing annotations on next frame (if any)
            next_frame_annotations = self.canvas.get_annotations()
            
            # Preserve mask ID and color if tracking is enabled
            if self.tracking_enabled and self.current_video_id:
                mask_id = instance_id
                # Get color from video colors
                if self.current_video_id not in self.video_mask_colors:
                    self.video_mask_colors[self.current_video_id] = {}
                color = self.video_mask_colors[self.current_video_id].get(mask_id)
            else:
                # Assign next available ID
                existing_ids = [ann.get('mask_id', ann.get('instance_id', 0)) 
                               for ann in next_frame_annotations]
                mask_id = max(existing_ids) + 1 if existing_ids else 1
                color = None
            
            # Add the copied mask to canvas
            self.canvas.add_mask(instance_mask, mask_id=mask_id, color=color, rebuild_viz=True)
            self._register_canvas_colors()
            
            # Set this instance as selected and start editing
            self.canvas.selected_mask_idx = mask_id
            self.canvas.start_editing_instance(mask_id)
            
            # Update instance list
            self.update_instance_list_from_canvas()
            
            # Mark frame as modified
            self.current_frame_modified = True
            
            self.status_label.setText(
                f"✓ Copied Instance ID {instance_id} (now in edit mode - adjust as needed)"
            )
            
            QMessageBox.information(
                self, "Instance Copied",
                f"Instance ID {instance_id} has been copied to the next frame.\n\n"
                f"The instance is now in edit mode - you can:\n"
                f"• Add positive points to expand the mask\n"
                f"• Add negative points to shrink the mask\n"
                f"• Click 'Finish Editing' when done\n\n"
                f"The frame will not be saved until you finish editing."
            )
            
        except Exception as e:
            import traceback
            error_msg = traceback.format_exc()
            QMessageBox.critical(
                self, "Copy Error",
                f"Error copying instance:\n{str(e)}\n\n{error_msg}"
            )
            self.status_label.setText("Copy failed")
            
    def delete_selected_instance(self):
        """Delete the currently selected instance"""
        if self.canvas.selected_mask_idx >= 0:
            self.canvas.clear_selected_instance()
            self.update_instance_list_from_canvas()
            self.status_label.setText("Instance deleted")
    
    def delete_all_instances(self):
        """Delete all instances in the current frame and remove annotation files"""
        # Check if there are any annotations to delete
        annotations = self.canvas.get_annotations()
        if not annotations:
            QMessageBox.information(
                self, "No Instances",
                "There are no instances to delete in the current frame."
            )
            return
        
        # Show confirmation dialog
        num_instances = len(annotations)
        reply = QMessageBox.question(
            self, "Delete All Instances",
            f"Are you sure you want to delete ALL {num_instances} instance(s) in this frame?\n\n"
            f"This will also remove the annotation files from disk.\n\n"
            f"This action cannot be undone.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No
        )
        
        if reply == QMessageBox.StandardButton.Yes:
            # Clear all annotations from canvas
            self.canvas.set_annotations([])
            self.update_instance_list_from_canvas()
            
            # Delete annotation files from disk if we have a project and video
            if self.project_path and self.current_video_id is not None:
                try:
                    from pathlib import Path
                    project_path = Path(self.project_path)
                    video_id = self.current_video_id
                    frame_idx = self.current_frame_idx
                    
                    # Delete PNG annotation file
                    png_file = project_path / 'annotations' / 'png' / video_id / f'frame_{frame_idx:06d}.png'
                    if png_file.exists():
                        png_file.unlink()
                        print(f"Deleted PNG annotation: {png_file}")
                    
                    # Delete JSON metadata file
                    json_file = project_path / 'annotations' / 'json' / video_id / f'frame_{frame_idx:06d}.json'
                    if json_file.exists():
                        json_file.unlink()
                        print(f"Deleted JSON metadata: {json_file}")
                    
                    # Delete bbox annotation file
                    bbox_file = project_path / 'annotations' / 'bbox' / video_id / f'frame_{frame_idx:06d}.json'
                    if bbox_file.exists():
                        bbox_file.unlink()
                        print(f"Deleted bbox annotation: {bbox_file}")
                    
                    # Delete pickle file if it exists (legacy format)
                    pkl_file = project_path / 'annotations' / 'pkl' / video_id / f'frame_{frame_idx:06d}.pkl'
                    if pkl_file.exists():
                        pkl_file.unlink()
                        print(f"Deleted pickle annotation: {pkl_file}")
                    
                    # Clear the annotation cache for this frame
                    if self.current_frame_idx in self.annotation_manager.frame_annotations:
                        del self.annotation_manager.frame_annotations[self.current_frame_idx]
                        print(f"Cleared cache for frame {self.current_frame_idx}")
                    
                    self.status_label.setText(f"✓ Deleted all {num_instances} instance(s) and removed annotation files")
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
                self.status_label.setText(f"✓ Deleted all {num_instances} instance(s)")
                self.current_frame_modified = True
    
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
            if self.canvas.editing_instance_id != selected_id:
                self.canvas.start_editing_instance(selected_id)
            
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
            
    def new_instance(self):
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
        
        # Add to annotation metadata to make it appear in the list
        if new_instance_id not in self.canvas.annotation_metadata:
            self.canvas.annotation_metadata[new_instance_id] = {
                'bbox': [0, 0, 0, 0],  # Placeholder bbox
                'bbox_only': True
            }
        
        # Invalidate cached instance IDs so new instance appears in list
        self.canvas._cached_instance_ids = None
        
        # Generate color for new instance
        if new_instance_id not in self.canvas.mask_colors:
            self.canvas.mask_colors[new_instance_id] = tuple(np.random.randint(0, 255, 3).tolist())
        
        # Update list to show new instance
        self.update_instance_list_from_canvas()
        
        # Find and select the new instance in the list
        for i in range(self.instance_list.count()):
            item_text = self.instance_list.item(i).text()
            if f"Instance ID: {new_instance_id}" in item_text:
                self.instance_list.setCurrentRow(i)
                break
        
        # Update status based on current tool
        current_tool = self.canvas.current_tool
        if current_tool == 'bbox':
            self.status_label.setText(f"Instance {new_instance_id} created - Click and drag to draw bounding box")
        elif current_tool in ['brush', 'eraser']:
            self.status_label.setText(f"Instance {new_instance_id} created - Use Brush to start drawing")
        else:
            self.status_label.setText(f"Instance {new_instance_id} created - Select a tool (Brush/BBox)")
    
    def resolve_overlaps(self):
        """Resolve overlapping masks across all instances"""
        overlap_count = self.canvas.resolve_all_overlaps()
        if overlap_count > 0:
            self.update_instance_list_from_canvas()
            self.status_label.setText(f"✓ Resolved {overlap_count} overlapping pixels")
        else:
            self.status_label.setText("No overlaps found")
    
    def _register_canvas_colors(self):
        """Register current canvas mask colors for the current video"""
        if not self.current_video_id:
            return
        
        # Initialize color mapping for this video if needed
        if self.current_video_id not in self.video_mask_colors:
            self.video_mask_colors[self.current_video_id] = {}
        
        # Register colors from canvas (mask_colors is dict {instance_id: (r,g,b)})
        for mask_id, color in self.canvas.mask_colors.items():
            if mask_id not in self.video_mask_colors[self.current_video_id]:
                # Convert numpy array to tuple if needed
                if isinstance(color, np.ndarray):
                    color = tuple(int(c) for c in color)
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
            self.canvas.add_mask(mask, mask_id, color, rebuild_viz=rebuild_viz)
        
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
        self.current_frame_modified = True
        
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
        if hasattr(self, 'yolo_toolbar') and hasattr(self.yolo_toolbar, 'checkpoint_path'):
            current_model_path = self.yolo_toolbar.checkpoint_path
        
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
                
                msg += "Would you like to load the new model into the coarse YOLO toolbar?"
                
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
                        self.yolo_toolbar._load_checkpoint_from_path(model_path, show_dialogs=False)
                        self.status_label.setText(f"✓ Loaded trained model: {Path(model_path).name}")
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
    
    def train_yolo_model_stage2(self):
        """Train YOLO fine-grained refinement model (Stage 2)"""
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
        
        # Show configuration dialog with Stage 2 specific parameters
        config_dialog = TrainingConfigDialog(self, current_model_path, stage2=True)
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
            
            # Create training worker for Stage 2
            worker = YOLOTrainingWorkerStage2(self.project_path, config)
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
                msg = "Stage 2 training completed successfully!\n\n"
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
                        self.status_label.setText(f"✓ Loaded trained Stage 2 model: {Path(model_path).name}")
                    else:
                        QMessageBox.warning(self, "Model Not Found", f"Model file not found: {model_path}")
            elif progress_dialog.training_failed:
                # Training failed
                QMessageBox.critical(
                    self,
                    "Training Failed",
                    "Stage 2 training failed. Check the training log for details.\n\n"
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
    
    def train_yolo_model_sahi(self):
        """Train YOLO SAHI model with enhanced augmentation for sliced inference"""
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
        
        # Get current SAHI YOLO model path
        current_model_path = None
        if hasattr(self, 'yolo_sahi_toolbar') and hasattr(self.yolo_sahi_toolbar, 'model_path'):
            current_model_path = self.yolo_sahi_toolbar.model_path
        
        # Show configuration dialog with SAHI specific parameters
        config_dialog = TrainingConfigDialog(self, current_model_path, stage2=False, sahi=True)
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
            
            # Create training worker for SAHI
            worker = YOLOTrainingWorkerSAHI(self.project_path, config)
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
                msg = "SAHI training completed successfully!\n\n"
                msg += f"Model saved to:\n{model_path}\n\n"
                
                if final_metrics:
                    msg += "Final Metrics:\n"
                    for key, value in final_metrics.items():
                        msg += f"  {key}: {value:.4f}\n"
                    msg += "\n"
                
                msg += "Would you like to load the new model into the SAHI YOLO toolbar?"
                
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
                        self.yolo_sahi_toolbar._load_checkpoint_from_path(model_path, show_dialogs=False)
                        self.status_label.setText(f"✓ Loaded trained SAHI model: {Path(model_path).name}")
                    else:
                        QMessageBox.warning(self, "Model Not Found", f"Model file not found: {model_path}")
            elif progress_dialog.training_failed:
                # Training failed
                QMessageBox.critical(
                    self,
                    "Training Failed",
                    "SAHI training failed. Check the training log for details.\n\n"
                    "The backup model has been preserved."
                )
            # else: training was stopped/cancelled, no action needed
    
    def train_yolo_model_beehavesque(self):
        """Train YOLO Beehavesque model with temporal frame context (prev/current/next as RGB)"""
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
        
        # Get current Beehavesque YOLO model path
        current_model_path = None
        if hasattr(self, 'yolo_beehavesque_toolbar') and hasattr(self.yolo_beehavesque_toolbar, 'model_path'):
            current_model_path = self.yolo_beehavesque_toolbar.model_path
        
        # Show configuration dialog with same parameters as SAHI (crop size, augmentation, etc.)
        config_dialog = TrainingConfigDialog(self, current_model_path, stage2=False, sahi=False, beehavesque=True)
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
            
            # Create training worker for Beehavesque
            worker = YOLOTrainingWorkerBeehavesque(self.project_path, config)
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
                msg = "Beehavesque training completed successfully!\n\n"
                msg += f"Model saved to:\n{model_path}\n\n"
                
                if final_metrics:
                    msg += "Final Metrics:\n"
                    for key, value in final_metrics.items():
                        msg += f"  {key}: {value:.4f}\n"
                    msg += "\n"
                
                msg += "Would you like to load the new model into the Beehavesque YOLO toolbar?"
                
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
                        self.yolo_beehavesque_toolbar._load_checkpoint_from_path(model_path, show_dialogs=False)
                        self.status_label.setText(f"✓ Loaded trained Beehavesque model: {Path(model_path).name}")
                    else:
                        QMessageBox.warning(self, "Model Not Found", f"Model file not found: {model_path}")
            elif progress_dialog.training_failed:
                # Training failed
                QMessageBox.critical(
                    self,
                    "Training Failed",
                    "Beehavesque training failed. Check the training log for details.\n\n"
                    "The backup model has been preserved."
                )
            # else: training was stopped/cancelled, no action needed
        
    def run_inference(self):
        """Run inference on current/all frames"""
        # TODO: Implement inference
        QMessageBox.information(self, "Inference", "Inference feature coming soon!")
        
    def save_annotations(self):
        """Save current annotations and regenerate COCO datasets"""
        if self.project_path:
            try:
                # Commit any pending edits before saving
                if self.canvas.editing_instance_id > 0:
                    self.canvas.commit_editing()
                
                # Save current frame annotations using video-based structure
                annotations = self.canvas.get_annotations()
                
                if annotations and self.current_video_id:
                    # Get frame index within video
                    frame_idx_in_video = self._get_frame_idx_in_video(self.current_frame_idx)
                    
                    # Save using video-based structure
                    self.annotation_manager.save_frame_annotations(
                        self.project_path, self.current_video_id,
                        frame_idx_in_video, annotations
                    )
                    
                    # Update cache
                    self.annotation_manager.set_frame_annotations(
                        self.current_frame_idx, annotations, video_id=self.current_video_id
                    )
                    
                    # Update cached max_mask_id for this video (performance optimization)
                    if self.current_video_id in self.video_next_mask_id:
                        max_id = self.video_next_mask_id[self.current_video_id] - 1
                        self._save_max_mask_id_to_metadata(self.current_video_id, max_id)
                    
                    num_instances = len(annotations)
                    self.status_label.setText(
                        f"Saving... {num_instances} instance(s) to {self.current_video_id}"
                    )
                else:
                    self.status_label.setText("✓ Nothing to save")
                
                # Regenerate COCO datasets for training
                self.status_label.setText("Regenerating COCO datasets...")
                QApplication.processEvents()
                
                train_videos = self.project_manager.get_videos_by_split('train')
                val_videos = self.project_manager.get_videos_by_split('val')
                
                coco_files_generated = []
                
                # Export train split if videos exist
                if train_videos:
                    train_paths = export_coco_per_video(
                        self.project_path,
                        train_videos,
                        'train',
                        class_names=self.annotation_manager.class_names,
                        image_width=self.annotation_manager.image_width,
                        image_height=self.annotation_manager.image_height
                    )
                    coco_files_generated.append(f"Training: {len(train_paths)} videos")
                
                # Export val split if videos exist
                if val_videos:
                    val_paths = export_coco_per_video(
                        self.project_path,
                        val_videos,
                        'val',
                        class_names=self.annotation_manager.class_names,
                        image_width=self.annotation_manager.image_width,
                        image_height=self.annotation_manager.image_height
                    )
                    coco_files_generated.append(f"Validation: {len(val_paths)} videos")
                
                # Show success message
                project_name = self.project_path.name
                coco_msg = "\n".join(coco_files_generated) if coco_files_generated else "No COCO files (no videos in splits)"
                
                if annotations and self.current_video_id:
                    success_msg = (
                        f"Annotations and COCO datasets saved!\n\n"
                        f"Project: {project_name}\n"
                        f"Video: {self.current_video_id}\n"
                        f"Frame: {frame_idx_in_video}\n"
                        f"Instances: {num_instances}\n\n"
                        f"COCO Datasets:\n{coco_msg}"
                    )
                else:
                    success_msg = (
                        f"COCO datasets regenerated!\n\n"
                        f"Project: {project_name}\n\n"
                        f"COCO Datasets:\n{coco_msg}"
                    )
                
                self.status_label.setText(
                    f"✓ Saved and regenerated COCO datasets"
                )
                QMessageBox.information(
                    self, 
                    "Save Successful", 
                    success_msg
                )
                    
            except Exception as e:
                import traceback
                error_msg = f"Failed to save annotations:\n\n{str(e)}\n\n{traceback.format_exc()}"
                self.status_label.setText("✗ Save failed")
                QMessageBox.critical(self, "Save Error", error_msg)
        else:
            QMessageBox.warning(self, "No Project", "Please create or open a project first")
    
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
    
    def export_coco_format(self):
        """Export annotations in COCO format (train, val, and test splits only - inference split is excluded)"""
        if not self.project_path:
            QMessageBox.warning(self, "No Project", "Please create or open a project first")
            return
        
        try:
            # Save current frame first
            if self.current_frame_idx is not None:
                annotations = self.canvas.get_annotations()
                self.annotation_manager.set_frame_annotations(
                    self.current_frame_idx, annotations, video_id=self.current_video_id
                )
            
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
            all_selected.extend([i in selected_indices for i in range(len(frame_files))])
        
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
                    # Found the video - select it (this will load the video's frames)
                    self.video_list.setCurrentRow(i)
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
                for i, (detection, track_id) in enumerate(matched_detections):
                    # Get or use existing color for this track
                    if self.current_video_id not in self.video_mask_colors:
                        self.video_mask_colors[self.current_video_id] = {}
                    
                    color = self.video_mask_colors[self.current_video_id].get(track_id)
                    rebuild_viz = (i == len(matched_detections) - 1)
                    self.canvas.add_mask(detection.mask, mask_id=track_id, color=color, rebuild_viz=rebuild_viz)
                
                # Register any new colors
                self._register_canvas_colors()
                
                # Update next_mask_id
                self.video_next_mask_id[self.current_video_id] = tracker.next_track_id
                
                # Mark frame as modified so annotations will be saved
                self.current_frame_modified = True
                
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
                for i, detection in enumerate(detections):
                    mask_id = next_id + i
                    rebuild_viz = (i == len(detections) - 1)
                    self.canvas.add_mask(detection.mask, mask_id=mask_id, rebuild_viz=rebuild_viz)
                
                self._register_canvas_colors()
                
                # Mark frame as modified so annotations will be saved
                self.current_frame_modified = True
                
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
            # Return focus to canvas so spacebar works for hiding masks
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
        
        # Check if instance segmentations already exist
        if self.canvas.combined_mask is not None and np.any(self.canvas.combined_mask > 0):
            QMessageBox.warning(
                self, "Segmentations Exist",
                "Cannot run YOLO bbox inference on frames that already have instance segmentations.\n\n"
                "Bounding boxes should only be used for frames without segmentation masks."
            )
            return
        
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
            
            # Use tracker to match detections to existing annotations
            if self.tracking_enabled and self.current_video_id:
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
                self.current_frame_modified = True
                
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
                self.current_frame_modified = True
                
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
            # Return focus to canvas so spacebar works for hiding masks
            self.canvas.setFocus()
    
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
                            frame_idx_in_video = self._get_frame_idx_in_video(next_frame_idx)
                            self.annotation_manager.set_frame_annotations(next_frame_idx, [], video_id=self.current_video_id)
                            self.save_worker.add_save_task(
                                self.project_path,
                                self.current_video_id,
                                frame_idx_in_video,
                                []
                            )
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
                        frame_idx_in_video = self._get_frame_idx_in_video(next_frame_idx)
                        # Update in-memory cache immediately
                        self.annotation_manager.set_frame_annotations(next_frame_idx, bbox_annotations, video_id=self.current_video_id)
                        # Queue background save - non-blocking!
                        self.save_worker.add_save_task(
                            self.project_path,
                            self.current_video_id,
                            frame_idx_in_video,
                            bbox_annotations
                        )
                    
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
                            frame_idx_in_video = self._get_frame_idx_in_video(next_frame_idx)
                            self.annotation_manager.set_frame_annotations(next_frame_idx, [], video_id=self.current_video_id)
                            self.save_worker.add_save_task(
                                self.project_path,
                                self.current_video_id,
                                frame_idx_in_video,
                                []
                            )
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
                        frame_idx_in_video = self._get_frame_idx_in_video(next_frame_idx)
                        # Update in-memory cache immediately
                        self.annotation_manager.set_frame_annotations(next_frame_idx, frame_annotations, video_id=self.current_video_id)
                        # Queue background save - non-blocking!
                        self.save_worker.add_save_task(
                            self.project_path,
                            self.current_video_id,
                            frame_idx_in_video,
                            frame_annotations
                        )
                    
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
                            frame_idx_in_video = self._get_frame_idx_in_video(next_frame_idx)
                            self.annotation_manager.set_frame_annotations(next_frame_idx, [], video_id=self.current_video_id)
                            self.save_worker.add_save_task(
                                self.project_path,
                                self.current_video_id,
                                frame_idx_in_video,
                                []
                            )
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
                        frame_idx_in_video = self._get_frame_idx_in_video(next_frame_idx)
                        # Update in-memory cache immediately
                        self.annotation_manager.set_frame_annotations(next_frame_idx, frame_annotations, video_id=self.current_video_id)
                        # Queue background save - non-blocking!
                        self.save_worker.add_save_task(
                            self.project_path,
                            self.current_video_id,
                            frame_idx_in_video,
                            frame_annotations
                        )
                    
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
    
    def on_box_inference_mode_requested(self, checked):
        """Handle box inference mode toggle from SAHI or Beehavesque toolbar"""
        self.box_inference_mode = checked
        if checked:
            # Switch to inference box tool when mode is enabled
            self.canvas.set_tool('inference_box')
            self.status_label.setText("Draw a box on the image (you can adjust it afterwards)")
        else:
            # Return to default tool when disabled
            self.canvas.set_tool('polygon')
            self.canvas.clear_inference_box()
            # Notify both toolbars that box is cleared
            self.yolo_sahi_toolbar.set_box_drawn(False)
            self.yolo_beehavesque_toolbar.set_box_drawn(False)
            self.status_label.setText("Box inference mode disabled")
    
    def on_box_inference_requested(self):
        """Handle request to run inference on the drawn box"""
        # Get the box from canvas
        box = self.canvas.get_inference_box()
        if not box:
            QMessageBox.warning(
                self, "No Box",
                "Please draw a box first."
            )
            return
        
        x1, y1, x2, y2 = box
        print(f"Running inference on box: ({x1}, {y1}) to ({x2}, {y2})")
        
        # Run inference
        self.run_yolo_inference_on_box(x1, y1, x2, y2)
        
        # Clear the box and reset both toolbars
        self.canvas.clear_inference_box()
        self.yolo_sahi_toolbar.set_box_drawn(False)
        self.yolo_beehavesque_toolbar.set_box_drawn(False)
        self.canvas.set_tool('polygon')
    
    def run_yolo_inference_on_box(self, x1, y1, x2, y2):
        """Run SAHI inference on a user-specified box region with smart merging"""
        if not self.yolo_sahi_toolbar.is_model_loaded():
            QMessageBox.warning(
                self, "No Model",
                "Please load a YOLO SAHI checkpoint first."
            )
            return
        
        if not self.frames or self.current_frame_idx >= len(self.frames):
            QMessageBox.warning(
                self, "No Frame",
                "No frame is currently loaded."
            )
            return
        
        try:
            import cv2
            
            # Get current frame path and image
            frame_path = self.frames[self.current_frame_idx]
            full_image = cv2.imread(str(frame_path))
            if full_image is None:
                raise ValueError(f"Failed to load image: {frame_path}")
            
            # Get current annotations before inference
            current_annotations = self.canvas.get_annotations()
            
            # Normalize box coordinates
            x1, x2 = int(min(x1, x2)), int(max(x1, x2))
            y1, y2 = int(min(y1, y2)), int(max(y1, y2))
            
            # Ensure box is within image bounds
            h, w = full_image.shape[:2]
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(w, x2)
            y2 = min(h, y2)
            
            print(f"Running box inference on region: ({x1}, {y1}) to ({x2}, {y2})")
            self.status_label.setText(f"Running SAHI inference on box region...")
            QApplication.processEvents()
            
            # Crop image to box region
            cropped_image = full_image[y1:y2, x1:x2]
            
            # Save cropped image to temporary file for SAHI
            import tempfile
            with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp_file:
                tmp_path = tmp_file.name
                cv2.imwrite(tmp_path, cropped_image)
            
            try:
                # Import SAHI
                from sahi import AutoDetectionModel
                from sahi.predict import get_sliced_prediction
                
                # Get model and SAHI parameters
                model = self.yolo_sahi_toolbar.get_model()
                sahi_params = self.yolo_sahi_toolbar.get_sahi_params()
                
                # Wrap YOLO model in SAHI AutoDetectionModel
                detection_model = AutoDetectionModel.from_pretrained(
                    model_type='yolov8',
                    model_path=str(self.yolo_sahi_toolbar.model_path),
                    confidence_threshold=0.5,
                    device='cuda:0' if model.device.type == 'cuda' else 'cpu'
                )
                
                # Run SAHI inference on cropped region
                result = get_sliced_prediction(
                    tmp_path,
                    detection_model,
                    slice_height=sahi_params['slice_height'],
                    slice_width=sahi_params['slice_width'],
                    overlap_height_ratio=sahi_params['overlap_height_ratio'],
                    overlap_width_ratio=sahi_params['overlap_width_ratio'],
                    postprocess_type=sahi_params['postprocess_type'],
                    postprocess_match_metric=sahi_params['postprocess_match_metric'],
                    postprocess_match_threshold=sahi_params['postprocess_match_threshold'],
                    postprocess_class_agnostic=sahi_params['postprocess_class_agnostic'],
                    verbose=0
                )
                
                if not result.object_prediction_list or len(result.object_prediction_list) == 0:
                    self.status_label.setText("No detections in box region")
                    QMessageBox.information(
                        self, "No Detections",
                        "SAHI did not detect any instances in the box region."
                    )
                    return
                
                # Convert SAHI results to Detection objects
                detections = self._sahi_results_to_detections(result, tmp_path)
                
                if not detections:
                    self.status_label.setText("No detections in box region")
                    return
                
                print(f"Found {len(detections)} detections in box region")
                
                # Transform detection masks back to full image coordinates
                for detection in detections:
                    # Create full-size mask
                    full_mask = np.zeros((h, w), dtype=np.uint8)
                    # Place cropped mask in correct position
                    full_mask[y1:y2, x1:x2] = detection.mask
                    detection.mask = full_mask
                    
                    # Update bounding box to full image coordinates
                    bbox_x1, bbox_y1, bbox_x2, bbox_y2 = detection.bbox
                    detection.bbox = (bbox_x1 + x1, bbox_y1 + y1, bbox_x2 + x1, bbox_y2 + y1)
                
                # Identify existing instances fully contained in box vs. edge instances
                contained_mask_ids = []  # IDs to remove
                edge_annotations = []  # Annotations partially overlapping box
                outside_annotations = []  # Annotations completely outside box
                
                for ann in current_annotations:
                    mask = ann['mask']
                    # Check overlap with box region
                    box_mask = np.zeros_like(mask)
                    box_mask[y1:y2, x1:x2] = 1
                    
                    mask_area = np.sum(mask > 0)
                    overlap_area = np.sum((mask > 0) & (box_mask > 0))
                    
                    if overlap_area == 0:
                        # No overlap - keep as is
                        outside_annotations.append(ann)
                    elif overlap_area == mask_area:
                        # Fully contained - mark for removal
                        contained_mask_ids.append(ann.get('mask_id'))
                        print(f"Removing fully contained instance {ann.get('mask_id')}")
                    else:
                        # Partially overlapping - candidate for merging
                        edge_annotations.append(ann)
                        print(f"Edge instance {ann.get('mask_id')}: {overlap_area}/{mask_area} overlap")
                
                # Merge edge instances with new detections
                merged_annotations = outside_annotations.copy()
                used_detections = set()
                
                for edge_ann in edge_annotations:
                    edge_mask = edge_ann['mask']
                    best_match_idx = None
                    best_overlap = 0
                    
                    # Find best matching detection using mask IoU
                    for i, detection in enumerate(detections):
                        if i in used_detections:
                            continue
                        
                        intersection = np.sum((edge_mask > 0) & (detection.mask > 0))
                        union = np.sum((edge_mask > 0) | (detection.mask > 0))
                        
                        if union > 0:
                            iou = intersection / union
                            if iou > best_overlap and iou > 0.1:  # Minimum 10% overlap
                                best_overlap = iou
                                best_match_idx = i
                    
                    if best_match_idx is not None:
                        # Merge detection with edge instance
                        detection = detections[best_match_idx]
                        merged_mask = np.maximum(edge_mask, detection.mask)
                        
                        # Update annotation with merged mask
                        edge_ann['mask'] = merged_mask
                        
                        # Recalculate bounding box
                        coords = np.column_stack(np.where(merged_mask > 0))
                        if len(coords) > 0:
                            y_min, x_min = coords.min(axis=0)
                            y_max, x_max = coords.max(axis=0)
                            edge_ann['bbox'] = [int(x_min), int(y_min), int(x_max), int(y_max)]
                        
                        # Recalculate contours
                        contours, _ = cv2.findContours(merged_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        edge_ann['contours'] = contours
                        
                        used_detections.add(best_match_idx)
                        merged_annotations.append(edge_ann)
                        print(f"Merged detection {best_match_idx} with edge instance {edge_ann.get('mask_id')} (IoU: {best_overlap:.2f})")
                    else:
                        # No good match - keep edge instance as is
                        merged_annotations.append(edge_ann)
                        print(f"No match for edge instance {edge_ann.get('mask_id')}, keeping as is")
                
                # Add remaining new detections (not merged)
                next_id = max([ann.get('mask_id', 0) for ann in merged_annotations], default=0) + 1
                for i, detection in enumerate(detections):
                    if i not in used_detections:
                        # Create annotation for new detection
                        mask_id = next_id
                        next_id += 1
                        
                        contours, _ = cv2.findContours(detection.mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        
                        new_ann = {
                            'mask': detection.mask,
                            'mask_id': mask_id,
                            'contours': contours,
                            'bbox': [int(x) for x in detection.bbox]
                        }
                        merged_annotations.append(new_ann)
                        print(f"Added new detection as instance {mask_id}")
                
                # Clean duplicate contours between instances
                if len(merged_annotations) > 1:
                    merged_annotations = self._clean_duplicate_contours(merged_annotations, overlap_threshold=0.5)
                
                # Build mask colors dictionary
                mask_colors_dict = {}
                if self.current_video_id and self.current_video_id in self.video_mask_colors:
                    for ann in merged_annotations:
                        mask_id = ann.get('mask_id')
                        if mask_id and mask_id in self.video_mask_colors[self.current_video_id]:
                            mask_colors_dict[mask_id] = self.video_mask_colors[self.current_video_id][mask_id]
                
                # Update canvas with merged annotations
                self.canvas.set_annotations(merged_annotations, mask_colors_dict)
                self._register_canvas_colors()
                
                # Mark frame as modified
                self.current_frame_modified = True
                
                # Update UI
                self.update_instance_list_from_canvas()
                self.status_label.setText(
                    f"✓ Box inference complete: {len(detections)} detected, "
                    f"{len(contained_mask_ids)} removed, {len(used_detections)} merged"
                )
                
                QMessageBox.information(
                    self, "Box Inference Complete",
                    f"Detected {len(detections)} instance(s) in box region.\n\n"
                    f"Removed {len(contained_mask_ids)} fully contained instance(s).\n"
                    f"Merged {len(used_detections)} with edge instance(s).\n"
                    f"Added {len(detections) - len(used_detections)} new instance(s)."
                )
                
            finally:
                # Clean up temporary file
                import os
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)
        
        except Exception as e:
            import traceback
            error_msg = traceback.format_exc()
            QMessageBox.critical(
                self, "Box Inference Error",
                f"Error running box inference:\n{str(e)}\n\n{error_msg}"
            )
            self.status_label.setText("Box inference failed")
        finally:
            # Return focus to canvas so spacebar works for hiding masks
            self.canvas.setFocus()
    
    def run_sahi_inference(self):
        """Run SAHI sliced inference on the current frame"""
        if not self.yolo_sahi_toolbar.is_model_loaded():
            QMessageBox.warning(
                self, "No Model",
                "Please load a YOLO SAHI checkpoint first."
            )
            return
        
        if not self.frames or self.current_frame_idx >= len(self.frames):
            QMessageBox.warning(
                self, "No Frame",
                "No frame is currently loaded."
            )
            return
        
        try:
            # Check if sahi is installed
            try:
                from sahi import AutoDetectionModel
                from sahi.predict import get_sliced_prediction, get_prediction
            except ImportError:
                QMessageBox.critical(
                    self, "SAHI Not Installed",
                    "SAHI library is not installed.\n\n"
                    "Please install it with:\n"
                    "pip install sahi"
                )
                return
            
            # Get current frame path
            frame_path = self.frames[self.current_frame_idx]
            
            # Get SAHI parameters from toolbar
            sahi_params = self.yolo_sahi_toolbar.get_sahi_params()
            
            # Get model
            model = self.yolo_sahi_toolbar.get_model()
            model_path = str(self.yolo_sahi_toolbar.model_path)
            
            self.status_label.setText("Running SAHI sliced inference...")
            QApplication.processEvents()  # Update UI
            
            # Create SAHI detection model
            detection_model = AutoDetectionModel.from_pretrained(
                model_type='ultralytics',  
                model_path=model_path,
                confidence_threshold=0.5,
                device='cuda:0' if torch.cuda.is_available() else 'cpu'
            )
            
            # Run sliced prediction with postprocessing parameters
            result = get_sliced_prediction(
                str(frame_path),
                detection_model,
                slice_height=sahi_params['slice_height'],
                slice_width=sahi_params['slice_width'],
                overlap_height_ratio=sahi_params['overlap_height_ratio'],
                overlap_width_ratio=sahi_params['overlap_width_ratio'],
                verbose=0,
                perform_standard_pred=False,
                postprocess_type=sahi_params['postprocess_type'],
                postprocess_match_metric=sahi_params['postprocess_match_metric'],
                postprocess_match_threshold=sahi_params['postprocess_match_threshold'],
                postprocess_class_agnostic=sahi_params['postprocess_class_agnostic'],
            )
            
            if not result or not result.object_prediction_list:
                self.status_label.setText("No detections found")
                QMessageBox.information(
                    self, "No Detections",
                    "SAHI did not detect any instances in the current frame."
                )
                return
            
            # Convert SAHI results to Detection objects
            detections = self._sahi_results_to_detections(result, frame_path)
            
            # FIRST: Clean overlapping contours from individual detections
            initial_count = len(detections)
            detections = self._clean_detections_contours(detections, overlap_threshold=0.15)
            print(f"SAHI: After contour cleaning: {len(detections)} detections")
            
            # THEN: Apply additional custom duplicate merging if needed
            # (SAHI's postprocessing should handle most cases, but this provides extra safety)
            detections = self._remove_duplicate_detections(detections, iou_threshold=0.3)
            if len(detections) < initial_count:
                print(f"SAHI: Custom merging combined {initial_count - len(detections)} overlapping detections")
            
            if not detections:
                self.status_label.setText("No detections found")
                QMessageBox.information(
                    self, "No Detections",
                    "SAHI did not detect any instances in the current frame."
                )
                return
            
            # Use tracker to match detections to existing annotations
            if self.tracking_enabled and self.current_video_id:
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
                
                print(f"SAHI Tracking: {num_matched} matched to existing tracks, {num_new} new tracks created")
                
                # Clear existing annotations and add matched detections
                self.canvas.set_annotations([])
                
                # Add matched detections with their assigned IDs
                for i, (detection, track_id) in enumerate(matched_detections):
                    # Get or use existing color for this track
                    if self.current_video_id not in self.video_mask_colors:
                        self.video_mask_colors[self.current_video_id] = {}
                    
                    color = self.video_mask_colors[self.current_video_id].get(track_id)
                    rebuild_viz = (i == len(matched_detections) - 1)
                    self.canvas.add_mask(detection.mask, mask_id=track_id, color=color, rebuild_viz=rebuild_viz)
                
                # Register any new colors
                self._register_canvas_colors()
                
                # Update next_mask_id
                self.video_next_mask_id[self.current_video_id] = tracker.next_track_id
                
                # Mark frame as modified so annotations will be saved
                self.current_frame_modified = True
                
                self.status_label.setText(
                    f"✓ SAHI inference complete: {len(matched_detections)} instances detected (with tracking)"
                )
            else:
                # No tracking - just add detections with sequential IDs
                canvas_annotations = self.canvas.get_annotations()
                next_id = max([ann.get('instance_id', ann.get('mask_id', 0)) for ann in canvas_annotations], default=0) + 1
                
                # Clear and add new detections
                self.canvas.set_annotations([])
                
                for i, detection in enumerate(detections):
                    mask_id = next_id + i
                    rebuild_viz = (i == len(detections) - 1)
                    self.canvas.add_mask(detection.mask, mask_id=mask_id, rebuild_viz=rebuild_viz)
                
                self._register_canvas_colors()
                
                # Mark frame as modified so annotations will be saved
                self.current_frame_modified = True
                
                self.status_label.setText(
                    f"✓ SAHI inference complete: {len(detections)} instances detected"
                )
            
            self.update_instance_list_from_canvas()
            
            # Show summary
            num_detected = len(detections)
            tracking_msg = " (with ID tracking)" if self.tracking_enabled and self.current_video_id else ""
            QMessageBox.information(
                self, "SAHI Inference Complete",
                f"SAHI detected {num_detected} instance(s) using sliced inference{tracking_msg}.\n\n"
                f"Slice size: {sahi_params['slice_width']}x{sahi_params['slice_height']}\n"
                f"Overlap ratio: {sahi_params['overlap_width_ratio']:.2f}\n"
                f"Postprocessing: {sahi_params['postprocess_type']} "
                f"({sahi_params['postprocess_match_metric']} @ {sahi_params['postprocess_match_threshold']:.2f})\n\n"
                f"The detections have been added as new instances.\n"
                f"You can now edit, delete, or propagate them as needed."
            )
            
        except ImportError as e:
            QMessageBox.critical(
                self, "Import Error",
                f"Could not import required libraries:\n{str(e)}\n\n"
                "Make sure ultralytics, sahi, and opencv-python are installed."
            )
        except Exception as e:
            import traceback
            error_msg = traceback.format_exc()
            QMessageBox.critical(
                self, "Inference Error",
                f"Error running SAHI inference:\n{str(e)}\n\n{error_msg}"
            )
            self.status_label.setText("SAHI inference failed")
        finally:
            # Return focus to canvas so spacebar works for hiding masks
            self.canvas.setFocus()
    
    def run_soho_inference(self):
        """Run SOHO (Sliced Overlapping Heuristic Optimization) inference on the current frame
        
        SOHO adds padding to avoid edge artifacts, creates overlapping slices,
        filters out detections near slice edges, then merges results.
        """
        # Debug mode flag - set to True to save slice visualizations
        DEBUG_SOHO = True
        
        if not self.yolo_sahi_toolbar.is_model_loaded():
            QMessageBox.warning(
                self, "No Model",
                "Please load a YOLO SAHI checkpoint first."
            )
            return
        
        if not self.frames or self.current_frame_idx >= len(self.frames):
            QMessageBox.warning(
                self, "No Frame",
                "No frame is currently loaded."
            )
            return
        
        try:
            import cv2
            import shutil
            from datetime import datetime
            
            # Setup debug folder if debug mode is enabled
            debug_folder = None
            if DEBUG_SOHO:
                # Use project-based tmp folder
                debug_folder = Path.cwd() / "tmp" / "soho_debug"
                
                # Clear and recreate the folder
                if debug_folder.exists():
                    shutil.rmtree(debug_folder)
                debug_folder.mkdir(parents=True, exist_ok=True)
                
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                print(f"\nSOHO Debug: Saving slice visualizations to {debug_folder}")
                print(f"Timestamp: {timestamp}")
                
                # Create README in debug folder
                readme_text = """SOHO Debug Visualizations
========================

This folder contains debug visualizations from SOHO (Sliced Overlapping Heuristic Optimization) inference.

Files:
------
00_slice_grid_overview.jpg
  - Overview of the padded image with all slice boundaries shown in magenta
  - Green rectangle shows the original image boundary within the padding
  - Numbers indicate slice processing order

slice_001.jpg, slice_002.jpg, etc.
  - Individual slice visualizations
  - Yellow rectangle: Edge filter boundary (50px margin)
  - RED boxes: Detections that were REJECTED (too close to slice edge)
  - GREEN boxes: Detections that were KEPT (away from edges)
  - Text overlay shows slice position, size, and detection counts

99_final_result.jpg
  - Final merged detections on the original image (no padding)
  - Green boxes show all detections after edge filtering and merging
  - Numbers indicate detection ID and confidence score

Parameters:
-----------
Padding: 100px around image (BORDER_CONSTANT, black)
Edge Filter: 50px margin per slice
Detections within 50px of any slice edge are removed entirely
After filtering, overlapping detections are merged using IoU threshold

Color Legend:
-------------
- Magenta: Slice boundaries (overview)
- Yellow: Edge filter zone (individual slices)
- Red: Rejected detections (too close to edge)
- Green: Kept detections (away from edges)
- Green (final): Final merged detections
"""
                readme_path = debug_folder / "README.txt"
                with open(readme_path, 'w') as f:
                    f.write(readme_text)
            
            # Get current frame path
            frame_path = self.frames[self.current_frame_idx]
            
            # Load original image
            original_img = cv2.imread(str(frame_path))
            if original_img is None:
                raise ValueError(f"Failed to load image: {frame_path}")
            
            orig_h, orig_w = original_img.shape[:2]
            
            # SOHO parameters
            pad_size = 100  # Padding around image
            edge_filter = 50  # Filter detections within this margin of each slice's edges
            
            # Get slicing parameters from toolbar
            sahi_params = self.yolo_sahi_toolbar.get_sahi_params()
            slice_h = sahi_params['slice_height']
            slice_w = sahi_params['slice_width']
            overlap_h = sahi_params['overlap_height_ratio']
            overlap_w = sahi_params['overlap_width_ratio']
            
            self.status_label.setText("Running SOHO inference (creating padded image)...")
            QApplication.processEvents()
            
            # Add black padding to avoid edge artifacts
            padded_img = cv2.copyMakeBorder(
                original_img,
                pad_size, pad_size, pad_size, pad_size,
                cv2.BORDER_CONSTANT,
                value=0
            )
            
            padded_h, padded_w = padded_img.shape[:2]
            
            # Calculate slice positions with overlap
            stride_h = int(slice_h * (1 - overlap_h))
            stride_w = int(slice_w * (1 - overlap_w))
            
            slice_positions = []
            y = 0
            while y < padded_h:
                x = 0
                while x < padded_w:
                    # Calculate slice bounds
                    y2 = min(y + slice_h, padded_h)
                    x2 = min(x + slice_w, padded_w)
                    
                    # Adjust start position if we hit the edge
                    y1 = max(0, y2 - slice_h)
                    x1 = max(0, x2 - slice_w)
                    
                    slice_positions.append((x1, y1, x2, y2))
                    
                    if x2 >= padded_w:
                        break
                    x += stride_w
                
                if y2 >= padded_h:
                    break
                y += stride_h
            
            print(f"SOHO: Created {len(slice_positions)} slices from {padded_w}x{padded_h} padded image")
            print(f"Slice size: {slice_w}x{slice_h}, Stride: {stride_w}x{stride_h}")
            
            # Create overview visualization showing all slice positions
            if DEBUG_SOHO and debug_folder:
                overview_img = padded_img.copy()
                
                # Draw black borders to show the 100px padding region
                # Top border
                cv2.rectangle(overview_img, (0, 0), (padded_w, pad_size), (0, 0, 0), -1)
                # Bottom border
                cv2.rectangle(overview_img, (0, padded_h - pad_size), (padded_w, padded_h), (0, 0, 0), -1)
                # Left border
                cv2.rectangle(overview_img, (0, 0), (pad_size, padded_h), (0, 0, 0), -1)
                # Right border
                cv2.rectangle(overview_img, (padded_w - pad_size, 0), (padded_w, padded_h), (0, 0, 0), -1)
                
                # Add text labels for padding regions
                cv2.putText(overview_img, "100px PADDING", (padded_w // 2 - 100, pad_size // 2),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                cv2.putText(overview_img, "100px PADDING", (padded_w // 2 - 100, padded_h - pad_size // 2),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                cv2.putText(overview_img, "100px", (pad_size // 4, padded_h // 2),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.putText(overview_img, "100px", (padded_w - pad_size + 10, padded_h // 2),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                for idx, (x1, y1, x2, y2) in enumerate(slice_positions):
                    # Draw slice boundary
                    cv2.rectangle(overview_img, (x1, y1), (x2, y2), (255, 0, 255), 2)  # Magenta
                    # Draw slice number
                    cv2.putText(
                        overview_img,
                        str(idx + 1),
                        (x1 + 10, y1 + 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (255, 255, 255),
                        2
                    )
                
                # Draw original image boundary within padding (green)
                cv2.rectangle(
                    overview_img,
                    (pad_size, pad_size),
                    (pad_size + orig_w, pad_size + orig_h),
                    (0, 255, 0),  # Green for original image boundary
                    3
                )
                
                # Save overview
                overview_path = debug_folder / "00_slice_grid_overview.jpg"
                cv2.imwrite(str(overview_path), overview_img)
                print(f"Saved slice grid overview to {overview_path}")
            
            # Get model
            model = self.yolo_sahi_toolbar.get_model()
            
            # Process each slice
            all_detections = []
            
            for idx, (x1, y1, x2, y2) in enumerate(slice_positions):
                self.status_label.setText(f"Running SOHO inference (slice {idx+1}/{len(slice_positions)})...")
                QApplication.processEvents()
                
                # Extract slice
                slice_img = padded_img[y1:y2, x1:x2]
                
                # Run YOLO inference on this slice
                results = model.predict(
                    source=slice_img,
                    conf=0.5,
                    iou=0.5,
                    retina_masks=True,
                    verbose=False
                )
                
                if not results or len(results) == 0 or results[0].masks is None:
                    continue
                
                # Convert results to detections
                slice_detections = self._yolo_results_to_detections(results[0], model)
                
                # Create visualization of this slice (only if debug mode)
                if DEBUG_SOHO and debug_folder:
                    vis_img = slice_img.copy()
                    
                    # Draw edge filter boundary (yellow)
                    cv2.rectangle(
                        vis_img,
                        (edge_filter, edge_filter),
                        (slice_img.shape[1] - edge_filter, slice_img.shape[0] - edge_filter),
                        (0, 255, 255),  # Yellow
                        2
                    )
                
                # Filter detections near slice edges
                slice_h_actual = y2 - y1
                slice_w_actual = x2 - x1
                
                filtered_detections = []
                rejected_detections = []
                for detection in slice_detections:
                    bbox = detection.bbox  # [x1_local, y1_local, x2_local, y2_local]
                    
                    # Check if detection is too close to any edge of this slice
                    is_too_close = (bbox[0] < edge_filter or bbox[2] > slice_w_actual - edge_filter or
                                   bbox[1] < edge_filter or bbox[3] > slice_h_actual - edge_filter)
                    
                    if is_too_close:
                        # Skip - too close to edge, likely partial detection
                        rejected_detections.append(detection)
                        continue
                    
                    # Transform coordinates from slice space to padded image space
                    # Adjust bbox
                    detection.bbox = [
                        bbox[0] + x1,
                        bbox[1] + y1,
                        bbox[2] + x1,
                        bbox[3] + y1
                    ]
                    
                    # Transform mask to padded image coordinates
                    # Create full padded image mask
                    full_mask = np.zeros((padded_h, padded_w), dtype=np.uint8)
                    full_mask[y1:y2, x1:x2] = detection.mask
                    detection.mask = full_mask
                    
                    filtered_detections.append(detection)
                
                # Draw rejected detections in red (debug mode only)
                if DEBUG_SOHO and debug_folder:
                    for detection in rejected_detections:
                        bbox = detection.bbox
                        cv2.rectangle(
                            vis_img,
                            (int(bbox[0]), int(bbox[1])),
                            (int(bbox[2]), int(bbox[3])),
                            (0, 0, 255),  # Red
                            2
                        )
                        # Add label
                        cv2.putText(
                            vis_img,
                            f"REJECT {detection.confidence:.2f}",
                            (int(bbox[0]), int(bbox[1]) - 5),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (0, 0, 255),
                            1
                        )
                    
                    # Draw kept detections in green
                    for detection in filtered_detections:
                        bbox = detection.bbox
                        # bbox is in padded image space now, convert back to slice space for visualization
                        bbox_slice = [bbox[0] - x1, bbox[1] - y1, bbox[2] - x1, bbox[3] - y1]
                        cv2.rectangle(
                            vis_img,
                            (int(bbox_slice[0]), int(bbox_slice[1])),
                            (int(bbox_slice[2]), int(bbox_slice[3])),
                            (0, 255, 0),  # Green
                            2
                        )
                        # Add label
                        cv2.putText(
                            vis_img,
                            f"KEEP {detection.confidence:.2f}",
                            (int(bbox_slice[0]), int(bbox_slice[1]) - 5),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (0, 255, 0),
                            1
                        )
                    
                    # Add slice info text
                    info_text = [
                        f"Slice {idx+1}/{len(slice_positions)}",
                        f"Position: ({x1},{y1})-({x2},{y2})",
                        f"Size: {slice_w_actual}x{slice_h_actual}",
                        f"Detected: {len(slice_detections)}",
                        f"Kept: {len(filtered_detections)}",
                        f"Rejected: {len(rejected_detections)}"
                    ]
                    
                    y_offset = 20
                    for text in info_text:
                        cv2.putText(
                            vis_img,
                            text,
                            (10, y_offset),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (255, 255, 255),
                            2
                        )
                        cv2.putText(
                            vis_img,
                            text,
                            (10, y_offset),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (0, 0, 0),
                            1
                        )
                        y_offset += 25
                    
                    # Save visualization
                    vis_path = debug_folder / f"slice_{idx+1:03d}.jpg"
                    cv2.imwrite(str(vis_path), vis_img)
                
                all_detections.extend(filtered_detections)
                print(f"Slice {idx+1}: {len(slice_detections)} detected, {len(filtered_detections)} after edge filter")
            
            print(f"SOHO: Total detections before merging: {len(all_detections)}")
            
            if not all_detections:
                self.status_label.setText("No detections found")
                QMessageBox.information(
                    self, "No Detections",
                    "SOHO did not detect any complete instances in the current frame."
                )
                return
            
            # FIRST: Clean overlapping contours from individual detections
            self.status_label.setText("Running SOHO inference (cleaning contours)...")
            QApplication.processEvents()
            cleaned_detections = self._clean_detections_contours(all_detections, overlap_threshold=0.15)
            print(f"SOHO: After contour cleaning: {len(cleaned_detections)} detections")
            
            # THEN: Merge duplicate detections
            self.status_label.setText("Running SOHO inference (merging duplicates)...")
            QApplication.processEvents()
            merged_detections = self._remove_duplicate_detections(cleaned_detections, iou_threshold=0.3)
            print(f"SOHO: After merging: {len(merged_detections)} detections")
            
            # Transform detections from padded space to original image space
            final_detections = []
            for detection in merged_detections:
                # Adjust bbox coordinates
                bbox = detection.bbox
                bbox_orig = [
                    max(0, bbox[0] - pad_size),
                    max(0, bbox[1] - pad_size),
                    min(orig_w, bbox[2] - pad_size),
                    min(orig_h, bbox[3] - pad_size)
                ]
                
                # Crop mask to original image region
                mask_orig = detection.mask[pad_size:pad_size+orig_h, pad_size:pad_size+orig_w]
                
                # Skip if mask is empty after cropping
                if np.sum(mask_orig > 0) == 0:
                    continue
                
                detection.bbox = bbox_orig
                detection.mask = mask_orig
                final_detections.append(detection)
            
            detections = final_detections
            print(f"SOHO: Final detections in original image space: {len(detections)}")
            
            # Create final result visualization (debug mode only)
            if DEBUG_SOHO and debug_folder:
                final_vis_img = original_img.copy()
                for i, detection in enumerate(detections):
                    bbox = detection.bbox
                    cv2.rectangle(
                        final_vis_img,
                        (int(bbox[0]), int(bbox[1])),
                        (int(bbox[2]), int(bbox[3])),
                        (0, 255, 0),  # Green
                        3
                    )
                    # Add detection number and confidence
                    cv2.putText(
                        final_vis_img,
                        f"#{i+1} {detection.confidence:.2f}",
                        (int(bbox[0]), int(bbox[1]) - 8),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 0),
                        2
                    )
                
                # Add summary text
                cv2.putText(
                    final_vis_img,
                    f"Final: {len(detections)} detections",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (0, 255, 0),
                    2
                )
                
                # Save final result
                final_path = debug_folder / "99_final_result.jpg"
                cv2.imwrite(str(final_path), final_vis_img)
                print(f"Saved final result to {final_path}")
            
            if not detections:
                self.status_label.setText("No detections found")
                QMessageBox.information(
                    self, "No Detections",
                    "SOHO did not detect any instances in the current frame."
                )
                return
            
            # Use tracker or add detections (same as SAHI)
            if self.tracking_enabled and self.current_video_id:
                tracker = self._get_or_create_tracker(self.current_video_id)
                
                matched_detections = tracker.match_detections_to_tracks(
                    detections,
                    self.current_frame_idx
                )
                
                existing_track_ids = set(tracker.active_tracks.keys())
                matched_ids = [track_id for _, track_id in matched_detections]
                num_matched = sum(1 for tid in matched_ids if tid in existing_track_ids or tid in tracker.lost_tracks)
                num_new = len(matched_detections) - num_matched
                
                print(f"SOHO Tracking: {num_matched} matched to existing tracks, {num_new} new tracks created")
                
                self.canvas.set_annotations([])
                
                for i, (detection, track_id) in enumerate(matched_detections):
                    if self.current_video_id not in self.video_mask_colors:
                        self.video_mask_colors[self.current_video_id] = {}
                    
                    color = self.video_mask_colors[self.current_video_id].get(track_id)
                    rebuild_viz = (i == len(matched_detections) - 1)
                    self.canvas.add_mask(detection.mask, mask_id=track_id, color=color, rebuild_viz=rebuild_viz)
                
                self._register_canvas_colors()
                
                self.video_next_mask_id[self.current_video_id] = tracker.next_track_id
                self.current_frame_modified = True
                
                self.status_label.setText(
                    f"✓ SOHO inference complete: {len(matched_detections)} instances detected (with tracking)"
                )
            else:
                # No tracking
                canvas_annotations = self.canvas.get_annotations()
                next_id = max([ann.get('instance_id', ann.get('mask_id', 0)) for ann in canvas_annotations], default=0) + 1
                
                self.canvas.set_annotations([])
                
                for i, detection in enumerate(detections):
                    mask_id = next_id + i
                    rebuild_viz = (i == len(detections) - 1)
                    self.canvas.add_mask(detection.mask, mask_id=mask_id, rebuild_viz=rebuild_viz)
                
                self._register_canvas_colors()
                
                self.current_frame_modified = True
                
                self.status_label.setText(
                    f"✓ SOHO inference complete: {len(detections)} instances detected"
                )
            
            # Detect markers (ArUco/QR codes) in new annotations
            self.detect_and_update_markers()
            
            self.update_instance_list_from_canvas()
            
            # Show summary
            num_detected = len(detections)
            tracking_msg = " (with ID tracking)" if self.tracking_enabled and self.current_video_id else ""
            debug_msg = f"\n\nDebug visualizations saved to:\n{debug_folder}" if DEBUG_SOHO and debug_folder else ""
            QMessageBox.information(
                self, "SOHO Inference Complete",
                f"SOHO detected {num_detected} instance(s){tracking_msg}.\n\n"
                f"Method: Custom slicing with edge filtering\n"
                f"Padding: {pad_size}px, Edge filter: {edge_filter}px\n"
                f"Slices processed: {len(slice_positions)}\n"
                f"Slice size: {slice_w}x{slice_h}, Overlap: {overlap_w:.0%}x{overlap_h:.0%}{debug_msg}\n\n"
                f"The detections have been added as new instances.\n"
                f"You can now edit, delete, or propagate them as needed."
            )
            
        except Exception as e:
            import traceback
            error_msg = traceback.format_exc()
            QMessageBox.critical(
                self, "SOHO Inference Error",
                f"Error running SOHO inference:\n{str(e)}\n\n{error_msg}"
            )
            self.status_label.setText("SOHO inference failed")
        finally:
            # Return focus to canvas so spacebar works for hiding masks
            self.canvas.setFocus()
    
    def propagate_soho_to_next_frame(self, target_frame_idx=None):
        """Propagate annotations to next frame using SOHO inference and ByteTrack matching
        
        Args:
            target_frame_idx: If provided, propagate to this specific frame index.
                            Otherwise, uses _get_next_frame_index() (respects filter).
        """
        if not self.yolo_sahi_toolbar.is_model_loaded():
            QMessageBox.warning(
                self, "No Model",
                "Please load a YOLO SAHI checkpoint first."
            )
            return
        
        # Check if we have current annotations
        current_annotations = self.canvas.get_annotations()
        if not current_annotations:
            QMessageBox.warning(
                self, "No Annotations",
                "No annotations to propagate. Annotate some instances first."
            )
            return
        
        # Check if tracking is enabled
        if not self.tracking_enabled or not self.current_video_id:
            QMessageBox.warning(
                self, "Tracking Required",
                "Please enable tracking mode first.\n\n"
                "Go to Video menu → Enable Track IDs to use propagation."
            )
            return
        
        # Find next frame
        if target_frame_idx is not None:
            # Use specified target frame (for sequential propagation)
            next_idx = target_frame_idx
            if next_idx >= len(self.frames) or next_idx < 0:
                self.status_label.setText("Invalid target frame index")
                return
        else:
            # Use filter-aware next frame (for manual single-frame propagation)
            next_idx = self._get_next_frame_index()
            if next_idx is None:
                self.status_label.setText("Already at last frame in current view")
                return
        
        try:
            import cv2
            
            # Save current frame annotations
            self.annotation_manager.set_frame_annotations(
                self.current_frame_idx, current_annotations, video_id=self.current_video_id
            )
            
            # Load next frame
            next_frame_path = self.frames[next_idx]
            next_frame_img = cv2.imread(str(next_frame_path))
            if next_frame_img is None:
                raise ValueError(f"Failed to load next frame: {next_frame_path}")
            
            orig_h, orig_w = next_frame_img.shape[:2]
            
            # Run SOHO on next frame (reuse SOHO logic inline)
            self.status_label.setText("Running SOHO on next frame...")
            QApplication.processEvents()
            
            # SOHO parameters
            pad_size = 100
            edge_filter = 50
            DEBUG_SOHO = False  # Disable debug output for propagation
            
            # Get slicing parameters
            sahi_params = self.yolo_sahi_toolbar.get_sahi_params()
            slice_h = sahi_params['slice_height']
            slice_w = sahi_params['slice_width']
            overlap_h = sahi_params['overlap_height_ratio']
            overlap_w = sahi_params['overlap_width_ratio']
            
            # Add black padding
            padded_img = cv2.copyMakeBorder(
                next_frame_img,
                pad_size, pad_size, pad_size, pad_size,
                cv2.BORDER_CONSTANT,
                value=0
            )
            padded_h, padded_w = padded_img.shape[:2]
            
            # Generate slice positions
            stride_h = int(slice_h * (1 - overlap_h))
            stride_w = int(slice_w * (1 - overlap_w))
            
            slice_positions = []
            y = 0
            while y < padded_h:
                x = 0
                while x < padded_w:
                    y2 = min(y + slice_h, padded_h)
                    x2 = min(x + slice_w, padded_w)
                    y1 = max(0, y2 - slice_h)
                    x1 = max(0, x2 - slice_w)
                    slice_positions.append((x1, y1, x2, y2))
                    if x2 >= padded_w:
                        break
                    x += stride_w
                if y2 >= padded_h:
                    break
                y += stride_h
            
            # Get model
            model = self.yolo_sahi_toolbar.get_model()
            
            # Process slices
            all_detections = []
            for idx, (x1, y1, x2, y2) in enumerate(slice_positions):
                self.status_label.setText(f"SOHO propagation: slice {idx+1}/{len(slice_positions)}...")
                QApplication.processEvents()
                
                slice_img = padded_img[y1:y2, x1:x2]
                
                results = model.predict(
                    source=slice_img,
                    conf=0.5,
                    iou=0.5,
                    retina_masks=True,
                    verbose=False
                )
                
                if not results or len(results) == 0 or results[0].masks is None:
                    continue
                
                slice_detections = self._yolo_results_to_detections(results[0], model)
                
                # Filter edge detections
                slice_h_actual = y2 - y1
                slice_w_actual = x2 - x1
                
                for detection in slice_detections:
                    bbox = detection.bbox
                    
                    if (bbox[0] < edge_filter or bbox[2] > slice_w_actual - edge_filter or
                        bbox[1] < edge_filter or bbox[3] > slice_h_actual - edge_filter):
                        continue
                    
                    # Transform to padded image space
                    detection.bbox = [
                        bbox[0] + x1,
                        bbox[1] + y1,
                        bbox[2] + x1,
                        bbox[3] + y1
                    ]
                    
                    full_mask = np.zeros((padded_h, padded_w), dtype=np.uint8)
                    full_mask[y1:y2, x1:x2] = detection.mask
                    detection.mask = full_mask
                    
                    all_detections.append(detection)
            
            print(f"SOHO Propagation: {len(all_detections)} detections before merging")
            
            if not all_detections:
                self.status_label.setText("No detections in next frame")
                QMessageBox.information(
                    self, "No Detections",
                    "SOHO did not detect any instances in the next frame."
                )
                return
            
            # FIRST: Clean overlapping contours from individual detections
            self.status_label.setText("Cleaning contours...")
            QApplication.processEvents()
            cleaned_detections = self._clean_detections_contours(all_detections, overlap_threshold=0.15)
            print(f"SOHO Propagation: {len(cleaned_detections)} after contour cleaning")
            
            # THEN: Merge duplicates
            self.status_label.setText("Merging duplicate detections...")
            QApplication.processEvents()
            merged_detections = self._remove_duplicate_detections(cleaned_detections, iou_threshold=0.3)
            print(f"SOHO Propagation: {len(merged_detections)} after merging")
            
            # Transform to original image space
            final_detections = []
            for detection in merged_detections:
                bbox = detection.bbox
                bbox_orig = [
                    max(0, bbox[0] - pad_size),
                    max(0, bbox[1] - pad_size),
                    min(orig_w, bbox[2] - pad_size),
                    min(orig_h, bbox[3] - pad_size)
                ]
                
                mask_orig = detection.mask[pad_size:pad_size+orig_h, pad_size:pad_size+orig_w]
                
                if np.sum(mask_orig > 0) == 0:
                    continue
                
                detection.bbox = bbox_orig
                detection.mask = mask_orig
                final_detections.append(detection)
            
            print(f"SOHO Propagation: {len(final_detections)} final detections")
            
            if not final_detections:
                self.status_label.setText("No valid detections")
                QMessageBox.information(
                    self, "No Detections",
                    "No valid detections after filtering."
                )
                return
            
            # Initialize tracker with current frame's annotations
            self.status_label.setText("Initializing tracker with current frame...")
            QApplication.processEvents()
            
            tracker = self._get_or_create_tracker(self.current_video_id)
            
            # Update tracker with current frame's tracks (without overwriting lost tracks)
            # Only update if this is a new track or if the track exists and needs updating
            current_track_ids = set()
            for ann in current_annotations:
                if 'mask' in ann:
                    mask_id = ann.get('mask_id', ann.get('instance_id', 1))
                    mask = ann['mask']
                    current_track_ids.add(mask_id)
                    
                    # Compute bbox from mask
                    if np.any(mask > 0):
                        y_indices, x_indices = np.where(mask > 0)
                        bbox = np.array([
                            float(x_indices.min()),
                            float(y_indices.min()),
                            float(x_indices.max()),
                            float(y_indices.max())
                        ])
                        
                        # Check if track already exists in active or lost tracks
                        if mask_id in tracker.active_tracks:
                            # Update existing active track
                            track = tracker.active_tracks[mask_id]
                            track.bbox = bbox
                            track.mask = mask
                            track.last_seen_frame = self.current_frame_idx
                            track.frames_lost = 0
                        elif mask_id in tracker.lost_tracks:
                            # Reactivate lost track
                            track = tracker.lost_tracks[mask_id]
                            track.bbox = bbox
                            track.mask = mask
                            track.last_seen_frame = self.current_frame_idx
                            track.frames_lost = 0
                            tracker.active_tracks[mask_id] = track
                            del tracker.lost_tracks[mask_id]
                        else:
                            # Create new track
                            track = Track(
                                track_id=mask_id,
                                bbox=bbox,
                                mask=mask,
                                last_seen_frame=self.current_frame_idx
                            )
                            tracker.active_tracks[mask_id] = track
                        
                        # Update next_track_id if needed
                        if mask_id >= tracker.next_track_id:
                            tracker.next_track_id = mask_id + 1
            
            print(f"SOHO Propagation: Tracker has {len(tracker.active_tracks)} active tracks and {len(tracker.lost_tracks)} lost tracks")
            
            # Match with tracker
            self.status_label.setText("Matching detections with tracker...")
            QApplication.processEvents()
            
            matched_detections = tracker.match_detections_to_tracks(
                final_detections,
                next_idx
            )
            
            existing_track_ids = set(ann.get('mask_id', ann.get('instance_id', 1)) for ann in current_annotations)
            matched_ids = [track_id for _, track_id in matched_detections]
            num_matched = sum(1 for tid in matched_ids if tid in existing_track_ids)
            num_new = len(matched_detections) - num_matched
            
            print(f"SOHO Propagation: {num_matched} matched to existing tracks, {num_new} new tracks")
            
            # Navigate to next frame
            self.load_frame(next_idx)
            
            # Clear canvas and add matched detections
            self.canvas.set_annotations([])
            
            for i, (detection, track_id) in enumerate(matched_detections):
                if self.current_video_id not in self.video_mask_colors:
                    self.video_mask_colors[self.current_video_id] = {}
                
                color = self.video_mask_colors[self.current_video_id].get(track_id)
                rebuild_viz = (i == len(matched_detections) - 1)
                self.canvas.add_mask(detection.mask, mask_id=track_id, color=color, rebuild_viz=rebuild_viz)
            
            self._register_canvas_colors()
            
            self.video_next_mask_id[self.current_video_id] = tracker.next_track_id
            self.current_frame_modified = True
            
            # Detect markers (ArUco/QR codes) in propagated annotations
            self.detect_and_update_markers()
            
            self.update_instance_list_from_canvas()
            
            self.status_label.setText(
                f"✓ SOHO propagated: {num_matched} tracked, {num_new} new ({len(matched_detections)} total)"
            )
            
        except Exception as e:
            import traceback
            error_msg = traceback.format_exc()
            QMessageBox.critical(
                self, "SOHO Propagation Error",
                f"Error during SOHO propagation:\n{str(e)}\n\n{error_msg}"
            )
            self.status_label.setText("SOHO propagation failed")
    
    def propagate_soho_to_selected(self):
        """Propagate annotations through frames until next selected frame using SOHO"""
        if not self.yolo_sahi_toolbar.is_model_loaded():
            QMessageBox.warning(
                self, "No Model",
                "Please load a YOLO SAHI checkpoint first."
            )
            return
        
        # Check if we have annotations
        current_annotations = self.canvas.get_annotations()
        if not current_annotations:
            QMessageBox.warning(
                self, "No Annotations",
                "No annotations to propagate. Annotate some instances first."
            )
            return
        
        # Check if tracking is enabled
        if not self.tracking_enabled or not self.current_video_id:
            QMessageBox.warning(
                self, "Tracking Required",
                "Please enable tracking mode first.\n\n"
                "Go to Video menu → Enable Track IDs to use propagation."
            )
            return
        
        # Find next selected frame (keyframe marked for annotation)
        next_selected_idx = None
        for idx in range(self.current_frame_idx + 1, len(self.frames)):
            if idx < len(self.frame_selected) and self.frame_selected[idx]:
                next_selected_idx = idx
                break
        
        if next_selected_idx is None:
            QMessageBox.information(
                self, "No Selected Frame",
                "No selected (train/val) keyframe found after current frame."
            )
            return
        
        # Calculate frames to propagate
        num_frames = next_selected_idx - self.current_frame_idx
        
        reply = QMessageBox.question(
            self, "Confirm Propagation",
            f"Propagate through {num_frames} frame(s) to next selected frame?\n\n"
            f"Current frame: {self.current_frame_idx + 1}\n"
            f"Target frame: {next_selected_idx + 1}\n\n"
            f"This will run SOHO inference on each frame and maintain track IDs.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        
        if reply != QMessageBox.StandardButton.Yes:
            return
        
        # Show progress dialog
        from PyQt6.QtWidgets import QProgressDialog
        progress = QProgressDialog(
            f"Propagating SOHO through {num_frames} frame(s)...",
            "Cancel",
            0,
            num_frames,
            self
        )
        progress.setWindowModality(Qt.WindowModality.WindowModal)
        progress.setMinimumDuration(0)
        progress.setValue(0)
        
        # Propagate frame by frame (sequentially, ignoring filter)
        successful_propagations = 0
        failed_frames = []
        was_canceled = False
        
        try:
            for i in range(num_frames):
                # Check if user canceled
                if progress.wasCanceled():
                    was_canceled = True
                    break
                
                current_frame = self.current_frame_idx
                target_frame = current_frame + 1
                
                # Update progress dialog
                progress.setLabelText(
                    f"Propagating SOHO to frame {target_frame + 1}...\n"
                    f"Frame {i + 1} of {num_frames}"
                )
                progress.setValue(i)
                QApplication.processEvents()
                
                # Run propagation to specific next frame (sequential, not filtered)
                try:
                    self.propagate_soho_to_next_frame(target_frame_idx=target_frame)
                    
                    # Check if we actually moved to the next frame
                    if self.current_frame_idx == target_frame:
                        successful_propagations += 1
                    else:
                        failed_frames.append(target_frame)
                        print(f"Warning: Propagation did not advance frame (current: {self.current_frame_idx}, expected: {target_frame})")
                        break
                    
                except Exception as e:
                    failed_frames.append(target_frame)
                    print(f"Error propagating to frame {target_frame}: {e}")
                    break
            
            # Complete the progress dialog
            progress.setValue(num_frames)
            progress.close()
            
            # Save the final frame's annotations explicitly
            if self.current_frame_modified and self.current_video_id:
                try:
                    annotations = self.canvas.get_annotations()
                    if annotations:
                        frame_idx_in_video = self._get_frame_idx_in_video(self.current_frame_idx)
                        self.annotation_manager.save_frame_annotations(
                            self.project_path, self.current_video_id,
                            frame_idx_in_video, annotations
                        )
                        print(f"  Saved final frame {self.current_frame_idx} (video frame {frame_idx_in_video})")
                except Exception as e:
                    print(f"Warning: Failed to save final frame: {e}")
            
            # Show summary
            if was_canceled:
                QMessageBox.information(
                    self, "Propagation Canceled",
                    f"Propagation was canceled by user.\n\n"
                    f"Successfully propagated through {successful_propagations} frame(s) before canceling.\n"
                    f"All processed frames have been saved."
                )
                self.status_label.setText(f"SOHO propagation canceled ({successful_propagations} frames saved)")
            elif failed_frames:
                QMessageBox.warning(
                    self, "Propagation Completed with Errors",
                    f"Successfully propagated through {successful_propagations} frame(s).\n\n"
                    f"Failed at frame(s): {', '.join(map(str, failed_frames))}\n\n"
                    f"All successfully processed frames have been saved."
                )
                self.status_label.setText(f"SOHO propagated through {successful_propagations} frames (with errors, saved)")
            else:
                QMessageBox.information(
                    self, "Propagation Complete",
                    f"Successfully propagated through {successful_propagations} frame(s) using SOHO.\n\n"
                    f"Track IDs have been maintained across frames."
                )
                self.status_label.setText(f"✓ SOHO propagated through {successful_propagations} frames (all saved)")
            
        except Exception as e:
            # Close progress dialog on error
            if 'progress' in locals():
                progress.close()
            
            import traceback
            error_msg = traceback.format_exc()
            QMessageBox.critical(
                self, "SOHO Propagation Error",
                f"Error during batch propagation:\n{str(e)}\n\n{error_msg}"
            )
            self.status_label.setText("SOHO propagation failed")
    
    def propagate_soho_through_video(self):
        """Propagate annotations through all remaining frames in the video using SOHO"""
        if not self.yolo_sahi_toolbar.is_model_loaded():
            QMessageBox.warning(
                self, "No Model",
                "Please load a YOLO SAHI checkpoint first."
            )
            return
        
        # Check if we have annotations
        current_annotations = self.canvas.get_annotations()
        if not current_annotations:
            QMessageBox.warning(
                self, "No Annotations",
                "No annotations to propagate. Annotate some instances first."
            )
            return
        
        # Check if tracking is enabled
        if not self.tracking_enabled or not self.current_video_id:
            QMessageBox.warning(
                self, "Tracking Required",
                "Please enable tracking mode first.\n\n"
                "Go to Video menu → Enable Track IDs to use propagation."
            )
            return
        
        # Calculate total frames to propagate (from current to end of video)
        # Get the last frame index in the current video
        last_frame_idx = None
        for idx in range(len(self.frames) - 1, self.current_frame_idx, -1):
            if idx < len(self.frame_video_ids) and self.frame_video_ids[idx] == self.current_video_id:
                last_frame_idx = idx
                break
        
        if last_frame_idx is None or last_frame_idx <= self.current_frame_idx:
            QMessageBox.information(
                self, "Already at End",
                "Already at or past the last frame of the current video."
            )
            return
        
        num_frames = last_frame_idx - self.current_frame_idx
        
        reply = QMessageBox.question(
            self, "Confirm Video Propagation",
            f"Propagate through all remaining {num_frames} frame(s) in the video?\n\n"
            f"Current frame: {self.current_frame_idx + 1}\n"
            f"Last frame: {last_frame_idx + 1}\n"
            f"Video ID: {self.current_video_id}\n\n"
            f"This will run SOHO inference on each frame and maintain track IDs.\n"
            f"This operation may take several minutes depending on the number of frames.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        
        if reply != QMessageBox.StandardButton.Yes:
            return
        
        # Create progress dialog
        from PyQt6.QtWidgets import QProgressDialog
        progress = QProgressDialog(
            "Initializing SOHO propagation...",
            "Cancel",
            0,
            num_frames,
            self
        )
        progress.setWindowTitle("Propagating Through Video")
        progress.setWindowModality(Qt.WindowModality.WindowModal)
        progress.setMinimumDuration(0)  # Show immediately
        progress.setValue(0)
        
        # Propagate frame by frame
        successful_propagations = 0
        failed_frames = []
        cancelled = False
        
        try:
            import time
            start_time = time.time()
            
            for i in range(num_frames):
                # Check if cancelled
                if progress.wasCanceled():
                    cancelled = True
                    break
                
                current_frame = self.current_frame_idx
                
                # Update progress dialog
                elapsed = time.time() - start_time
                avg_time_per_frame = elapsed / (i + 1) if i > 0 else 0
                remaining_frames = num_frames - i
                estimated_remaining = avg_time_per_frame * remaining_frames
                
                progress.setLabelText(
                    f"Propagating frame {i+1}/{num_frames}\n"
                    f"Current frame: {current_frame + 1} → {current_frame + 2}\n"
                    f"Elapsed: {elapsed:.1f}s | Remaining: ~{estimated_remaining:.0f}s\n"
                    f"Average: {avg_time_per_frame:.2f}s/frame"
                )
                progress.setValue(i)
                QApplication.processEvents()
                
                self.status_label.setText(f"Propagating frame {i+1}/{num_frames}...")
                QApplication.processEvents()
                
                # Run propagation to next frame
                try:
                    self.propagate_soho_to_next_frame()
                    
                    # Check if we actually moved to the next frame
                    if self.current_frame_idx == current_frame + 1:
                        successful_propagations += 1
                    else:
                        failed_frames.append(current_frame + 1)
                        print(f"Warning: Propagation did not advance frame (stuck at {self.current_frame_idx})")
                        break
                    
                    # Check if we've left the current video
                    if self.current_frame_idx >= len(self.frame_video_ids) or \
                       self.frame_video_ids[self.current_frame_idx] != self.current_video_id:
                        print(f"Reached end of video at frame {self.current_frame_idx}")
                        break
                    
                except Exception as e:
                    failed_frames.append(current_frame + 1)
                    print(f"Error propagating to frame {current_frame + 1}: {e}")
                    
                    # Close progress dialog temporarily
                    progress.hide()
                    
                    # Ask if user wants to continue
                    reply = QMessageBox.question(
                        self, "Propagation Error",
                        f"Error at frame {current_frame + 1}: {str(e)}\n\n"
                        f"Continue with remaining frames?",
                        QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
                    )
                    
                    if reply != QMessageBox.StandardButton.Yes:
                        progress.close()
                        break
                    
                    # Show progress dialog again
                    progress.show()
            
            # Close progress dialog
            progress.setValue(num_frames)
            progress.close()
            
            # Save the final frame's annotations explicitly
            if self.current_frame_modified and self.current_video_id:
                try:
                    annotations = self.canvas.get_annotations()
                    if annotations:
                        frame_idx_in_video = self._get_frame_idx_in_video(self.current_frame_idx)
                        self.annotation_manager.save_frame_annotations(
                            self.project_path, self.current_video_id,
                            frame_idx_in_video, annotations
                        )
                        print(f"  Saved final frame {self.current_frame_idx} (video frame {frame_idx_in_video})")
                except Exception as e:
                    print(f"Warning: Failed to save final frame: {e}")
            
            # Show summary
            total_time = time.time() - start_time
            
            if cancelled:
                QMessageBox.information(
                    self, "Propagation Cancelled",
                    f"Propagation was cancelled by user.\n\n"
                    f"Successfully propagated through {successful_propagations} frame(s) in {total_time:.1f} seconds before cancelling.\n"
                    f"All processed frames have been saved."
                )
                self.status_label.setText(
                    f"Propagation cancelled after {successful_propagations} frames (all saved)"
                )
            elif failed_frames:
                QMessageBox.warning(
                    self, "Propagation Completed with Errors",
                    f"Successfully propagated through {successful_propagations} frame(s) in {total_time:.1f} seconds.\n\n"
                    f"Failed at frame(s): {', '.join(map(str, failed_frames))}\n\n"
                    f"All successfully processed frames have been saved."
                )
                self.status_label.setText(
                    f"✓ SOHO propagated {successful_propagations} frames (with errors, saved)"
                )
            else:
                QMessageBox.information(
                    self, "Video Propagation Complete",
                    f"Successfully propagated through {successful_propagations} frame(s) using SOHO.\n\n"
                    f"Total time: {total_time:.1f} seconds\n"
                    f"Average: {total_time/successful_propagations:.2f} seconds/frame\n\n"
                    f"Track IDs have been maintained across all frames."
                )
                self.status_label.setText(
                    f"✓ SOHO propagated through entire video: {successful_propagations} frames in {total_time:.1f}s (all saved)"
                )
            
        except Exception as e:
            import traceback
            error_msg = traceback.format_exc()
            progress.close()
            QMessageBox.critical(
                self, "SOHO Propagation Error",
                f"Error during video propagation:\n{str(e)}\n\n{error_msg}"
            )
            self.status_label.setText("SOHO video propagation failed")
    
    def run_beehavesque_inference(self):
        """Run Beehavesque inference with temporal context (currently a stub)"""
        QMessageBox.information(
            self, "Coming Soon",
            "Beehavesque inference is under development.\n\n"
            "This will use previous, current, and next frames as RGB channels for inference."
        )
    
    def run_beehavesque_soho_inference(self):
        """Run Beehavesque SOHO inference on the current frame with temporal context"""
        # Debug mode flag - set to True to save slice visualizations
        DEBUG_SOHO = True
        
        if not self.yolo_beehavesque_toolbar.is_model_loaded():
            QMessageBox.warning(
                self, "No Model",
                "Please load a YOLO Beehavesque checkpoint first."
            )
            return
        
        if not self.frames or self.current_frame_idx >= len(self.frames):
            QMessageBox.warning(
                self, "No Frame",
                "No frame is currently loaded."
            )
            return
        
        try:
            import cv2
            import shutil
            from datetime import datetime
            
            # Setup debug folder if debug mode is enabled
            debug_folder = None
            if DEBUG_SOHO:
                # Use project-based tmp folder
                debug_folder = Path.cwd() / "tmp" / "beehavesque_soho_debug"
                
                # Clear and recreate the folder
                if debug_folder.exists():
                    shutil.rmtree(debug_folder)
                debug_folder.mkdir(parents=True, exist_ok=True)
                
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                print(f"\nBeehavesque SOHO Debug: Saving visualizations to {debug_folder}")
                print(f"Timestamp: {timestamp}")
            
            # Get current frame path
            frame_path = self.frames[self.current_frame_idx]
            
            # Create temporal image (prev, current, next frames)
            self.status_label.setText("Creating temporal image (prev/current/next)...")
            QApplication.processEvents()
            
            temporal_img = self._create_beehavesque_temporal_image(frame_path)
            if temporal_img is None:
                raise ValueError("Failed to create temporal image")
            
            orig_h, orig_w = temporal_img.shape[:2]
            
            # Save temporal image for debugging
            if DEBUG_SOHO and debug_folder:
                temporal_vis_path = debug_folder / "00_temporal_image.jpg"
                cv2.imwrite(str(temporal_vis_path), temporal_img)
                print(f"Saved temporal image to {temporal_vis_path}")
                
                # Create README in debug folder
                readme_text = """BeeHaveSquE SOHO Debug Visualizations
======================================

This folder contains debug visualizations from BeeHaveSquE SOHO inference.

Files:
------
00_temporal_image.jpg
  - Temporal image created from prev/current/next frames
  - Used as input for BeeHaveSquE model inference

00_slice_grid_overview.jpg
  - Overview of the padded image with all slice boundaries shown in magenta
  - Green rectangle shows the original image boundary within the padding
  - Numbers indicate slice processing order

slice_001.jpg, slice_002.jpg, etc.
  - Individual slice visualizations
  - Yellow rectangle: Edge filter boundary (50px margin)
  - RED boxes: Detections that were REJECTED (too close to slice edge)
  - GREEN boxes: Detections that were KEPT (away from edges)
  - Text overlay shows slice position, size, and detection counts

99_final_result.jpg
  - Final merged detections on the original temporal image (no padding)
  - Green boxes show all detections after edge filtering and merging
  - Numbers indicate detection ID and confidence score

Parameters:
-----------
Model: BeeHaveSquE (temporal context: prev/current/next frames)
Padding: 100px around image (BORDER_CONSTANT, black)
Edge Filter: 50px margin per slice
Detections within 50px of any slice edge are removed entirely
After filtering, overlapping detections are merged using IoU threshold

Color Legend:
-------------
- Magenta: Slice boundaries (overview)
- Yellow: Edge filter zone (individual slices)
- Red: Rejected detections (too close to edge)
- Green: Kept detections (away from edges)
- Green (final): Final merged detections
"""
                readme_path = debug_folder / "README.txt"
                with open(readme_path, 'w') as f:
                    f.write(readme_text)
            
            # SOHO parameters
            pad_size = 100  # Padding around image
            edge_filter = 50  # Filter detections within this margin of each slice's edges
            
            # Get slicing parameters from toolbar
            sahi_params = self.yolo_beehavesque_toolbar.get_sahi_params()
            slice_h = sahi_params['slice_height']
            slice_w = sahi_params['slice_width']
            overlap_h = sahi_params['overlap_height_ratio']
            overlap_w = sahi_params['overlap_width_ratio']
            
            self.status_label.setText("Running Beehavesque SOHO inference (creating padded image)...")
            QApplication.processEvents()
            
            # Add black padding to avoid edge artifacts
            padded_img = cv2.copyMakeBorder(
                temporal_img,
                pad_size, pad_size, pad_size, pad_size,
                cv2.BORDER_CONSTANT,
                value=0
            )
            
            padded_h, padded_w = padded_img.shape[:2]
            
            # Calculate slice positions with overlap
            stride_h = int(slice_h * (1 - overlap_h))
            stride_w = int(slice_w * (1 - overlap_w))
            
            slice_positions = []
            y = 0
            while y < padded_h:
                x = 0
                while x < padded_w:
                    # Calculate slice bounds
                    y2 = min(y + slice_h, padded_h)
                    x2 = min(x + slice_w, padded_w)
                    
                    # Adjust start position if we hit the edge
                    y1 = max(0, y2 - slice_h)
                    x1 = max(0, x2 - slice_w)
                    
                    slice_positions.append((x1, y1, x2, y2))
                    
                    if x2 >= padded_w:
                        break
                    x += stride_w
                
                if y2 >= padded_h:
                    break
                y += stride_h
            
            print(f"Beehavesque SOHO: Created {len(slice_positions)} slices from {padded_w}x{padded_h} padded image")
            print(f"Slice size: {slice_w}x{slice_h}, Stride: {stride_w}x{stride_h}")
            
            # Create overview visualization showing all slice positions
            if DEBUG_SOHO and debug_folder:
                overview_img = padded_img.copy()
                
                # Draw black borders to show the 100px padding region
                # Top border
                cv2.rectangle(overview_img, (0, 0), (padded_w, pad_size), (0, 0, 0), -1)
                # Bottom border
                cv2.rectangle(overview_img, (0, padded_h - pad_size), (padded_w, padded_h), (0, 0, 0), -1)
                # Left border
                cv2.rectangle(overview_img, (0, 0), (pad_size, padded_h), (0, 0, 0), -1)
                # Right border
                cv2.rectangle(overview_img, (padded_w - pad_size, 0), (padded_w, padded_h), (0, 0, 0), -1)
                
                # Add text labels for padding regions
                cv2.putText(overview_img, "100px PADDING", (padded_w // 2 - 100, pad_size // 2),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                cv2.putText(overview_img, "100px PADDING", (padded_w // 2 - 100, padded_h - pad_size // 2),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                cv2.putText(overview_img, "100px", (pad_size // 4, padded_h // 2),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.putText(overview_img, "100px", (padded_w - pad_size + 10, padded_h // 2),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                for idx, (x1, y1, x2, y2) in enumerate(slice_positions):
                    # Draw slice boundary
                    cv2.rectangle(overview_img, (x1, y1), (x2, y2), (255, 0, 255), 2)  # Magenta
                    # Draw slice number
                    cv2.putText(
                        overview_img,
                        str(idx + 1),
                        (x1 + 10, y1 + 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (255, 255, 255),
                        2
                    )
                
                # Draw original image boundary within padding (green)
                cv2.rectangle(
                    overview_img,
                    (pad_size, pad_size),
                    (pad_size + orig_w, pad_size + orig_h),
                    (0, 255, 0),  # Green for original image boundary
                    3
                )
                
                # Save overview
                overview_path = debug_folder / "00_slice_grid_overview.jpg"
                cv2.imwrite(str(overview_path), overview_img)
                print(f"Saved slice grid overview to {overview_path}")
            
            # Get model
            model = self.yolo_beehavesque_toolbar.get_model()
            
            # Process each slice
            all_detections = []
            
            for idx, (x1, y1, x2, y2) in enumerate(slice_positions):
                self.status_label.setText(f"Running Beehavesque SOHO (slice {idx+1}/{len(slice_positions)})...")
                QApplication.processEvents()
                
                # Extract slice
                slice_img = padded_img[y1:y2, x1:x2]
                
                # Run YOLO inference on this slice
                results = model.predict(
                    source=slice_img,
                    conf=0.5,
                    iou=0.8,
                    retina_masks=True,
                    verbose=False
                )
                
                if not results or len(results) == 0 or results[0].masks is None:
                    continue
                
                # Convert results to detections
                slice_detections = self._yolo_results_to_detections(results[0], model)
                
                # Create visualization of this slice (only if debug mode)
                if DEBUG_SOHO and debug_folder:
                    vis_img = slice_img.copy()
                    
                    # Draw edge filter boundary (yellow)
                    cv2.rectangle(
                        vis_img,
                        (edge_filter, edge_filter),
                        (slice_img.shape[1] - edge_filter, slice_img.shape[0] - edge_filter),
                        (0, 255, 255),  # Yellow
                        2
                    )
                
                # Filter detections near slice edges
                slice_h_actual = y2 - y1
                slice_w_actual = x2 - x1
                
                filtered_detections = []
                rejected_detections = []
                for detection in slice_detections:
                    bbox = detection.bbox  # [x1_local, y1_local, x2_local, y2_local]
                    
                    # Check if detection is too close to any edge of this slice
                    is_too_close = (bbox[0] < edge_filter or bbox[2] > slice_w_actual - edge_filter or
                                   bbox[1] < edge_filter or bbox[3] > slice_h_actual - edge_filter)
                    
                    if is_too_close:
                        # Skip - too close to edge, likely partial detection
                        rejected_detections.append(detection)
                        continue
                    
                    # Transform coordinates from slice space to padded image space
                    detection.bbox = [
                        bbox[0] + x1,
                        bbox[1] + y1,
                        bbox[2] + x1,
                        bbox[3] + y1
                    ]
                    
                    # Transform mask to padded image coordinates
                    full_mask = np.zeros((padded_h, padded_w), dtype=np.uint8)
                    full_mask[y1:y2, x1:x2] = detection.mask
                    detection.mask = full_mask
                    
                    filtered_detections.append(detection)
                
                # Draw rejected detections in red (debug mode only)
                if DEBUG_SOHO and debug_folder:
                    for detection in rejected_detections:
                        bbox = detection.bbox
                        cv2.rectangle(
                            vis_img,
                            (int(bbox[0]), int(bbox[1])),
                            (int(bbox[2]), int(bbox[3])),
                            (0, 0, 255),  # Red
                            2
                        )
                        # Add label
                        cv2.putText(
                            vis_img,
                            f"REJECT {detection.confidence:.2f}",
                            (int(bbox[0]), int(bbox[1]) - 5),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (0, 0, 255),
                            1
                        )
                    
                    # Draw kept detections in green
                    for detection in filtered_detections:
                        bbox = detection.bbox
                        # bbox is in padded image space now, convert back to slice space for visualization
                        bbox_slice = [bbox[0] - x1, bbox[1] - y1, bbox[2] - x1, bbox[3] - y1]
                        cv2.rectangle(
                            vis_img,
                            (int(bbox_slice[0]), int(bbox_slice[1])),
                            (int(bbox_slice[2]), int(bbox_slice[3])),
                            (0, 255, 0),  # Green
                            2
                        )
                        # Add label
                        cv2.putText(
                            vis_img,
                            f"KEEP {detection.confidence:.2f}",
                            (int(bbox_slice[0]), int(bbox_slice[1]) - 5),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (0, 255, 0),
                            1
                        )
                    
                    # Add slice info text
                    info_text = [
                        f"Slice {idx+1}/{len(slice_positions)}",
                        f"Position: ({x1},{y1})-({x2},{y2})",
                        f"Size: {slice_w_actual}x{slice_h_actual}",
                        f"Detected: {len(slice_detections)}",
                        f"Kept: {len(filtered_detections)}",
                        f"Rejected: {len(rejected_detections)}"
                    ]
                    
                    y_offset = 20
                    for text in info_text:
                        cv2.putText(
                            vis_img,
                            text,
                            (10, y_offset),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (255, 255, 255),
                            2
                        )
                        cv2.putText(
                            vis_img,
                            text,
                            (10, y_offset),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (0, 0, 0),
                            1
                        )
                        y_offset += 25
                    
                    # Save visualization
                    vis_path = debug_folder / f"slice_{idx+1:03d}.jpg"
                    cv2.imwrite(str(vis_path), vis_img)
                
                all_detections.extend(filtered_detections)
                print(f"Slice {idx+1}: {len(slice_detections)} detected, {len(filtered_detections)} after edge filter")
            
            print(f"Beehavesque SOHO: Total detections before merging: {len(all_detections)}")
            
            if not all_detections:
                self.status_label.setText("No detections found")
                QMessageBox.information(
                    self, "No Detections",
                    "Beehavesque SOHO did not detect any instances in the current frame."
                )
                return
            
            # Clean overlapping contours from individual detections
            self.status_label.setText("Running Beehavesque SOHO (cleaning contours)...")
            QApplication.processEvents()
            cleaned_detections = self._clean_detections_contours(all_detections, overlap_threshold=0.15)
            print(f"Beehavesque SOHO: After contour cleaning: {len(cleaned_detections)} detections")
            
            # Merge duplicate detections
            self.status_label.setText("Running Beehavesque SOHO (merging duplicates)...")
            QApplication.processEvents()
            merged_detections = self._remove_duplicate_detections(cleaned_detections, iou_threshold=0.3)
            print(f"Beehavesque SOHO: After merging: {len(merged_detections)} detections")
            
            # Transform detections from padded space to original image space
            final_detections = []
            for detection in merged_detections:
                # Adjust bbox coordinates
                bbox = detection.bbox
                bbox_orig = [
                    max(0, bbox[0] - pad_size),
                    max(0, bbox[1] - pad_size),
                    min(orig_w, bbox[2] - pad_size),
                    min(orig_h, bbox[3] - pad_size)
                ]
                
                # Crop mask to original image region
                mask_orig = detection.mask[pad_size:pad_size+orig_h, pad_size:pad_size+orig_w]
                
                # Skip if mask is empty after cropping
                if np.sum(mask_orig > 0) == 0:
                    continue
                
                detection.bbox = bbox_orig
                detection.mask = mask_orig
                final_detections.append(detection)
            
            detections = final_detections
            print(f"Beehavesque SOHO: Final detections: {len(detections)}")
            
            # Create final result visualization (debug mode only)
            if DEBUG_SOHO and debug_folder:
                # Load original current frame for final visualization
                current_frame_img = cv2.imread(str(frame_path))
                if current_frame_img is not None:
                    final_vis_img = current_frame_img.copy()
                    for i, detection in enumerate(detections):
                        bbox = detection.bbox
                        cv2.rectangle(
                            final_vis_img,
                            (int(bbox[0]), int(bbox[1])),
                            (int(bbox[2]), int(bbox[3])),
                            (0, 255, 0),  # Green
                            3
                        )
                        # Add detection number and confidence
                        cv2.putText(
                            final_vis_img,
                            f"#{i+1} {detection.confidence:.2f}",
                            (int(bbox[0]), int(bbox[1]) - 8),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            (0, 255, 0),
                            2
                        )
                    
                    # Add summary text
                    cv2.putText(
                        final_vis_img,
                        f"Final: {len(detections)} detections",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.0,
                        (0, 255, 0),
                        2
                    )
                    
                    # Save final result
                    final_path = debug_folder / "99_final_result.jpg"
                    cv2.imwrite(str(final_path), final_vis_img)
                    print(f"Saved final result to {final_path}")
            
            if not detections:
                self.status_label.setText("No detections found")
                QMessageBox.information(
                    self, "No Detections",
                    "Beehavesque SOHO did not detect any instances in the current frame."
                )
                return
            
            # Use tracker or add detections (same as SAHI)
            if self.tracking_enabled and self.current_video_id:
                tracker = self._get_or_create_tracker(self.current_video_id)
                
                matched_detections = tracker.match_detections_to_tracks(
                    detections,
                    self.current_frame_idx
                )
                
                existing_track_ids = set(tracker.active_tracks.keys())
                matched_ids = [track_id for _, track_id in matched_detections]
                num_matched = sum(1 for tid in matched_ids if tid in existing_track_ids or tid in tracker.lost_tracks)
                num_new = len(matched_detections) - num_matched
                
                print(f"Beehavesque SOHO Tracking: {num_matched} matched to existing tracks, {num_new} new tracks created")
                
                self.canvas.set_annotations([])
                
                for i, (detection, track_id) in enumerate(matched_detections):
                    if self.current_video_id not in self.video_mask_colors:
                        self.video_mask_colors[self.current_video_id] = {}
                    
                    color = self.video_mask_colors[self.current_video_id].get(track_id)
                    rebuild_viz = (i == len(matched_detections) - 1)
                    self.canvas.add_mask(detection.mask, mask_id=track_id, color=color, rebuild_viz=rebuild_viz)
                
                self._register_canvas_colors()
                
                self.video_next_mask_id[self.current_video_id] = tracker.next_track_id
                self.current_frame_modified = True
                
                self.status_label.setText(
                    f"✓ Beehavesque SOHO: {len(matched_detections)} instances detected (with tracking)"
                )
            else:
                # No tracking
                canvas_annotations = self.canvas.get_annotations()
                next_id = max([ann.get('instance_id', ann.get('mask_id', 0)) for ann in canvas_annotations], default=0) + 1
                
                self.canvas.set_annotations([])
                
                for i, detection in enumerate(detections):
                    mask_id = next_id + i
                    rebuild_viz = (i == len(detections) - 1)
                    self.canvas.add_mask(detection.mask, mask_id=mask_id, rebuild_viz=rebuild_viz)
                
                self._register_canvas_colors()
                
                self.current_frame_modified = True
                
                self.status_label.setText(
                    f"✓ Beehavesque SOHO: {len(detections)} instances detected"
                )
            
            # Detect markers (ArUco/QR codes) in new annotations
            self.detect_and_update_markers()
            
            self.update_instance_list_from_canvas()
            
            # Show summary
            num_detected = len(detections)
            tracking_msg = " (with ID tracking)" if self.tracking_enabled and self.current_video_id else ""
            debug_msg = f"\n\nDebug visualizations saved to:\n{debug_folder}" if DEBUG_SOHO and debug_folder else ""
            QMessageBox.information(
                self, "Beehavesque SOHO Complete",
                f"Beehavesque SOHO detected {num_detected} instance(s){tracking_msg}.\n\n"
                f"Method: Temporal context (prev/current/next frames) + SOHO slicing\n"
                f"Padding: {pad_size}px, Edge filter: {edge_filter}px\n"
                f"Slices processed: {len(slice_positions)}\n"
                f"Slice size: {slice_w}x{slice_h}, Overlap: {overlap_w:.0%}x{overlap_h:.0%}{debug_msg}\n\n"
                f"The detections have been added as new instances.\n"
                f"You can now edit, delete, or propagate them as needed."
            )
            
        except Exception as e:
            import traceback
            error_msg = f"Beehavesque SOHO inference failed:\n{str(e)}\n\n{traceback.format_exc()}"
            QMessageBox.critical(
                self, "Beehavesque SOHO Error",
                error_msg
            )
            self.status_label.setText("Beehavesque SOHO failed")
        finally:
            # Return focus to canvas so spacebar works for hiding masks
            self.canvas.setFocus()
    
    def _create_beehavesque_temporal_image(self, current_frame_path):
        """Create temporal image from prev/current/next frames for beehavesque inference
        
        Args:
            current_frame_path: Path to the current frame
            
        Returns:
            np.ndarray: 3-channel RGB image where R=prev, G=current, B=next (all grayscale)
        """
        import cv2
        
        # Get video frames directory
        current_frame_path = Path(current_frame_path)
        video_frames_dir = current_frame_path.parent
        
        # Get all frames in this video sorted by name
        all_frames = sorted(list(video_frames_dir.glob('*.jpg')) + 
                          list(video_frames_dir.glob('*.png')))
        
        # Find current frame index
        try:
            current_idx = all_frames.index(current_frame_path)
        except ValueError:
            # Frame not in list, use current frame for all channels
            current_img = cv2.imread(str(current_frame_path), cv2.IMREAD_GRAYSCALE)
            if current_img is None:
                return None
            return np.stack([current_img, current_img, current_img], axis=2)
        
        # Get previous frame (or duplicate current if at start)
        if current_idx > 0:
            prev_frame_path = all_frames[current_idx - 1]
        else:
            prev_frame_path = current_frame_path
        
        # Get next frame (or duplicate current if at end)
        if current_idx < len(all_frames) - 1:
            next_frame_path = all_frames[current_idx + 1]
        else:
            next_frame_path = current_frame_path
        
        # Load frames as grayscale
        prev_img = cv2.imread(str(prev_frame_path), cv2.IMREAD_GRAYSCALE)
        current_img = cv2.imread(str(current_frame_path), cv2.IMREAD_GRAYSCALE)
        next_img = cv2.imread(str(next_frame_path), cv2.IMREAD_GRAYSCALE)
        
        if prev_img is None or current_img is None or next_img is None:
            return None
        
        # Ensure all frames have the same dimensions
        h, w = current_img.shape
        if prev_img.shape != (h, w):
            prev_img = cv2.resize(prev_img, (w, h))
        if next_img.shape != (h, w):
            next_img = cv2.resize(next_img, (w, h))
        
        # Stack as RGB channels: R=prev, G=current, B=next
        temporal_img = np.stack([prev_img, current_img, next_img], axis=2)
        
        return temporal_img
    
    def on_beehavesque_box_inference_requested(self):
        """Handle request to run beehavesque inference on the drawn box"""
        # Get the box from canvas
        box = self.canvas.get_inference_box()
        if not box:
            QMessageBox.warning(
                self, "No Box",
                "Please draw a box first."
            )
            return
        
        x1, y1, x2, y2 = box
        print(f"Running beehavesque inference on box: ({x1}, {y1}) to ({x2}, {y2})")
        
        # Run inference
        self.run_beehavesque_inference_on_box(x1, y1, x2, y2)
        
        # Clear the box and reset both toolbars
        self.canvas.clear_inference_box()
        self.yolo_sahi_toolbar.set_box_drawn(False)
        self.yolo_beehavesque_toolbar.set_box_drawn(False)
        self.canvas.set_tool('polygon')
    
    def run_beehavesque_inference_on_box(self, x1, y1, x2, y2):
        """Run Beehavesque SOHO inference on a user-specified box region"""
        if not self.yolo_beehavesque_toolbar.is_model_loaded():
            QMessageBox.warning(
                self, "No Model",
                "Please load a YOLO Beehavesque checkpoint first."
            )
            return
        
        if not self.frames or self.current_frame_idx >= len(self.frames):
            QMessageBox.warning(
                self, "No Frame",
                "No frame is currently loaded."
            )
            return
        
        try:
            import cv2
            
            # Get current frame path
            frame_path = self.frames[self.current_frame_idx]
            
            # Create temporal image
            self.status_label.setText("Creating temporal image...")
            QApplication.processEvents()
            
            temporal_img = self._create_beehavesque_temporal_image(frame_path)
            if temporal_img is None:
                raise ValueError("Failed to create temporal image")
            
            # Get current annotations before inference
            current_annotations = self.canvas.get_annotations()
            
            # Normalize box coordinates
            x1, x2 = int(min(x1, x2)), int(max(x1, x2))
            y1, y2 = int(min(y1, y2)), int(max(y1, y2))
            
            # Ensure box is within image bounds
            h, w = temporal_img.shape[:2]
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(w, x2)
            y2 = min(h, y2)
            
            print(f"Running beehavesque box inference on region: ({x1}, {y1}) to ({x2}, {y2})")
            self.status_label.setText(f"Running Beehavesque inference on box region...")
            QApplication.processEvents()
            
            # Crop temporal image to box region
            cropped_temporal = temporal_img[y1:y2, x1:x2]
            
            # SOHO parameters for box region
            pad_size = 50  # Smaller padding for box inference
            edge_filter = 25  # Smaller edge filter
            
            # Get slicing parameters
            sahi_params = self.yolo_beehavesque_toolbar.get_sahi_params()
            slice_h = sahi_params['slice_height']
            slice_w = sahi_params['slice_width']
            overlap_h = sahi_params['overlap_height_ratio']
            overlap_w = sahi_params['overlap_width_ratio']
            
            # Add padding to cropped region
            padded_crop = cv2.copyMakeBorder(
                cropped_temporal,
                pad_size, pad_size, pad_size, pad_size,
                cv2.BORDER_CONSTANT,
                value=0
            )
            
            padded_h, padded_w = padded_crop.shape[:2]
            
            # Calculate slice positions
            stride_h = int(slice_h * (1 - overlap_h))
            stride_w = int(slice_w * (1 - overlap_w))
            
            slice_positions = []
            y = 0
            while y < padded_h:
                x = 0
                while x < padded_w:
                    y2 = min(y + slice_h, padded_h)
                    x2 = min(x + slice_w, padded_w)
                    y1_slice = max(0, y2 - slice_h)
                    x1_slice = max(0, x2 - slice_w)
                    slice_positions.append((x1_slice, y1_slice, x2, y2))
                    if x2 >= padded_w:
                        break
                    x += stride_w
                if y2 >= padded_h:
                    break
                y += stride_h
            
            # Get model
            model = self.yolo_beehavesque_toolbar.get_model()
            
            # Process slices
            all_detections = []
            for idx, (sx1, sy1, sx2, sy2) in enumerate(slice_positions):
                self.status_label.setText(f"Beehavesque box inference: slice {idx+1}/{len(slice_positions)}...")
                QApplication.processEvents()
                
                slice_img = padded_crop[sy1:sy2, sx1:sx2]
                
                results = model.predict(
                    source=slice_img,
                    conf=0.5,
                    iou=0.5,
                    retina_masks=True,
                    verbose=False
                )
                
                if not results or len(results) == 0 or results[0].masks is None:
                    continue
                
                slice_detections = self._yolo_results_to_detections(results[0], model)
                
                # Filter edge detections
                slice_h_actual = sy2 - sy1
                slice_w_actual = sx2 - sx1
                
                for detection in slice_detections:
                    bbox = detection.bbox
                    
                    if (bbox[0] < edge_filter or bbox[2] > slice_w_actual - edge_filter or
                        bbox[1] < edge_filter or bbox[3] > slice_h_actual - edge_filter):
                        continue
                    
                    # Transform to padded crop space
                    detection.bbox = [
                        bbox[0] + sx1,
                        bbox[1] + sy1,
                        bbox[2] + sx1,
                        bbox[3] + sy1
                    ]
                    
                    full_mask = np.zeros((padded_h, padded_w), dtype=np.uint8)
                    full_mask[sy1:sy2, sx1:sx2] = detection.mask
                    detection.mask = full_mask
                    
                    all_detections.append(detection)
            
            print(f"Beehavesque box inference: {len(all_detections)} detections before merging")
            
            if not all_detections:
                self.status_label.setText("No detections in box region")
                QMessageBox.information(
                    self, "No Detections",
                    "Beehavesque did not detect any instances in the box region."
                )
                return
            
            # Clean and merge detections
            cleaned_detections = self._clean_detections_contours(all_detections, overlap_threshold=0.15)
            merged_detections = self._remove_duplicate_detections(cleaned_detections, iou_threshold=0.3)
            
            print(f"Beehavesque box inference: {len(merged_detections)} after merging")
            
            # Transform detections to original image coordinates
            crop_h, crop_w = cropped_temporal.shape[:2]
            final_detections = []
            
            for detection in merged_detections:
                bbox = detection.bbox
                bbox_crop = [
                    max(0, bbox[0] - pad_size),
                    max(0, bbox[1] - pad_size),
                    min(crop_w, bbox[2] - pad_size),
                    min(crop_h, bbox[3] - pad_size)
                ]
                
                mask_crop = detection.mask[pad_size:pad_size+crop_h, pad_size:pad_size+crop_w]
                
                if np.sum(mask_crop > 0) == 0:
                    continue
                
                # Transform to full image coordinates
                full_mask = np.zeros((h, w), dtype=np.uint8)
                full_mask[y1:y2, x1:x2] = mask_crop
                
                bbox_full = [
                    bbox_crop[0] + x1,
                    bbox_crop[1] + y1,
                    bbox_crop[2] + x1,
                    bbox_crop[3] + y1
                ]
                
                detection.bbox = bbox_full
                detection.mask = full_mask
                final_detections.append(detection)
            
            print(f"Beehavesque box inference: {len(final_detections)} final detections")
            
            # Identify existing instances to remove/keep
            contained_mask_ids = []  # IDs to remove
            outside_annotations = []  # Annotations completely outside box
            
            box_mask = np.zeros((h, w), dtype=np.uint8)
            box_mask[y1:y2, x1:x2] = 1
            
            for ann in current_annotations:
                mask = ann['mask']
                mask_area = np.sum(mask > 0)
                overlap_area = np.sum((mask > 0) & (box_mask > 0))
                
                if overlap_area == 0:
                    # No overlap - keep as is
                    outside_annotations.append(ann)
                elif overlap_area == mask_area:
                    # Fully contained - mark for removal
                    contained_mask_ids.append(ann.get('mask_id'))
                    print(f"Removing fully contained instance {ann.get('mask_id')}")
            
            # Clear canvas and add back non-contained annotations
            self.canvas.clear_annotations()
            for ann in outside_annotations:
                self.canvas.add_annotation(ann['mask'], mask_id=ann.get('mask_id'))
            
            # Add new detections
            for detection in final_detections:
                self.canvas.add_annotation(detection.mask)
            
            self.canvas.update()
            self.status_label.setText(
                f"✓ Beehavesque box inference: Added {len(final_detections)} detections"
            )
            
        except Exception as e:
            import traceback
            error_msg = f"Beehavesque box inference failed:\n{str(e)}\n\n{traceback.format_exc()}"
            QMessageBox.critical(
                self, "Box Inference Error",
                error_msg
            )
            self.status_label.setText("Box inference failed")
        finally:
            # Return focus to canvas so spacebar works for hiding masks
            self.canvas.setFocus()
    
    def propagate_beehavesque_to_next_frame(self, target_frame_idx=None):
        """Propagate annotations to next frame using Beehavesque inference and ByteTrack matching
        
        Automatically finds the last annotated frame and propagates from there.
        
        Args:
            target_frame_idx: If provided, propagate to this specific frame index.
                            Otherwise, uses _get_next_frame_index() (respects filter).
        """
        if not self.yolo_beehavesque_toolbar.is_model_loaded():
            QMessageBox.warning(
                self, "No Model",
                "Please load a YOLO Beehavesque checkpoint first."
            )
            return
        
        # Check if tracking is enabled
        if not self.tracking_enabled or not self.current_video_id:
            QMessageBox.warning(
                self, "Tracking Required",
                "Please enable tracking mode first.\n\n"
                "Go to Video menu → Enable Track IDs to use propagation."
            )
            return
        
        # Find the last annotated frame (source)
        last_annotated_idx = None
        for idx in range(self.current_frame_idx, -1, -1):
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
            QMessageBox.warning(
                self, "No Annotations",
                "No annotated frames found. Please annotate some instances first."
            )
            return
        
        # Load annotations from last annotated frame
        frame_idx_in_video = self._get_frame_idx_in_video(last_annotated_idx)
        source_annotations = self.annotation_manager.load_frame_annotations(
            self.project_path, self.current_video_id, frame_idx_in_video
        )
        
        if not source_annotations:
            QMessageBox.warning(
                self, "No Annotations",
                "No annotations found in last annotated frame."
            )
            return
        
        print(f"Propagating from last annotated frame {last_annotated_idx} (video frame {frame_idx_in_video})")
        
        # Determine target frame
        if target_frame_idx is not None:
            # Use specified target frame (for sequential propagation)
            next_idx = target_frame_idx
            if next_idx >= len(self.frames) or next_idx < 0:
                self.status_label.setText("Invalid target frame index")
                return
        else:
            # Use next frame after last annotated frame
            next_idx = last_annotated_idx + 1
            if next_idx >= len(self.frames):
                self.status_label.setText("Already at last frame")
                return
        
        print(f"Target frame: {next_idx}")
        
        try:
            import cv2
            
            # Get next frame path
            next_frame_path = self.frames[next_idx]
            
            # Create temporal image for next frame
            self.status_label.setText(f"Creating temporal image for frame {next_idx}...")
            QApplication.processEvents()
            
            temporal_img = self._create_beehavesque_temporal_image(next_frame_path)
            if temporal_img is None:
                raise ValueError("Failed to create temporal image for next frame")
            
            orig_h, orig_w = temporal_img.shape[:2]
            
            # Run SOHO on next frame
            self.status_label.setText(f"Running Beehavesque SOHO on frame {next_idx}...")
            QApplication.processEvents()
            
            # SOHO parameters
            pad_size = 100
            edge_filter = 50
            
            # Get slicing parameters
            sahi_params = self.yolo_beehavesque_toolbar.get_sahi_params()
            slice_h = sahi_params['slice_height']
            slice_w = sahi_params['slice_width']
            overlap_h = sahi_params['overlap_height_ratio']
            overlap_w = sahi_params['overlap_width_ratio']
            
            # Add black padding
            padded_img = cv2.copyMakeBorder(
                temporal_img,
                pad_size, pad_size, pad_size, pad_size,
                cv2.BORDER_CONSTANT,
                value=0
            )
            padded_h, padded_w = padded_img.shape[:2]
            
            # Generate slice positions
            stride_h = int(slice_h * (1 - overlap_h))
            stride_w = int(slice_w * (1 - overlap_w))
            
            slice_positions = []
            y = 0
            while y < padded_h:
                x = 0
                while x < padded_w:
                    y2 = min(y + slice_h, padded_h)
                    x2 = min(x + slice_w, padded_w)
                    y1 = max(0, y2 - slice_h)
                    x1 = max(0, x2 - slice_w)
                    slice_positions.append((x1, y1, x2, y2))
                    if x2 >= padded_w:
                        break
                    x += stride_w
                if y2 >= padded_h:
                    break
                y += stride_h
            
            # Get model
            model = self.yolo_beehavesque_toolbar.get_model()
            
            # Process slices
            all_detections = []
            for idx, (x1, y1, x2, y2) in enumerate(slice_positions):
                self.status_label.setText(f"Beehavesque propagation: slice {idx+1}/{len(slice_positions)}...")
                QApplication.processEvents()
                
                slice_img = padded_img[y1:y2, x1:x2]
                
                results = model.predict(
                    source=slice_img,
                    conf=0.5,
                    iou=0.5,
                    retina_masks=True,
                    verbose=False
                )
                
                if not results or len(results) == 0 or results[0].masks is None:
                    continue
                
                slice_detections = self._yolo_results_to_detections(results[0], model)
                
                # Filter edge detections
                slice_h_actual = y2 - y1
                slice_w_actual = x2 - x1
                
                for detection in slice_detections:
                    bbox = detection.bbox
                    
                    if (bbox[0] < edge_filter or bbox[2] > slice_w_actual - edge_filter or
                        bbox[1] < edge_filter or bbox[3] > slice_h_actual - edge_filter):
                        continue
                    
                    # Transform to padded image space
                    detection.bbox = [
                        bbox[0] + x1,
                        bbox[1] + y1,
                        bbox[2] + x1,
                        bbox[3] + y1
                    ]
                    
                    full_mask = np.zeros((padded_h, padded_w), dtype=np.uint8)
                    full_mask[y1:y2, x1:x2] = detection.mask
                    detection.mask = full_mask
                    
                    all_detections.append(detection)
            
            print(f"Beehavesque Propagation: {len(all_detections)} detections before merging")
            
            if not all_detections:
                self.status_label.setText("No detections in next frame")
                QMessageBox.information(
                    self, "No Detections",
                    "Beehavesque SOHO did not detect any instances in the next frame."
                )
                return
            
            # Clean overlapping contours
            self.status_label.setText("Cleaning contours...")
            QApplication.processEvents()
            cleaned_detections = self._clean_detections_contours(all_detections, overlap_threshold=0.15)
            print(f"Beehavesque Propagation: {len(cleaned_detections)} after contour cleaning")
            
            # Merge duplicates
            self.status_label.setText("Merging duplicate detections...")
            QApplication.processEvents()
            merged_detections = self._remove_duplicate_detections(cleaned_detections, iou_threshold=0.3)
            print(f"Beehavesque Propagation: {len(merged_detections)} after merging")
            
            # Transform to original image space
            final_detections = []
            for detection in merged_detections:
                bbox = detection.bbox
                bbox_orig = [
                    max(0, bbox[0] - pad_size),
                    max(0, bbox[1] - pad_size),
                    min(orig_w, bbox[2] - pad_size),
                    min(orig_h, bbox[3] - pad_size)
                ]
                
                mask_orig = detection.mask[pad_size:pad_size+orig_h, pad_size:pad_size+orig_w]
                
                if np.sum(mask_orig > 0) == 0:
                    continue
                
                detection.bbox = bbox_orig
                detection.mask = mask_orig
                final_detections.append(detection)
            
            print(f"Beehavesque Propagation: {len(final_detections)} final detections")
            
            if not final_detections:
                self.status_label.setText("No valid detections")
                return
            
            # Initialize tracker and match detections to tracks
            self.status_label.setText("Matching detections to existing tracks...")
            QApplication.processEvents()
            
            tracker = self._get_or_create_tracker(self.current_video_id)
            
            # Manually initialize tracker with existing track IDs from source frame
            # This ensures continuity of instance IDs across frames
            source_track_ids = []
            for ann in source_annotations:
                if 'mask' in ann:
                    mask = ann['mask']
                    mask_id = ann.get('mask_id', ann.get('instance_id', 1))
                    
                    if np.any(mask > 0):
                        coords = np.where(mask > 0)
                        y1, y2 = coords[0].min(), coords[0].max()
                        x1, x2 = coords[1].min(), coords[1].max()
                        bbox = np.array([x1, y1, x2, y2])
                        
                        # Create Track object with existing mask_id
                        track = Track(
                            track_id=mask_id,
                            bbox=bbox,
                            mask=mask,
                            last_seen_frame=last_annotated_idx
                        )
                        
                        # Add to active tracks with the correct ID
                        tracker.active_tracks[mask_id] = track
                        source_track_ids.append(mask_id)
                        
                        # Update next_track_id to avoid conflicts
                        if mask_id >= tracker.next_track_id:
                            tracker.next_track_id = mask_id + 1
            
            print(f"Initialized tracker with {len(source_track_ids)} tracks from source frame: {source_track_ids}")
            
            # Match next frame detections to existing tracks
            matched_results = tracker.match_detections_to_tracks(final_detections, next_idx)
            
            # Count how many matched to existing vs new tracks
            matched_ids = [track_id for _, track_id in matched_results]
            num_matched = sum(1 for tid in matched_ids if tid in source_track_ids)
            num_new = len(matched_results) - num_matched
            
            print(f"Beehavesque Propagation: {num_matched} matched to existing tracks, {num_new} new tracks created")
            
            # Navigate to next frame (this auto-saves the previous frame and loads the image)
            self.load_frame(next_idx)
            
            # Clear existing annotations and add matched ones
            self.canvas.set_annotations([])
            
            for i, (detection, track_id) in enumerate(matched_results):
                # Preserve colors for existing track IDs
                if self.current_video_id not in self.video_mask_colors:
                    self.video_mask_colors[self.current_video_id] = {}
                
                color = self.video_mask_colors[self.current_video_id].get(track_id)
                rebuild_viz = (i == len(matched_results) - 1)  # Only rebuild on last
                self.canvas.add_mask(detection.mask, mask_id=track_id, color=color, rebuild_viz=rebuild_viz)
            
            # Register any new colors that were generated
            self._register_canvas_colors()
            
            # Update next mask ID for this video
            self.video_next_mask_id[self.current_video_id] = tracker.next_track_id
            
            # Mark frame as modified so annotations will be saved
            self.current_frame_modified = True
            
            # Detect markers (ArUco/QR codes) in propagated annotations
            self.detect_and_update_markers()
            
            # Update instance list
            self.update_instance_list_from_canvas()
            
            self.status_label.setText(
                f"✓ Beehavesque propagated: {num_matched} tracked, {num_new} new ({len(matched_results)} total)"
            )
            print(f"Propagation complete: {num_matched} tracked instances, {num_new} new")
            
        except Exception as e:
            import traceback
            error_msg = f"Beehavesque propagation failed:\n{str(e)}\n\n{traceback.format_exc()}"
            QMessageBox.critical(
                self, "Propagation Error",
                error_msg
            )
            self.status_label.setText("Propagation failed")
    
    def propagate_beehavesque_to_selected(self):
        """Propagate Beehavesque through all frames from last annotated to selected frame"""
        if not self.yolo_beehavesque_toolbar.is_model_loaded():
            QMessageBox.warning(
                self, "No Model",
                "Please load a YOLO Beehavesque checkpoint first."
            )
            return
        
        # Get target frame from user
        target_frame, ok = QInputDialog.getInt(
            self,
            "Select Target Frame",
            f"Enter frame number to propagate to (0-{len(self.frames)-1}):",
            value=self.current_frame_idx + 1,
            min=0,
            max=len(self.frames) - 1
        )
        
        if ok:
            # Find the last annotated frame
            last_annotated_idx = None
            for idx in range(self.current_frame_idx, -1, -1):
                if idx < len(self.frame_video_ids) and self.frame_video_ids[idx] == self.current_video_id:
                    frame_idx_in_video = self._get_frame_idx_in_video(idx)
                    annotations = self.annotation_manager.load_frame_annotations(
                        self.project_path, self.current_video_id, frame_idx_in_video
                    )
                    if annotations and len(annotations) > 0:
                        last_annotated_idx = idx
                        break
            
            if last_annotated_idx is None:
                QMessageBox.warning(
                    self, "No Annotations",
                    "No annotated frames found. Please annotate some instances first."
                )
                return
            
            if target_frame <= last_annotated_idx:
                QMessageBox.warning(
                    self, "Invalid Target",
                    f"Target frame ({target_frame}) must be after last annotated frame ({last_annotated_idx})."
                )
                return
            
            # Propagate through all intermediate frames sequentially with progress dialog
            from PyQt6.QtWidgets import QProgressDialog
            import time
            
            total_frames = target_frame - last_annotated_idx
            print(f"Propagating from frame {last_annotated_idx} through to frame {target_frame}")
            
            progress = QProgressDialog("Initializing propagation...", "Cancel", 0, total_frames, self)
            progress.setWindowTitle("BeeHaveSquE Propagation")
            progress.setWindowModality(Qt.WindowModality.WindowModal)
            progress.setMinimumDuration(0)  # Show immediately
            
            start_time = time.time()
            frame_times = []
            
            try:
                for i, frame_idx in enumerate(range(last_annotated_idx + 1, target_frame + 1), start=1):
                    if progress.wasCanceled():
                        # Save current frame before canceling
                        if self.current_frame_modified and self.current_video_id:
                            try:
                                annotations = self.canvas.get_annotations()
                                if annotations:
                                    frame_idx_in_video = self._get_frame_idx_in_video(self.current_frame_idx)
                                    self.annotation_manager.save_frame_annotations(
                                        self.project_path, self.current_video_id,
                                        frame_idx_in_video, annotations
                                    )
                                    print(f"  Saved current frame before canceling: {self.current_frame_idx} (video frame {frame_idx_in_video})")
                            except Exception as e:
                                print(f"Warning: Failed to save before cancel: {e}")
                        
                        self.status_label.setText(f"Propagation canceled at frame {frame_idx-1} (all processed frames saved)")
                        return
                    
                    frame_start = time.time()
                    
                    print(f"Propagating to frame {frame_idx}...")
                    self.propagate_beehavesque_to_next_frame(target_frame_idx=frame_idx)
                    
                    frame_time = time.time() - frame_start
                    frame_times.append(frame_time)
                    
                    # Calculate ETA based on average frame time
                    avg_time = sum(frame_times) / len(frame_times)
                    remaining_frames = total_frames - i
                    eta_seconds = avg_time * remaining_frames
                    
                    if eta_seconds < 60:
                        eta_str = f"{int(eta_seconds)}s"
                    else:
                        eta_str = f"{int(eta_seconds // 60)}m {int(eta_seconds % 60)}s"
                    
                    progress.setValue(i)
                    progress.setLabelText(
                        f"Processing frame {i}/{total_frames}\n"
                        f"Time per frame: {frame_time:.2f}s\n"
                        f"ETA: {eta_str}"
                    )
                    QApplication.processEvents()
                
                progress.setValue(total_frames)
                
                # Save the final frame's annotations explicitly
                if self.current_frame_modified and self.current_video_id:
                    try:
                        annotations = self.canvas.get_annotations()
                        if annotations:
                            frame_idx_in_video = self._get_frame_idx_in_video(self.current_frame_idx)
                            self.annotation_manager.save_frame_annotations(
                                self.project_path, self.current_video_id,
                                frame_idx_in_video, annotations
                            )
                            print(f"  Saved final frame {self.current_frame_idx} (video frame {frame_idx_in_video})")
                    except Exception as e:
                        print(f"Warning: Failed to save final frame: {e}")
                
                total_time = time.time() - start_time
                self.status_label.setText(
                    f"✓ Propagated through frames {last_annotated_idx+1} to {target_frame} "
                    f"({total_time:.1f}s total, all saved)"
                )
            finally:
                progress.close()
    
    def propagate_beehavesque_through_video(self):
        """Propagate Beehavesque through entire video"""
        if not self.yolo_beehavesque_toolbar.is_model_loaded():
            QMessageBox.warning(
                self, "No Model",
                "Please load a YOLO Beehavesque checkpoint first."
            )
            return
        
        # Check if we have current annotations
        current_annotations = self.canvas.get_annotations()
        if not current_annotations:
            QMessageBox.warning(
                self, "No Annotations",
                "No annotations to propagate. Annotate some instances first."
            )
            return
        
        # Check if tracking is enabled
        if not self.tracking_enabled or not self.current_video_id:
            QMessageBox.warning(
                self, "Tracking Required",
                "Please enable tracking mode first.\n\n"
                "Go to Video menu → Enable Track IDs to use propagation."
            )
            return
        
        # Ask for confirmation
        reply = QMessageBox.question(
            self,
            "Propagate Through Video",
            f"This will propagate annotations from frame {self.current_frame_idx} "
            f"through the rest of the video ({len(self.frames) - self.current_frame_idx - 1} frames).\n\n"
            "This may take several minutes. Continue?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        
        if reply != QMessageBox.StandardButton.Yes:
            return
        
        from PyQt6.QtWidgets import QProgressDialog
        import time
        
        start_frame = self.current_frame_idx
        total_frames = len(self.frames) - start_frame - 1
        
        progress = QProgressDialog("Initializing propagation...", "Cancel", 0, total_frames, self)
        progress.setWindowTitle("BeeHaveSquE Propagation")
        progress.setWindowModality(Qt.WindowModality.WindowModal)
        progress.setMinimumDuration(0)  # Show immediately
        
        start_time = time.time()
        frame_times = []
        
        try:
            # Propagate through each subsequent frame
            target_frame = start_frame + 1
            frames_processed = 0
            
            while target_frame < len(self.frames):
                if progress.wasCanceled():
                    # Save current frame before canceling
                    if self.current_frame_modified and self.current_video_id:
                        try:
                            annotations = self.canvas.get_annotations()
                            if annotations:
                                frame_idx_in_video = self._get_frame_idx_in_video(self.current_frame_idx)
                                self.annotation_manager.save_frame_annotations(
                                    self.project_path, self.current_video_id,
                                    frame_idx_in_video, annotations
                                )
                                print(f"  Saved current frame before canceling: {self.current_frame_idx} (video frame {frame_idx_in_video})")
                        except Exception as e:
                            print(f"Warning: Failed to save before cancel: {e}")
                    
                    self.status_label.setText(f"Propagation canceled at frame {target_frame} (all processed frames saved)")
                    return
                
                frame_start = time.time()
                
                # Propagate to next frame
                self.propagate_beehavesque_to_next_frame(target_frame_idx=target_frame)
                
                frame_time = time.time() - frame_start
                frame_times.append(frame_time)
                frames_processed += 1
                
                # Calculate ETA based on average frame time
                avg_time = sum(frame_times) / len(frame_times)
                remaining_frames = total_frames - frames_processed
                eta_seconds = avg_time * remaining_frames
                
                if eta_seconds < 60:
                    eta_str = f"{int(eta_seconds)}s"
                else:
                    eta_str = f"{int(eta_seconds // 60)}m {int(eta_seconds % 60)}s"
                
                progress.setValue(frames_processed)
                progress.setLabelText(
                    f"Processing frame {frames_processed}/{total_frames}\n"
                    f"Time per frame: {frame_time:.2f}s\n"
                    f"ETA: {eta_str}"
                )
                QApplication.processEvents()
                
                # Check if propagation was successful (we should now be on target_frame)
                if self.current_frame_idx != target_frame:
                    break
                
                target_frame += 1
            
            progress.setValue(total_frames)
            
            # Save the final frame's annotations explicitly
            if self.current_frame_modified and self.current_video_id:
                try:
                    annotations = self.canvas.get_annotations()
                    if annotations:
                        frame_idx_in_video = self._get_frame_idx_in_video(self.current_frame_idx)
                        self.annotation_manager.save_frame_annotations(
                            self.project_path, self.current_video_id,
                            frame_idx_in_video, annotations
                        )
                        print(f"  Saved final frame {self.current_frame_idx} (video frame {frame_idx_in_video})")
                except Exception as e:
                    print(f"Warning: Failed to save final frame: {e}")
            
            total_time = time.time() - start_time
            self.status_label.setText(
                f"✓ Beehavesque propagation complete: processed {frames_processed} frames ({total_time:.1f}s total, all saved)"
            )
            
            QMessageBox.information(
                self, "Propagation Complete",
                f"Successfully propagated annotations through {frames_processed} frames in {total_time:.1f}s\n"
                f"Average: {total_time/frames_processed:.2f}s per frame\n\n"
                f"All frames have been saved."
            )
            
        except Exception as e:
            import traceback
            error_msg = f"Batch propagation failed:\n{str(e)}\n\n{traceback.format_exc()}"
            QMessageBox.critical(
                self, "Propagation Error",
                error_msg
            )
            self.status_label.setText("Batch propagation failed")
        finally:
            progress.close()
    
    def run_beehavesque_validation(self):
        """Run full pipeline validation on validation set"""
        if not self.yolo_beehavesque_toolbar.is_model_loaded():
            QMessageBox.warning(
                self, "No Model",
                "Please load a YOLO Beehavesque checkpoint first."
            )
            return
        
        if not self.project_path:
            QMessageBox.warning(
                self, "No Project",
                "Please open a project first."
            )
            return
        
        # Find validation videos (videos with at least one validation frame with annotations)
        validation_videos = []
        print(f"\nScanning for validation videos...")
        
        # Get all validation videos from project
        all_val_videos = self.project_manager.get_videos_by_split('val')
        print(f"Total validation videos in project (from get_videos_by_split): {len(all_val_videos)}")
        print(f"Video IDs returned: {all_val_videos}")
        
        for idx, video_id in enumerate(all_val_videos):
            print(f"\n[{idx+1}/{len(all_val_videos)}] Checking video: {video_id}")
            
            # Check for annotations in PNG+JSON format
            png_annotations_dir = self.project_path / 'annotations/png' / video_id
            json_annotations_dir = self.project_path / 'annotations/json' / video_id
            
            print(f"  PNG annotations dir: {png_annotations_dir} (exists: {png_annotations_dir.exists()})")
            print(f"  JSON annotations dir: {json_annotations_dir} (exists: {json_annotations_dir.exists()})")
            
            # Collect all annotation files (need both PNG and JSON)
            annotation_files = []
            
            # Check PNG+JSON format (need both files)
            if png_annotations_dir.exists() and json_annotations_dir.exists():
                png_files = list(png_annotations_dir.glob('frame_*.png'))
                json_files = list(json_annotations_dir.glob('frame_*.json'))
                # Only count files that have both PNG and JSON
                png_stems = {f.stem for f in png_files}
                json_stems = {f.stem for f in json_files}
                valid_png_stems = png_stems & json_stems
                annotation_files.extend([(png_annotations_dir / f'{stem}.png', 'png') for stem in valid_png_stems])
                print(f"  Found {len(valid_png_stems)} PNG+JSON annotation pairs")
            else:
                print(f"  Annotation directories not found for {video_id}")
            
            if not annotation_files:
                print(f"  ✗ {video_id}: No annotation files found")
                continue
            
            # Check which frames have valid annotations
            val_frame_count = 0
            has_gt_annotations = False
            
            for ann_file, format_type in annotation_files:
                # Extract frame number from filename (e.g., frame_000123.pkl -> 123)
                frame_idx = int(ann_file.stem.split('_')[1])
                
                # Load and check if it has any instances
                annotations = self.annotation_manager.load_frame_annotations(
                    self.project_path, video_id, frame_idx
                )
                if annotations and len(annotations) > 0:
                    val_frame_count += 1
                    has_gt_annotations = True
            
            print(f"  Valid annotated frames: {val_frame_count}")
            
            if has_gt_annotations:
                validation_videos.append(video_id)
                print(f"  ✓ {video_id}: {val_frame_count} annotated frames - ADDED TO VALIDATION LIST")
            else:
                print(f"  ✗ {video_id}: {len(annotation_files)} annotation files (no valid annotations)")
        
        print(f"\n{'='*60}")
        print(f"FINAL: Found {len(validation_videos)} videos with validation data")
        print(f"Video IDs: {validation_videos}")
        print(f"{'='*60}\n")
        
        if not validation_videos:
            QMessageBox.warning(
                self, "No Validation Data",
                "No validation frames with annotations found in the project.\n\n"
                "Please mark some frames as 'val' split and ensure they have ground truth annotations."
            )
            return
        
        # Show configuration dialog
        config_dialog = ValidationConfigDialog(self)
        
        if config_dialog.exec():
            config = config_dialog.get_config()
            
            print(f"Starting validation on {len(validation_videos)} videos:")
            for vid in validation_videos:
                print(f"  - {vid}")
            
            # Create progress dialog
            progress_dialog = ValidationProgressDialog(self)
            progress_dialog.setWindowTitle("BeeHaveSquE Pipeline Validation")
            
            # Create validation worker
            worker = ValidationWorker(self, config, validation_videos)
            progress_dialog.set_worker(worker)
            
            # Start validation
            worker.start()
            
            # Show progress dialog (blocks until closed)
            progress_dialog.exec()
            
            # Handle completion
            if progress_dialog.validation_completed:
                # Validation completed successfully
                results_path = progress_dialog.results_path
                final_metrics = progress_dialog.final_metrics
                
                # Build completion message
                msg = "BeeHaveSquE pipeline validation completed successfully!\n\n"
                msg += f"Results saved to:\n{results_path}\n\n"
                
                if final_metrics:
                    msg += "Overall Metrics:\n"
                    msg += f"  MOTA: {final_metrics.get('mota', 0):.3f}\n"
                    msg += f"  IDF1: {final_metrics.get('idf1', 0):.3f}\n"
                    msg += f"  Precision: {final_metrics.get('precision', 0):.3f}\n"
                    msg += f"  Recall: {final_metrics.get('recall', 0):.3f}\n"
                    msg += f"  ID Switches: {final_metrics.get('id_switches', 0)}\n"
                    msg += f"  Total Frames: {final_metrics.get('total_frames', 0)}\n"
                
                QMessageBox.information(
                    self,
                    "Validation Complete",
                    msg
                )
                
                self.status_label.setText(f"✓ Validation complete: MOTA={final_metrics.get('mota', 0):.3f}")
            
            elif progress_dialog.validation_failed:
                # Show error message
                error_msg = progress_dialog.error_message or "Unknown error"
                QMessageBox.critical(
                    self,
                    "Validation Failed",
                    f"BeeHaveSquE validation failed:\n\n{error_msg}\n\n"
                    "Check the console for detailed error messages."
                )
                self.status_label.setText("Validation failed")
    
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
                if 'mask' in ann:
                    mask_id = ann.get('mask_id', ann.get('instance_id', 1))
                    mask = ann['mask']
                    
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
                        continue  # Skip empty masks
                    
                    track = Track(
                        track_id=mask_id,
                        bbox=bbox,
                        mask=mask,
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
            # Rebuild the canvas with updated IDs
            self.canvas.set_annotations([])
            
            for i, (detection, annotation) in enumerate(zip(current_detections, current_annotations)):
                if i in detection_to_track_id:
                    # This detection matched - use the matched track ID
                    new_mask_id = detection_to_track_id[i]
                else:
                    # Unmatched - keep original ID
                    new_mask_id = annotation.get('mask_id', annotation.get('instance_id', i + 1))
                
                # Get or preserve color
                if self.current_video_id not in self.video_mask_colors:
                    self.video_mask_colors[self.current_video_id] = {}
                
                color = self.video_mask_colors[self.current_video_id].get(new_mask_id)
                rebuild_viz = (i == len(current_detections) - 1)  # Only rebuild on last
                self.canvas.add_mask(detection.mask, mask_id=new_mask_id, color=color, rebuild_viz=rebuild_viz)
            
            # Register any new colors that were generated
            self._register_canvas_colors()
            
            # Mark frame as modified so annotations will be saved
            self.current_frame_modified = True
            
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
    
    def refine_selected_mask(self):
        """Refine the currently selected instance mask using YOLO"""
        if not self.yolo_refine_toolbar.is_model_loaded():
            QMessageBox.warning(
                self, "No Model",
                "Please load a YOLO checkpoint first."
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
            
            self.status_label.setText("Running YOLO refinement on crop...")
            QApplication.processEvents()
            
            success, message, stats = self._refine_mask_by_index(selected_idx)
            
            if success:
                # Update visualization
                self.canvas.update_mask_visualization(selected_idx)
                self.canvas.annotation_changed.emit()
                
                # Show success message
                self.status_label.setText(
                    f"✓ Mask refined (IoU: {stats['iou']:.3f}, "
                    f"area: {stats['refined_area']}px, Δ{stats['area_diff']:+d}px)"
                )
            else:
                # Show failure message
                self.status_label.setText(f"Refinement failed: {message}")
                
                if "No detections" in message:
                    QMessageBox.information(
                        self, "No Detections",
                        "YOLO did not detect any instances in the crop.\n\n"
                        "Try adjusting the crop padding or using a different model."
                    )
                elif "No match" in message:
                    QMessageBox.information(
                        self, "No Match",
                        f"Could not find a prediction with sufficient overlap.\n\n"
                        f"{message}\n\n"
                        f"The original mask may be very different from what the model predicts."
                    )
                else:
                    QMessageBox.warning(
                        self, "Refinement Failed",
                        f"Failed to refine mask:\n{message}"
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
            # Return focus to canvas so spacebar works for hiding masks
            self.canvas.setFocus()
    
    def _refine_mask_by_index(self, mask_idx):
        """
        Refine a mask by ID using YOLO.
        
        Args:
            mask_idx: Instance ID (mask_id) of the mask to refine
            
        Returns:
            Tuple of (success: bool, message: str, stats: dict)
        """
        import cv2
        
        try:
            # Validate mask exists
            if self.canvas.combined_mask is None or mask_idx <= 0:
                return False, "Invalid mask ID", {}
            
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
            
            # Get padding
            padding = self.yolo_refine_toolbar.get_padding()
            
            # Add padding to bounding box
            h, w = self.canvas.current_image.shape[:2]
            crop_y1 = max(0, y_min - padding)
            crop_y2 = min(h, y_max + padding + 1)
            crop_x1 = max(0, x_min - padding)
            crop_x2 = min(w, x_max + padding + 1)
            
            # Crop the image
            crop = self.canvas.current_image[crop_y1:crop_y2, crop_x1:crop_x2]
            
            if crop.size == 0:
                return False, "Invalid crop", {}
            
            # Debug: Print crop info
            print(f"\n=== YOLO Refinement Debug ===")
            print(f"Original mask bbox: ({x_min}, {y_min}) to ({x_max}, {y_max})")
            print(f"Crop region: ({crop_x1}, {crop_y1}) to ({crop_x2}, {crop_y2})")
            print(f"Crop size: {crop.shape[1]}x{crop.shape[0]} (WxH)")
            print(f"Padding used: {padding}px")
            
            # Run YOLO on the crop
            model = self.yolo_refine_toolbar.get_model()
            
            # Convert crop for YOLO (it expects BGR or 3-channel)
            if len(crop.shape) == 2:
                # Grayscale: convert to BGR
                crop_bgr = cv2.cvtColor(crop, cv2.COLOR_GRAY2BGR)
            else:
                # RGB: convert to BGR
                crop_bgr = cv2.cvtColor(crop, cv2.COLOR_RGB2BGR)
            
            results = model.predict(
                source=crop_bgr,
                conf=0.5,  # Lower confidence threshold for refinement
                iou=0.45, 
                augment=False,  # Disable test-time augmentation for consistent results
                retina_masks=True,  # Return masks at original image size (no resize needed)
                verbose=False, 
            )
            
            if not results or len(results) == 0 or results[0].masks is None or len(results[0].masks) == 0:
                print(f"No YOLO detections in crop")
                return False, "No detections", {}
            
            result = results[0]
            print(f"YOLO found {len(result.masks)} predictions in crop")
            
            # Crop the original mask to the same region
            original_mask_crop = original_mask[crop_y1:crop_y2, crop_x1:crop_x2]
            original_area = np.sum(original_mask_crop > 0)
            
            # Calculate original mask's center in crop coordinates
            if original_area > 0:
                y_coords, x_coords = np.where(original_mask_crop > 0)
                orig_center_x = np.mean(x_coords)
                orig_center_y = np.mean(y_coords)
            else:
                orig_center_x = crop.shape[1] / 2
                orig_center_y = crop.shape[0] / 2
            
            print(f"Original mask in crop: area={original_area}px, center=({orig_center_x:.1f}, {orig_center_y:.1f})")
            
            # Calculate expected crop dimensions
            expected_height = crop_y2 - crop_y1
            expected_width = crop_x2 - crop_x1
            
            # Find the prediction with highest IoU with the original mask
            # Also ensure it's not too much larger (reject if >3x original size)
            best_iou = 0
            best_pred_idx = -1
            best_score = 0
            
            for i in range(len(result.masks)):
                # Get predicted mask (already at crop size with retina_masks=True)
                pred_mask = result.masks[i].data[0].cpu().numpy()
                
                print(f"  Prediction {i}: mask shape {pred_mask.shape}, expected ({expected_height}, {expected_width})")
                
                # Validate shape - with retina_masks=True, should already match crop size
                if pred_mask.shape != (expected_height, expected_width):
                    print(f"    WARNING: Shape mismatch! Resizing from {pred_mask.shape} to ({expected_height}, {expected_width})")
                    pred_mask = cv2.resize(
                        pred_mask,
                        (expected_width, expected_height),  # cv2.resize expects (width, height)
                        interpolation=cv2.INTER_LINEAR
                    )
                
                # Binarize
                pred_mask_binary = (pred_mask > 0.5).astype('uint8')
                original_mask_binary = (original_mask_crop > 0).astype('uint8')
                
                # Calculate IoU
                intersection = np.sum(np.logical_and(pred_mask_binary, original_mask_binary))
                union = np.sum(np.logical_or(pred_mask_binary, original_mask_binary))
                pred_area = np.sum(pred_mask_binary)
                
                if union > 0:
                    iou = intersection / union
                    
                    # Calculate predicted mask's center
                    if pred_area > 0:
                        y_coords, x_coords = np.where(pred_mask_binary > 0)
                        pred_center_x = np.mean(x_coords)
                        pred_center_y = np.mean(y_coords)
                        
                        # Distance between centers
                        center_dist = np.sqrt((pred_center_x - orig_center_x)**2 + (pred_center_y - orig_center_y)**2)
                    else:
                        center_dist = float('inf')
                    
                    # Area ratio (predicted / original)
                    area_ratio = pred_area / original_area if original_area > 0 else 0
                    
                    print(f"    IoU: {iou:.3f}, area: {pred_area}px (ratio: {area_ratio:.2f}x), center_dist: {center_dist:.1f}px")
                    
                    # Reject predictions that are way too large or have centers too far apart
                    # This prevents selecting predictions that merge multiple bees
                    if area_ratio > 3.0:
                        print(f"    -> Rejected: too large (>{area_ratio:.1f}x original)")
                        continue
                    
                    if center_dist > max(50, np.sqrt(original_area) * 0.5):
                        print(f"    -> Rejected: center too far ({center_dist:.1f}px)")
                        continue
                    
                    # Scoring: prioritize IoU but penalize large area differences
                    # Score = IoU - penalty for size difference
                    size_penalty = abs(area_ratio - 1.0) * 0.1  # Penalize deviations from 1.0x
                    score = iou - size_penalty
                    
                    if score > best_score:
                        best_iou = iou
                        best_pred_idx = i
                        best_score = score
                        print(f"    -> New best (score: {score:.3f})")
            
            if best_pred_idx < 0 or best_iou < 0.1:
                print(f"No good match found (best IoU: {best_iou:.3f}, best score: {best_score:.3f})")
                return False, f"No match (best IoU: {best_iou:.3f})", {"iou": best_iou}
            
            print(f"Selected prediction {best_pred_idx} with IoU {best_iou:.3f}, score {best_score:.3f}")
            
            # Get the best prediction mask (already at crop size with retina_masks=True)
            best_pred_mask = result.masks[best_pred_idx].data[0].cpu().numpy()

            # Binarize
            best_pred_mask = (best_pred_mask > 0.5).astype('uint8') * 255

            # Validate shape before placement
            if best_pred_mask.shape != (expected_height, expected_width):
                return False, f"Shape mismatch: mask is {best_pred_mask.shape}, expected ({expected_height}, {expected_width})", {}
            
            # Extract only the largest contour to remove any neighboring bee artifacts
            contours, _ = cv2.findContours(best_pred_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if len(contours) > 0:
                # Find the largest contour
                largest_contour = max(contours, key=cv2.contourArea)
                largest_area = cv2.contourArea(largest_contour)
                
                print(f"Found {len(contours)} contour(s), keeping largest (area: {largest_area:.0f}px)")
                
                # Create clean mask with only the largest contour
                best_pred_mask = np.zeros_like(best_pred_mask)
                cv2.drawContours(best_pred_mask, [largest_contour], 0, 255, -1)  # -1 fills the contour
            else:
                print("Warning: No contours found in predicted mask")
            
            # Create full-size mask
            refined_mask = np.zeros_like(original_mask)
            refined_mask[crop_y1:crop_y2, crop_x1:crop_x2] = best_pred_mask

            # Update the editing mask directly (don't use add_mask which writes to combined_mask)
            self.canvas.editing_mask = refined_mask
            self.canvas._update_editing_visualization()
            
            # Calculate improvement
            original_area = np.sum(original_mask > 0)
            refined_area = np.sum(refined_mask > 0)
            area_diff = refined_area - original_area
            
            stats = {
                "iou": best_iou,
                "original_area": original_area,
                "refined_area": refined_area,
                "area_diff": area_diff
            }
            
            return True, "Success", stats
            
        except Exception as e:
            import traceback
            error_msg = traceback.format_exc()
            return False, f"Error: {str(e)}", {}
    
    def refine_all_masks(self):
        """Refine all instance masks in the current frame using YOLO"""
        if not self.yolo_refine_toolbar.is_model_loaded():
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
        
        if self.canvas.combined_mask is None or not np.any(self.canvas.combined_mask > 0):
            QMessageBox.warning(
                self, "No Instances",
                "There are no instances in the current frame to refine."
            )
            return
        
        try:
            # Get unique instance IDs
            instance_ids = np.unique(self.canvas.combined_mask)
            instance_ids = instance_ids[instance_ids > 0]
            num_masks = len(instance_ids)
            
            # Ask for confirmation
            reply = QMessageBox.question(
                self, "Refine All Masks",
                f"This will refine all {num_masks} instance(s) in the current frame.\n\n"
                f"Continue?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel
            )
            
            if reply != QMessageBox.StandardButton.Yes:
                return
            
            # Track results
            refined_count = 0
            failed_count = 0
            total_iou = 0
            
            # Process each instance
            for i, instance_id in enumerate(instance_ids):
                self.status_label.setText(f"Refining instance {i + 1}/{num_masks} (ID: {instance_id})...")
                QApplication.processEvents()
                
                success, message, stats = self._refine_mask_by_index(instance_id)
                
                if success:
                    refined_count += 1
                    total_iou += stats.get("iou", 0)
                else:
                    failed_count += 1
            
            # Commit the last instance that was edited
            if self.canvas.editing_instance_id > 0:
                self.canvas.commit_editing()
            
            # Show summary
            avg_iou = total_iou / refined_count if refined_count > 0 else 0
            summary = (
                f"Refinement complete!\n\n"
                f"Refined: {refined_count}/{num_masks}\n"
                f"Failed: {failed_count}/{num_masks}\n"
                f"Average IoU: {avg_iou:.3f}"
            )
            
            self.status_label.setText(
                f"✓ Refined {refined_count}/{num_masks} masks (avg IoU: {avg_iou:.3f})"
            )
            
            QMessageBox.information(
                self, "Refinement Complete",
                summary
            )
            
        except Exception as e:
            import traceback
            error_msg = traceback.format_exc()
            QMessageBox.critical(
                self, "Refinement Error",
                f"Error refining masks:\n{str(e)}\n\n{error_msg}"
            )
            self.status_label.setText("Refinement failed")
        finally:
            # Return focus to canvas so spacebar works for hiding masks
            self.canvas.setFocus()
            
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
            # Return focus to canvas so spacebar works for hiding masks
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
        """Refine all instances in current frame using instance-focused YOLO with overlap resolution"""
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
        mask_ids = sorted(list(instance_ids))
        
        if len(mask_ids) == 0:
            QMessageBox.information(
                self, "No Instances",
                "No instances found in current frame.\n\n"
                "You need either segmentation masks or bounding box annotations."
            )
            return
        
        # Ask for confirmation
        reply = QMessageBox.question(
            self,
            "Refine All Instances",
            f"Refine all {len(mask_ids)} instances in this frame?\n\n"
            f"This will predict masks for all instances, then resolve overlaps\n"
            f"by choosing the prediction with highest confidence per pixel.\n\n"
            f"Works with both segmentation masks and bbox-only annotations.",
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
            print(f"Refining {len(mask_ids)} instances with overlap resolution")
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
            
            # OPTIMIZED: Direct batch update of combined_mask instead of per-instance editing
            # This is much faster than calling start_editing_instance/commit_editing for each instance
            
            # Initialize combined_mask if needed
            if self.canvas.combined_mask is None:
                self.canvas.combined_mask = np.zeros((h, w), dtype=np.int32)
            
            # Clear existing masks for instances that were refined
            for mask_id in predictions.keys():
                self.canvas.combined_mask[self.canvas.combined_mask == mask_id] = 0
            
            # Apply all refined masks at once from instance_map
            self.canvas.combined_mask = instance_map.astype(np.int32)
            
            # Update metadata and calculate statistics
            total_area_diff = 0
            for mask_id, pred_data in predictions.items():
                # If this was a bbox-only annotation, mark it as no longer bbox-only
                if pred_data['from_bbox'] and mask_id in self.canvas.annotation_metadata:
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
            # Return focus to canvas so spacebar works for hiding masks
            self.canvas.setFocus()
            
    def closeEvent(self, event):
        """Handle window close event"""
        # Ask for confirmation before closing
        reply = QMessageBox.question(
            self, "Close Application",
            "Are you sure you want to close the application?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        
        if reply == QMessageBox.StandardButton.No:
            event.ignore()
            return
        
        # Automatically save current frame before closing
        if hasattr(self, 'project_path') and self.project_path and hasattr(self, 'current_video_id') and self.current_video_id:
            try:
                # Commit any pending edits
                if hasattr(self, 'canvas') and self.canvas.editing_instance_id > 0:
                    self.canvas.commit_editing()
                
                # Save current frame annotations
                annotations = self.canvas.get_annotations()
                if annotations:
                    frame_idx_in_video = self._get_frame_idx_in_video(self.current_frame_idx)
                    self.annotation_manager.save_frame_annotations(
                        self.project_path, self.current_video_id,
                        frame_idx_in_video, annotations
                    )
                    print(f"Saved frame {frame_idx_in_video} before closing")
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
