"""
Main application window
"""

import numpy as np
import torch
from PyQt6.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QSplitter, QFileDialog, QMessageBox, QStatusBar,
                             QDockWidget, QListWidget, QToolBar, QLabel, 
                             QApplication, QDialog, QPushButton, QScrollArea)
from PyQt6.QtCore import Qt, pyqtSignal, QSettings, QThread
from PyQt6.QtGui import QAction, QKeySequence
from pathlib import Path
import queue
import time

from .canvas import ImageCanvas
from .toolbar import AnnotationToolbar
from .yolo_toolbar import YOLOToolbar
from .yolo_refine_toolbar import YOLORefinementToolbar
from .yolo_sahi_toolbar import YOLOSAHIToolbar
from .sam2_toolbar import SAM2Toolbar
from .dialogs import VideoImportDialog, ProjectDialog
from .training_dialog import TrainingConfigDialog, TrainingProgressDialog
from core.video_processor import VideoProcessor
from core.annotation import AnnotationManager
from core.project_manager import ProjectManager
from core.instance_tracker import InstanceTracker, Detection, Track
from core.frame_cache import FrameCache, PreloadWorker
from training.coco_video_export import export_coco_per_video
from training.yolo_trainer import YOLOTrainingWorker
from training.yolo_trainer_stage2 import YOLOTrainingWorkerStage2
from training.yolo_trainer_sahi import YOLOTrainingWorkerSAHI


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
    
    def __init__(self, sam2_checkpoint=None, coarse_yolo_checkpoint=None, fine_yolo_checkpoint=None):
        super().__init__()
        
        self.video_processor = VideoProcessor()
        self.annotation_manager = AnnotationManager(max_cache_size=2)  # Very small cache - annotations with masks are huge (~100MB/frame)
        self.project_manager = ProjectManager()
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
        self.preload_worker = PreloadWorker(self.frame_cache)
        self.preload_worker.start()
        
        # SAM2 will be initialized when user loads checkpoint via SAM2Toolbar
        self.sam2 = None
        
        # Store checkpoint paths for use in init_ui
        self.sam2_checkpoint = sam2_checkpoint
        self.coarse_yolo_checkpoint = coarse_yolo_checkpoint
        self.fine_yolo_checkpoint = fine_yolo_checkpoint
        
        self.current_frame_idx = 0
        self.frames = []
        self.frame_video_ids = []  # Track which video each frame belongs to
        self.frame_splits = []  # Track split (train/val) for each frame
        self.frame_selected = []  # Track if frame is selected for train/val
        self.frame_list_to_frames_map = []  # Map list row to actual frame index
        self.project_path = None
        self.split_filter = 'all'  # 'all', 'train', or 'val'
        self.video_next_mask_id = {}  # Track next_mask_id per video for unique IDs
        self.video_mask_colors = {}  # Track mask colors per video: {video_id: {mask_id: (r,g,b)}}
        self.box_inference_mode = False  # Track if box inference mode is active
        
        # Track unsaved changes to avoid unnecessary saves
        self.current_frame_modified = False
        
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
    
    def closeEvent(self, event):
        """Handle window close event"""
        # Save current frame if it has unsaved changes
        if hasattr(self, 'current_frame_modified') and self.current_frame_modified:
            if hasattr(self, 'project_path') and self.project_path and hasattr(self, 'current_video_id') and self.current_video_id:
                try:
                    # Commit any pending edits
                    if hasattr(self, 'canvas') and self.canvas.editing_instance_id > 0:
                        self.canvas.commit_editing()
                    
                    annotations = self.canvas.get_annotations()
                    if annotations:
                        frame_idx_in_video = self._get_frame_idx_in_video(self.current_frame_idx)
                        # Do a blocking save for final frame
                        self.annotation_manager.save_frame_annotations(
                            self.project_path, self.current_video_id,
                            frame_idx_in_video, annotations
                        )
                        print(f"Saved unsaved changes for frame {frame_idx_in_video} before closing")
                except Exception as e:
                    print(f"Warning: Failed to save on close: {e}")
        
        # Stop and wait for save worker to finish
        if hasattr(self, 'save_worker'):
            self.save_worker.stop()
            self.save_worker.wait(2000)  # Wait up to 2 seconds
        
        # Stop preload worker
        if hasattr(self, 'preload_worker'):
            self.preload_worker.stop()
            self.preload_worker.join(timeout=1.0)  # Wait up to 1 second
        
        event.accept()
    
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
        self.canvas.annotation_changed.connect(self.update_instance_list_from_canvas)
        self.canvas.setToolTip("Hold Spacebar to temporarily hide masks | Press F to fit image to window")
        
        # Create annotation toolbar
        self.toolbar = AnnotationToolbar(self)
        self.toolbar.tool_changed.connect(self.on_tool_changed)
        self.toolbar.brush_size_changed.connect(self.canvas.set_brush_size)
        self.toolbar.mask_opacity_changed.connect(self.canvas.set_mask_opacity)
        self.toolbar.clear_instance_requested.connect(self.clear_selected_instance)
        self.toolbar.new_instance_requested.connect(self.new_instance)
        self.toolbar.resolve_overlaps_requested.connect(self.resolve_overlaps)
        
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
        
        # Create YOLO toolbar
        if self.coarse_yolo_checkpoint:
            print(f"Loading coarse YOLO checkpoint from command line: {self.coarse_yolo_checkpoint}")
        self.yolo_toolbar = YOLOToolbar(self, checkpoint_path=self.coarse_yolo_checkpoint)
        self.yolo_toolbar.inference_requested.connect(self.run_yolo_inference)
        self.yolo_toolbar.track_from_last_requested.connect(self.track_from_last_frame)
        
        # Create YOLO refinement toolbar
        if self.fine_yolo_checkpoint:
            print(f"Loading fine YOLO checkpoint from command line: {self.fine_yolo_checkpoint}")
        self.yolo_refine_toolbar = YOLORefinementToolbar(self, checkpoint_path=self.fine_yolo_checkpoint)
        self.yolo_refine_toolbar.refine_requested.connect(self.refine_selected_mask)
        self.yolo_refine_toolbar.refine_all_requested.connect(self.refine_all_masks)
        
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
        
        # Create layout
        central_widget = QWidget()
        layout = QVBoxLayout(central_widget)
        layout.addWidget(self.toolbar)
        layout.addWidget(self.sam2_toolbar)
        layout.addWidget(self.yolo_toolbar)
        layout.addWidget(self.yolo_refine_toolbar)
        layout.addWidget(self.yolo_sahi_toolbar)
        layout.addWidget(self.canvas)
        self.setCentralWidget(central_widget)
        
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
    
    def _clean_duplicate_contours(self, annotations, overlap_threshold=0.5):
        """Remove duplicate contours from instances where a contour overlaps heavily with another instance
        
        This handles cases where an instance has picked up contours from other bees.
        For each contour in each instance, if it overlaps significantly with another instance,
        remove it from the original instance but keep the rest of that instance intact.
        
        Args:
            annotations: List of annotation dicts with 'mask' key
            overlap_threshold: Threshold for overlap (intersection / smaller_area). 
                             If a contour from instance A has overlap > threshold with instance B,
                             remove it from A.
            
        Returns:
            List of cleaned annotations
        """
        import cv2
        
        if len(annotations) <= 1:
            return annotations
        
        print(f"Cleaning duplicate contours from {len(annotations)} instances...")
        
        # Extract contours for each instance
        instance_contours = []
        for ann in annotations:
            mask = ann['mask']
            if not isinstance(mask, np.ndarray):
                instance_contours.append([])
                continue
            
            # Find all contours for this instance
            binary_mask = (mask > 0).astype(np.uint8)
            contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Store contours with their areas
            contours_with_areas = []
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > 10:  # Filter out very small noise contours
                    contours_with_areas.append((contour, area))
            
            instance_contours.append(contours_with_areas)
        
        # Track which contours to remove from which instances
        # Format: {instance_idx: [contour_indices_to_remove]}
        contours_to_remove = {i: [] for i in range(len(annotations))}
        
        total_removed = 0
        
        # Compare contours across different instances
        for i in range(len(annotations)):
            for j in range(len(annotations)):
                if i == j:
                    continue
                
                # Compare each contour from instance i with full mask of instance j
                mask_j = annotations[j]['mask']
                binary_mask_j = (mask_j > 0).astype(np.uint8)
                
                for contour_idx, (contour_i, area_i) in enumerate(instance_contours[i]):
                    if contour_idx in contours_to_remove[i]:
                        continue  # Already marked for removal
                    
                    # Create a temporary mask for this single contour
                    h, w = annotations[i]['mask'].shape
                    contour_mask = np.zeros((h, w), dtype=np.uint8)
                    cv2.drawContours(contour_mask, [contour_i], 0, 255, -1)
                    
                    # Calculate overlap with instance j
                    intersection = np.logical_and(contour_mask > 0, binary_mask_j > 0).sum()
                    
                    if intersection > 0:
                        # Calculate overlap as intersection / area of this contour
                        overlap_ratio = intersection / area_i
                        
                        if overlap_ratio > overlap_threshold:
                            # This contour overlaps heavily with instance j
                            # Mark it for removal from instance i
                            contours_to_remove[i].append(contour_idx)
                            total_removed += 1
                            print(f"  Removing contour {contour_idx} from instance {i} (area={area_i:.0f}px, overlap={overlap_ratio:.1%} with instance {j})")
                            break  # Move to next contour
        
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
                for contour_idx, (contour, area) in enumerate(instance_contours[i]):
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
        
        delete_action = menu.addAction("Delete Video...")
        
        action = menu.exec(self.video_list.mapToGlobal(position))
        
        if action == delete_action:
            self.delete_video(video_id)
    
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
            
            # Delete annotations directory
            annotations_dir = self.project_path / 'annotations' / 'pkl' / video_id
            if annotations_dir.exists():
                shutil.rmtree(annotations_dir)
                print(f"Deleted annotations: {annotations_dir}")
            
            # Delete video file from input_data (check both train and val)
            deleted_video = False
            
            # Try to find the video in both splits if split detection fails
            for split in ['train', 'val']:
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
        
        # Get all videos from both train and val splits
        train_videos = self.project_manager.get_videos_by_split('train')
        val_videos = self.project_manager.get_videos_by_split('val')
        
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
        self.split_filter_combo.addItems(['All Frames', 'Training Only', 'Validation Only'])
        self.split_filter_combo.currentTextChanged.connect(self.on_split_filter_changed)
        filter_layout.addWidget(self.split_filter_combo)
        layout.addLayout(filter_layout)
        
        # Frame list
        self.frame_list = QListWidget()
        self.frame_list.currentRowChanged.connect(self.on_frame_changed)
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
        
        # Delete instance
        delete_instance = QShortcut(QKeySequence(Qt.Key.Key_Delete), self)
        delete_instance.activated.connect(self.delete_selected_instance)
        
        backspace_instance = QShortcut(QKeySequence(Qt.Key.Key_Backspace), self)
        backspace_instance.activated.connect(self.delete_selected_instance)
        
        # Clear instance mask and points
        clear_instance = QShortcut(QKeySequence(Qt.Key.Key_C), self)
        clear_instance.activated.connect(self.clear_selected_instance)
    
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
        split_combo.addItems(['train', 'val'])
        layout.addWidget(split_combo)
        
        # Frames for training/validation selector
        layout.addWidget(QLabel("Number of frames for training/validation:"))
        frames_spin = QSpinBox()
        frames_spin.setMinimum(1)
        frames_spin.setMaximum(1000)
        frames_spin.setValue(15)
        layout.addWidget(frames_spin)
        
        # Info text
        info_label = QLabel("All frames will be extracted, but only selected frames\nwill be marked for training/validation.")
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
            split: 'train' or 'val'
            n_selected: Number of frames to mark for training/validation
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
                error_msg = '\n'.join([f"{f['video_id']}: {f['error']}" 
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
                with open(metadata_file) as f:
                    metadata = json.load(f)
                    selected_indices = set(metadata.get('selected_frames', []))
            
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
            
            # Create label with split indicator (only for selected frames)
            if is_selected and split != 'unknown':
                split_tag = f"[{split.upper()}] "
            else:
                split_tag = ""
            label = f"{split_tag}Frame {i:04d}"
            self.frame_list.addItem(label)
            
            # Store mapping from list row to actual frame index
            self.frame_list_to_frames_map.append(i)
            
    def load_frame(self, idx):
        """Load a specific frame"""
        if 0 <= idx < len(self.frames):
            # Auto-save current frame annotations before loading new frame (only if modified)
            if self.current_frame_idx != idx and self.project_path and self.current_frame_modified:
                try:
                    # Commit any pending edits before saving current frame
                    if self.canvas.editing_instance_id > 0:
                        self.canvas.commit_editing()
                    
                    annotations = self.canvas.get_annotations()
                    if annotations and self.current_video_id:  # Only save if there are annotations and we know the video
                        # Update video next_mask_id tracking
                        if self.current_video_id in self.video_next_mask_id:
                            self.video_next_mask_id[self.current_video_id] = self.canvas.next_mask_id
                        
                        # Update in-memory cache
                        self.annotation_manager.set_frame_annotations(
                            self.current_frame_idx, annotations
                        )
                        # Get actual frame index within video (not the list index)
                        frame_idx_in_video = self._get_frame_idx_in_video(self.current_frame_idx)
                        # Queue background save - non-blocking!
                        self.save_worker.add_save_task(
                            self.project_path, self.current_video_id, 
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
                selected_indices=selected_indices
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
                        # Find highest mask_id in existing annotations for this video
                        max_id = 0
                        if self.project_path:
                            annotations_dir = self.project_path / 'annotations' / 'pkl' / self.current_video_id
                            if annotations_dir.exists():
                                for ann_file in annotations_dir.glob('frame_*.pkl'):
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
                        self.video_next_mask_id[self.current_video_id] = max_id + 1
                    
                    # Set canvas next_mask_id from video tracking
                    self.canvas.next_mask_id = self.video_next_mask_id[self.current_video_id]
                
                # Load annotations for this frame
                # First check cache, then load from disk if not present
                annotations = self.annotation_manager.get_frame_annotations(idx)
                annotation_source = "cached" if annotations else "disk"
                if not annotations and self.project_path and self.current_video_id:
                    # Not in cache - load from disk
                    frame_idx_in_video = self._get_frame_idx_in_video(idx)
                    annotations = self.annotation_manager.load_frame_annotations(
                        self.project_path, self.current_video_id, frame_idx_in_video
                    )
                    # Update cache
                    if annotations:
                        self.annotation_manager.set_frame_annotations(idx, annotations)
                
                # Debug: print annotation info
                if annotations:
                    num_instances = len(annotations)
                    total_mask_mb = sum(ann['mask'].nbytes / (1024*1024) for ann in annotations if 'mask' in ann)
                    print(f"  Annotations ({annotation_source}): {num_instances} instances, {total_mask_mb:.1f}MB total")
                
                # Clean up duplicate contours between instances
                if annotations and len(annotations) > 1:
                    annotations = self._clean_duplicate_contours(annotations, overlap_threshold=0.5)
                
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
            
    def on_tool_changed(self, tool_name):
        """Handle tool change"""
        self.canvas.set_tool(tool_name)
        
        # When switching away from SAM2 tools, uncheck SAM2 toolbar buttons
        if tool_name not in ['sam2_prompt', 'sam2_box']:
            self.sam2_toolbar.uncheck_tools()
        
        # When switching to SAM2 tools, uncheck annotation toolbar buttons
        if tool_name in ['sam2_prompt', 'sam2_box']:
            self.toolbar.uncheck_all_tools()
    
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
            # Box is drawn and stored in canvas - just notify toolbar
            self.yolo_sahi_toolbar.set_box_drawn(True)
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
                self.canvas.add_mask(mask)
                
                # Register color for this mask
                self._register_canvas_colors()
                
                self.update_instance_list_from_canvas()
                self.status_label.setText("SAM2 prediction added")
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
        if visible:
            self.status_label.setText("Masks visible")
        else:
            self.status_label.setText("⚠ Masks hidden (release Spacebar to show)")
    
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
        annotations = self.annotation_manager.get_frame_annotations(self.current_frame_idx)
        for i, ann in enumerate(annotations):
            label = ann.get('label', 'Unknown')
            self.instance_list.addItem(f"{label} #{i+1}")
            
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
        
        # Get unique instance IDs from combined mask
        instance_ids = np.unique(self.canvas.combined_mask)
        instance_ids = instance_ids[instance_ids > 0].tolist()  # Exclude background (0)
        
        # Add editing instance if currently in editing mode
        if self.canvas.editing_instance_id > 0 and self.canvas.editing_mask is not None:
            if self.canvas.editing_instance_id not in instance_ids:
                instance_ids.append(self.canvas.editing_instance_id)
        
        # Sort for consistent ordering
        instance_ids = sorted(instance_ids)
        
        for instance_id in instance_ids:
            # Calculate area from appropriate source
            if instance_id == self.canvas.editing_instance_id and self.canvas.editing_mask is not None:
                area = np.sum(self.canvas.editing_mask > 0)
            else:
                area = np.sum(self.canvas.combined_mask == instance_id)
            self.instance_list.addItem(f"Instance ID: {instance_id} (area: {area})")
        
        # Update instance labels on canvas if they are currently visible
        if self.canvas.labels_visible:
            self.canvas.update_instance_labels()

        if active_instance_id in instance_ids:
            active_index = instance_ids.index(active_instance_id)
            self.instance_list.blockSignals(True)
            self.instance_list.setCurrentRow(active_index)
            self.instance_list.blockSignals(False)
    
    def on_instance_changed(self, idx):
        """Handle instance selection change
        
        Args:
            idx: Row index in the instance list widget
        """
        if idx >= 0 and self.canvas.combined_mask is not None:
            # Get unique instance IDs (same logic as update_instance_list_from_canvas)
            instance_ids = np.unique(self.canvas.combined_mask)
            instance_ids = instance_ids[instance_ids > 0].tolist()  # Exclude background
            
            # Add editing instance if present
            if self.canvas.editing_instance_id > 0 and self.canvas.editing_mask is not None:
                if self.canvas.editing_instance_id not in instance_ids:
                    instance_ids.append(self.canvas.editing_instance_id)
            
            # Sort to match list order
            instance_ids = sorted(instance_ids)
            
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
        
        # Get unique instance IDs
        instance_ids = np.unique(self.canvas.combined_mask)
        instance_ids = instance_ids[instance_ids > 0]  # Exclude background
        
        if idx >= 0 and idx < len(instance_ids):
            current_id = int(instance_ids[idx])
            
            # Show input dialog
            from PyQt6.QtWidgets import QInputDialog
            new_id, ok = QInputDialog.getInt(
                self,
                "Edit Instance ID",
                f"Enter new ID for Instance ID {current_id}:",
                value=current_id,
                min=1,
                max=9999
            )
            
            if ok and new_id != current_id:
                # Check if ID already exists
                if new_id in instance_ids:
                    QMessageBox.warning(
                        self,
                        "Duplicate ID",
                        f"Bee ID {new_id} already exists. Please choose a different ID."
                    )
                    return
                
                # Update ID in combined mask (replace all pixels with current_id to new_id)
                self.canvas.combined_mask[self.canvas.combined_mask == current_id] = new_id
                
                # Update color mapping
                if current_id in self.canvas.mask_colors:
                    self.canvas.mask_colors[new_id] = self.canvas.mask_colors[current_id]
                    del self.canvas.mask_colors[current_id]
                
                # Update next_mask_id if necessary
                if new_id >= self.canvas.next_mask_id:
                    self.canvas.next_mask_id = new_id + 1
                
                # Mark frame as modified so it will be saved
                self.current_frame_modified = True
                
                # Rebuild visualization
                self.canvas.rebuild_visualizations()
                
                # Refresh the list
                self.update_instance_list_from_canvas()
                self.instance_list.setCurrentRow(idx)
                self.status_label.setText(f"Updated bee ID to {new_id}")
    
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
                self.current_frame_idx, current_annotations
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
                    target_idx, updated_annotations
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
                self.current_frame_idx, current_annotations
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
            
    def new_instance(self):
        """Start annotating a new instance"""
        # Start new SAM2 instance (clears prompts, resets active mask)
        self.canvas.start_new_sam2_instance()
        # Clear instance list selection
        self.instance_list.clearSelection()
        self.status_label.setText("Ready to annotate new bee - Use Point/Box tool or Brush")
    
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
        """Propagate current annotations to next frame"""
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
            self.current_frame_idx, current_annotations
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
                    next_frame_image
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
                        self.current_frame_idx, annotations
                    )
                    
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
    
    def export_coco_format(self):
        """Export annotations in COCO format (train and val splits)"""
        if not self.project_path:
            QMessageBox.warning(self, "No Project", "Please create or open a project first")
            return
        
        try:
            # Save current frame first
            if self.current_frame_idx is not None:
                annotations = self.canvas.get_annotations()
                self.annotation_manager.set_frame_annotations(
                    self.current_frame_idx, annotations
                )
            
            # Get train and validation videos
            train_videos = self.project_manager.get_videos_by_split('train')
            val_videos = self.project_manager.get_videos_by_split('val')
            
            if not train_videos and not val_videos:
                QMessageBox.warning(
                    self,
                    "No Videos",
                    "No videos found in train or validation splits.\n\n"
                    "Please import videos and assign them to splits first."
                )
                return
            
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
                
                # Clean up duplicate contours between instances
                current_annotations = self.canvas.get_annotations()
                if len(current_annotations) > 1:
                    cleaned_annotations = self._clean_duplicate_contours(current_annotations, overlap_threshold=0.5)
                    # Reapply cleaned annotations
                    mask_colors_dict = {ann['mask_id']: self.video_mask_colors[self.current_video_id].get(ann['mask_id']) 
                                       for ann in cleaned_annotations if ann.get('mask_id')}
                    self.canvas.set_annotations(cleaned_annotations, mask_colors_dict)
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
    
    def on_box_inference_mode_requested(self, checked):
        """Handle box inference mode toggle from SAHI toolbar"""
        self.box_inference_mode = checked
        if checked:
            # Switch to inference box tool when mode is enabled
            self.canvas.set_tool('inference_box')
            self.status_label.setText("Draw a box on the image (you can adjust it afterwards)")
        else:
            # Return to default tool when disabled
            self.canvas.set_tool('polygon')
            self.canvas.clear_inference_box()
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
        
        # Clear the box and reset toolbar
        self.canvas.clear_inference_box()
        self.yolo_sahi_toolbar.set_box_drawn(False)
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
            
            # Apply additional custom duplicate merging if needed
            # (SAHI's postprocessing should handle most cases, but this provides extra safety by merging any remaining overlaps)
            initial_count = len(detections)
            detections = self._remove_duplicate_detections(detections, iou_threshold=0.5)
            if len(detections) < initial_count:
                print(f"Custom merging combined {initial_count - len(detections)} overlapping detections into merged instances")
            
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
                
                # Clean up duplicate contours between instances
                current_annotations = self.canvas.get_annotations()
                if len(current_annotations) > 1:
                    cleaned_annotations = self._clean_duplicate_contours(current_annotations, overlap_threshold=0.5)
                    # Reapply cleaned annotations
                    mask_colors_dict = {ann['mask_id']: self.video_mask_colors[self.current_video_id].get(ann['mask_id']) 
                                       for ann in cleaned_annotations if ann.get('mask_id')}
                    self.canvas.set_annotations(cleaned_annotations, mask_colors_dict)
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
                
                # Clean up duplicate contours between instances
                current_annotations = self.canvas.get_annotations()
                if len(current_annotations) > 1:
                    cleaned_annotations = self._clean_duplicate_contours(current_annotations, overlap_threshold=0.5)
                    # Reapply cleaned annotations
                    mask_colors_dict = {ann['mask_id']: self.canvas.mask_colors.get(ann['mask_id']) 
                                       for ann in cleaned_annotations if ann.get('mask_id')}
                    self.canvas.set_annotations(cleaned_annotations, mask_colors_dict)
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
Padding: 100px around image (BORDER_REFLECT_101)
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
            
            # Add padding using reflection to avoid border artifacts
            padded_img = cv2.copyMakeBorder(
                original_img,
                pad_size, pad_size, pad_size, pad_size,
                cv2.BORDER_REFLECT_101
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
            
            self.status_label.setText("Running SOHO inference (merging duplicates)...")
            QApplication.processEvents()
            
            # Merge duplicate detections
            merged_detections = self._remove_duplicate_detections(all_detections, iou_threshold=0.5)
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
                
                # Clean duplicate contours
                current_annotations = self.canvas.get_annotations()
                if len(current_annotations) > 1:
                    cleaned_annotations = self._clean_duplicate_contours(current_annotations, overlap_threshold=0.5)
                    mask_colors_dict = {ann['mask_id']: self.video_mask_colors[self.current_video_id].get(ann['mask_id']) 
                                       for ann in cleaned_annotations if ann.get('mask_id')}
                    self.canvas.set_annotations(cleaned_annotations, mask_colors_dict)
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
                
                # Clean duplicate contours
                current_annotations = self.canvas.get_annotations()
                if len(current_annotations) > 1:
                    cleaned_annotations = self._clean_duplicate_contours(current_annotations, overlap_threshold=0.5)
                    mask_colors_dict = {ann['mask_id']: self.canvas.mask_colors.get(ann['mask_id']) 
                                       for ann in cleaned_annotations if ann.get('mask_id')}
                    self.canvas.set_annotations(cleaned_annotations, mask_colors_dict)
                    self._register_canvas_colors()
                
                self.current_frame_modified = True
                
                self.status_label.setText(
                    f"✓ SOHO inference complete: {len(detections)} instances detected"
                )
            
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
    
    def propagate_soho_to_next_frame(self):
        """Propagate annotations to next frame using SOHO inference and ByteTrack matching"""
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
        next_idx = self._get_next_frame_index()
        if next_idx is None:
            self.status_label.setText("Already at last frame in current view")
            return
        
        try:
            import cv2
            
            # Save current frame annotations
            self.annotation_manager.set_frame_annotations(
                self.current_frame_idx, current_annotations
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
            
            # Add padding
            padded_img = cv2.copyMakeBorder(
                next_frame_img,
                pad_size, pad_size, pad_size, pad_size,
                cv2.BORDER_REFLECT_101
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
            
            # Merge duplicates
            self.status_label.setText("Merging duplicate detections...")
            QApplication.processEvents()
            
            merged_detections = self._remove_duplicate_detections(all_detections, iou_threshold=0.5)
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
            
            # Clean duplicate contours
            current_annotations = self.canvas.get_annotations()
            if len(current_annotations) > 1:
                cleaned_annotations = self._clean_duplicate_contours(current_annotations, overlap_threshold=0.5)
                mask_colors_dict = {ann['mask_id']: self.video_mask_colors[self.current_video_id].get(ann['mask_id']) 
                                   for ann in cleaned_annotations if ann.get('mask_id')}
                self.canvas.set_annotations(cleaned_annotations, mask_colors_dict)
                self._register_canvas_colors()
            
            self.video_next_mask_id[self.current_video_id] = tracker.next_track_id
            self.current_frame_modified = True
            
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
        
        # Find next selected frame
        next_selected_idx = None
        for idx in range(self.current_frame_idx + 1, len(self.frames)):
            if idx < len(self.frame_splits) and self.frame_splits[idx] != 'none':
                next_selected_idx = idx
                break
        
        if next_selected_idx is None:
            QMessageBox.information(
                self, "No Selected Frame",
                "No selected (train/val) frame found after current frame."
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
        
        # Propagate frame by frame
        successful_propagations = 0
        failed_frames = []
        
        try:
            for i in range(num_frames):
                current_frame = self.current_frame_idx
                
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
                    
                except Exception as e:
                    failed_frames.append(current_frame + 1)
                    print(f"Error propagating to frame {current_frame + 1}: {e}")
                    break
            
            # Show summary
            if failed_frames:
                QMessageBox.warning(
                    self, "Propagation Completed with Errors",
                    f"Successfully propagated through {successful_propagations} frame(s).\n\n"
                    f"Failed at frame(s): {', '.join(map(str, failed_frames))}"
                )
            else:
                QMessageBox.information(
                    self, "Propagation Complete",
                    f"Successfully propagated through {successful_propagations} frame(s) using SOHO.\n\n"
                    f"Track IDs have been maintained across frames."
                )
            
            self.status_label.setText(f"✓ SOHO propagated through {successful_propagations} frames")
            
        except Exception as e:
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
            
            # Show summary
            total_time = time.time() - start_time
            
            if cancelled:
                QMessageBox.information(
                    self, "Propagation Cancelled",
                    f"Propagation was cancelled by user.\n\n"
                    f"Successfully propagated through {successful_propagations} frame(s) in {total_time:.1f} seconds before cancelling."
                )
                self.status_label.setText(
                    f"Propagation cancelled after {successful_propagations} frames"
                )
            elif failed_frames:
                QMessageBox.warning(
                    self, "Propagation Completed with Errors",
                    f"Successfully propagated through {successful_propagations} frame(s) in {total_time:.1f} seconds.\n\n"
                    f"Failed at frame(s): {', '.join(map(str, failed_frames))}"
                )
                self.status_label.setText(
                    f"✓ SOHO propagated {successful_propagations} frames (with errors)"
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
                    f"✓ SOHO propagated through entire video: {successful_propagations} frames in {total_time:.1f}s"
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
            
    def closeEvent(self, event):
        """Handle window close event"""
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
        
        # Ask to save if there are unsaved changes
        if self.annotation_manager.has_unsaved_changes():
            reply = QMessageBox.question(
                self, "Unsaved Changes",
                "Do you want to save your changes before closing?",
                QMessageBox.StandardButton.Save | 
                QMessageBox.StandardButton.Discard | 
                QMessageBox.StandardButton.Cancel
            )
            
            if reply == QMessageBox.StandardButton.Save:
                self.save_annotations()
            elif reply == QMessageBox.StandardButton.Cancel:
                event.ignore()
                return
                
        event.accept()
