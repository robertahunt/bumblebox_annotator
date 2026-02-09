"""
Main application window
"""

import numpy as np
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
from .sam2_toolbar import SAM2Toolbar
from .dialogs import VideoImportDialog, ProjectDialog
from core.video_processor import VideoProcessor
from core.annotation import AnnotationManager
from core.project_manager import ProjectManager
from training.coco_video_export import export_coco_with_tracking
# from training.trainer import HumanInTheLoopTrainer  # Detectron2 training removed


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
        self.annotation_manager = AnnotationManager()
        self.project_manager = ProjectManager()
        self.trainer = None  # Initialized when training starts
        self.current_video_id = None  # Track current video being annotated
        
        # Initialize background save worker
        self.save_worker = SaveWorker(self.annotation_manager)
        self.save_worker.error_occurred.connect(self.on_save_error)
        self.save_worker.save_started.connect(self.on_save_started)
        self.save_worker.save_completed.connect(self.on_save_completed)
        self.save_worker.start()
        
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
        
        # Create default projects directory
        self.default_projects_dir = Path(__file__).parent.parent / 'projects'
        self.default_projects_dir.mkdir(exist_ok=True)
        
        self.init_ui()
        self.setup_shortcuts()
        self.load_settings()
    
    def closeEvent(self, event):
        """Handle window close event"""
        # Stop and wait for save worker to finish
        if hasattr(self, 'save_worker'):
            self.save_worker.stop()
            self.save_worker.wait(2000)  # Wait up to 2 seconds
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
        
        # Create YOLO toolbar
        if self.coarse_yolo_checkpoint:
            print(f"Loading coarse YOLO checkpoint from command line: {self.coarse_yolo_checkpoint}")
        self.yolo_toolbar = YOLOToolbar(self, checkpoint_path=self.coarse_yolo_checkpoint)
        self.yolo_toolbar.inference_requested.connect(self.run_yolo_inference)
        
        # Create YOLO refinement toolbar
        if self.fine_yolo_checkpoint:
            print(f"Loading fine YOLO checkpoint from command line: {self.fine_yolo_checkpoint}")
        self.yolo_refine_toolbar = YOLORefinementToolbar(self, checkpoint_path=self.fine_yolo_checkpoint)
        self.yolo_refine_toolbar.refine_requested.connect(self.refine_selected_mask)
        self.yolo_refine_toolbar.refine_all_requested.connect(self.refine_all_masks)
        
        # Create layout
        central_widget = QWidget()
        layout = QVBoxLayout(central_widget)
        layout.addWidget(self.toolbar)
        layout.addWidget(self.sam2_toolbar)
        layout.addWidget(self.yolo_toolbar)
        layout.addWidget(self.yolo_refine_toolbar)
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
        
        # Detectron2 training removed - use YOLO training notebook instead
        # train_action = QAction("Start &Training...", self)
        # train_action.setShortcut("Ctrl+T")
        # train_action.triggered.connect(self.start_training)
        # model_menu.addAction(train_action)
        
        predict_action = QAction("Run &Inference", self)
        predict_action.setShortcut("Ctrl+R")
        predict_action.triggered.connect(self.run_inference)
        model_menu.addAction(predict_action)
        
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
    
    def on_video_selected(self, row):
        """Handle video selection from video list"""
        if row < 0 or row >= self.video_list.count():
            return
        
        item = self.video_list.item(row)
        video_id = item.data(Qt.ItemDataRole.UserRole)  # Store video_id in item data
        
        if video_id:
            # Load this video's frames
            self.load_video_frames(video_id)
    
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
        
        self.instance_list = QListWidget()
        self.instance_list.currentRowChanged.connect(self.on_instance_changed)
        self.instance_list.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.instance_list.customContextMenuRequested.connect(self.show_instance_context_menu)
        dock.setWidget(self.instance_list)
        
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
            
            self.current_video_id = video_id
            
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
            
            # Update UI
            self.update_video_list()  # Refresh video list to update current video highlighting
            self.update_frame_list()
            if self.frames:
                self.load_frame(0)
            
            self.status_label.setText(
                f"Loaded {video_id} ({split}): {len(self.frames)} frames"
            )
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load video frames: {str(e)}")
            import traceback
            traceback.print_exc()
                
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
            # Auto-save current frame annotations before loading new frame
            if self.current_frame_idx != idx and self.project_path:
                try:
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
            
            frame = self.frames[idx]
            
            # Check if frame is a file path or numpy array
            if isinstance(frame, Path) or isinstance(frame, str):
                # It's a file path
                if not Path(frame).exists():
                    QMessageBox.warning(self, "Error", f"Frame file not found: {frame}")
                    return
            
            try:
                # Pass frame directly (can be path or array)
                self.canvas.load_image(frame)
                
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
                if not annotations and self.project_path and self.current_video_id:
                    # Not in cache - load from disk
                    frame_idx_in_video = self._get_frame_idx_in_video(idx)
                    annotations = self.annotation_manager.load_frame_annotations(
                        self.project_path, self.current_video_id, frame_idx_in_video
                    )
                    # Update cache
                    if annotations:
                        self.annotation_manager.set_frame_annotations(idx, annotations)
                self.canvas.set_annotations(annotations)
                
                self.update_instance_list_from_canvas()
                
                # Find the list row for this frame index
                try:
                    list_row = self.frame_list_to_frames_map.index(idx)
                    self.frame_list.setCurrentRow(list_row)
                except ValueError:
                    # Frame not in current filter view
                    pass
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
        print("SAM2 loaded successfully via toolbar!")
        # Only update status label if it exists (may not exist during init)
        if hasattr(self, 'status_label'):
            self.status_label.setText("✓ SAM2 model loaded and ready")
        
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
                
                # Call SAM2 with all accumulated points
                mask = self.sam2.predict_with_points(
                    self.canvas.current_image,
                    positive_points,
                    negative_points
                )
                print(f"Mask shape: {mask.shape}, unique values: {np.unique(mask)}")
                
                # Check if we're editing an existing selected instance
                if self.canvas.selected_mask_idx >= 0 and self.canvas.selected_mask_idx < len(self.canvas.masks):
                    # Update the selected instance
                    self.canvas.masks[self.canvas.selected_mask_idx] = mask
                    self.canvas.update_mask_visualization(self.canvas.selected_mask_idx)
                    self.canvas.active_sam2_mask_idx = self.canvas.selected_mask_idx
                # Check if we're editing an active SAM2 mask
                elif self.canvas.active_sam2_mask_idx >= 0:
                    # Update the active mask
                    self.canvas.masks[self.canvas.active_sam2_mask_idx] = mask
                    self.canvas.update_mask_visualization(self.canvas.active_sam2_mask_idx)
                else:
                    # Create new mask
                    self.canvas.add_mask(mask)
                    self.canvas.active_sam2_mask_idx = len(self.canvas.masks) - 1
                
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
        """Handle box drawn on canvas for SAM2 prompting"""
        print(f"Box drawn: ({x1}, {y1}) to ({x2}, {y2})")
        print(f"SAM2 available: {self.sam2 is not None}")
        print(f"Current tool: {self.canvas.current_tool}")
        
        if self.sam2 and self.canvas.current_tool == 'sam2_box':
            try:
                print("Calling SAM2 box prediction...")
                mask = self.sam2.predict_with_box(
                    self.canvas.current_image,
                    x1, y1, x2, y2
                )
                print(f"Mask shape: {mask.shape}, unique values: {np.unique(mask)}")
                self.canvas.add_mask(mask)
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
        for i, mask in enumerate(self.canvas.masks):
            area = np.sum(mask > 0)
            # Use persistent mask ID instead of array index
            mask_id = self.canvas.mask_ids[i] if i < len(self.canvas.mask_ids) else i+1
            self.instance_list.addItem(f"Instance ID: {mask_id} (area: {area})")
            
    def on_instance_changed(self, idx):
        """Handle instance selection change"""
        if idx >= 0:
            self.canvas.set_selected_instance(idx)
            self.canvas.highlight_instance(idx)
            self.status_label.setText(f"Selected instance {idx+1} - Use Brush/Eraser to edit")
    
    def on_instance_double_clicked(self, item):
        """Handle double-click on instance to edit ID"""
        idx = self.instance_list.currentRow()
        if idx >= 0 and idx < len(self.canvas.mask_ids):
            current_id = self.canvas.mask_ids[idx]
            
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
                if new_id in self.canvas.mask_ids:
                    QMessageBox.warning(
                        self,
                        "Duplicate ID",
                        f"Bee ID {new_id} already exists. Please choose a different ID."
                    )
                    return
                
                # Update the ID
                self.canvas.mask_ids[idx] = new_id
                
                # Update next_mask_id if necessary
                if new_id >= self.canvas.next_mask_id:
                    self.canvas.next_mask_id = new_id + 1
                
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
        
        action = menu.exec(self.instance_list.mapToGlobal(position))
        
        if action == edit_id_action:
            # Trigger the same edit dialog as double-click
            item = self.instance_list.item(idx)
            if item:
                self.on_instance_double_clicked(item)
        elif action == delete_action:
            self.delete_selected_instance()
            
    def delete_selected_instance(self):
        """Delete the currently selected instance"""
        if self.canvas.selected_mask_idx >= 0:
            self.canvas.delete_selected_instance()
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
        
        # Load next frame image
        if isinstance(next_frame, Path) or isinstance(next_frame, str):
            import cv2
            next_frame_image = cv2.imread(str(next_frame))
            if next_frame_image is None:
                QMessageBox.warning(self, "Error", f"Failed to load frame {next_idx}")
                return
            next_frame_image = cv2.cvtColor(next_frame_image, cv2.COLOR_BGR2RGB)
        else:
            next_frame_image = next_frame
        
        # Use SAM2 to propagate masks if available
        propagated_masks = []
        if self.sam2 and current_masks:
            try:
                self.status_label.setText("Propagating with SAM2...")
                QApplication.processEvents()
                
                propagated_masks = self.sam2.propagate_masks_to_frame(
                    self.current_frame_idx,
                    current_masks,
                    next_frame_image
                )
                
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
                    # Still load the frame, but without masks
                    self.load_frame(next_idx)
                    self.update_instance_list_from_canvas()
                    self.status_label.setText(f"Frame {next_idx} loaded - propagation failed, annotate manually")
                    return
                    
            except Exception as e:
                print(f"Error: SAM2 propagation failed: {e}")
                import traceback
                traceback.print_exc()
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
        
        # Clear any existing annotations on the next frame before propagating
        self.annotation_manager.set_frame_annotations(next_idx, [])
        
        # Load the next frame (this will now have no masks)
        self.load_frame(next_idx)
        
        # Add successfully propagated masks to canvas with their original IDs
        for i, mask in enumerate(propagated_masks):
            # Use the corresponding mask_id from the current frame
            if i < len(current_mask_ids):
                mask_id = current_mask_ids[i]
            else:
                mask_id = None  # Let canvas assign a new ID
            self.canvas.add_mask(mask, mask_id)
        
        # Update video next_mask_id tracking after adding masks
        if self.current_video_id:
            self.video_next_mask_id[self.current_video_id] = self.canvas.next_mask_id
        
        self.update_instance_list_from_canvas()
        
        self.status_label.setText(f"✓ Propagated {len(propagated_masks)} instance(s) to frame {next_idx} (replaced existing)")
            
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
        
    def run_inference(self):
        """Run inference on current/all frames"""
        # TODO: Implement inference
        QMessageBox.information(self, "Inference", "Inference feature coming soon!")
        
    def save_annotations(self):
        """Save current annotations and regenerate COCO datasets"""
        if self.project_path:
            try:
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
                    train_path = export_coco_with_tracking(
                        self.project_path,
                        train_videos,
                        'train',
                        class_names=self.annotation_manager.class_names,
                        image_width=self.annotation_manager.image_width,
                        image_height=self.annotation_manager.image_height
                    )
                    coco_files_generated.append(f"Training: {train_path.name}")
                
                # Export val split if videos exist
                if val_videos:
                    val_path = export_coco_with_tracking(
                        self.project_path,
                        val_videos,
                        'val',
                        class_names=self.annotation_manager.class_names,
                        image_width=self.annotation_manager.image_width,
                        image_height=self.annotation_manager.image_height
                    )
                    coco_files_generated.append(f"Validation: {val_path.name}")
                
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
        """Export annotations in COCO format"""
        if not self.project_path:
            QMessageBox.warning(self, "No Project", "Please create or open a project first")
            return
        
        try:
            # Ask user where to save
            default_path = self.project_path / 'annotations' / 'annotations_coco.json'
            file_path, _ = QFileDialog.getSaveFileName(
                self,
                "Export COCO Format",
                str(default_path),
                "JSON Files (*.json);;All Files (*)"
            )
            
            if file_path:
                # Save current frame first
                annotations = self.canvas.get_annotations()
                self.annotation_manager.set_frame_annotations(
                    self.current_frame_idx, annotations
                )
                
                # Export to COCO format
                self.annotation_manager.export_coco(file_path)
                
                QMessageBox.information(
                    self,
                    "Export Successful",
                    f"COCO format annotations exported successfully!\n\n"
                    f"File: {Path(file_path).name}\n"
                    f"Location: {Path(file_path).parent}"
                )
                self.status_label.setText(f"✓ Exported COCO format to {Path(file_path).name}")
                
        except Exception as e:
            error_msg = f"Failed to export COCO format:\n\n{str(e)}"
            self.status_label.setText("✗ Export failed")
            QMessageBox.critical(self, "Export Error", error_msg)
            import traceback
            traceback.print_exc()
        else:
            QMessageBox.warning(self, "No Project", "Please create or open a project first")
            
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
        if self.frames:
            self.load_frame(0)
                
    def load_settings(self):
        """Load application settings"""
        settings = QSettings()
        geometry = settings.value('geometry')
        if geometry:
            self.restoreGeometry(geometry)
    
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
                conf=0.25,  # Confidence threshold
                iou=0.45,   # NMS IoU threshold
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
            
            # Convert YOLO results to annotations
            import cv2
            annotations = []
            
            for i in range(len(result.masks)):
                # Get mask as numpy array (already in image coordinates)
                mask = result.masks[i].data[0].cpu().numpy()
                
                # Resize mask to original image size if needed
                if mask.shape != (result.orig_shape[0], result.orig_shape[1]):
                    mask = cv2.resize(
                        mask, 
                        (result.orig_shape[1], result.orig_shape[0]),
                        interpolation=cv2.INTER_LINEAR
                    )
                
                # Convert to binary mask (threshold at 0.5)
                mask = (mask > 0.5).astype('uint8')
                
                # Get confidence and class
                conf = float(result.boxes[i].conf[0])
                cls = int(result.boxes[i].cls[0])
                
                # Create annotation
                annotation = {
                    'mask': mask,
                    'confidence': conf,
                    'class': cls,
                    'class_name': model.names[cls] if cls < len(model.names) else f'class_{cls}'
                }
                
                annotations.append(annotation)
            
            # Add annotations to the annotation manager
            # Get the next available instance ID
            existing_annotations = self.annotation_manager.get_frame_annotations(self.current_frame_idx)
            next_id = max([ann.get('instance_id', ann.get('mask_id', 0)) for ann in existing_annotations], default=-1) + 1
            
            for i, ann in enumerate(annotations):
                instance_id = next_id + i
                # Create annotation dict in the format expected by AnnotationManager
                annotation_dict = {
                    'mask': ann['mask'],
                    'instance_id': instance_id,
                    'label': f'instance_{instance_id}',
                    'area': int(np.sum(ann['mask'] > 0)),
                    'confidence': ann.get('confidence', 1.0),
                    'class': ann.get('class', 0),
                    'class_name': ann.get('class_name', 'bee')
                }
                self.annotation_manager.add_annotation(
                    self.current_frame_idx,
                    annotation_dict
                )
            
            # Refresh the display
            updated_annotations = self.annotation_manager.get_frame_annotations(self.current_frame_idx)
            self.canvas.set_annotations(updated_annotations)
            self.update_instance_list()
            
            # Update status
            self.status_label.setText(
                f"✓ YOLO inference complete: {len(annotations)} instances detected"
            )
            
            # Show summary
            QMessageBox.information(
                self, "Inference Complete",
                f"YOLO detected {len(annotations)} instance(s) in the current frame.\n\n"
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
                self, "Import Error",
                f"Could not import required libraries:\n{str(e)}\n\n"
                "Make sure ultralytics and opencv-python are installed."
            )
        except Exception as e:
            import traceback
            error_msg = traceback.format_exc()
            QMessageBox.critical(
                self, "Refinement Error",
                f"Error refining mask:\n{str(e)}\n\n{error_msg}"
            )
            self.status_label.setText("Refinement failed")
    
    def _refine_mask_by_index(self, mask_idx):
        """
        Refine a mask by index using YOLO.
        
        Args:
            mask_idx: Index of the mask in self.canvas.masks
            
        Returns:
            Tuple of (success: bool, message: str, stats: dict)
        """
        import cv2
        
        try:
            # Validate mask index
            if mask_idx >= len(self.canvas.masks):
                return False, "Invalid mask index", {}
            
            original_mask = self.canvas.masks[mask_idx]
            if original_mask is None or np.sum(original_mask > 0) == 0:
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
            
            # Run YOLO on the crop
            model = self.yolo_refine_toolbar.get_model()
            
            # Convert crop to BGR for YOLO (it expects BGR)
            crop_bgr = cv2.cvtColor(crop, cv2.COLOR_RGB2BGR)
            
            results = model.predict(
                source=crop_bgr,
                conf=0.1,  # Lower confidence threshold for refinement
                iou=0.45,
                augment=False,  # Disable test-time augmentation for consistent results
                verbose=False
            )
            
            if not results or len(results) == 0 or results[0].masks is None or len(results[0].masks) == 0:
                return False, "No detections", {}
            
            result = results[0]
            
            # Crop the original mask to the same region
            original_mask_crop = original_mask[crop_y1:crop_y2, crop_x1:crop_x2]
            
            # Find the prediction with highest IoU with the original mask
            best_iou = 0
            best_pred_idx = -1
            
            for i in range(len(result.masks)):
                # Get predicted mask
                pred_mask = result.masks[i].data[0].cpu().numpy()
                
                # Resize to crop size if needed
                if pred_mask.shape != (crop.shape[0], crop.shape[1]):
                    pred_mask = cv2.resize(
                        pred_mask,
                        (crop.shape[1], crop.shape[0]),
                        interpolation=cv2.INTER_LINEAR
                    )
                
                # Binarize
                pred_mask_binary = (pred_mask > 0.5).astype('uint8')
                original_mask_binary = (original_mask_crop > 0).astype('uint8')
                
                # Calculate IoU
                intersection = np.sum(np.logical_and(pred_mask_binary, original_mask_binary))
                union = np.sum(np.logical_or(pred_mask_binary, original_mask_binary))
                
                if union > 0:
                    iou = intersection / union
                    if iou > best_iou:
                        best_iou = iou
                        best_pred_idx = i
            
            if best_pred_idx < 0 or best_iou < 0.1:
                return False, f"No match (best IoU: {best_iou:.3f})", {"iou": best_iou}
            
            # Get the best prediction mask
            best_pred_mask = result.masks[best_pred_idx].data[0].cpu().numpy()
            
            # Resize to crop size if needed
            if best_pred_mask.shape != (crop.shape[0], crop.shape[1]):
                best_pred_mask = cv2.resize(
                    best_pred_mask,
                    (crop.shape[1], crop.shape[0]),
                    interpolation=cv2.INTER_LINEAR
                )
            
            # Binarize
            best_pred_mask = (best_pred_mask > 0.5).astype('uint8') * 255
            
            # Create full-size mask
            refined_mask = np.zeros_like(original_mask)
            refined_mask[crop_y1:crop_y2, crop_x1:crop_x2] = best_pred_mask
            
            # Update the mask in canvas
            self.canvas.masks[mask_idx] = refined_mask
            
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
        
        if not self.canvas.masks or len(self.canvas.masks) == 0:
            QMessageBox.warning(
                self, "No Instances",
                "There are no instances in the current frame to refine."
            )
            return
        
        try:
            num_masks = len(self.canvas.masks)
            
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
            
            # Process each mask
            for idx in range(num_masks):
                self.status_label.setText(f"Refining instance {idx + 1}/{num_masks}...")
                QApplication.processEvents()
                
                success, message, stats = self._refine_mask_by_index(idx)
                
                if success:
                    refined_count += 1
                    total_iou += stats.get("iou", 0)
                    # Update visualization for this mask
                    self.canvas.update_mask_visualization(idx)
                else:
                    failed_count += 1
            
            # Emit annotation changed signal
            if refined_count > 0:
                self.canvas.annotation_changed.emit()
            
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
