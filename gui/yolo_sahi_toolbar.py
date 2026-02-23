"""
YOLO SAHI (Sliced Aided Hyper Inference) toolbar for running sliced inference
"""

from PyQt6.QtWidgets import (QWidget, QHBoxLayout, QVBoxLayout, QToolButton, QPushButton,
                             QLabel, QFileDialog, QMessageBox, QSpinBox, QDoubleSpinBox,
                             QMenu, QComboBox)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QAction
from pathlib import Path


class YOLOSAHIToolbar(QWidget):
    """Toolbar for YOLO SAHI model inference with sliced windows"""
    
    inference_requested = pyqtSignal()  # Signal to request SAHI inference on current frame
    box_inference_mode_requested = pyqtSignal(bool)  # Signal to enable/disable box drawing mode
    box_inference_requested = pyqtSignal()  # Signal to run inference on drawn box
    soho_inference_requested = pyqtSignal()  # Signal to request SOHO inference on current frame
    propagate_soho_requested = pyqtSignal()  # Signal to propagate SOHO to next frame
    propagate_soho_to_selected_requested = pyqtSignal()  # Signal to propagate SOHO to selected frame
    propagate_soho_through_video_requested = pyqtSignal()  # Signal to propagate SOHO through entire video
    track_from_last_frame_requested = pyqtSignal()  # Signal to track from last annotated frame
    
    def __init__(self, parent=None, checkpoint_path=None):
        super().__init__(parent)
        
        self.model_path = None
        self.model = None
        self.box_inference_mode = False
        self.box_is_drawn = False  # Track if inference box is drawn
        
        # SAHI parameters
        self.slice_height = 1280
        self.slice_width = 1280
        self.overlap_ratio = 0.55
        
        # Postprocessing parameters
        self.postprocess_type = "GREEDYNMM"
        self.postprocess_match_metric = "IOS"
        self.postprocess_match_threshold = 0.65
        self.postprocess_class_agnostic = True
        
        self.init_ui()
        
        # Load checkpoint if provided
        if checkpoint_path:
            self._load_checkpoint_from_path(checkpoint_path)
        
    def init_ui(self):
        """Initialize UI"""
        # Main vertical layout to stack two rows
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(5, 5, 5, 5)
        main_layout.setSpacing(5)
        
        # --- Row 1: Model management and slice parameters ---
        row1_layout = QHBoxLayout()
        row1_layout.setSpacing(10)
        
        # YOLO SAHI section label
        row1_layout.addWidget(QLabel("<b>SAHI YOLO:</b>"))
        
        # Load model button
        self.load_model_btn = QPushButton("Load Checkpoint...")
        self.load_model_btn.setToolTip("Load a SAHI-trained YOLO model checkpoint (.pt file)")
        self.load_model_btn.clicked.connect(self.load_checkpoint)
        row1_layout.addWidget(self.load_model_btn)
        
        # Train menu button
        self.train_menu_btn = QToolButton()
        self.train_menu_btn.setText("Train...")
        self.train_menu_btn.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        self.train_menu_btn.setToolTip("Train a SAHI model")
        
        # Create menu
        train_menu = QMenu(self)
        
        # Add train action
        train_action = QAction("Train SAHI Model", self)
        train_action.setToolTip("Train a new SAHI YOLO model on full images with enhanced augmentation")
        train_action.triggered.connect(self.on_train_model)
        train_menu.addAction(train_action)
        
        self.train_menu_btn.setMenu(train_menu)
        row1_layout.addWidget(self.train_menu_btn)
        
        # Model status label
        self.model_status_label = QLabel("No model loaded")
        self.model_status_label.setStyleSheet("color: gray; font-style: italic;")
        row1_layout.addWidget(self.model_status_label)
        
        # Separator
        row1_layout.addWidget(self.create_separator())
        
        # SAHI slice parameters
        row1_layout.addWidget(QLabel("Slice Size:"))
        self.slice_size_spinbox = QSpinBox()
        self.slice_size_spinbox.setMinimum(320)
        self.slice_size_spinbox.setMaximum(3000)
        self.slice_size_spinbox.setSingleStep(64)
        self.slice_size_spinbox.setValue(self.slice_width)
        self.slice_size_spinbox.setSuffix(" px")
        self.slice_size_spinbox.setToolTip("Size of each slice (width and height)")
        self.slice_size_spinbox.valueChanged.connect(self.on_slice_size_changed)
        row1_layout.addWidget(self.slice_size_spinbox)
        
        row1_layout.addWidget(QLabel("Overlap:"))
        self.overlap_spinbox = QDoubleSpinBox()
        self.overlap_spinbox.setMinimum(0.0)
        self.overlap_spinbox.setMaximum(0.95)
        self.overlap_spinbox.setSingleStep(0.05)
        self.overlap_spinbox.setValue(self.overlap_ratio)
        self.overlap_spinbox.setDecimals(2)
        self.overlap_spinbox.setToolTip("Overlap ratio between slices (0.0 to 0.5)")
        self.overlap_spinbox.valueChanged.connect(self.on_overlap_changed)
        row1_layout.addWidget(self.overlap_spinbox)
        
        row1_layout.addStretch()
        
        # --- Row 2: Inference and propagation buttons ---
        row2_layout = QHBoxLayout()
        row2_layout.setSpacing(10)
        
        # Inference section label
        row2_layout.addWidget(QLabel("<b>SOHO Inference:</b>"))
        
        # Run SOHO inference button
        self.soho_btn = QPushButton("Run SOHO")
        self.soho_btn.setToolTip(
            "SOHO: Sliced Overlapping Heuristic Optimization\n"
            "Run SOHO inference on the current frame to detect instances.\n"
            "Uses custom slicing with edge filtering to prevent split detections."
        )
        self.soho_btn.setEnabled(False)  # Disabled until model is loaded
        self.soho_btn.clicked.connect(self.on_soho_requested)
        row2_layout.addWidget(self.soho_btn)
        
        # Box inference button (state-based)
        self.box_inference_btn = QPushButton("Draw Box")
        self.box_inference_btn.setToolTip(
            "Draw a box on the image to run SAHI inference only on that region.\n"
            "After drawing, you can adjust the box, then click again to run inference."
        )
        self.box_inference_btn.setEnabled(False)  # Disabled until model is loaded
        self.box_inference_btn.clicked.connect(self.on_box_inference_btn_clicked)
        row2_layout.addWidget(self.box_inference_btn)
        
        # Separator
        row2_layout.addWidget(self.create_separator())
        
        # Propagation section label
        row2_layout.addWidget(QLabel("<b>Propagation:</b>"))
        
        # Propagate SOHO to next frame button
        self.propagate_soho_btn = QPushButton("Propagate to Next")
        self.propagate_soho_btn.setToolTip(
            "Run SOHO on the next frame and match detections to current frame instances.\n"
            "Uses ByteTrack-style matching to maintain consistent IDs across frames."
        )
        self.propagate_soho_btn.setEnabled(False)  # Disabled until model is loaded
        self.propagate_soho_btn.clicked.connect(self.on_propagate_soho_requested)
        row2_layout.addWidget(self.propagate_soho_btn)
        
        # Propagate SOHO to selected frame button
        self.propagate_soho_to_selected_btn = QPushButton("Propagate to Selected")
        self.propagate_soho_to_selected_btn.setToolTip(
            "Run SOHO and propagate through all frames until the next selected frame.\n"
            "Maintains track IDs across multiple frames using ByteTrack matching."
        )
        self.propagate_soho_to_selected_btn.setEnabled(False)  # Disabled until model is loaded
        self.propagate_soho_to_selected_btn.clicked.connect(self.on_propagate_soho_to_selected_requested)
        row2_layout.addWidget(self.propagate_soho_to_selected_btn)
        
        # Propagate SOHO through entire video button
        self.propagate_soho_through_video_btn = QPushButton("Propagate Through Video")
        self.propagate_soho_through_video_btn.setToolTip(
            "Run SOHO and propagate through all remaining frames in the video.\n"
            "Maintains track IDs across all frames using ByteTrack matching."
        )
        self.propagate_soho_through_video_btn.setEnabled(False)  # Disabled until model is loaded
        self.propagate_soho_through_video_btn.clicked.connect(self.on_propagate_soho_through_video_requested)
        row2_layout.addWidget(self.propagate_soho_through_video_btn)
        
        # Track from last frame button
        self.track_from_last_frame_btn = QPushButton("Track from Last")
        self.track_from_last_frame_btn.setToolTip(
            "Match current frame instances to the last annotated frame using tracking.\n"
            "No inference - just ByteTrack matching to assign consistent IDs."
        )
        self.track_from_last_frame_btn.setEnabled(True)  # Always enabled - method handles requirements
        self.track_from_last_frame_btn.clicked.connect(self.on_track_from_last_frame_requested)
        row2_layout.addWidget(self.track_from_last_frame_btn)
        
        row2_layout.addStretch()
        
        # Add both rows to main layout
        main_layout.addLayout(row1_layout)
        main_layout.addLayout(row2_layout)
        
    def create_separator(self):
        """Create a vertical separator"""
        from PyQt6.QtWidgets import QFrame
        line = QFrame()
        line.setFrameShape(QFrame.Shape.VLine)
        line.setFrameShadow(QFrame.Shadow.Sunken)
        return line
    
    def on_slice_size_changed(self, value):
        """Handle slice size change"""
        self.slice_height = value
        self.slice_width = value
    
    def on_overlap_changed(self, value):
        """Handle overlap ratio change"""
        self.overlap_ratio = value
    
    def on_postprocess_type_changed(self, value):
        """Handle postprocess type change"""
        self.postprocess_type = value
    
    def on_postprocess_metric_changed(self, value):
        """Handle postprocess metric change"""
        self.postprocess_match_metric = value
    
    def on_postprocess_threshold_changed(self, value):
        """Handle postprocess threshold change"""
        self.postprocess_match_threshold = value
    
    def on_train_model(self):
        """Handle train model button click"""
        # Find the main window and call its training method
        parent_window = self.parent()
        while parent_window and not hasattr(parent_window, 'train_yolo_model_sahi'):
            parent_window = parent_window.parent()
        
        if parent_window and hasattr(parent_window, 'train_yolo_model_sahi'):
            parent_window.train_yolo_model_sahi()
        else:
            QMessageBox.warning(
                self,
                "Training Not Available",
                "Training function not found. Please ensure you're running from the main window."
            )
        
    def load_checkpoint(self):
        """Load a YOLO checkpoint file via file dialog"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select YOLO SAHI Checkpoint",
            str(Path.home()),
            "PyTorch Model Files (*.pt);;All Files (*)"
        )
        
        if not file_path:
            return
        
        self._load_checkpoint_from_path(file_path, show_dialogs=True)
    
    def _load_checkpoint_from_path(self, file_path, show_dialogs=False):
        """Load a YOLO checkpoint from a file path"""
        try:
            # Import ultralytics YOLO
            from ultralytics import YOLO
            
            # Load the model
            model = YOLO(file_path)
            
            # Verify it's a segmentation model
            if not hasattr(model, 'predictor') or 'segment' not in str(model.task):
                raise ValueError("Model is not a segmentation model. Please load a YOLO segmentation model (trained with -seg variant).")
            
            # Store model
            self.model = model
            self.model_path = Path(file_path)
            
            # Update UI
            model_name = self.model_path.name
            self.model_status_label.setText(f"✓ {model_name}")
            self.model_status_label.setStyleSheet("color: green;")
            self.soho_btn.setEnabled(True)
            self.box_inference_btn.setEnabled(True)
            self.propagate_soho_btn.setEnabled(True)
            self.propagate_soho_to_selected_btn.setEnabled(True)
            self.propagate_soho_through_video_btn.setEnabled(True)
            
            if show_dialogs:
                QMessageBox.information(
                    self,
                    "Model Loaded",
                    f"Successfully loaded YOLO SAHI model:\n{model_name}\n\n"
                    f"Task: {model.task}\n"
                    f"Ready for sliced inference!"
                )
            else:
                print(f"SAHI YOLO loaded successfully: {model_name} (Task: {model.task})")
            
        except ImportError:
            error_msg = "Could not import ultralytics. Please install it with: pip install ultralytics"
            if show_dialogs:
                QMessageBox.critical(self, "Import Error", error_msg)
            else:
                print(f"ERROR: {error_msg}")
        except Exception as e:
            error_msg = f"Failed to load model: {str(e)}"
            if show_dialogs:
                QMessageBox.critical(self, "Error Loading Model", error_msg)
            else:
                print(f"ERROR: {error_msg}")
            self.model = None
            self.model_path = None
            self.model_status_label.setText("Failed to load model")
            self.model_status_label.setStyleSheet("color: red;")
            self.soho_btn.setEnabled(False)
            self.box_inference_btn.setEnabled(False)
            self.propagate_soho_btn.setEnabled(False)
            self.propagate_soho_to_selected_btn.setEnabled(False)
            self.propagate_soho_through_video_btn.setEnabled(False)
    
    def on_inference_requested(self):
        """Handle inference button click"""
        if self.model is None:
            QMessageBox.warning(
                self,
                "No Model",
                "Please load a YOLO SAHI checkpoint first."
            )
            return
            
        self.inference_requested.emit()
    
    def on_soho_requested(self):
        """Handle SOHO button click"""
        if self.model is None:
            QMessageBox.warning(
                self,
                "No Model",
                "Please load a YOLO SAHI checkpoint first."
            )
            return
            
        self.soho_inference_requested.emit()
    
    def on_propagate_soho_requested(self):
        """Handle propagate SOHO to next frame button click"""
        if self.model is None:
            QMessageBox.warning(
                self,
                "No Model",
                "Please load a YOLO SAHI checkpoint first."
            )
            return
            
        self.propagate_soho_requested.emit()
    
    def on_propagate_soho_to_selected_requested(self):
        """Handle propagate SOHO to selected frame button click"""
        if self.model is None:
            QMessageBox.warning(
                self,
                "No Model",
                "Please load a YOLO SAHI checkpoint first."
            )
            return
            
        self.propagate_soho_to_selected_requested.emit()
    
    def on_propagate_soho_through_video_requested(self):
        """Handle propagate SOHO through entire video button click"""
        if self.model is None:
            QMessageBox.warning(
                self,
                "No Model",
                "Please load a YOLO SAHI checkpoint first."
            )
            return
            
        self.propagate_soho_through_video_requested.emit()
    
    def on_track_from_last_frame_requested(self):
        """Handle track from last frame button click"""
        # No model check needed - tracking works on existing annotations
        self.track_from_last_frame_requested.emit()
    
    def on_box_inference_btn_clicked(self):
        """Handle box inference button click - two states: draw box or run inference"""
        if self.model is None:
            QMessageBox.warning(
                self,
                "No Model",
                "Please load a YOLO SAHI checkpoint first."
            )
            return
        
        if not self.box_is_drawn:
            # State 1: Enable drawing mode
            self.box_inference_mode = True
            self.box_inference_mode_requested.emit(True)
            self.box_inference_btn.setText("Drawing Box...")
            self.box_inference_btn.setStyleSheet("background-color: #4CAF50; color: white;")
        else:
            # State 2: Run inference on drawn box
            self.box_inference_requested.emit()
            # Button will be reset by set_box_drawn(False) after inference
    
    def set_box_drawn(self, is_drawn):
        """Update button state when box is drawn or cleared"""  
        self.box_is_drawn = is_drawn
        if is_drawn:
            # Box is drawn - change button to run inference
            self.box_inference_btn.setText("Run Inference on Box")
            self.box_inference_btn.setStyleSheet("background-color: #FF9800; color: white;")  # Orange
        else:
            # Box cleared - reset to drawing mode
            self.box_inference_btn.setText("Draw Box for Inference")
            self.box_inference_btn.setStyleSheet("")
            self.box_inference_mode = False
    
    def on_box_inference_mode_clicked(self, checked):
        """Legacy method - kept for compatibility"""
        pass
    
    def get_model(self):
        """Get the loaded YOLO model"""
        return self.model
    
    def is_model_loaded(self):
        """Check if a model is loaded"""
        return self.model is not None
    
    def get_sahi_params(self):
        """Get SAHI parameters"""
        return {
            'slice_height': self.slice_height,
            'slice_width': self.slice_width,
            'overlap_height_ratio': self.overlap_ratio,
            'overlap_width_ratio': self.overlap_ratio,
            'postprocess_type': self.postprocess_type,
            'postprocess_match_metric': self.postprocess_match_metric,
            'postprocess_match_threshold': self.postprocess_match_threshold,
            'postprocess_class_agnostic': self.postprocess_class_agnostic
        }
