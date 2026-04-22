"""
YOLO bounding box detection toolbar for running trained bbox detection models
"""

from PyQt6.QtWidgets import (QWidget, QHBoxLayout, QToolButton, QPushButton,
                             QLabel, QFileDialog, QMessageBox)
from PyQt6.QtCore import Qt, pyqtSignal
from pathlib import Path


class YOLOBBoxToolbar(QWidget):
    """Toolbar for YOLO bounding box detection model inference"""
    
    inference_requested = pyqtSignal()  # Signal to request inference on current frame
    track_from_last_requested = pyqtSignal()  # Signal to request tracking from last annotated frame
    propagate_requested = pyqtSignal()  # Signal to request propagation through video
    
    def __init__(self, parent=None, checkpoint_path=None):
        super().__init__(parent)
        
        self.model_path = None
        self.model = None
        
        self.init_ui()
        
        # Load checkpoint if provided
        if checkpoint_path:
            self._load_checkpoint_from_path(checkpoint_path)
        
    def init_ui(self):
        """Initialize UI"""
        layout = QHBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(10)
        
        # YOLO Model section label
        layout.addWidget(QLabel("<b>Use YOLO BBox Detection to Find Bees:</b>"))
        
        # Load model button
        self.load_model_btn = QPushButton("Load Checkpoint...")
        self.load_model_btn.setToolTip("Load a trained YOLO detection model checkpoint (.pt file)")
        self.load_model_btn.clicked.connect(self.load_checkpoint)
        layout.addWidget(self.load_model_btn)
        
        # Model status label
        self.model_status_label = QLabel("No model loaded")
        self.model_status_label.setStyleSheet("color: gray; font-style: italic;")
        layout.addWidget(self.model_status_label)
        
        # Separator
        layout.addWidget(self.create_separator())
        
        # Run inference button
        self.inference_btn = QPushButton("Run on Current Frame")
        self.inference_btn.setToolTip("Run YOLO bbox detection on the currently displayed frame")
        self.inference_btn.setEnabled(False)  # Disabled until model is loaded
        self.inference_btn.clicked.connect(self.on_inference_requested)
        layout.addWidget(self.inference_btn)
        
        # Separator
        layout.addWidget(self.create_separator())
        
        # Track from last frame button
        self.track_btn = QPushButton("Track from Last Frame")
        self.track_btn.setToolTip("Match YOLO detections on current frame to annotations from the last annotated frame")
        self.track_btn.setEnabled(False)  # Disabled until model is loaded
        self.track_btn.clicked.connect(self.on_track_requested)
        layout.addWidget(self.track_btn)
        
        # Separator
        layout.addWidget(self.create_separator())
        
        # Propagate button
        self.propagate_btn = QPushButton("Propagate Through Video")
        self.propagate_btn.setToolTip("Run YOLO detection and tracking through multiple frames")
        self.propagate_btn.setEnabled(False)  # Disabled until model is loaded
        self.propagate_btn.clicked.connect(self.on_propagate_requested)
        layout.addWidget(self.propagate_btn)
        
        layout.addStretch()
        
    def create_separator(self):
        """Create a vertical separator"""
        from PyQt6.QtWidgets import QFrame
        line = QFrame()
        line.setFrameShape(QFrame.Shape.VLine)
        line.setFrameShadow(QFrame.Shadow.Sunken)
        return line
        
    def load_checkpoint(self):
        """Load a YOLO checkpoint file via file dialog"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select YOLO Detection Checkpoint",
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
            
            # Verify it's a detection model (not segmentation or other tasks)
            if not hasattr(model, 'predictor'):
                raise ValueError("Invalid YOLO model. Please load a valid YOLO detection model.")
            
            # Check task - should be 'detect' for bbox-only models
            task = str(model.task) if hasattr(model, 'task') else 'detect'
            if 'segment' in task or 'pose' in task or 'classify' in task:
                QMessageBox.warning(
                    self,
                    "Model Type Mismatch",
                    f"This toolbar is for bounding box detection models.\n\n"
                    f"The loaded model appears to be: {task}\n\n"
                    f"For segmentation models, use the 'YOLO Coarse' toolbar instead.\n\n"
                    f"Continuing anyway - bounding boxes will be extracted from predictions."
                )
            
            # Store model
            self.model = model
            self.model_path = Path(file_path)
            
            # Update UI
            model_name = self.model_path.name
            self.model_status_label.setText(f"✓ {model_name}")
            self.model_status_label.setStyleSheet("color: green;")
            self.inference_btn.setEnabled(True)
            self.track_btn.setEnabled(True)
            self.propagate_btn.setEnabled(True)
            
            if show_dialogs:
                QMessageBox.information(
                    self,
                    "Model Loaded",
                    f"Successfully loaded YOLO detection model:\n{model_name}\n\n"
                    f"Task: {task}\n"
                    f"Ready for bounding box detection!"
                )
            else:
                print(f"YOLO BBox Detection loaded successfully: {model_name} (Task: {task})")
            
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
            self.inference_btn.setEnabled(False)
            self.track_btn.setEnabled(False)
            self.propagate_btn.setEnabled(False)
    
    def on_inference_requested(self):
        """Handle inference button click"""
        if self.model is None:
            QMessageBox.warning(
                self,
                "No Model",
                "Please load a YOLO checkpoint first."
            )
            return
            
        self.inference_requested.emit()
    
    def on_track_requested(self):
        """Handle track from last frame button click"""
        if self.model is None:
            QMessageBox.warning(
                self,
                "No Model",
                "Please load a YOLO checkpoint first."
            )
            return
            
        self.track_from_last_requested.emit()
    
    def on_propagate_requested(self):
        """Handle propagate button click"""
        if self.model is None:
            QMessageBox.warning(
                self,
                "No Model",
                "Please load a YOLO checkpoint first."
            )
            return
            
        self.propagate_requested.emit()
    
    def get_model(self):
        """Get the loaded YOLO model"""
        return self.model
    
    def is_model_loaded(self):
        """Check if a model is loaded"""
        return self.model is not None
