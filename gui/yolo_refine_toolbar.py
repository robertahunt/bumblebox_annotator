"""
YOLO refinement toolbar for refining instance masks using a trained model
"""

from PyQt6.QtWidgets import (QWidget, QHBoxLayout, QPushButton,
                             QLabel, QFileDialog, QMessageBox, QSpinBox)
from PyQt6.QtCore import Qt, pyqtSignal
from pathlib import Path


class YOLORefinementToolbar(QWidget):
    """Toolbar for YOLO mask refinement"""
    
    refine_requested = pyqtSignal()  # Signal to request mask refinement
    refine_all_requested = pyqtSignal()  # Signal to request refinement of all masks
    
    def __init__(self, parent=None, checkpoint_path=None):
        super().__init__(parent)
        
        self.model_path = None
        self.model = None
        self.padding = 50  # Default padding around crop
        
        self.init_ui()
        
        # Load checkpoint if provided
        if checkpoint_path:
            self._load_checkpoint_from_path(checkpoint_path)
        
    def init_ui(self):
        """Initialize UI"""
        layout = QHBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(10)
        
        # YOLO Refinement section label
        layout.addWidget(QLabel("<b>Run Fine Grained YOLO model to Refine Instance Masks:</b>"))
        
        # Load model button
        self.load_model_btn = QPushButton("Load Checkpoint...")
        self.load_model_btn.setToolTip("Load a trained YOLO segmentation model checkpoint (.pt file)")
        self.load_model_btn.clicked.connect(self.load_checkpoint)
        layout.addWidget(self.load_model_btn)
        
        # Model status label
        self.model_status_label = QLabel("No model loaded")
        self.model_status_label.setStyleSheet("color: gray; font-style: italic;")
        layout.addWidget(self.model_status_label)
        
        # Separator
        layout.addWidget(self.create_separator())
        
        # Crop padding label and spinbox
        layout.addWidget(QLabel("Crop Padding:"))
        self.padding_spinbox = QSpinBox()
        self.padding_spinbox.setMinimum(10)
        self.padding_spinbox.setMaximum(500)
        self.padding_spinbox.setValue(self.padding)
        self.padding_spinbox.setSuffix(" px")
        self.padding_spinbox.setToolTip("Padding around instance bounding box when cropping")
        self.padding_spinbox.valueChanged.connect(self.on_padding_changed)
        layout.addWidget(self.padding_spinbox)
        
        # Separator
        layout.addWidget(self.create_separator())
        
        # Refine mask button
        self.refine_btn = QPushButton("Refine Mask")
        self.refine_btn.setToolTip(
            "Refine the currently selected instance mask using the trained model.\n"
            "The model will be run on a crop around the instance and the mask will be updated."
        )
        self.refine_btn.setEnabled(False)  # Disabled until model is loaded
        self.refine_btn.clicked.connect(self.on_refine_requested)
        layout.addWidget(self.refine_btn)
        
        # Refine all button
        self.refine_all_btn = QPushButton("Refine All")
        self.refine_all_btn.setToolTip(
            "Refine all instances in the current frame using the trained model.\n"
            "Each instance will be processed sequentially."
        )
        self.refine_all_btn.setEnabled(False)  # Disabled until model is loaded
        self.refine_all_btn.clicked.connect(self.on_refine_all_requested)
        layout.addWidget(self.refine_all_btn)
        
        layout.addStretch()
        
    def create_separator(self):
        """Create a vertical separator"""
        from PyQt6.QtWidgets import QFrame
        line = QFrame()
        line.setFrameShape(QFrame.Shape.VLine)
        line.setFrameShadow(QFrame.Shadow.Sunken)
        return line
    
    def on_padding_changed(self, value):
        """Handle padding change"""
        self.padding = value
        
    def load_checkpoint(self):
        """Load a YOLO checkpoint file via file dialog"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select YOLO Checkpoint",
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
            self.refine_btn.setEnabled(True)
            self.refine_all_btn.setEnabled(True)
            
            if show_dialogs:
                QMessageBox.information(
                    self,
                    "Model Loaded",
                    f"Successfully loaded YOLO refinement model:\n{model_name}\n\n"
                    f"Task: {model.task}\n"
                    f"Ready for mask refinement!"
                )
            else:
                print(f"Fine YOLO loaded successfully: {model_name} (Task: {model.task})")
            
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
            self.refine_btn.setEnabled(False)
            self.refine_all_btn.setEnabled(False)
    
    def on_refine_requested(self):
        """Handle refine button click"""
        if self.model is None:
            QMessageBox.warning(
                self,
                "No Model",
                "Please load a YOLO checkpoint first."
            )
            return
            
        self.refine_requested.emit()
    
    def on_refine_all_requested(self):
        """Handle refine all button click"""
        if self.model is None:
            QMessageBox.warning(
                self,
                "No Model",
                "Please load a YOLO checkpoint first."
            )
            return
            
        self.refine_all_requested.emit()
    
    def get_model(self):
        """Get the loaded YOLO model"""
        return self.model
    
    def get_padding(self):
        """Get the current crop padding"""
        return self.padding
    
    def is_model_loaded(self):
        """Check if a model is loaded"""
        return self.model is not None
