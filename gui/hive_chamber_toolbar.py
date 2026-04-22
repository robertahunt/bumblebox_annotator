"""
Hive and Chamber YOLO inference toolbar for running trained hive/chamber models
"""

from PyQt6.QtWidgets import (QWidget, QHBoxLayout, QToolButton, QPushButton,
                             QLabel, QFileDialog, QMessageBox, QFrame)
from PyQt6.QtCore import Qt, pyqtSignal
from pathlib import Path


class HiveChamberToolbar(QWidget):
    """Toolbar for Hive, Chamber, and Pollen YOLO model inference"""
    
    # Signals for running inference
    hive_inference_requested = pyqtSignal()  # Signal to request hive inference on current frame
    chamber_inference_requested = pyqtSignal()  # Signal to request chamber inference on current frame
    pollen_inference_requested = pyqtSignal()  # Signal to request pollen inference on current frame
    both_inference_requested = pyqtSignal()  # Signal to run all three models on current frame
    
    def __init__(self, parent=None, hive_checkpoint_path=None, chamber_checkpoint_path=None, pollen_checkpoint_path=None):
        super().__init__(parent)
        
        self.hive_model_path = None
        self.hive_model = None
        self.chamber_model_path = None
        self.chamber_model = None
        self.pollen_model_path = None
        self.pollen_model = None
        
        self.init_ui()
        
        # Load checkpoints if provided
        if hive_checkpoint_path:
            self._load_hive_checkpoint_from_path(hive_checkpoint_path)
        if chamber_checkpoint_path:
            self._load_chamber_checkpoint_from_path(chamber_checkpoint_path)
        if pollen_checkpoint_path:
            self._load_pollen_checkpoint_from_path(pollen_checkpoint_path)
        
    def init_ui(self):
        """Initialize UI"""
        layout = QHBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(10)
        
        # Section label
        layout.addWidget(QLabel("<b>Hive/Chamber/Pollen Detection:</b>"))
        
        # === Hive Model Section ===
        
        # Load hive model button
        self.load_hive_btn = QPushButton("Load Hive Model...")
        self.load_hive_btn.setToolTip("Load a trained YOLO hive detection model (.pt file)")
        self.load_hive_btn.clicked.connect(self.load_hive_checkpoint)
        layout.addWidget(self.load_hive_btn)
        
        # Hive model status label
        self.hive_status_label = QLabel("Not loaded")
        self.hive_status_label.setStyleSheet("color: gray; font-style: italic;")
        layout.addWidget(self.hive_status_label)
        
        # Run hive inference button
        self.hive_inference_btn = QPushButton("Run Hive")
        self.hive_inference_btn.setToolTip("Run hive detection on the current frame")
        self.hive_inference_btn.setEnabled(False)  # Disabled until model is loaded
        self.hive_inference_btn.clicked.connect(self.on_hive_inference_requested)
        layout.addWidget(self.hive_inference_btn)
        
        # Separator
        layout.addWidget(self.create_separator())
        
        # === Chamber Model Section ===
        
        # Load chamber model button
        self.load_chamber_btn = QPushButton("Load Chamber Model...")
        self.load_chamber_btn.setToolTip("Load a trained YOLO chamber detection model (.pt file)")
        self.load_chamber_btn.clicked.connect(self.load_chamber_checkpoint)
        layout.addWidget(self.load_chamber_btn)
        
        # Chamber model status label
        self.chamber_status_label = QLabel("Not loaded")
        self.chamber_status_label.setStyleSheet("color: gray; font-style: italic;")
        layout.addWidget(self.chamber_status_label)
        
        # Run chamber inference button
        self.chamber_inference_btn = QPushButton("Run Chamber")
        self.chamber_inference_btn.setToolTip("Run chamber detection on the current frame")
        self.chamber_inference_btn.setEnabled(False)  # Disabled until model is loaded
        self.chamber_inference_btn.clicked.connect(self.on_chamber_inference_requested)
        layout.addWidget(self.chamber_inference_btn)
        
        # Separator
        layout.addWidget(self.create_separator())
        
        # === Pollen Model Section ===
        
        # Load pollen model button
        self.load_pollen_btn = QPushButton("Load Pollen Model...")
        self.load_pollen_btn.setToolTip("Load a trained YOLO pollen detection model (.pt file)")
        self.load_pollen_btn.clicked.connect(self.load_pollen_checkpoint)
        layout.addWidget(self.load_pollen_btn)
        
        # Pollen model status label
        self.pollen_status_label = QLabel("Not loaded")
        self.pollen_status_label.setStyleSheet("color: gray; font-style: italic;")
        layout.addWidget(self.pollen_status_label)
        
        # Run pollen inference button
        self.pollen_inference_btn = QPushButton("Run Pollen")
        self.pollen_inference_btn.setToolTip("Run pollen detection on the current frame")
        self.pollen_inference_btn.setEnabled(False)  # Disabled until model is loaded
        self.pollen_inference_btn.clicked.connect(self.on_pollen_inference_requested)
        layout.addWidget(self.pollen_inference_btn)
        
        # Separator
        layout.addWidget(self.create_separator())
        
        # === Run All Button ===
        
        # Run all button
        self.both_inference_btn = QPushButton("Run All")
        self.both_inference_btn.setToolTip("Run all loaded models (hive, chamber, pollen) on the current frame")
        self.both_inference_btn.setEnabled(False)  # Disabled until at least one model is loaded
        self.both_inference_btn.clicked.connect(self.on_both_inference_requested)
        layout.addWidget(self.both_inference_btn)
        
        layout.addStretch()
        
    def create_separator(self):
        """Create a vertical separator"""
        line = QFrame()
        line.setFrameShape(QFrame.Shape.VLine)
        line.setFrameShadow(QFrame.Shadow.Sunken)
        return line
        
    def load_hive_checkpoint(self):
        """Load a YOLO hive checkpoint file via file dialog"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select YOLO Hive Checkpoint",
            str(Path.home()),
            "PyTorch Model Files (*.pt);;All Files (*)"
        )
        
        if not file_path:
            return
        
        self._load_hive_checkpoint_from_path(file_path, show_dialogs=True)
    
    def _load_hive_checkpoint_from_path(self, file_path, show_dialogs=False):
        """Load a YOLO hive checkpoint from a file path"""
        try:
            # Import ultralytics YOLO
            from ultralytics import YOLO
            
            # Load the model
            model = YOLO(file_path)
            
            # Verify it's a segmentation model
            if not hasattr(model, 'predictor') or 'segment' not in str(model.task):
                raise ValueError("Model is not a segmentation model. Please load a YOLO segmentation model (trained with -seg variant).")
            
            # Store model
            self.hive_model = model
            self.hive_model_path = Path(file_path)
            
            # Update UI
            model_name = self.hive_model_path.name
            self.hive_status_label.setText(f"✓ {model_name}")
            self.hive_status_label.setStyleSheet("color: green;")
            self.hive_inference_btn.setEnabled(True)
            
            # Enable "Run Both" button if both models are loaded
            self._update_both_button()
            
            if show_dialogs:
                QMessageBox.information(
                    self,
                    "Hive Model Loaded",
                    f"Successfully loaded YOLO hive model:\n{model_name}\n\n"
                    f"Task: {model.task}\n"
                    f"Ready for inference!"
                )
            else:
                print(f"Hive YOLO loaded successfully: {model_name} (Task: {model.task})")
                
        except Exception as e:
            if show_dialogs:
                QMessageBox.critical(
                    self,
                    "Load Failed",
                    f"Failed to load YOLO hive model:\n{str(e)}"
                )
            else:
                print(f"Failed to load hive model: {str(e)}")
            
    def load_chamber_checkpoint(self):
        """Load a YOLO chamber checkpoint file via file dialog"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select YOLO Chamber Checkpoint",
            str(Path.home()),
            "PyTorch Model Files (*.pt);;All Files (*)"
        )
        
        if not file_path:
            return
        
        self._load_chamber_checkpoint_from_path(file_path, show_dialogs=True)
    
    def _load_chamber_checkpoint_from_path(self, file_path, show_dialogs=False):
        """Load a YOLO chamber checkpoint from a file path"""
        try:
            # Import ultralytics YOLO
            from ultralytics import YOLO
            
            # Load the model
            model = YOLO(file_path)
            
            # Verify it's a segmentation model
            if not hasattr(model, 'predictor') or 'segment' not in str(model.task):
                raise ValueError("Model is not a segmentation model. Please load a YOLO segmentation model (trained with -seg variant).")
            
            # Store model
            self.chamber_model = model
            self.chamber_model_path = Path(file_path)
            
            # Update UI
            model_name = self.chamber_model_path.name
            self.chamber_status_label.setText(f"✓ {model_name}")
            self.chamber_status_label.setStyleSheet("color: green;")
            self.chamber_inference_btn.setEnabled(True)
            
            # Enable "Run Both" button if both models are loaded
            self._update_both_button()
            
            if show_dialogs:
                QMessageBox.information(
                    self,
                    "Chamber Model Loaded",
                    f"Successfully loaded YOLO chamber model:\n{model_name}\n\n"
                    f"Task: {model.task}\n"
                    f"Ready for inference!"
                )
            else:
                print(f"Chamber YOLO loaded successfully: {model_name} (Task: {model.task})")
                
        except Exception as e:
            if show_dialogs:
                QMessageBox.critical(
                    self,
                    "Load Failed",
                    f"Failed to load YOLO chamber model:\n{str(e)}"
                )
            else:
                print(f"Failed to load chamber model: {str(e)}")
    
    def load_pollen_checkpoint(self):
        """Load a YOLO pollen checkpoint file via file dialog"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select YOLO Pollen Checkpoint",
            str(Path.home()),
            "PyTorch Model Files (*.pt);;All Files (*)"
        )
        
        if not file_path:
            return
        
        self._load_pollen_checkpoint_from_path(file_path, show_dialogs=True)
    
    def _load_pollen_checkpoint_from_path(self, file_path, show_dialogs=False):
        """Load a YOLO pollen checkpoint from a file path"""
        try:
            # Import ultralytics YOLO
            from ultralytics import YOLO
            
            # Load the model
            model = YOLO(file_path)
            
            # Verify it's a segmentation model
            if not hasattr(model, 'predictor') or 'segment' not in str(model.task):
                raise ValueError("Model is not a segmentation model. Please load a YOLO segmentation model (trained with -seg variant).")
            
            # Store model
            self.pollen_model = model
            self.pollen_model_path = Path(file_path)
            
            # Update UI
            model_name = self.pollen_model_path.name
            self.pollen_status_label.setText(f"✓ {model_name}")
            self.pollen_status_label.setStyleSheet("color: green;")
            self.pollen_inference_btn.setEnabled(True)
            
            # Enable "Run All" button if any models are loaded
            self._update_both_button()
            
            if show_dialogs:
                QMessageBox.information(
                    self,
                    "Pollen Model Loaded",
                    f"Successfully loaded YOLO pollen model:\n{model_name}\n\n"
                    f"Task: {model.task}\n"
                    f"Ready for inference!"
                )
            else:
                print(f"Pollen YOLO loaded successfully: {model_name} (Task: {model.task})")
                
        except Exception as e:
            if show_dialogs:
                QMessageBox.critical(
                    self,
                    "Load Failed",
                    f"Failed to load YOLO pollen model:\n{str(e)}"
                )
            else:
                print(f"Failed to load pollen model: {str(e)}")
    
    def _update_both_button(self):
        """Update the 'Run All' button based on whether any models are loaded"""
        any_loaded = (self.hive_model is not None or 
                     self.chamber_model is not None or 
                     self.pollen_model is not None)
        self.both_inference_btn.setEnabled(any_loaded)
        
    def on_hive_inference_requested(self):
        """Handle hive inference button click"""
        self.hive_inference_requested.emit()
        
    def on_chamber_inference_requested(self):
        """Handle chamber inference button click"""
        self.chamber_inference_requested.emit()
    
    def on_pollen_inference_requested(self):
        """Handle pollen inference button click"""
        self.pollen_inference_requested.emit()
        
    def on_both_inference_requested(self):
        """Handle run all button click"""
        self.both_inference_requested.emit()
    
    def get_hive_model(self):
        """Get the loaded hive model"""
        return self.hive_model
    
    def get_chamber_model(self):
        """Get the loaded chamber model"""
        return self.chamber_model
    
    def get_pollen_model(self):
        """Get the loaded pollen model"""
        return self.pollen_model
    
    def is_hive_model_loaded(self):
        """Check if hive model is loaded"""
        return self.hive_model is not None
    
    def is_chamber_model_loaded(self):
        """Check if chamber model is loaded"""
        return self.chamber_model is not None
    
    def is_pollen_model_loaded(self):
        """Check if pollen model is loaded"""
        return self.pollen_model is not None
