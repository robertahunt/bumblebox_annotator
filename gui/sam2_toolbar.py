"""
SAM2 toolbar for segmentation and propagation
"""

from PyQt6.QtWidgets import (QWidget, QHBoxLayout, QVBoxLayout, QPushButton, QToolButton,
                             QLabel, QFileDialog, QMessageBox, QButtonGroup)
from PyQt6.QtCore import Qt, pyqtSignal
from pathlib import Path


class SAM2Toolbar(QWidget):
    """Toolbar for SAM2 segmentation and propagation"""
    
    tool_changed = pyqtSignal(str)  # Signal when SAM2 tool is selected
    clear_prompts_requested = pyqtSignal()
    propagate_requested = pyqtSignal()
    propagate_to_selected_requested = pyqtSignal()
    unload_requested = pyqtSignal()  # Signal to unload SAM2 checkpoint
    sam2_loaded = pyqtSignal(object)  # Signal when SAM2 is loaded (emits SAM2Integrator)
    finetune_requested = pyqtSignal()  # Signal when fine-tuning is requested
    run_on_bbox_requested = pyqtSignal()  # Signal to run SAM2 on selected instance's bbox
    
    def __init__(self, parent=None, checkpoint_path=None):
        super().__init__(parent)
        
        self.checkpoint_path = None
        self.config_name = "sam2_hiera_l.yaml"  # Default config
        
        self.init_ui()
        
        # Load checkpoint if provided
        if checkpoint_path:
            self._load_checkpoint_from_path(checkpoint_path)
        
    def init_ui(self):
        """Initialize UI with two rows"""
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(5, 5, 5, 5)
        main_layout.setSpacing(5)
        
        # First row: Model loading and tools
        row1 = QHBoxLayout()
        row1.setSpacing(10)
        
        # SAM2 section label
        row1.addWidget(QLabel("<b>SAM2:</b>"))
        
        # Load checkpoint button
        self.load_checkpoint_btn = QPushButton("Load Checkpoint...")
        self.load_checkpoint_btn.setToolTip("Load SAM2 checkpoint file (.pt)")
        self.load_checkpoint_btn.clicked.connect(self.load_checkpoint)
        row1.addWidget(self.load_checkpoint_btn)
        
        # Unload checkpoint button
        self.unload_checkpoint_btn = QPushButton("Unload")
        self.unload_checkpoint_btn.setToolTip("Unload SAM2 checkpoint to free memory")
        self.unload_checkpoint_btn.clicked.connect(self.unload_checkpoint)
        self.unload_checkpoint_btn.setEnabled(False)
        row1.addWidget(self.unload_checkpoint_btn)
        
        # Checkpoint status label
        self.checkpoint_status_label = QLabel("No model loaded")
        self.checkpoint_status_label.setStyleSheet("color: gray; font-style: italic;")
        row1.addWidget(self.checkpoint_status_label)
        
        # Separator
        row1.addWidget(self.create_separator())
        
        # Tool buttons (only enabled when model is loaded)
        self.button_group = QButtonGroup(self)
        self.button_group.setExclusive(True)  # Exclusive so buttons stay checked
        
        row1.addWidget(QLabel("Tools:"))
        
        self.point_btn = self.create_tool_button("Point", "sam2_prompt")
        self.point_btn.setToolTip("SAM2 point prompting: Left-click for positive, Right-click for negative")
        self.point_btn.setEnabled(False)
        row1.addWidget(self.point_btn)
        
        self.box_btn = self.create_tool_button("Box", "sam2_box")
        self.box_btn.setToolTip("SAM2 box prompting: Click and drag to draw a bounding box")
        self.box_btn.setEnabled(False)
        row1.addWidget(self.box_btn)
        
        row1.addStretch()
        main_layout.addLayout(row1)
        
        # Second row: Actions and refinement
        row2 = QHBoxLayout()
        row2.setSpacing(10)
        
        row2.addWidget(QLabel("Actions:"))
        
        self.clear_prompts_btn = QToolButton()
        self.clear_prompts_btn.setText("Clear Points")
        self.clear_prompts_btn.setToolTip("Clear all SAM2 prompt points for the current instance")
        self.clear_prompts_btn.setEnabled(False)
        self.clear_prompts_btn.clicked.connect(self.on_clear_prompts)
        row2.addWidget(self.clear_prompts_btn)
        
        self.run_on_bbox_btn = QToolButton()
        self.run_on_bbox_btn.setText("Run on Instance Bbox")
        self.run_on_bbox_btn.setToolTip("Run SAM2 on the selected instance's bounding box to refine the mask")
        self.run_on_bbox_btn.setEnabled(False)
        self.run_on_bbox_btn.clicked.connect(self.on_run_on_bbox)
        row2.addWidget(self.run_on_bbox_btn)
        
        row2.addWidget(self.create_separator())
        
        self.propagate_btn = QToolButton()
        self.propagate_btn.setText("Propagate to Next →")
        self.propagate_btn.setToolTip("Propagate current frame masks to next frame using SAM2")
        self.propagate_btn.setEnabled(False)
        self.propagate_btn.clicked.connect(self.on_propagate)
        row2.addWidget(self.propagate_btn)
        
        self.propagate_to_selected_btn = QToolButton()
        self.propagate_to_selected_btn.setText("Propagate to Selected →→")
        self.propagate_to_selected_btn.setToolTip("Propagate through all frames until next training/validation frame")
        self.propagate_to_selected_btn.setEnabled(False)
        self.propagate_to_selected_btn.clicked.connect(self.on_propagate_to_selected)
        row2.addWidget(self.propagate_to_selected_btn)
        
        row2.addWidget(self.create_separator())
        
        # Fine-tune button
        self.finetune_btn = QPushButton("Fine-tune SAM2")
        self.finetune_btn.setToolTip("Fine-tune SAM2 model on your annotated data")
        self.finetune_btn.clicked.connect(self.on_finetune_requested)
        self.finetune_btn.setEnabled(False)
        row2.addWidget(self.finetune_btn)
        
        row2.addStretch()
        main_layout.addLayout(row2)
        
    def create_tool_button(self, text, tool_name):
        """Create a tool button"""
        btn = QToolButton()
        btn.setText(text)
        btn.setCheckable(True)
        btn.setProperty('tool_name', tool_name)
        btn.clicked.connect(lambda checked: self.on_tool_clicked(tool_name, checked))
        self.button_group.addButton(btn)
        return btn
        
    def create_separator(self):
        """Create a vertical separator"""
        from PyQt6.QtWidgets import QFrame
        line = QFrame()
        line.setFrameShape(QFrame.Shape.VLine)
        line.setFrameShadow(QFrame.Shadow.Sunken)
        return line
    
    def on_tool_clicked(self, tool_name, checked):
        """Handle tool button click"""
        if checked:
            self.tool_changed.emit(tool_name)
    
    def load_checkpoint(self):
        """Load a SAM2 checkpoint file via file dialog"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select SAM2 Checkpoint",
            str(Path.home()),
            "PyTorch Model Files (*.pt *.pth);;All Files (*)"
        )
        
        if not file_path:
            return
        
        self._load_checkpoint_from_path(file_path, show_dialogs=True)
    
    def _load_checkpoint_from_path(self, file_path, show_dialogs=False):
        """Load a SAM2 checkpoint from a file path"""
        try:
            # Import SAM2 integrator
            from core.sam2_integrator import SAM2Integrator
            
            # Load the model
            self.checkpoint_status_label.setText("Loading...")
            self.checkpoint_status_label.setStyleSheet("color: orange;")
            
            if show_dialogs:
                QMessageBox.information(
                    self,
                    "Loading SAM2",
                    f"Loading SAM2 model from:\n{Path(file_path).name}\n\n"
                    "This may take a moment..."
                )
            
            from PyQt6.QtWidgets import QApplication
            QApplication.processEvents()
            
            print(f"Loading SAM2 from: {file_path}")
            
            sam2 = SAM2Integrator(file_path, self.config_name)
            
            # Store checkpoint path
            self.checkpoint_path = Path(file_path)
            
            # Update UI
            checkpoint_name = self.checkpoint_path.name
            self.checkpoint_status_label.setText(f"✓ {checkpoint_name}")
            self.checkpoint_status_label.setStyleSheet("color: green;")
            
            # Enable tool buttons and unload button
            self.point_btn.setEnabled(True)
            self.box_btn.setEnabled(True)
            self.clear_prompts_btn.setEnabled(True)
            self.run_on_bbox_btn.setEnabled(True)
            self.propagate_btn.setEnabled(True)
            self.propagate_to_selected_btn.setEnabled(True)
            self.unload_checkpoint_btn.setEnabled(True)
            self.finetune_btn.setEnabled(True)
            
            # Emit signal that SAM2 is loaded
            print(f"SAM2Toolbar: Emitting sam2_loaded signal with sam2={sam2}")
            self.sam2_loaded.emit(sam2)
            
            if show_dialogs:
                QMessageBox.information(
                    self,
                    "SAM2 Loaded",
                    f"Successfully loaded SAM2 model:\n{checkpoint_name}\n\n"
                    f"Ready for segmentation!"
                )
            else:
                print(f"SAM2 loaded successfully: {checkpoint_name}")
            
        except ImportError as e:
            error_msg = f"Could not import SAM2 integration: {str(e)}\nMake sure SAM2 is properly installed."
            if show_dialogs:
                QMessageBox.critical(self, "Import Error", error_msg)
            else:
                print(f"ERROR: {error_msg}")
            self.checkpoint_status_label.setText("Import failed")
            self.checkpoint_status_label.setStyleSheet("color: red;")
        except Exception as e:
            error_msg = f"Failed to load SAM2 checkpoint: {str(e)}"
            if show_dialogs:
                QMessageBox.critical(self, "Error Loading Checkpoint", error_msg)
            else:
                print(f"ERROR: {error_msg}")
            self.checkpoint_path = None
            self.checkpoint_status_label.setText("Failed to load")
            self.checkpoint_status_label.setStyleSheet("color: red;")
    
    def on_clear_prompts(self):
        """Handle clear prompts button"""
        self.clear_prompts_requested.emit()
        
    def on_propagate(self):
        """Handle propagate button"""
        self.propagate_requested.emit()
    
    def on_propagate_to_selected(self):
        """Handle propagate to selected button"""
        self.propagate_to_selected_requested.emit()
    
    def on_finetune_requested(self):
        """Handle fine-tune button"""
        self.finetune_requested.emit()
    
    def on_run_on_bbox(self):
        """Handle run on bbox button"""
        self.run_on_bbox_requested.emit()
    
    def is_checkpoint_loaded(self):
        """Check if a checkpoint is loaded"""
        return self.checkpoint_path is not None
    
    def uncheck_tools(self):
        """Uncheck all SAM2 tool buttons without emitting signals"""
        for btn in self.button_group.buttons():
            btn.blockSignals(True)
            btn.setChecked(False)
            btn.blockSignals(False)
    
    def unload_checkpoint(self):
        """Unload the SAM2 checkpoint to free memory"""
        # Confirm with user
        from PyQt6.QtWidgets import QMessageBox
        reply = QMessageBox.question(
            self,
            "Unload SAM2 Checkpoint",
            "This will unload the SAM2 model from memory to free up resources.\n\n"
            "You can reload it later if needed.\n\n"
            "Continue?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        
        if reply == QMessageBox.StandardButton.Yes:
            # Uncheck all tools
            self.uncheck_tools()
            
            # Emit signal to parent to clear SAM2 instance
            self.unload_requested.emit()
            
            # Reset UI state
            self.checkpoint_path = None
            self.checkpoint_status_label.setText("No model loaded")
            self.checkpoint_status_label.setStyleSheet("color: gray; font-style: italic;")
            
            # Disable tool buttons
            self.point_btn.setEnabled(False)
            self.box_btn.setEnabled(False)
            self.clear_prompts_btn.setEnabled(False)
            self.run_on_bbox_btn.setEnabled(False)
            self.propagate_btn.setEnabled(False)
            self.propagate_to_selected_btn.setEnabled(False)
            self.unload_checkpoint_btn.setEnabled(False)
            self.finetune_btn.setEnabled(False)
            
            QMessageBox.information(
                self,
                "SAM2 Unloaded",
                "SAM2 checkpoint has been unloaded from memory."
            )
