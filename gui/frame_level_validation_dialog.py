"""
Frame-level validation dialog for analyzing predictions vs ground truth
"""

from PyQt6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel, 
                             QPushButton, QGroupBox, QFormLayout, 
                             QDoubleSpinBox, QLineEdit, QFileDialog,
                             QProgressBar, QTextEdit, QCheckBox)
from PyQt6.QtCore import Qt, pyqtSlot
from PyQt6.QtGui import QFont
from pathlib import Path


class FrameLevelValidationConfigDialog(QDialog):
    """Dialog for configuring frame-level validation analysis"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Frame-Level Validation Analysis")
        self.setModal(True)
        self.setMinimumWidth(600)
        
        self.config = None
        
        self.init_ui()
        
    def init_ui(self):
        """Initialize UI"""
        layout = QVBoxLayout(self)
        
        # Description
        desc_label = QLabel(
            "Analyze predictions vs ground truth for all validation frames.\n\n"
            "This will:\n"
            "• Run inference on all validation frames with bee annotations\n"
            "• Match predictions to ground truth bounding boxes\n"
            "• Optionally analyze hive and chamber predictions\n"
            "• Generate detailed CSV reports\n"
        )
        desc_label.setWordWrap(True)
        layout.addWidget(desc_label)
        
        # Model selection group
        model_group = QGroupBox("Required Models")
        model_layout = QFormLayout()
        
        # Bee BBox model (required)
        bbox_layout = QHBoxLayout()
        self.bbox_model_edit = QLineEdit()
        self.bbox_model_edit.setPlaceholderText("Select YOLO bounding box detection model...")
        self.bbox_model_edit.setReadOnly(True)
        bbox_layout.addWidget(self.bbox_model_edit)
        
        self.bbox_browse_btn = QPushButton("Browse...")
        self.bbox_browse_btn.clicked.connect(self.browse_bbox_model)
        bbox_layout.addWidget(self.bbox_browse_btn)
        
        model_layout.addRow("Bee BBox Model:", bbox_layout)
        
        model_group.setLayout(model_layout)
        layout.addWidget(model_group)
        
        # Optional models group
        optional_group = QGroupBox("Optional Models")
        optional_layout = QFormLayout()
        
        # Hive model (optional)
        hive_layout = QHBoxLayout()
        self.hive_model_edit = QLineEdit()
        self.hive_model_edit.setPlaceholderText("Optional: Select YOLO hive segmentation model...")
        self.hive_model_edit.setReadOnly(True)
        hive_layout.addWidget(self.hive_model_edit)
        
        self.hive_browse_btn = QPushButton("Browse...")
        self.hive_browse_btn.clicked.connect(self.browse_hive_model)
        hive_layout.addWidget(self.hive_browse_btn)
        
        self.hive_clear_btn = QPushButton("Clear")
        self.hive_clear_btn.clicked.connect(lambda: self.hive_model_edit.clear())
        hive_layout.addWidget(self.hive_clear_btn)
        
        optional_layout.addRow("Hive Model:", hive_layout)
        
        # Chamber model (optional)
        chamber_layout = QHBoxLayout()
        self.chamber_model_edit = QLineEdit()
        self.chamber_model_edit.setPlaceholderText("Optional: Select YOLO chamber segmentation model...")
        self.chamber_model_edit.setReadOnly(True)
        chamber_layout.addWidget(self.chamber_model_edit)
        
        self.chamber_browse_btn = QPushButton("Browse...")
        self.chamber_browse_btn.clicked.connect(self.browse_chamber_model)
        chamber_layout.addWidget(self.chamber_browse_btn)
        
        self.chamber_clear_btn = QPushButton("Clear")
        self.chamber_clear_btn.clicked.connect(lambda: self.chamber_model_edit.clear())
        chamber_layout.addWidget(self.chamber_clear_btn)
        
        optional_layout.addRow("Chamber Model:", chamber_layout)
        
        optional_group.setLayout(optional_layout)
        layout.addWidget(optional_group)
        
        # Parameters group
        params_group = QGroupBox("Parameters")
        params_layout = QFormLayout()
        
        # Confidence threshold
        self.conf_threshold_spin = QDoubleSpinBox()
        self.conf_threshold_spin.setRange(0.01, 0.99)
        self.conf_threshold_spin.setValue(0.5)
        self.conf_threshold_spin.setSingleStep(0.05)
        self.conf_threshold_spin.setDecimals(2)
        self.conf_threshold_spin.setToolTip("Minimum confidence for accepting predictions")
        params_layout.addRow("Confidence Threshold:", self.conf_threshold_spin)
        
        # IoU threshold for matching
        self.iou_threshold_spin = QDoubleSpinBox()
        self.iou_threshold_spin.setRange(0.1, 0.95)
        self.iou_threshold_spin.setValue(0.5)
        self.iou_threshold_spin.setSingleStep(0.05)
        self.iou_threshold_spin.setDecimals(2)
        self.iou_threshold_spin.setToolTip("IoU threshold for matching predictions to ground truth")
        params_layout.addRow("Matching IoU Threshold:", self.iou_threshold_spin)
        
        params_group.setLayout(params_layout)
        layout.addWidget(params_group)
        
        # Output options group
        output_group = QGroupBox("Output Options")
        output_layout = QVBoxLayout()
        
        self.debug_mode_check = QCheckBox("Debug mode (process first frame only)")
        self.debug_mode_check.setChecked(False)
        self.debug_mode_check.setToolTip("Only process the first validation frame to quickly check output format")
        output_layout.addWidget(self.debug_mode_check)
        
        self.save_visualizations_check = QCheckBox("Save visualization images for all frames")
        self.save_visualizations_check.setChecked(True)
        self.save_visualizations_check.setToolTip("Save images with color-coded bounding boxes (Green=Matched, Red=False Negative, Pink=False Positive) and hive/chamber outlines (Yellow=Hive, Blue=Chambers)")
        output_layout.addWidget(self.save_visualizations_check)
        
        output_group.setLayout(output_layout)
        layout.addWidget(output_group)
        
        # Output location info
        info_label = QLabel(
            "Results will be saved to:\n"
            "<project_path>/frame_level_validation/<timestamp>/"
        )
        info_label.setStyleSheet("color: gray; font-style: italic;")
        info_label.setWordWrap(True)
        layout.addWidget(info_label)
        
        # Buttons
        button_layout = QHBoxLayout()
        
        self.run_btn = QPushButton("Start Analysis")
        self.run_btn.clicked.connect(self.validate_and_accept)
        self.run_btn.setDefault(True)
        button_layout.addWidget(self.run_btn)
        
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.clicked.connect(self.reject)
        button_layout.addWidget(self.cancel_btn)
        
        layout.addLayout(button_layout)
        
    def browse_bbox_model(self):
        """Browse for YOLO bbox detection model"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select YOLO BBox Detection Model",
            str(Path.home()),
            "PyTorch Model Files (*.pt);;All Files (*)"
        )
        
        if file_path:
            self.bbox_model_edit.setText(file_path)
    
    def browse_hive_model(self):
        """Browse for YOLO hive segmentation model"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select YOLO Hive Segmentation Model",
            str(Path.home()),
            "PyTorch Model Files (*.pt);;All Files (*)"
        )
        
        if file_path:
            self.hive_model_edit.setText(file_path)
    
    def browse_chamber_model(self):
        """Browse for YOLO chamber segmentation model"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select YOLO Chamber Segmentation Model",
            str(Path.home()),
            "PyTorch Model Files (*.pt);;All Files (*)"
        )
        
        if file_path:
            self.chamber_model_edit.setText(file_path)
    
    def validate_and_accept(self):
        """Validate inputs before accepting"""
        # Check that bbox model is provided
        if not self.bbox_model_edit.text():
            from PyQt6.QtWidgets import QMessageBox
            QMessageBox.warning(
                self,
                "Missing Required Model",
                "Please select a YOLO bounding box detection model.\n\n"
                "This is required for analyzing bee predictions."
            )
            return
        
        # Check that bbox model file exists
        bbox_path = Path(self.bbox_model_edit.text())
        if not bbox_path.exists():
            from PyQt6.QtWidgets import QMessageBox
            QMessageBox.warning(
                self,
                "Model Not Found",
                f"BBox model file not found:\n{bbox_path}"
            )
            return
        
        # Check optional models if provided
        hive_path = self.hive_model_edit.text()
        if hive_path and not Path(hive_path).exists():
            from PyQt6.QtWidgets import QMessageBox
            QMessageBox.warning(
                self,
                "Model Not Found",
                f"Hive model file not found:\n{hive_path}"
            )
            return
        
        chamber_path = self.chamber_model_edit.text()
        if chamber_path and not Path(chamber_path).exists():
            from PyQt6.QtWidgets import QMessageBox
            QMessageBox.warning(
                self,
                "Model Not Found",
                f"Chamber model file not found:\n{chamber_path}"
            )
            return
        
        # All validation passed
        self.accept()
    
    def get_config(self):
        """Get validation configuration"""
        return {
            'bbox_model_path': self.bbox_model_edit.text(),
            'hive_model_path': self.hive_model_edit.text() or None,
            'chamber_model_path': self.chamber_model_edit.text() or None,
            'conf_threshold': self.conf_threshold_spin.value(),
            'iou_threshold': self.iou_threshold_spin.value(),
            'save_visualizations': self.save_visualizations_check.isChecked(),
            'debug_mode': self.debug_mode_check.isChecked()
        }


class FrameLevelValidationProgressDialog(QDialog):
    """Dialog showing frame-level validation progress"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Frame-Level Validation Analysis")
        self.setModal(True)
        self.setMinimumSize(700, 500)
        
        self.init_ui()
        
    def init_ui(self):
        """Initialize UI"""
        layout = QVBoxLayout(self)
        
        # Status label
        self.status_label = QLabel("Initializing...")
        self.status_label.setStyleSheet("font-weight: bold; font-size: 12pt;")
        layout.addWidget(self.status_label)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setMinimum(0)
        self.progress_bar.setMaximum(100)
        layout.addWidget(self.progress_bar)
        
        # Time remaining label
        self.time_remaining_label = QLabel("Estimating time remaining...")
        self.time_remaining_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.time_remaining_label.setStyleSheet("color: #666; font-style: italic;")
        layout.addWidget(self.time_remaining_label)
        
        # Stats group
        stats_group = QGroupBox("Statistics")
        stats_layout = QFormLayout()
        
        self.videos_label = QLabel("0 / 0")
        stats_layout.addRow("Videos:", self.videos_label)
        
        self.frames_label = QLabel("0 / 0")
        stats_layout.addRow("Frames:", self.frames_label)
        
        self.matches_label = QLabel("0")
        stats_layout.addRow("Matched Bees:", self.matches_label)
        
        self.fp_label = QLabel("0")
        stats_layout.addRow("False Positives:", self.fp_label)
        
        self.fn_label = QLabel("0")
        stats_layout.addRow("False Negatives:", self.fn_label)
        
        stats_group.setLayout(stats_layout)
        layout.addWidget(stats_group)
        
        # Log area
        log_label = QLabel("Progress Log:")
        layout.addWidget(log_label)
        
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMaximumHeight(200)
        font = QFont("Monospace")
        font.setStyleHint(QFont.StyleHint.TypeWriter)
        self.log_text.setFont(font)
        layout.addWidget(self.log_text)
        
        # Buttons
        button_layout = QHBoxLayout()
        
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.clicked.connect(self.on_cancel)
        button_layout.addWidget(self.cancel_btn)
        
        self.close_btn = QPushButton("Close")
        self.close_btn.setEnabled(False)  # Enabled when complete
        self.close_btn.clicked.connect(self.accept)
        button_layout.addWidget(self.close_btn)
        
        layout.addLayout(button_layout)
        
        self.cancelled = False
        
    def on_cancel(self):
        """Handle cancel button"""
        self.cancelled = True
        self.cancel_btn.setEnabled(False)
        self.log("Cancelling...")
    
    @pyqtSlot(str)
    def update_status(self, status: str):
        """Update status label"""
        self.status_label.setText(status)
    
    @pyqtSlot(int, int, int, int)
    def update_stats(self, videos_done: int, total_videos: int, 
                    frames_done: int, total_frames: int):
        """Update statistics"""
        self.videos_label.setText(f"{videos_done} / {total_videos}")
        self.frames_label.setText(f"{frames_done} / {total_frames}")
        
        # Update progress bar
        if total_frames > 0:
            progress = int(100 * frames_done / total_frames)
            self.progress_bar.setValue(progress)
    
    @pyqtSlot(str)
    def update_time_remaining(self, time_str: str):
        """Update time remaining estimate"""
        self.time_remaining_label.setText(f"Estimated time remaining: {time_str}")
    
    @pyqtSlot(int, int, int)
    def update_match_stats(self, matches: int, fps: int, fns: int):
        """Update matching statistics"""
        self.matches_label.setText(str(matches))
        self.fp_label.setText(str(fps))
        self.fn_label.setText(str(fns))
    
    @pyqtSlot(str)
    def log(self, message: str):
        """Add message to log"""
        self.log_text.append(message)
        # Auto-scroll to bottom
        scrollbar = self.log_text.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())
    
    @pyqtSlot(str)
    def on_complete(self, results_path: str):
        """Handle completion"""
        self.status_label.setText("✓ Analysis Complete")
        self.status_label.setStyleSheet("font-weight: bold; font-size: 12pt; color: green;")
        self.progress_bar.setValue(100)
        self.time_remaining_label.setText("Complete!")
        self.cancel_btn.setEnabled(False)
        self.close_btn.setEnabled(True)
        self.log(f"\n=== Analysis Complete ===")
        self.log(f"Results saved to: {results_path}")
    
    @pyqtSlot(str)
    def on_error(self, error_msg: str):
        """Handle error"""
        self.status_label.setText("✗ Analysis Failed")
        self.status_label.setStyleSheet("font-weight: bold; font-size: 12pt; color: red;")
        self.cancel_btn.setEnabled(False)
        self.close_btn.setEnabled(True)
        self.log(f"\n=== ERROR ===")
        self.log(error_msg)
