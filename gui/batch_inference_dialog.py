"""
Batch inference dialog for running inference on arbitrary PNG images
"""

from PyQt6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel, 
                             QPushButton, QGroupBox, QFormLayout, 
                             QDoubleSpinBox, QLineEdit, QFileDialog,
                             QProgressBar, QTextEdit, QCheckBox, QMessageBox,
                             QRadioButton, QButtonGroup, QComboBox)
from PyQt6.QtCore import Qt, pyqtSlot
from PyQt6.QtGui import QFont
from pathlib import Path


class BatchInferenceConfigDialog(QDialog):
    """Dialog for configuring batch image inference"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Batch Image Inference")
        self.setModal(True)
        self.setMinimumWidth(600)
        
        self.config = None
        
        self.init_ui()
        
    def init_ui(self):
        """Initialize UI"""
        layout = QVBoxLayout(self)
        
        # Description
        desc_label = QLabel(
            "Run inference on PNG images from a folder.\n\n"
            "This will:\n"
            "• Discover all PNG images recursively in the selected folder\n"
            "• Run bee detection, and optionally hive/chamber segmentation\n"
            "• Generate detailed CSV reports with predictions\n"
            "• Optionally export annotations and visualizations\n"
        )
        desc_label.setWordWrap(True)
        layout.addWidget(desc_label)
        
        # Input folder selection
        folder_group = QGroupBox("Input Folder")
        folder_layout = QHBoxLayout()
        
        self.input_folder_edit = QLineEdit()
        self.input_folder_edit.setPlaceholderText("Select folder containing PNG images...")
        self.input_folder_edit.setReadOnly(True)
        folder_layout.addWidget(self.input_folder_edit)
        
        self.folder_browse_btn = QPushButton("Browse...")
        self.folder_browse_btn.clicked.connect(self.browse_input_folder)
        folder_layout.addWidget(self.folder_browse_btn)
        
        folder_group.setLayout(folder_layout)
        layout.addWidget(folder_group)
        
        # Model selection group
        model_group = QGroupBox("Bee Detection Model (Required)")
        model_layout = QVBoxLayout()
        
        # Model type selection
        type_label = QLabel("Detection Type:")
        type_label.setStyleSheet("font-weight: bold;")
        model_layout.addWidget(type_label)
        
        type_radio_layout = QHBoxLayout()
        self.bbox_radio = QRadioButton("Bounding Box")
        self.seg_radio = QRadioButton("Segmentation")
        self.bbox_radio.setChecked(True)
        self.bbox_radio.setToolTip("Use bounding box detection model")
        self.seg_radio.setToolTip("Use segmentation model (includes masks, polygons, centroids)")
        
        self.model_type_group = QButtonGroup()
        self.model_type_group.addButton(self.bbox_radio)
        self.model_type_group.addButton(self.seg_radio)
        
        type_radio_layout.addWidget(self.bbox_radio)
        type_radio_layout.addWidget(self.seg_radio)
        type_radio_layout.addStretch()
        model_layout.addLayout(type_radio_layout)
        
        # Connect radio buttons to update UI
        self.bbox_radio.toggled.connect(self.update_distance_method_visibility)
        self.seg_radio.toggled.connect(self.update_distance_method_visibility)
        
        # Bee model path
        bee_model_label = QLabel("Model File:")
        bee_model_label.setStyleSheet("font-weight: bold; margin-top: 10px;")
        model_layout.addWidget(bee_model_label)
        
        bbox_layout = QHBoxLayout()
        self.bbox_model_edit = QLineEdit()
        self.bbox_model_edit.setPlaceholderText("Select YOLO model file...")
        self.bbox_model_edit.setReadOnly(True)
        bbox_layout.addWidget(self.bbox_model_edit)
        
        self.bbox_browse_btn = QPushButton("Browse...")
        self.bbox_browse_btn.clicked.connect(self.browse_bbox_model)
        bbox_layout.addWidget(self.bbox_browse_btn)
        
        model_layout.addLayout(bbox_layout)
        
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
        
        # Distance calculation method (for segmentation models only)
        self.distance_method_label = QLabel("Distance Method:")
        self.distance_method_combo = QComboBox()
        self.distance_method_combo.addItems([
            "contour (Fast & accurate)",
            "bbox_filter (Fastest, recommended)",
            "downsample (Very fast, approximate)",
            "full (Slowest, exact)"
        ])
        self.distance_method_combo.setCurrentIndex(0)  # Default to contour
        self.distance_method_combo.setToolTip(
            "Method for calculating distances between segmentation masks:\n"
            "• contour: Edge pixels only (fast & accurate)\n"
            "• bbox_filter: Smart filtering + contours (best balance)\n"
            "• downsample: Reduced resolution (very fast)\n"
            "• full: All pixels (slow but exact)"
        )
        params_layout.addRow(self.distance_method_label, self.distance_method_combo)
        
        # Initially hide distance method (bbox is default)
        self.distance_method_label.hide()
        self.distance_method_combo.hide()
        
        params_group.setLayout(params_layout)
        layout.addWidget(params_group)
        
        # Output options group
        output_group = QGroupBox("Output Options")
        output_layout = QVBoxLayout()
        
        self.save_annotations_check = QCheckBox("Save annotations (PNG+JSON format)")
        self.save_annotations_check.setChecked(False)
        self.save_annotations_check.setToolTip("Export predictions as PNG+JSON annotations, preserving folder structure")
        output_layout.addWidget(self.save_annotations_check)
        
        self.save_visualizations_check = QCheckBox("Save visualization images")
        self.save_visualizations_check.setChecked(True)
        self.save_visualizations_check.setToolTip("Save images with color-coded bounding boxes and hive/chamber outlines")
        output_layout.addWidget(self.save_visualizations_check)
        
        self.debug_mode_check = QCheckBox("Debug mode (process first image only)")
        self.debug_mode_check.setChecked(False)
        self.debug_mode_check.setToolTip("Only process the first image to quickly check output format")
        output_layout.addWidget(self.debug_mode_check)
        
        output_group.setLayout(output_layout)
        layout.addWidget(output_group)
        
        # Output location info
        info_label = QLabel(
            "Results will be saved to:\n"
            "<project_path>/batch_inference/<bbox|segmentation>/<timestamp>/"
        )
        info_label.setStyleSheet("color: gray; font-style: italic;")
        info_label.setWordWrap(True)
        layout.addWidget(info_label)
        
        # Buttons
        button_layout = QHBoxLayout()
        
        self.run_btn = QPushButton("Start Inference")
        self.run_btn.clicked.connect(self.validate_and_accept)
        self.run_btn.setDefault(True)
        button_layout.addWidget(self.run_btn)
        
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.clicked.connect(self.reject)
        button_layout.addWidget(self.cancel_btn)
        
        layout.addLayout(button_layout)
        
    def browse_input_folder(self):
        """Browse for input folder containing images"""
        folder_path = QFileDialog.getExistingDirectory(
            self,
            "Select Folder Containing PNG Images",
            str(Path.home()),
            QFileDialog.Option.ShowDirsOnly
        )
        
        if folder_path:
            self.input_folder_edit.setText(folder_path)
    
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
    
    def update_distance_method_visibility(self):
        """Show/hide distance method controls based on model type selection"""
        is_segmentation = self.seg_radio.isChecked()
        self.distance_method_label.setVisible(is_segmentation)
        self.distance_method_combo.setVisible(is_segmentation)
    
    def validate_and_accept(self):
        """Validate inputs and accept dialog"""
        # Check input folder
        if not self.input_folder_edit.text():
            QMessageBox.warning(self, "Missing Input", "Please select an input folder.")
            return
        
        input_folder = Path(self.input_folder_edit.text())
        if not input_folder.exists() or not input_folder.is_dir():
            QMessageBox.warning(self, "Invalid Folder", "Selected folder does not exist.")
            return
        
        # Check required model
        if not self.bbox_model_edit.text():
            QMessageBox.warning(self, "Missing Model", "Please select a bee detection model (required).")
            return
        
        bbox_model_path = Path(self.bbox_model_edit.text())
        if not bbox_model_path.exists():
            QMessageBox.warning(self, "Invalid Model", "Bee detection model file does not exist.")
            return
        
        # Build config
        # Extract distance method from combo box (format: "method_name (description)")
        distance_method_text = self.distance_method_combo.currentText()
        distance_method = distance_method_text.split(' ')[0]  # Get first word before space
        
        self.config = {
            'input_folder': input_folder,
            'bbox_model': bbox_model_path,
            'bee_model_type': 'segmentation' if self.seg_radio.isChecked() else 'bbox',
            'hive_model': Path(self.hive_model_edit.text()) if self.hive_model_edit.text() else None,
            'chamber_model': Path(self.chamber_model_edit.text()) if self.chamber_model_edit.text() else None,
            'conf_threshold': self.conf_threshold_spin.value(),
            'distance_method': distance_method,
            'save_annotations': self.save_annotations_check.isChecked(),
            'save_visualizations': self.save_visualizations_check.isChecked(),
            'debug_mode': self.debug_mode_check.isChecked()
        }
        
        self.accept()
    
    def get_config(self):
        """Get configuration dictionary"""
        return self.config


class BatchInferenceProgressDialog(QDialog):
    """Dialog showing progress of batch inference"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Batch Inference Progress")
        self.setModal(True)
        self.setMinimumWidth(700)
        self.setMinimumHeight(500)
        
        self.init_ui()
        
    def init_ui(self):
        """Initialize UI"""
        layout = QVBoxLayout(self)
        
        # Status label
        self.status_label = QLabel("Initializing...")
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        layout.addWidget(self.progress_bar)
        
        # Time remaining label
        self.time_label = QLabel("Estimated time remaining: calculating...")
        layout.addWidget(self.time_label)
        
        # Statistics group
        stats_group = QGroupBox("Statistics")
        stats_layout = QFormLayout()
        
        self.images_processed_label = QLabel("0")
        stats_layout.addRow("Images Processed:", self.images_processed_label)
        
        self.total_images_label = QLabel("0")
        stats_layout.addRow("Total Images:", self.total_images_label)
        
        self.bees_detected_label = QLabel("0")
        stats_layout.addRow("Total Bees Detected:", self.bees_detected_label)
        
        self.chambers_detected_label = QLabel("0")
        stats_layout.addRow("Total Chambers Detected:", self.chambers_detected_label)
        
        self.hives_detected_label = QLabel("0")
        stats_layout.addRow("Total Hives Detected:", self.hives_detected_label)
        
        stats_group.setLayout(stats_layout)
        layout.addWidget(stats_group)
        
        # Log output
        log_label = QLabel("Progress Log:")
        layout.addWidget(log_label)
        
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMaximumHeight(200)
        self.log_text.setFont(QFont("Courier", 9))
        layout.addWidget(self.log_text)
        
        # Buttons
        button_layout = QHBoxLayout()
        
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.clicked.connect(self.request_cancel)
        button_layout.addWidget(self.cancel_btn)
        
        self.close_btn = QPushButton("Close")
        self.close_btn.clicked.connect(self.accept)
        self.close_btn.setEnabled(False)
        button_layout.addWidget(self.close_btn)
        
        layout.addLayout(button_layout)
        
        self.cancel_requested = False
        
    def request_cancel(self):
        """Request cancellation"""
        self.cancel_requested = True
        self.cancel_btn.setEnabled(False)
        self.status_label.setText("Cancellation requested... finishing current image...")
        
    @pyqtSlot(str)
    def update_status(self, status: str):
        """Update status label"""
        self.status_label.setText(status)
        
    @pyqtSlot(int, int)
    def update_progress(self, current: int, total: int):
        """Update progress bar"""
        if total > 0:
            progress = int((current / total) * 100)
            self.progress_bar.setValue(progress)
            self.images_processed_label.setText(str(current))
            self.total_images_label.setText(str(total))
    
    @pyqtSlot(str)
    def update_time_remaining(self, time_str: str):
        """Update time remaining label"""
        self.time_label.setText(f"Estimated time remaining: {time_str}")
    
    @pyqtSlot(int, int, int)
    def update_stats(self, bees: int, chambers: int, hives: int):
        """Update detection statistics"""
        self.bees_detected_label.setText(str(bees))
        self.chambers_detected_label.setText(str(chambers))
        self.hives_detected_label.setText(str(hives))
    
    @pyqtSlot(str)
    def append_log(self, message: str):
        """Append message to log"""
        self.log_text.append(message)
        # Auto-scroll to bottom
        scrollbar = self.log_text.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())
    
    @pyqtSlot(bool)
    def on_complete(self, success: bool):
        """Called when analysis is complete"""
        self.cancel_btn.setEnabled(False)
        self.close_btn.setEnabled(True)
        
        if success:
            self.status_label.setText("Analysis complete!")
        else:
            self.status_label.setText("Analysis failed or was cancelled")
        
    def analysis_complete(self):
        """Called when analysis is complete (deprecated, use on_complete)"""
        self.on_complete(True)
        self.close_btn.setEnabled(True)
        self.progress_bar.setValue(100)
        self.time_label.setText("Complete!")
