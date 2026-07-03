"""
Dialog for batch video inference with tracking and ArUco detection
"""

from PyQt6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel, 
                             QPushButton, QGroupBox, QFormLayout, 
                             QDoubleSpinBox, QSpinBox, QLineEdit, QFileDialog,
                             QProgressBar, QTextEdit, QCheckBox, QMessageBox,
                             QComboBox, QRadioButton, QButtonGroup, QScrollArea, QWidget)
from PyQt6.QtCore import Qt, pyqtSlot
from PyQt6.QtGui import QFont
from pathlib import Path
from datetime import datetime


class BatchVideoInferenceConfigDialog(QDialog):
    """Dialog for configuring batch video inference with tracking"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent_window = parent
        self.setWindowTitle("Batch Video Inference with Tracking")
        self.setModal(True)
        self.setMinimumSize(700, 800)
        
        self.config = None
        self.selected_files = []
        
        self.init_ui()
        
    def init_ui(self):
        """Initialize UI"""
        main_layout = QVBoxLayout(self)
        
        # Description
        desc_label = QLabel(
            "<h3>Batch Video Inference with Tracking</h3>"
            "Process videos with bee detection, tracking, ArUco detection, and spatial analysis.<br><br>"
            "<b>Outputs:</b><br>"
            "• bee_detections.csv - Per-frame bee data with spatial metrics<br>"
            "• bee_velocity.csv - Average velocity and frame transitions per bee<br>"
            "• hive_detections.csv - Averaged hive pixels and centroid per chamber (when hive model is provided)<br>"
            "• chamber_detections.csv - Averaged chamber pixels and centroid per chamber<br>"
            "• Optional: Annotated frame images with tracking trails"
        )
        desc_label.setWordWrap(True)
        main_layout.addWidget(desc_label)
        
        # Create scroll area for the rest
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        
        content_widget = QWidget()
        layout = QVBoxLayout(content_widget)
        
        # Video selection group
        video_group = QGroupBox("Video Selection")
        video_layout = QVBoxLayout()
        
        # Radio buttons for folder vs files
        self.folder_radio = QRadioButton("Select folder (process all videos recursively)")
        self.files_radio = QRadioButton("Select individual video files")
        self.folder_radio.setChecked(True)
        
        source_group = QButtonGroup(self)
        source_group.addButton(self.folder_radio)
        source_group.addButton(self.files_radio)
        
        video_layout.addWidget(self.folder_radio)
        video_layout.addWidget(self.files_radio)
        
        # Folder selection
        folder_layout = QHBoxLayout()
        self.input_folder_edit = QLineEdit()
        self.input_folder_edit.setPlaceholderText("Select folder containing videos...")
        self.input_folder_edit.setReadOnly(True)
        folder_layout.addWidget(self.input_folder_edit)
        
        self.folder_browse_btn = QPushButton("Browse...")
        self.folder_browse_btn.clicked.connect(self.browse_input_folder)
        folder_layout.addWidget(self.folder_browse_btn)
        
        video_layout.addLayout(folder_layout)
        
        # File selection
        files_layout = QHBoxLayout()
        self.input_files_edit = QLineEdit()
        self.input_files_edit.setPlaceholderText("Select video files...")
        self.input_files_edit.setReadOnly(True)
        self.input_files_edit.setEnabled(False)
        files_layout.addWidget(self.input_files_edit)
        
        self.files_browse_btn = QPushButton("Browse...")
        self.files_browse_btn.clicked.connect(self.browse_input_files)
        self.files_browse_btn.setEnabled(False)
        files_layout.addWidget(self.files_browse_btn)
        
        video_layout.addLayout(files_layout)
        
        # Connect radio buttons to enable/disable fields
        self.folder_radio.toggled.connect(self._update_selection_controls)
        
        video_group.setLayout(video_layout)
        layout.addWidget(video_group)
        
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
        self.seg_radio.setToolTip("Use segmentation model (includes masks for visualization)")
        
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
        
        bee_layout = QHBoxLayout()
        self.bee_model_edit = QLineEdit()
        self.bee_model_edit.setPlaceholderText("Select YOLO model file...")
        self.bee_model_edit.setReadOnly(True)
        bee_layout.addWidget(self.bee_model_edit)
        
        self.bee_browse_btn = QPushButton("Browse...")
        self.bee_browse_btn.clicked.connect(self.browse_bee_model)
        bee_layout.addWidget(self.bee_browse_btn)
        
        model_layout.addLayout(bee_layout)
        
        model_group.setLayout(model_layout)
        layout.addWidget(model_group)
        
        # Optional models group
        optional_group = QGroupBox("Optional Models")
        optional_layout = QFormLayout()
        
        # Hive model
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
        
        # Chamber model
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
        
        # Tracking configuration group
        tracking_group = QGroupBox("Tracking Algorithm")
        tracking_layout = QFormLayout()
        
        # Algorithm selection
        self.tracking_algo_combo = QComboBox()
        self.tracking_algo_combo.addItems(["ByteTrack", "SimpleIoU", "Centroid"])
        self.tracking_algo_combo.setCurrentText("Centroid")
        self.tracking_algo_combo.currentTextChanged.connect(self._update_tracking_params)
        tracking_layout.addRow("Algorithm:", self.tracking_algo_combo)
        
        # ByteTrack parameters
        self.bytetrack_params = QWidget()
        bt_layout = QFormLayout()
        bt_layout.setContentsMargins(0, 0, 0, 0)
        
        self.bt_high_conf_spin = QDoubleSpinBox()
        self.bt_high_conf_spin.setRange(0.1, 0.99)
        self.bt_high_conf_spin.setValue(0.5)
        self.bt_high_conf_spin.setSingleStep(0.05)
        self.bt_high_conf_spin.setDecimals(2)
        bt_layout.addRow("  High confidence threshold:", self.bt_high_conf_spin)
        
        self.bt_high_iou_spin = QDoubleSpinBox()
        self.bt_high_iou_spin.setRange(0.1, 0.95)
        self.bt_high_iou_spin.setValue(0.6)
        self.bt_high_iou_spin.setSingleStep(0.05)
        self.bt_high_iou_spin.setDecimals(2)
        bt_layout.addRow("  High IoU threshold:", self.bt_high_iou_spin)
        
        self.bt_low_iou_spin = QDoubleSpinBox()
        self.bt_low_iou_spin.setRange(0.05, 0.9)
        self.bt_low_iou_spin.setValue(0.3)
        self.bt_low_iou_spin.setSingleStep(0.05)
        self.bt_low_iou_spin.setDecimals(2)
        bt_layout.addRow("  Low IoU threshold:", self.bt_low_iou_spin)
        
        self.bt_max_lost_spin = QSpinBox()
        self.bt_max_lost_spin.setRange(1, 100)
        self.bt_max_lost_spin.setValue(10)
        bt_layout.addRow("  Max frames lost:", self.bt_max_lost_spin)
        
        self.bt_mask_iou_check = QCheckBox("Use mask IoU")
        self.bt_mask_iou_check.setChecked(True)
        bt_layout.addRow("", self.bt_mask_iou_check)
        
        self.bytetrack_params.setLayout(bt_layout)
        tracking_layout.addRow(self.bytetrack_params)
        
        # SimpleIoU parameters
        self.simpleiou_params = QWidget()
        siou_layout = QFormLayout()
        siou_layout.setContentsMargins(0, 0, 0, 0)
        
        self.siou_threshold_spin = QDoubleSpinBox()
        self.siou_threshold_spin.setRange(0.1, 0.95)
        self.siou_threshold_spin.setValue(0.5)
        self.siou_threshold_spin.setSingleStep(0.05)
        self.siou_threshold_spin.setDecimals(2)
        siou_layout.addRow("  IoU threshold:", self.siou_threshold_spin)
        
        self.siou_mask_iou_check = QCheckBox("Use mask IoU")
        self.siou_mask_iou_check.setChecked(True)
        siou_layout.addRow("", self.siou_mask_iou_check)
        
        self.simpleiou_params.setLayout(siou_layout)
        self.simpleiou_params.hide()
        tracking_layout.addRow(self.simpleiou_params)
        
        # Centroid parameters
        self.centroid_params = QWidget()
        cent_layout = QFormLayout()
        cent_layout.setContentsMargins(0, 0, 0, 0)
        
        self.cent_max_dist_spin = QSpinBox()
        self.cent_max_dist_spin.setRange(10, 2000)
        self.cent_max_dist_spin.setValue(200)
        self.cent_max_dist_spin.setSingleStep(50)
        cent_layout.addRow("  Max distance (pixels):", self.cent_max_dist_spin)
        
        self.cent_max_missing_spin = QSpinBox()
        self.cent_max_missing_spin.setRange(1, 30)
        self.cent_max_missing_spin.setValue(1)
        cent_layout.addRow("  Max frames missing:", self.cent_max_missing_spin)
        
        self.centroid_params.setLayout(cent_layout)
        self.centroid_params.hide()
        tracking_layout.addRow(self.centroid_params)
        
        tracking_group.setLayout(tracking_layout)
        layout.addWidget(tracking_group)
        
        # Detection and tracking parameters group
        detection_group = QGroupBox("Detection & Tracking Parameters")
        detection_layout = QFormLayout()
        
        self.confidence_spin = QDoubleSpinBox()
        self.confidence_spin.setRange(0.01, 0.99)
        self.confidence_spin.setValue(0.5)
        self.confidence_spin.setSingleStep(0.05)
        self.confidence_spin.setDecimals(2)
        detection_layout.addRow("Confidence threshold:", self.confidence_spin)
        
        self.nms_iou_spin = QDoubleSpinBox()
        self.nms_iou_spin.setRange(0.01, 0.95)
        self.nms_iou_spin.setValue(0.45)
        self.nms_iou_spin.setSingleStep(0.05)
        self.nms_iou_spin.setDecimals(2)
        detection_layout.addRow("NMS IoU threshold:", self.nms_iou_spin)

        self.compute_spatial_metrics_check = QCheckBox("Compute spatial metrics")
        self.compute_spatial_metrics_check.setChecked(True)
        self.compute_spatial_metrics_check.setToolTip(
            "Calculate hive distance, number of bees per chamber, and bee-to-bee distance metrics.\n"
            "Disable this for faster tracking-only runs; spatial metric columns will be left blank."
        )
        self.compute_spatial_metrics_check.stateChanged.connect(self.update_distance_method_visibility)
        detection_layout.addRow("", self.compute_spatial_metrics_check)

        # Distance calculation method for bee-to-bee spatial metrics
        self.distance_method_label = QLabel("Bee distance method:")
        self.distance_method_combo = QComboBox()
        self.distance_method_combo.addItems([
            "centroid (Fastest, center-to-center)",
            "contour (Mask edge-to-edge)",
            "bbox_filter (Mask edge-to-edge, filtered)",
            "downsample (Mask approximate)",
            "full (Mask exact, slowest)"
        ])
        self.distance_method_combo.setCurrentIndex(0)  # Default to current fast behavior
        self.distance_method_combo.setToolTip(
            "Method for calculating bee-to-bee spatial distances:\n"
            "• centroid: Center-to-center distance\n"
            "• contour: Mask edge-to-edge distance\n"
            "• bbox_filter: Mask edge-to-edge with filtering\n"
            "• downsample: Approximate mask distance\n"
            "• full: Exact mask distance, slowest\n"
            "Mask methods fall back to centroid distance when masks are unavailable."
        )
        detection_layout.addRow(self.distance_method_label, self.distance_method_combo)
        
        detection_group.setLayout(detection_layout)
        layout.addWidget(detection_group)
        
        # ArUco detection group
        aruco_group = QGroupBox("ArUco Detection")
        aruco_layout = QVBoxLayout()
        
        self.enable_aruco_check = QCheckBox("Enable ArUco marker detection on bees")
        self.enable_aruco_check.setChecked(True)
        self.enable_aruco_check.setToolTip(
            "Detect ArUco/QR markers on individual bees for ID tracking.\n"
            "Note: Chamber ordering is based on YOLO chamber segmentation (left to right)."
        )
        self.enable_aruco_check.stateChanged.connect(self._update_aruco_controls)
        aruco_layout.addWidget(self.enable_aruco_check)

        aruco_form = QFormLayout()

        self.aruco_dictionary_combo = QComboBox()
        self.aruco_dictionary_combo.addItems([
            "Auto 4x4 dictionaries",
            "4x4_50",
            "4x4_100",
            "4x4_250",
            "4x4_1000",
            "5x5_50",
            "5x5_100",
            "5x5_250",
            "5x5_1000",
            "6x6_50",
            "6x6_100",
            "6x6_250",
            "6x6_1000",
            "7x7_50",
            "7x7_100",
            "7x7_250",
            "7x7_1000",
        ])
        self.aruco_dictionary_combo.setCurrentText("Auto 4x4 dictionaries")
        aruco_form.addRow("Dictionary:", self.aruco_dictionary_combo)

        self.aruco_optimize_check = QCheckBox("Optimize parameter bank before each video")
        self.aruco_optimize_check.setChecked(False)
        self.aruco_optimize_check.setToolTip(
            "Sample frames from each video, sweep ArUco detector parameters, and use a compact parameter bank during tracking."
        )
        self.aruco_optimize_check.stateChanged.connect(self._update_aruco_controls)
        aruco_form.addRow("", self.aruco_optimize_check)

        self.aruco_profile_combo = QComboBox()
        self.aruco_profile_combo.addItems(["quick", "balanced", "deep", "daily"])
        self.aruco_profile_combo.setCurrentText("daily")
        aruco_form.addRow("Optimization profile:", self.aruco_profile_combo)

        self.aruco_sample_frames_spin = QSpinBox()
        self.aruco_sample_frames_spin.setRange(1, 500)
        self.aruco_sample_frames_spin.setValue(12)
        aruco_form.addRow("Sample frames/video:", self.aruco_sample_frames_spin)

        self.aruco_max_combinations_spin = QSpinBox()
        self.aruco_max_combinations_spin.setRange(1, 10000)
        self.aruco_max_combinations_spin.setValue(750)
        aruco_form.addRow("Max combinations:", self.aruco_max_combinations_spin)

        self.aruco_bank_size_spin = QSpinBox()
        self.aruco_bank_size_spin.setRange(1, 20)
        self.aruco_bank_size_spin.setValue(5)
        aruco_form.addRow("Parameter bank size:", self.aruco_bank_size_spin)

        self.aruco_expected_tags_spin = QDoubleSpinBox()
        self.aruco_expected_tags_spin.setRange(0, 10000)
        self.aruco_expected_tags_spin.setDecimals(1)
        self.aruco_expected_tags_spin.setValue(0)
        self.aruco_expected_tags_spin.setToolTip("Use 0 to disable expected-count scoring.")
        aruco_form.addRow("Expected visible tags:", self.aruco_expected_tags_spin)

        tag_list_layout = QHBoxLayout()
        self.aruco_tag_list_edit = QLineEdit()
        self.aruco_tag_list_edit.setPlaceholderText("Optional tag allowlist file...")
        self.aruco_tag_list_edit.setReadOnly(True)
        tag_list_layout.addWidget(self.aruco_tag_list_edit)
        self.aruco_tag_list_browse_btn = QPushButton("Browse...")
        self.aruco_tag_list_browse_btn.clicked.connect(self.browse_aruco_tag_list)
        tag_list_layout.addWidget(self.aruco_tag_list_browse_btn)
        self.aruco_tag_list_clear_btn = QPushButton("Clear")
        self.aruco_tag_list_clear_btn.clicked.connect(lambda: self.aruco_tag_list_edit.clear())
        tag_list_layout.addWidget(self.aruco_tag_list_clear_btn)
        aruco_form.addRow("Tag list:", tag_list_layout)

        self.aruco_sweep_min_perimeter_edit = QLineEdit()
        self.aruco_sweep_min_perimeter_edit.setPlaceholderText("e.g. 0.019153")
        aruco_form.addRow("Sweep min perimeter:", self.aruco_sweep_min_perimeter_edit)

        self.aruco_sweep_max_perimeter_edit = QLineEdit()
        self.aruco_sweep_max_perimeter_edit.setPlaceholderText("e.g. 0.052808")
        aruco_form.addRow("Sweep max perimeter:", self.aruco_sweep_max_perimeter_edit)

        self.aruco_sweep_win_min_edit = QLineEdit()
        self.aruco_sweep_win_min_edit.setPlaceholderText("e.g. 3")
        aruco_form.addRow("Sweep thresh win min:", self.aruco_sweep_win_min_edit)

        self.aruco_sweep_win_max_edit = QLineEdit()
        self.aruco_sweep_win_max_edit.setPlaceholderText("e.g. 30,50,70,90,110,130,150")
        aruco_form.addRow("Sweep thresh win max:", self.aruco_sweep_win_max_edit)

        self.aruco_sweep_win_step_edit = QLineEdit()
        self.aruco_sweep_win_step_edit.setPlaceholderText("e.g. 3")
        aruco_form.addRow("Sweep thresh win step:", self.aruco_sweep_win_step_edit)

        self.aruco_sweep_poly_edit = QLineEdit()
        self.aruco_sweep_poly_edit.setPlaceholderText("e.g. 0.08")
        aruco_form.addRow("Sweep polygon approx:", self.aruco_sweep_poly_edit)

        self.aruco_sweep_constant_edit = QLineEdit()
        self.aruco_sweep_constant_edit.setPlaceholderText("e.g. 3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19")
        aruco_form.addRow("Sweep threshold constant:", self.aruco_sweep_constant_edit)

        aruco_layout.addLayout(aruco_form)
        
        aruco_group.setLayout(aruco_layout)
        layout.addWidget(aruco_group)
        
        # Output options group
        output_group = QGroupBox("Output Options")
        output_layout = QVBoxLayout()
        
        # Output folder
        output_folder_layout = QHBoxLayout()
        output_folder_layout.addWidget(QLabel("Output folder:"))
        
        self.output_folder_edit = QLineEdit()
        self.output_folder_edit.setPlaceholderText("Select output folder for CSVs and visualizations...")
        self.output_folder_edit.setReadOnly(True)
        output_folder_layout.addWidget(self.output_folder_edit)
        
        self.output_browse_btn = QPushButton("Browse...")
        self.output_browse_btn.clicked.connect(self.browse_output_folder)
        output_folder_layout.addWidget(self.output_browse_btn)
        
        output_layout.addLayout(output_folder_layout)
        
        self.save_visualizations_check = QCheckBox("Generate annotated frame visualizations")
        self.save_visualizations_check.setChecked(False)
        self.save_visualizations_check.setToolTip(
            "Create annotated frame images with:\n"
            "• Bee bounding boxes with IDs and ArUco codes\n"
            "• Tracking trails\n"
            "• Chamber boundaries\n"
            "• Hive segmentation\n"
            "Saved under output_folder/visualizations/<video_id>/"
        )
        output_layout.addWidget(self.save_visualizations_check)

        self.verbose_output_check = QCheckBox("Verbose output")
        self.verbose_output_check.setChecked(False)
        self.verbose_output_check.setToolTip(
            "Show detailed frame timing, GPU memory, and ArUco diagnostic messages.\n"
            "Leave unchecked for a quieter processing log."
        )
        output_layout.addWidget(self.verbose_output_check)
        
        output_group.setLayout(output_layout)
        layout.addWidget(output_group)
        
        # Set content widget to scroll area
        scroll.setWidget(content_widget)
        main_layout.addWidget(scroll)
        
        # Buttons
        button_layout = QHBoxLayout()
        button_layout.addStretch()
        
        self.run_btn = QPushButton("Run Batch Inference")
        self.run_btn.setDefault(True)
        self.run_btn.clicked.connect(self.accept)
        button_layout.addWidget(self.run_btn)
        
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        button_layout.addWidget(cancel_btn)
        
        main_layout.addLayout(button_layout)
        
        # Set default output folder to project_path/batch_video_inference/{timestamp}
        if self.parent_window and hasattr(self.parent_window, 'project_path') and self.parent_window.project_path:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            default_output = self.parent_window.project_path / 'batch_video_inference' / timestamp
            self.output_folder_edit.setText(str(default_output))
        
        # Initialize UI state to match default selections
        self._update_tracking_params("Centroid")  # Show Centroid params since that's the default
        self.update_distance_method_visibility()  # Update distance method visibility
        self._update_aruco_controls()
    
    def _update_selection_controls(self):
        """Enable/disable folder vs file controls based on radio selection"""
        folder_mode = self.folder_radio.isChecked()
        
        self.input_folder_edit.setEnabled(folder_mode)
        self.folder_browse_btn.setEnabled(folder_mode)
        
        self.input_files_edit.setEnabled(not folder_mode)
        self.files_browse_btn.setEnabled(not folder_mode)
    
    def _update_tracking_params(self, algo_name):
        """Show/hide tracking parameters based on selected algorithm"""
        self.bytetrack_params.setVisible(algo_name == "ByteTrack")
        self.simpleiou_params.setVisible(algo_name == "SimpleIoU")
        self.centroid_params.setVisible(algo_name == "Centroid")
    
    def update_distance_method_visibility(self):
        """Enable the distance method selector when spatial metrics are calculated."""
        compute_spatial_metrics = (
            self.compute_spatial_metrics_check.isChecked()
            if hasattr(self, 'compute_spatial_metrics_check')
            else True
        )
        self.distance_method_label.setEnabled(compute_spatial_metrics)
        self.distance_method_combo.setEnabled(compute_spatial_metrics)

    def _update_aruco_controls(self):
        """Enable/disable ArUco optimization controls."""
        if not hasattr(self, 'enable_aruco_check'):
            return

        aruco_enabled = self.enable_aruco_check.isChecked()
        optimize_enabled = aruco_enabled and self.aruco_optimize_check.isChecked()
        base_controls = [
            self.aruco_dictionary_combo,
            self.aruco_optimize_check,
            self.aruco_tag_list_edit,
            self.aruco_tag_list_browse_btn,
            self.aruco_tag_list_clear_btn,
        ]
        optimize_controls = [
            self.aruco_profile_combo,
            self.aruco_sample_frames_spin,
            self.aruco_max_combinations_spin,
            self.aruco_bank_size_spin,
            self.aruco_expected_tags_spin,
            self.aruco_sweep_min_perimeter_edit,
            self.aruco_sweep_max_perimeter_edit,
            self.aruco_sweep_win_min_edit,
            self.aruco_sweep_win_max_edit,
            self.aruco_sweep_win_step_edit,
            self.aruco_sweep_poly_edit,
            self.aruco_sweep_constant_edit,
        ]

        for control in base_controls:
            control.setEnabled(aruco_enabled)
        for control in optimize_controls:
            control.setEnabled(optimize_enabled)
    
    def browse_input_folder(self):
        """Browse for input folder"""
        folder = QFileDialog.getExistingDirectory(
            self,
            "Select Folder Containing Videos",
            str(Path.home())
        )
        
        if folder:
            self.input_folder_edit.setText(folder)
    
    def browse_input_files(self):
        """Browse for input video files"""
        files, _ = QFileDialog.getOpenFileNames(
            self,
            "Select Video Files",
            str(Path.home()),
            "Video Files (*.mp4 *.avi *.mov *.mkv *.mjpeg *.mjpg);;All Files (*)"
        )
        
        if files:
            self.selected_files = files
            if len(files) == 1:
                self.input_files_edit.setText(files[0])
            else:
                self.input_files_edit.setText(f"{len(files)} files selected")
    
    def browse_bee_model(self):
        """Browse for bee detection model"""
        model_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Bee Detection Model",
            str(Path.home()),
            "YOLO Models (*.pt *.onnx);;All Files (*)"
        )
        
        if model_path:
            self.bee_model_edit.setText(model_path)
    
    def browse_hive_model(self):
        """Browse for hive segmentation model"""
        model_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Hive Segmentation Model",
            str(Path.home()),
            "YOLO Models (*.pt *.onnx);;All Files (*)"
        )
        
        if model_path:
            self.hive_model_edit.setText(model_path)
    
    def browse_chamber_model(self):
        """Browse for chamber segmentation model"""
        model_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Chamber Segmentation Model",
            str(Path.home()),
            "YOLO Models (*.pt *.onnx);;All Files (*)"
        )
        
        if model_path:
            self.chamber_model_edit.setText(model_path)
    
    def browse_output_folder(self):
        """Browse for output folder"""
        folder = QFileDialog.getExistingDirectory(
            self,
            "Select Output Folder",
            str(Path.home())
        )
        
        if folder:
            self.output_folder_edit.setText(folder)

    def browse_aruco_tag_list(self):
        """Browse for optional ArUco tag allowlist"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select ArUco Tag List",
            str(Path.home()),
            "Tag Lists (*.txt *.csv *.json);;All Files (*)"
        )
        if file_path:
            self.aruco_tag_list_edit.setText(file_path)

    def _parse_sweep_values(self, raw_text: str, label: str, value_type: str):
        text = str(raw_text or "").strip()
        if not text:
            return []

        values = []
        for token in text.split(','):
            token = token.strip()
            if not token:
                continue
            try:
                if value_type == "int":
                    value = float(token)
                    if not value.is_integer():
                        raise ValueError
                    values.append(int(value))
                else:
                    values.append(float(token))
            except ValueError:
                raise ValueError(f"{label} contains an invalid {value_type}: {token}")
        return values

    def _collect_aruco_sweep_overrides(self):
        fields = [
            ("minMarkerPerimeterRate", self.aruco_sweep_min_perimeter_edit, "float", "Sweep min perimeter"),
            ("maxMarkerPerimeterRate", self.aruco_sweep_max_perimeter_edit, "float", "Sweep max perimeter"),
            ("adaptiveThreshWinSizeMin", self.aruco_sweep_win_min_edit, "int", "Sweep threshold window min"),
            ("adaptiveThreshWinSizeMax", self.aruco_sweep_win_max_edit, "int", "Sweep threshold window max"),
            ("adaptiveThreshWinSizeStep", self.aruco_sweep_win_step_edit, "int", "Sweep threshold window step"),
            ("polygonalApproxAccuracyRate", self.aruco_sweep_poly_edit, "float", "Sweep polygon approx"),
            ("adaptiveThreshConstant", self.aruco_sweep_constant_edit, "int", "Sweep threshold constant"),
        ]

        overrides = {}
        for key, widget, value_type, label in fields:
            values = self._parse_sweep_values(widget.text(), label, value_type)
            if values:
                overrides[key] = values
        return overrides
    
    def accept(self):
        """Validate and accept dialog"""
        # Validate inputs
        if self.folder_radio.isChecked():
            if not self.input_folder_edit.text():
                QMessageBox.warning(self, "Input Required", "Please select an input folder.")
                return
            video_source = Path(self.input_folder_edit.text())
            if not video_source.exists():
                QMessageBox.warning(self, "Invalid Path", "Input folder does not exist.")
                return
        else:
            if not self.selected_files:
                QMessageBox.warning(self, "Input Required", "Please select video files.")
                return
            video_source = self.selected_files
        
        if not self.bee_model_edit.text():
            QMessageBox.warning(self, "Model Required", "Please select a bee detection model.")
            return
        
        bee_model_path = Path(self.bee_model_edit.text())
        if not bee_model_path.exists():
            QMessageBox.warning(self, "Invalid Path", "Bee model does not exist.")
            return
        
        # Hive model is optional. If omitted, hive-distance metrics are left blank
        # and hive_detections.csv is not produced.
        hive_model_path = None
        if self.hive_model_edit.text():
            hive_model_path = Path(self.hive_model_edit.text())
            if not hive_model_path.exists():
                QMessageBox.warning(self, "Invalid Path", "Hive model does not exist.")
                return
        
        # Chamber model is optional
        chamber_model_path = None
        if self.chamber_model_edit.text():
            chamber_model_path = Path(self.chamber_model_edit.text())
            if not chamber_model_path.exists():
                QMessageBox.warning(self, "Invalid Path", "Chamber model does not exist.")
                return
        
        if not self.output_folder_edit.text():
            QMessageBox.warning(self, "Output Required", "Please select an output folder.")
            return
        
        output_folder = Path(self.output_folder_edit.text())
        if not output_folder.exists():
            output_folder.mkdir(parents=True, exist_ok=True)

        tag_list_path = self.aruco_tag_list_edit.text().strip()
        if tag_list_path and not Path(tag_list_path).exists():
            QMessageBox.warning(self, "Invalid Path", "ArUco tag list file does not exist.")
            return

        aruco_dictionary_text = self.aruco_dictionary_combo.currentText()
        aruco_dictionary_mode = 'auto_4x4' if aruco_dictionary_text.startswith("Auto") else 'single'
        aruco_dictionary = '4x4_100' if aruco_dictionary_mode == 'auto_4x4' else aruco_dictionary_text

        if self.enable_aruco_check.isChecked() and self.aruco_optimize_check.isChecked() and aruco_dictionary_mode == 'auto_4x4':
            QMessageBox.warning(
                self,
                "Dictionary Required",
                "Please choose a specific ArUco dictionary when parameter-bank optimization is enabled."
            )
            return

        try:
            aruco_sweep_overrides = self._collect_aruco_sweep_overrides()
        except ValueError as exc:
            QMessageBox.warning(self, "Invalid ArUco Sweep", str(exc))
            return

        expected_tags = self.aruco_expected_tags_spin.value()
        expected_tags = expected_tags if expected_tags > 0 else None
        
        # Build tracking config based on selected algorithm
        tracking_algo = self.tracking_algo_combo.currentText()
        
        if tracking_algo == "ByteTrack":
            tracking_config = {
                'algorithm': 'bytetrack',
                'high_conf_threshold': self.bt_high_conf_spin.value(),
                'high_iou_threshold': self.bt_high_iou_spin.value(),
                'low_iou_threshold': self.bt_low_iou_spin.value(),
                'max_frames_lost': self.bt_max_lost_spin.value(),
                'use_mask_iou': self.bt_mask_iou_check.isChecked()
            }
        elif tracking_algo == "SimpleIoU":
            tracking_config = {
                'algorithm': 'simple_iou',
                'iou_threshold': self.siou_threshold_spin.value(),
                'use_mask_iou': self.siou_mask_iou_check.isChecked()
            }
        else:  # Centroid
            tracking_config = {
                'algorithm': 'centroid',
                'max_distance': self.cent_max_dist_spin.value(),
                'max_frames_missing': self.cent_max_missing_spin.value()
            }
        
        # Get model type and distance method
        bee_model_type = 'segmentation' if self.seg_radio.isChecked() else 'bbox'
        
        # Parse distance method from combo box text
        distance_method_text = self.distance_method_combo.currentText()
        distance_method = distance_method_text.split(' ')[0]  # Extract first word (e.g., "contour" from "contour (Fast & accurate)")
        
        # Build config
        self.config = {
            'video_source': video_source,
            'folder_mode': self.folder_radio.isChecked(),
            'bee_model_path': str(bee_model_path),
            'bee_model_type': bee_model_type,
            'distance_method': distance_method,
            'hive_model_path': str(hive_model_path) if hive_model_path else None,
            'chamber_model_path': str(chamber_model_path) if chamber_model_path else None,
            'tracking_config': tracking_config,
            'confidence_threshold': self.confidence_spin.value(),
            'nms_iou_threshold': self.nms_iou_spin.value(),
            'compute_spatial_metrics': self.compute_spatial_metrics_check.isChecked(),
            'enable_aruco': self.enable_aruco_check.isChecked(),
            'aruco_dictionary': aruco_dictionary,
            'aruco_dictionary_mode': aruco_dictionary_mode,
            'tag_list_path': tag_list_path or None,
            'allowed_tag_ids': [],
            'aruco_optimization': {
                'enabled': self.enable_aruco_check.isChecked() and self.aruco_optimize_check.isChecked(),
                'dictionary': aruco_dictionary,
                'profile': self.aruco_profile_combo.currentText(),
                'sample_frames': self.aruco_sample_frames_spin.value(),
                'max_combinations': self.aruco_max_combinations_spin.value(),
                'bank_size': self.aruco_bank_size_spin.value(),
                'expected_tags': expected_tags,
                'sweep_overrides': aruco_sweep_overrides,
            },
            'output_folder': str(output_folder),
            'save_visualizations': self.save_visualizations_check.isChecked(),
            'verbose_output': self.verbose_output_check.isChecked()
        }
        
        super().accept()


class BatchVideoInferenceProgressDialog(QDialog):
    """Dialog for showing batch video inference progress"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Batch Video Inference Progress")
        self.setModal(True)
        self.setMinimumSize(700, 500)
        
        self.init_ui()
    
    def init_ui(self):
        """Initialize UI"""
        layout = QVBoxLayout(self)
        
        # Status label
        self.status_label = QLabel("Initializing...")
        self.status_label.setFont(QFont("Arial", 10, QFont.Weight.Bold))
        layout.addWidget(self.status_label)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setMinimum(0)
        self.progress_bar.setMaximum(100)
        self.progress_bar.setValue(0)
        layout.addWidget(self.progress_bar)
        
        # Current video label
        self.current_video_label = QLabel("")
        layout.addWidget(self.current_video_label)
        
        # Log output
        log_label = QLabel("Processing Log:")
        layout.addWidget(log_label)
        
        self.log_output = QTextEdit()
        self.log_output.setReadOnly(True)
        self.log_output.setFont(QFont("Courier", 9))
        layout.addWidget(self.log_output)
        
        # Buttons
        button_layout = QHBoxLayout()
        button_layout.addStretch()
        
        self.stop_btn = QPushButton("Stop")
        self.stop_btn.clicked.connect(self.request_stop)
        button_layout.addWidget(self.stop_btn)
        
        self.close_btn = QPushButton("Close")
        self.close_btn.setEnabled(False)
        self.close_btn.clicked.connect(self.accept)
        button_layout.addWidget(self.close_btn)
        
        layout.addLayout(button_layout)
        
        self.should_stop = False
    
    def request_stop(self):
        """Request processing to stop"""
        self.should_stop = True
        self.stop_btn.setEnabled(False)
        self.status_label.setText("Stopping...")
    
    @pyqtSlot(str)
    def update_status(self, status):
        """Update status label"""
        self.status_label.setText(status)
    
    @pyqtSlot(int, int)
    def update_progress(self, current, total):
        """Update progress bar"""
        if total > 0:
            percentage = int((current / total) * 100)
            self.progress_bar.setValue(percentage)
            self.current_video_label.setText(f"Processing video {current} of {total}")
    
    @pyqtSlot(str)
    def append_log(self, message):
        """Append message to log"""
        self.log_output.append(message)
    
    @pyqtSlot()
    def processing_complete(self):
        """Called when processing is complete"""
        self.stop_btn.setEnabled(False)
        self.close_btn.setEnabled(True)
        self.status_label.setText("✓ Processing complete!")
        self.progress_bar.setValue(100)

    @pyqtSlot()
    def processing_stopped(self):
        """Called when processing is stopped by the user"""
        self.stop_btn.setEnabled(False)
        self.close_btn.setEnabled(True)
        self.status_label.setText("⚠️ Processing stopped")
    
    @pyqtSlot(str)
    def processing_failed(self, error):
        """Called when processing fails"""
        self.stop_btn.setEnabled(False)
        self.close_btn.setEnabled(True)
        self.status_label.setText(f"❌ Processing failed: {error}")
