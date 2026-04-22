"""
Dialog for configuring and running tracking algorithm validation
"""

from PyQt6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel, 
                             QPushButton, QGroupBox, QFormLayout, 
                             QDoubleSpinBox, QSpinBox, QCheckBox, QFileDialog,
                             QLineEdit, QListWidget, QListWidgetItem, QTabWidget,
                             QWidget, QProgressBar, QTextEdit, QSplitter, QScrollArea,
                             QRadioButton, QButtonGroup)
from PyQt6.QtCore import Qt, pyqtSignal, pyqtSlot
from PyQt6.QtGui import QFont
from pathlib import Path
import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


class TrackingValidationConfigDialog(QDialog):
    """Dialog for configuring tracking validation"""
    
    def __init__(self, sequences, parent=None):
        super().__init__(parent)
        self.sequences = sequences  # List of TrackingSequence objects to validate
        self.parent_window = parent
        self.setWindowTitle("Tracking Algorithm Validation")
        self.setModal(True)
        self.setMinimumSize(800, 700)
        
        self.config = None
        self.init_ui()
    
    def init_ui(self):
        """Initialize UI"""
        layout = QVBoxLayout(self)
        
        # Header
        header = QLabel(
            "<h3>Tracking Algorithm Validation</h3>"
            "Compare multiple tracking algorithms on ground truth sequences"
        )
        layout.addWidget(header)
        
        # Create scroll area for the main content
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        
        # Container widget for scrollable content
        content_widget = QWidget()
        content_layout = QVBoxLayout(content_widget)
        
        # Detection source selection
        source_group = QGroupBox("Detection Source")
        source_layout = QVBoxLayout()
        
        self.use_model_radio = QRadioButton("Use Model (YOLO inference)")
        self.use_gt_radio = QRadioButton("Use Ground Truth (perfect detections)")
        self.use_model_radio.setChecked(True)
        self.use_gt_radio.setToolTip(
            "Use ground truth annotations as detections to evaluate tracking in isolation.\n"
            "This removes detection errors as a confounding factor."
        )
        
        source_button_group = QButtonGroup(self)
        source_button_group.addButton(self.use_model_radio)
        source_button_group.addButton(self.use_gt_radio)
        
        source_layout.addWidget(self.use_model_radio)
        source_layout.addWidget(self.use_gt_radio)
        source_group.setLayout(source_layout)
        content_layout.addWidget(source_group)
        
        # Detection model (only shown when use_model_radio is selected)
        self.model_group = QGroupBox("Detection Model")
        model_layout = QHBoxLayout()
        
        self.model_edit = QLineEdit()
        self.model_edit.setPlaceholderText("Select YOLO detection model...")
        self.model_edit.setReadOnly(True)
        model_layout.addWidget(self.model_edit)
        
        browse_btn = QPushButton("Browse...")
        browse_btn.clicked.connect(self.browse_model)
        model_layout.addWidget(browse_btn)
        
        self.model_group.setLayout(model_layout)
        content_layout.addWidget(self.model_group)
        
        # Connect radio buttons to show/hide model controls
        self.use_model_radio.toggled.connect(self._update_model_controls_visibility)
        self.use_gt_radio.toggled.connect(self._update_model_controls_visibility)
        
        # Algorithms to test
        algo_group = QGroupBox("Algorithms to Test")
        algo_layout = QVBoxLayout()
        
        algo_layout.addWidget(QLabel("Select one or more tracking algorithms to compare:"))
        
        # ByteTrack
        self.bytetrack_check = QCheckBox("ByteTrack (current implementation)")
        self.bytetrack_check.setChecked(True)
        algo_layout.addWidget(self.bytetrack_check)
        
        # ByteTrack parameters
        bytetrack_params_widget = QWidget()
        bytetrack_params_layout = QFormLayout()
        bytetrack_params_layout.setContentsMargins(30, 0, 0, 0)
        
        self.bt_high_conf_spin = QDoubleSpinBox()
        self.bt_high_conf_spin.setRange(0.1, 0.99)
        self.bt_high_conf_spin.setValue(0.5)
        self.bt_high_conf_spin.setSingleStep(0.05)
        self.bt_high_conf_spin.setDecimals(2)
        bytetrack_params_layout.addRow("  High confidence threshold:", self.bt_high_conf_spin)
        
        self.bt_high_iou_spin = QDoubleSpinBox()
        self.bt_high_iou_spin.setRange(0.1, 0.95)
        self.bt_high_iou_spin.setValue(0.6)
        self.bt_high_iou_spin.setSingleStep(0.05)
        self.bt_high_iou_spin.setDecimals(2)
        bytetrack_params_layout.addRow("  High IoU threshold:", self.bt_high_iou_spin)
        
        self.bt_low_iou_spin = QDoubleSpinBox()
        self.bt_low_iou_spin.setRange(0.05, 0.9)
        self.bt_low_iou_spin.setValue(0.3)
        self.bt_low_iou_spin.setSingleStep(0.05)
        self.bt_low_iou_spin.setDecimals(2)
        bytetrack_params_layout.addRow("  Low IoU threshold:", self.bt_low_iou_spin)
        
        self.bt_max_lost_spin = QSpinBox()
        self.bt_max_lost_spin.setRange(1, 100)
        self.bt_max_lost_spin.setValue(10)
        bytetrack_params_layout.addRow("  Max frames lost:", self.bt_max_lost_spin)
        
        self.bt_mask_iou_check = QCheckBox("Use mask IoU (vs bbox IoU)")
        self.bt_mask_iou_check.setChecked(True)
        bytetrack_params_layout.addRow("", self.bt_mask_iou_check)
        
        bytetrack_params_widget.setLayout(bytetrack_params_layout)
        algo_layout.addWidget(bytetrack_params_widget)
        
        # Simple IoU tracker (baseline)
        self.simple_iou_check = QCheckBox("Simple IoU Matching (baseline)")
        self.simple_iou_check.setChecked(True)
        self.simple_iou_check.setToolTip("Simple greedy IoU matching without track management")
        algo_layout.addWidget(self.simple_iou_check)
        
        simple_params_widget = QWidget()
        simple_params_layout = QFormLayout()
        simple_params_layout.setContentsMargins(30, 0, 0, 0)
        
        self.simple_iou_spin = QDoubleSpinBox()
        self.simple_iou_spin.setRange(0.1, 0.95)
        self.simple_iou_spin.setValue(0.5)
        self.simple_iou_spin.setSingleStep(0.05)
        self.simple_iou_spin.setDecimals(2)
        simple_params_layout.addRow("  IoU threshold:", self.simple_iou_spin)
        
        simple_params_widget.setLayout(simple_params_layout)
        algo_layout.addWidget(simple_params_widget)
        
        # Centroid tracking
        self.centroid_check = QCheckBox("Centroid Distance Matching")
        self.centroid_check.setChecked(True)
        self.centroid_check.setToolTip("Match based on centroid distance")
        algo_layout.addWidget(self.centroid_check)
        
        centroid_params_widget = QWidget()
        centroid_params_layout = QFormLayout()
        centroid_params_layout.setContentsMargins(30, 0, 0, 0)
        
        self.centroid_max_dist_spin = QSpinBox()
        self.centroid_max_dist_spin.setRange(10, 1000)
        self.centroid_max_dist_spin.setValue(200)
        self.centroid_max_dist_spin.setSingleStep(10)
        centroid_params_layout.addRow("  Max distance (pixels):", self.centroid_max_dist_spin)
        
        self.centroid_max_age_spin = QSpinBox()
        self.centroid_max_age_spin.setRange(1, 30)
        self.centroid_max_age_spin.setValue(1)
        self.centroid_max_age_spin.setSingleStep(1)
        self.centroid_max_age_spin.setToolTip("Remove tracks not seen for this many frames")
        centroid_params_layout.addRow("  Max frames missing:", self.centroid_max_age_spin)
        
        centroid_params_widget.setLayout(centroid_params_layout)
        algo_layout.addWidget(centroid_params_widget)
        
        algo_group.setLayout(algo_layout)
        content_layout.addWidget(algo_group)
        
        # Validation sequences
        seq_group = QGroupBox(f"Validation Sequences ({len(self.sequences)} selected)")
        seq_layout = QVBoxLayout()
        
        # Checkbox for including all validation video sequences
        self.include_val_check = QCheckBox("Include all sequences from validation videos")
        self.include_val_check.setChecked(True)
        self.include_val_check.setToolTip(
            "Automatically include all available tracking sequences from videos in the 'val' split.\n"
            "This ensures comprehensive evaluation across the validation set."
        )
        seq_layout.addWidget(self.include_val_check)
        
        seq_list = QListWidget()
        seq_list.setMaximumHeight(100)
        for seq in self.sequences:
            item_text = f"{seq.video_id}: Frames {seq.start_frame}-{seq.end_frame} ({seq.length} frames)"
            if seq.notes:
                item_text += f" - {seq.notes[:30]}..."
            seq_list.addItem(item_text)
        seq_layout.addWidget(seq_list)
        
        seq_group.setLayout(seq_layout)
        content_layout.addWidget(seq_group)
        
        # Metrics
        metrics_group = QGroupBox("Metrics to Calculate")
        metrics_layout = QVBoxLayout()
        
        self.mota_check = QCheckBox("MOTA (Multiple Object Tracking Accuracy)")
        self.mota_check.setChecked(True)
        self.mota_check.setToolTip("Overall tracking accuracy: 1 - (FP + FN + ID_switches) / GT_instances")
        metrics_layout.addWidget(self.mota_check)
        
        self.motp_check = QCheckBox("MOTP (Multiple Object Tracking Precision)")
        self.motp_check.setChecked(True)
        self.motp_check.setToolTip("Average IoU of correctly matched instances")
        metrics_layout.addWidget(self.motp_check)
        
        self.idf1_check = QCheckBox("IDF1 (ID F1 Score)")
        self.idf1_check.setChecked(True)
        self.idf1_check.setToolTip("Measures long-term tracking consistency")
        metrics_layout.addWidget(self.idf1_check)
        
        self.id_switches_check = QCheckBox("ID Switches")
        self.id_switches_check.setChecked(True)
        self.id_switches_check.setToolTip("Number of times tracked IDs switch to different ground truth IDs")
        metrics_layout.addWidget(self.id_switches_check)
        
        self.fragmentation_check = QCheckBox("Fragmentation")
        self.fragmentation_check.setChecked(True)
        self.fragmentation_check.setToolTip("Number of times a track is fragmented (lost and re-found)")
        metrics_layout.addWidget(self.fragmentation_check)
        
        metrics_group.setLayout(metrics_layout)
        content_layout.addWidget(metrics_group)
        
        # Matching parameters
        match_group = QGroupBox("Ground Truth Matching")
        match_layout = QFormLayout()
        
        self.gt_iou_spin = QDoubleSpinBox()
        self.gt_iou_spin.setRange(0.1, 0.95)
        self.gt_iou_spin.setValue(0.5)
        self.gt_iou_spin.setSingleStep(0.05)
        self.gt_iou_spin.setDecimals(2)
        self.gt_iou_spin.setToolTip("IoU threshold for matching predictions to ground truth")
        match_layout.addRow("IoU Threshold:", self.gt_iou_spin)
        
        # Model inference parameters (only shown when using model)
        self.model_params_label = QLabel("Model Inference Parameters:")
        self.model_params_label.setStyleSheet("font-weight: bold; margin-top: 10px;")
        match_layout.addRow(self.model_params_label)
        
        self.conf_threshold_spin = QDoubleSpinBox()
        self.conf_threshold_spin.setRange(0.01, 0.99)
        self.conf_threshold_spin.setValue(0.5)
        self.conf_threshold_spin.setSingleStep(0.05)
        self.conf_threshold_spin.setDecimals(2)
        self.conf_threshold_spin.setToolTip("Minimum detection confidence to consider")
        match_layout.addRow("Min Confidence:", self.conf_threshold_spin)
        
        self.nms_iou_spin = QDoubleSpinBox()
        self.nms_iou_spin.setRange(0.1, 0.95)
        self.nms_iou_spin.setValue(0.5)
        self.nms_iou_spin.setSingleStep(0.05)
        self.nms_iou_spin.setDecimals(2)
        self.nms_iou_spin.setToolTip("NMS IoU threshold for suppressing overlapping detections during inference")
        match_layout.addRow("NMS IoU Threshold:", self.nms_iou_spin)
        
        match_group.setLayout(match_layout)
        content_layout.addWidget(match_group)
        
        # Output options
        output_group = QGroupBox("Output Options")
        output_layout = QVBoxLayout()
        
        self.save_viz_check = QCheckBox("Save visualization images")
        self.save_viz_check.setChecked(True)
        self.save_viz_check.setToolTip("Save annotated frames showing tracking results")
        output_layout.addWidget(self.save_viz_check)
        
        self.save_csv_check = QCheckBox("Save detailed CSV per algorithm")
        self.save_csv_check.setChecked(True)
        output_layout.addWidget(self.save_csv_check)
        
        output_group.setLayout(output_layout)
        content_layout.addWidget(output_group)
        
        # Add stretch to push content up in scroll area
        content_layout.addStretch()
        
        # Set content widget and add scroll area to main layout
        scroll.setWidget(content_widget)
        layout.addWidget(scroll)
        
        # Buttons
        button_layout = QHBoxLayout()
        
        self.run_btn = QPushButton("▶ Run Validation")
        self.run_btn.clicked.connect(self.validate_and_accept)
        self.run_btn.setDefault(True)
        self.run_btn.setStyleSheet("font-weight: bold; padding: 10px;")
        button_layout.addWidget(self.run_btn)
        
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        button_layout.addWidget(cancel_btn)
        
        layout.addLayout(button_layout)
    
    def browse_model(self):
        """Browse for YOLO model"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select YOLO Model",
            str(Path.home()),
            "PyTorch Model Files (*.pt);;All Files (*)"
        )
        
        if file_path:
            self.model_edit.setText(file_path)
    
    def _update_model_controls_visibility(self):
        """Show or hide model-related controls based on detection source"""
        use_model = self.use_model_radio.isChecked()
        self.model_group.setVisible(use_model)
        self.model_params_label.setVisible(use_model)
        self.conf_threshold_spin.setVisible(use_model)
        self.nms_iou_spin.setVisible(use_model)
        
        # Also update label visibility in form layout
        form_layout = self.conf_threshold_spin.parent().layout()
        if isinstance(form_layout, QFormLayout):
            for i in range(form_layout.rowCount()):
                label_item = form_layout.itemAt(i, QFormLayout.ItemRole.LabelRole)
                if label_item and label_item.widget():
                    label_text = label_item.widget().text()
                    if label_text in ["Min Confidence:", "NMS IoU Threshold:"]:
                        label_item.widget().setVisible(use_model)
    
    def validate_and_accept(self):
        """Validate inputs and accept dialog"""
        # Check model only if using model mode
        if self.use_model_radio.isChecked() and not self.model_edit.text():
            from PyQt6.QtWidgets import QMessageBox
            QMessageBox.warning(self, "Missing Model", "Please select a detection model.")
            return
        
        # Check at least one algorithm selected
        if not any([self.bytetrack_check.isChecked(), 
                   self.simple_iou_check.isChecked(),
                   self.centroid_check.isChecked()]):
            from PyQt6.QtWidgets import QMessageBox
            QMessageBox.warning(self, "No Algorithms", "Please select at least one tracking algorithm.")
            return
        
        # Get sequences to validate
        sequences_to_validate = self.sequences.copy()
        initial_count = len(sequences_to_validate)
        
        # If include_val_check is checked, add all sequences from validation videos
        if self.include_val_check.isChecked() and self.parent_window:
            # Get all sequences from validation videos
            if hasattr(self.parent_window, 'tracking_sequence_manager') and hasattr(self.parent_window, 'project_manager'):
                val_videos = self.parent_window.project_manager.get_videos_by_split('val')
                print(f"Found {len(val_videos)} validation videos: {val_videos}")
                
                # Get all sequences for validation videos
                all_sequences = self.parent_window.tracking_sequence_manager.get_all_sequences()
                print(f"Total sequences in project: {len(all_sequences)}")
                
                added_count = 0
                for seq in all_sequences:
                    if seq.video_id in val_videos and seq.enabled:
                        # Add if not already in the list
                        if not any(s.sequence_id == seq.sequence_id for s in sequences_to_validate):
                            sequences_to_validate.append(seq)
                            added_count += 1
                            print(f"  Added sequence: {seq.sequence_id} from {seq.video_id}")
                
                print(f"Added {added_count} sequences from validation videos (total: {len(sequences_to_validate)})")
            else:
                print("Warning: Could not access tracking_sequence_manager or project_manager")
        
        # Show message if additional sequences were added
        if self.include_val_check.isChecked() and len(sequences_to_validate) > initial_count:
            from PyQt6.QtWidgets import QMessageBox
            QMessageBox.information(
                self,
                "Additional Sequences",
                f"Including {len(sequences_to_validate) - initial_count} additional sequences from validation videos.\\n\\n"
                f"Total sequences to validate: {len(sequences_to_validate)}"
            )
        
        # Build config
        use_ground_truth = self.use_gt_radio.isChecked()
        self.config = {
            'use_ground_truth': use_ground_truth,
            'model_path': self.model_edit.text() if not use_ground_truth else None,
            'sequences': sequences_to_validate,
            'algorithms': {},
            'metrics': {
                'mota': self.mota_check.isChecked(),
                'motp': self.motp_check.isChecked(),
                'idf1': self.idf1_check.isChecked(),
                'id_switches': self.id_switches_check.isChecked(),
                'fragmentation': self.fragmentation_check.isChecked(),
            },
            'gt_iou_threshold': self.gt_iou_spin.value(),
            'min_confidence': self.conf_threshold_spin.value() if not use_ground_truth else None,
            'nms_iou_threshold': self.nms_iou_spin.value() if not use_ground_truth else None,
            'save_visualizations': self.save_viz_check.isChecked(),
            'save_csv': self.save_csv_check.isChecked(),
        }
        
        # Add selected algorithms
        if self.bytetrack_check.isChecked():
            self.config['algorithms']['bytetrack'] = {
                'conf_threshold_high': self.bt_high_conf_spin.value(),
                'iou_threshold_high': self.bt_high_iou_spin.value(),
                'iou_threshold_low': self.bt_low_iou_spin.value(),
                'max_frames_lost': self.bt_max_lost_spin.value(),
                'use_mask_iou': self.bt_mask_iou_check.isChecked(),
            }
        
        if self.simple_iou_check.isChecked():
            self.config['algorithms']['simple_iou'] = {
                'iou_threshold': self.simple_iou_spin.value(),
            }
        
        if self.centroid_check.isChecked():
            self.config['algorithms']['centroid'] = {
                'max_distance': self.centroid_max_dist_spin.value(),
                'max_frames_missing': self.centroid_max_age_spin.value(),
            }
        
        self.accept()


class TrackingValidationProgressDialog(QDialog):
    """Dialog showing tracking validation progress"""
    
    stop_requested = pyqtSignal()
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Tracking Validation Progress")
        self.setModal(True)
        self.setMinimumSize(900, 700)
        
        self.init_ui()
    
    def init_ui(self):
        """Initialize UI"""
        layout = QVBoxLayout(self)
        
        # Status
        self.status_label = QLabel("Initializing...")
        self.status_label.setStyleSheet("font-weight: bold; font-size: 12px;")
        layout.addWidget(self.status_label)
        
        # Progress bar
        self.progress = QProgressBar()
        layout.addWidget(self.progress)
        
        # Splitter for metrics and log
        splitter = QSplitter(Qt.Orientation.Vertical)
        
        # Real-time metrics
        metrics_widget = QWidget()
        metrics_layout = QVBoxLayout(metrics_widget)
        metrics_layout.addWidget(QLabel("<b>Real-Time Metrics</b>"))
        
        self.metrics_label = QLabel("Waiting for results...")
        self.metrics_label.setStyleSheet(
            "padding: 10px; background-color: #f5f5f5; color: #000000; "
            "border-radius: 5px; font-family: monospace;"
        )
        self.metrics_label.setWordWrap(True)
        metrics_layout.addWidget(self.metrics_label)
        
        # Matplotlib canvas for live plot
        self.figure = Figure(figsize=(8, 4))
        self.canvas = FigureCanvas(self.figure)
        metrics_layout.addWidget(self.canvas)
        
        splitter.addWidget(metrics_widget)
        
        # Log
        log_widget = QWidget()
        log_layout = QVBoxLayout(log_widget)
        log_layout.addWidget(QLabel("<b>Validation Log</b>"))
        
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setFontFamily("monospace")
        self.log_text.setStyleSheet("background-color: #2b2b2b; color: #f0f0f0;")
        log_layout.addWidget(self.log_text)
        
        splitter.addWidget(log_widget)
        
        splitter.setSizes([400, 300])
        layout.addWidget(splitter)
        
        # Buttons
        button_layout = QHBoxLayout()
        
        self.stop_btn = QPushButton("⏹ Stop Validation")
        self.stop_btn.clicked.connect(self.stop_requested.emit)
        button_layout.addWidget(self.stop_btn)
        
        self.close_btn = QPushButton("Close")
        self.close_btn.clicked.connect(self.accept)
        self.close_btn.setEnabled(False)
        button_layout.addWidget(self.close_btn)
        
        layout.addLayout(button_layout)
        
        # Initialize plot
        self.init_plot()
    
    def init_plot(self):
        """Initialize metrics plot"""
        self.figure.clear()
        self.ax = self.figure.add_subplot(111)
        self.ax.set_xlabel('Sequence')
        self.ax.set_ylabel('Score')
        self.ax.set_title('Tracking Metrics Comparison')
        self.ax.grid(True, alpha=0.3)
        self.ax.legend()
        self.canvas.draw()
    
    @pyqtSlot(str)
    def update_status(self, status):
        """Update status label"""
        self.status_label.setText(status)
    
    @pyqtSlot(int, int)
    def update_progress(self, current, total):
        """Update progress bar"""
        self.progress.setMaximum(total)
        self.progress.setValue(current)
    
    @pyqtSlot(str)
    def append_log(self, message):
        """Append message to log"""
        self.log_text.append(message)
        # Auto-scroll
        scrollbar = self.log_text.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())
    
    @pyqtSlot(dict)
    def update_metrics(self, metrics_data):
        """Update metrics display and plot"""
        # Format metrics text
        lines = ["<b>Current Results:</b><br>"]
        
        for algo_name, metrics in metrics_data.items():
            lines.append(f"<br><b>{algo_name.upper()}:</b>")
            if 'mota' in metrics:
                lines.append(f"  MOTA: {metrics['mota']:.2%}")
            if 'motp' in metrics:
                lines.append(f"  MOTP: {metrics['motp']:.3f}")
            if 'idf1' in metrics:
                lines.append(f"  IDF1: {metrics['idf1']:.2%}")
            if 'id_switches' in metrics:
                lines.append(f"  ID Switches: {metrics['id_switches']}")
            if 'precision' in metrics:
                lines.append(f"  Precision: {metrics['precision']:.2%}")
            if 'recall' in metrics:
                lines.append(f"  Recall: {metrics['recall']:.2%}")
        
        self.metrics_label.setText("<br>".join(lines))
        
        # Update plot
        self.update_plot(metrics_data)
    
    def update_plot(self, metrics_data):
        """Update metrics comparison plot"""
        self.ax.clear()
        
        algorithms = list(metrics_data.keys())
        if not algorithms:
            return
        
        # Plot MOTA, IDF1, precision, recall
        metric_names = ['mota', 'idf1', 'precision', 'recall']
        metric_labels = ['MOTA', 'IDF1', 'Precision', 'Recall']
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
        
        x = range(len(algorithms))
        width = 0.2
        
        for i, (metric, label, color) in enumerate(zip(metric_names, metric_labels, colors)):
            values = [metrics_data[algo].get(metric, 0) for algo in algorithms]
            offset = (i - len(metric_names)/2) * width + width/2
            self.ax.bar([xi + offset for xi in x], values, width, label=label, color=color)
        
        self.ax.set_xlabel('Algorithm')
        self.ax.set_ylabel('Score')
        self.ax.set_title('Tracking Metrics Comparison')
        self.ax.set_xticks(x)
        self.ax.set_xticklabels(algorithms)
        self.ax.set_ylim(0, 1.0)
        self.ax.grid(True, alpha=0.3, axis='y')
        self.ax.legend()
        
        try:
            self.figure.tight_layout()
        except Exception:
            pass  # Ignore tight_layout warnings
        self.canvas.draw()
    
    @pyqtSlot()
    def validation_finished(self):
        """Mark validation as complete"""
        self.stop_btn.setEnabled(False)
        self.close_btn.setEnabled(True)
        self.status_label.setText("✓ Validation Complete!")
    
    @pyqtSlot(str)
    def validation_error(self, error_msg):
        """Handle validation error"""
        self.stop_btn.setEnabled(False)
        self.close_btn.setEnabled(True)
        self.status_label.setText(f"❌ Error: {error_msg}")
