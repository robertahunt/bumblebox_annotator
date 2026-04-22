"""
Validation configuration and progress dialog for BeeHaveSquE pipeline validation
"""

from PyQt6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel, 
                             QPushButton, QProgressBar, QTextEdit, QGroupBox,
                             QFormLayout, QDoubleSpinBox, QCheckBox, QSpinBox)
from PyQt6.QtCore import Qt, pyqtSlot
from PyQt6.QtGui import QFont
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


class ValidationConfigDialog(QDialog):
    """Dialog for configuring BeeHaveSquE pipeline validation"""
    
    def __init__(self, parent=None, model_path=None):
        super().__init__(parent)
        self.setWindowTitle("Configure Pipeline Validation")
        self.setModal(True)
        self.setMinimumWidth(500)
        
        self.model_path = model_path
        self.config = None
        
        self.init_ui()
        
    def init_ui(self):
        """Initialize UI"""
        layout = QVBoxLayout(self)
        
        # Description
        desc_label = QLabel(
            "This will test the complete BeeHaveSquE pipeline:\n"
            "1. Run inference on first validation frame\n"
            "2. Match to ground truth (one-time ID mapping)\n"
            "3. Autonomously propagate through video using tracking\n"
            "4. Evaluate tracking performance against ground truth\n\n"
            "This tests true autonomous tracking ability."
        )
        desc_label.setWordWrap(True)
        layout.addWidget(desc_label)
        
        # Configuration group
        config_group = QGroupBox("Validation Parameters")
        config_layout = QFormLayout()
        
        # IoU threshold for detection matching
        self.iou_threshold_spin = QDoubleSpinBox()
        self.iou_threshold_spin.setRange(0.1, 0.95)
        self.iou_threshold_spin.setValue(0.5)
        self.iou_threshold_spin.setSingleStep(0.05)
        self.iou_threshold_spin.setDecimals(2)
        self.iou_threshold_spin.setToolTip("IoU threshold for counting a detection as correct")
        config_layout.addRow("IoU Threshold:", self.iou_threshold_spin)
        
        # Confidence threshold
        self.conf_threshold_spin = QDoubleSpinBox()
        self.conf_threshold_spin.setRange(0.1, 0.95)
        self.conf_threshold_spin.setValue(0.5)
        self.conf_threshold_spin.setSingleStep(0.05)
        self.conf_threshold_spin.setDecimals(2)
        self.conf_threshold_spin.setToolTip("Confidence threshold for detections")
        config_layout.addRow("Confidence Threshold:", self.conf_threshold_spin)
        
        # ID switch threshold
        self.id_switch_iou_spin = QDoubleSpinBox()
        self.id_switch_iou_spin.setRange(0.1, 0.95)
        self.id_switch_iou_spin.setValue(0.3)
        self.id_switch_iou_spin.setSingleStep(0.05)
        self.id_switch_iou_spin.setDecimals(2)
        self.id_switch_iou_spin.setToolTip("IoU threshold for detecting ID switches")
        config_layout.addRow("ID Switch Threshold:", self.id_switch_iou_spin)
        
        config_group.setLayout(config_layout)
        layout.addWidget(config_group)
        
        # SOHO Inference Parameters
        soho_group = QGroupBox("SOHO Inference Parameters")
        soho_layout = QFormLayout()
        
        # Slice size
        self.slice_size_spin = QSpinBox()
        self.slice_size_spin.setRange(256, 2048)
        self.slice_size_spin.setValue(640)
        self.slice_size_spin.setSingleStep(64)
        self.slice_size_spin.setToolTip("Size of image slices for inference (square)")
        soho_layout.addRow("Slice Size (px):", self.slice_size_spin)
        
        # Overlap ratio
        self.overlap_ratio_spin = QDoubleSpinBox()
        self.overlap_ratio_spin.setRange(0.0, 0.9)
        self.overlap_ratio_spin.setValue(0.5)
        self.overlap_ratio_spin.setSingleStep(0.1)
        self.overlap_ratio_spin.setDecimals(2)
        self.overlap_ratio_spin.setToolTip("Overlap ratio between adjacent slices (0.5 = 50% overlap)")
        soho_layout.addRow("Overlap Ratio:", self.overlap_ratio_spin)
        
        # Edge filter
        self.edge_filter_spin = QSpinBox()
        self.edge_filter_spin.setRange(0, 200)
        self.edge_filter_spin.setValue(50)
        self.edge_filter_spin.setSingleStep(10)
        self.edge_filter_spin.setToolTip("Filter detections within this many pixels from slice edges")
        soho_layout.addRow("Edge Filter (px):", self.edge_filter_spin)
        
        soho_group.setLayout(soho_layout)
        layout.addWidget(soho_group)
        
        # Metrics to compute
        metrics_group = QGroupBox("Metrics to Compute")
        metrics_layout = QVBoxLayout()
        
        self.compute_mota_check = QCheckBox("MOTA (Multiple Object Tracking Accuracy)")
        self.compute_mota_check.setChecked(True)
        self.compute_mota_check.setEnabled(False)  # Always compute
        metrics_layout.addWidget(self.compute_mota_check)
        
        self.compute_idf1_check = QCheckBox("IDF1 (ID F1 Score)")
        self.compute_idf1_check.setChecked(True)
        metrics_layout.addWidget(self.compute_idf1_check)
        
        self.compute_map_check = QCheckBox("mAP (Mean Average Precision)")
        self.compute_map_check.setChecked(True)
        metrics_layout.addWidget(self.compute_map_check)
        
        metrics_group.setLayout(metrics_layout)
        layout.addWidget(metrics_group)
        
        # Visualization options
        viz_group = QGroupBox("Visualization Options")
        viz_layout = QVBoxLayout()
        
        self.save_viz_check = QCheckBox("Save comparison images (first & last frames)")
        self.save_viz_check.setChecked(True)
        viz_layout.addWidget(self.save_viz_check)
        
        self.save_all_frames_check = QCheckBox("Save all validation frames (warning: large)")
        self.save_all_frames_check.setChecked(False)
        viz_layout.addWidget(self.save_all_frames_check)
        
        viz_group.setLayout(viz_layout)
        layout.addWidget(viz_group)
        
        # Output location
        output_group = QGroupBox("Output")
        output_layout = QVBoxLayout()
        
        if self.model_path:
            from pathlib import Path
            model_folder = Path(self.model_path).parent
            output_label = QLabel(f"Results will be saved to:\n{model_folder}/validation_results_[timestamp]/")
            output_label.setWordWrap(True)
            output_layout.addWidget(output_label)
        else:
            output_label = QLabel("Results will be saved to model folder")
            output_layout.addWidget(output_label)
        
        output_group.setLayout(output_layout)
        layout.addWidget(output_group)
        
        # Buttons
        button_layout = QHBoxLayout()
        
        self.run_btn = QPushButton("Start Validation")
        self.run_btn.clicked.connect(self.accept)
        self.run_btn.setDefault(True)
        button_layout.addWidget(self.run_btn)
        
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.clicked.connect(self.reject)
        button_layout.addWidget(self.cancel_btn)
        
        layout.addLayout(button_layout)
        
    def get_config(self):
        """Get validation configuration"""
        return {
            'iou_threshold': self.iou_threshold_spin.value(),
            'conf_threshold': self.conf_threshold_spin.value(),
            'id_switch_iou': self.id_switch_iou_spin.value(),
            'slice_size': self.slice_size_spin.value(),
            'overlap_ratio': self.overlap_ratio_spin.value(),
            'edge_filter': self.edge_filter_spin.value(),
            'compute_idf1': self.compute_idf1_check.isChecked(),
            'compute_map': self.compute_map_check.isChecked(),
            'save_visualizations': self.save_viz_check.isChecked(),
            'save_all_frames': self.save_all_frames_check.isChecked()
        }


class ValidationProgressDialog(QDialog):
    """Dialog showing validation progress with real-time metrics"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("BeeHaveSquE Pipeline Validation")
        self.setModal(True)
        self.setMinimumSize(900, 700)
        
        self.validation_worker = None
        self.metrics_history = {
            'mota': [],
            'idf1': [],
            'precision': [],
            'recall': []
        }
        
        self.validation_completed = False
        self.validation_failed = False
        self.results_path = None
        self.final_metrics = None
        self.error_message = None
        self.total_videos = 0  # Track total video count
        
        self.init_ui()
        
    def init_ui(self):
        """Initialize UI"""
        layout = QVBoxLayout(self)
        
        # Status label
        self.status_label = QLabel("Initializing validation...")
        self.status_label.setFont(QFont("Arial", 12, QFont.Weight.Bold))
        layout.addWidget(self.status_label)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        layout.addWidget(self.progress_bar)
        
        # Current video/frame info
        self.video_frame_label = QLabel("Video: - | Frame: -")
        layout.addWidget(self.video_frame_label)
        
        # Current metrics group
        metrics_group = QGroupBox("Current Metrics")
        metrics_layout = QFormLayout()
        
        self.mota_label = QLabel("N/A")
        self.idf1_label = QLabel("N/A")
        self.precision_label = QLabel("N/A")
        self.recall_label = QLabel("N/A")
        self.id_switches_label = QLabel("0")
        
        metrics_layout.addRow("MOTA:", self.mota_label)
        metrics_layout.addRow("IDF1:", self.idf1_label)
        metrics_layout.addRow("Precision:", self.precision_label)
        metrics_layout.addRow("Recall:", self.recall_label)
        metrics_layout.addRow("ID Switches:", self.id_switches_label)
        
        metrics_group.setLayout(metrics_layout)
        layout.addWidget(metrics_group)
        
        # Metrics plot
        self.figure = Figure(figsize=(8, 4))
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)
        
        self._init_empty_plot()
        
        # Log output
        log_group = QGroupBox("Validation Log")
        log_layout = QVBoxLayout()
        
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMaximumHeight(150)
        self.log_text.setFont(QFont("Courier", 9))
        log_layout.addWidget(self.log_text)
        
        log_group.setLayout(log_layout)
        layout.addWidget(log_group)
        
        # Buttons
        button_layout = QHBoxLayout()
        
        self.stop_btn = QPushButton("Stop Validation")
        self.stop_btn.clicked.connect(self.stop_validation)
        self.stop_btn.setEnabled(False)
        button_layout.addWidget(self.stop_btn)
        
        button_layout.addStretch()
        
        self.close_btn = QPushButton("Close")
        self.close_btn.clicked.connect(self.reject)
        self.close_btn.setEnabled(False)
        button_layout.addWidget(self.close_btn)
        
        layout.addLayout(button_layout)
        
    def set_worker(self, worker):
        """Set the validation worker and connect signals"""
        self.validation_worker = worker
        
        worker.video_started.connect(self.on_video_started)
        worker.frame_processed.connect(self.on_frame_processed)
        worker.video_completed.connect(self.on_video_completed)
        worker.validation_complete.connect(self.on_validation_complete)
        worker.validation_failed.connect(self.on_validation_failed)
        worker.finished.connect(self.on_worker_finished)
        
        self.stop_btn.setEnabled(True)
    
    def stop_validation(self):
        """Request validation to stop"""
        if self.validation_worker:
            self.log("Stopping validation...")
            self.status_label.setText("Stopping validation...")
            self.validation_worker.stop()
            self.stop_btn.setEnabled(False)
            self.close_btn.setEnabled(True)
    
    @pyqtSlot()
    def on_worker_finished(self):
        """Handle worker thread finishing"""
        if not self.close_btn.isEnabled():
            self.close_btn.setEnabled(True)
            if not self.validation_completed and not self.validation_failed:
                self.status_label.setText("Validation stopped by user")
                self.log("Validation stopped.")
    
    @pyqtSlot(str, int, int, int)
    def on_video_started(self, video_id, num_frames, video_num, total_videos):
        """Handle video processing start"""
        self.total_videos = total_videos  # Store for reference
        self.status_label.setText(f"Processing video {video_num}/{total_videos}: {video_id}")
        self.video_frame_label.setText(f"Video: {video_id} | Frame: 0/{num_frames}")
        self.log(f"\n=== Starting video {video_num}/{total_videos}: {video_id} ({num_frames} validation frames) ===")
    
    @pyqtSlot(str, int, int, dict)
    def on_frame_processed(self, video_id, frame_idx, total_frames, metrics):
        """Handle frame processing update"""
        # Update progress
        self.video_frame_label.setText(f"Video: {video_id} | Frame: {frame_idx}/{total_frames}")
        
        # Update metrics if available
        if metrics:
            if 'mota' in metrics:
                self.mota_label.setText(f"{metrics['mota']:.3f}")
            if 'idf1' in metrics:
                self.idf1_label.setText(f"{metrics['idf1']:.3f}")
            if 'precision' in metrics:
                self.precision_label.setText(f"{metrics['precision']:.3f}")
            if 'recall' in metrics:
                self.recall_label.setText(f"{metrics['recall']:.3f}")
            if 'id_switches' in metrics:
                self.id_switches_label.setText(f"{metrics['id_switches']}")
    
    @pyqtSlot(str, dict, int, int)
    def on_video_completed(self, video_id, video_metrics, video_num, total_videos):
        """Handle video completion"""
        # Update overall progress
        progress = int((video_num / total_videos) * 100)
        self.progress_bar.setValue(progress)
        self.status_label.setText(f"Completed video {video_num}/{total_videos}: {video_id}")
        
        # Log video results
        self.log(f"✓ Video {video_num}/{total_videos} ({video_id}) complete:")
        self.log(f"  MOTA: {video_metrics.get('mota', 0):.3f}")
        self.log(f"  IDF1: {video_metrics.get('idf1', 0):.3f}")
        self.log(f"  ID Switches: {video_metrics.get('id_switches', 0)}")
        
        # Add to history
        for key in ['mota', 'idf1', 'precision', 'recall']:
            if key in video_metrics:
                self.metrics_history[key].append((video_num, video_metrics[key]))
        
        self.update_plot()
    
    @pyqtSlot(dict, str)
    def on_validation_complete(self, overall_metrics, results_path):
        """Handle validation completion"""
        self.validation_completed = True
        self.results_path = results_path
        self.final_metrics = overall_metrics
        self.progress_bar.setValue(100)
        self.status_label.setText("✓ Validation Complete")
        
        self.log("\n" + "="*50)
        self.log("VALIDATION COMPLETE")
        self.log("="*50)
        self.log(f"MOTA: {overall_metrics.get('mota', 0):.3f}")
        self.log(f"IDF1: {overall_metrics.get('idf1', 0):.3f}")
        self.log(f"Precision: {overall_metrics.get('precision', 0):.3f}")
        self.log(f"Recall: {overall_metrics.get('recall', 0):.3f}")
        self.log(f"ID Switches: {overall_metrics.get('id_switches', 0)}")
        self.log(f"\nResults saved to: {results_path}")
        
        self.stop_btn.setEnabled(False)
        self.close_btn.setEnabled(True)
    
    @pyqtSlot(str)
    def on_validation_failed(self, error_msg):
        """Handle validation failure"""
        self.validation_failed = True
        self.error_message = error_msg
        self.status_label.setText("✗ Validation Failed")
        self.log(f"\nERROR: {error_msg}")
        
        self.stop_btn.setEnabled(False)
        self.close_btn.setEnabled(True)
    
    def log(self, message):
        """Add message to log"""
        self.log_text.append(message)
        self.log_text.verticalScrollBar().setValue(
            self.log_text.verticalScrollBar().maximum()
        )
    
    def _init_empty_plot(self):
        """Initialize empty plot structure"""
        self.figure.clear()
        ax1 = self.figure.add_subplot(121)
        ax2 = self.figure.add_subplot(122)
        
        ax1.set_xlabel('Video #')
        ax1.set_ylabel('Score')
        ax1.set_title('MOTA & IDF1')
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim([0, 1])
        
        ax2.set_xlabel('Video #')
        ax2.set_ylabel('Score')
        ax2.set_title('Precision & Recall')
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim([0, 1])
        
        self.canvas.draw()
    
    def update_plot(self):
        """Update metrics plot"""
        self.figure.clear()
        ax1 = self.figure.add_subplot(121)
        ax2 = self.figure.add_subplot(122)
        
        # Plot MOTA & IDF1
        if self.metrics_history['mota']:
            videos, mota_vals = zip(*self.metrics_history['mota'])
            ax1.plot(videos, mota_vals, 'b-o', label='MOTA', markersize=4)
        
        if self.metrics_history['idf1']:
            videos, idf1_vals = zip(*self.metrics_history['idf1'])
            ax1.plot(videos, idf1_vals, 'r-s', label='IDF1', markersize=4)
        
        ax1.set_xlabel('Video #')
        ax1.set_ylabel('Score')
        ax1.set_title('MOTA & IDF1')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim([0, 1])
        
        # Plot Precision & Recall
        if self.metrics_history['precision']:
            videos, prec_vals = zip(*self.metrics_history['precision'])
            ax2.plot(videos, prec_vals, 'g-o', label='Precision', markersize=4)
        
        if self.metrics_history['recall']:
            videos, rec_vals = zip(*self.metrics_history['recall'])
            ax2.plot(videos, rec_vals, 'm-s', label='Recall', markersize=4)
        
        ax2.set_xlabel('Video #')
        ax2.set_ylabel('Score')
        ax2.set_title('Precision & Recall')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim([0, 1])
        
        self.figure.tight_layout()
        self.canvas.draw()
