"""
Dialog and worker for creating a single-video tracking visualization MP4.
"""

import gc
import time
from pathlib import Path
from typing import Dict

import torch
from PyQt6.QtCore import QThread, pyqtSignal, QSettings
from PyQt6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QDialog,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QRadioButton,
    QScrollArea,
    QSpinBox,
    QDoubleSpinBox,
    QVBoxLayout,
    QWidget,
)
from ultralytics import YOLO

from core.batch_video_processor import BatchVideoProcessor
from core.visualization_generator import VisualizationGenerator


class TrackingVisualizationConfigDialog(QDialog):
    """Configure single-video tracking visualization export."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent_window = parent
        self.setWindowTitle("Create Tracking Visualization Video")
        self.setModal(True)
        self.setMinimumSize(700, 820)
        self.config = None
        self.default_output_dir = self._default_output_dir()
        self.settings = QSettings("BeeWhere", "TrackingVisualization")

        self.init_ui()
        self._restore_settings()

    def init_ui(self):
        main_layout = QVBoxLayout(self)

        desc_label = QLabel(
            "<h3>Create Tracking Visualization Video</h3>"
            "Run bee detection and tracking on one video, then export an annotated MP4 "
            "with IDs, trails, masks/boxes, chambers, hive overlay, and ArUco labels when enabled."
        )
        desc_label.setWordWrap(True)
        main_layout.addWidget(desc_label)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        content_widget = QWidget()
        layout = QVBoxLayout(content_widget)

        input_group = QGroupBox("Input and Output")
        input_layout = QFormLayout()

        video_layout = QHBoxLayout()
        self.video_edit = QLineEdit()
        self.video_edit.setReadOnly(True)
        self.video_edit.setPlaceholderText("Select one video file...")
        video_layout.addWidget(self.video_edit)
        video_btn = QPushButton("Browse...")
        video_btn.clicked.connect(self.browse_video)
        video_layout.addWidget(video_btn)
        input_layout.addRow("Input video:", video_layout)

        output_layout = QHBoxLayout()
        self.output_edit = QLineEdit()
        self.output_edit.setReadOnly(True)
        self.output_edit.setPlaceholderText("Select output MP4 path...")
        output_layout.addWidget(self.output_edit)
        output_btn = QPushButton("Browse...")
        output_btn.clicked.connect(self.browse_output)
        output_layout.addWidget(output_btn)
        input_layout.addRow("Output MP4:", output_layout)

        self.preview_check = QCheckBox("Preview only - first 5 frames")
        self.preview_check.setChecked(False)
        self.preview_check.setToolTip("Run only the first 5 frames so you can quickly inspect the visual style and tracking parameters.")
        input_layout.addRow("", self.preview_check)

        input_group.setLayout(input_layout)
        layout.addWidget(input_group)

        model_group = QGroupBox("Bee Detection Model")
        model_layout = QVBoxLayout()

        type_layout = QHBoxLayout()
        self.bbox_radio = QRadioButton("Bounding Box")
        self.seg_radio = QRadioButton("Segmentation")
        self.bbox_radio.setChecked(True)
        self.model_type_group = QButtonGroup()
        self.model_type_group.addButton(self.bbox_radio)
        self.model_type_group.addButton(self.seg_radio)
        type_layout.addWidget(QLabel("Detection type:"))
        type_layout.addWidget(self.bbox_radio)
        type_layout.addWidget(self.seg_radio)
        type_layout.addStretch()
        model_layout.addLayout(type_layout)

        bee_layout = QHBoxLayout()
        self.bee_model_edit = QLineEdit()
        self.bee_model_edit.setReadOnly(True)
        self.bee_model_edit.setPlaceholderText("Select YOLO bee model...")
        bee_layout.addWidget(self.bee_model_edit)
        bee_btn = QPushButton("Browse...")
        bee_btn.clicked.connect(self.browse_bee_model)
        bee_layout.addWidget(bee_btn)
        model_layout.addLayout(bee_layout)

        model_group.setLayout(model_layout)
        layout.addWidget(model_group)

        optional_group = QGroupBox("Optional Overlay Models")
        optional_layout = QFormLayout()

        hive_layout = QHBoxLayout()
        self.hive_model_edit = QLineEdit()
        self.hive_model_edit.setReadOnly(True)
        self.hive_model_edit.setPlaceholderText("Optional hive segmentation model...")
        hive_layout.addWidget(self.hive_model_edit)
        hive_btn = QPushButton("Browse...")
        hive_btn.clicked.connect(self.browse_hive_model)
        hive_layout.addWidget(hive_btn)
        hive_clear_btn = QPushButton("Clear")
        hive_clear_btn.clicked.connect(lambda: self.hive_model_edit.clear())
        hive_layout.addWidget(hive_clear_btn)
        optional_layout.addRow("Hive model:", hive_layout)

        chamber_layout = QHBoxLayout()
        self.chamber_model_edit = QLineEdit()
        self.chamber_model_edit.setReadOnly(True)
        self.chamber_model_edit.setPlaceholderText("Optional chamber segmentation model...")
        chamber_layout.addWidget(self.chamber_model_edit)
        chamber_btn = QPushButton("Browse...")
        chamber_btn.clicked.connect(self.browse_chamber_model)
        chamber_layout.addWidget(chamber_btn)
        chamber_clear_btn = QPushButton("Clear")
        chamber_clear_btn.clicked.connect(lambda: self.chamber_model_edit.clear())
        chamber_layout.addWidget(chamber_clear_btn)
        optional_layout.addRow("Chamber model:", chamber_layout)

        optional_group.setLayout(optional_layout)
        layout.addWidget(optional_group)

        tracking_group = QGroupBox("Tracking Algorithm")
        tracking_layout = QFormLayout()

        self.tracking_algo_combo = QComboBox()
        self.tracking_algo_combo.addItems(["ByteTrack", "SimpleIoU", "Centroid"])
        self.tracking_algo_combo.setCurrentText("Centroid")
        self.tracking_algo_combo.currentTextChanged.connect(self._update_tracking_params)
        tracking_layout.addRow("Algorithm:", self.tracking_algo_combo)

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
        tracking_layout.addRow(self.simpleiou_params)

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
        tracking_layout.addRow(self.centroid_params)

        tracking_group.setLayout(tracking_layout)
        layout.addWidget(tracking_group)

        detection_group = QGroupBox("Detection and Visualization Options")
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
        self.enable_aruco_check = QCheckBox("Use ArUco codes when assigning bee IDs")
        self.enable_aruco_check.setChecked(True)
        detection_layout.addRow("", self.enable_aruco_check)
        self.compute_spatial_metrics_check = QCheckBox("Compute spatial metrics for overlays")
        self.compute_spatial_metrics_check.setChecked(False)
        detection_layout.addRow("", self.compute_spatial_metrics_check)
        self.visualization_mode_combo = QComboBox()
        self.visualization_mode_combo.addItems(["Pretty", "Science"])
        self.visualization_mode_combo.setCurrentText("Pretty")
        self.visualization_mode_combo.setToolTip(
            "Pretty: clean MP4 with stable colors by bee ID and minimal overlays.\n"
            "Science: diagnostic style with frame counts, detection counts, and new-ID emphasis."
        )
        detection_layout.addRow("Visualization mode:", self.visualization_mode_combo)
        self.verbose_output_check = QCheckBox("Verbose output")
        self.verbose_output_check.setChecked(False)
        detection_layout.addRow("", self.verbose_output_check)
        detection_group.setLayout(detection_layout)
        layout.addWidget(detection_group)

        scroll.setWidget(content_widget)
        main_layout.addWidget(scroll)

        button_layout = QHBoxLayout()
        button_layout.addStretch()
        run_btn = QPushButton("Create MP4")
        run_btn.clicked.connect(self.accept)
        button_layout.addWidget(run_btn)
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        button_layout.addWidget(cancel_btn)
        main_layout.addLayout(button_layout)

        self._update_tracking_params("Centroid")

    def _update_tracking_params(self, algo_name):
        self.bytetrack_params.setVisible(algo_name == "ByteTrack")
        self.simpleiou_params.setVisible(algo_name == "SimpleIoU")
        self.centroid_params.setVisible(algo_name == "Centroid")

    def browse_video(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Video",
            str(Path.home()),
            "Video Files (*.mp4 *.avi *.mov *.mkv *.mjpeg *.mjpg);;All Files (*)",
        )
        if path:
            self.video_edit.setText(path)
            if not self.output_edit.text():
                source = Path(path)
                self.output_edit.setText(str(self.default_output_dir / f"{source.stem}_tracking_visualization.mp4"))

    def browse_output(self):
        default = self.output_edit.text() or str(self.default_output_dir / "tracking_visualization.mp4")
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Tracking Visualization",
            default,
            "Video Files (*.mp4 *.avi);;All Files (*)",
        )
        if path:
            if Path(path).suffix == "":
                path = f"{path}.mp4"
            self.output_edit.setText(path)

    def _default_output_dir(self) -> Path:
        """Return the default folder for example tracking visualization videos."""
        for parent in Path(__file__).resolve().parents:
            candidate = parent / "03_data_outputs" / "00_examples" / "videos"
            if candidate.parent.exists():
                candidate.mkdir(parents=True, exist_ok=True)
                return candidate

        candidate = Path.cwd() / "03_data_outputs" / "00_examples" / "videos"
        candidate.mkdir(parents=True, exist_ok=True)
        return candidate

    def browse_bee_model(self):
        self._browse_model(self.bee_model_edit, "Select Bee Detection Model")

    def browse_hive_model(self):
        self._browse_model(self.hive_model_edit, "Select Hive Segmentation Model")

    def browse_chamber_model(self):
        self._browse_model(self.chamber_model_edit, "Select Chamber Segmentation Model")

    def _browse_model(self, line_edit: QLineEdit, title: str):
        path, _ = QFileDialog.getOpenFileName(
            self,
            title,
            str(Path.home()),
            "PyTorch Model Files (*.pt);;All Files (*)",
        )
        if path:
            line_edit.setText(path)

    def accept(self):
        video_path = Path(self.video_edit.text())
        if not self.video_edit.text() or not video_path.exists():
            QMessageBox.warning(self, "Input Required", "Please select a valid input video.")
            return

        bee_model_path = Path(self.bee_model_edit.text())
        if not self.bee_model_edit.text() or not bee_model_path.exists():
            QMessageBox.warning(self, "Model Required", "Please select a valid bee detection model.")
            return

        output_path = Path(self.output_edit.text())
        if not self.output_edit.text():
            QMessageBox.warning(self, "Output Required", "Please select an output MP4 path.")
            return
        output_path.parent.mkdir(parents=True, exist_ok=True)

        hive_model_path = Path(self.hive_model_edit.text()) if self.hive_model_edit.text() else None
        if hive_model_path and not hive_model_path.exists():
            QMessageBox.warning(self, "Invalid Path", "Hive model does not exist.")
            return

        chamber_model_path = Path(self.chamber_model_edit.text()) if self.chamber_model_edit.text() else None
        if chamber_model_path and not chamber_model_path.exists():
            QMessageBox.warning(self, "Invalid Path", "Chamber model does not exist.")
            return

        tracking_algo = self.tracking_algo_combo.currentText()
        if tracking_algo == "ByteTrack":
            tracking_config = {
                'algorithm': 'bytetrack',
                'high_conf_threshold': self.bt_high_conf_spin.value(),
                'high_iou_threshold': self.bt_high_iou_spin.value(),
                'low_iou_threshold': self.bt_low_iou_spin.value(),
                'max_frames_lost': self.bt_max_lost_spin.value(),
                'use_mask_iou': self.bt_mask_iou_check.isChecked(),
            }
        elif tracking_algo == "SimpleIoU":
            tracking_config = {
                'algorithm': 'simple_iou',
                'iou_threshold': self.siou_threshold_spin.value(),
                'use_mask_iou': self.siou_mask_iou_check.isChecked(),
            }
        else:
            tracking_config = {
                'algorithm': 'centroid',
                'max_distance': self.cent_max_dist_spin.value(),
                'max_frames_missing': self.cent_max_missing_spin.value(),
            }

        self.config = {
            'video_path': str(video_path),
            'output_path': str(output_path),
            'preview': self.preview_check.isChecked(),
            'bee_model_path': str(bee_model_path),
            'bee_model_type': 'segmentation' if self.seg_radio.isChecked() else 'bbox',
            'hive_model_path': str(hive_model_path) if hive_model_path else None,
            'chamber_model_path': str(chamber_model_path) if chamber_model_path else None,
            'tracking_config': tracking_config,
            'confidence_threshold': self.confidence_spin.value(),
            'nms_iou_threshold': self.nms_iou_spin.value(),
            'enable_aruco': self.enable_aruco_check.isChecked(),
            'compute_spatial_metrics': self.compute_spatial_metrics_check.isChecked(),
            'distance_method': 'centroid',
            'visualization_mode': self.visualization_mode_combo.currentText().lower(),
            'verbose_output': self.verbose_output_check.isChecked(),
        }
        
        # Save settings for next time dialog is opened
        self._save_settings()
        
        super().accept()

    def _restore_settings(self):
        """Restore bee model path and model type from previous session."""
        bee_model_path = self.settings.value("bee_model_path", "")
        model_type = self.settings.value("bee_model_type", "bbox")
        
        if bee_model_path and Path(bee_model_path).exists():
            self.bee_model_edit.setText(bee_model_path)
        
        if model_type == "segmentation":
            self.seg_radio.setChecked(True)
        else:
            self.bbox_radio.setChecked(True)
    
    def _save_settings(self):
        """Save bee model path and model type for next session."""
        bee_model_path = self.bee_model_edit.text()
        if bee_model_path:
            self.settings.setValue("bee_model_path", bee_model_path)
        
        model_type = "segmentation" if self.seg_radio.isChecked() else "bbox"
        self.settings.setValue("bee_model_type", model_type)
        self.settings.sync()


class TrackingVisualizationWorker(QThread):
    """Run one-video tracking and export an annotated MP4."""

    status_updated = pyqtSignal(str)
    progress_updated = pyqtSignal(int, int)
    log_message = pyqtSignal(str)
    visualization_complete = pyqtSignal(str)
    visualization_stopped = pyqtSignal()
    visualization_failed = pyqtSignal(str)

    def __init__(self, config: Dict):
        super().__init__()
        self.config = config
        self.should_stop = False
        self.verbose_output = bool(config.get('verbose_output', False))

    def stop(self):
        self.should_stop = True

    def _log_verbose(self, message: str):
        if self.verbose_output:
            self.log_message.emit(message)

    def run(self):
        try:
            video_path = Path(self.config['video_path'])
            output_path = Path(self.config['output_path'])
            max_frames = 5 if self.config.get('preview') else None

            self.log_message.emit("=== Tracking Visualization Video ===")
            self.log_message.emit(f"Video: {video_path}")
            self.log_message.emit(f"Output: {output_path}")
            self.log_message.emit(f"Mode: {'Preview first 5 frames' if max_frames else 'Full video'}")
            self.log_message.emit(f"Tracking algorithm: {self.config['tracking_config']['algorithm']}")
            self.log_message.emit(f"ArUco detection: {'Enabled' if self.config['enable_aruco'] else 'Disabled'}")

            self.status_updated.emit("Loading models...")
            bee_model = YOLO(self.config['bee_model_path'])
            hive_model = YOLO(self.config['hive_model_path']) if self.config.get('hive_model_path') else None
            chamber_model = YOLO(self.config['chamber_model_path']) if self.config.get('chamber_model_path') else None

            for model in (bee_model, hive_model, chamber_model):
                if model and hasattr(model, 'model') and hasattr(model.model, 'eval'):
                    model.model.eval()

            tracker = self._initialize_tracker()

            self.status_updated.emit("Running tracking...")
            processor = BatchVideoProcessor(
                video_path=video_path,
                video_id=video_path.stem,
                bee_model=bee_model,
                hive_model=hive_model,
                chamber_model=chamber_model,
                tracker=tracker,
                confidence_threshold=self.config['confidence_threshold'],
                nms_iou_threshold=self.config['nms_iou_threshold'],
                enable_aruco=self.config['enable_aruco'],
                output_folder=output_path.parent,
                distance_method=self.config.get('distance_method', 'centroid'),
                bee_model_type=self.config.get('bee_model_type', 'bbox'),
                compute_spatial_metrics=self.config.get('compute_spatial_metrics', False),
                store_masks=True,
                log_callback=self.log_message.emit,
                stop_callback=lambda: self.should_stop,
                timing_log_interval=1,
                cleanup_interval=25,
                verbose_output=self.verbose_output,
                max_frames=max_frames,
                high_quality_masks=True,
            )

            start = time.perf_counter()
            success = processor.process()
            elapsed = time.perf_counter() - start
            if not success and (self.should_stop or getattr(processor, 'was_stopped', False)):
                self.visualization_stopped.emit()
                return
            if not success:
                raise RuntimeError("Video processing failed")

            self.log_message.emit(
                f"Tracked {processor.frame_count} frame(s), "
                f"{len(processor.get_bee_detections())} bee detections in {elapsed:.1f}s"
            )

            self.status_updated.emit("Writing MP4...")
            viz_gen = VisualizationGenerator(
                video_path=video_path,
                output_folder=output_path.parent,
                video_id=video_path.stem,
                bee_detections=processor.get_bee_detections(),
                chamber_frame_data=processor.get_chamber_frame_data(),
                chambers_by_frame=processor.get_chambers_by_frame(),
                hive_masks_by_frame=processor.get_hive_masks_by_frame(),
                bee_masks_by_frame=processor.get_bee_masks_by_frame(),
                aruco_markers_by_frame=processor.get_aruco_markers_by_frame(),
                max_frame=processor.frame_count,
                log_callback=self.log_message.emit,
                verbose_output=self.verbose_output,
                show_chambers=chamber_model is not None,
                show_chamber_info=chamber_model is not None,
                bubble_outlines=True,
                visualization_mode=self.config.get('visualization_mode', 'pretty'),
            )

            if not viz_gen.generate_video(output_path):
                raise RuntimeError("No annotated frames were written to the MP4")

            self.progress_updated.emit(1, 1)
            self.log_message.emit(f"✓ Tracking visualization saved: {output_path}")
            self.visualization_complete.emit(str(output_path))

            del viz_gen, processor
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        except Exception as e:
            import traceback
            self.log_message.emit(f"❌ Tracking visualization failed: {str(e)}")
            self._log_verbose(traceback.format_exc())
            self.visualization_failed.emit(str(e))

    def _initialize_tracker(self):
        tracking_config = self.config['tracking_config']
        algo = tracking_config['algorithm']

        if algo == 'bytetrack':
            from core.instance_tracker import InstanceTracker
            return InstanceTracker(config={
                'high_conf_threshold': tracking_config['high_conf_threshold'],
                'high_iou_threshold': tracking_config['high_iou_threshold'],
                'low_iou_threshold': tracking_config['low_iou_threshold'],
                'max_frames_lost': tracking_config['max_frames_lost'],
                'use_mask_iou': tracking_config['use_mask_iou'],
            })

        if algo == 'simple_iou':
            from gui.tracking_validation_worker import SimpleIoUTracker
            return SimpleIoUTracker(
                iou_threshold=tracking_config['iou_threshold'],
                use_mask_iou=tracking_config['use_mask_iou'],
            )

        if algo == 'centroid':
            from gui.tracking_validation_worker import CentroidTracker
            return CentroidTracker(
                max_distance=tracking_config['max_distance'],
                max_frames_missing=tracking_config['max_frames_missing'],
            )

        raise ValueError(f"Unknown tracking algorithm: {algo}")
