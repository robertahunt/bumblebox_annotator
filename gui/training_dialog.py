"""
Training progress dialog for YOLO model training
"""

from PyQt6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel, 
                             QPushButton, QProgressBar, QTextEdit, QGroupBox,
                             QFormLayout, QSpinBox, QComboBox, QCheckBox)
from PyQt6.QtCore import Qt, pyqtSlot
from PyQt6.QtGui import QFont
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


class TrainingProgressDialog(QDialog):
    """Dialog showing YOLO training progress with metrics"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("YOLO Training Progress")
        self.setModal(True)
        self.setMinimumSize(800, 600)
        
        self.training_worker = None
        # Store metrics as lists of (epoch, value) tuples to keep them synchronized
        self.metrics_history = {
            'train_loss': [],  # [(epoch, value), ...]
            'val_loss': [],
            'mAP50': [],
            'mAP50-95': []
        }
        
        # Track completion status
        self.training_completed = False
        self.training_failed = False
        self.model_path = None
        self.final_metrics = {}
        
        self.init_ui()
        
    def init_ui(self):
        """Initialize UI"""
        layout = QVBoxLayout(self)
        
        # Status label
        self.status_label = QLabel("Initializing...")
        self.status_label.setFont(QFont("Arial", 12, QFont.Weight.Bold))
        layout.addWidget(self.status_label)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        layout.addWidget(self.progress_bar)
        
        # Current metrics group
        metrics_group = QGroupBox("Current Metrics")
        metrics_layout = QFormLayout()
        
        self.epoch_label = QLabel("0 / 0")
        self.train_loss_label = QLabel("N/A")
        self.val_loss_label = QLabel("N/A")
        self.map50_label = QLabel("N/A")
        self.map50_95_label = QLabel("N/A")
        
        metrics_layout.addRow("Epoch:", self.epoch_label)
        metrics_layout.addRow("Train Loss:", self.train_loss_label)
        metrics_layout.addRow("Val Loss:", self.val_loss_label)
        metrics_layout.addRow("mAP50:", self.map50_label)
        metrics_layout.addRow("mAP50-95:", self.map50_95_label)
        
        metrics_group.setLayout(metrics_layout)
        layout.addWidget(metrics_group)
        
        # Training curves plot
        self.figure = Figure(figsize=(8, 4))
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)
        
        # Initialize empty plot
        self._init_empty_plot()
        
        # Log output
        log_group = QGroupBox("Training Log")
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
        
        self.stop_btn = QPushButton("Stop Training")
        self.stop_btn.clicked.connect(self.stop_training)
        self.stop_btn.setEnabled(False)
        button_layout.addWidget(self.stop_btn)
        
        button_layout.addStretch()
        
        self.close_btn = QPushButton("Close")
        self.close_btn.clicked.connect(self.reject)
        self.close_btn.setEnabled(False)
        button_layout.addWidget(self.close_btn)
        
        layout.addLayout(button_layout)
        
    def set_worker(self, worker):
        """Set the training worker and connect signals"""
        self.training_worker = worker
        
        worker.stage_update.connect(self.on_stage_update)
        worker.progress_update.connect(self.on_progress_update)
        worker.training_complete.connect(self.on_training_complete)
        worker.training_failed.connect(self.on_training_failed)
        worker.finished.connect(self.on_worker_finished)  # Connect to QThread's finished signal
        
        self.stop_btn.setEnabled(True)
    
    def stop_training(self):
        """Request training to stop"""
        if self.training_worker:
            self.log("Stopping training...")
            self.status_label.setText("Stopping training...")
            self.training_worker.stop()
            self.stop_btn.setEnabled(False)
            # Enable close button immediately so user can close once worker stops
            self.close_btn.setEnabled(True)
    
    @pyqtSlot()
    def on_worker_finished(self):
        """Handle worker thread finishing (for any reason)"""
        # Make sure close button is enabled when worker thread finishes
        if not self.close_btn.isEnabled():
            self.close_btn.setEnabled(True)
            if not self.training_completed and not self.training_failed:
                # Training was stopped by user
                self.status_label.setText("Training stopped by user")
                self.log("Training stopped.")
            
    @pyqtSlot(str)
    def on_stage_update(self, message):
        """Handle stage update"""
        self.status_label.setText(message)
        self.log(message)
        
    @pyqtSlot(int, int, dict)
    def on_progress_update(self, epoch, total_epochs, metrics):
        """Handle progress update"""
        # Update progress bar
        progress = int((epoch / total_epochs) * 100)
        self.progress_bar.setValue(progress)
        
        # Update status
        self.status_label.setText(f"Training: Epoch {epoch}/{total_epochs}")
        
        # Update metrics display
        self.epoch_label.setText(f"{epoch} / {total_epochs}")
        
        if metrics:
            if 'train_loss' in metrics:
                self.train_loss_label.setText(f"{metrics['train_loss']:.4f}")
            if 'val_loss' in metrics:
                self.val_loss_label.setText(f"{metrics['val_loss']:.4f}")
            if 'mAP50' in metrics:
                self.map50_label.setText(f"{metrics['mAP50']:.4f}")
            if 'mAP50-95' in metrics:
                self.map50_95_label.setText(f"{metrics['mAP50-95']:.4f}")
            
            # Store metrics history as (epoch, value) tuples
            for key in ['train_loss', 'val_loss', 'mAP50', 'mAP50-95']:
                if key in metrics:
                    self.metrics_history[key].append((epoch, metrics[key]))
            
            # Update plot
            self.update_plot()
            
        # Log
        log_msg = f"Epoch {epoch}/{total_epochs}"
        if metrics:
            log_msg += f" - Loss: {metrics.get('train_loss', 0):.4f}"
            log_msg += f" - mAP50: {metrics.get('mAP50', 0):.4f}"
        self.log(log_msg)
    
    def _init_empty_plot(self):
        """Initialize empty plot structure"""
        # Create subplots
        ax1 = self.figure.add_subplot(121)
        ax2 = self.figure.add_subplot(122)
        
        # Set up loss plot
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training & Validation Loss')
        ax1.grid(True, alpha=0.3)
        ax1.text(0.5, 0.5, 'Waiting for training data...', 
                ha='center', va='center', transform=ax1.transAxes,
                fontsize=10, color='gray')
        
        # Set up mAP plot
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('mAP')
        ax2.set_title('Validation mAP')
        ax2.grid(True, alpha=0.3)
        ax2.text(0.5, 0.5, 'Waiting for training data...', 
                ha='center', va='center', transform=ax2.transAxes,
                fontsize=10, color='gray')
        
        self.figure.tight_layout()
        self.canvas.draw()
        
    def update_plot(self):
        """Update training curves plot"""
        self.figure.clear()
        
        # Check if we have any metrics
        has_data = any(len(v) > 0 for v in self.metrics_history.values())
        if not has_data:
            return
            
        # Create subplots
        ax1 = self.figure.add_subplot(121)
        ax2 = self.figure.add_subplot(122)
        
        # Plot losses
        has_loss_data = False
        if self.metrics_history['train_loss']:
            epochs, values = zip(*self.metrics_history['train_loss'])
            ax1.plot(epochs, values, 'b-', label='Train Loss', linewidth=2)
            has_loss_data = True
        if self.metrics_history['val_loss']:
            epochs, values = zip(*self.metrics_history['val_loss'])
            ax1.plot(epochs, values, 'r-', label='Val Loss', linewidth=2)
            has_loss_data = True
        
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training Loss')
        if has_loss_data:
            ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot mAP
        has_map_data = False
        if self.metrics_history['mAP50']:
            epochs, values = zip(*self.metrics_history['mAP50'])
            ax2.plot(epochs, values, 'g-', label='mAP50', linewidth=2)
            has_map_data = True
        if self.metrics_history['mAP50-95']:
            epochs, values = zip(*self.metrics_history['mAP50-95'])
            ax2.plot(epochs, values, 'm-', label='mAP50-95', linewidth=2)
            has_map_data = True
        
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('mAP')
        ax2.set_title('Validation mAP')
        if has_map_data:
            ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        self.figure.tight_layout()
        self.canvas.draw()
        
    @pyqtSlot(str, dict)
    def on_training_complete(self, model_path, final_metrics):
        """Handle training completion"""
        self.training_completed = True
        self.model_path = model_path
        self.final_metrics = final_metrics
        
        self.progress_bar.setValue(100)
        self.status_label.setText("✓ Training Complete!")
        
        msg = f"Training complete!\nModel saved to: {model_path}\n\n"
        msg += "Final Metrics:\n"
        for key, value in final_metrics.items():
            msg += f"  {key}: {value:.4f}\n"
        
        self.log(msg)
        
        self.stop_btn.setEnabled(False)
        self.close_btn.setEnabled(True)
        
    @pyqtSlot(str)
    def on_training_failed(self, error_message):
        """Handle training failure"""
        self.training_failed = True
        
        self.status_label.setText("✗ Training Failed")
        self.log(f"ERROR: {error_message}")
        
        self.stop_btn.setEnabled(False)
        self.close_btn.setEnabled(True)
        
    def log(self, message):
        """Append message to log"""
        self.log_text.append(message)
        # Auto-scroll to bottom
        self.log_text.verticalScrollBar().setValue(
            self.log_text.verticalScrollBar().maximum()
        )


class TrainingConfigDialog(QDialog):
    """Dialog for configuring YOLO training parameters"""
    
    def __init__(self, parent=None, current_model_path=None, stage2=False, sahi=False):
        super().__init__(parent)
        self.stage2 = stage2
        self.sahi = sahi
        
        if sahi:
            title = "Configure SAHI Training"
        elif stage2:
            title = "Configure Stage 2 Training"
        else:
            title = "Configure YOLO Training"
        
        self.setWindowTitle(title)
        self.setModal(True)
        self.current_model_path = current_model_path
        
        self.init_ui()
        
    def init_ui(self):
        """Initialize UI"""
        layout = QVBoxLayout(self)
        
        # Info label
        if self.sahi:
            info_text = (
                "Configure parameters for training the SAHI model with enhanced augmentation.\n"
                "This model is trained on full images with aggressive data augmentation,\n"
                "optimized for sliced inference on large images.\n"
                "The current model will be backed up before training starts."
            )
        elif self.stage2:
            info_text = (
                "Configure parameters for training the Stage 2 fine-grained refinement model.\n"
                "Training crops are generated around each bee, with all visible bees labeled.\n"
                "The current model will be backed up before training starts."
            )
        else:
            info_text = (
                "Configure parameters for training the YOLO coarse detection model.\n"
                "The current model will be backed up before training starts."
            )
        info_label = QLabel(info_text)
        info_label.setWordWrap(True)
        layout.addWidget(info_label)
        
        # Training parameters
        params_group = QGroupBox("Training Parameters")
        params_layout = QFormLayout()
        
        # Epochs
        self.epochs_spin = QSpinBox()
        self.epochs_spin.setRange(1, 1000)
        if self.sahi:
            self.epochs_spin.setValue(100)  # More epochs for SAHI training
        else:
            self.epochs_spin.setValue(50)
        self.epochs_spin.setSuffix(" epochs")
        params_layout.addRow("Epochs:", self.epochs_spin)
        
        # Batch size
        self.batch_spin = QSpinBox()
        self.batch_spin.setRange(1, 64)
        if self.sahi:
            self.batch_spin.setValue(8)  # Smaller batch for larger images
        else:
            self.batch_spin.setValue(8)
        params_layout.addRow("Batch Size:", self.batch_spin)
        
        # Image size
        self.imgsz_combo = QComboBox()
        if self.sahi:
            self.imgsz_combo.addItems(['640', '800', '1024', '1280', '1536'])
            self.imgsz_combo.setCurrentText('640')  # Larger size for SAHI
        else:
            self.imgsz_combo.addItems(['320', '480', '640', '800', '1024'])
            self.imgsz_combo.setCurrentText('640')
        params_layout.addRow("Image Size:", self.imgsz_combo)
        
        # Learning rate
        self.lr_combo = QComboBox()
        self.lr_combo.addItems(['0.0001', '0.0005', '0.001', '0.005', '0.01'])
        self.lr_combo.setCurrentText('0.001')
        params_layout.addRow("Learning Rate:", self.lr_combo)
        
        # Patience (early stopping)
        self.patience_spin = QSpinBox()
        self.patience_spin.setRange(1, 100)
        if self.sahi:
            self.patience_spin.setValue(20)  # More patience for SAHI
        else:
            self.patience_spin.setValue(10)
        self.patience_spin.setSuffix(" epochs")
        self.patience_spin.setToolTip("Stop training if no improvement for this many epochs")
        params_layout.addRow("Patience:", self.patience_spin)
        
        params_group.setLayout(params_layout)
        layout.addWidget(params_group)
        
        # Data Export Options
        export_group = QGroupBox("Data Export")
        export_layout = QFormLayout()
        
        self.export_coco_checkbox = QCheckBox()
        self.export_coco_checkbox.setChecked(True)
        self.export_coco_checkbox.setToolTip(
            "Export COCO annotations before training starts.\n"
            "Recommended to ensure training uses latest annotations."
        )
        export_layout.addRow("Export COCO Annotations:", self.export_coco_checkbox)
        
        export_group.setLayout(export_layout)
        layout.addWidget(export_group)
        
        # SAHI specific parameters
        if self.sahi:
            sahi_group = QGroupBox("SAHI Random Crop Parameters")
            sahi_layout = QFormLayout()
            
            # Crop size
            self.crop_size_spin = QSpinBox()
            self.crop_size_spin.setRange(320, 1536)
            self.crop_size_spin.setSingleStep(64)
            self.crop_size_spin.setValue(640)
            self.crop_size_spin.setSuffix(" px")
            self.crop_size_spin.setToolTip("Size of random crops (should match SAHI slice size at inference)")
            sahi_layout.addRow("Crop Size:", self.crop_size_spin)
            
            # Maximum number of crops per image
            self.max_crops_spin = QSpinBox()
            self.max_crops_spin.setRange(1, 20)
            self.max_crops_spin.setValue(10)
            self.max_crops_spin.setToolTip(
                "Maximum crops per image. Actual number is determined by instance count:\n"
                "1-2 instances → 2 crops, 3-5 instances → 3 crops,\n"
                "6-10 instances → 5 crops, 11+ instances → 8 crops"
            )
            sahi_layout.addRow("Max Crops Per Image:", self.max_crops_spin)
            
            # Include full images checkbox
            self.include_full_checkbox = QCheckBox()
            self.include_full_checkbox.setChecked(True)
            self.include_full_checkbox.setToolTip("Include full images in addition to crops")
            sahi_layout.addRow("Include Full Images:", self.include_full_checkbox)
            
            sahi_group.setLayout(sahi_layout)
            layout.addWidget(sahi_group)
        
        # Stage 2 specific parameters
        if self.stage2:
            stage2_group = QGroupBox("Stage 2 Crop Parameters")
            stage2_layout = QFormLayout()
            
            # Crop padding
            self.crop_padding_spin = QSpinBox()
            self.crop_padding_spin.setRange(0, 100)
            self.crop_padding_spin.setValue(30)
            self.crop_padding_spin.setSuffix("%")
            self.crop_padding_spin.setToolTip("Percentage of bbox size to add as padding around crops")
            stage2_layout.addRow("Crop Padding:", self.crop_padding_spin)
            
            # Min crop size
            self.min_crop_size_spin = QSpinBox()
            self.min_crop_size_spin.setRange(32, 512)
            self.min_crop_size_spin.setValue(128)
            self.min_crop_size_spin.setSuffix(" px")
            self.min_crop_size_spin.setToolTip("Minimum crop size (smaller crops will be skipped)")
            stage2_layout.addRow("Min Crop Size:", self.min_crop_size_spin)
            
            stage2_group.setLayout(stage2_layout)
            layout.addWidget(stage2_group)
        
        # Model name
        name_group = QGroupBox("Model Name")
        name_layout = QFormLayout()
        
        self.name_combo = QComboBox()
        self.name_combo.setEditable(True)
        if self.sahi:
            self.name_combo.addItems(['bee_segmentation_sahi', 'bee_segmentation_sahi_v2', 'bee_sahi'])
            self.name_combo.setCurrentText('bee_segmentation_sahi')
        elif self.stage2:
            self.name_combo.addItems(['bee_segmentation_stage2', 'bee_segmentation_stage2_v2', 'bee_stage2'])
            self.name_combo.setCurrentText('bee_segmentation_stage2')
        else:
            self.name_combo.addItems(['bee_segmentation', 'bee_segmentation2', 'bee_segmentation_v2'])
            self.name_combo.setCurrentText('bee_segmentation')
        name_layout.addRow("Experiment Name:", self.name_combo)
        
        name_group.setLayout(name_layout)
        layout.addWidget(name_group)
        
        # Buttons
        button_layout = QHBoxLayout()
        
        start_btn = QPushButton("Start Training")
        start_btn.clicked.connect(self.accept)
        button_layout.addWidget(start_btn)
        
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        button_layout.addWidget(cancel_btn)
        
        layout.addLayout(button_layout)
        
    def get_config(self):
        """Get training configuration"""
        config = {
            'epochs': self.epochs_spin.value(),
            'batch': self.batch_spin.value(),
            'imgsz': int(self.imgsz_combo.currentText()),
            'lr0': float(self.lr_combo.currentText()),
            'patience': self.patience_spin.value(),
            'name': self.name_combo.currentText(),
            'current_model_path': self.current_model_path,
            'export_coco': self.export_coco_checkbox.isChecked()
        }
        
        # Add SAHI specific parameters
        if self.sahi:
            config['crop_size'] = self.crop_size_spin.value()
            config['max_crops_per_image'] = self.max_crops_spin.value()
            config['include_full_images'] = self.include_full_checkbox.isChecked()
        
        # Add Stage 2 specific parameters
        if self.stage2:
            config['crop_padding'] = self.crop_padding_spin.value() / 100.0  # Convert to decimal
            config['min_crop_size'] = self.min_crop_size_spin.value()
        
        return config
