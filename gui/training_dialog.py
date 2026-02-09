"""
Training configuration dialog for Detectron2
"""

from PyQt6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel, 
                             QPushButton, QSpinBox, QDoubleSpinBox, QComboBox,
                             QFileDialog, QLineEdit, QGroupBox, QCheckBox,
                             QTextEdit, QProgressBar, QSlider)
from PyQt6.QtCore import Qt, pyqtSignal, QThread
from pathlib import Path


class TrainingDialog(QDialog):
    """Dialog for configuring and running Detectron2 training"""
    
    training_started = pyqtSignal()
    training_completed = pyqtSignal(dict)  # Emit results when done
    
    def __init__(self, project_path: Path, train_stats: dict, val_stats: dict, parent=None):
        super().__init__(parent)
        self.project_path = project_path
        self.train_stats = train_stats
        self.val_stats = val_stats
        
        self.setWindowTitle("Train Instance Segmentation Model")
        self.setMinimumWidth(600)
        self.setMinimumHeight(700)
        
        self.init_ui()
        
    def init_ui(self):
        """Initialize UI components"""
        layout = QVBoxLayout()
        
        # Dataset info
        dataset_group = QGroupBox("Dataset Information")
        dataset_layout = QVBoxLayout()
        
        info_text = (
            f"Training: {self.train_stats['num_images']} images, "
            f"{self.train_stats['num_annotations']} annotations, "
            f"{self.train_stats['num_unique_tracks']} unique tracks\n"
            f"Validation: {self.val_stats['num_images']} images, "
            f"{self.val_stats['num_annotations']} annotations"
        )
        dataset_layout.addWidget(QLabel(info_text))
        dataset_group.setLayout(dataset_layout)
        layout.addWidget(dataset_group)
        
        # Model configuration
        model_group = QGroupBox("Model Configuration")
        model_layout = QVBoxLayout()
        
        # Architecture selection
        arch_layout = QHBoxLayout()
        arch_layout.addWidget(QLabel("Architecture:"))
        self.arch_combo = QComboBox()
        self.arch_combo.addItems([
            "Mask R-CNN R50-FPN (Fast)",
            "Mask R-CNN R101-FPN (Better)",
            "Mask R-CNN X101-FPN (Best)"
        ])
        self.arch_combo.setCurrentIndex(0)
        arch_layout.addWidget(self.arch_combo)
        model_layout.addLayout(arch_layout)
        
        # Pretrained weights
        weights_layout = QHBoxLayout()
        weights_layout.addWidget(QLabel("Pretrained Weights:"))
        self.weights_combo = QComboBox()
        self.weights_combo.addItems([
            "COCO (Recommended)",
            "Custom Checkpoint..."
        ])
        weights_layout.addWidget(self.weights_combo)
        self.weights_browse = QPushButton("Browse...")
        self.weights_browse.clicked.connect(self.browse_weights)
        self.weights_browse.setEnabled(False)
        self.weights_combo.currentIndexChanged.connect(
            lambda idx: self.weights_browse.setEnabled(idx == 1)
        )
        weights_layout.addWidget(self.weights_browse)
        model_layout.addLayout(weights_layout)
        
        self.weights_path = None
        
        model_group.setLayout(model_layout)
        layout.addWidget(model_group)
        
        # Training hyperparameters
        hyperparam_group = QGroupBox("Training Hyperparameters")
        hyperparam_layout = QVBoxLayout()
        
        # Learning rate
        lr_layout = QHBoxLayout()
        lr_layout.addWidget(QLabel("Learning Rate:"))
        self.lr_spin = QDoubleSpinBox()
        self.lr_spin.setDecimals(5)
        self.lr_spin.setRange(0.00001, 0.01)
        self.lr_spin.setValue(0.00025)
        self.lr_spin.setSingleStep(0.00001)
        lr_layout.addWidget(self.lr_spin)
        hyperparam_layout.addLayout(lr_layout)
        
        # Batch size
        batch_layout = QHBoxLayout()
        batch_layout.addWidget(QLabel("Batch Size:"))
        self.batch_spin = QSpinBox()
        self.batch_spin.setRange(1, 16)
        self.batch_spin.setValue(1)  # Default to 1 for limited GPU memory
        batch_layout.addWidget(self.batch_spin)
        batch_layout.addWidget(QLabel("(use 1 for <8GB GPU)"))
        hyperparam_layout.addLayout(batch_layout)
        
        # Max iterations (auto-calculated but editable)
        iter_layout = QHBoxLayout()
        iter_layout.addWidget(QLabel("Max Iterations:"))
        self.iter_spin = QSpinBox()
        self.iter_spin.setRange(10, 10000)
        self.iter_spin.setSingleStep(100)
        
        # Auto-calculate based on dataset size
        num_train = self.train_stats['num_images']
        suggested_iter = self._calculate_iterations(num_train)
        self.iter_spin.setValue(10)  # Set to 10 for quick debugging
        
        iter_layout.addWidget(self.iter_spin)
        iter_layout.addWidget(QLabel(f"(~{suggested_iter//100} validations)"))
        hyperparam_layout.addLayout(iter_layout)
        
        # Validation period
        val_layout = QHBoxLayout()
        val_layout.addWidget(QLabel("Validation Every:"))
        self.val_spin = QSpinBox()
        self.val_spin.setRange(10, 5000)
        self.val_spin.setValue(100)
        self.val_spin.setSingleStep(50)
        self.val_spin.setSuffix(" iterations")
        val_layout.addWidget(self.val_spin)
        hyperparam_layout.addLayout(val_layout)
        
        hyperparam_group.setLayout(hyperparam_layout)
        layout.addWidget(hyperparam_group)
        
        # Augmentation options
        aug_group = QGroupBox("Data Augmentation")
        aug_layout = QVBoxLayout()
        
        self.aug_flip_h = QCheckBox("Horizontal Flip")
        self.aug_flip_h.setChecked(True)
        aug_layout.addWidget(self.aug_flip_h)
        
        self.aug_flip_v = QCheckBox("Vertical Flip")
        self.aug_flip_v.setChecked(True)
        aug_layout.addWidget(self.aug_flip_v)
        
        # Crop size slider
        crop_layout = QHBoxLayout()
        self.aug_crop = QCheckBox("Random Crop")
        self.aug_crop.setChecked(True)
        self.aug_crop.toggled.connect(lambda checked: self.crop_slider.setEnabled(checked))
        crop_layout.addWidget(self.aug_crop)
        
        self.crop_slider = QSlider(Qt.Orientation.Horizontal)
        self.crop_slider.setRange(50, 95)
        self.crop_slider.setValue(80)
        self.crop_slider.setTickInterval(5)
        self.crop_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.crop_label = QLabel("80%")
        self.crop_slider.valueChanged.connect(lambda v: self.crop_label.setText(f"{v}%"))
        crop_layout.addWidget(self.crop_slider)
        crop_layout.addWidget(self.crop_label)
        aug_layout.addLayout(crop_layout)
        
        self.aug_rotation = QCheckBox("Random Rotation (±15°)")
        self.aug_rotation.setChecked(True)
        aug_layout.addWidget(self.aug_rotation)
        
        self.aug_brightness = QCheckBox("Brightness/Contrast Jitter")
        self.aug_brightness.setChecked(True)
        aug_layout.addWidget(self.aug_brightness)
        
        self.aug_color = QCheckBox("Color Jitter")
        self.aug_color.setChecked(True)
        aug_layout.addWidget(self.aug_color)
        
        aug_group.setLayout(aug_layout)
        layout.addWidget(aug_group)
        
        # Progress area (hidden initially)
        self.progress_group = QGroupBox("Training Progress")
        self.progress_group.setVisible(False)
        progress_layout = QVBoxLayout()
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        progress_layout.addWidget(self.progress_bar)
        
        self.progress_text = QTextEdit()
        self.progress_text.setReadOnly(True)
        self.progress_text.setMaximumHeight(150)
        progress_layout.addWidget(self.progress_text)
        
        self.progress_group.setLayout(progress_layout)
        layout.addWidget(self.progress_group)
        
        # Buttons
        button_layout = QHBoxLayout()
        self.start_button = QPushButton("Start Training")
        self.start_button.clicked.connect(self.start_training)
        button_layout.addWidget(self.start_button)
        
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.clicked.connect(self.reject)
        button_layout.addWidget(self.cancel_button)
        
        layout.addLayout(button_layout)
        
        self.setLayout(layout)
    
    def _calculate_iterations(self, num_images: int) -> int:
        """Calculate suggested iterations"""
        images_per_epoch = num_images / 2  # batch_size = 2
        
        if num_images < 50:
            target_epochs = 50
        elif num_images < 200:
            target_epochs = 30
        else:
            target_epochs = 20
        
        max_iter = int(images_per_epoch * target_epochs)
        max_iter = max(100, min(5000, max_iter))
        max_iter = (max_iter // 100) * 100
        
        return max_iter
    
    def browse_weights(self):
        """Browse for custom checkpoint"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Checkpoint",
            str(self.project_path),
            "PyTorch Checkpoints (*.pth *.pkl);;All Files (*)"
        )
        
        if file_path:
            self.weights_path = Path(file_path)
    
    def get_config(self) -> dict:
        """Get training configuration"""
        # Map architecture selection
        arch_map = {
            0: "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml",
            1: "COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml",
            2: "COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml"
        }
        
        config = {
            'architecture': arch_map[self.arch_combo.currentIndex()],
            'pretrained_weights': self.weights_path if self.weights_combo.currentIndex() == 1 else None,
            'learning_rate': float(self.lr_spin.value()),
            'batch_size': int(self.batch_spin.value()),
            'max_iter': int(self.iter_spin.value()),
            'val_period': int(self.val_spin.value()),
            'augmentation': {
                'flip_horizontal': self.aug_flip_h.isChecked(),
                'flip_vertical': self.aug_flip_v.isChecked(),
                'crop': self.aug_crop.isChecked(),
                'crop_size': self.crop_slider.value() / 100.0,
                'rotation': self.aug_rotation.isChecked(),
                'brightness': self.aug_brightness.isChecked(),
                'color_jitter': self.aug_color.isChecked()
            }
        }
        
        return config
    
    def start_training(self):
        """Start training process"""
        self.progress_group.setVisible(True)
        self.start_button.setEnabled(False)
        self.cancel_button.setEnabled(False)
        self.progress_text.append("Configuration accepted!")
        self.progress_text.append("")
        
        # Accept dialog but keep it visible
        super().accept()
    
    def update_progress(self, iteration: int, max_iter: int, loss: float):
        """Update progress bar and text"""
        progress = int((iteration / max_iter) * 100)
        self.progress_bar.setValue(progress)
        self.progress_text.append(f"Iteration {iteration}/{max_iter}: loss={loss:.4f}")
        
        # Auto-scroll to bottom
        self.progress_text.verticalScrollBar().setValue(
            self.progress_text.verticalScrollBar().maximum()
        )
    
    def training_finished(self, success: bool, message: str):
        """Handle training completion"""
        self.progress_text.append(f"\n{'SUCCESS' if success else 'FAILED'}: {message}")
        self.start_button.setEnabled(True)
        self.cancel_button.setEnabled(True)
        self.cancel_button.setText("Close")
        
        # Don't auto-close - let user review results
        # if success:
        #     self.accept()
