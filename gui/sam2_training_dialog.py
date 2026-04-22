"""
SAM2 training configuration dialog
"""

from PyQt6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel, 
                             QPushButton, QGroupBox, QFormLayout, QSpinBox, 
                             QComboBox, QCheckBox, QDoubleSpinBox)
from PyQt6.QtCore import Qt


class SAM2TrainingConfigDialog(QDialog):
    """Dialog for configuring SAM2 fine-tuning parameters"""
    
    def __init__(self, parent=None, checkpoint_path=None, config_name=None):
        super().__init__(parent)
        self.checkpoint_path = checkpoint_path
        self.config_name = config_name
        
        self.setWindowTitle("Configure SAM2 Fine-tuning")
        self.setModal(True)
        self.setMinimumWidth(500)
        
        self.init_ui()
        
    def init_ui(self):
        """Initialize UI"""
        layout = QVBoxLayout(self)
        
        # Info label
        info_text = (
            "Configure parameters for fine-tuning SAM2 on your annotated data.\n\n"
            "SAM2 will be trained to better segment objects in your specific domain by:\n"
            "• Training the mask decoder and prompt encoder\n"
            "• Optionally training the image encoder (requires more GPU memory)\n"
            "• Learning from your segmentation masks with point prompts\n"
            "• Using random crops around annotations for efficient training\n\n"
            "Training will use your existing COCO annotations. Make sure you have "
            "annotated some frames first."
        )
        info_label = QLabel(info_text)
        info_label.setWordWrap(True)
        layout.addWidget(info_label)
        
        # Model info (if checkpoint is loaded)
        if self.checkpoint_path and self.config_name:
            from pathlib import Path
            model_info = QLabel(
                f"<b>Current Model:</b> {Path(self.checkpoint_path).name}<br>"
                f"<b>Config:</b> {self.config_name}"
            )
            model_info.setStyleSheet("background-color: #f0f0f0; padding: 5px; border-radius: 3px;")
            layout.addWidget(model_info)
        
        # Training parameters
        params_group = QGroupBox("Training Parameters")
        params_layout = QFormLayout()
        
        # Crop size
        self.crop_size_combo = QComboBox()
        self.crop_size_combo.addItems(['640', '800', '1024'])
        self.crop_size_combo.setCurrentText('1024')
        self.crop_size_combo.setToolTip(
            "Size of random crops extracted from images for training.\n"
            "Larger crops capture more context but use more memory.\n"
            "Default: 1024 (SAM2's maximum input size)"
        )
        params_layout.addRow("Crop Size:", self.crop_size_combo)
        
        # Number of training steps
        self.steps_spin = QSpinBox()
        self.steps_spin.setRange(100, 100000)
        self.steps_spin.setSingleStep(1000)
        self.steps_spin.setValue(25000)
        self.steps_spin.setToolTip(
            "Number of training steps (iterations).\n"
            "Recommended: 25,000 steps for good results.\n"
            "You should see improvement after ~5,000 steps."
        )
        params_layout.addRow("Training Steps:", self.steps_spin)
        
        # Learning rate
        self.lr_combo = QComboBox()
        self.lr_combo.addItems(['1e-6', '5e-6', '1e-5', '5e-5', '1e-4'])
        self.lr_combo.setCurrentText('1e-5')
        self.lr_combo.setToolTip(
            "Learning rate for the optimizer.\n"
            "Default: 1e-5 (recommended by SAM2 authors)"
        )
        params_layout.addRow("Learning Rate:", self.lr_combo)
        
        # Weight decay
        self.weight_decay_combo = QComboBox()
        self.weight_decay_combo.addItems(['1e-5', '4e-5', '1e-4', '4e-4'])
        self.weight_decay_combo.setCurrentText('4e-5')
        self.weight_decay_combo.setToolTip(
            "Weight decay for regularization.\n"
            "Default: 4e-5 (recommended by SAM2 authors)"
        )
        params_layout.addRow("Weight Decay:", self.weight_decay_combo)
        
        # Save interval
        self.save_interval_spin = QSpinBox()
        self.save_interval_spin.setRange(100, 10000)
        self.save_interval_spin.setSingleStep(100)
        self.save_interval_spin.setValue(1000)
        self.save_interval_spin.setToolTip(
            "How often to save checkpoint during training.\n"
            "A checkpoint will also be saved at the end of training."
        )
        params_layout.addRow("Save Every (steps):", self.save_interval_spin)
        
        params_group.setLayout(params_layout)
        layout.addWidget(params_group)
        
        # Advanced options
        advanced_group = QGroupBox("Advanced Options")
        advanced_layout = QVBoxLayout()
        
        # Train image encoder checkbox
        self.train_encoder_checkbox = QCheckBox("Train Image Encoder")
        self.train_encoder_checkbox.setChecked(False)
        self.train_encoder_checkbox.setToolTip(
            "Enable training of the image encoder (largest component).\n"
            "⚠️ WARNING: This requires strong GPU (16GB+ VRAM) and may need code modifications.\n"
            "Only enable if you have sufficient GPU memory.\n"
            "Training just the mask decoder and prompt encoder usually gives good results."
        )
        self.train_encoder_checkbox.setStyleSheet("font-weight: bold; color: #ff6600;")
        advanced_layout.addWidget(self.train_encoder_checkbox)
        
        advanced_group.setLayout(advanced_layout)
        layout.addWidget(advanced_group)
        
        # Data export options
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
        
        # Buttons
        button_layout = QHBoxLayout()
        
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        button_layout.addWidget(cancel_btn)
        
        button_layout.addStretch()
        
        start_btn = QPushButton("Start Fine-tuning")
        start_btn.setDefault(True)
        start_btn.clicked.connect(self.accept)
        start_btn.setStyleSheet(
            "QPushButton { background-color: #4CAF50; color: white; font-weight: bold; padding: 5px 15px; }"
            "QPushButton:hover { background-color: #45a049; }"
        )
        button_layout.addWidget(start_btn)
        
        layout.addLayout(button_layout)
        
    def get_config(self):
        """Get training configuration as dictionary"""
        config = {
            'crop_size': int(self.crop_size_combo.currentText()),
            'num_steps': self.steps_spin.value(),
            'learning_rate': float(self.lr_combo.currentText()),
            'weight_decay': float(self.weight_decay_combo.currentText()),
            'save_interval': self.save_interval_spin.value(),
            'train_image_encoder': self.train_encoder_checkbox.isChecked(),
            'export_coco': self.export_coco_checkbox.isChecked(),
            'checkpoint_path': self.checkpoint_path,
            'config_name': self.config_name
        }
        return config
