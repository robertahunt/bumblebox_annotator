"""
Various dialog windows
"""

from PyQt6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel,
                             QLineEdit, QPushButton, QFileDialog, QSpinBox,
                             QCheckBox, QComboBox, QGroupBox, QFormLayout)
from PyQt6.QtCore import Qt
from pathlib import Path


class VideoImportDialog(QDialog):
    """Dialog for video import settings"""
    
    def __init__(self, parent=None, video_path=None):
        super().__init__(parent)
        self.video_path = video_path
        self.setWindowTitle("Import Video Settings")
        self.setModal(True)
        
        self.init_ui()
        
    def init_ui(self):
        """Initialize UI"""
        layout = QVBoxLayout(self)
        
        # Video info
        if self.video_path:
            info_label = QLabel(f"Video: {Path(self.video_path).name}")
            layout.addWidget(info_label)
            
        # Frame extraction settings
        group = QGroupBox("Frame Extraction")
        form = QFormLayout()
        
        self.fps_spin = QSpinBox()
        self.fps_spin.setMinimum(1)
        self.fps_spin.setMaximum(60)
        self.fps_spin.setValue(10)
        self.fps_spin.setSuffix(" fps")
        form.addRow("Extract at:", self.fps_spin)
        
        self.start_frame_spin = QSpinBox()
        self.start_frame_spin.setMinimum(0)
        self.start_frame_spin.setMaximum(1000000)
        self.start_frame_spin.setValue(0)
        form.addRow("Start frame:", self.start_frame_spin)
        
        self.end_frame_spin = QSpinBox()
        self.end_frame_spin.setMinimum(0)
        self.end_frame_spin.setMaximum(1000000)
        self.end_frame_spin.setValue(0)
        self.end_frame_spin.setSpecialValueText("End")
        form.addRow("End frame:", self.end_frame_spin)
        
        self.resize_check = QCheckBox("Resize frames")
        form.addRow("", self.resize_check)
        
        self.width_spin = QSpinBox()
        self.width_spin.setMinimum(100)
        self.width_spin.setMaximum(4000)
        self.width_spin.setValue(1280)
        self.width_spin.setEnabled(False)
        form.addRow("Width:", self.width_spin)
        
        self.height_spin = QSpinBox()
        self.height_spin.setMinimum(100)
        self.height_spin.setMaximum(4000)
        self.height_spin.setValue(720)
        self.height_spin.setEnabled(False)
        form.addRow("Height:", self.height_spin)
        
        self.resize_check.toggled.connect(self.width_spin.setEnabled)
        self.resize_check.toggled.connect(self.height_spin.setEnabled)
        
        group.setLayout(form)
        layout.addWidget(group)
        
        # Buttons
        button_layout = QHBoxLayout()
        
        ok_btn = QPushButton("OK")
        ok_btn.clicked.connect(self.accept)
        button_layout.addWidget(ok_btn)
        
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        button_layout.addWidget(cancel_btn)
        
        layout.addLayout(button_layout)
        
    def get_settings(self):
        """Get import settings"""
        settings = {
            'fps': self.fps_spin.value(),
            'start_frame': self.start_frame_spin.value(),
            'end_frame': self.end_frame_spin.value() if self.end_frame_spin.value() > 0 else None,
        }
        
        if self.resize_check.isChecked():
            settings['resize'] = (self.width_spin.value(), self.height_spin.value())
            
        return settings


class ProjectDialog(QDialog):
    """Dialog for creating/opening projects"""
    
    def __init__(self, parent=None, mode='new', default_dir=None):
        super().__init__(parent)
        self.mode = mode
        self.default_dir = default_dir
        self.setWindowTitle("New Project" if mode == 'new' else "Open Project")
        self.setModal(True)
        
        self.init_ui()
        
    def init_ui(self):
        """Initialize UI"""
        layout = QVBoxLayout(self)
        
        form = QFormLayout()
        
        # Project name
        self.name_edit = QLineEdit()
        self.name_edit.setPlaceholderText("My Bee Project")
        form.addRow("Project Name:", self.name_edit)
        
        # Project path
        path_layout = QHBoxLayout()
        self.path_edit = QLineEdit()
        # Set default path if provided
        if self.default_dir:
            from pathlib import Path
            self.path_edit.setText(str(Path(self.default_dir)))
        else:
            self.path_edit.setPlaceholderText("/path/to/project")
        path_layout.addWidget(self.path_edit)
        
        browse_btn = QPushButton("Browse...")
        browse_btn.clicked.connect(self.browse_path)
        path_layout.addWidget(browse_btn)
        
        form.addRow("Location:", path_layout)
        
        layout.addLayout(form)
        
        # Buttons
        button_layout = QHBoxLayout()
        
        ok_btn = QPushButton("Create" if self.mode == 'new' else "Open")
        ok_btn.clicked.connect(self.accept)
        button_layout.addWidget(ok_btn)
        
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        button_layout.addWidget(cancel_btn)
        
        layout.addLayout(button_layout)
        
    def browse_path(self):
        """Browse for project path"""
        start_dir = str(self.default_dir) if self.default_dir else ""
        path = QFileDialog.getExistingDirectory(
            self, "Select Project Directory", start_dir
        )
        if path:
            self.path_edit.setText(path)
            
    def get_project_info(self):
        """Get project information"""
        info = {
            'name': self.name_edit.text(),
            'path': self.path_edit.text(),
        }
        
        return info


class TrainingDialog(QDialog):
    """Dialog for training configuration"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Training Configuration")
        self.setModal(True)
        self.setMinimumWidth(400)
        
        self.init_ui()
        
    def init_ui(self):
        """Initialize UI"""
        layout = QVBoxLayout(self)
        
        # Training parameters
        group = QGroupBox("Training Parameters")
        form = QFormLayout()
        
        self.epochs_spin = QSpinBox()
        self.epochs_spin.setMinimum(1)
        self.epochs_spin.setMaximum(1000)
        self.epochs_spin.setValue(100)
        form.addRow("Max Epochs:", self.epochs_spin)
        
        self.batch_size_spin = QSpinBox()
        self.batch_size_spin.setMinimum(1)
        self.batch_size_spin.setMaximum(64)
        self.batch_size_spin.setValue(8)
        form.addRow("Batch Size:", self.batch_size_spin)
        
        self.lr_combo = QComboBox()
        self.lr_combo.addItems(["0.0001", "0.001", "0.01"])
        self.lr_combo.setCurrentIndex(1)
        self.lr_combo.setEditable(True)
        form.addRow("Learning Rate:", self.lr_combo)
        
        self.val_split_spin = QSpinBox()
        self.val_split_spin.setMinimum(0)
        self.val_split_spin.setMaximum(50)
        self.val_split_spin.setValue(20)
        self.val_split_spin.setSuffix("%")
        form.addRow("Validation Split:", self.val_split_spin)
        
        group.setLayout(form)
        layout.addWidget(group)
        
        # Human-in-the-loop settings
        hitl_group = QGroupBox("Human-in-the-Loop")
        hitl_form = QFormLayout()
        
        self.auto_retrain_check = QCheckBox("Auto-retrain on corrections")
        self.auto_retrain_check.setChecked(True)
        hitl_form.addRow("", self.auto_retrain_check)
        
        self.retrain_interval_spin = QSpinBox()
        self.retrain_interval_spin.setMinimum(1)
        self.retrain_interval_spin.setMaximum(100)
        self.retrain_interval_spin.setValue(10)
        self.retrain_interval_spin.setSuffix(" corrections")
        hitl_form.addRow("Retrain after:", self.retrain_interval_spin)
        
        hitl_group.setLayout(hitl_form)
        layout.addWidget(hitl_group)
        
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
        return {
            'epochs': self.epochs_spin.value(),
            'batch_size': self.batch_size_spin.value(),
            'learning_rate': float(self.lr_combo.currentText()),
            'val_split': self.val_split_spin.value() / 100.0,
            'auto_retrain': self.auto_retrain_check.isChecked(),
            'retrain_interval': self.retrain_interval_spin.value()
        }


class EditBeeIdDialog(QDialog):
    """Dialog for editing bee instance ID with option to propagate through video"""
    
    def __init__(self, parent=None, current_id=None, existing_ids=None, has_next_frames=False):
        super().__init__(parent)
        self.current_id = current_id
        self.existing_ids = existing_ids or []
        self.has_next_frames = has_next_frames
        self.propagate_through_video = False  # Flag to indicate if user wants to propagate
        
        self.setWindowTitle("Edit Bee ID")
        self.setModal(True)
        
        self.init_ui()
        
    def init_ui(self):
        """Initialize UI"""
        layout = QVBoxLayout(self)
        
        # Info label
        info_label = QLabel(f"Current Bee ID: {self.current_id}")
        layout.addWidget(info_label)
        
        # ID input
        id_layout = QHBoxLayout()
        id_layout.addWidget(QLabel("New Bee ID:"))
        
        self.id_spin = QSpinBox()
        self.id_spin.setMinimum(1)
        self.id_spin.setMaximum(9999)
        self.id_spin.setValue(self.current_id)
        id_layout.addWidget(self.id_spin)
        
        layout.addLayout(id_layout)
        
        # Buttons
        button_layout = QVBoxLayout()
        
        # OK button (update current frame only)
        ok_btn = QPushButton("Update Current Frame Only")
        ok_btn.setDefault(True)
        ok_btn.clicked.connect(self.accept)
        button_layout.addWidget(ok_btn)
        
        # Propagate button (update current + all subsequent frames)
        if self.has_next_frames:
            propagate_btn = QPushButton("Update Current Frame + All Subsequent Frames")
            propagate_btn.clicked.connect(self.accept_with_propagation)
            button_layout.addWidget(propagate_btn)
        
        # Cancel button
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        button_layout.addWidget(cancel_btn)
        
        layout.addLayout(button_layout)
        
    def accept_with_propagation(self):
        """Accept dialog and set flag to propagate through video"""
        self.propagate_through_video = True
        self.accept()
        
    def get_new_id(self):
        """Get the new bee ID"""
        return self.id_spin.value()
    
    def should_propagate(self):
        """Return whether to propagate through video"""
        return self.propagate_through_video
