"""
Annotation toolbar with tools and controls
"""

import math
from PyQt6.QtWidgets import (QWidget, QHBoxLayout, QVBoxLayout, QToolButton, QButtonGroup,
                             QSlider, QLabel, QSpinBox, QComboBox, QCheckBox)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QIcon


class AnnotationToolbar(QWidget):
    """Toolbar for annotation tools"""
    
    tool_changed = pyqtSignal(str)
    brush_size_changed = pyqtSignal(int)
    mask_opacity_changed = pyqtSignal(int)
    clear_instance_requested = pyqtSignal()
    new_instance_requested = pyqtSignal()
    delete_all_requested = pyqtSignal()
    detect_aruco_requested = pyqtSignal()
    clear_all_aruco_requested = pyqtSignal()
    show_segmentations_changed = pyqtSignal(bool)
    show_bboxes_changed = pyqtSignal(bool)
    annotation_type_changed = pyqtSignal(str)  # Annotation type selection (bee/hive/chamber)
    annotation_type_visibility_changed = pyqtSignal(str, bool)  # annotation_type, visible
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        self.init_ui()
        
    def init_ui(self):
        """Initialize UI with two rows"""
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(5, 5, 5, 5)
        main_layout.setSpacing(5)
        
        # First row: Tool selection buttons
        row1 = QHBoxLayout()
        row1.setSpacing(5)
        
        # Tool buttons
        self.button_group = QButtonGroup(self)
        self.button_group.setExclusive(True)
        
        # Pan tool
        self.pan_btn = self.create_tool_button("Pan", "pan")
        self.pan_btn.setChecked(True)
        row1.addWidget(self.pan_btn)
        
        row1.addWidget(self.create_separator())
        
        # Editing tools
        row1.addWidget(QLabel("Edit:"))
        self.brush_btn = self.create_tool_button("Brush", "brush")
        row1.addWidget(self.brush_btn)
        
        self.eraser_btn = self.create_tool_button("Eraser", "eraser")
        row1.addWidget(self.eraser_btn)
        
        self.bbox_btn = self.create_tool_button("BBox", "bbox")
        self.bbox_btn.setToolTip("Draw or edit bounding box")
        row1.addWidget(self.bbox_btn)
        
        row1.addWidget(self.create_separator())
        
        # Annotation type selection
        row1.addWidget(QLabel("Type:"))
        self.annotation_type_combo = QComboBox()
        self.annotation_type_combo.addItems(["Bee", "Hive", "Chamber"])
        self.annotation_type_combo.setCurrentText("Bee")
        self.annotation_type_combo.setToolTip("Select annotation type (video-level for Hive/Chamber)")
        self.annotation_type_combo.currentTextChanged.connect(self.on_annotation_type_changed)
        row1.addWidget(self.annotation_type_combo)
        
        row1.addWidget(self.create_separator())
        
        # Category visibility toggles
        row1.addWidget(QLabel("Show:"))
        
        self.show_bees_checkbox = QCheckBox("Bees")
        self.show_bees_checkbox.setChecked(True)
        self.show_bees_checkbox.setToolTip("Show/hide bee annotations")
        self.show_bees_checkbox.stateChanged.connect(lambda state: self.on_annotation_type_visibility_changed('bee', state))
        row1.addWidget(self.show_bees_checkbox)
        
        self.show_hives_checkbox = QCheckBox("Hives")
        self.show_hives_checkbox.setChecked(False)
        self.show_hives_checkbox.setToolTip("Show/hide hive annotations (video-level)")
        self.show_hives_checkbox.setStyleSheet("QCheckBox { background-color: rgba(255, 255, 0, 50); padding: 2px; }")
        self.show_hives_checkbox.stateChanged.connect(lambda state: self.on_annotation_type_visibility_changed('hive', state))
        row1.addWidget(self.show_hives_checkbox)
        
        self.show_chambers_checkbox = QCheckBox("Chambers")
        self.show_chambers_checkbox.setChecked(False)
        self.show_chambers_checkbox.setToolTip("Show/hide chamber annotations (video-level)")
        self.show_chambers_checkbox.setStyleSheet("QCheckBox { background-color: rgba(255, 0, 0, 50); padding: 2px; }")
        self.show_chambers_checkbox.stateChanged.connect(lambda state: self.on_annotation_type_visibility_changed('chamber', state))
        row1.addWidget(self.show_chambers_checkbox)
        
        # Keep old checkboxes for backward compatibility with segmentation/bbox view modes
        self.segmentation_checkbox = QCheckBox("Segmentations")
        self.segmentation_checkbox.setChecked(True)
        self.segmentation_checkbox.setToolTip("Show/hide all segmentation masks (Ctrl+Shift+S)")
        self.segmentation_checkbox.stateChanged.connect(self.on_show_segmentations_changed)
        row1.addWidget(self.segmentation_checkbox)
        
        self.bbox_checkbox = QCheckBox("BBoxes")
        self.bbox_checkbox.setChecked(True)
        self.bbox_checkbox.setToolTip("Show/hide bounding boxes (Ctrl+Shift+B)")
        self.bbox_checkbox.stateChanged.connect(self.on_show_bboxes_changed)
        row1.addWidget(self.bbox_checkbox)
        
        row1.addStretch()
        main_layout.addLayout(row1)
        
        # Second row: Controls and action buttons
        row2 = QHBoxLayout()
        row2.setSpacing(5)
        
        # Brush size control (log_2 scale up to 300)
        row2.addWidget(QLabel("Brush Size:"))
        self.brush_size_slider = QSlider(Qt.Orientation.Horizontal)
        self.brush_size_slider.setMinimum(0)
        self.brush_size_slider.setMaximum(100)
        # Default to brush size 10: slider_value = 100 * log2(10) / log2(300) ≈ 40
        self.brush_size_slider.setValue(20)
        self.brush_size_slider.setMinimumWidth(120)
        self.brush_size_slider.valueChanged.connect(self.on_brush_size_changed)
        row2.addWidget(self.brush_size_slider)
        
        self.brush_size_label = QLabel("10")
        self.brush_size_label.setMinimumWidth(25)
        row2.addWidget(self.brush_size_label)
        
        row2.addWidget(self.create_separator())
        
        # Mask opacity control
        row2.addWidget(QLabel("Opacity:"))
        self.opacity_slider = QSlider(Qt.Orientation.Horizontal)
        self.opacity_slider.setMinimum(10)
        self.opacity_slider.setMaximum(255)
        self.opacity_slider.setValue(64)
        self.opacity_slider.setMinimumWidth(120)
        self.opacity_slider.valueChanged.connect(self.on_opacity_changed)
        row2.addWidget(self.opacity_slider)
        
        self.opacity_label = QLabel("25%")
        self.opacity_label.setMinimumWidth(35)
        row2.addWidget(self.opacity_label)
        
        row2.addWidget(self.create_separator())
        
        # Instance controls
        self.clear_instance_btn = QToolButton()
        self.clear_instance_btn.setText("Clear Instance")
        self.clear_instance_btn.setToolTip("Clear selected instance mask and points (C)")
        self.clear_instance_btn.clicked.connect(self.on_clear_instance)
        row2.addWidget(self.clear_instance_btn)
        
        self.new_instance_btn = QToolButton()
        self.new_instance_btn.setText("New Instance")
        self.new_instance_btn.clicked.connect(self.on_new_instance)
        row2.addWidget(self.new_instance_btn)
        
        self.detect_aruco_btn = QToolButton()
        self.detect_aruco_btn.setText("Detect ArUco")
        self.detect_aruco_btn.setToolTip("Detect ArUco markers on bee instances only")
        self.detect_aruco_btn.setStyleSheet("QToolButton { color: blue; font-weight: bold; }")
        self.detect_aruco_btn.clicked.connect(self.on_detect_aruco)
        row2.addWidget(self.detect_aruco_btn)
        
        self.clear_aruco_btn = QToolButton()
        self.clear_aruco_btn.setText("Clear ArUco")
        self.clear_aruco_btn.setToolTip("Remove all ArUco tracking for this video")
        self.clear_aruco_btn.setStyleSheet("QToolButton { color: orange; font-weight: bold; }")
        self.clear_aruco_btn.clicked.connect(self.on_clear_all_aruco)
        row2.addWidget(self.clear_aruco_btn)
        
        self.delete_all_btn = QToolButton()
        self.delete_all_btn.setText("Delete All Bee Instances")
        self.delete_all_btn.setToolTip("Delete all bee instances in the current frame and remove annotations from disk (preserves hive/chamber)")
        self.delete_all_btn.setStyleSheet("QToolButton { color: red; font-weight: bold; }")
        self.delete_all_btn.clicked.connect(self.on_delete_all)
        row2.addWidget(self.delete_all_btn)
        
        row2.addStretch()
        main_layout.addLayout(row2)
        
    def create_tool_button(self, text, tool_name):
        """Create a tool button"""
        btn = QToolButton()
        btn.setText(text)
        btn.setCheckable(True)
        btn.setProperty('tool_name', tool_name)
        btn.clicked.connect(lambda: self.on_tool_clicked(tool_name))
        self.button_group.addButton(btn)
        return btn
        
    def create_separator(self):
        """Create a vertical separator"""
        from PyQt6.QtWidgets import QFrame
        line = QFrame()
        line.setFrameShape(QFrame.Shape.VLine)
        line.setFrameShadow(QFrame.Shadow.Sunken)
        return line
        
    def on_tool_clicked(self, tool_name):
        """Handle tool button click"""
        self.tool_changed.emit(tool_name)
        
    def on_brush_size_changed(self, value):
        """Handle brush size change (converts from log_2 scale)"""
        # Convert slider value (0-100) to brush size (1-300) on log_2 scale
        # brush_size = 2^(slider_value * log2(300) / 100)
        if value == 0:
            brush_size = 1
        else:
            exponent = value * math.log2(1000) / 100
            brush_size = round(2 ** exponent)
        
        self.brush_size_label.setText(str(brush_size))
        self.brush_size_changed.emit(brush_size)
    
    def on_opacity_changed(self, value):
        """Handle opacity change"""
        percentage = int((value / 255) * 100)
        self.opacity_label.setText(f"{percentage}%")
        self.mask_opacity_changed.emit(value)
        
    def on_clear_instance(self):
        """Handle clear instance button"""
        self.clear_instance_requested.emit()
        
    def on_new_instance(self):
        """Handle new instance button"""
        self.new_instance_requested.emit()
    
    def on_detect_aruco(self):
        """Handle detect ArUco button"""
        self.detect_aruco_requested.emit()
    
    def on_clear_all_aruco(self):
        """Handle clear all ArUco button"""
        self.clear_all_aruco_requested.emit()
    
    def on_delete_all(self):
        """Handle delete all button"""
        self.delete_all_requested.emit()
        
    def on_show_segmentations_changed(self, state):
        """Handle show segmentations checkbox"""
        self.show_segmentations_changed.emit(state == Qt.CheckState.Checked.value)
    
    def on_show_bboxes_changed(self, state):
        """Handle show bboxes checkbox"""
        self.show_bboxes_changed.emit(state == Qt.CheckState.Checked.value)
    
    def on_annotation_type_changed(self, type_text):
        """Handle annotation type dropdown selection"""
        # Convert display name to internal name
        type_map = {'Bee': 'bee', 'Hive': 'hive', 'Chamber': 'chamber'}
        annotation_type = type_map.get(type_text, 'bee')
        self.annotation_type_changed.emit(annotation_type)
    
    def on_annotation_type_visibility_changed(self, annotation_type, state):
        """Handle annotation type visibility checkbox"""
        visible = (state == Qt.CheckState.Checked.value)
        self.annotation_type_visibility_changed.emit(annotation_type, visible)
        
    def set_tool(self, tool_name):
        """Set active tool"""
        for btn in self.button_group.buttons():
            if btn.property('tool_name') == tool_name:
                btn.setChecked(True)
                self.tool_changed.emit(tool_name)
                break
    
    def uncheck_all_tools(self):
        """Uncheck all tool buttons without emitting signals"""
        for btn in self.button_group.buttons():
            btn.blockSignals(True)
            btn.setChecked(False)
            btn.blockSignals(False)
