"""
Annotation toolbar with tools and controls
"""

from PyQt6.QtWidgets import (QWidget, QHBoxLayout, QVBoxLayout, QToolButton, QButtonGroup,
                             QSlider, QLabel, QSpinBox, QComboBox)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QIcon


class AnnotationToolbar(QWidget):
    """Toolbar for annotation tools"""
    
    tool_changed = pyqtSignal(str)
    brush_size_changed = pyqtSignal(int)
    mask_opacity_changed = pyqtSignal(int)
    clear_instance_requested = pyqtSignal()
    new_instance_requested = pyqtSignal()
    resolve_overlaps_requested = pyqtSignal()
    
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
        
        row1.addStretch()
        main_layout.addLayout(row1)
        
        # Second row: Controls and action buttons
        row2 = QHBoxLayout()
        row2.setSpacing(5)
        
        # Brush size control
        row2.addWidget(QLabel("Brush Size:"))
        self.brush_size_slider = QSlider(Qt.Orientation.Horizontal)
        self.brush_size_slider.setMinimum(1)
        self.brush_size_slider.setMaximum(50)
        self.brush_size_slider.setValue(10)
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
        self.opacity_slider.setValue(128)
        self.opacity_slider.setMinimumWidth(120)
        self.opacity_slider.valueChanged.connect(self.on_opacity_changed)
        row2.addWidget(self.opacity_slider)
        
        self.opacity_label = QLabel("50%")
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
        
        self.resolve_overlaps_btn = QToolButton()
        self.resolve_overlaps_btn.setText("Resolve Overlaps")
        self.resolve_overlaps_btn.setToolTip("Remove overlapping pixels from all instances (later instances have priority)")
        self.resolve_overlaps_btn.clicked.connect(self.on_resolve_overlaps)
        row2.addWidget(self.resolve_overlaps_btn)
        
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
        """Handle brush size change"""
        self.brush_size_label.setText(str(value))
        self.brush_size_changed.emit(value)
    
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
    
    def on_resolve_overlaps(self):
        """Handle resolve overlaps button"""
        self.resolve_overlaps_requested.emit()
        
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
