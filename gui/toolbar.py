"""
Annotation toolbar with tools and controls
"""

from core.categories import BROOD_CATEGORIES, BROOD_DESCRIPTIONS, CATEGORY_LABELS, category_label

import math
from PyQt6.QtWidgets import (QWidget, QHBoxLayout, QVBoxLayout, QToolButton, QButtonGroup,
                             QSlider, QLabel, QSpinBox, QCheckBox, QMenu, QComboBox)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QAction, QIcon


class AnnotationToolbar(QWidget):
    """Toolbar for annotation tools"""
    
    tool_changed = pyqtSignal(str)
    brush_size_changed = pyqtSignal(int)
    brush_cursor_preview_changed = pyqtSignal(bool)
    mask_opacity_changed = pyqtSignal(int)
    clear_instance_requested = pyqtSignal()
    new_instance_requested = pyqtSignal(str)
    delete_all_requested = pyqtSignal()
    detect_aruco_requested = pyqtSignal()
    clear_all_aruco_requested = pyqtSignal()
    show_segmentations_changed = pyqtSignal(bool)
    show_bboxes_changed = pyqtSignal(bool)
    annotation_type_changed = pyqtSignal(str)  # Annotation type selection (bee/hive/chamber)
    annotation_type_visibility_changed = pyqtSignal(str, bool)  # annotation_type, visible
    clear_no_draw_zones_requested = pyqtSignal()
    set_measurement_scale_requested = pyqtSignal()
    clear_measurement_scale_requested = pyqtSignal()
    clear_measurement_line_requested = pyqtSignal()
    imaging_setup_changed = pyqtSignal(str)
    apply_imaging_setup_requested = pyqtSignal(str)
    new_imaging_setup_requested = pyqtSignal()
    
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

        self.no_draw_zone_btn = self.create_tool_button("No Draw Zone", "no_draw_zone")
        self.no_draw_zone_btn.setToolTip(
            "Protect a polygon from Brush and Eraser strokes: click vertices, "
            "double-click or press Enter to close"
        )
        row1.addWidget(self.no_draw_zone_btn)

        self.measure_btn = self.create_tool_button("Measure", "measure")
        self.measure_btn.setToolTip(
            "Click and drag a line to measure its length in image pixels and, "
            "after calibration, centimeters"
        )
        row1.addWidget(self.measure_btn)
        
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
        
        self.show_pollen_checkbox = QCheckBox("Pollen")
        self.show_pollen_checkbox.setChecked(False)
        self.show_pollen_checkbox.setToolTip("Show/hide pollen annotations (video-level)")
        self.show_pollen_checkbox.setStyleSheet("QCheckBox { background-color: rgba(255, 165, 0, 50); padding: 2px; }")
        self.show_pollen_checkbox.stateChanged.connect(lambda state: self.on_annotation_type_visibility_changed('pollen', state))
        row1.addWidget(self.show_pollen_checkbox)
        
        self.brood_visibility_actions = {}
        self.brood_visibility_button = QToolButton()
        self.brood_visibility_button.setText("Brood")
        self.brood_visibility_button.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        brood_menu = QMenu(self.brood_visibility_button)
        for category in BROOD_CATEGORIES:
            action = brood_menu.addAction(category_label(category))
            action.setToolTip(BROOD_DESCRIPTIONS[category])
            action.setCheckable(True)
            action.setChecked(False)
            action.toggled.connect(
                lambda checked, cat=category: self.annotation_type_visibility_changed.emit(cat, checked))
            self.brood_visibility_actions[category] = action
        self.brood_visibility_button.setMenu(brood_menu)
        row1.addWidget(self.brood_visibility_button)

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
        
        # Brush size control (log_2 scale up to 1000)
        row2.addWidget(QLabel("Brush Size:"))
        self.brush_size_slider = QSlider(Qt.Orientation.Horizontal)
        self.brush_size_slider.setMinimum(0)
        self.brush_size_slider.setMaximum(100)
        self.brush_size_slider.setValue(self._brush_size_to_slider_value(10))
        self.brush_size_slider.setMinimumWidth(120)
        self.brush_size_slider.valueChanged.connect(self.on_brush_size_changed)
        row2.addWidget(self.brush_size_slider)
        
        self.brush_size_label = QLabel("10")
        self.brush_size_label.setMinimumWidth(25)
        row2.addWidget(self.brush_size_label)

        self.brush_cursor_checkbox = QCheckBox("Brush cursor")
        self.brush_cursor_checkbox.setChecked(True)
        self.brush_cursor_checkbox.setToolTip(
            "Show a brush-size cursor matching the stroke diameter"
        )
        self.brush_cursor_checkbox.stateChanged.connect(self.on_brush_cursor_preview_changed)
        row2.addWidget(self.brush_cursor_checkbox)

        self.clear_no_draw_zones_btn = QToolButton()
        self.clear_no_draw_zones_btn.setText("Clear Zones")
        self.clear_no_draw_zones_btn.setToolTip(
            "Remove every No Draw Zone from the current frame"
        )
        self.clear_no_draw_zones_btn.clicked.connect(
            self.clear_no_draw_zones_requested.emit
        )
        row2.addWidget(self.clear_no_draw_zones_btn)
        
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
        self.new_instance_btn.setToolTip("Choose object type for a new instance")
        self.new_instance_btn.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        self.new_instance_menu = QMenu(self.new_instance_btn)
        for category, label in CATEGORY_LABELS.items():
            action = QAction(label, self.new_instance_menu)
            if category in BROOD_DESCRIPTIONS:
                action.setToolTip(BROOD_DESCRIPTIONS[category])
            action.triggered.connect(
                lambda checked=False, cat=category: self.new_instance_requested.emit(cat)
            )
            self.new_instance_menu.addAction(action)
        self.new_instance_btn.setMenu(self.new_instance_menu)
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

        # Third row: physical-distance calibration and measurement controls.
        row3 = QHBoxLayout()
        row3.setSpacing(5)
        row3.addWidget(QLabel("Imaging setup:"))

        self.imaging_setup_combo = QComboBox()
        self.imaging_setup_combo.setMinimumWidth(180)
        self.imaging_setup_combo.setToolTip(
            "Choose an imaging setup for the selected video or videos. Videos "
            "assigned to the same setup share one pixel-to-centimeter calibration."
        )
        self.imaging_setup_combo.currentIndexChanged.connect(
            self.on_imaging_setup_changed
        )
        row3.addWidget(self.imaging_setup_combo)

        self.apply_imaging_setup_btn = QToolButton()
        self.apply_imaging_setup_btn.setText("Apply to Selected")
        self.apply_imaging_setup_btn.setToolTip(
            "Apply the setup shown in the list to every selected sidebar video"
        )
        self.apply_imaging_setup_btn.clicked.connect(
            self.on_apply_imaging_setup
        )
        row3.addWidget(self.apply_imaging_setup_btn)

        self.new_imaging_setup_btn = QToolButton()
        self.new_imaging_setup_btn.setText("New Setup")
        self.new_imaging_setup_btn.setToolTip(
            "Create a named imaging setup and assign all selected videos to it"
        )
        self.new_imaging_setup_btn.clicked.connect(
            self.new_imaging_setup_requested.emit
        )
        row3.addWidget(self.new_imaging_setup_btn)

        row3.addWidget(self.create_separator())
        row3.addWidget(QLabel("Physical scale:"))

        self.measurement_scale_label = QLabel("Not calibrated (pixels only)")
        self.measurement_scale_label.setMinimumWidth(270)
        self.measurement_scale_label.setToolTip(
            "Physical scale for the current video. Draw a Measure line over an "
            "object of known length, then choose Set Scale from Line."
        )
        row3.addWidget(self.measurement_scale_label)

        self.set_measurement_scale_btn = QToolButton()
        self.set_measurement_scale_btn.setText("Set Scale from Line")
        self.set_measurement_scale_btn.setToolTip(
            "Use the most recently drawn Measure line as a known physical distance"
        )
        self.set_measurement_scale_btn.clicked.connect(
            self.set_measurement_scale_requested.emit
        )
        row3.addWidget(self.set_measurement_scale_btn)

        self.clear_measurement_scale_btn = QToolButton()
        self.clear_measurement_scale_btn.setText("Clear Scale")
        self.clear_measurement_scale_btn.setToolTip(
            "Remove the saved pixel-to-centimeter calibration for the current video"
        )
        self.clear_measurement_scale_btn.clicked.connect(
            self.clear_measurement_scale_requested.emit
        )
        self.clear_measurement_scale_btn.setEnabled(False)
        row3.addWidget(self.clear_measurement_scale_btn)

        self.clear_measurement_line_btn = QToolButton()
        self.clear_measurement_line_btn.setText("Clear Line")
        self.clear_measurement_line_btn.setToolTip(
            "Remove the temporary measurement line from the canvas"
        )
        self.clear_measurement_line_btn.clicked.connect(
            self.clear_measurement_line_requested.emit
        )
        row3.addWidget(self.clear_measurement_line_btn)

        row3.addStretch()
        main_layout.addLayout(row3)

        self.set_imaging_setups([], None)
        
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
        # Convert slider value (0-100) to brush size (1-1000) on log_2 scale
        # brush_size = 2^(slider_value * log2(1000) / 100)
        if value == 0:
            brush_size = 1
        else:
            exponent = value * math.log2(1000) / 100
            brush_size = round(2 ** exponent)
        
        self.brush_size_label.setText(str(brush_size))
        self.brush_size_changed.emit(brush_size)

    def on_brush_cursor_preview_changed(self, state):
        """Handle brush cursor footprint checkbox changes."""
        self.brush_cursor_preview_changed.emit(state == Qt.CheckState.Checked.value)

    def _brush_size_to_slider_value(self, brush_size):
        """Convert a brush size in pixels to the toolbar's log-scale slider value."""
        brush_size = max(1, min(1000, int(brush_size)))
        if brush_size <= 1:
            return 0
        return max(0, min(100, round(100 * math.log2(brush_size) / math.log2(1000))))

    def set_brush_size(self, brush_size, emit=True):
        """Set brush size from code while keeping slider and label in sync."""
        brush_size = max(1, min(1000, int(brush_size)))
        slider_value = self._brush_size_to_slider_value(brush_size)

        self.brush_size_slider.blockSignals(True)
        self.brush_size_slider.setValue(slider_value)
        self.brush_size_slider.blockSignals(False)
        self.brush_size_label.setText(str(brush_size))

        if emit:
            self.brush_size_changed.emit(brush_size)
    
    def on_opacity_changed(self, value):
        """Handle opacity change"""
        percentage = int((value / 255) * 100)
        self.opacity_label.setText(f"{percentage}%")
        self.mask_opacity_changed.emit(value)
        
    def on_clear_instance(self):
        """Handle clear instance button"""
        self.clear_instance_requested.emit()
        
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
        self.annotation_type_changed.emit(self._display_text_to_annotation_type(type_text))

    def _display_text_to_annotation_type(self, type_text):
        """Convert display text to internal annotation type."""
        type_map = {label: category for category, label in CATEGORY_LABELS.items()}
        return type_map.get(type_text, 'bee')
    
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

    def on_imaging_setup_changed(self, index):
        """Emit the selected setup name, or an empty string for unassigned."""
        setup_name = self.imaging_setup_combo.itemData(index)
        self.imaging_setup_changed.emit(str(setup_name or ""))

    def on_apply_imaging_setup(self):
        """Re-emit the displayed setup for an explicit batch assignment."""
        setup_name = self.imaging_setup_combo.currentData()
        self.apply_imaging_setup_requested.emit(str(setup_name or ""))

    def set_imaging_setups(self, setup_names, current_setup=None):
        """Populate the setup selector without changing video assignments."""
        self.imaging_setup_combo.blockSignals(True)
        self.imaging_setup_combo.clear()
        self.imaging_setup_combo.addItem("Video-specific / unassigned", "")
        for setup_name in sorted(setup_names, key=str.casefold):
            self.imaging_setup_combo.addItem(str(setup_name), str(setup_name))

        target_index = 0
        if current_setup:
            for index in range(self.imaging_setup_combo.count()):
                if self.imaging_setup_combo.itemData(index) == current_setup:
                    target_index = index
                    break
        self.imaging_setup_combo.setCurrentIndex(target_index)
        self.imaging_setup_combo.blockSignals(False)

    def set_measurement_scale(self, pixels_per_cm, setup_name=None):
        """Update the physical-scale readout for the current video."""
        self.measurement_scale_label.setToolTip(
            "Physical scale for the current video. Draw a Measure line over an "
            "object of known length, then choose Set Scale from Line."
        )
        if pixels_per_cm is None or pixels_per_cm <= 0:
            if setup_name:
                self.measurement_scale_label.setText("Shared setup not calibrated (pixels only)")
                self.measurement_scale_label.setToolTip(
                    f'Imaging setup "{setup_name}" has no physical calibration yet.'
                )
            else:
                self.measurement_scale_label.setText("Not calibrated (pixels only)")
            self.clear_measurement_scale_btn.setEnabled(False)
            return

        cm_per_pixel = 1.0 / float(pixels_per_cm)
        source_text = "Shared setup: " if setup_name else "Video-specific: "
        self.measurement_scale_label.setText(
            f"{source_text}{float(pixels_per_cm):.3f} px/cm "
            f"(1 px = {cm_per_pixel:.6f} cm)"
        )
        if setup_name:
            self.measurement_scale_label.setToolTip(
                f'Inherited from imaging setup "{setup_name}".'
            )
        self.clear_measurement_scale_btn.setEnabled(True)
    
    def uncheck_all_tools(self):
        """Uncheck all tool buttons without emitting signals"""
        for btn in self.button_group.buttons():
            btn.blockSignals(True)
            btn.setChecked(False)
            btn.blockSignals(False)
