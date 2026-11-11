"""Toolbar for the optional keypoint workflow (hidden unless the project uses keypoints)."""

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import QCheckBox, QComboBox, QLabel, QPushButton, QToolButton, QWidget

from core import keypoints as kp
from .toolbar import FlowLayout


class KeypointToolbar(QWidget):
    """Place Keypoints tool, next-keypoint chooser, and per-frame keypoint actions."""

    tool_changed = pyqtSignal(str)
    settings_requested = pyqtSignal()

    HELP = (
        "Place Keypoints: click a bee, then click its keypoints in order (the next one is "
        "shown here and advances automatically). Drag a point to move it. Right-click a point "
        "to mark it hidden/visible or delete it. Esc finishes the bee, Backspace removes the "
        "hovered or last point, N skips to the next keypoint name, H toggles 'place as hidden'. "
        "Ctrl+Z / Ctrl+Shift+Z undo and redo keypoint edits on this frame."
    )

    def __init__(self, layer, parent=None):
        super().__init__(parent)
        self.layer = layer
        layout = FlowLayout(self, margin=5, hspacing=10, vspacing=4)

        layout.addWidget(QLabel("<b>Keypoints:</b>"))

        self.place_btn = QToolButton()
        self.place_btn.setText("Place Keypoints")
        self.place_btn.setCheckable(True)
        self.place_btn.setToolTip(self.HELP)
        self.place_btn.clicked.connect(self._on_place_clicked)
        layout.addWidget(self.place_btn)

        layout.addWidget(QLabel("Next:"))
        self.next_combo = QComboBox()
        self.next_combo.setToolTip("Keypoint placed by the next click (advances automatically)")
        self.next_combo.activated.connect(self._on_next_chosen)
        layout.addWidget(self.next_combo)

        self.hidden_checkbox = QCheckBox("Place as hidden")
        self.hidden_checkbox.setToolTip(
            "Mark new points as labelled but not visible (occluded), e.g. a head under "
            "another bee. Shortcut: H")
        self.hidden_checkbox.toggled.connect(self.layer.set_place_occluded)
        layout.addWidget(self.hidden_checkbox)

        self.show_checkbox = QCheckBox("Show keypoints")
        self.show_checkbox.setChecked(True)
        self.show_checkbox.toggled.connect(self.layer.set_visible)
        layout.addWidget(self.show_checkbox)

        self.marker_btn = QPushButton("Thorax from ArUco Tags")
        self.marker_btn.clicked.connect(self._on_fill_from_markers)
        layout.addWidget(self.marker_btn)

        self.clear_btn = QPushButton("Clear Bee's Keypoints")
        self.clear_btn.setToolTip("Remove every keypoint of the bee being labelled")
        self.clear_btn.clicked.connect(lambda: self.layer.clear_instance(self.layer.target_id))
        layout.addWidget(self.clear_btn)

        self.settings_btn = QPushButton("Keypoint Settings...")
        self.settings_btn.setToolTip("Choose this project's keypoints, or turn keypoints off")
        self.settings_btn.clicked.connect(self.settings_requested.emit)
        layout.addWidget(self.settings_btn)

        self.status_label = QLabel()
        self.status_label.setMinimumWidth(320)
        layout.addWidget(self.status_label)

        self.layer.state_changed.connect(self.refresh)
        self.refresh()

    # Interface shared with the other tool toolbars.
    def uncheck_tools(self):
        self.place_btn.blockSignals(True)
        self.place_btn.setChecked(False)
        self.place_btn.blockSignals(False)

    def set_tool_checked(self, checked):
        self.place_btn.blockSignals(True)
        self.place_btn.setChecked(bool(checked))
        self.place_btn.blockSignals(False)

    def _on_place_clicked(self, checked):
        self.tool_changed.emit('keypoint' if checked else 'pan')

    def _on_next_chosen(self, index):
        name = self.next_combo.itemData(index)
        if name:
            self.layer.set_active_name(name)

    def _on_fill_from_markers(self):
        count = self.layer.fill_from_markers(only_missing=True)
        name = self.layer.config.marker_keypoint if self.layer.config else 'thorax'
        self.status_label.setText(
            f"Set {name} from ArUco tags for {count} bee{'s' if count != 1 else ''}"
            if count else f"No bees with an ArUco tag and a missing {name}")

    def refresh(self):
        layer = self.layer
        config = layer.config
        enabled = config is not None
        for widget in (self.place_btn, self.next_combo, self.hidden_checkbox, self.show_checkbox,
                       self.clear_btn):
            widget.setEnabled(enabled)

        names = list(config.names) if config else []
        current = [self.next_combo.itemData(i) for i in range(self.next_combo.count())]
        if current != names:
            self.next_combo.blockSignals(True)
            self.next_combo.clear()
            for name in names:
                self.next_combo.addItem(name.replace('_', ' '), name)
            self.next_combo.blockSignals(False)
        if layer.active_name in names:
            self.next_combo.blockSignals(True)
            self.next_combo.setCurrentIndex(names.index(layer.active_name))
            self.next_combo.blockSignals(False)

        if self.hidden_checkbox.isChecked() != layer.place_occluded:
            self.hidden_checkbox.blockSignals(True)
            self.hidden_checkbox.setChecked(layer.place_occluded)
            self.hidden_checkbox.blockSignals(False)

        marker_name = config.marker_keypoint if config else None
        self.marker_btn.setVisible(bool(marker_name))
        if marker_name:
            self.marker_btn.setText(f"{marker_name.replace('_', ' ').title()} from ArUco Tags")
            self.marker_btn.setToolTip(
                f"Set {marker_name} to the ArUco tag centre for every bee on this frame that has a "
                f"detected tag and no {marker_name} yet (one undo step)")
        self.marker_btn.setEnabled(enabled and bool(marker_name))
        self.clear_btn.setEnabled(enabled and bool(layer.target_id))

        if not enabled:
            self.status_label.setText("Keypoints are off for this project")
        elif layer.target_id:
            done = kp.summary_text(layer.keypoints_for(layer.target_id), config.names)
            nxt = layer.active_name.replace('_', ' ') if layer.active_name else 'all placed'
            self.status_label.setText(f"Bee {layer.target_id}: {done}   |   next: {nxt}")
        else:
            labelled = layer.labelled_instance_count()
            self.status_label.setText(
                f"Click a bee to start   |   {labelled} bee{'s' if labelled != 1 else ''} "
                f"with keypoints on this frame")
