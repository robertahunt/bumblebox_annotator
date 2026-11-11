"""Project setting: whether to use keypoints, and which ones."""

from PyQt6.QtWidgets import (QCheckBox, QComboBox, QDialog, QDialogButtonBox, QFormLayout, QLabel,
                             QLineEdit, QMessageBox, QPlainTextEdit, QVBoxLayout)

from core import keypoints as kp


class KeypointSettingsDialog(QDialog):
    """Edit the project's keypoint configuration. ``result_config()`` after accept()."""

    def __init__(self, stored_config=None, enabled=False, parent=None):
        super().__init__(parent)
        self.setWindowTitle('Keypoints')
        config = stored_config or kp.default_config()
        self._result = None

        layout = QVBoxLayout(self)
        intro = QLabel(
            'Keypoints are optional landmarks on each bee (for example head, thorax and '
            'abdomen tip), saved with its mask. Projects that leave this off are unchanged.')
        intro.setWordWrap(True)
        layout.addWidget(intro)

        self.enabled_checkbox = QCheckBox('Use keypoints in this project')
        self.enabled_checkbox.setChecked(bool(enabled))
        layout.addWidget(self.enabled_checkbox)

        form = QFormLayout()
        self.names_edit = QPlainTextEdit('\n'.join(config.names))
        self.names_edit.setToolTip('One keypoint per line, in the order you will usually click them')
        self.names_edit.setFixedHeight(110)
        form.addRow('Keypoints (in order):', self.names_edit)
        self.edges_edit = QLineEdit(kp.format_edges(config.edges))
        self.edges_edit.setToolTip('Lines drawn between keypoints, e.g. head-thorax, thorax-abdomen_tip')
        form.addRow('Skeleton lines:', self.edges_edit)
        self.marker_combo = QComboBox()
        form.addRow('Fill from ArUco tag:', self.marker_combo)
        layout.addLayout(form)
        self.names_edit.textChanged.connect(self._refresh_marker_choices)
        self._refresh_marker_choices(selected=config.marker_keypoint)

        note = QLabel(
            'This edits annotations/project.json, like the annotation policy. Contributors get it '
            'with the next project update. Work they already have can only be submitted for your '
            'review after a settings change, so change this between assignment rounds. Keypoints '
            'already saved in frames are never deleted here: turning keypoints off or removing a '
            'name only hides those points.')
        note.setWordWrap(True)
        note.setStyleSheet('color: palette(mid);')
        layout.addWidget(note)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Save |
                                   QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self._accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        self.resize(480, 0)

    def _refresh_marker_choices(self, selected=None):
        if selected is None:
            selected = self.marker_combo.currentData()
        try:
            names = kp.parse_names(self.names_edit.toPlainText())
        except ValueError:
            return
        self.marker_combo.blockSignals(True)
        self.marker_combo.clear()
        self.marker_combo.addItem('(none)', None)
        for name in names:
            self.marker_combo.addItem(name, name)
        index = self.marker_combo.findData(selected) if selected else 0
        self.marker_combo.setCurrentIndex(max(index, 0))
        self.marker_combo.blockSignals(False)

    def _accept(self):
        try:
            config = kp.KeypointConfig(
                names=kp.parse_names(self.names_edit.toPlainText()),
                edges=kp.parse_edges(self.edges_edit.text()),
                marker_keypoint=self.marker_combo.currentData(),
            )
        except ValueError as exc:
            QMessageBox.warning(self, 'Keypoints', str(exc))
            return
        self._result = (config, self.enabled_checkbox.isChecked())
        self.accept()

    def result_config(self):
        """``(KeypointConfig, enabled)`` after the dialog was accepted, else None."""
        return self._result
