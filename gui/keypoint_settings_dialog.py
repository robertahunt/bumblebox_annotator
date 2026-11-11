"""Project setting: whether to use keypoints, and which ones."""

from PyQt6.QtWidgets import (QCheckBox, QComboBox, QDialog, QDialogButtonBox, QFormLayout,
                             QGroupBox, QLabel, QLineEdit, QMessageBox, QPlainTextEdit,
                             QPushButton, QVBoxLayout)

from core import keypoints as kp

# Explicit line breaks throughout: Qt sizes windows with word-wrapped labels too short,
# which squeezed the keypoint list into the text below it.
NOTES = (
    'Saved in annotations/project.json, like the annotation policy.<br>'
    'Contributors get it with the next project update.',
    'Change it between assignment rounds: after a settings change, work that<br>'
    'contributors already have can only be submitted for your review.',
    'Saved keypoints are never deleted here. Turning keypoints off or<br>'
    'removing a name only hides those points.',
)


class KeypointSettingsDialog(QDialog):
    """Edit the project's keypoint configuration. ``result_config()`` after accept()."""

    def __init__(self, stored_config=None, enabled=False, parent=None):
        super().__init__(parent)
        self.setWindowTitle('Keypoints')
        self.setMinimumWidth(680)
        config = stored_config or kp.default_config()
        self._result = None

        layout = QVBoxLayout(self)
        layout.setSpacing(10)

        intro = QLabel(
            'Keypoints are optional landmarks on each bee, such as the head, thorax,<br>'
            'abdomen tip and antenna tips, saved with the bee\'s mask.<br>'
            'Projects that leave this off are unchanged.')
        layout.addWidget(intro)

        self.enabled_checkbox = QCheckBox('Use keypoints in this project')
        self.enabled_checkbox.setChecked(bool(enabled))
        layout.addWidget(self.enabled_checkbox)

        form = QFormLayout()
        form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.DontWrapRows)
        form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        form.setVerticalSpacing(8)

        self.names_edit = QPlainTextEdit('\n'.join(config.names))
        line = self.names_edit.fontMetrics().lineSpacing()
        self.names_edit.setMinimumHeight(line * 7 + 12)
        self.names_edit.setToolTip('One keypoint per line, in the order you will usually click them')
        form.addRow('Keypoints:', self.names_edit)
        names_help = QLabel('One per line, in clicking order.<br>'
                            'Left and right mean the bee\'s own left and right.<br>'
                            'Skip points you can\'t see clearly.')
        form.addRow('', names_help)

        self.edges_edit = QLineEdit(kp.format_edges(config.edges))
        self.edges_edit.setToolTip('Lines drawn between keypoints, e.g. head-thorax, thorax-abdomen_tip')
        form.addRow('Skeleton lines:', self.edges_edit)

        self.marker_combo = QComboBox()
        self.marker_combo.setToolTip('Keypoint that "from ArUco Tags" sets to each tag centre')
        form.addRow('Fill from ArUco tag:', self.marker_combo)
        layout.addLayout(form)
        self.names_edit.textChanged.connect(self._refresh_marker_choices)
        self._refresh_marker_choices(selected=config.marker_keypoint)

        self.defaults_btn = QPushButton('Restore Bee Defaults')
        self.defaults_btn.setToolTip(', '.join(kp.DEFAULT_KEYPOINT_NAMES))
        self.defaults_btn.clicked.connect(self.restore_defaults)
        layout.addWidget(self.defaults_btn)

        notes = QGroupBox('Good to know')
        notes_layout = QVBoxLayout(notes)
        for text in NOTES:
            label = QLabel('• ' + text)
            notes_layout.addWidget(label)
        layout.addWidget(notes)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Save |
                                   QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self._accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def restore_defaults(self):
        default = kp.default_config()
        self.names_edit.setPlainText('\n'.join(default.names))
        self.edges_edit.setText(kp.format_edges(default.edges))
        self._refresh_marker_choices(selected=default.marker_keypoint)

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
