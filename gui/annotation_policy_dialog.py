"""Project label intent and imported-frame review controls."""

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (QAbstractItemView, QCheckBox, QComboBox, QDialog,
                            QDialogButtonBox, QHeaderView, QLabel, QMessageBox,
                            QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget)

from core.annotation_policy import (MEANINGS, SCOPES, category_policies, default_policy,
                                    guidance, policy_text, save_policy, temporal_support)
from core.annotation_scope import read_project_info
from core.categories import CATEGORIES, category_label
from core.import_review import review_rows


class PolicyTable(QWidget):
    def __init__(self, info, parent=None, *, new=False):
        super().__init__(parent)
        self.new = new
        self.info = info
        self.meanings = {}
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.table = QTableWidget(len(CATEGORIES), 4)
        self.table.setHorizontalHeaderLabels(['Category', 'Saved scope', 'Label meaning', 'Temporal analysis'])
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table.verticalHeader().hide()
        self.table.setWordWrap(True)
        self.table.setMinimumHeight(280)
        entries = default_policy(info)['categories'] if new else category_policies(info)
        for row, category in enumerate(CATEGORIES):
            for column, text in ((0, category_label(category)), (1, SCOPES[entries[category]['scope']]),
                                 (3, temporal_support(category))):
                item = QTableWidgetItem(text)
                item.setToolTip(text)
                self.table.setItem(row, column, item)
            combo = QComboBox()
            for key, label in MEANINGS.items():
                combo.addItem(label, key)
            combo.setCurrentIndex(combo.findData(entries[category]['meaning']))
            combo.currentIndexChanged.connect(lambda unused, c=category: self.update_guidance(c))
            self.table.setCellWidget(row, 2, combo)
            self.meanings[category] = combo
        self.table.resizeRowsToContents()
        self.table.cellClicked.connect(lambda row, column: self.update_guidance(CATEGORIES[row]))
        layout.addWidget(self.table)
        self.help = QLabel()
        self.help.setWordWrap(True)
        self.help.setMinimumHeight(48)
        layout.addWidget(self.help)
        self.update_guidance('hive')

    def entries(self):
        entries = category_policies(self.info)
        for category, combo in self.meanings.items():
            entries[category]['meaning'] = combo.currentData()
        return entries

    def update_guidance(self, category):
        if hasattr(self, 'help'):
            self.help.setText(category_label(category) + ': ' + guidance(self.entries()[category]))

    def set_hive_scope(self, scope):
        self.info = {'hive_annotation_scope': scope}
        row = CATEGORIES.index('hive')
        self.table.item(row, 1).setText(SCOPES[scope])
        self.meanings['hive'].setCurrentIndex(self.meanings['hive'].findData(
            'visible' if scope == 'frame' else 'full_extent'))
        self.update_guidance('hive')


class AnnotationPolicyDialog(QDialog):
    policy_saved = pyqtSignal()

    def __init__(self, project, contributor=None, parent=None):
        super().__init__(parent)
        self.project = project
        self.contributor = contributor
        self.setWindowTitle('Annotation Policy')
        self.resize(980, 700)
        layout = QVBoxLayout(self)
        label = QLabel('Project: ' + str(project))
        label.setWordWrap(True)
        label.setTextFormat(Qt.TextFormat.PlainText)
        layout.addWidget(label)
        self.policy = PolicyTable(read_project_info(project), self)
        layout.addWidget(self.policy)
        note = QLabel('Scope is fixed for existing projects. Use Import From Project for scope conversion. '
                      'Temporal maps are batch outputs, not training labels. Changing label meaning does not edit masks.')
        note.setWordWrap(True)
        layout.addWidget(note)
        self.confirm = QCheckBox('These meanings describe my intended labels; imported conflicts still need frame review')
        layout.addWidget(self.confirm)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Close)
        self.save_button = buttons.button(QDialogButtonBox.StandardButton.Save)
        self.save_button.setEnabled(False)
        self.confirm.toggled.connect(self.save_button.setEnabled)
        buttons.accepted.connect(self.save)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def save(self):
        try:
            if not self.confirm.isChecked():
                return
            save_policy(self.project, self.policy.entries(), self.contributor)
            self.policy_saved.emit()
            self.accept()
        except Exception as exc:
            QMessageBox.warning(self, 'Policy Not Saved', str(exc))


class ImportedFrameReviewDialog(QDialog):
    def __init__(self, project, parent=None):
        super().__init__(parent)
        self.target = None
        self.setWindowTitle('Imported Frame Review')
        self.resize(1100, 650)
        layout = QVBoxLayout(self)
        self.rows = review_rows(project)
        pending = sum(row['status'] == 'pending' for row in self.rows)
        layout.addWidget(QLabel(f'{pending} pending / {len(self.rows)} imported frames'))
        self.table = QTableWidget(len(self.rows), 5)
        self.table.setHorizontalHeaderLabels(['Video / frame', 'Review', 'Categories', 'Policy differences', 'Reason'])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table.verticalHeader().hide()
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        current = category_policies(read_project_info(project))
        for row, entry in enumerate(self.rows):
            requirements = entry['task']['requirements']
            policies = [f"{category_label(c)}: {policy_text(r['source_policy'])} -> {policy_text(current[c])}"
                        for c, r in requirements.items()]
            reasons = sorted({reason for r in requirements.values() for reason in r['reasons']})
            values = [f"{entry['video']} / {entry['frame']}", entry['status'].title(),
                      ', '.join(category_label(c) for c in requirements), '\n'.join(policies), '\n'.join(reasons)]
            for col, value in enumerate(values):
                item = QTableWidgetItem(value)
                item.setToolTip(value)
                self.table.setItem(row, col, item)
        self.table.resizeRowsToContents()
        layout.addWidget(self.table)
        note = QLabel('Review each frame against its destination policy. A shared-mask edit affects other frames; '
                      'saved label or policy changes reopen completed reviews. Training selection is separate.')
        note.setWordWrap(True)
        layout.addWidget(note)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Open | QDialogButtonBox.StandardButton.Close)
        self.open_button = buttons.button(QDialogButtonBox.StandardButton.Open)
        self.open_button.setText('Open Frame')
        self.open_button.setEnabled(False)
        self.table.itemSelectionChanged.connect(lambda: self.open_button.setEnabled(self.table.currentRow() >= 0))
        self.table.cellDoubleClicked.connect(lambda row, column: self.open_frame())
        buttons.accepted.connect(self.open_frame)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        if self.rows:
            self.table.selectRow(next((i for i, row in enumerate(self.rows) if row['status'] == 'pending'), 0))

    def open_frame(self):
        row = self.table.currentRow()
        if row >= 0:
            self.target = (self.rows[row]['video'], self.rows[row]['frame'])
            self.accept()
