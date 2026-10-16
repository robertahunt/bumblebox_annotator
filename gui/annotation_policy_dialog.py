"""Project label intent and imported-frame review controls."""

from PyQt6.QtCore import Qt, pyqtSignal, QSignalBlocker
from PyQt6.QtGui import QPalette
from PyQt6.QtWidgets import (QAbstractItemView, QCheckBox, QComboBox, QDialog,
                            QDialogButtonBox, QHeaderView, QLabel, QMessageBox,
                            QPushButton, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget)

from core.annotation_policy import (MEANINGS, SCOPES, category_policies, default_policy,
                                    guidance, policy_needs_confirmation, policy_signature, policy_text,
                                    save_policy, temporal_support)
from core.annotation_scope import read_project_info
from core.categories import CATEGORIES, BROOD_CATEGORIES, category_label
from core.import_review import review_rows
from core.assigned_work import workspace


class PolicyTable(QWidget):
    hive_scope_changed = pyqtSignal(str)

    def __init__(self, info, parent=None, *, new=False):
        super().__init__(parent)
        self.new = new
        self.info = info
        self.meanings = {}
        self.scopes = {}
        self.groups = {}
        for category in CATEGORIES:
            group = 'brood' if category in BROOD_CATEGORIES else category
            self.groups.setdefault(group, []).append(category)
        self.rows = list(self.groups)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.table = QTableWidget(len(self.rows), 4)
        self.table.setHorizontalHeaderLabels(['Category', 'Saved scope', 'Label meaning', 'Temporal analysis'])
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table.verticalHeader().hide()
        self.table.setWordWrap(True)
        self.table.setMinimumHeight(280)
        entries = default_policy(info)['categories'] if new else category_policies(info)
        for row, group in enumerate(self.rows):
            members = self.groups[group]
            category = members[0]
            label = 'Brood' if group == 'brood' else category_label(category)
            for column, text in ((0, label), (3, temporal_support(category))):
                item = QTableWidgetItem(text)
                item.setToolTip(text)
                self.table.setItem(row, column, item)
            if group == 'brood':
                self.table.item(row, 0).setToolTip('All brood and queen brood stages; segmentation labels remain separate.')
            scope_combo = QComboBox()
            saved_scope = entries[category]['scope']
            for key, scope_label in SCOPES.items():
                scope_combo.addItem(scope_label, key)
                if key != saved_scope and not (new and group == 'hive'):
                    tip = 'Requires project import conversion' if group == 'hive' else 'Not implemented yet'
                    scope_combo.setItemData(scope_combo.findData(key), tip, Qt.ItemDataRole.ToolTipRole)
            scope_combo.setCurrentIndex(scope_combo.findData(saved_scope))
            scope_combo.currentIndexChanged.connect(lambda unused, g=group: self._scope_requested(g))
            self.table.setCellWidget(row, 1, scope_combo)
            self.scopes[group] = scope_combo
            combo = QComboBox()
            meanings = {entries[c]['meaning'] for c in members}
            if len(meanings) > 1:
                combo.addItem('Mixed (keep existing)', None)
            for key, label in MEANINGS.items():
                combo.addItem(label, key)
            combo.setCurrentIndex(combo.findData(next(iter(meanings))) if len(meanings) == 1 else 0)
            combo.currentIndexChanged.connect(lambda unused, g=group: self.update_guidance(g))
            self.table.setCellWidget(row, 2, combo)
            self.meanings[group] = combo
        self.table.resizeRowsToContents()
        self.table.cellClicked.connect(lambda row, column: self.update_guidance(self.rows[row]))
        layout.addWidget(self.table)
        self.help = QLabel()
        self.help.setWordWrap(True)
        self.help.setMinimumHeight(48)
        layout.addWidget(self.help)
        self.update_guidance('hive')

    def entries(self):
        entries = category_policies(self.info)
        for group, combo in self.meanings.items():
            # A mixed legacy group stays unchanged until the user chooses a common meaning.
            if combo.currentData() is not None:
                for category in self.groups[group]:
                    entries[category]['meaning'] = combo.currentData()
        return entries

    def update_guidance(self, group):
        if hasattr(self, 'help'):
            if group == 'brood' and self.meanings[group].currentData() is None:
                self.help.setText('Brood: saved stage meanings differ. They remain unchanged until you choose '
                                  'a common meaning for all brood and queen brood stages.')
            else:
                label = 'Brood (all brood and queen brood stages)' if group == 'brood' else category_label(group)
                self.help.setText(label + ': ' + guidance(self.entries()[self.groups[group][0]]))

    def _scope_requested(self, group):
        combo = self.scopes[group]
        saved = category_policies(self.info)[self.groups[group][0]]['scope']
        requested = combo.currentData()
        if requested == saved:
            return
        if self.new and group == 'hive':
            self.set_hive_scope(requested)
            return
        with QSignalBlocker(combo):
            combo.setCurrentIndex(combo.findData(saved))
        self.update_guidance(group)
        if group == 'hive':
            title = 'Hive Scope Conversion Required'
            message = (
                'Hive annotations support both scopes, but an existing project cannot switch them here.\n\n'
                f'Create a destination project with hive scope "{SCOPES[requested]}", then use '
                'File > Import From Project... in that project to preview and convert copies of the annotations.')
        else:
            label = 'Brood (all brood and queen brood stages)' if group == 'brood' else category_label(group)
            title = 'Scope Not Implemented'
            message = f'Sorry, "{SCOPES[requested]}" annotations for {label} have not been implemented yet.'
        QMessageBox.information(
            self, title, message + f'\n\nScope remains "{SCOPES[saved]}". '
            'Your masks and saved scope have not been changed.')

    def set_hive_scope(self, scope):
        if scope not in SCOPES:
            raise ValueError(f'Unknown hive scope: {scope}')
        combo = self.scopes['hive']
        if not self.new:
            combo.setCurrentIndex(combo.findData(scope))
            return
        if scope == category_policies(self.info)['hive']['scope']:
            return
        self.info = {'hive_annotation_scope': scope}
        with QSignalBlocker(combo):
            combo.setCurrentIndex(combo.findData(scope))
        self.meanings['hive'].setCurrentIndex(self.meanings['hive'].findData(
            'visible' if scope == 'frame' else 'full_extent'))
        self.update_guidance('hive')
        self.hive_scope_changed.emit(scope)


class AnnotationPolicyDialog(QDialog):
    policy_saved = pyqtSignal()
    policy_reviewed = pyqtSignal(str)

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
        info = read_project_info(project)
        pending = policy_needs_confirmation(info)
        self.saved_signature = policy_signature(info)
        self.editing = pending
        self.setWindowTitle('Finish Configuring Annotation Policy' if pending else 'Review Saved Annotation Policy')
        self.reminder_label = QLabel(
            'Annotation policy setup is incomplete. Choose label meanings for all categories and save to finish. '
            'Until then, this reminder will return when you reopen this project, including after restarting the app.')
        self.reminder_label.setWordWrap(True)
        self.reminder_label.setVisible(pending)
        layout.addWidget(self.reminder_label)
        self.policy = PolicyTable(info, self)
        layout.addWidget(self.policy)
        self.edit_button = QPushButton('Edit Policy')
        self.edit_button.clicked.connect(self.begin_editing)
        self.edit_button.setVisible(not pending)
        layout.addWidget(self.edit_button)
        note = QLabel('Saved scopes cannot be changed here. Hive scope conversion is available through Import From Project; '
                      'alternate scopes for the other categories are not implemented yet. '
                      'Temporal maps are batch outputs, not training labels. Changing label meaning does not edit masks.')
        note.setWordWrap(True)
        layout.addWidget(note)
        self.confirm = QCheckBox('These meanings describe my intended labels; imported conflicts still need frame review')
        layout.addWidget(self.confirm)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Close)
        self.defer_button = buttons.button(QDialogButtonBox.StandardButton.Close)
        self.defer_button.setText('Remind Me Later' if pending else 'Continue')
        if pending:
            self.defer_button.setToolTip('Close without saving these choices. Remind me when this project is reopened.')
        self.save_button = buttons.button(QDialogButtonBox.StandardButton.Save)
        self.save_button.setEnabled(False)
        self.save_button.setVisible(pending)
        self.confirm.setVisible(pending)
        self._set_controls_editable(pending)
        self.confirm.toggled.connect(self.save_button.setEnabled)
        buttons.accepted.connect(self.save)
        buttons.rejected.connect(self.continue_or_defer)
        layout.addWidget(buttons)

        self.assigned = bool(workspace(project))
        if self.assigned:
            self.editing = False
            self.edit_button.hide()
            self.confirm.hide()
            self.save_button.hide()
            self._set_controls_editable(False)
            self.setWindowTitle('Review Assigned Annotation Policy')

    def _set_controls_editable(self, editable):
        for combo in (*self.policy.meanings.values(), *self.policy.scopes.values()):
            palette = combo.palette()
            for role in (QPalette.ColorRole.Text, QPalette.ColorRole.ButtonText):
                palette.setColor(QPalette.ColorGroup.Disabled, role,
                                 palette.color(QPalette.ColorGroup.Active, role))
            combo.setPalette(palette)
            combo.setEnabled(editable)

    def begin_editing(self):
        if self.assigned:
            return
        self.editing = True
        self.setWindowTitle('Edit Annotation Policy')
        self._set_controls_editable(True)
        self.edit_button.hide()
        self.confirm.show()
        self.save_button.show()
        self.defer_button.setText('Cancel')

    def continue_or_defer(self):
        if not self.editing:
            self.policy_reviewed.emit(self.saved_signature)
        self.reject()

    def save(self):
        try:
            if not self.editing or not self.confirm.isChecked():
                return
            if policy_signature(read_project_info(self.project)) != self.saved_signature:
                raise ValueError('The saved policy changed while this dialog was open. Reopen it before editing.')
            save_policy(self.project, self.policy.entries(), self.contributor)
            self.saved_signature = policy_signature(read_project_info(self.project))
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
