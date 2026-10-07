"""Selective project import with explicit annotation-scope reconciliation."""

from pathlib import Path

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (QAbstractItemView, QCheckBox, QComboBox, QDialog,
                            QDialogButtonBox, QFileDialog, QFormLayout, QHBoxLayout,
                            QHeaderView, QLabel, QLineEdit, QListWidget, QListWidgetItem,
                            QMessageBox, QPushButton, QSplitter, QStyle, QTableWidget,
                            QTableWidgetItem, QTextEdit, QTreeWidget, QTreeWidgetItem,
                            QVBoxLayout, QWidget)

from core.annotation_scope import frame_categories, read_project_info
from core.categories import CATEGORIES, category_label
from core.project_import import inspect_project, prepare_import
from gui.project_sync_dialog import SyncProgressDialog


class ImportPreviewDialog(QDialog):
    def __init__(self, plan, parent=None):
        super().__init__(parent)
        self.setWindowTitle('Review Project Import')
        self.resize(1080, 640)
        layout = QVBoxLayout(self)
        destination = QLabel('Destination: ' + str(plan.destination))
        destination.setWordWrap(True)
        layout.addWidget(destination)
        self.table = QTableWidget(len(plan.rows), 6)
        self.table.setHorizontalHeaderLabels(['Video', 'Scope', 'Source frame', 'Category', 'Count', 'Action'])
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.verticalHeader().hide()
        for row, record in enumerate(plan.rows):
            for col, key in enumerate(('video', 'scope', 'frame', 'category', 'instances', 'action')):
                value = category_label(record[key]) if key == 'category' else str(record[key])
                item = QTableWidgetItem(value)
                item.setToolTip(', '.join(map(str, record['frame_indices']))
                                if key == 'frame' and 'frame_indices' in record else value)
                self.table.setItem(row, col, item)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        layout.addWidget(self.table, 1)
        warnings = QTextEdit()
        warnings.setReadOnly(True)
        warnings.setPlainText('\n\n'.join(plan.warnings))
        warnings.setMaximumHeight(150)
        warnings.setVisible(bool(plan.warnings))
        layout.addWidget(warnings)
        self.confirm_scope = QCheckBox('Apply the chosen reference masks to the whole video')
        self.confirm_scope.setVisible(any('will apply to the whole video' in w for w in plan.warnings))
        layout.addWidget(self.confirm_scope)
        self.confirm_replace = QCheckBox('Replace the listed destination categories (originals will be backed up)')
        self.confirm_replace.setVisible(any(r['action'] == 'Replace' for r in plan.rows))
        layout.addWidget(self.confirm_replace)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        self.import_button = buttons.button(QDialogButtonBox.StandardButton.Ok)
        self.import_button.setText('Import Copies')
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        self.confirm_scope.toggled.connect(self.update_confirmation)
        self.confirm_replace.toggled.connect(self.update_confirmation)
        layout.addWidget(buttons)
        self.update_confirmation()

    def update_confirmation(self):
        self.import_button.setEnabled(all(box.isHidden() or box.isChecked()
                                          for box in (self.confirm_scope, self.confirm_replace)))


class ProjectImportDialog(QDialog):
    def __init__(self, destination, contributor=None, parent=None):
        super().__init__(parent)
        self.destination = Path(destination)
        self.contributor = contributor
        self.catalog = None
        self.result = None
        self.hive_sources = {}
        self.setWindowTitle('Import From Project')
        self.resize(1080, 760)
        layout = QVBoxLayout(self)
        form = QFormLayout()
        self.source = QLineEdit()
        self.source.setReadOnly(True)
        browse = QPushButton()
        browse.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_DirOpenIcon))
        browse.setToolTip('Choose source project')
        browse.clicked.connect(self.browse)
        source_row = QHBoxLayout()
        source_row.addWidget(self.source)
        source_row.addWidget(browse)
        form.addRow('Source project:', source_row)
        target = QLabel(str(self.destination))
        target.setWordWrap(True)
        form.addRow('Destination:', target)
        layout.addLayout(form)

        filters = QHBoxLayout()
        self.annotated_only = QCheckBox('Annotated frames only')
        self.annotated_only.setChecked(True)
        self.annotated_only.toggled.connect(self.filter_frames)
        filters.addWidget(self.annotated_only)
        filters.addStretch()
        for icon, tooltip, checked in ((QStyle.StandardPixmap.SP_DialogApplyButton, 'Select visible frames', True),
                                        (QStyle.StandardPixmap.SP_DialogResetButton, 'Clear frame selection', False)):
            button = QPushButton()
            button.setIcon(self.style().standardIcon(icon))
            button.setToolTip(tooltip)
            button.clicked.connect(lambda unused=False, state=checked: self.select_visible(state))
            filters.addWidget(button)
        layout.addLayout(filters)
        splitter = QSplitter()
        self.frames = QTreeWidget()
        self.frames.setHeaderLabels(['Video / frame', 'Annotations', 'Hive reference frame'])
        self.frames.setAlternatingRowColors(True)
        self.frames.header().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.frames.setColumnWidth(1, 95)
        self.frames.setColumnWidth(2, 195)
        self.frames.itemChanged.connect(self.update_count)
        splitter.addWidget(self.frames)
        category_panel = QWidget()
        category_layout = QVBoxLayout(category_panel)
        category_layout.setContentsMargins(0, 0, 0, 0)
        category_layout.addWidget(QLabel('Categories'))
        self.categories = QListWidget()
        self.categories.setMinimumWidth(225)
        self.categories.setWordWrap(True)
        self.categories.setTextElideMode(Qt.TextElideMode.ElideNone)
        for category in CATEGORIES:
            item = QListWidgetItem(category_label(category))
            item.setData(Qt.ItemDataRole.UserRole, category)
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(Qt.CheckState.Checked)
            item.setToolTip(category_label(category))
            self.categories.addItem(item)
        self.categories.itemChanged.connect(self.filter_frames)
        category_layout.addWidget(self.categories)
        splitter.addWidget(category_panel)
        splitter.setSizes([760, 300])
        layout.addWidget(splitter, 1)

        options = QFormLayout()
        self.split = QComboBox()
        for text, value in [('Keep source split', 'preserve'), ('Training', 'train'),
                            ('Validation', 'val'), ('Test', 'test'), ('Inference', 'inference')]:
            self.split.addItem(text, value)
        options.addRow('Destination split:', self.split)
        self.conflicts = QComboBox()
        self.conflicts.addItem('Keep destination annotations', False)
        self.conflicts.addItem('Replace selected categories', True)
        options.addRow('Existing annotations:', self.conflicts)
        self.videos = QCheckBox('Copy original videos')
        self.videos.setToolTip('Optional. Frames and labels work without videos; video-based tools need the originals.')
        options.addRow('', self.videos)
        layout.addLayout(options)
        self.count = QLabel('0 frames selected')
        layout.addWidget(self.count)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Cancel)
        self.preview_button = buttons.addButton('Preview Import...', QDialogButtonBox.ButtonRole.ActionRole)
        self.preview_button.setEnabled(False)
        self.preview_button.clicked.connect(self.preview_import)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def browse(self):
        path = QFileDialog.getExistingDirectory(self, 'Choose Source Project', str(self.destination.parent))
        if path:
            try:
                self.load_source(path)
            except Exception as exc:
                QMessageBox.warning(self, 'Source Project Unavailable', str(exc))

    def load_source(self, path):
        if Path(path).resolve() == self.destination.resolve():
            raise ValueError('Choose a different project as the source.')
        catalog = SyncProgressDialog('Read Source Project', lambda progress, cancelled: inspect_project(path), self).run()
        self.catalog = catalog
        self.source.setText(str(catalog['path']))
        self.frames.blockSignals(True)
        self.frames.clear()
        self.hive_sources = {}
        convert_hive = ('hive' in frame_categories(catalog['info'])
                        and 'hive' not in frame_categories(read_project_info(self.destination)))
        self.frames.setColumnHidden(2, not convert_hive)
        for video, record in catalog['videos'].items():
            parent = QTreeWidgetItem([video, record['split']])
            parent.setData(0, Qt.ItemDataRole.UserRole, video)
            parent.setToolTip(0, video)
            parent.setFlags(parent.flags() | Qt.ItemFlag.ItemIsAutoTristate | Qt.ItemFlag.ItemIsUserCheckable)
            parent.setCheckState(0, Qt.CheckState.Unchecked)
            self.frames.addTopLevelItem(parent)
            if convert_hive:
                combo = QComboBox()
                combo.addItem('Earliest selected annotation', None)
                for index, frame in record['frames'].items():
                    if frame['local_hive']:
                        combo.addItem(f'Frame {index}', index)
                combo.setToolTip('Use all hive instances from one selected frame; other frames are not combined.')
                self.frames.setItemWidget(parent, 2, combo)
                self.hive_sources[video] = combo
            for index, frame in record['frames'].items():
                child = QTreeWidgetItem([f'Frame {index}', str(sum(frame['counts'].values()))])
                child.setData(0, Qt.ItemDataRole.UserRole, index)
                child.setFlags(child.flags() | Qt.ItemFlag.ItemIsUserCheckable)
                child.setCheckState(0, Qt.CheckState.Checked if frame['selected'] and frame['counts'] else Qt.CheckState.Unchecked)
                parent.addChild(child)
            parent.setExpanded(len(catalog['videos']) <= 4)
        self.frames.blockSignals(False)
        self.filter_frames()

    def selected_categories(self):
        return [self.categories.item(i).data(Qt.ItemDataRole.UserRole)
                for i in range(self.categories.count())
                if self.categories.item(i).checkState() == Qt.CheckState.Checked]

    def filter_frames(self, *unused):
        if not self.catalog:
            return
        categories = set(self.selected_categories())
        for i in range(self.frames.topLevelItemCount()):
            parent = self.frames.topLevelItem(i)
            video = parent.data(0, Qt.ItemDataRole.UserRole)
            for j in range(parent.childCount()):
                child = parent.child(j)
                frame = self.catalog['videos'][video]['frames'][child.data(0, Qt.ItemDataRole.UserRole)]
                count = sum(n for c, n in frame['counts'].items() if c in categories)
                child.setHidden(self.annotated_only.isChecked() and count == 0)
                child.setText(1, str(count))
        self.update_count()

    def selections(self):
        result = {}
        for i in range(self.frames.topLevelItemCount()):
            parent = self.frames.topLevelItem(i)
            indices = [parent.child(j).data(0, Qt.ItemDataRole.UserRole) for j in range(parent.childCount())
                       if not parent.child(j).isHidden() and parent.child(j).checkState(0) == Qt.CheckState.Checked]
            if indices:
                result[parent.data(0, Qt.ItemDataRole.UserRole)] = indices
        return result

    def select_visible(self, checked):
        self.frames.blockSignals(True)
        for i in range(self.frames.topLevelItemCount()):
            parent = self.frames.topLevelItem(i)
            for j in range(parent.childCount()):
                if not parent.child(j).isHidden():
                    parent.child(j).setCheckState(0, Qt.CheckState.Checked if checked else Qt.CheckState.Unchecked)
        self.frames.blockSignals(False)
        self.update_count()

    def update_count(self, *unused):
        number = sum(map(len, self.selections().values()))
        self.count.setText(f'{number} frames selected')
        self.preview_button.setEnabled(number > 0 and bool(self.selected_categories()))

    def preview_import(self):
        plan = None
        try:
            # Snapshot widget values on the GUI thread before starting file work.
            source = self.catalog['path']
            selections, categories = self.selections(), self.selected_categories()
            options = dict(split=self.split.currentData(), include_videos=self.videos.isChecked(),
                           replace=self.conflicts.currentData(),
                           hive_frames={video: combo.currentData() for video, combo in self.hive_sources.items()},
                           contributor=self.contributor)
            plan = SyncProgressDialog('Prepare Import Preview', lambda progress, cancelled: prepare_import(
                source, self.destination, selections, categories, **options,
                progress=progress, cancelled=cancelled), self).run()
            preview = ImportPreviewDialog(plan, self)
            if preview.exec() != QDialog.DialogCode.Accepted:
                return
            self.result = SyncProgressDialog('Import Project Data', plan.apply, self).run()
            self.accept()
        except Exception as exc:
            QMessageBox.warning(self, 'Import Not Completed', str(exc))
        finally:
            if plan:
                plan.close()
