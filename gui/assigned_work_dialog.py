"""Coordinator assignments and contributor downloads, without exposing archive internals."""

from pathlib import Path

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (QAbstractItemView, QComboBox, QDialog, QDialogButtonBox,
    QFileDialog, QHBoxLayout, QHeaderView, QLabel, QLineEdit, QMessageBox,
    QPushButton, QTableWidget, QTableWidgetItem, QVBoxLayout)

from core.contributors import contributor_profile
from core.project_sync import _read_json
from core.project_update import PublishedProject
from core.assigned_work import (accept_submission, assignment_status, catalog, download_plan,
    download_work, latest_snapshot, my_projects, parse_frames, read_assignments, save_assignments)
from gui.project_sync_dialog import SyncProgressDialog, setting_json


def _table(headers):
    table = QTableWidget(0, len(headers))
    table.setHorizontalHeaderLabels(headers)
    table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
    table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
    table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
    table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
    table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
    return table


class AssignmentManagerDialog(QDialog):
    def __init__(self, project, configuration, contributor, parent=None):
        super().__init__(parent)
        self.project, self.configuration, self.contributor = project, configuration, contributor
        self.profiles = {identity: {'id': identity, 'name': name}
                         for name, identity in setting_json('contributors/profiles', {}).items()}
        self.profiles[contributor['id']] = {k: contributor[k] for k in ('id', 'name')}
        self.setWindowTitle('Video Assignments')
        self.resize(1080, 580)
        layout = QVBoxLayout(self)
        self.summary = QLabel()
        self.summary.setWordWrap(True)
        layout.addWidget(self.summary)
        self.table = _table(['Video', 'Split', 'Published frames', 'Contributor', 'Assigned frames', 'Status'])
        layout.addWidget(self.table)
        row = QHBoxLayout()
        import_button = QPushButton('Import Contributor Profile...')
        import_button.clicked.connect(self.import_profile)
        row.addWidget(import_button)
        refresh = QPushButton('Refresh')
        refresh.clicked.connect(self.confirm_refresh)
        row.addWidget(refresh)
        self.accept_work = QPushButton('Accept Reviewed Submission...')
        self.accept_work.clicked.connect(self.accept_selected)
        row.addWidget(self.accept_work)
        row.addStretch()
        layout.addLayout(row)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Close)
        buttons.button(QDialogButtonBox.StandardButton.Save).setText('Save Assignments')
        buttons.accepted.connect(self.save)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        self.refresh()

    def refresh(self):
        def load(progress, cancelled):
            view = PublishedProject(self.project, self.configuration)
            data = read_assignments(view)
            if data['coordinator'] and data['coordinator']['id'] != self.contributor['id']:
                raise ValueError('Assignment management belongs to ' + data['coordinator']['name'] +
                                 '. Use My Assigned Work for your assignments.')
            snapshot, _ = latest_snapshot(view)
            return data, catalog(snapshot), {v: assignment_status(view, e) for v, e in data['videos'].items()}
        self.data, self.available, self.statuses = SyncProgressDialog('Read Video Assignments', load, self).run()
        for entry in self.data['videos'].values():
            self.profiles[entry['contributor']['id']] = entry['contributor']
        for actor, frames in getattr(self, 'controls', {}).values():
            actor.hide()
            frames.hide()
        self.table.setRowCount(0)
        self.videos = sorted(self.available)
        self.table.setRowCount(len(self.videos))
        self.controls = {}
        for row, video in enumerate(self.videos):
            info = self.available[video]
            entry = self.data['videos'].get(video)
            for col, value in ((0, video), (1, info['split']), (2, str(len(info['frames']))),
                               (5, self.statuses.get(video, ('Unassigned', None))[0])):
                self.table.setItem(row, col, QTableWidgetItem(value))
            actor = QComboBox()
            actor.addItem('Unassigned', None)
            for identity, profile in sorted(self.profiles.items(), key=lambda item: item[1]['name'].casefold()):
                actor.addItem(profile['name'] + ' (' + identity[:8] + ')', identity)
            if entry:
                actor.setCurrentIndex(actor.findData(entry['contributor']['id']))
            frames = QLineEdit('All' if not entry or entry['frames'] == info['frames'] else
                               ','.join(str(i) for i in entry['frames']))
            frames.setMinimumWidth(110)
            frames.setToolTip('All, individual frame numbers, or ranges such as 0,5-10')
            self.table.setCellWidget(row, 3, actor)
            self.table.setCellWidget(row, 4, frames)
            self.controls[video] = actor, frames
        self.summary.setText(f'{len(self.data["videos"])} assigned videos | Coordinator: '
                             + (self.data['coordinator'] or self.contributor)['name'])

    def confirm_refresh(self):
        if QMessageBox.question(self, 'Refresh Assignments?', 'Discard unsaved assignment choices and reload?',
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No) == QMessageBox.StandardButton.Yes:
            try:
                self.refresh()
            except Exception as exc:
                QMessageBox.warning(self, 'Refresh Failed', str(exc))

    def import_profile(self):
        path, _ = QFileDialog.getOpenFileName(self, 'Import Contributor Profile', '', 'JSON (*.json)')
        if not path:
            return
        try:
            profile = contributor_profile(_read_json(path))
            self.profiles[profile['id']] = {key: profile[key] for key in ('id', 'name')}
            for actor, _ in self.controls.values():
                index = actor.findData(profile['id'])
                text = profile['name'] + ' (' + profile['id'][:8] + ')'
                if index < 0:
                    actor.addItem(text, profile['id'])
                else:
                    actor.setItemText(index, text)
        except Exception as exc:
            QMessageBox.warning(self, 'Profile Not Imported', str(exc))

    def choices(self):
        return {video: {'contributor': self.profiles[actor.currentData()],
                        'frames': parse_frames(frames.text(), self.available[video]['frames'])}
                for video, (actor, frames) in self.controls.items() if actor.currentData()}

    def save(self):
        try:
            choices = self.choices()
            old = {v: {k: e[k] for k in ('contributor', 'frames')} for v, e in self.data['videos'].items()}
            if old == choices:
                return
            if QMessageBox.question(self, 'Save Video Assignments?',
                    'Each video has one assignee. Changing its contributor or frames retires the previous assignment. '
                    'Existing local work is preserved but cannot publish under the retired assignment. Continue?',
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                    QMessageBox.StandardButton.No) != QMessageBox.StandardButton.Yes:
                return
            SyncProgressDialog('Save Video Assignments',
                lambda progress, cancelled: save_assignments(self.project, self.configuration, self.contributor,
                                                             choices, self.data['revision']), self).run()
            self.refresh()
        except Exception as exc:
            QMessageBox.warning(self, 'Assignments Not Saved', str(exc))

    def accept_selected(self):
        try:
            old = {v: {k: e[k] for k in ('contributor', 'frames')} for v, e in self.data['videos'].items()}
            if self.choices() != old:
                raise ValueError('Save or refresh your assignment choices before accepting a submission.')
        except ValueError as exc:
            QMessageBox.warning(self, 'Unsaved Assignment Choices', str(exc))
            return
        row = self.table.currentRow()
        if row < 0:
            return
        video = self.videos[row]
        status, revision = self.statuses.get(video, ('Unassigned', None))
        if status != 'Ready for Review':
            QMessageBox.information(self, 'Not Ready for Review', 'Select a video marked Ready for Review.')
            return
        if QMessageBox.question(self, 'Accept Reviewed Submission?',
                f'Confirm that you updated your local project and reviewed {video}, including any video-wide masks.',
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No) != QMessageBox.StandardButton.Yes:
            return
        try:
            SyncProgressDialog('Accept Submission', lambda progress, cancelled:
                accept_submission(self.project, self.configuration, self.contributor, video, revision), self).run()
            self.refresh()
        except Exception as exc:
            QMessageBox.warning(self, 'Submission Not Accepted', str(exc))


class AssignedDownloadDialog(QDialog):
    def __init__(self, configuration, contributor, parent=None):
        super().__init__(parent)
        self.configuration, self.contributor = configuration, contributor
        self.download_result = None
        self.setWindowTitle('My Assigned Work')
        self.resize(820, 460)
        layout = QVBoxLayout(self)
        self.projects = QComboBox()
        layout.addWidget(self.projects)
        self.table = _table(['Video', 'Assigned frames', 'Download'])
        layout.addWidget(self.table)
        self.summary = QLabel('')
        self.summary.setWordWrap(True)
        layout.addWidget(self.summary)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Cancel)
        self.download = buttons.addButton('Download Assigned Work...', QDialogButtonBox.ButtonRole.ActionRole)
        self.download.clicked.connect(self.download_selected)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        self.available = SyncProgressDialog('Find My Assignments', lambda progress, cancelled:
            my_projects(configuration, contributor), self).run()
        for project in self.available:
            self.projects.addItem(project['name'], project['id'])
        self.projects.currentIndexChanged.connect(self.populate)
        self.populate()

    def populate(self):
        index = self.projects.currentIndex()
        videos = self.available[index]['videos'] if index >= 0 else {}
        self.videos = sorted(videos)
        self.table.setRowCount(len(self.videos))
        for row, video in enumerate(self.videos):
            self.table.setItem(row, 0, QTableWidgetItem(video))
            self.table.setItem(row, 1, QTableWidgetItem(str(len(videos[video]['frames']))))
            check = QTableWidgetItem()
            check.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsUserCheckable)
            check.setCheckState(Qt.CheckState.Checked)
            self.table.setItem(row, 2, check)
        self.download.setEnabled(bool(videos))
        self.summary.setText(f'Contributor: {self.contributor["name"]} | {len(videos)} assigned videos' if videos else
                             'No assignments for this contributor profile. Ask the coordinator to assign your exported profile.')

    def download_selected(self):
        selected = [v for row, v in enumerate(self.videos)
                    if self.table.item(row, 2).checkState() == Qt.CheckState.Checked]
        if not selected:
            return
        try:
            project_id = self.projects.currentData()
            plan = SyncProgressDialog('Estimate Assigned Download', lambda progress, cancelled:
                download_plan(self.configuration, project_id, self.contributor, selected), self).run()
            count = sum(len(e['frames']) for e in plan['assignments'].values())
            if QMessageBox.question(self, 'Download Assigned Work?',
                    f'{len(selected)} videos | {count} frame images | {plan["bytes"] / 1024**2:.1f} MiB\n\n'
                    'Original videos are excluded. Video-wide masks affect all frames of their video, '
                    'including frames not downloaded.', QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                    QMessageBox.StandardButton.No) != QMessageBox.StandardButton.Yes:
                return
            parent = QFileDialog.getExistingDirectory(self, 'Choose a Local Parent Folder', str(Path.home()))
            if not parent:
                return
            from core.project_sync import _folder_label
            name = _folder_label(self.projects.currentText(), 'project') + '-assigned'
            destination = Path(parent) / name
            suffix = 2
            while destination.exists():
                destination = Path(parent) / f'{name}-{suffix}'
                suffix += 1
            self.download_result = SyncProgressDialog('Download Assigned Work', lambda progress, cancelled:
                download_work(self.configuration, project_id, self.contributor, selected, destination,
                              progress, cancelled), self).run()
            self.accept()
        except Exception as exc:
            QMessageBox.warning(self, 'Work Not Downloaded', str(exc))
