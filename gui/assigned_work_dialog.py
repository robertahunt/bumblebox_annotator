"""Coordinator assignments and contributor downloads, without exposing archive internals."""

from pathlib import Path

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (QAbstractItemView, QComboBox, QDialog, QDialogButtonBox,
    QFileDialog, QFormLayout, QHBoxLayout, QHeaderView, QInputDialog, QLabel, QLineEdit, QListWidget, QMessageBox,
    QPushButton, QTableWidget, QTableWidgetItem, QToolButton, QVBoxLayout, QWidget)

from core.contributors import contributor_profile
from core.project_sync import _read_json
from core.project_update import PublishedProject
from core.assigned_work import (accept_submission, assignment_status, catalog, download_plan,
    download_work, latest_snapshot, my_projects, parse_frames, read_assignments, save_assignments,
    transfer_director)
from gui.project_sync_dialog import (SyncProgressDialog, TransferConfirmationDialog,
                                     setting_json, save_setting, load_contributor_profiles)
from gui.workflow_help import help_button


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
    assignments_changed = pyqtSignal()

    def __init__(self, project, configuration, contributor, parent=None, *, initial=None, embedded=False):
        super().__init__(parent)
        if embedded:
            self.setWindowFlags(Qt.WindowType.Widget)
        self.project, self.configuration, self.contributor = project, configuration, contributor
        self.profiles = {identity: {key: profile[key] for key in ('id', 'name')}
                         for identity, profile in load_contributor_profiles().items()}
        self.profiles[contributor['id']] = {k: contributor[k] for k in ('id', 'name')}
        for value in setting_json('assignments/profiles', {}).values():
            profile = contributor_profile(dict(value, format=1, kind='bumblebox-contributor'))
            self.profiles[profile['id']] = {k: profile[k] for k in ('id', 'name')}
        self.setWindowTitle('Video Assignments')
        self.resize(1080, 580)
        layout = QVBoxLayout(self)
        self.summary = QLabel()
        self.summary.setWordWrap(True)
        layout.addWidget(self.summary)
        filters = QHBoxLayout()
        self.search = QLineEdit()
        self.search.setPlaceholderText('Search videos or contributors')
        self.filter = QComboBox()
        self.filter.addItems(['All videos', 'Unassigned', 'Assigned', 'Ready for Review'])
        filters.addWidget(self.search, 1)
        filters.addWidget(self.filter)
        layout.addLayout(filters)
        bulk = QHBoxLayout()
        self.select_visible = QPushButton('Select Visible')
        self.select_visible.clicked.connect(self.check_visible)
        clear = QPushButton('Clear Selection')
        clear.clicked.connect(self.clear_checked)
        self.assign_selected = QPushButton('Assign Selected Videos...')
        self.assign_selected.setEnabled(False)
        self.assign_selected.clicked.connect(self.bulk_assign)
        bulk.addWidget(self.select_visible)
        bulk.addWidget(clear)
        bulk.addStretch()
        bulk.addWidget(self.assign_selected)
        layout.addLayout(bulk)
        self.table = _table(['Video', 'Split', 'Published frames', 'Contributor', 'Assigned frames', 'Status'])
        self.table.itemChanged.connect(self.update_selection)
        layout.addWidget(self.table)
        self.search.textChanged.connect(self.filter_videos)
        self.filter.currentIndexChanged.connect(self.filter_videos)
        self.contributors_toggle = QToolButton()
        self.contributors_toggle.setText('Contributors')
        self.contributors_toggle.setCheckable(True)
        self.contributors_toggle.setArrowType(Qt.ArrowType.RightArrow)
        self.contributors_toggle.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        contributors_header = QHBoxLayout()
        contributors_header.addWidget(self.contributors_toggle)
        contributors_header.addStretch()
        layout.insertLayout(1, contributors_header)
        self.contributors_panel = QWidget()
        contributors_layout = QVBoxLayout(self.contributors_panel)
        self.roster = QListWidget()
        self.roster.setMaximumHeight(110)
        contributors_layout.addWidget(self.roster)
        import_button = QPushButton('Add Contributor: Import Profile...')
        import_button.clicked.connect(self.import_profile)
        contributors_header.addWidget(import_button)
        layout.insertWidget(2, self.contributors_panel)
        self.contributors_panel.hide()
        self.contributors_toggle.toggled.connect(self.show_contributors)
        row = QHBoxLayout()
        refresh = QPushButton('Refresh')
        refresh.clicked.connect(self.confirm_refresh)
        row.addWidget(refresh)
        self.accept_work = QPushButton('Accept Reviewed Submission...')
        self.accept_work.clicked.connect(self.accept_selected)
        row.addWidget(self.accept_work)
        self.transfer_button = QPushButton('Transfer Director Role...')
        self.transfer_button.clicked.connect(self.transfer)
        row.addWidget(self.transfer_button)
        row.addStretch()
        layout.addLayout(row)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Close)
        self.save_table = buttons.button(QDialogButtonBox.StandardButton.Save)
        self.save_table.setText('Save Table Edits')
        buttons.accepted.connect(self.save)
        buttons.rejected.connect(self.reject)
        if embedded:
            buttons.button(QDialogButtonBox.StandardButton.Close).hide()
        layout.addWidget(buttons)
        self.refresh(initial)
        from gui.shared_profiles import profile_sync_service
        profile_sync_service().changed.connect(self.add_discovered_profiles)

    def add_discovered_profiles(self):
        """Update assignee choices without discarding any in-progress table edits."""
        added = False
        for identity, profile in load_contributor_profiles().items():
            if identity in self.profiles:
                continue
            self.profiles[identity] = {key: profile[key] for key in ('id', 'name')}
            for actor, _ in self.controls.values():
                actor.addItem(profile['name'] + ' (' + identity[:8] + ')', identity)
            added = True
        if added:
            self.refresh_roster()

    def refresh(self, initial=None):
        for identity, profile in load_contributor_profiles().items():
            self.profiles.setdefault(identity, {key: profile[key] for key in ('id', 'name')})
        def load(progress, cancelled):
            view = PublishedProject(self.project, self.configuration)
            data = read_assignments(view)
            if not data['coordinator']:
                raise ValueError('Set the project director in the Assignments workspace first.')
            if data['coordinator']['id'] != self.contributor['id']:
                raise ValueError('Assignment management belongs to ' + data['coordinator']['name'] +
                                 '. Use My Assigned Work for your assignments.')
            snapshot, _ = latest_snapshot(view)
            return data, catalog(snapshot), {v: assignment_status(view, e) for v, e in data['videos'].items()}
        self.data, self.available, self.statuses = initial or SyncProgressDialog('Read Video Assignments', load, self).run()
        for entry in self.data['videos'].values():
            self.profiles[entry['contributor']['id']] = entry['contributor']
        for actor, frames in getattr(self, 'controls', {}).values():
            actor.blockSignals(True)
            frames.blockSignals(True)
            actor.hide()
            frames.hide()
        self.table.blockSignals(True)
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
            check = self.table.item(row, 0)
            check.setFlags(check.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            check.setCheckState(Qt.CheckState.Unchecked)
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
            actor.currentIndexChanged.connect(self.update_edit_state)
            frames.textChanged.connect(self.update_edit_state)
        ready = sum(status == 'Ready for Review' for status, _ in self.statuses.values())
        self.summary.setText(f'{len(self.available) - len(self.data["videos"])} unassigned | '
                             f'{len(self.data["videos"])} assigned | {ready} ready for review | Project director: '
                             + self.data['coordinator']['name'])
        self.transfer_button.setEnabled(bool(self.data['coordinator']))
        self.table.blockSignals(False)
        self.refresh_roster()
        self.filter_videos()
        self.update_edit_state()

    def update_edit_state(self):
        old = {v: {k: e[k] for k in ('contributor', 'frames')} for v, e in self.data['videos'].items()}
        try:
            changed = self.choices() != old
        except ValueError:
            changed = True
        self.save_table.setEnabled(changed)

    def show_contributors(self, visible):
        self.contributors_panel.setVisible(visible)
        self.contributors_toggle.setArrowType(Qt.ArrowType.DownArrow if visible else Qt.ArrowType.RightArrow)

    def refresh_roster(self):
        self.roster.clear()
        for profile in sorted(self.profiles.values(), key=lambda p: p['name'].casefold()):
            self.roster.addItem(profile['name'] + ' (' + profile['id'][:8] + ')')
        self.contributors_toggle.setText(f'Contributors ({len(self.profiles)})')

    def filter_videos(self):
        for row, video in enumerate(self.videos):
            entry = self.data['videos'].get(video)
            name = entry['contributor']['name'] if entry else ''
            status = self.statuses.get(video, ('Unassigned', None))[0]
            mode = self.filter.currentText()
            visible = (self.search.text().casefold() in (video + ' ' + name).casefold() and
                       (mode == 'All videos' or mode == status or mode == 'Assigned' and bool(entry)))
            self.table.setRowHidden(row, not visible)
            if not visible:
                self.table.item(row, 0).setCheckState(Qt.CheckState.Unchecked)
        self.update_selection()

    def checked_videos(self):
        return [video for row, video in enumerate(self.videos) if not self.table.isRowHidden(row)
                and self.table.item(row, 0).checkState() == Qt.CheckState.Checked]

    def update_selection(self):
        count = len(self.checked_videos())
        self.assign_selected.setEnabled(bool(count))
        self.assign_selected.setText(f'Assign Selected Videos ({count})...' if count else 'Assign Selected Videos...')

    def check_visible(self):
        for row in range(self.table.rowCount()):
            if not self.table.isRowHidden(row):
                self.table.item(row, 0).setCheckState(Qt.CheckState.Checked)

    def clear_checked(self):
        for row in range(self.table.rowCount()):
            self.table.item(row, 0).setCheckState(Qt.CheckState.Unchecked)

    def bulk_assign(self):
        selected = self.checked_videos()
        if not selected:
            return
        try:
            old = {v: {k: e[k] for k in ('contributor', 'frames')} for v, e in self.data['videos'].items()}
            if self.choices() != old:
                raise ValueError('Save or refresh your table edits before making a bulk assignment.')
            dialog = BulkAssignmentDialog(selected, self.available, self.profiles, self)
            if dialog.exec() != QDialog.DialogCode.Accepted:
                return
            choices = dict(old, **dialog.choices)
            SyncProgressDialog('Assign Work', lambda progress, cancelled:
                save_assignments(self.project, self.configuration, self.contributor,
                                 choices, self.data['revision']), self).run()
            self.refresh()
            self.assignments_changed.emit()
        except Exception as exc:
            QMessageBox.warning(self, 'Assignments Not Saved', str(exc))

    def transfer(self):
        try:
            old = {v: {k: e[k] for k in ('contributor', 'frames')} for v, e in self.data['videos'].items()}
            if self.choices() != old:
                raise ValueError('Save or refresh your assignment choices before transferring the role.')
            profiles = [p for identity, p in self.profiles.items() if identity != self.contributor['id']]
            if not profiles:
                raise ValueError('Import the new director\'s contributor profile first.')
            names = [p['name'] + ' (' + p['id'][:8] + ')' for p in profiles]
            name, accepted = QInputDialog.getItem(self, 'Transfer Project Director', 'New director:', names, 0, False)
            if not accepted:
                return
            successor = profiles[names.index(name)]
            if QMessageBox.question(self, 'Transfer Director Role?',
                    f"Make {successor['name']} the only director of this project?\n\n"
                    'You will no longer be able to manage its assignments or accept submissions. '
                    'Existing video assignments and annotations will not change.',
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
                    QMessageBox.StandardButton.Cancel) != QMessageBox.StandardButton.Yes:
                return
            SyncProgressDialog('Transfer Project Director', lambda progress, cancelled:
                transfer_director(self.project, self.configuration, self.contributor, successor,
                                  self.data['revision']), self).run()
            self.assignments_changed.emit()
            self.accept()
        except Exception as exc:
            QMessageBox.warning(self, 'Director Role Not Transferred', str(exc))

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
            remembered = setting_json('assignments/profiles', {})
            remembered[profile['id']] = self.profiles[profile['id']]
            save_setting('assignments/profiles', remembered)
            for actor, _ in self.controls.values():
                index = actor.findData(profile['id'])
                text = profile['name'] + ' (' + profile['id'][:8] + ')'
                if index < 0:
                    actor.addItem(text, profile['id'])
                else:
                    actor.setItemText(index, text)
            self.refresh_roster()
            self.contributors_toggle.setChecked(True)
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
            self.assignments_changed.emit()
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
            self.assignments_changed.emit()
        except Exception as exc:
            QMessageBox.warning(self, 'Submission Not Accepted', str(exc))


class BulkAssignmentDialog(QDialog):
    def __init__(self, videos, available, profiles, parent=None):
        super().__init__(parent)
        self.videos, self.available, self.profiles = videos, available, profiles
        self.choices = None
        self.setWindowTitle('Assign Selected Videos')
        self.resize(600, 280)
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel(f'{len(videos)} videos selected'))
        form = QFormLayout()
        self.contributor = QComboBox()
        for identity, profile in sorted(profiles.items(), key=lambda pair: pair[1]['name'].casefold()):
            self.contributor.addItem(profile['name'] + ' (' + identity[:8] + ')', identity)
        self.contributor.setCurrentIndex(-1)
        self.scope = QComboBox()
        self.scope.addItems(['All extracted frames', 'Specific frames'])
        self.frames = QLineEdit()
        self.frames.setPlaceholderText('Example: 0,5-10')
        self.frames.setEnabled(False)
        self.scope.currentIndexChanged.connect(lambda index: self.frames.setEnabled(index == 1))
        form.addRow('Assign to', self.contributor)
        form.addRow('Frames', self.scope)
        form.addRow('Frame numbers', self.frames)
        layout.addLayout(form)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.button(QDialogButtonBox.StandardButton.Ok).setText('Review Assignment...')
        buttons.accepted.connect(self.review)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def review(self):
        try:
            actor = self.profiles.get(self.contributor.currentData())
            if not actor:
                raise ValueError('Choose a contributor.')
            ranges = self.frames.text() if self.scope.currentIndex() else 'All'
            choices = {v: {'contributor': actor, 'frames': parse_frames(ranges, self.available[v]['frames'])}
                       for v in self.videos}
            details = '\n'.join(f'{v}: {len(c["frames"])} frames' for v, c in choices.items())
            review = TransferConfirmationDialog('Review Assignment',
                f'Assign to: {actor["name"]}\nFrames: {ranges}\n\n{details}\n\n'
                'Changing an existing assignee or frame range retires that assignment. '
                'Keep any unpublished work for director review before reassigning.\n\n'
                'This saves assignments only. The contributor downloads their selected files on their own computer.',
                'Assign Work', self)
            if review.exec() == QDialog.DialogCode.Accepted:
                self.choices = choices
                self.accept()
        except ValueError as exc:
            QMessageBox.warning(self, 'Check Assignment', str(exc))


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
        actions = QHBoxLayout()
        actions.addWidget(buttons, 1)
        actions.addWidget(help_button(self, 'download'))
        layout.addLayout(actions)
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
