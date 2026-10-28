"""Coordinator assignments and contributor downloads, without exposing archive internals."""

from pathlib import Path

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (QAbstractItemView, QComboBox, QDialog, QDialogButtonBox,
    QFileDialog, QFormLayout, QHBoxLayout, QHeaderView, QInputDialog, QLabel, QLineEdit, QListWidget, QMessageBox,
    QPushButton, QStyle, QTableWidget, QTableWidgetItem, QToolButton, QVBoxLayout, QWidget)

from core.contributors import contributor_profile
from core.project_sync import _read_json
from core.project_update import PublishedProject
from core.assigned_work import (accept_submission, assignment_status, catalog, download_plan,
    download_work, latest_snapshot, my_projects, parse_frames, read_assignments, save_assignments,
    transfer_director, project_members, add_project_member)
from gui.project_sync_dialog import (SyncProgressDialog, TransferConfirmationDialog,
                                     setting_json, save_setting, load_contributor_profiles,
                                     remember_contributor_profile)
from gui.workflow_help import help_button
from gui.annotation_work import WorkFilters, task_combo
from gui.video_browsing import VideoBrowserControls, ModifiedTableItem, SORT_ROLE, VideoRowColors
from core.annotation_work import published_index, summarize, task_label
from core.shared_project_review import compare_saved_annotations
from gui.shared_version import SharedVersionHeader


def _table(headers):
    table = QTableWidget(0, len(headers))
    table.setHorizontalHeaderLabels(headers)
    VideoRowColors(table)
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
        self.profiles = {}
        self.scanning_masks = False
        self.setWindowTitle('Video Assignments')
        self.resize(1080, 580)
        layout = QVBoxLayout(self)
        self.shared_version = SharedVersionHeader(project, configuration, self)
        self.shared_version.refresh_requested.connect(self.confirm_refresh)
        self.shared_version.compare_requested.connect(self.compare_with_shared)
        self.finished.connect(self.shared_version.cancel)
        layout.addWidget(self.shared_version)
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
        self.video_browser = VideoBrowserControls()
        layout.addWidget(self.video_browser)
        self.video_browser.changed.connect(self.filter_videos)
        self.finished.connect(self.video_browser.cancel)
        self.work_filters = WorkFilters()
        layout.addWidget(self.work_filters)
        self.work_filters.changed.connect(self.filter_videos)
        mask_status_row = QHBoxLayout()
        self.mask_status = QLabel()
        self.mask_status.setWordWrap(True)
        mask_status_row.addWidget(self.mask_status, 1)
        self.scan_masks = QPushButton('Scan Mask Status')
        self.scan_masks.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_BrowserReload))
        self.scan_masks.setToolTip('Read published annotations to fill in missing mask summaries. '
                                   'No frame images are downloaded and no annotations are changed.')
        self.scan_masks.clicked.connect(self.scan_mask_status)
        mask_status_row.addWidget(self.scan_masks)
        layout.addLayout(mask_status_row)
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
        self.table = _table(['Video', 'Split', 'Published frames', 'Contributor', 'Assigned frames',
                             'Submission', 'Masks', 'Annotation work', 'Assigned task',
                             'Experiment / setup', 'Modified (local)'])
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Interactive)
        self.table.setColumnWidth(0, 350)
        self.table.itemChanged.connect(self.update_selection)
        layout.addWidget(self.table)
        self.search.textChanged.connect(self.filter_videos)
        self.filter.currentIndexChanged.connect(self.filter_videos)
        self.contributors_toggle = QToolButton()
        self.contributors_toggle.setText('People')
        self.contributors_toggle.setCheckable(True)
        self.contributors_toggle.setArrowType(Qt.ArrowType.RightArrow)
        self.contributors_toggle.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        contributors_header = QHBoxLayout()
        contributors_header.addWidget(self.contributors_toggle)
        contributors_header.addStretch()
        layout.insertLayout(2, contributors_header)
        self.contributors_panel = QWidget()
        contributors_layout = QVBoxLayout(self.contributors_panel)
        self.roster = QListWidget()
        self.roster.setMaximumHeight(110)
        contributors_layout.addWidget(self.roster)
        add_button = QPushButton('Add Contributor...')
        add_button.clicked.connect(self.add_contributor)
        contributors_header.addWidget(add_button)
        import_button = QPushButton('Import Profile...')
        import_button.clicked.connect(self.import_profile)
        contributors_layout.addWidget(import_button)
        layout.insertWidget(3, self.contributors_panel)
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

    def refresh(self, initial=None):
        def load(progress, cancelled):
            progress('Reading published assignments...')
            view = PublishedProject(self.project, self.configuration)
            data = read_assignments(view)
            if not data['coordinator']:
                raise ValueError('Set the project director in the Assignments workspace first.')
            if data['coordinator']['id'] != self.contributor['id']:
                raise ValueError('Assignment management belongs to ' + data['coordinator']['name'] +
                                 '. Use My Assigned Work for your assignments.')
            snapshot, _ = latest_snapshot(view)
            return (data, catalog(snapshot, include_work=True, progress=progress, cancelled=cancelled),
                    {v: assignment_status(view, e) for v, e in data['videos'].items()}, snapshot)
        loaded = initial or SyncProgressDialog('Read Video Assignments', load, self).run()
        self.data, self.available, self.statuses = loaded[:3]
        self.snapshot = loaded[3] if len(loaded) > 3 else None
        self.shared_version.set_snapshot(self.snapshot)
        self.profiles = project_members(self.data)
        self.video_browser.set_context(self.project, self.available,
            {v: entry['experiment'] for v, entry in self.available.items() if entry.get('experiment')})
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
                self.table.item(row, col).setToolTip(value)
            check = self.table.item(row, 0)
            check.setData(Qt.ItemDataRole.UserRole, video)
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
        self.summary.setText('Published annotations | ' + f'{len(self.available) - len(self.data["videos"])} unassigned | '
                             f'{len(self.data["videos"])} assigned | {ready} ready for review | Project director: '
                             + self.data['coordinator']['name'])
        self.transfer_button.setEnabled(bool(self.data['coordinator']))
        self.table.blockSignals(False)
        self.refresh_roster()
        self.filter_videos()
        self.update_edit_state()
        self.update_mask_status()
        self.video_browser.schedule()

    def compare_with_shared(self):
        if not self.snapshot:
            return
        try:
            # Deliberately do not save or publish anything: the review is of files on disk.
            snapshot = self.snapshot
            review = SyncProgressDialog('Compare Local and Shared Annotations', lambda progress, cancelled:
                compare_saved_annotations(self.project, snapshot, progress=progress, cancelled=cancelled), self).run()
            from gui.publication_review_dialog import PublicationReviewDialog
            PublicationReviewDialog({'review': review}, self).exec()
        except Exception as exc:
            QMessageBox.warning(self, 'Comparison Not Completed', str(exc))

    def update_mask_status(self):
        summaries = [summarize(info.get('annotation_work') or {'scan_state': 'not_scanned'}, None)
                     for info in self.available.values()]
        unscanned = sum(summary['state'] == 'not_scanned' for summary in summaries)
        unavailable = sum(summary['state'] == 'unknown' for summary in summaries)
        missing = unscanned + unavailable
        text = f'Mask status available for {len(self.available) - missing}/{len(self.available)} videos'
        if self.scanning_masks:
            text += ' | Scanning published mask status...'
        else:
            if unscanned:
                text += f' | {unscanned} not scanned; use Scan Mask Status'
            if unavailable:
                text += f' | {unavailable} unavailable; see row details'
        self.mask_status.setText(text)
        self.scan_masks.setEnabled(bool(missing) and not self.scanning_masks)

    def work_summary(self, video, task):
        entry = self.available[video].get('annotation_work') or {'scan_state': 'not_scanned'}
        summary = summarize(entry, task)
        if self.scanning_masks and summary['state'] in ('not_scanned', 'unknown'):
            return {'presence': 'loading', 'state': 'loading', 'mask_text': 'Scanning...',
                    'work_text': 'Scanning...', 'detail': 'Reading published annotations; status is not yet available.'}
        return summary

    def scan_mask_status(self):
        if self.scanning_masks:
            return
        self.scanning_masks = True
        self.filter_videos()
        self.update_mask_status()
        error = None
        try:
            def scan(progress, cancelled):
                snapshot = self.snapshot
                if snapshot is None:
                    progress('Reading the published project version...')
                    snapshot, _ = latest_snapshot(PublishedProject(self.project, self.configuration))
                return snapshot, published_index(snapshot, progress=progress, cancelled=cancelled)
            snapshot, index = SyncProgressDialog('Scan Published Mask Status', scan, self).run()
            self.snapshot = snapshot
            if self.shared_version.key() != (tuple(snapshot['reference']), snapshot['manifest_hash']):
                self.shared_version.set_snapshot(snapshot)
            for video, entry in self.available.items():
                entry['annotation_work'] = index['videos'].get(video)
        except Exception as exc:
            error = str(exc)
        finally:
            self.scanning_masks = False
            self.filter_videos()
            self.update_mask_status()
            if error is not None:
                self.mask_status.setText(self.mask_status.text() + '\nMask status scan not completed: ' + error)

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
            role = 'Director' if profile['id'] == self.data['coordinator']['id'] else 'Contributor'
            self.roster.addItem(profile['name'] + ' (' + profile['id'][:8] + ') | ' + role)
        self.contributors_toggle.setText(f'People ({len(self.profiles)})')

    def filter_videos(self):
        if not hasattr(self, 'videos'):
            return
        task = self.work_filters.task.currentData()
        visibility = {}
        for row, video in enumerate(self.videos):
            entry = self.data['videos'].get(video)
            name = entry['contributor']['name'] if entry else ''
            status = self.statuses.get(video, ('Unassigned', None))[0]
            mode = self.filter.currentText()
            summary = self.work_summary(video, task)
            for col, value in ((6, summary['mask_text']), (7, summary['work_text']),
                               (8, task_label((entry or {}).get('task'))),
                               (9, self.available[video].get('experiment') or 'Not assigned')):
                item = QTableWidgetItem(value)
                item.setToolTip('Published annotations. ' + summary.get('detail', ''))
                self.table.setItem(row, col, item)
            modified = ModifiedTableItem(self.video_browser.date_text(video))
            modified.setData(SORT_ROLE, self.video_browser.key(video))
            modified.setToolTip(self.video_browser.tooltip(video) + '\nDates refer to this local project, not the shared backup.')
            self.table.setItem(row, 10, modified)
            visibility[video] = (self.video_browser.accepts(video) and
                       self.work_filters.accepts(summary, self.available[video]['split']) and
                       self.search.text().casefold() in (video + ' ' + name).casefold() and
                       (mode == 'All videos' or mode == status or mode == 'Assigned' and bool(entry)))
        self.table.sortItems(10 if self.video_browser.sort.currentData() == 'recent' else 0,
                             Qt.SortOrder.AscendingOrder)
        self.videos = [self.table.item(row, 0).data(Qt.ItemDataRole.UserRole)
                       for row in range(self.table.rowCount())]
        for row, video in enumerate(self.videos):
            self.table.setRowHidden(row, not visibility[video])
            if not visibility[video]:
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
            dialog = BulkAssignmentDialog(selected, self.available, self.profiles, self,
                                          task=self.work_filters.task.currentData())
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
                raise ValueError('Add the new director in People first.')
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
            if QMessageBox.question(self, 'Add Project Contributor?',
                    f"Add {profile['name']} ({profile['id'][:8]}) to this project and the connected lab?",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                    QMessageBox.StandardButton.No) == QMessageBox.StandardButton.Yes:
                self.enroll(profile)
        except Exception as exc:
            QMessageBox.warning(self, 'Profile Not Imported', str(exc))

    def add_contributor(self):
        from gui.project_people import AddContributorDialog
        dialog = AddContributorDialog(self.profiles, self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            try:
                self.enroll(dialog.profile)
            except Exception as exc:
                QMessageBox.warning(self, 'Contributor Not Added', str(exc))

    def enroll(self, profile):
        local = load_contributor_profiles().get(profile['id'])
        portable = contributor_profile(dict(profile, format=1, kind='bumblebox-contributor'))
        if local and local != portable:
            raise ValueError('This identity has a different local name. No existing profile was overwritten.')
        data = SyncProgressDialog('Add Project Contributor', lambda progress, cancelled:
            add_project_member(self.project, self.configuration, self.contributor, profile,
                               self.data['revision']), self).run()
        remember_contributor_profile(portable)
        sharing = setting_json('contributors/shared', {})
        if sharing.get('enabled') and sharing.get('root_id') == self.configuration.get('root_id'):
            sharing['publish_ids'] = sorted(set(sharing.get('publish_ids', [])) | {profile['id']})
            save_setting('contributors/shared', sharing)
            from gui.shared_profiles import profile_sync_service
            profile_sync_service().refresh()
        # Adding a person changes only the roster, never pending table choices.
        self.data = data
        self.profiles = project_members(data)
        for actor, _ in self.controls.values():
            if actor.findData(profile['id']) < 0:
                actor.addItem(profile['name'] + ' (' + profile['id'][:8] + ')', profile['id'])
        self.refresh_roster()
        self.contributors_toggle.setChecked(True)
        self.assignments_changed.emit()

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
    def __init__(self, videos, available, profiles, parent=None, *, task=None):
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
        self.task = task_combo(self)
        self.task.setCurrentIndex(self.task.findData(task))
        self.scope = QComboBox()
        self.scope.addItems(['All extracted frames', 'Specific frames'])
        self.frames = QLineEdit()
        self.frames.setPlaceholderText('Example: 0,5-10')
        self.frames.setEnabled(False)
        self.scope.currentIndexChanged.connect(lambda index: self.frames.setEnabled(index == 1))
        form.addRow('Assign to', self.contributor)
        form.addRow('Work on', self.task)
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
            for choice in choices.values():
                choice['task'] = self.task.currentData()
            details = '\n'.join(f'{v}: {len(c["frames"])} frames' for v, c in choices.items())
            review = TransferConfirmationDialog('Review Assignment',
                f'Assign to: {actor["name"]}\nWork on: {task_label(self.task.currentData())}\nFrames: {ranges}\n\n{details}\n\n'
                'Changing an existing assignee, task, or frame range retires that assignment. '
                'Keep any unpublished work for director review before reassigning.\n\n'
                'This saves assignments only. The contributor downloads their selected files on their own computer.',
                'Assign Work', self)
            if review.exec() == QDialog.DialogCode.Accepted:
                self.choices = choices
                self.accept()
        except ValueError as exc:
            QMessageBox.warning(self, 'Check Assignment', str(exc))


class AssignedDownloadDialog(QDialog):
    def __init__(self, configuration, contributor, parent=None, *, project_id=None):
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
        if project_id is not None:
            self.projects.setCurrentIndex(self.projects.findData(project_id))
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
                             'No assignments for this contributor profile. Ask the project director to add you in People and assign videos.')

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
