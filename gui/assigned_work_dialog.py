"""Coordinator assignments and contributor downloads, without exposing archive internals."""

from pathlib import Path

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (QAbstractItemView, QComboBox, QDialog, QDialogButtonBox,
    QFileDialog, QFormLayout, QHBoxLayout, QHeaderView, QInputDialog, QLabel, QLineEdit, QListWidget, QMessageBox,
    QPushButton, QStyle, QTableWidget, QTableWidgetItem, QToolButton, QVBoxLayout, QWidget)

from core.contributors import contributor_profile
from core.project_sync import _read_json
from core.project_update import PublishedProject
from core.assigned_work import (accept_submission, assignment_status, catalog,
    latest_snapshot, my_projects, parse_frames, read_assignments, save_assignments,
    transfer_director, project_members, add_project_member)
from core.assigned_workspace import (PreparedAssignedDownload, managed_directory, matching_copy, existing_copies,
                                     assigned_download_overview, DOWNLOAD_NEEDED, DOWNLOAD_LABELS)
from gui.project_sync_dialog import (SyncProgressDialog, TransferConfirmationDialog,
                                     setting_json, save_setting, load_contributor_profiles,
                                     remember_contributor_profile)
from gui.workflow_help import help_button
from gui.annotation_work import WorkFilters, task_combo
from gui.assigned_review import AssignedReviewViews
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


def managed_copy_key(configuration, project_id, contributor):
    return '/'.join((configuration['root_id'], project_id, contributor['id']))


def assigned_local_path(configuration, project_id, contributor):
    saved = setting_json('assigned_work/locations', {}).get(managed_copy_key(configuration, project_id, contributor))
    path = Path(saved) if saved else managed_directory(configuration, project_id, contributor)
    if saved and not path.is_dir():
        raise ValueError('Your saved working copy is unavailable:\n' + str(path) +
                         '\n\nReconnect or restore this folder. No replacement copy has been created.')
    if saved or path.exists():
        matching_copy(path, configuration, project_id, contributor)
        return str(path)
    return ''


class AssignedDownloadDialog(QDialog):
    DOWNLOAD_COLUMN = 4

    def __init__(self, configuration, contributor, parent=None, *, project_id=None):
        super().__init__(parent)
        self.configuration, self.contributor = configuration, contributor
        self.download_result = None
        self.setWindowTitle('My Assigned Work')
        self.resize(1100, 560)
        layout = QVBoxLayout(self)
        self.projects = QComboBox()
        layout.addWidget(self.projects)
        filters = QFormLayout()
        review_row = QHBoxLayout()
        self.review_view = QComboBox()
        for label, value in AssignedReviewViews.VIEWS:
            self.review_view.addItem(label, value)
        self.review_view.setToolTip('Published review status, separate from local mask completion.')
        review_row.addWidget(self.review_view, 1)
        self.refresh_button = QToolButton()
        self.refresh_button.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_BrowserReload))
        self.refresh_button.setToolTip('Refresh review and download status')
        self.refresh_button.setAccessibleName('Refresh review and download status')
        self.refresh_button.clicked.connect(self.refresh_status)
        review_row.addWidget(self.refresh_button)
        filters.addRow('Review:', review_row)
        self.local_filter = QComboBox()
        for label, value in (('All videos', 'all'), ('Needs download / refresh', 'needed'),
                             ('Already downloaded', 'downloaded'), ('Not checked', 'unknown')):
            self.local_filter.addItem(label, value)
        filters.addRow('On this computer:', self.local_filter)
        layout.addLayout(filters)
        self.table = _table(['Video', 'Review status', 'On this computer', 'Assigned frames', 'Download'])
        layout.addWidget(self.table)
        selection = QHBoxLayout()
        select_visible = QPushButton('Select Visible')
        select_visible.clicked.connect(lambda: self.set_checked(True))
        clear = QPushButton('Clear Selection')
        clear.clicked.connect(lambda: self.set_checked(False))
        selection.addWidget(select_visible)
        selection.addWidget(clear)
        selection.addStretch()
        layout.addLayout(selection)
        self.status_note = QLabel()
        self.status_note.setWordWrap(True)
        self.status_note.setTextFormat(Qt.TextFormat.PlainText)
        layout.addWidget(self.status_note)
        self.summary = QLabel('')
        self.summary.setWordWrap(True)
        layout.addWidget(self.summary)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Cancel)
        self.open_copy = buttons.addButton('Open Working Copy', QDialogButtonBox.ButtonRole.ActionRole)
        self.open_copy.clicked.connect(self.open_working_copy)
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
        self.review_view.currentIndexChanged.connect(self.filter_videos)
        self.local_filter.currentIndexChanged.connect(self.filter_videos)
        self.table.itemChanged.connect(self.update_selection)
        self.populate()

    def populate(self):
        index = self.projects.currentIndex()
        videos = self.available[index]['videos'] if index >= 0 else {}
        self.overview = None
        self.overview_error = ''
        for combo in (self.review_view, self.local_filter):
            combo.blockSignals(True)
            combo.setCurrentIndex(0)
            combo.blockSignals(False)
        self.videos = sorted(videos)
        self.table.blockSignals(True)
        self.table.setRowCount(len(self.videos))
        for row, video in enumerate(self.videos):
            item = QTableWidgetItem(video)
            item.setToolTip(video)
            self.table.setItem(row, 0, item)
            self.table.setItem(row, 1, QTableWidgetItem('Not checked'))
            self.table.setItem(row, 2, QTableWidgetItem('Not checked'))
            self.table.setItem(row, 3, QTableWidgetItem(str(len(videos[video]['frames']))))
            check = QTableWidgetItem()
            check.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsUserCheckable)
            check.setCheckState(Qt.CheckState.Unchecked)
            self.table.setItem(row, self.DOWNLOAD_COLUMN, check)
        self.table.blockSignals(False)
        self.refresh_status(initial=True)

    def known_copy_paths(self, project_id):
        parent = self.parent()
        paths = list(getattr(parent, 'recent_projects', []))
        paths.append(getattr(parent, 'project_path', None))
        workflow = getattr(parent, 'workflow', None)
        if workflow:
            settings = workflow.settings
            actor, root = self.contributor['id'], self.configuration['root_id']
            paths += settings.value('workflow/recent_projects/' + actor, [], type=list)
            paths.append(settings.value(f'workflow/shared_projects/{actor}/{root}/{project_id}', '', type=str))
        return paths

    def refresh_status(self, _checked=False, *, initial=False):
        project_id = self.projects.currentData()
        self.overview, self.overview_error = None, ''
        self.download.setEnabled(False)
        self.open_copy.setEnabled(False)
        if self.videos:
            saved = setting_json('assigned_work/locations', {}).get(
                managed_copy_key(self.configuration, project_id, self.contributor))
            known = self.known_copy_paths(project_id)
            assignments = self.available[self.projects.currentIndex()]['videos']
            try:
                self.overview = SyncProgressDialog('Check Assigned Work', lambda progress, cancelled:
                    assigned_download_overview(self.configuration, project_id, self.contributor, assignments,
                        saved_path=saved, known_paths=known, progress=progress, cancelled=cancelled), self).run()
            except Exception as exc:
                self.overview_error = str(exc)
        data = self.overview or {}
        self.table.blockSignals(True)
        for row, video in enumerate(self.videos):
            self.table.item(row, 1).setText(data.get('statuses', {}).get(video, 'Not checked'))
            local = data.get('local', {}).get(video, 'unknown')
            self.table.item(row, 2).setText(DOWNLOAD_LABELS[local])
            self.table.item(row, 2).setToolTip(data.get('local_error') or
                'Frame availability and published-version metadata only. '
                'The transfer preview checks saved local edits and conflicts before updating files.')
            if initial:
                self.table.item(row, self.DOWNLOAD_COLUMN).setCheckState(
                    Qt.CheckState.Checked if local in DOWNLOAD_NEEDED else Qt.CheckState.Unchecked)
        for i, (label, value) in enumerate(AssignedReviewViews.VIEWS):
            count = sum(AssignedReviewViews.group(data.get('statuses', {}).get(v)) == value for v in self.videos)
            if value == 'all':
                count = len(self.videos)
            elif self.videos and not self.overview:
                count = '?'
            self.review_view.setItemText(i, f'{label} ({count})')
        self.table.blockSignals(False)
        self.open_copy.setEnabled(bool(data.get('local_path')))
        self.open_copy.setToolTip(data.get('local_path', ''))
        self.filter_videos()

    def selected_videos(self):
        return [v for row, v in enumerate(self.videos) if not self.table.isRowHidden(row)
                and self.table.item(row, self.DOWNLOAD_COLUMN).checkState() == Qt.CheckState.Checked]

    def set_checked(self, checked):
        self.table.blockSignals(True)
        for row in range(self.table.rowCount()):
            if not checked or not self.table.isRowHidden(row):
                self.table.item(row, self.DOWNLOAD_COLUMN).setCheckState(
                    Qt.CheckState.Checked if checked else Qt.CheckState.Unchecked)
        self.table.blockSignals(False)
        self.update_selection()

    def filter_videos(self):
        data = self.overview or {}
        selected_review, selected_local = self.review_view.currentData(), self.local_filter.currentData()
        self.table.blockSignals(True)
        for row, video in enumerate(self.videos):
            local = data.get('local', {}).get(video, 'unknown')
            visible = (selected_review == 'all' or
                       AssignedReviewViews.group(data.get('statuses', {}).get(video)) == selected_review) and (
                selected_local == 'all' or selected_local == local or
                (selected_local == 'needed' and local in DOWNLOAD_NEEDED))
            self.table.setRowHidden(row, not visible)
            if not visible:
                self.table.item(row, self.DOWNLOAD_COLUMN).setCheckState(Qt.CheckState.Unchecked)
        self.table.blockSignals(False)
        self.update_selection()

    def update_selection(self):
        count = len(self.selected_videos())
        shown = sum(not self.table.isRowHidden(row) for row in range(len(self.videos)))
        self.download.setEnabled(bool(count and self.overview))
        self.download.setText(f'Download Selected ({count})...' if count else 'Download Selected...')
        self.summary.setText(f'Contributor: {self.contributor["name"]} | {len(self.videos)} assigned videos | '
                             f'{shown} shown | {count} selected' if self.videos else
                             'No assignments for this contributor profile. Ask the project director to add you in People and assign videos.')
        error = self.overview_error or (self.overview or {}).get('local_error')
        self.status_note.setText(('Status check incomplete: ' + error) if error else
                                 'No videos match these filters.' if self.videos and not shown else '')
        self.status_note.setVisible(bool(self.status_note.text()))

    def open_working_copy(self):
        path = (self.overview or {}).get('local_path')
        if not path:
            return
        try:
            matching_copy(path, self.configuration, self.projects.currentData(), self.contributor)
            locations = setting_json('assigned_work/locations', {})
            locations[managed_copy_key(self.configuration, self.projects.currentData(), self.contributor)] = path
            save_setting('assigned_work/locations', locations)
            self.download_result = {'path': path, 'updated': False, 'bytes': 0}
            self.accept()
        except Exception as exc:
            QMessageBox.warning(self, 'Working Copy Unavailable', str(exc))

    def download_selected(self):
        selected = self.selected_videos()
        if not selected or not self.overview:
            return
        try:
            project_id = self.projects.currentData()
            destination = self.choose_working_copy(project_id)
            if destination is None:
                return
            plan = SyncProgressDialog('Estimate Assigned Download', lambda progress, cancelled:
                PreparedAssignedDownload(self.configuration, project_id, self.contributor, selected,
                                         destination, progress, cancelled), self).run()
            details = (f'{len(plan.new_videos)} new videos | {len(plan.updated_videos)} updated videos\n'
                       f'{plan.file_count} files to add/update/remove | {plan.bytes / 1024**2:.1f} MiB to write\n\n'
                       f'Local working copy:\n{destination}\n\n'
                       'Existing local edits are kept. Original videos are excluded. '
                       'This same folder will be used for future assignment downloads.')
            rollback_bytes = sum(record['size'] for record in plan.before.values() if record)
            if rollback_bytes:
                details += f'\n\nRollback copies of affected files: {rollback_bytes / 1024**2:.1f} MiB.'
            if plan.kept_edits:
                details += '\n\nLocal edits kept: ' + ', '.join(plan.kept_edits)
            if plan.retained:
                details += ('\n\nNo longer assigned (files kept locally, hidden from active work and not uploaded): '
                            + ', '.join(plan.retained))
            if TransferConfirmationDialog('Update My Assigned Work', details,
                    'Download and Open' if plan.file_count else 'Open Working Copy', self).exec() != QDialog.DialogCode.Accepted:
                return
            self.download_result = SyncProgressDialog('Download Assigned Work', lambda progress, cancelled:
                plan.apply(progress, cancelled), self).run()
            locations = setting_json('assigned_work/locations', {})
            locations[managed_copy_key(self.configuration, project_id, self.contributor)] = str(destination)
            save_setting('assigned_work/locations', locations)
            if self.download_result.get('warning'):
                QMessageBox.warning(self, 'Working Copy Updated', self.download_result['warning'])
            self.accept()
        except Exception as exc:
            QMessageBox.warning(self, 'Work Not Downloaded', str(exc))

    def choose_working_copy(self, project_id):
        registered = assigned_local_path(self.configuration, project_id, self.contributor)
        if registered:
            return Path(registered)
        copies = existing_copies(self.known_copy_paths(project_id), self.configuration, project_id, self.contributor)
        if copies:
            choice, accepted = QInputDialog.getItem(self, 'Continue an Existing Working Copy',
                'Choose the copy to keep using on this computer. Other copies are left untouched.\n'
                'If edits are spread across copies, ask the director to reconcile them before continuing.',
                [str(path) for path in copies], 0, False)
            return Path(choice) if accepted else None
        return managed_directory(self.configuration, project_id, self.contributor)
