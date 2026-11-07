"""Assignment discovery and explicit project-director setup."""

from pathlib import Path
from copy import deepcopy
from datetime import datetime, timezone

from PyQt6.QtCore import Qt, QTimer, QSize
from PyQt6.QtWidgets import (QBoxLayout, QComboBox, QDialog, QHeaderView, QHBoxLayout, QLabel,
                            QListWidget, QListWidgetItem, QMessageBox, QPushButton, QScrollArea,
                            QStyle, QToolButton, QTreeWidget, QTreeWidgetItem, QVBoxLayout, QWidget)

from core.assigned_work import (assignment_management, assignment_presence, establish_director,
                                require_contributor, shared_assignment_progress)
from core.project_sync import PROJECT_MARKER, _read_json
from gui.assigned_work_dialog import AssignmentManagerDialog
from gui.project_sync_dialog import (OperationWorker, SyncProgressDialog,
                                     setting_json)


def text_label(text=''):
    label = QLabel(text)
    label.setTextFormat(Qt.TextFormat.PlainText)
    label.setWordWrap(True)
    return label


class AssignmentInbox(QWidget):
    """A failed or stale background check must never become a 'no assignments' notice."""
    def __init__(self, controller):
        super().__init__()
        self.controller = controller
        self.window = window = controller.window
        self.worker = None
        self.notice = None
        self.shown_empty = set()
        self.stopping = False
        self.result = None
        self.shared_projects = []
        self.project_actor = None
        self.progress_worker = None
        self.progress_context = None
        self.progress_result = None
        self.progress_error = None
        self.progress_requested = False
        self.progress_timer = QTimer(self)
        self.progress_timer.setSingleShot(True)
        self.progress_timer.timeout.connect(self.check_progress)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 8, 0, 8)
        row = QHBoxLayout()
        self.summary = text_label('Assignments not checked')
        row.addWidget(self.summary, 1)
        self.check = QToolButton()
        self.check.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_BrowserReload))
        self.check.setAccessibleName('Check assignments again')
        self.check.setToolTip('Check assignments again')
        self.check.clicked.connect(self.retry_connection)
        row.addWidget(self.check)
        layout.addLayout(row)
        self.share = QPushButton('Shared Drive Settings...')
        self.share.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_DriveNetIcon))
        self.share.setToolTip('View or change the shared folder used for profiles, assignments, and backups')
        self.share.clicked.connect(self.configure_profiles)
        self.configure = QPushButton('Set Up Shared Drive...')
        self.configure.clicked.connect(self.connect_folder)
        self.manage = QPushButton('Manage Assignments...')
        self.manage.clicked.connect(self.manage_selected)
        for button in (self.share, self.configure):
            row.insertWidget(row.count() - 1, button)
            button.hide()
        from gui.drive_login import DriveLoginButton
        self.login = DriveLoginButton(self)
        self.login.completed.connect(self.retry_connection)
        layout.addWidget(self.login)
        self.login.hide()
        self.body = QBoxLayout(QBoxLayout.Direction.LeftToRight)
        self.body.setSpacing(24)
        picker = QWidget()
        picker_layout = QVBoxLayout(picker)
        picker_layout.setContentsMargins(0, 0, 0, 0)
        picker_layout.addWidget(text_label('Projects'))
        self.projects = QListWidget()
        self.projects.setAccessibleName('Projects')
        self.projects.setMinimumWidth(200)
        self.projects.setMinimumHeight(180)
        self.projects.setTextElideMode(Qt.TextElideMode.ElideRight)
        self.projects.itemActivated.connect(lambda item: self.open_selected())
        self.projects.currentItemChanged.connect(self.selection_changed)
        picker_layout.addWidget(self.projects)
        self.body.addWidget(picker, 2)
        self.detail = QWidget()
        detail = QVBoxLayout(self.detail)
        detail.setContentsMargins(0, 0, 0, 0)
        detail.setSpacing(12)
        self.project_title = text_label('Choose a project')
        self.project_title.setObjectName('selectedProjectTitle')
        detail.addWidget(self.project_title)
        self.project_status = text_label()
        detail.addWidget(self.project_status)
        self.copy_label = text_label('Local working copy')
        detail.addWidget(self.copy_label)
        self.copies = QComboBox()
        self.copies.setAccessibleName('Local working copy')
        self.copies.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.copies.setMinimumContentsLength(12)
        self.copies.currentIndexChanged.connect(self.copy_changed)
        detail.addWidget(self.copies)
        self.assignment_summary = text_label()
        detail.addWidget(self.assignment_summary)
        self.videos = QTreeWidget()
        self.videos.setHeaderLabels(['Assigned video', 'Work on', 'Frames', 'Shared Status'])
        self.videos.headerItem().setToolTip(3, 'Published progress on the shared drive, not unuploaded local edits')
        self.videos.setRootIsDecorated(False)
        self.videos.setAlternatingRowColors(True)
        self.videos.setSelectionMode(QTreeWidget.SelectionMode.NoSelection)
        self.videos.setMinimumHeight(160)
        self.videos.header().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        for column in (1, 2, 3):
            self.videos.header().setSectionResizeMode(column, QHeaderView.ResizeMode.ResizeToContents)
        detail.addWidget(self.videos, 1)
        self.progress_note = text_label()
        detail.addWidget(self.progress_note)
        self.activity_note = text_label()
        detail.addWidget(self.activity_note)
        self.review_notice = QPushButton()
        self.review_notice.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_FileDialogDetailedView))
        self.review_notice.clicked.connect(self.review_selected)
        self.review_notice.hide()
        detail.addWidget(self.review_notice)
        self.open = QPushButton('Download Assigned Work...')
        self.open.setObjectName('openSelectedProject')
        self.open.setMinimumHeight(40)
        self.open.clicked.connect(self.open_selected)
        detail.addWidget(self.open)
        self.download_more = QPushButton('Download Assigned Work...')
        self.download_more.setToolTip('Add or refresh assignments in your existing working copy; local edits are kept.')
        self.download_more.clicked.connect(self.download_selected)
        detail.addWidget(self.download_more)
        detail.addWidget(self.manage)
        self.body.addWidget(self.detail, 3)
        layout.addLayout(self.body, 1)
        self.download_more.hide()
        self.selection_changed()
        self.timer = QTimer(self)
        self.timer.setInterval(60000)
        self.timer.timeout.connect(self.refresh)
        self.timer.start()

    def context(self):
        actor = self.window.contributor_session or {}
        configuration = setting_json('project_sync/destination', {})
        return (actor.get('id'), actor.get('session_id'), str(self.window.project_path or ''),
                configuration.get('root_id'), configuration.get('root'))

    def reset(self):
        self.result = None
        self.shared_projects = []
        self.project_actor = (self.window.contributor_session or {}).get('id')
        self.rebuild_projects()
        if self.notice:
            self.notice.close()
            self.notice = None
        QTimer.singleShot(0, self.refresh)

    def refresh(self):
        if self.stopping:
            return
        if self.worker:
            # Completion checks the captured context and restarts if it became stale.
            return
        actor = dict(self.window.contributor_session or {})
        if not actor:
            self.summary.setText('Select a contributor profile to check assignments')
            return
        from gui.lab_startup import working_locally
        if working_locally():
            self.summary.setText('Working locally | Shared assignments not checked')
            self.summary.setToolTip('')
            self.shared_projects = []
            self.result = None
            self.rebuild_projects()
            self.login.show()
            self.share.show()
            self.configure.hide()
            self.check.setEnabled(True)
            return
        self.result = None
        for button in (self.share, self.configure):
            button.hide()
        self.summary.setText('Checking assignments...')
        self.check.setEnabled(False)
        # Local projects stay usable while the shared-drive lookup is in progress.
        self.download_more.setEnabled(False)
        context = self.context()
        project = self.window.project_path
        configuration = dict(setting_json('project_sync/destination', {}))
        def load(progress, cancelled):
            result = assignment_presence(configuration, actor, project, cancelled)
            if project:
                try:
                    result['projects'] = assignment_presence(configuration, actor, None, cancelled).get('projects', [])
                except (OSError, ValueError) as exc:
                    # An unrelated unreadable archive must not hide this project's valid role.
                    result['project_list_error'] = str(exc)
            return result
        worker = OperationWorker(load, self)
        self.worker = worker
        worker.finished.connect(lambda: self.finished(worker, context, actor))
        worker.start()

    def retry_connection(self):
        from gui.lab_startup import check_session_connection, working_locally
        if working_locally() and not check_session_connection(
                (self.window.contributor_session or {}).get('name', ''), self.window):
            return
        self.refresh()
        self.window.workflow.check_shared()
        self.window.workflow.automatic_work.retry()

    def finished(self, worker, context, actor):
        if worker is not self.worker:
            return
        self.worker = None
        worker.deleteLater()
        if self.stopping:
            return
        from gui.lab_startup import working_locally
        if working_locally():
            self.refresh()
            return
        self.check.setEnabled(True)
        if context != self.context():
            self.refresh()
            return
        if worker.error:
            self.summary.setText('Unable to check assignments. Local work is still available.')
            self.summary.setToolTip(str(worker.error))
            self.share.show()
            self.configure.hide()
            self.login.show()
            self.shared_projects = []
            self.rebuild_projects()
            return
        self.login.hide()
        self.result = result = worker.result
        status = result['status']
        messages = {
            'unconfigured': 'Local projects | Shared drive not set up',
            'unpublished': 'Shared drive connected',
            'setup': 'Shared drive connected',
            'director': 'Shared drive connected',
            'assigned': 'Shared drive connected',
            'empty': 'Shared drive connected | No work assigned',
        }
        self.summary.setText(messages[status])
        self.summary.setToolTip('')
        if result.get('project_list_error'):
            self.summary.setText(messages[status] + '\nOther shared projects could not be checked.')
            self.summary.setToolTip(result['project_list_error'])
            self.login.show()
        self.configure.setVisible(status == 'unconfigured')
        self.share.setVisible(status != 'unconfigured')
        if self.window.project_path:
            self.window.workflow.assignment_role_changed(result)
        self.show_projects(result.get('projects', []))
        notice_key = (context[0], context[1], context[3])
        if status == 'empty' and notice_key not in self.shown_empty:
            self.shown_empty.add(notice_key)
            self.show_empty(actor, result)

    def show_empty(self, actor, result):
        if self.notice:
            self.notice.close()
        notice = QDialog(self.window)
        notice.setWindowTitle('No Work Assigned Yet')
        notice.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        notice.setModal(False)
        notice.resize(560, 230)
        layout = QVBoxLayout(notice)
        director = result.get('director')
        manager = director['name'] if director else 'your project director'
        layout.addWidget(text_label(f'No work is assigned to {actor["name"]} in {result["scope"]}.\n\n'
            f'Ask {manager} to add you in People and assign videos to this profile. '
            'Local work remains available.'))
        row = QHBoxLayout()
        share = QPushButton('Shared Drive Settings...')
        share.clicked.connect(self.configure_profiles)
        retry = QPushButton('Check Again')
        retry.clicked.connect(lambda: (notice.close(), self.refresh()))
        close = QPushButton('Continue Locally')
        close.setDefault(True)
        close.clicked.connect(notice.close)
        for button in (share, retry, close):
            row.addWidget(button)
        layout.addLayout(row)
        notice.finished.connect(lambda: self.notice_finished(notice))
        self.notice = notice
        notice.show()

    def notice_finished(self, notice):
        if self.notice is notice:
            self.notice = None

    def configure_profiles(self):
        from gui.shared_profiles import LabConnectionDialog
        if LabConnectionDialog(self.window).exec() == QDialog.DialogCode.Accepted:
            self.window.invalidate_backup_estimate()
            self.refresh()

    def connect_folder(self):
        self.configure_profiles()

    def show_projects(self, projects):
        self.shared_projects = projects
        self.project_actor = (self.window.contributor_session or {}).get('id')
        self.rebuild_projects()
        self.request_progress(force=True)

    def rebuild_projects(self):
        """Merge shared identities with local copies without conflating names or deleting copies."""
        selected = self.projects.currentItem()
        identity = selected.data(Qt.ItemDataRole.UserRole).get('_key') if selected else None
        old_copy = self.copies.currentData()
        flow = self.controller
        actor = self.window.contributor_session or {}
        if self.project_actor != actor.get('id'):
            self.shared_projects = []
            self.project_actor = actor.get('id')
            identity = old_copy = None
        entries = {('shared', p['id']): dict(p, _key=('shared', p['id']), _paths=[], _shared=True)
                   for p in self.shared_projects}
        recent = flow.settings.value('workflow/recent_projects/' + actor.get('id', ''), [], type=list)
        paths = list(dict.fromkeys(str(p) for p in [self.window.project_path, flow.last_project(), *recent] if p))
        for entry in entries.values():
            local = self.local_path(entry)
            if local and local not in paths:
                paths.append(local)
        configuration = setting_json('project_sync/destination', {})
        for local in paths:
            path = Path(local)
            if configuration.get('root') and path.absolute().is_relative_to(Path(configuration['root']).absolute()):
                continue  # A shared backup is not a local working copy; avoid probing an offline mount here.
            try:
                state = require_contributor(path, actor)
                marker_path = path / 'annotations' / PROJECT_MARKER
                marker = _read_json(marker_path) if marker_path.is_file() else {}
                key = ('shared', marker.get('project_id'))
                if state and state['root_id'] != configuration.get('root_id'):
                    key = ('local', local)
                if key not in entries:
                    key = ('local', local)
                if key not in entries:
                    entries[key] = {'id': marker.get('project_id'), 'name': path.name,
                                    'role': 'local', 'count': 0, 'videos': {},
                                    '_key': key, '_paths': [], '_shared': False}
                entry = entries[key]
                exists = (path / 'annotations' / 'project.json').is_file()
                if exists:
                    entry['_paths'].append(local)
                if not entry['_shared'] and state:
                    entry['videos'] = state['assignments']
                    entry['count'] = len(state['assignments'])
            except (OSError, ValueError, KeyError):
                continue  # Never offer another contributor's selective working copy.
        self.projects.blockSignals(True)
        self.projects.clear()
        selected_row = 0
        selection_found = False
        for index, project in enumerate(entries.values()):
            availability = 'On this computer' if project['_paths'] else (
                'Shared drive' if project['_shared'] else 'Local folder unavailable')
            item = QListWidgetItem(f"{project['name']}\n{availability}")
            item.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_DirIcon))
            item.setToolTip(project['name'] + '\n' + '\n'.join(project['_paths']))
            item.setData(Qt.ItemDataRole.UserRole, project)
            item.setSizeHint(QSize(0, 64))
            self.projects.addItem(item)
            if project['_key'] == identity:
                selected_row = index
                selection_found = True
            elif not selection_found and (old_copy or str(self.window.project_path)) in project['_paths']:
                selected_row = index
        self.projects.setCurrentRow(selected_row)
        self.projects.blockSignals(False)
        self.selection_changed()
        index = self.copies.findData(old_copy)
        if index >= 0:
            self.copies.setCurrentIndex(index)

    def resizeEvent(self, event):
        self.body.setDirection(QBoxLayout.Direction.TopToBottom if self.width() < 720 else
                               QBoxLayout.Direction.LeftToRight)
        super().resizeEvent(event)

    def local_path(self, project):
        from gui.assigned_work_dialog import assigned_local_path
        actor = self.window.contributor_session['id']
        configuration = setting_json('project_sync/destination', {})
        root = configuration.get('root_id', '')
        if root:
            try:
                managed = assigned_local_path(configuration, project['id'], self.window.contributor_session)
                if managed:
                    return managed
            except (OSError, ValueError, KeyError):
                return ''  # Do not silently fall back to a different copy when managed work is unavailable.
        local = self.controller.settings.value(
            f'workflow/shared_projects/{actor}/{root}/{project["id"]}', '', type=str)
        if not local:
            return ''
        if configuration.get('root') and Path(local).absolute().is_relative_to(Path(configuration['root']).absolute()):
            return ''
        try:
            path = Path(local)
            marker = _read_json(path / 'annotations' / PROJECT_MARKER)
            state = require_contributor(path, self.window.contributor_session)
            if marker.get('project_id') != project['id'] or (state and state['root_id'] != root):
                return ''
            if not (path / 'annotations' / 'project.json').is_file():
                return ''
        except (OSError, ValueError, KeyError):
            return ''
        return local

    def selection_changed(self):
        item = self.projects.currentItem()
        project = item.data(Qt.ItemDataRole.UserRole) if item else None
        paths = project.get('_paths', []) if project else []
        self.copies.blockSignals(True)
        self.copies.clear()
        for path in paths:
            self.copies.addItem(path, path)
        current = self.copies.findData(str(self.window.project_path))
        if current >= 0:
            self.copies.setCurrentIndex(current)
        self.copies.blockSignals(False)
        self.copy_label.setVisible(len(paths) > 1)
        self.copies.setVisible(len(paths) > 1)
        self.project_title.setText(project['name'] if project else 'Choose a project')
        self.videos.clear()
        if project:
            from core.annotation_work import task_label
            for video, assignment in sorted(project.get('videos', {}).items()):
                row = QTreeWidgetItem([video, task_label(assignment.get('task')), str(len(assignment['frames']))])
                row.setData(0, Qt.ItemDataRole.UserRole, video)
                row.setToolTip(0, video)
                self.videos.addTopLevelItem(row)
        self.videos.setVisible(self.videos.topLevelItemCount() > 0)
        self.copy_changed()
        self.request_progress()

    def selected_progress_context(self):
        from gui.lab_startup import working_locally
        item = self.projects.currentItem()
        project = item.data(Qt.ItemDataRole.UserRole) if item else None
        if working_locally() or not project or not project.get('_shared'):
            return None
        actor = self.window.contributor_session or {}
        configuration = setting_json('project_sync/destination', {})
        return (configuration.get('root'), configuration.get('root_id'), actor.get('id'),
                actor.get('session_id'), project['id'],
                tuple(sorted((v, a['id'], tuple(a['frames']), a.get('task'))
                             for v, a in project['videos'].items())))

    def request_progress(self, *, force=False):
        if self.stopping:
            return
        context = self.selected_progress_context()
        if context != self.progress_context:
            self.progress_context = context
            self.progress_result = self.progress_error = None
            self.progress_requested = context is not None
            self.progress_timer.stop()
            if self.progress_worker:
                self.progress_worker.requestInterruption()
        elif force and context is not None and not self.progress_worker:
            self.progress_requested = True
            self.progress_result = self.progress_error = None
        if self.progress_requested and not self.progress_worker:
            self.progress_timer.start(0)
        self.update_progress_rows()

    def check_progress(self):
        if self.stopping or self.progress_worker or not self.progress_context:
            return
        if self.selected_progress_context() != self.progress_context:
            self.request_progress()
            return
        project = self.projects.currentItem().data(Qt.ItemDataRole.UserRole)
        configuration = dict(setting_json('project_sync/destination', {}))
        actor = dict(self.window.contributor_session or {})
        assignments = deepcopy(project['videos'])
        project_id = project['id']
        worker = OperationWorker(lambda progress, cancelled:
            shared_assignment_progress(configuration, project_id, actor, assignments, cancelled), self)
        worker.context = self.progress_context
        self.progress_worker = worker
        self.progress_requested = False
        worker.finished.connect(lambda: self.progress_finished(worker))
        worker.start()
        self.update_progress_rows()

    def progress_finished(self, worker):
        if worker is not self.progress_worker:
            return
        self.progress_worker = None
        worker.deleteLater()
        if self.stopping:
            return
        if worker.context != self.selected_progress_context() or worker.isInterruptionRequested():
            self.progress_requested = self.selected_progress_context() is not None
            self.request_progress()
            return
        self.progress_result = worker.result if not worker.error else None
        self.progress_error = str(worker.error) if worker.error else None
        self.update_progress_rows()

    def update_progress_rows(self):
        pending = bool(self.progress_worker or self.progress_requested or self.progress_timer.isActive())
        for index in range(self.videos.topLevelItemCount()):
            row = self.videos.topLevelItem(index)
            video = row.data(0, Qt.ItemDataRole.UserRole)
            record = (self.progress_result or {}).get('videos', {}).get(video)
            text = 'Checking...' if self.progress_context and pending else 'Not checked'
            detail = 'Local edits on either computer are not visible here until uploaded.'
            if record:
                text = record['status']
                summary = record.get('completion') or {}
                detail += '\nPublished completion: ' + summary.get('work_text', 'Unavailable')
                detail += '\n' + summary.get('detail', '')
                if summary.get('state') not in ('unknown', 'not_scanned', 'needs_review') and 'finished' in summary:
                    if summary.get('shared'):
                        text += '\nVideo task finished' if summary['finished'] == summary['total'] else '\nVideo task unfinished'
                    else:
                        text += f'\n{summary["finished"]}/{summary["total"]} frames finished'
                if record['status'] == 'In Progress':
                    detail += '\nIn Progress can mean downloaded or uploaded; it does not itself mean finished.'
            elif self.progress_error:
                text = 'Unavailable'
                detail += '\n' + self.progress_error
            row.setText(3, text)
            row.setToolTip(3, detail)
        self.progress_note.setVisible(self.videos.topLevelItemCount() > 0)
        if self.progress_result:
            checked = datetime.fromisoformat(self.progress_result['checked_at'].replace('Z', '+00:00')).astimezone()
            self.progress_note.setText('Shared progress checked ' + checked.strftime('%H:%M:%S'))
        elif self.progress_error:
            self.progress_note.setText('Shared progress unavailable')
        else:
            self.progress_note.setText('Checking shared progress...' if self.progress_context and pending else
                                       'Shared progress not checked')
        self.progress_note.setToolTip(self.progress_error or
            'Only uploaded work is included. This check does not download or replace annotations.')
        self.update_activity()

    def update_activity(self):
        result = self.progress_result or {}
        reports = result.get('activity', [])
        def when(value):
            try:
                return datetime.fromisoformat(value.replace('Z', '+00:00')).astimezone().strftime('%b %d %H:%M')
            except (ValueError, AttributeError, TypeError):
                return 'unknown time'
        lines = []
        for report in reports:
            name = report.get('device', {}).get('name', 'Unknown computer')
            person = report.get('contributor', {}).get('name', 'Unknown contributor')
            text = f'{person} on {name}: last heard from {when(report.get("reported_at"))}'
            if report.get('state') == 'closed':
                text += ' (session ended)'
            else:
                try:
                    stamp = datetime.fromisoformat(report['reported_at'].replace('Z', '+00:00'))
                    if (datetime.now(timezone.utc) - stamp).total_seconds() > 65 * 60:
                        text += ' (connection status unknown)'
                except (ValueError, TypeError, KeyError):
                    text += ' (connection status unknown)'
            if report.get('last_edited_at'):
                text += f'; last edit {when(report["last_edited_at"])}'
            if report.get('local_save_state') == 'failed':
                text += '; local save needs attention'
            elif report.get('local_save_state') in ('unsaved', 'saving'):
                text += '; local save pending'
            if report.get('pending'):
                text += '; reported work waiting to upload'
            lines.append(text)
        self.activity_note.setText('\n'.join(lines[:3]))
        self.activity_note.setVisible(bool(lines))
        self.activity_note.setToolTip('\n'.join(lines) + '\n\nReports are sent every 30 minutes while '
            'the app is open. An old report means current connection is unknown, not necessarily offline. '
            'Reported edits may still be local; Shared Status shows verified uploads.')
        ready = result.get('review_ready', [])
        self.review_notice.setVisible(bool(ready))
        self.review_notice.setText(f'{len(ready)} {"video" if len(ready) == 1 else "videos"} ready for review')
        self.review_notice.setToolTip('\n'.join(r['video'] + ' - ' + r['contributor']['name'] for r in ready))
        button = getattr(self.controller, 'review_notification', None)
        if button:
            button.setVisible(bool(ready))
            button.setText(f'{len(ready)} ready for review')
            button.setToolTip(self.project_title.text() + '\n' + self.review_notice.toolTip())

    def copy_changed(self, *_):
        item = self.projects.currentItem()
        project = item.data(Qt.ItemDataRole.UserRole) if item else None
        local = self.copies.currentData() or ''
        shared = bool(project and project.get('_shared'))
        count = project['count'] if project else 0
        self.project_status.setText(('Project director' if project['role'] == 'director' else
            'Contributor' if shared else 'Local project') if project else 'No projects available yet')
        if local:
            self.project_status.setText(self.project_status.text() + ' | Ready on this computer')
        elif project and not shared:
            self.project_status.setText('Local folder unavailable')
        self.project_status.setToolTip(local)
        self.assignment_summary.setText(
            (f'{count} video' + ('s' if count != 1 else '') + ' assigned to you' if count else
             'No videos assigned to you') if shared else
            ('Downloaded assignments | Shared status not checked' if count else
             'Assignments not checked for this project'))
        self.assignment_summary.setVisible(bool(project))
        self.open.setText(('Open My Assigned Work' if count else 'Open Project')
                          if local or not shared else 'Download My Assigned Work...')
        self.open.setVisible(bool(project))
        self.open.setEnabled(bool(local or (shared and count)))
        self.open.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_DirOpenIcon if local else
                                                   QStyle.StandardPixmap.SP_ArrowDown))
        self.open.setToolTip('' if project and (local or project['count']) else
            'Open a local project copy or ask your director to assign work.')
        self.download_more.setVisible(bool(local and shared and count))
        self.download_more.setEnabled(bool(shared and count))
        self.download_more.setText('Download / Update Assigned Work...')
        current_setup = (local and str(self.window.project_path) == local and self.result and
                         self.result['status'] in ('setup', 'unpublished'))
        self.manage.setVisible(bool(local and ((project and project['role'] == 'director') or current_setup)))

    def manage_selected(self):
        local = self.copies.currentData()
        if not local:
            return
        if str(self.window.project_path) != local:
            self.window.open_recent_project(local)
        if str(self.window.project_path) == local:
            self.window.manage_assignments()

    def review_selected(self):
        local = self.copies.currentData()
        if not local:
            QMessageBox.information(self.window, 'Open a Local Project',
                'Choose a local director project copy before reviewing submissions. Shared backups are not edited in place.')
            return
        if str(self.window.project_path) != local:
            self.window.open_recent_project(local)
        if str(self.window.project_path) == local:
            self.window.review_submissions()

    def download_selected(self):
        item = self.projects.currentItem()
        if item:
            project = item.data(Qt.ItemDataRole.UserRole)
            if not project.get('_shared'):
                return
            if not self.window.open_assigned_work(project_id=project['id']):
                return
            if self.local_path(project) and self.window.project_path and project.get('videos'):
                if Path(self.local_path(project)) == self.window.project_path:
                    self.show_assigned_work(project)

    def open_selected(self):
        item = self.projects.currentItem()
        if not item:
            return
        project = item.data(Qt.ItemDataRole.UserRole)
        local = self.copies.currentData() or ''
        if local:
            if str(self.window.project_path) != local:
                self.window.open_recent_project(local)
            if str(self.window.project_path) != local:
                return  # Opening can be cancelled or rejected; do not navigate the old project.
            if project.get('videos'):
                self.show_assigned_work(project)
            else:
                self.window.workflow.activate('Annotate')
                self.window.video_dock.show()
        elif project.get('_shared') and project['count']:
            self.download_selected()

    def show_assigned_work(self, project):
        try:
            self._show_assigned_work(project)
        except (OSError, ValueError, KeyError) as exc:
            QMessageBox.warning(self.window, 'Assigned Work Unavailable', str(exc))

    def _show_assigned_work(self, project):
        """Navigate a full project or a selective copy without changing its storage mode."""
        w = self.window
        assignments = project['videos']
        state = require_contributor(w.project_path, w.contributor_session)
        available, missing = {}, []
        for video, assignment in assignments.items():
            if state:
                stored = state['assignments'].get(video, {})
                if any(stored.get(key) != assignment.get(key) for key in ('id', 'frames', 'task')):
                    missing.append(video)
                    continue
            folder = w.project_manager.get_frames_dir(video)
            frames = [frame for frame in assignment['frames']
                      if (folder / f'frame_{frame:06d}.jpg').is_file()]
            if frames:
                available[video] = frames
            if len(frames) != len(assignment['frames']):
                missing.append(video)
        if not available:
            QMessageBox.information(w, 'Assigned Work Not Available Locally',
                'This project does not contain usable frames for your current assignments. '
                'Choose Download Assigned Work on Home to add assignments to your working copy. '
                'Existing local edits will be kept; conflicts stop for review.')
            return

        browser = w.video_browser
        browser.set_my_work(w.project_path, {v: assignments[v] for v in available}, enabled=True)
        # Old experiment, split, or completion filters must not hide newly opened work.
        filters = w.annotation_work.filters
        w.annotation_work.review.view.setCurrentIndex(0)
        for combo in (browser.experiment, w.video_split_filter_combo, w.split_filter_combo,
                      filters.show, filters.presence, filters.split):
            combo.blockSignals(True)
            combo.setCurrentIndex(0)
            combo.blockSignals(False)
        w.video_split_filter = 'all'
        w.split_filter = 'all'
        w.tracking_filter_checkbox.blockSignals(True)
        w.tracking_filter_checkbox.setChecked(False)
        w.tracking_filter_checkbox.blockSignals(False)
        filters.update_filter_label()
        w.workflow.activate('Annotate')
        w.video_dock.show()
        w.frame_dock.show()
        video = w.current_video_id if w.current_video_id in available else sorted(available)[0]
        if w.current_video_id != video or not w.frames:
            w.load_video_frames(video)
        if w.current_video_id == video and w.frames:
            current = w._get_frame_idx_in_video(w.current_frame_idx)
            if current not in available[video]:
                target = available[video][0]
                index = next((i for i in range(len(w.frames)) if w._get_frame_idx_in_video(i) == target), None)
                if index is not None:
                    w.load_frame(index)
        task = assignments[video].get('task')
        filters.task.setCurrentIndex(max(0, filters.task.findData(task)))
        w.update_video_list()
        w.update_frame_list()
        w.annotation_work.review.schedule()
        w.video_list.clearSelection()
        for row in range(w.video_list.count()):
            item = w.video_list.item(row)
            if item.data(Qt.ItemDataRole.UserRole) == video:
                w.video_list.setCurrentItem(item)
                w.video_list.scrollToItem(item)
                break
        w.status_label.setText(f'My assigned work: {len(available)} of {len(assignments)} videos available locally')
        if missing:
            QMessageBox.information(w, 'Some Assigned Work Needs Downloading',
                f'{len(missing)} assigned videos have missing frames or changed assignments. '
                'Use Download Assigned Work on Home for the current assignments. '
                'Existing local edits have not been replaced.')

    def stop(self):
        self.stopping = True
        self.timer.stop()
        self.progress_timer.stop()
        if self.notice:
            self.notice.close()
        workers = [worker for worker in (self.worker, self.progress_worker) if worker and worker.isRunning()]
        for worker in workers:
            worker.requestInterruption()
        return all([worker.wait(2000) for worker in workers])


class AssignmentWorkspace(QScrollArea):
    def __init__(self, window):
        super().__init__()
        self.window = window
        self.panel = None
        self.context_key = None
        self.worker = None
        self.generation = 0
        self.refresh_pending = False
        self.stopping = False
        self.cancel_requested = False
        self.setWidgetResizable(True)
        self.content = QWidget()
        self.layout = QVBoxLayout(self.content)
        self.layout.setContentsMargins(24, 24, 24, 24)
        self.title = text_label('Assignments')
        font = self.title.font()
        font.setPointSize(18)
        font.setBold(True)
        self.title.setFont(font)
        self.layout.addWidget(self.title)
        self.status = text_label()
        self.layout.addWidget(self.status)
        self.setup = QWidget()
        self.setup_layout = QVBoxLayout(self.setup)
        self.setup_layout.setContentsMargins(0, 0, 0, 0)
        self.layout.addWidget(self.setup)
        self.retry = QPushButton('Check Setup Again')
        self.retry.clicked.connect(self.refresh)
        self.layout.addWidget(self.retry)
        self.cancel_check = QPushButton('Cancel Check')
        self.cancel_check.clicked.connect(self.cancel)
        self.layout.addWidget(self.cancel_check)
        self.cancel_check.hide()
        self.layout.addStretch()
        self.setWidget(self.content)

    def clear(self):
        self.generation += 1
        self.refresh_pending = False
        if self.worker:
            self.worker.requestInterruption()
        self.context_key = None
        self.status.show()
        self.setup.show()
        self.retry.show()
        if self.panel:
            self.panel.video_browser.cancel()
            self.panel.shared_version.cancel()
            self.panel.hide()
            self.layout.removeWidget(self.panel)
            self.panel.deleteLater()
            self.panel = None
        while self.setup_layout.count():
            widget = self.setup_layout.takeAt(0).widget()
            if widget:
                widget.deleteLater()

    def activate(self):
        key = self.window.workflow.home.inbox.context()
        if self.context_key != key:
            self.refresh()

    def confirm_discard(self):
        if self.panel:
            try:
                old = {v: {k: e[k] for k in ('contributor', 'frames')} for v, e in self.panel.data['videos'].items()}
                dirty = old != self.panel.choices()
            except ValueError:
                dirty = True
            if dirty and QMessageBox.question(self, 'Discard Unsaved Assignment Choices?',
                    'This will discard unsaved assignment table edits. Continue?',
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                    QMessageBox.StandardButton.No) != QMessageBox.StandardButton.Yes:
                return False
        return True

    def refresh(self):
        if self.worker:
            self.refresh_pending = True
            self.worker.requestInterruption()
            return
        if not self.confirm_discard():
            return
        self.stopping = False
        self.cancel_requested = False
        self.clear()
        window = self.window
        from gui.lab_startup import working_locally
        if working_locally():
            self.status.setText('Working locally. Reconnect from Home to manage shared assignments.')
            self.retry.hide()
            return
        if not window.project_path or not window.contributor_session:
            self.status.setText('Open a local project and choose your profile first.')
            return
        configuration = setting_json('project_sync/destination', {})
        project = window.project_path
        contributor = dict(window.contributor_session)
        context = window.workflow.home.inbox.context()
        generation = self.generation
        self.status.setText('Checking assignment setup...')
        self.retry.setEnabled(False)
        self.cancel_check.setEnabled(True)
        self.cancel_check.show()
        worker = OperationWorker(lambda progress, cancelled: assignment_management(
            project, configuration, contributor, progress=progress, cancelled=cancelled), self)
        self.worker = worker
        worker.progress.connect(lambda message: self.show_progress(worker, generation, message))
        worker.finished.connect(lambda: self.checked(worker, generation, context, configuration))
        worker.start()

    def show_progress(self, worker, generation, message):
        if worker is self.worker and generation == self.generation and not self.cancel_requested:
            self.status.setText(message)

    def cancel(self):
        self.refresh_pending = False
        self.cancel_requested = True
        if self.worker:
            self.worker.requestInterruption()
            self.cancel_check.setEnabled(False)
            self.status.setText('Cancelling after the current file operation. Local work is still available.')

    def stop(self):
        self.stopping = True
        self.refresh_pending = False
        stopped = self.panel.video_browser.stop() if self.panel else True
        if self.panel:
            stopped = self.panel.shared_version.stop() and stopped
        if self.worker and self.worker.isRunning():
            self.worker.requestInterruption()
            return self.worker.wait(2000) and stopped
        return stopped

    def checked(self, worker, generation, context, configuration):
        if worker is not self.worker:
            return
        self.worker = None
        worker.deleteLater()
        self.retry.setEnabled(True)
        self.cancel_check.hide()
        if self.stopping:
            return
        window = self.window
        if (self.refresh_pending or generation != self.generation or
                context != window.workflow.home.inbox.context()):
            if self.refresh_pending or self.isVisible():
                self.refresh()
            return
        if self.cancel_requested:
            self.status.setText('Assignment check cancelled. Local work is still available.')
            return
        if worker.error:
            self.status.setText('Unable to load assignments: ' + str(worker.error))
            return
        try:
            state = worker.result
            self.setup_revision = state.get('data', {}).get('revision')
            self.context_key = window.workflow.home.inbox.context()
            if state.get('restricted'):
                self.status.setText('Assignments are managed by ' + state['director']['name'] + '.')
                return
            if state['needs']:
                self.status.setText('Assignment setup')
                actions = {
                    'connection': ('Shared folder not connected', 'Connect Shared Folder...', window.configure_project_sync),
                    'backup': ('Publish the project before assigning videos', 'Prepare Project Backup...', self.prepare_backup),
                    'compact_backup': ('A deduplicated backup is required', 'Backup Settings...', window.configure_project_sync),
                    'policy': ('The published annotation policy is incomplete', 'Review Annotation Policy...', window.show_annotation_policy),
                    'frames': ('No frame images are published', 'Add Video...', window.show_add_video_dialog),
                    'director': ('No project director has been set', 'Set Project Director...', self.set_director),
                }
                for need in state['needs']:
                    message, title, callback = actions[need]
                    self.setup_layout.addWidget(text_label(message))
                    button = QPushButton(title)
                    button.clicked.connect(callback)
                    self.setup_layout.addWidget(button)
                if any(need in state['needs'] for need in ('policy', 'frames', 'compact_backup')):
                    button = QPushButton('Prepare Updated Backup...')
                    button.clicked.connect(self.prepare_backup)
                    self.setup_layout.addWidget(button)
                return
            self.status.setText('')
            self.status.hide()
            self.setup.hide()
            self.retry.hide()
            self.panel = AssignmentManagerDialog(window.project_path, configuration, window.contributor_session,
                self.content, initial=(state['data'], state['available'], state['statuses'], state['snapshot']), embedded=True)
            self.layout.insertWidget(self.layout.count() - 1, self.panel, 1)
            self.panel.finished.connect(self.refresh)
            self.panel.assignments_changed.connect(window.workflow.home.inbox.reset)
        except Exception as exc:
            self.status.setText('Unable to load assignments: ' + str(exc))

    def prepare_backup(self):
        self.window.workflow.activate('Manage')

    def set_director(self):
        actor = dict(self.window.contributor_session)
        answer = QMessageBox.question(self, 'Set Project Director?',
            f'Make {actor["name"]} the director of this project?\n\n'
            'This profile will manage assignments and accept submissions. '
            'The role applies only to this project and can be transferred explicitly later.',
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Cancel)
        if answer != QMessageBox.StandardButton.Yes:
            return
        try:
            configuration = setting_json('project_sync/destination', {})
            # The backend rejects a director created by another computer in the meantime.
            SyncProgressDialog('Set Project Director', lambda progress, cancelled:
                establish_director(self.window.project_path, configuration, actor, self.setup_revision), self).run()
            self.window.workflow.home.inbox.reset()
            self.refresh()
        except Exception as exc:
            QMessageBox.warning(self, 'Director Not Set', str(exc))
