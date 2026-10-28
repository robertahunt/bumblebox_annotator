"""Assignment discovery and explicit project-director setup."""

from pathlib import Path

from PyQt6.QtCore import Qt, QTimer, QSize, pyqtSignal
from PyQt6.QtWidgets import (QDialog, QHBoxLayout, QLabel, QListWidget, QListWidgetItem, QMessageBox, QPushButton,
                            QScrollArea, QStyle, QToolButton, QVBoxLayout, QWidget)

from core.assigned_work import (assignment_management, assignment_presence, establish_director,
                                require_contributor)
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
    projects_visible = pyqtSignal(bool)

    def __init__(self, window):
        super().__init__()
        self.window = window
        self.worker = None
        self.notice = None
        self.shown_empty = set()
        self.stopping = False
        self.result = None
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 8, 0, 8)
        row = QHBoxLayout()
        self.summary = text_label('Assignments not checked')
        row.addWidget(self.summary, 1)
        self.check = QToolButton()
        self.check.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_BrowserReload))
        self.check.setAccessibleName('Check assignments again')
        self.check.setToolTip('Check assignments again')
        self.check.clicked.connect(self.refresh)
        row.addWidget(self.check)
        layout.addLayout(row)
        actions = QHBoxLayout()
        self.share = QPushButton('Lab Connection...')
        self.share.setToolTip('Make contributor profiles available on other computers through the shared folder')
        self.share.clicked.connect(self.configure_profiles)
        self.download = QPushButton('Download Assigned Work...')
        self.download.clicked.connect(lambda: window.workflow.join_shared())
        self.configure = QPushButton('Connect to Lab...')
        self.configure.clicked.connect(self.connect_folder)
        self.manage = QPushButton('Open Assignments')
        self.manage.clicked.connect(window.manage_assignments)
        for button in (self.share, self.download, self.configure, self.manage):
            actions.addWidget(button)
            button.hide()
        actions.addStretch()
        layout.addLayout(actions)
        from gui.drive_login import DriveLoginButton
        self.login = DriveLoginButton(self)
        self.login.completed.connect(self.refresh)
        layout.addWidget(self.login)
        self.login.hide()
        self.projects = QListWidget()
        self.projects.setAccessibleName('My shared projects')
        self.projects.setTextElideMode(Qt.TextElideMode.ElideRight)
        self.projects.itemActivated.connect(lambda item: self.open_selected())
        self.projects.currentItemChanged.connect(self.selection_changed)
        layout.addWidget(self.projects)
        self.open = QPushButton('Download Assigned Work...')
        self.open.clicked.connect(self.open_selected)
        layout.addWidget(self.open)
        self.download_more = QPushButton('Download Assigned Work...')
        self.download_more.setToolTip('Add or refresh assignments in your existing working copy; local edits are kept.')
        self.download_more.clicked.connect(self.download_selected)
        layout.addWidget(self.download_more)
        self.download_more.hide()
        self.projects.hide()
        self.open.hide()
        self.timer = QTimer(self)
        self.timer.setInterval(60000)
        self.timer.timeout.connect(lambda: self.refresh() if self.isVisible() else None)
        self.timer.start()

    def context(self):
        actor = self.window.contributor_session or {}
        configuration = setting_json('project_sync/destination', {})
        return (actor.get('id'), actor.get('session_id'), str(self.window.project_path or ''),
                configuration.get('root_id'), configuration.get('root'))

    def reset(self):
        self.result = None
        self.projects.clear()
        self.projects.hide()
        self.open.hide()
        self.projects_visible.emit(False)
        self.download_more.hide()
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
        self.result = None
        for button in (self.share, self.download, self.configure, self.manage):
            button.hide()
        self.summary.setText('Checking assignments...')
        self.check.setEnabled(False)
        self.open.setEnabled(False)
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

    def finished(self, worker, context, actor):
        if worker is not self.worker:
            return
        self.worker = None
        worker.deleteLater()
        if self.stopping:
            return
        self.check.setEnabled(True)
        if context != self.context():
            self.refresh()
            return
        if worker.error:
            self.summary.setText('Unable to check assignments. Local work is still available.')
            self.summary.setToolTip(str(worker.error))
            self.configure.show()
            self.login.show()
            self.projects.hide()
            self.open.hide()
            self.download_more.hide()
            self.projects_visible.emit(False)
            return
        self.login.hide()
        self.result = result = worker.result
        status, count = result['status'], result['count']
        scope = result.get('scope', 'this project')
        messages = {
            'unconfigured': 'Shared folder not configured',
            'unpublished': 'This local project has not been published for assignments',
            'setup': 'Project director setup is required',
            'director': f'Project director | {count} videos assigned to you',
            'assigned': f'{count} videos assigned to {actor["name"]} in {scope}',
            'empty': f'No work assigned to {actor["name"]} in {scope}',
        }
        self.summary.setText(messages[status])
        self.summary.setToolTip('')
        if result.get('project_list_error'):
            self.summary.setText(messages[status] + '\nOther shared projects could not be checked.')
            self.summary.setToolTip(result['project_list_error'])
            self.login.show()
        self.configure.setVisible(status == 'unconfigured')
        self.share.setVisible(status != 'unconfigured')
        self.download.setVisible(False)
        self.manage.setVisible(bool(self.window.project_path) and status in ('director', 'setup', 'unpublished'))
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
        share = QPushButton('Connect to Lab...')
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
        selected = self.projects.currentItem()
        identity = selected.data(Qt.ItemDataRole.UserRole)['id'] if selected else None
        self.projects.clear()
        for project in projects:
            role = 'Director' if project['role'] == 'director' else 'Contributor'
            item = QListWidgetItem(f"{project['name']} | {role} | {project['count']} assigned videos")
            item.setToolTip(item.text())
            item.setData(Qt.ItemDataRole.UserRole, project)
            item.setSizeHint(QSize(0, 40))
            self.projects.addItem(item)
            if project['id'] == identity:
                self.projects.setCurrentItem(item)
        self.projects.setFixedHeight(min(4, len(projects)) * 40 + 4)
        self.projects.setVisible(bool(projects))
        self.open.setVisible(bool(projects))
        self.projects_visible.emit(bool(projects))
        if projects and self.projects.currentRow() < 0:
            self.projects.setCurrentRow(0)
        self.selection_changed()

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
        local = self.window.workflow.settings.value(
            f'workflow/shared_projects/{actor}/{root}/{project["id"]}', '', type=str)
        if not local:
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
        local = self.local_path(project) if project else ''
        self.open.setText(('Open My Assigned Work' if project['count'] else 'Open Local Project')
                          if local else 'Download Assigned Work...')
        self.open.setEnabled(bool(project and (local or project['count'])))
        self.open.setToolTip('' if project and (local or project['count']) else
            'No videos assigned. Directors can open their local project below to assign work.')
        self.download_more.setVisible(bool(local and project['count']))
        self.download_more.setEnabled(bool(project and project['count']))

    def download_selected(self):
        item = self.projects.currentItem()
        if item:
            project = item.data(Qt.ItemDataRole.UserRole)
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
        local = self.local_path(project)
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
        elif project['count']:
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
        if self.notice:
            self.notice.close()
        if self.worker and self.worker.isRunning():
            self.worker.requestInterruption()
            return self.worker.wait(2000)
        return True


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
