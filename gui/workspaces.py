"""Project home and task workspaces around the existing annotation editor."""

import hashlib
import math
from pathlib import Path

from PyQt6.QtCore import QObject, QPointF, QSize, Qt, QTimer
from PyQt6.QtGui import QAction, QColor, QPalette, QTransform
from PyQt6.QtWidgets import (QCheckBox, QComboBox, QDockWidget, QHBoxLayout,
                            QLabel, QListWidget, QListWidgetItem, QMessageBox,
                            QPushButton, QScrollArea, QStackedWidget, QStyle,
                            QTabBar, QToolBar, QToolButton, QVBoxLayout, QWidget,
                            QSizePolicy)

from core.assigned_work import workspace
from core.project_activity import project_activity
from gui.project_sync_dialog import OperationWorker, setting_json
from gui.workflow_help import (ASSIGNED_LABEL, BACKUP_LABEL, RESTORE_LABEL, UPDATE_LABEL,
                               add_help_row, help_button)
from gui.assignment_workspace import AssignmentInbox, AssignmentWorkspace


def label(text, *, heading=False):
    widget = QLabel(text)
    widget.setTextFormat(Qt.TextFormat.PlainText)
    widget.setWordWrap(True)
    if heading:
        font = widget.font()
        font.setPointSize(18)
        font.setBold(True)
        widget.setFont(font)
    return widget


def command(layout, text, callback, icon=None):
    button = QPushButton(text)
    button.setMinimumHeight(32)
    if icon is not None:
        button.setIcon(button.style().standardIcon(icon))
    button.clicked.connect(callback)
    layout.addWidget(button)
    return button


class ProjectHome(QWidget):
    def __init__(self, controller):
        super().__init__()
        self.setObjectName('projectHome')
        font = self.font()
        font.setPointSize(max(11, font.pointSize()))
        self.setFont(font)
        # Keep the landing page legible under a dark system theme without changing the editor.
        palette = self.palette()
        for role, color in ((QPalette.ColorRole.Window, '#f4f6f8'),
                            (QPalette.ColorRole.WindowText, '#202b32'),
                            (QPalette.ColorRole.Base, '#ffffff'),
                            (QPalette.ColorRole.Text, '#202b32'),
                            (QPalette.ColorRole.Button, '#ffffff'),
                            (QPalette.ColorRole.ButtonText, '#202b32'),
                            (QPalette.ColorRole.Highlight, '#dceee9'),
                            (QPalette.ColorRole.HighlightedText, '#163e35')):
            palette.setColor(role, QColor(color))
        self.setPalette(palette)
        self.setAutoFillBackground(True)
        self.setStyleSheet('''
            QWidget#projectHome, QWidget#projectHomePage { background: #f4f6f8; }
            QWidget#projectHome QLabel, QWidget#projectHome QPushButton,
            QWidget#projectHome QToolButton, QWidget#projectHome QListWidget,
            QWidget#projectHome QCheckBox { font-size: 11pt; }
            QWidget#projectHome QLabel { color: #202b32; background: transparent; }
            QWidget#projectHome QLabel#homeTitle { font-size: 18pt; font-weight: bold; }
            QWidget#projectHome QLabel#homeIdentity { color: #53636d; }
            QWidget#projectHome QPushButton, QWidget#projectHome QToolButton {
                color: #202b32; background: #ffffff; border: 1px solid #aebdc5;
                border-radius: 4px; padding: 8px 12px;
            }
            QWidget#projectHome QPushButton:hover, QWidget#projectHome QToolButton:hover {
                background: #e9f0f3; border-color: #718b98;
            }
            QWidget#projectHome QPushButton:focus, QWidget#projectHome QToolButton:focus {
                border: 2px solid #247b69;
            }
            QWidget#projectHome QPushButton#openSelectedProject {
                background: #246f60; color: #ffffff; border-color: #246f60;
            }
            QWidget#projectHome QPushButton#openSelectedProject:hover { background: #195b4e; }
            QWidget#projectHome QPushButton:disabled {
                background: #e4e9ec; color: #687780; border-color: #c8d1d7;
            }
            QWidget#projectHome QToolButton#contextHelp { padding: 0; border-radius: 14px; }
            QWidget#projectHome QToolButton#recoveryToggle {
                background: transparent; border: none; padding: 8px 0; text-align: left;
            }
            QWidget#projectHome QListWidget {
                background: #ffffff; color: #202b32; border: 1px solid #aebdc5; border-radius: 4px;
            }
            QWidget#projectHome QListWidget::item { padding: 8px; }
            QWidget#projectHome QListWidget::item:selected { background: #dceee9; color: #163e35; }
            QWidget#projectHome QListWidget::item:hover:!selected { background: #eef3f6; }
            QWidget#projectHome QCheckBox { color: #394b56; spacing: 8px; }
        ''')
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        scroll = QScrollArea()
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        scroll.setWidgetResizable(True)
        page = QWidget()
        page.setObjectName('projectHomePage')
        page.setPalette(palette)
        page.setAutoFillBackground(True)
        page_layout = QVBoxLayout(page)
        page_layout.setContentsMargins(24, 24, 24, 24)
        content = QWidget()
        content.setMaximumWidth(840)
        content.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        layout = QVBoxLayout(content)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)
        title = label('Choose a project', heading=True)
        title.setObjectName('homeTitle')
        layout.addWidget(title)
        self.identity = label('')
        self.identity.setObjectName('homeIdentity')
        layout.addWidget(self.identity)
        self.inbox = AssignmentInbox(controller.window)
        layout.addWidget(self.inbox)
        layout.addSpacing(12)
        self.recent_heading = label('Recent projects')
        font = self.recent_heading.font()
        font.setBold(True)
        self.recent_heading.setFont(font)
        layout.addWidget(self.recent_heading)
        self.recent = QListWidget()
        self.recent.setAccessibleName('Recent projects')
        self.recent.setTextElideMode(Qt.TextElideMode.ElideMiddle)
        self.recent.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.recent.itemActivated.connect(lambda item: self.open_project())
        layout.addWidget(self.recent)
        self.open_selected = command(layout, 'Open Selected Project', self.open_project,
                                     QStyle.StandardPixmap.SP_DirOpenIcon)
        self.open_selected.setObjectName('openSelectedProject')
        self.open_selected.setDefault(True)
        self.context = label('')
        self.context.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        layout.addWidget(self.context)
        self.recent.itemSelectionChanged.connect(self.selection_changed)
        layout.addSpacing(18)
        self.other_heading = label('Other projects')
        self.other_heading.setFont(font)
        layout.addWidget(self.other_heading)
        command(layout, 'Open Local Project...', controller.window.open_project,
                QStyle.StandardPixmap.SP_DirOpenIcon)
        download = QPushButton('Download Assigned Work...')
        download.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_DriveNetIcon))
        download.clicked.connect(controller.join_shared)
        add_help_row(layout, download, 'download')
        command(layout, 'Create New Project...', controller.window.new_project,
                QStyle.StandardPixmap.SP_FileIcon)
        layout.addSpacing(10)
        self.recovery_toggle = QToolButton()
        self.recovery_toggle.setObjectName('recoveryToggle')
        self.recovery_toggle.setText('Recovery')
        self.recovery_toggle.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.recovery_toggle.setArrowType(Qt.ArrowType.RightArrow)
        self.recovery_toggle.setCheckable(True)
        layout.addWidget(self.recovery_toggle)
        self.recovery = QWidget()
        recovery_layout = QVBoxLayout(self.recovery)
        recovery_layout.setContentsMargins(0, 0, 0, 0)
        restore = QPushButton(RESTORE_LABEL)
        restore.clicked.connect(controller.window.restore_project_backup)
        add_help_row(recovery_layout, restore, 'restore')
        layout.addWidget(self.recovery)
        self.recovery.hide()
        self.recovery_toggle.toggled.connect(self.show_recovery)
        self.auto_resume = QCheckBox('Reopen my last project on startup')
        self.auto_resume.setChecked(controller.settings.value('workflow/auto_resume', False, type=bool))
        self.auto_resume.toggled.connect(lambda value: controller.settings.setValue('workflow/auto_resume', value))
        layout.addWidget(self.auto_resume)
        center = QHBoxLayout()
        center.addStretch()
        center.addWidget(content, 1)
        center.addStretch()
        page_layout.addLayout(center)
        page_layout.addStretch()
        scroll.setWidget(page)
        outer.addWidget(scroll)
        self.controller = controller

    def show_recovery(self, visible):
        self.recovery.setVisible(visible)
        self.recovery_toggle.setArrowType(Qt.ArrowType.DownArrow if visible else Qt.ArrowType.RightArrow)

    def selection_changed(self):
        item = self.recent.currentItem()
        path = item.data(Qt.ItemDataRole.UserRole) if item else ''
        current = path and str(self.controller.window.project_path) == path
        self.open_selected.setEnabled(bool(item))
        self.open_selected.setText('Return to Project' if current else 'Open Selected Project')
        self.context.setText(path)
        self.context.setToolTip(path)
        self.context.setVisible(bool(path))

    def open_project(self):
        item = self.recent.currentItem()
        if item:
            path = item.data(Qt.ItemDataRole.UserRole)
            if str(self.controller.window.project_path) == path:
                self.controller.resume()
            else:
                self.controller.window.open_recent_project(path)

    def refresh(self):
        window = self.controller.window
        session = window.contributor_session
        self.identity.setText('Working as ' + session['name'] if session else 'Contributor not selected')
        self.recent.clear()
        target = window.project_path or self.controller.last_project()
        paths = list(dict.fromkeys(str(path) for path in [target, *window.recent_projects] if path))
        for path in paths:
            item = QListWidgetItem(Path(path).name)
            item.setData(Qt.ItemDataRole.UserRole, str(path))
            item.setToolTip(str(path))
            item.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_DirIcon))
            item.setSizeHint(QSize(0, 44))
            self.recent.addItem(item)
        self.recent.setFixedHeight(min(4, len(paths)) * 44 + 4)
        for widget in (self.recent_heading, self.recent, self.open_selected):
            widget.setVisible(bool(paths))
        self.other_heading.setText('Other projects' if paths else 'Projects')
        if paths:
            self.recent.setCurrentRow(0)
        self.selection_changed()


class WorkspaceController(QObject):
    names = ('Annotate', 'Review', 'Train', 'Assignments', 'Manage')

    def __init__(self, window, settings):
        super().__init__(window)
        self.window, self.settings = window, settings
        self.current = 'Home'
        self.worker = None
        self.generation = 0
        self.status = None
        self.assignment_role = None
        self.dock_visibility = {}
        self.editor = window.takeCentralWidget()
        self.shell = QWidget()
        layout = QVBoxLayout(self.shell)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        header_widget = QWidget()
        header_widget.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        header = QHBoxLayout(header_widget)
        header.setContentsMargins(8, 6, 8, 6)
        home = QToolButton()
        home.setIcon(window.style().standardIcon(QStyle.StandardPixmap.SP_DirHomeIcon))
        home.setToolTip('Project Home')
        home.setAccessibleName('Project Home')
        home.setMinimumSize(32, 32)
        home.clicked.connect(self.show_home)
        header.addWidget(home)
        self.project_label = label('No project')
        self.project_label.setWordWrap(False)
        self.project_label.setMinimumWidth(140)
        self.project_label.setMaximumWidth(240)
        header.addWidget(self.project_label)
        self.tabs = QTabBar()
        self.tabs.setExpanding(False)
        self.tabs.setDrawBase(False)
        for name in self.names:
            self.tabs.addTab(name)
        self.tabs.setMinimumWidth(self.tabs.sizeHint().width())
        self.tabs.currentChanged.connect(lambda i: self.activate(self.names[i]))
        header.addWidget(self.tabs)
        header.addStretch()
        self.person = QToolButton()
        self.person.setToolTip('Change contributor or import/export your profile')
        self.person.setMaximumWidth(200)
        self.person.clicked.connect(window.change_contributor)
        header.addWidget(self.person)
        self.navigation = QToolBar('Workspace', window)
        self.navigation.setMovable(False)
        self.navigation.setFloatable(False)
        self.navigation.addWidget(header_widget)
        window.addToolBar(Qt.ToolBarArea.TopToolBarArea, self.navigation)

        self.status_bar = QWidget()
        status_layout = QHBoxLayout(self.status_bar)
        status_layout.setContentsMargins(8, 0, 8, 0)
        self.local_status = label('Saved locally')
        self.shared_status = label('Local project')
        status_layout.addWidget(self.local_status)
        status_layout.addWidget(self.shared_status, 1)
        self.check_button = QToolButton()
        self.check_button.setIcon(window.style().standardIcon(QStyle.StandardPixmap.SP_BrowserReload))
        self.check_button.setToolTip('Check shared project for updates')
        self.check_button.setAccessibleName('Check shared project for updates')
        self.check_button.clicked.connect(self.check_shared)
        status_layout.addWidget(self.check_button)
        self.update_button = command(status_layout, UPDATE_LABEL, window.update_current_project)
        self.update_help = help_button(self.status_bar, 'update')
        status_layout.addWidget(self.update_help)
        self.upload_button = command(status_layout, BACKUP_LABEL, self.upload,
                                     QStyle.StandardPixmap.SP_ArrowUp)
        self.upload_help = help_button(self.status_bar,
                                      lambda: 'assigned' if workspace(window.project_path) else 'backup')
        status_layout.addWidget(self.upload_help)
        layout.addWidget(self.status_bar)
        self.stack = QStackedWidget()
        layout.addWidget(self.stack, 1)
        self.home = ProjectHome(self)
        self.stack.addWidget(self.home)
        self.stack.addWidget(self.editor)
        self.train_page = self.command_page('Train and Evaluate', [
            ('Train Segmentation Model', window.train_yolo_model),
            ('Train Instance-Focused Model', window.train_yolo_model_instance_focused),
            ('Train Bounding-Box Model', window.train_yolo_bbox_model),
            ('Evaluate Validation Frames', window.analyze_frame_level_validation),
            ('Batch Image Inference', window.batch_image_inference),
            ('Batch Video Inference', window.batch_video_inference),
        ])
        self.backup_summary = label('Transfer size not estimated')
        self.manage_page = self.command_page('Manage Project', [
            ('Video Assignments and Submissions', window.manage_assignments),
            ('Annotation Policy', window.show_annotation_policy),
            ('Add Video', window.show_add_video_dialog),
            ('Import From Project', window.import_project_data),
            ('Shared Folder Settings', window.configure_project_sync),
            ('Estimate Transfer Size', window.estimate_backup_action),
            ('', self.backup_summary),
            ('Review Changes', window.review_backup_action),
            (BACKUP_LABEL, window.sync_project_action),
            (UPDATE_LABEL, window.update_current_project),
            ('Local Rollback History', lambda: window.update_current_project(history=True)),
            ('Project Overview', window.show_project_overview),
        ])
        self.assignments_page = AssignmentWorkspace(window)
        self.stack.addWidget(self.assignments_page)
        window.setCentralWidget(self.shell)
        self._setup_assistance()
        self._setup_details()
        self.docks = window.findChildren(QDockWidget)
        for dock in self.docks:
            self.dock_visibility[dock] = dock is not window.tracking_sequences_dock
        self.save_timer = QTimer(self)
        self.save_timer.setSingleShot(True)
        self.save_timer.setInterval(2500)
        self.save_timer.timeout.connect(self.autosave)
        self.update_button.hide()
        self.update_help.hide()
        self.show_home()

    def command_page(self, title, commands):
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setContentsMargins(28, 24, 28, 24)
        layout.setSpacing(12)
        layout.addWidget(label(title, heading=True))
        for text, callback in commands:
            if isinstance(callback, QWidget):
                layout.addWidget(callback)
                continue
            if isinstance(callback, QAction):
                button = QToolButton()
                button.setDefaultAction(callback)
                button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
                button.setMinimumHeight(32)
                button.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
            else:
                button = QPushButton(text)
                button.setMinimumHeight(32)
                button.clicked.connect(callback)
            topic = {BACKUP_LABEL: 'backup', UPDATE_LABEL: 'update',
                     'Estimate Transfer Size': 'estimate', 'Review Changes': 'review'}.get(text)
            if topic:
                row = QWidget()
                row.setMaximumWidth(560)
                row_layout = QVBoxLayout(row)
                row_layout.setContentsMargins(0, 0, 0, 0)
                add_help_row(row_layout, button, topic)
                layout.addWidget(row)
            else:
                button.setMaximumWidth(520)
                layout.addWidget(button)
        layout.addStretch()
        scroll.setWidget(content)
        self.stack.addWidget(scroll)
        return scroll

    def _setup_assistance(self):
        window = self.window
        layout = self.editor.layout()
        self.assist = QWidget()
        assist_layout = QVBoxLayout(self.assist)
        assist_layout.setContentsMargins(0, 0, 0, 0)
        self.assist_choice = QComboBox()
        for title, toolbar, action in window._optional_top_toolbar_entries():
            layout.removeWidget(toolbar)
            assist_layout.addWidget(toolbar)
            self.assist_choice.addItem(title)
            action.setChecked(False)
            action.triggered.connect(lambda checked, widget=toolbar: self.select_assistance(widget, checked))
        self.assist_choice.currentIndexChanged.connect(self.refresh_assistance)
        layout.insertWidget(0, self.assist)
        controls = QHBoxLayout()
        self.assist_toggle = QCheckBox('Model assistance')
        self.assist_toggle.toggled.connect(self.refresh_assistance)
        controls.addWidget(self.assist_toggle)
        controls.addWidget(self.assist_choice)
        controls.addStretch()
        policy = QPushButton('Annotation Policy')
        policy.clicked.connect(window.show_annotation_policy)
        controls.addWidget(policy)
        layout.insertLayout(0, controls)
        self.refresh_assistance()

    def select_assistance(self, toolbar, checked):
        entries = self.window._optional_top_toolbar_entries()
        index = next(i for i, entry in enumerate(entries) if entry[1] is toolbar)
        if checked:
            self.assist_choice.setCurrentIndex(index)
            self.assist_toggle.setChecked(True)
            self.activate('Annotate')
        elif self.assist_choice.currentIndex() == index:
            self.assist_toggle.setChecked(False)
        self.refresh_assistance()

    def refresh_assistance(self, *_):
        enabled = self.assist_toggle.isChecked()
        self.assist.setVisible(enabled)
        self.assist_choice.setVisible(enabled)
        for i, (_, widget, action) in enumerate(self.window._optional_top_toolbar_entries()):
            selected = enabled and self.assist_choice.currentIndex() == i
            widget.setVisible(selected)
            action.setChecked(selected)
        if self.window.project_path:
            self.settings.setValue(self.key('assistance'), enabled)
            self.settings.setValue(self.key('assistance_model'), self.assist_choice.currentIndex())

    def _setup_details(self):
        toolbar = self.window.toolbar
        toolbar.layout().removeWidget(toolbar.visibility_panel)
        self.window.instance_dock.widget().layout().insertWidget(0, toolbar.visibility_panel)
        self.details = QCheckBox('Additional tools')
        self.details.setToolTip('ArUco, calibration, protected areas, and bulk deletion')
        toolbar.layout().insertWidget(2, self.details)
        self.details.toggled.connect(self.refresh_details)
        undo_row = QHBoxLayout()
        for text, callback, icon in (
            ('Undo', self.window.undo, QStyle.StandardPixmap.SP_ArrowBack),
            ('Redo', self.window.redo, QStyle.StandardPixmap.SP_ArrowForward),
        ):
            button = QToolButton()
            button.setIcon(self.window.style().standardIcon(icon))
            button.setToolTip(text)
            button.setAccessibleName(text)
            button.setMinimumSize(30, 30)
            button.clicked.connect(callback)
            undo_row.addWidget(button)
        toolbar.layout().removeWidget(self.details)
        undo_row.addWidget(self.details)
        undo_row.addStretch()
        toolbar.layout().insertLayout(2, undo_row)
        self.refresh_details()

    def refresh_details(self, *_):
        toolbar = self.window.toolbar
        enabled = self.details.isChecked()
        for widget in (toolbar.detect_aruco_btn, toolbar.load_aruco_csv_btn, toolbar.clear_aruco_btn,
                       toolbar.delete_all_btn, toolbar.clear_no_draw_zones_btn):
            widget.setVisible(enabled)
        for widget in toolbar.calibration_widgets:
            widget.setVisible(enabled and not self.assigned())

    def assigned(self):
        return bool(workspace(self.window.project_path))

    def key(self, suffix):
        identity = (self.window.contributor_session or {}).get('id', '')
        path = str(Path(self.window.project_path).resolve()) if self.window.project_path else ''
        digest = hashlib.sha256((identity + '\n' + path).encode()).hexdigest()
        return 'workflow/projects/' + digest + '/' + suffix

    def last_project(self):
        identity = (self.window.contributor_session or {}).get('id', '')
        return self.settings.value('workflow/last_project/' + identity, '', type=str)

    def identity_changed(self):
        self.window.invalidate_backup_estimate()
        session = self.window.contributor_session
        name = session['name'] if session else 'Choose Contributor'
        self.person.setText(self.person.fontMetrics().elidedText(name, Qt.TextElideMode.ElideRight, 180))
        self.person.setToolTip(name + '\nChange contributor or import/export your profile')
        self.upload_button.setText(ASSIGNED_LABEL if self.assigned() else BACKUP_LABEL)
        self.home.refresh()
        self.assignment_role = None
        self.home.inbox.reset()

    def project_loaded(self):
        self.generation += 1
        self.status = None
        self.assignments_page.clear()
        self.identity_changed()
        project = self.window.project_path
        self.project_label.setText(project.name)
        self.project_label.setToolTip(str(project))
        identity = (self.window.contributor_session or {}).get('id', '')
        self.settings.setValue('workflow/last_project/' + identity, str(project))
        for i, name in enumerate(self.names):
            self.tabs.setTabEnabled(i, not self.assigned() or name == 'Annotate')
            self.tabs.setTabVisible(i, not self.assigned() or name == 'Annotate')
        enabled = self.settings.value(self.key('assistance'), False, type=bool)
        model = self.settings.value(self.key('assistance_model'), 0, type=int)
        self.assist_toggle.blockSignals(True)
        self.assist_choice.blockSignals(True)
        self.assist_choice.setCurrentIndex(max(0, min(model, self.assist_choice.count() - 1)))
        self.assist_toggle.setChecked(enabled)
        self.assist_toggle.blockSignals(False)
        self.assist_choice.blockSignals(False)
        self.refresh_assistance()
        self.refresh_details()
        target = self.settings.value(self.key('workspace'), 'Annotate', type=str)
        self.activate(target)
        self.update_save_status()
        self.update_button.hide()
        self.update_help.hide()
        self.shared_status.setText('Shared status not checked' if self.linked() else 'Local project')
        self.check_button.setVisible(self.linked())
        self.upload_button.setVisible(self.linked())
        self.upload_help.setVisible(self.linked())
        if self.linked():
            QTimer.singleShot(0, self.check_shared)
        QTimer.singleShot(0, self.restore_view)

    def assignment_role_changed(self, result):
        self.assignment_role = result.get('director')
        index = self.names.index('Assignments')
        actor = self.window.contributor_session or {}
        allowed = not self.assigned() and (not self.assignment_role or
                                          self.assignment_role['id'] == actor.get('id'))
        self.tabs.blockSignals(True)
        self.tabs.setTabVisible(index, allowed)
        self.tabs.setTabEnabled(index, allowed)
        self.tabs.blockSignals(False)
        if not allowed and self.current == 'Assignments':
            self.activate('Home')

    def restore_position(self):
        video = self.settings.value(self.key('video'), '', type=str)
        if video:
            self.window.last_project_path = str(self.window.project_path)
            self.window.last_video_id = video
            self.window.last_frame_index_in_video = self.settings.value(self.key('frame'), 0, type=int)

    def remember_position(self):
        window = self.window
        if window.project_path and window.current_video_id and window.frames:
            self.settings.setValue(self.key('video'), window.current_video_id)
            self.settings.setValue(self.key('frame'), window._get_frame_idx_in_video(window.current_frame_idx))

    def remember_view(self):
        if not self.window.project_path or not self.window.canvas.image_item:
            return
        self.remember_position()
        state = self.window.canvas.get_view_state()
        transform, center = state['transform'], state['center']
        self.settings.setValue(self.key('view'), [transform.m11(), transform.m12(), transform.m21(),
            transform.m22(), transform.dx(), transform.dy(), center.x(), center.y(), state['zoom_factor']])

    def restore_view(self):
        values = self.settings.value(self.key('view'))
        if not isinstance(values, list) or len(values) != 9:
            return
        try:
            values = [float(value) for value in values]
            if not all(math.isfinite(value) for value in values) or values[0] <= 0 or values[3] <= 0:
                return
            self.window.canvas.restore_view_state({'transform': QTransform(*values[:6]),
                'center': QPointF(*values[6:8]), 'zoom_factor': values[8]})
        except (ValueError, TypeError):
            return

    def resume(self):
        if self.window.project_path:
            self.activate(self.settings.value(self.key('workspace'), 'Annotate', type=str))
        elif self.last_project():
            self.window.open_recent_project(self.last_project())

    def join_shared(self):
        if not setting_json('project_sync/destination', {}).get('root_id'):
            self.window.configure_project_sync()
        if setting_json('project_sync/destination', {}).get('root_id'):
            self.window.open_assigned_work()

    def show_home(self):
        self.activate('Home')
        self.home.refresh()
        if self.window.contributor_session:
            QTimer.singleShot(0, self.home.inbox.refresh)

    def activate(self, name):
        if name not in (*self.names, 'Home'):
            name = 'Annotate'
        if not self.window.project_path:
            name = 'Home'
        elif self.assigned() and name not in ('Home', 'Annotate'):
            name = 'Annotate'
        elif name == 'Assignments' and self.assignment_role and self.assignment_role['id'] != (
                self.window.contributor_session or {}).get('id'):
            name = 'Home'
        leaving_editor = self.current in ('Annotate', 'Review')
        entering_editor = name in ('Annotate', 'Review')
        if self.window.is_playing and not entering_editor:
            self.window.toggle_play()
        if leaving_editor and not entering_editor:
            self.remember_view()
            for dock in self.docks:
                self.dock_visibility[dock] = not dock.isHidden()
        self.current = name
        self.tabs.blockSignals(True)
        self.tabs.setCurrentIndex(self.names.index(name) if name in self.names else -1)
        self.tabs.blockSignals(False)
        self.tabs.setEnabled(bool(self.window.project_path))
        self.tabs.setVisible(name != 'Home')
        self.stack.setCurrentWidget(self.editor if entering_editor else
                                    self.train_page if name == 'Train' else
                                    self.assignments_page if name == 'Assignments' else
                                    self.manage_page if name == 'Manage' else self.home)
        for dock in self.docks:
            show = entering_editor and self.dock_visibility.get(dock, False)
            if dock is self.window.tracking_sequences_dock:
                show = name == 'Review' and not self.assigned()
            dock.setVisible(show)
        self.window.import_review_controls.setVisible(name == 'Review' and not self.assigned())
        self.window.video_batch_help.hide()
        self.window.tracking_filter_checkbox.setVisible(name == 'Review' and not self.assigned())
        self.status_bar.setVisible(bool(self.window.project_path))
        if name != 'Home':
            self.settings.setValue(self.key('workspace'), name)
        if name == 'Assignments':
            self.assignments_page.activate()
        if entering_editor:
            self.window.canvas.setFocus()
            if not leaving_editor:
                QTimer.singleShot(0, self.restore_view)

    def linked(self):
        if not self.window.project_path:
            return False
        if self.assigned():
            return True
        record = setting_json('project_sync/projects', {}).get(str(self.window.project_path.resolve()), {})
        return bool(record.get('enabled'))

    def pending(self):
        return self.settings.value(self.key('unpublished'), False, type=bool)

    def edited(self):
        self.window.invalidate_backup_estimate()
        self.generation += 1
        self.update_save_status()
        self.save_timer.start()

    def saved(self):
        self.window.invalidate_backup_estimate()
        self.generation += 1
        self.settings.setValue(self.key('unpublished'), True)
        self.settings.sync()
        self.update_save_status()

    def uploaded(self):
        self.generation += 1
        self.settings.setValue(self.key('unpublished'), False)
        self.settings.sync()
        self.update_save_status()
        self.shared_status.setText('Upload verified')
        self.update_button.hide()
        self.update_help.hide()

    def update_save_status(self):
        window = self.window
        if window.save_worker.failed_tasks:
            text = 'Local save failed'
        elif window.save_worker.save_queue.unfinished_tasks:
            text = 'Saving locally...'
        elif window.current_frame_modified:
            text = 'Unsaved changes'
        else:
            text = 'Saved locally'
        if self.linked() and self.pending():
            text += ' | Changes to upload'
        self.local_status.setText(text)

    def autosave(self):
        window = self.window
        if not window.project_path or window._sync_in_progress or not window.current_frame_modified:
            return
        if window.save_worker.save_queue.unfinished_tasks or getattr(window.canvas, 'is_drawing', False):
            self.save_timer.start()
            return
        try:
            # Snapshot the active mask without committing it or clearing brush undo.
            window._save_annotation_sources(annotations=window.canvas.get_annotations(), use_background=True)
            self.update_save_status()
        except Exception as exc:
            self.local_status.setText('Local save failed')
            self.local_status.setToolTip(str(exc))

    def upload(self):
        self.window.sync_current_project()

    def check_shared(self):
        if not self.linked() or (self.worker and self.worker.isRunning()) or self.window._sync_in_progress:
            return
        project = str(self.window.project_path)
        generation = self.generation
        configuration = setting_json('project_sync/destination', {})
        self.shared_status.setText('Checking shared project...')
        self.check_button.setEnabled(False)
        self.worker = OperationWorker(lambda progress, cancelled:
                                      project_activity(project, configuration, cancelled), self)
        worker = self.worker
        worker.finished.connect(lambda: self.checked(worker, project, generation))
        worker.start()

    def checked(self, worker, project, generation):
        if worker is not self.worker:
            return
        self.check_button.setEnabled(True)
        self.worker = None
        worker.deleteLater()
        if str(self.window.project_path) != project:
            if self.linked():
                QTimer.singleShot(0, self.check_shared)
            return
        if generation != self.generation:
            self.shared_status.setText('Local work changed during the check; check again for shared status')
            self.update_button.hide()
            self.update_help.hide()
            return
        if worker.error:
            self.shared_status.setText('Shared folder unavailable or needs attention; local work is available')
            self.shared_status.setToolTip(str(worker.error))
            self.update_button.hide()
            self.update_help.hide()
            return
        result = worker.result
        self.status = result
        self.shared_status.setText(result['message'])
        self.shared_status.setToolTip(result['message'])
        self.update_button.setVisible(result['status'] in ('update', 'attention') and not self.assigned())
        self.update_help.setVisible(not self.update_button.isHidden())
        if generation == self.generation and result['changed'] is not None:
            self.settings.setValue(self.key('unpublished'), result['changed'])
        director = result.get('director')
        if director:
            actor = self.window.contributor_session or {}
            role = 'Project director' if director['id'] == actor.get('id') else 'Contributor'
            self.person.setText(self.person.fontMetrics().elidedText(
                actor.get('name', '') + ' | ' + role, Qt.TextElideMode.ElideRight, 180))
            self.person.setToolTip(actor.get('name', '') + ' | ' + role + '\nProject director: ' + director['name']
                                   + '\nChange contributor or import/export your profile')
        self.update_save_status()

    def stop_check(self):
        self.save_timer.stop()
        stopped = self.home.inbox.stop()
        if self.worker and self.worker.isRunning():
            self.worker.requestInterruption()
            stopped = self.worker.wait(2000) and stopped
        return stopped

    def confirm_upload_on_close(self):
        if not self.linked() or not self.pending():
            return True
        box = QMessageBox(self.window)
        box.setTextFormat(Qt.TextFormat.PlainText)
        box.setWindowTitle('Upload Saved Progress?')
        box.setText('Your work is saved on this computer. Upload it to the shared project before closing?')
        upload = box.addButton('Upload and Close', QMessageBox.ButtonRole.AcceptRole)
        local = box.addButton('Close with Work Saved Locally', QMessageBox.ButtonRole.ActionRole)
        box.addButton(QMessageBox.StandardButton.Cancel)
        box.setDefaultButton(upload)
        box.exec()
        if box.clickedButton() is local:
            return True
        if box.clickedButton() is upload:
            return bool(self.window.sync_current_project())
        return False

    def prepare_close(self):
        if not self.assignments_page.confirm_discard():
            return False
        self.save_timer.stop()
        self.remember_view()
        window = self.window
        try:
            if window.project_path and window.current_frame_modified:
                window._save_annotation_sources(annotations=window.canvas.get_annotations())
            window.save_worker.wait_until_idle()
        except Exception as exc:
            QMessageBox.warning(window, 'Local Work Not Saved',
                                str(exc) + '\nThe application will stay open.')
            return False
        if not self.confirm_upload_on_close():
            return False
        if not self.stop_check():
            window.status_label.setText('Shared-folder check is stopping; please close again in a moment')
            return False
        return True
