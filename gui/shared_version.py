"""Visible shared-version provenance with nonblocking, read-only freshness checks."""

from datetime import datetime
from pathlib import Path

from PyQt6.QtCore import QTimer, Qt, QUrl, pyqtSignal, pyqtSlot
from PyQt6.QtGui import QDesktopServices
from PyQt6.QtWidgets import (QApplication, QHBoxLayout, QLabel, QMessageBox,
                            QPushButton, QStyle, QToolButton, QVBoxLayout, QWidget)

from core.shared_project_review import check_shared_version
from gui.project_sync_dialog import OperationWorker


def display_time(value):
    if not value:
        return 'Not recorded'
    try:
        return datetime.fromisoformat(value.replace('Z', '+00:00')).astimezone().strftime('%b %d, %Y %H:%M %Z')
    except (ValueError, TypeError):
        return str(value)


class VersionCheckWorker(OperationWorker):
    @pyqtSlot()
    def cancel(self):
        self.requestInterruption()


class SharedVersionHeader(QWidget):
    refresh_requested = pyqtSignal()
    compare_requested = pyqtSignal()

    def __init__(self, project, configuration, parent=None):
        super().__init__(parent)
        self.project, self.configuration = project, dict(configuration)
        self.snapshot = None
        self.worker = None
        self.stopping = False
        self.last_checked = None
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.title = QLabel()
        font = self.title.font()
        font.setBold(True)
        self.title.setFont(font)
        self.publication = QLabel()
        self.freshness = QLabel('Version freshness not checked')
        for label in (self.title, self.publication, self.freshness):
            label.setTextFormat(Qt.TextFormat.PlainText)
            label.setWordWrap(True)
            layout.addWidget(label)
        row = QHBoxLayout()
        self.location = QToolButton()
        self.location.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_DirOpenIcon))
        self.location.setToolTip('Open the displayed shared version folder')
        self.location.setAccessibleName('Open shared version location')
        self.location.clicked.connect(self.open_location)
        self.details = QToolButton()
        self.details.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_MessageBoxInformation))
        self.details.setToolTip('Shared version details and local project location')
        self.details.clicked.connect(self.show_details)
        self.check = QToolButton()
        self.check.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_BrowserReload))
        self.check.setToolTip('Check for a newer shared version without changing this table')
        self.check.setAccessibleName('Check shared version')
        self.check.clicked.connect(self.check_version)
        self.reload = QPushButton('Refresh Assignments')
        self.reload.clicked.connect(self.refresh_requested)
        self.reload.hide()
        self.compare = QPushButton('Review Local vs Shared Changes...')
        self.compare.clicked.connect(self.compare_requested)
        for button in (self.location, self.details, self.check, self.reload, self.compare):
            row.addWidget(button)
        row.addStretch()
        layout.addLayout(row)
        self.timer = QTimer(self)
        self.timer.setInterval(60000)
        self.timer.timeout.connect(self.automatic_check)
        self.timer.start()

    def key(self):
        return (tuple(self.snapshot['reference']), self.snapshot['manifest_hash']) if self.snapshot else None

    def set_snapshot(self, snapshot):
        self.snapshot = snapshot
        self.last_checked = None
        self.title.setText(Path(self.project).name + ' | Shared version')
        manifest = snapshot['manifest'] if snapshot else {}
        self.publication.setText('Published by ' + manifest.get('contributor', {}).get('name', 'Not recorded') +
                                 ' | Published ' + display_time(manifest.get('created_at')))
        self.freshness.setText('Version freshness not checked' if snapshot else 'Refresh to load shared version details')
        self.freshness.setToolTip('')
        self.reload.hide()
        for button in (self.location, self.details, self.compare):
            button.setEnabled(bool(snapshot))
        self.check.setEnabled(bool(snapshot) and self.worker is None)
        if self.worker:
            self.worker.cancel()
        QTimer.singleShot(0, self.automatic_check)

    def showEvent(self, event):
        super().showEvent(event)
        if self.last_checked is None:
            QTimer.singleShot(0, self.automatic_check)

    @pyqtSlot()
    def automatic_check(self):
        if self.isVisible():
            self.check_version()

    def check_version(self):
        if self.stopping or self.worker or not self.snapshot:
            return
        snapshot = self.snapshot
        self.freshness.setText('Checking shared version...')
        self.freshness.setToolTip('')
        self.check.setEnabled(False)
        worker = VersionCheckWorker(lambda progress, cancelled:
            check_shared_version(self.project, self.configuration, snapshot, cancelled=cancelled), QApplication.instance())
        worker.version_key = self.key()
        self.worker = worker
        self.destroyed.connect(worker.cancel)
        worker.finished.connect(self.checked)
        worker.finished.connect(worker.deleteLater)
        worker.start()

    @pyqtSlot()
    def checked(self):
        worker, self.worker = self.worker, None
        if self.stopping:
            return
        self.check.setEnabled(bool(self.snapshot))
        if worker.version_key != self.key():
            self.automatic_check()
            return
        if worker.error:
            last = '\nLast successful check: ' + display_time(self.last_checked) if self.last_checked else ''
            self.freshness.setText('Unable to check shared version. Displayed version unchanged.' + last)
            self.freshness.setToolTip(str(worker.error))
            return
        self.last_checked = worker.result['checked_at']
        changed = worker.result['changed']
        self.freshness.setText(('Shared publications changed. Refresh to inspect the current version.' if changed else
                               'No newer shared publication found.') + '\nLast checked: ' + display_time(self.last_checked))
        self.reload.setVisible(changed)

    def open_location(self):
        if self.snapshot:
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(self.snapshot['manifest_path'].parent)))

    def show_details(self):
        if self.snapshot:
            QMessageBox.information(self, 'Shared Version Details',
                self.title.text() + '\n' + self.publication.text() +
                '\nVersion: ' + self.snapshot['reference'][1] +
                '\nShared manifest: ' + str(self.snapshot['manifest_path']) +
                '\nLocal working copy: ' + str(self.project) +
                '\nLast successful version check: ' + display_time(self.last_checked))

    def cancel(self):
        self.stopping = True
        self.timer.stop()
        if self.worker:
            self.worker.cancel()

    def stop(self):
        self.cancel()
        return self.worker.wait(2000) if self.worker and self.worker.isRunning() else True
