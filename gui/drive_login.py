"""Explicit, nonblocking drive sign-in with an independent terminal window."""

from PyQt6.QtCore import QTimer, pyqtSignal
from PyQt6.QtWidgets import QInputDialog, QMessageBox, QPushButton, QStyle

from core.drive_login import start_login


class DriveLoginButton(QPushButton):
    completed = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__('Sign In to Drive...', parent)
        self.setAutoDefault(False)
        self.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_ComputerIcon))
        self.setToolTip('Open a terminal for Kerberos (kinit). Your password stays in that terminal.')
        self.process = None
        self.timer = QTimer(self)
        self.timer.setInterval(300)
        self.timer.timeout.connect(self.poll)
        self.clicked.connect(self.sign_in)

    def sign_in(self):
        from gui.project_sync_dialog import setting_json, save_setting
        if self.process is not None:
            return
        principal, accepted = QInputDialog.getText(self, 'Research Drive Sign-In',
            'Kerberos username (not your contributor display name):',
            text=setting_json('lab/kerberos_principal', ''))
        if not accepted:
            return
        try:
            self.process = start_login(principal)
        except (ValueError, OSError) as exc:
            QMessageBox.warning(self, 'Terminal Sign-In Unavailable', str(exc))
            return
        save_setting('lab/kerberos_principal', principal.strip())
        self.setEnabled(False)
        self.setText('Complete Sign-In, Then Press Enter in Terminal')
        self.timer.start()

    def poll(self):
        if self.process is None or self.process.poll() is None:
            return
        code = self.process.returncode
        self.process = None
        self.timer.stop()
        self.setEnabled(True)
        self.setText('Sign In to Drive...')
        if code:
            QMessageBox.warning(self, 'Sign-In Did Not Complete',
                'The terminal sign-in did not report success, or its window was closed early. '
                'Read the Kerberos error in that terminal when retrying, or run kinit in your '
                'own terminal. The app will now recheck the drive; no project data was transferred.')
        # Even exit code zero does not prove the drive is mounted or writable.
        self.completed.emit()
