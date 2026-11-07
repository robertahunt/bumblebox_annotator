"""Exercise profile management through the chooser without an actual identity."""

from PyQt6.QtCore import QTimer
from PyQt6.QtWidgets import QApplication


def manage_profiles(chooser, operation):
    def respond():
        manager = QApplication.activeModalWidget()
        operation(manager)
        manager.reject()
    QTimer.singleShot(0, respond)
    chooser.manage.click()
