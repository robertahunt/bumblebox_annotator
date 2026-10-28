"""Shared wording and accessible contextual help for project transfers."""

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QHBoxLayout, QMessageBox, QToolButton


BACKUP_LABEL = 'Back Up Local Project'
ASSIGNED_LABEL = 'Upload Assigned Annotations'
UPDATE_LABEL = 'Review Shared Updates...'
RESTORE_LABEL = 'Restore from Backup...'

HELP = {
    'backup': (BACKUP_LABEL,
        'Save a version of the managed files in this local project to the shared folder. '
        'Extracted frames and annotations are included; original videos are optional.\n\n'
        'Files not present locally are NOT added from an older backup. Older backups remain available. '
        'This is not a way to combine contributors\' work.\n\n'
        'Estimate Transfer Size, then Review Changes before backing up. '
        'For a downloaded assignment, use Upload Assigned Annotations instead.\n\n'
        'Uploaded content is verified. Unchanged stored content gets presence and size checks only. '
        'Use Check Backup Integrity separately to read and verify every file in a selected backup.'),
    'assigned': (ASSIGNED_LABEL,
        'The routine upload for a downloaded assignment. Send permitted annotation changes '
        'while preserving shared files you did not download. Deleting a downloaded annotation '
        'can also be an outgoing change; shared-video masks affect undownloaded frames too.\n\n'
        'Conflicting changes are blocked for review. Uploading saves progress; '
        'Submit Current Video for Review separately marks a video ready for review.\n\n'
        'Uploaded content is verified. Unchanged published files are referenced without '
        'rechecking them; use Check Backup Integrity for a full stored-content check.'),
    'download': ('Download Assigned Work',
        'Keep one local working copy per project and contributor. Downloads add new assignments '
        'and update unchanged local work in that same folder. Local edits are kept; conflicts stop '
        'for director review. Original videos are not downloaded.\n\n'
        'Working copies live in persistent application data, outside the app installation. '
        'An existing downloaded copy can be reused instead. Work no longer assigned stays locally '
        'but is not uploaded. Upload Assigned Annotations sends progress back.'),
    'update': ('Review Shared Updates',
        'Compare published work with your existing local project before applying an update. '
        'Conflicts require review; this does not silently replace local edits or upload them. '
        'Applied changes have local rollback history.\n\n'
        'Use this for an existing full local project, not Restore from Backup. '
        'Assigned copies use their separate assignment workflow.'),
    'restore': ('Restore from Backup',
        'Recovery or full-project setup, not a routine way to submit annotations. '
        'Choose a completed published version, then create a verified copy in a NEW local folder. '
        'Existing local projects are never overwritten.\n\n'
        'The entire selected version is restored, which may require substantial disk space. '
        'Videos excluded from that backup cannot be restored from it.\n\n'
        'The newest timestamp does not guarantee that a version includes every contributor\'s work. '
        'Check the contributor and publication date, and ask the project director which version to use. '
        'Restoring neither merges versions nor uploads your local changes.'),
    'estimate': ('Estimate Transfer Size',
        'Check which local files need storage on the shared drive. The project inventory is '
        'the whole local dataset, not the amount to transfer. Estimated additional storage '
        'accounts for files already stored there.\n\n'
        'Required before Back Up Local Project. Changes to files or backup settings require a new estimate.'),
    'review': ('Review Changes',
        'Inspect outgoing additions, changes, removals, and excluded files grouped by video. '
        'This compares the estimated local backup with your previous published version.\n\n'
        'It does not transfer data or approve annotation quality. A file removed from this '
        'new backup remains available in older backups.'),
    'submit': ('Submit Current Video for Review',
        'Upload assigned annotation changes and mark the current video ready for the project '
        'director to review. Use Upload Assigned Annotations to save progress without marking it ready.'),
}


def help_button(parent, topic):
    """Keep help usable even when the associated command is disabled."""
    button = QToolButton(parent)
    button.setObjectName('contextHelp')
    button.setText('?')
    button.setFixedSize(28, 28)
    button.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

    def details():
        return HELP[topic() if callable(topic) else topic]

    def refresh():
        title, _ = details()
        button.setAccessibleName('Help: ' + title)
        button.setToolTip('About ' + title)

    def show():
        refresh()
        title, text = details()
        box = QMessageBox(button.window())
        box.setWindowTitle(title)
        box.setTextFormat(Qt.TextFormat.PlainText)
        box.setIcon(QMessageBox.Icon.Information)
        box.setText(text)
        box.exec()

    refresh()
    if callable(topic):
        button.setAccessibleName('Help: Project transfer')
        button.setToolTip('About this project transfer')
    button.clicked.connect(show)
    return button


def add_help_row(layout, button, topic):
    row = QHBoxLayout()
    row.setSpacing(8)
    row.addWidget(button, 1)
    help_widget = help_button(button.parentWidget(), topic)
    row.addWidget(help_widget)
    layout.addLayout(row)
    return help_widget
