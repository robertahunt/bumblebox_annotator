"""Searchable outgoing file changes, grouped by video, from a saved estimate."""

from collections import Counter, defaultdict

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (QAbstractItemView, QComboBox, QDialog, QDialogButtonBox,
                            QHBoxLayout, QHeaderView, QLabel, QLineEdit, QStyle,
                            QToolButton, QTreeWidget, QTreeWidgetItem, QVBoxLayout)

from gui.project_sync_dialog import _size_transfer_dialog


def _size(value):
    if value is None:
        return '-'
    for unit in ('B', 'KiB', 'MiB', 'GiB', 'TiB'):
        if value < 1024 or unit == 'TiB':
            return f'{value:,.0f} B' if unit == 'B' else f'{value:,.1f} {unit}'
        value /= 1024


class PublicationReviewDialog(QDialog):
    def __init__(self, estimate, parent=None):
        super().__init__(parent)
        self.setWindowTitle('Review Changes Before Publishing')
        self.review = estimate['review']
        self.rows = self.review['rows']
        self.groups = {}
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(10)
        baseline = self.review['baseline']
        baseline_text = (f"Compared with {baseline['contributor_name']}'s published backup: "
                         f"{baseline['created_at']} | {baseline['revision']}" if baseline else
                         'First backup for this contributor. All included files are additions.')
        self.baseline = QLabel(baseline_text)
        self.baseline.setTextFormat(Qt.TextFormat.PlainText)
        self.baseline.setWordWrap(True)
        self.baseline.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(self.baseline)
        self.summary = QLabel()
        self.summary.setWordWrap(True)
        layout.addWidget(self.summary)

        filters = QHBoxLayout()
        self.search = QLineEdit()
        self.search.setPlaceholderText('Search videos or files')
        self.search.setAccessibleName('Search outgoing changes')
        self.search.setClearButtonEnabled(True)
        self.actions = QComboBox()
        self.actions.setAccessibleName('Filter by change type')
        self.actions.addItem('All changes', None)
        for action in ('Added', 'Changed', 'Removed', 'Excluded'):
            self.actions.addItem(action, action)
        filters.addWidget(self.search, 1)
        filters.addWidget(self.actions)
        for expand, icon, title in ((True, QStyle.StandardPixmap.SP_ArrowDown, 'Expand video groups'),
                                    (False, QStyle.StandardPixmap.SP_ArrowUp, 'Collapse video groups')):
            button = QToolButton()
            button.setIcon(self.style().standardIcon(icon))
            button.setToolTip(title)
            button.setAccessibleName(title)
            button.clicked.connect(lambda checked=False, expand=expand: self.expand_groups(expand))
            filters.addWidget(button)
        layout.addLayout(filters)

        self.table = QTreeWidget()
        self.table.setHeaderLabels(['Video / file', 'Change', 'Scope', 'Before', 'After'])
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.table.setUniformRowHeights(True)
        self.table.setAlternatingRowColors(True)
        self.table.setTextElideMode(Qt.TextElideMode.ElideMiddle)
        self.table.header().setStretchLastSection(False)
        self.table.header().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        for column, width in ((1, 92), (2, 140), (3, 82), (4, 82)):
            self.table.header().setSectionResizeMode(column, QHeaderView.ResizeMode.Interactive)
            self.table.setColumnWidth(column, width)
        self.table.itemExpanded.connect(self.populate_group)
        self.table.currentItemChanged.connect(self.show_details)
        layout.addWidget(self.table, 1)
        self.empty = QLabel()
        self.empty.setWordWrap(True)
        layout.addWidget(self.empty)
        self.details = QLabel()
        self.details.setTextFormat(Qt.TextFormat.PlainText)
        self.details.setWordWrap(True)
        self.details.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.details.setMinimumHeight(42)
        layout.addWidget(self.details)
        note = QLabel('Removed and excluded files remain in older backups. '
                      'File changes are not counts of edited masks. Shared-video annotations affect the whole video.')
        note.setWordWrap(True)
        layout.addWidget(note)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        self.search.textChanged.connect(self.refresh)
        self.actions.currentIndexChanged.connect(self.refresh)
        _size_transfer_dialog(self, 1100, 720, minimum_height=480)
        self.refresh()

    def refresh(self, *_):
        query, action = self.search.text().strip().casefold(), self.actions.currentData()
        self.table.clear()
        self.groups = defaultdict(list)
        for row in self.rows:
            if action and row['action'] != action:
                continue
            haystack = ' '.join((row['path'], row['video'] or 'Project files', row['scope'])).casefold()
            if query and query not in haystack:
                continue
            self.groups[row['video']].append(row)
        visible = sum(len(rows) for rows in self.groups.values())
        for video in sorted(self.groups, key=lambda key: (key is not None, (key or '').casefold())):
            rows = self.groups[video]
            counts = Counter(row['action'] for row in rows)
            count = len(rows)
            item = QTreeWidgetItem([video if video is not None else 'Project files',
                                   f"{count:,} {'file' if count == 1 else 'files'}"])
            item.setData(0, Qt.ItemDataRole.UserRole, video)
            item.setChildIndicatorPolicy(QTreeWidgetItem.ChildIndicatorPolicy.ShowIndicator)
            for column in range(self.table.columnCount()):
                font = item.font(column)
                font.setBold(True)
                item.setFont(column, font)
            item.setToolTip(0, (video or 'Project files') + '\n' +
                            ', '.join(f'{count:,} {kind.lower()}' for kind, count in counts.items()))
            self.table.addTopLevelItem(item)
        counts = self.review['counts']
        self.summary.setText(' | '.join(f"{counts.get(kind, 0):,} {kind.lower()}"
                                       for kind in ('Added', 'Changed', 'Removed', 'Excluded')) +
                             f" | {self.review['unchanged']:,} unchanged\n"
                             f'Showing {visible:,} of {len(self.rows):,} changed/excluded files')
        self.empty.setVisible(not visible)
        self.empty.setText('No changes match these filters.' if self.rows else 'No file changes since this backup.')
        self.details.clear()
        # Populate only expanded groups; a first backup can contain tens of thousands of files.
        if visible <= 100:
            self.expand_groups(True)

    def expand_groups(self, expand):
        for index in range(self.table.topLevelItemCount()):
            item = self.table.topLevelItem(index)
            if expand:
                self.populate_group(item)
            item.setExpanded(expand)

    def populate_group(self, item):
        if item.parent() is not None or item.childCount():
            return
        for row in self.groups[item.data(0, Qt.ItemDataRole.UserRole)]:
            child = QTreeWidgetItem([row['path'], row['action'], row['scope'],
                                    _size(row['before_bytes']), _size(row['after_bytes'])])
            child.setData(0, Qt.ItemDataRole.UserRole, row)
            for column in range(5):
                child.setToolTip(column, child.text(column))
            child.setToolTip(1, {
                'Added': 'New path in this contributor\'s version. Its contents may already be stored remotely.',
                'Changed': 'File contents differ; timestamp-only changes are not included.',
                'Removed': 'Absent from the new version. Older backups are retained.',
                'Excluded': 'Omitted because Include original videos is unchecked. Older backups are retained.',
            }[row['action']])
            for column, key in ((3, 'before_bytes'), (4, 'after_bytes')):
                child.setTextAlignment(column, Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
                if row[key] is not None:
                    child.setToolTip(column, f'{row[key]:,} bytes')
            item.addChild(child)

    def show_details(self, item, previous):
        if item is None:
            self.details.clear()
        elif item.parent() is None:
            self.details.setText(item.toolTip(0))
        else:
            row = item.data(0, Qt.ItemDataRole.UserRole)
            self.details.setText(f"{row['path']}\n{row['action']} | {row['scope']}")
