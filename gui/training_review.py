"""Review one saved image at a time in the normal annotation canvas."""

from pathlib import Path

from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtWidgets import (QComboBox, QDockWidget, QHBoxLayout, QLabel,
                            QMessageBox, QPushButton, QVBoxLayout, QWidget)

from core.training_review import create_queue, record_decision
from core.validation_review import selected_frame_indices


class TrainingReviewDock(QDockWidget):
    def __init__(self, window):
        super().__init__('Bee training review', window)
        self.setObjectName('beeTrainingReviewDock')
        self.window = window
        self.project = None
        self.queue = None
        self.position = None
        self.busy = False
        body = QWidget()
        layout = QVBoxLayout(body)
        self.label = QLabel()
        self.label.setTextFormat(Qt.TextFormat.PlainText)
        self.label.setWordWrap(True)
        layout.addWidget(self.label)
        note = QLabel('Inspect all visible bees; edit masks in the canvas if needed. '
                      'Complete adds this frame to training. Incomplete excludes it. '
                      'Both save and advance. Re-export COCO before training.')
        note.setWordWrap(True)
        layout.addWidget(note)
        row = QHBoxLayout()
        self.filter = QComboBox()
        for label, key in [('Pending', 'pending'), ('Incomplete', 'incomplete'),
                           ('Complete', 'complete'), ('All frames', 'all')]:
            self.filter.addItem(label, key)
        row.addWidget(self.filter)
        self.previous = QPushButton('Previous')
        self.complete = QPushButton('Complete — add & next')
        self.incomplete = QPushButton('Incomplete — exclude & next')
        self.skip = QPushButton('Skip / next')
        self.reopen = QPushButton('Open review frame')
        for button in (self.previous, self.complete, self.incomplete, self.skip, self.reopen):
            row.addWidget(button)
        layout.addLayout(row)
        self.setWidget(body)
        self.filter.currentIndexChanged.connect(lambda: self.navigate(reset=True))
        self.previous.clicked.connect(lambda: self.navigate(-1))
        self.skip.clicked.connect(lambda: self.navigate())
        self.complete.clicked.connect(lambda: self.decide(True))
        self.incomplete.clicked.connect(lambda: self.decide(False))
        self.reopen.clicked.connect(self.open_current)
        self.timer = QTimer(self)
        self.timer.setInterval(250)
        self.timer.timeout.connect(self.refresh_enabled)
        self.timer.start()

    def start(self):
        self.project = Path(self.window.project_path)
        self.queue = create_queue(self.project)
        self.position = None
        if hasattr(self.window, 'workflow'):
            self.window.workflow.activate('Annotate')
        self.show()
        self.raise_()
        self.navigate(reset=True)

    def matching(self):
        return [i for i, item in enumerate(self.queue['items'])
                if self.filter.currentData() in ('all', item['status'])]

    def same_project(self):
        return self.project is not None and self.window.project_path is not None and (
            Path(self.window.project_path) == self.project)

    def current_matches(self):
        if not self.same_project() or self.position is None:
            return False
        item = self.queue['items'][self.position]
        w = self.window
        return (w.current_video_id == item['video'] and 0 <= w.current_frame_idx < len(w.frames)
                and Path(w.frames[w.current_frame_idx]).name == f"frame_{item['frame']:06d}.jpg"
                and Path(w.frames[w.current_frame_idx]).parent.name == item['video'])

    def refresh_enabled(self):
        enabled = not self.busy and self.current_matches()
        self.complete.setEnabled(enabled)
        self.incomplete.setEnabled(enabled)
        for button in (self.previous, self.skip, self.reopen):
            button.setEnabled(not self.busy and self.same_project() and self.position is not None)
        self.filter.setEnabled(not self.busy and self.same_project())

    def save_current(self):
        w = self.window
        if w.current_video_id and 0 <= w.current_frame_idx < len(w.frames):
            w._save_annotation_sources(use_background=False)
            w.save_worker.wait_until_idle()

    def navigate(self, direction=1, reset=False):
        if not self.same_project() or self.busy:
            return
        indices = self.matching()
        if not indices:
            self.position = None
            self.update_label()
            self.refresh_enabled()
            return
        if reset or self.position is None:
            position = indices[0]
        else:
            later = [i for i in indices if i > self.position] if direction > 0 else [i for i in indices if i < self.position]
            position = (later[0] if later else indices[0]) if direction > 0 else (later[-1] if later else indices[-1])
        self.position = position
        self.open_current()

    def update_label(self):
        counts = {status: sum(i['status'] == status for i in self.queue['items'])
                  for status in ('pending', 'complete', 'incomplete')}
        text = f"{counts['complete']} complete · {counts['incomplete']} incomplete · {counts['pending']} pending"
        if self.position is not None:
            item = self.queue['items'][self.position]
            text += f"\n{self.position + 1}/{len(self.queue['items'])} · {item['video']} · frame {item['frame']} · {item['status']}"
        else:
            text += '\nNo frames in this view. Choose another view to revisit decisions.'
        self.label.setText(text)

    def open_current(self):
        if not self.same_project() or self.position is None or self.busy:
            return
        self.busy = True
        self.refresh_enabled()
        try:
            self.save_current()
            w = self.window
            if w.is_playing:
                w.toggle_play()
            item = self.queue['items'][self.position]
            target = self.project / 'frames' / item['video'] / f"frame_{item['frame']:06d}.jpg"
            if not target.is_file():
                raise ValueError('Source image missing: ' + str(target))
            # Unselected candidates must remain visible in the frame sidebar.
            w.split_filter_combo.setCurrentIndex(0)
            if w.current_video_id != item['video']:
                w.load_video_frames(item['video'])
            index = next((i for i, path in enumerate(w.frames) if Path(path) == target), None)
            if index is None:
                raise ValueError('The review frame could not be loaded.')
            w.load_frame(index)
            if not self.current_matches():
                raise ValueError('Frame loading did not finish. Review decisions are disabled.')
            self.update_label()
        except Exception as exc:
            QMessageBox.warning(self, 'Review Frame Unavailable', str(exc))
        finally:
            self.busy = False
            self.refresh_enabled()

    def decide(self, complete):
        if self.busy or not self.current_matches():
            return
        self.busy = True
        self.refresh_enabled()
        success = False
        try:
            self.save_current()
            item = self.queue['items'][self.position]
            w = self.window
            self.queue = record_decision(self.project, item['video'], item['frame'], complete,
                                         getattr(w, 'contributor_session', None))
            selected = set(selected_frame_indices(self.project, item['video']))
            w.frame_selected = [int(Path(path).stem.split('_')[1]) in selected for path in w.frames]
            w.frame_list.blockSignals(True)
            try:
                w.update_frame_list()
                if w.current_frame_idx in w.frame_list_to_frames_map:
                    w.frame_list.setCurrentRow(w.frame_list_to_frames_map.index(w.current_frame_idx))
            finally:
                w.frame_list.blockSignals(False)
            w.coco_export_dirty = True
            if hasattr(w, 'workflow'):
                w.workflow.saved()
            success = True
        except Exception as exc:
            QMessageBox.warning(self, 'Review Decision Not Saved', str(exc))
        finally:
            self.busy = False
            self.refresh_enabled()
        if success:
            self.navigate()
