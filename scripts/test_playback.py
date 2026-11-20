#!/usr/bin/env python3
"""Play and Pause: stepping through the frames in view on a disposable project."""

import gc
import os
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np
from PyQt6.QtCore import QSettings, Qt
from PyQt6.QtTest import QTest
from PyQt6.QtWidgets import QApplication, QMessageBox

FRAMES = 4


class PlaybackTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        from core.contributors import new_session
        from core.project_manager import ProjectManager
        from core.project_sync import _write_json
        from gui.main_window import MainWindow
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        project = root / 'project'
        ProjectManager().create_project(project, 'Playback', hive_annotation_scope='frame')
        frames = project / 'frames/video'
        frames.mkdir(parents=True, exist_ok=True)
        for index in range(FRAMES):
            cv2.imwrite(str(frames / f'frame_{index:06d}.jpg'), np.full((64, 96, 3), 40 * index, np.uint8))
        _write_json(frames / 'video_metadata.json', {'selected_frames': list(range(FRAMES)), 'split': 'train',
                                                    'total_frames': FRAMES})
        (project / 'input_data/train').mkdir(parents=True, exist_ok=True)
        (project / 'input_data/train/video.mp4').touch()

        settings = QSettings(str(root / 'settings.ini'), QSettings.Format.IniFormat)
        for target in ('gui.main_window.QSettings', 'gui.project_sync_dialog.QSettings'):
            mock = patch(target, lambda *a, **k: settings)
            mock.start()
            self.addCleanup(mock.stop)
        env = patch.dict(os.environ, {'XDG_DATA_HOME': str(root / 'appdata')})
        env.start()
        self.addCleanup(env.stop)
        with patch.object(MainWindow, 'load_settings'):
            self.window = w = MainWindow()

        def close():
            with patch.object(QMessageBox, 'question', return_value=QMessageBox.StandardButton.Discard):
                w.close()
            self.app.processEvents()
        self.addCleanup(close)
        w.set_contributor(new_session('Annotator'))
        with patch.object(QMessageBox, 'critical') as errors, patch.object(w, 'show_annotation_policy'):
            w.load_project(project)
            errors.assert_not_called()
        for row in range(w.video_list.count()):
            item = w.video_list.item(row)
            if item.data(Qt.ItemDataRole.UserRole) == 'video':
                w.video_list.setCurrentItem(item)
                w._load_video_list_item(item)
        self.app.processEvents()
        self.assertEqual(len(w.frame_list_to_frames_map), FRAMES)
        self.go_to(0)

        # Record each frame shown, as its row in the frame list.
        self.shown = []
        self.timer_active_during_load = []
        real_load = w.load_frame

        def load_frame(index):
            self.timer_active_during_load.append(w.play_timer.isActive())
            real_load(index)
            self.shown.append(w.frame_list_to_frames_map.index(w.current_frame_idx))
        w.load_frame = load_frame

    def go_to(self, row):
        self.window.load_frame(self.window.frame_list_to_frames_map[row])
        self.app.processEvents()

    def row(self):
        return self.window.frame_list_to_frames_map.index(self.window.current_frame_idx)

    def wait_until_stopped(self, seconds=10):
        deadline = time.monotonic() + seconds
        while self.window.is_playing and time.monotonic() < deadline:
            QTest.qWait(10)
        self.assertFalse(self.window.is_playing, 'playback did not stop')

    def show_window(self):
        """On screen and active, so keyboard focus moves as it does in use."""
        self.window.show()
        self.assertTrue(QTest.qWaitForWindowActive(self.window))

    def wait_for_frames(self, count, seconds=5):
        deadline = time.monotonic() + seconds
        while len(self.shown) < count and time.monotonic() < deadline:
            QTest.qWait(5)
        self.assertGreaterEqual(len(self.shown), count, 'no frame was played')

    def assert_stopped(self):
        w = self.window
        self.assertFalse(w.is_playing)
        self.assertFalse(w.play_timer.isActive())
        self.assertIn('Play', w.play_button.text())
        self.assertTrue(w.instance_list.isEnabled())
        self.assertEqual(gc.get_freeze_count(), 0)

    def test_play_runs_to_the_last_frame_and_stops(self):
        w = self.window
        self.shown.clear()
        w.play_button.click()
        self.assertTrue(w.is_playing)
        self.assertIn('Pause', w.play_button.text())
        self.wait_until_stopped()
        self.assertEqual(self.shown, [1, 2, 3])
        self.assertEqual(self.row(), FRAMES - 1)
        self.assert_stopped()
        # The next frame is scheduled only once the current one has loaded, so
        # clicks are handled between frames however long a frame takes.
        self.assertTrue(w.play_timer.isSingleShot())
        self.assertEqual(self.timer_active_during_load[-3:], [False, False, False])

    def test_play_on_the_last_frame_starts_again_from_the_first(self):
        self.go_to(FRAMES - 1)
        self.shown.clear()
        self.window.play_button.click()
        self.assertTrue(self.window.is_playing)
        self.wait_until_stopped()
        self.assertEqual(self.shown, [0, 1, 2, 3])
        self.assert_stopped()

    def test_pause_stops_at_the_frame_on_screen(self):
        w = self.window
        self.shown.clear()
        w.play_button.click()
        deadline = time.monotonic() + 5
        while not self.shown and time.monotonic() < deadline:
            QTest.qWait(5)
        w.play_button.click()  # Pause
        self.assert_stopped()
        shown = list(self.shown)
        QTest.qWait(400)
        self.assertEqual(self.shown, shown)
        self.assertLess(self.row(), FRAMES - 1)
        # Play carries on from there.
        w.play_button.click()
        self.wait_until_stopped()
        self.assertEqual(self.row(), FRAMES - 1)

    def test_pause_works_when_a_frame_loads_between_press_and_release(self):
        # A click by hand is a press, a pause, and a release; frames load in between.
        w = self.window
        self.show_window()
        focused = QApplication.focusWidget()
        self.assertIsNotNone(focused)
        self.shown.clear()
        w.play_button.click()
        QTest.mousePress(w.play_button, Qt.MouseButton.LeftButton)
        self.wait_for_frames(1)
        self.assertTrue(w.is_playing)
        self.assertTrue(w.play_button.isDown())
        QTest.mouseRelease(w.play_button, Qt.MouseButton.LeftButton)
        self.assert_stopped()
        self.assertLess(self.row(), FRAMES - 1)
        # The keyboard stays where it was, so the canvas shortcuts carry on working.
        self.assertIs(QApplication.focusWidget(), focused)

    def test_playing_leaves_the_keyboard_where_it_is(self):
        # Taking the keyboard on every frame cancelled clicks held on any button.
        w = self.window
        self.show_window()
        w.next_frame_button.setFocus()
        self.assertIs(QApplication.focusWidget(), w.next_frame_button)
        self.shown.clear()
        w.play_button.click()
        QTest.mousePress(w.next_frame_button, Qt.MouseButton.LeftButton)
        self.wait_for_frames(1)
        self.assertIs(QApplication.focusWidget(), w.next_frame_button)
        self.assertTrue(w.next_frame_button.isDown())
        w.stop_playback()
        QTest.mouseRelease(w.next_frame_button, Qt.MouseButton.LeftButton)
        # Going to a frame by hand still hands the keyboard to the canvas.
        self.assertIs(QApplication.focusWidget(), w.canvas)

    def test_instance_list_waits_for_the_end_and_video_wide_masks_are_read_once(self):
        w = self.window
        listed, read = [], []
        real_tags, real_read = w._get_instance_to_aruco_map, w.annotation_manager.load_video_annotations
        # The tag lookup is the first step of rebuilding the instance list.
        w._get_instance_to_aruco_map = lambda: listed.append(w.is_playing) or real_tags()
        w.annotation_manager.load_video_annotations = lambda *a: read.append(w.is_playing) or real_read(*a)
        self.go_to(1)
        self.go_to(0)
        self.assertEqual(read, [False, False])  # ordinary navigation reads them every time
        listed.clear()
        read.clear()
        w.play_button.click()
        self.assertFalse(w.instance_list.isEnabled())
        self.wait_until_stopped()
        self.assertEqual(self.shown[-3:], [1, 2, 3])
        self.assertEqual(read, [True])           # once for the three frames played
        self.assertEqual(listed, [False])        # the list was rebuilt once, after stopping
        self.assert_stopped()

    def test_edits_while_playing_drop_the_kept_video_wide_masks(self):
        w = self.window
        read = []
        real_read = w.annotation_manager.load_video_annotations
        w.annotation_manager.load_video_annotations = lambda *a: read.append(1) or real_read(*a)
        w.start_playback()
        w._play_next_frame()
        self.assertEqual(len(read), 1)
        w.current_frame_modified = True  # an edit made while playing is saved on the way to the next frame
        with patch.object(w, '_save_annotation_sources'):
            w._play_next_frame()
        self.assertEqual(len(read), 2)
        w.stop_playback()
        self.assert_stopped()

    def test_a_frame_that_cannot_be_shown_stops_playback(self):
        w = self.window
        attempts = []
        w.load_frame = attempts.append  # shows nothing
        w.play_button.click()
        self.wait_until_stopped()
        self.assertEqual(len(attempts), 1)
        self.assertEqual(self.row(), 0)
        self.assert_stopped()

    def test_closing_while_playing_stops_playback(self):
        w = self.window
        w.play_button.click()
        self.assertGreater(gc.get_freeze_count(), 0)
        with patch.object(QMessageBox, 'question', return_value=QMessageBox.StandardButton.Discard):
            w.close()
        self.assertFalse(w.is_playing)
        self.assertEqual(gc.get_freeze_count(), 0)


if __name__ == '__main__':
    unittest.main()
