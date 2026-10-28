"""Experiment filters and local modification ordering on disposable projects."""

import os
from pathlib import Path
import tempfile
import time
from threading import Event
import unittest
from unittest.mock import patch

from PyQt6.QtCore import QCoreApplication, QEvent, Qt
from PyQt6.QtGui import QColor, QPalette
from PyQt6.QtTest import QTest

from core.assigned_work import catalog, latest_snapshot
from core.project_sync import _write_json, sync_project
from core.video_browsing import experiment_assignments, local_modification_times
from gui.assigned_work_dialog import AssignmentManagerDialog
from gui.video_browsing import VideoBrowserControls
from scripts import test_assigned_work_gui as fixtures


def set_annotation_dates(project, video, stamp):
    for kind in ('json', 'png', 'bbox', 'work'):
        folder = project / 'annotations' / kind / video
        if folder.exists():
            for path in folder.iterdir():
                os.utime(path, ns=(stamp, stamp))
            os.utime(folder, ns=(stamp, stamp))


class VideoBrowsingTests(unittest.TestCase):
    def test_dates_ignore_frame_images_original_videos_and_exports(self):
        with tempfile.TemporaryDirectory() as temporary:
            project = Path(temporary)
            mask = project / 'annotations/json/alpha/frame_000000.json'
            mask.parent.mkdir(parents=True)
            mask.write_bytes(b'not JSON: no payload parsing should occur')
            set_annotation_dates(project, 'alpha', 100_000_000_000)
            for name in ('frames/alpha/frame_000000.jpg', 'input_data/train/alpha.mp4',
                         'annotations/coco/alpha.json', 'annotations/imaging_setups.json'):
                path = project / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b'newer, irrelevant')
            result = local_modification_times(project, ['alpha', 'beta'])
            self.assertEqual(result['modified'], {'alpha': 100_000_000_000, 'beta': None})
            self.assertFalse(result['errors'])
            mask.unlink()
            os.utime(mask.parent, ns=(200_000_000_000, 200_000_000_000))
            self.assertEqual(local_modification_times(project, ['alpha'])['modified']['alpha'], 200_000_000_000)

    def test_cancellation_invalid_ids_and_unreadable_sources(self):
        with tempfile.TemporaryDirectory() as temporary:
            project = Path(temporary)
            with self.assertRaises(InterruptedError):
                local_modification_times(project, ['alpha'], lambda: True)
            with self.assertRaises(ValueError):
                local_modification_times(project, ['../elsewhere'])
            folder = project / 'annotations/json'
            folder.mkdir(parents=True)
            (folder / 'alpha').symlink_to(project, target_is_directory=True)
            result = local_modification_times(project, ['alpha'])
            self.assertIsNone(result['modified']['alpha'])
            self.assertIn('alpha', result['errors'])

    def test_experiment_names_use_existing_profiles_only(self):
        config = {'setups': {'2021 bumblebox': {}, '2024 rig build': {}},
                  'video_assignments': {'alpha': '2021 bumblebox', 'beta': 'missing'}}
        self.assertEqual(experiment_assignments(config), {'alpha': '2021 bumblebox'})
        with self.assertRaises(ValueError):
            experiment_assignments({'setups': []})


class VideoBrowsingGuiTests(unittest.TestCase):
    setUpClass = fixtures.AssignedWorkGuiTests.__dict__['setUpClass']
    setUp = fixtures.AssignedWorkGuiTests.setUp
    tearDown = fixtures.AssignedWorkGuiTests.tearDown
    make_window = fixtures.AssignedWorkGuiTests.make_window
    view = fixtures.AssignedWorkGuiTests.view
    download = fixtures.AssignedWorkGuiTests.download

    def setups(self):
        _write_json(self.source / 'annotations/imaging_setups.json', {
            'setups': {'2021 bumblebox': {}, '2024 rig build': {}},
            'video_assignments': {'alpha': '2021 bumblebox', 'beta': '2024 rig build'}})

    def settle(self, controls):
        deadline = time.monotonic() + 5
        while controls.timer.isActive() or controls.worker or controls.pending:
            QTest.qWait(10)
            self.assertLess(time.monotonic(), deadline)

    def visible(self, window):
        return [window.video_list.item(i).data(Qt.ItemDataRole.UserRole)
                for i in range(window.video_list.count()) if not window.video_list.item(i).isHidden()]

    def test_sidebar_experiment_filter_is_exact_and_does_not_change_calibration_or_canvas(self):
        self.setups()
        window = self.make_window(self.source)
        controls = window.video_browser
        before = (self.source / 'annotations/imaging_setups.json').read_bytes()
        transform, current, frame = window.canvas.transform(), window.current_video_id, window.current_frame_idx
        window.video_list.selectAll()
        controls.experiment.setCurrentIndex(controls.experiment.findData('2024 rig build'))
        self.assertEqual(self.visible(window), ['beta'])
        self.assertEqual(window._selected_video_ids(), ['beta'])
        self.assertEqual((window.current_video_id, window.current_frame_idx), (current, frame))
        self.assertEqual(window.canvas.transform(), transform)
        self.assertEqual((self.source / 'annotations/imaging_setups.json').read_bytes(), before)
        controls.experiment.setCurrentIndex(controls.experiment.findData(''))
        self.assertEqual(self.visible(window), [])
        self.assertEqual(window._selected_video_ids(), [])
        controls.experiment.setCurrentIndex(0)
        self.assertEqual(self.visible(window), ['alpha', 'beta'])
        window.resize(1500, 1050)
        window.show()
        self.app.processEvents()
        window.video_dock.grab().save('/tmp/bumblebox-experiment-video-browser.png')

    def test_recent_sort_crosses_splits_preserves_selection_and_updates_after_save(self):
        self.setups()
        destination = self.source / 'input_data/val/beta.mp4'
        destination.parent.mkdir(parents=True)
        (self.source / 'input_data/train/beta.mp4').rename(destination)
        window = self.make_window(self.source)
        self.settle(window.annotation_work)
        set_annotation_dates(self.source, 'alpha', 100_000_000_000)
        set_annotation_dates(self.source, 'beta', 200_000_000_000)
        controls = window.video_browser
        current, transform = window.current_video_id, window.canvas.transform()
        selected = window.video_list.item(0)
        selected.setSelected(True)
        controls.sort.setCurrentIndex(controls.sort.findData('recent'))
        self.settle(controls)
        self.assertEqual(self.visible(window), ['beta', 'alpha'])
        self.assertTrue(selected.isSelected())
        self.assertEqual(window.current_video_id, current)
        self.assertEqual(window.canvas.transform(), transform)
        set_annotation_dates(self.source, 'alpha', 300_000_000_000)
        window.on_save_completed(0)
        self.settle(controls)
        self.assertEqual(self.visible(window), ['alpha', 'beta'])
        controls.sort.setCurrentIndex(0)
        self.assertEqual(self.visible(window), ['alpha', 'beta'])

    def test_sidebar_stripes_follow_visible_rows_in_light_and_dark_palettes(self):
        window = self.make_window(self.source)
        self.settle(window.annotation_work)
        window.resize(1500, 1050)
        window.show()
        videos = window.video_list
        videos.clearSelection()
        videos.setCurrentRow(-1)
        first, second = videos.item(0), videos.item(1)
        for theme, base, alternate, text, stripe in (
            ('light', '#ffffff', '#eeeeee', '#202020', '#e7e7e7'),
            ('dark', '#242424', '#343434', '#eeeeee', '#444444'),
        ):
            with self.subTest(theme=theme):
                palette = videos.palette()
                for role, color in ((QPalette.ColorRole.Base, base),
                                    (QPalette.ColorRole.AlternateBase, alternate),
                                    (QPalette.ColorRole.Text, text)):
                    palette.setColor(role, QColor(color))
                videos.setPalette(palette)
                first.setHidden(False)
                self.app.processEvents()
                def background(item):
                    videos.doItemsLayout()
                    image = videos.viewport().grab().toImage()
                    ratio = image.devicePixelRatio()
                    return image.pixelColor(int((videos.viewport().width() - 8) * ratio),
                                            int(videos.visualItemRect(item).center().y() * ratio)).name()
                self.assertEqual(background(first), base)
                self.assertEqual(background(second), stripe)
                window.video_dock.grab().save(f'/tmp/bumblebox-video-stripes-{theme}.png')
                first.setHidden(True)
                self.assertEqual(background(second), base)
                second.setSelected(True)
                self.assertNotIn(background(second), (base, stripe))
                second.setSelected(False)

    def test_sidebar_work_filters_separate_untouched_and_unchecked_categories(self):
        window = self.make_window(self.source)
        self.settle(window.annotation_work)
        filters = window.annotation_work.filters
        filters.task.setCurrentIndex(filters.task.findData('hive'))
        filters.show.setCurrentIndex(filters.show.findData('not_started'))
        self.assertEqual(self.visible(window), [window.current_video_id])
        self.assertIn('kept visible', window.annotation_work.note.text())
        filters.show.setCurrentIndex(filters.show.findData('not_checked'))
        self.assertEqual(self.visible(window), ['alpha', 'beta'])
        filters.task.setCurrentIndex(filters.task.findData('pollen'))
        self.assertEqual(self.visible(window), [window.current_video_id])
        filters.show.setCurrentIndex(filters.show.findData('not_started'))
        self.assertEqual(self.visible(window), ['alpha', 'beta'])

    def test_assignment_table_sorts_and_filters_without_mixing_row_identities(self):
        self.setups()
        sync_project(self.source, self.configuration, self.coordinator)
        snapshot, _ = latest_snapshot(self.view())
        self.assertEqual(catalog(snapshot)['beta']['experiment'], '2024 rig build')
        set_annotation_dates(self.source, 'alpha', 100_000_000_000)
        set_annotation_dates(self.source, 'beta', 200_000_000_000)
        dialog = AssignmentManagerDialog(self.source, self.configuration, self.coordinator)
        try:
            controls = dialog.video_browser
            dialog.controls['alpha'][1].setText('0-1')
            dialog.check_visible()
            controls.sort.setCurrentIndex(controls.sort.findData('recent'))
            self.settle(controls)
            self.assertEqual(dialog.videos, ['beta', 'alpha'])
            self.assertEqual(dialog.checked_videos(), ['beta', 'alpha'])
            self.assertEqual(dialog.choices()['alpha']['frames'], [0, 1])
            controls.experiment.setCurrentIndex(controls.experiment.findData('2021 bumblebox'))
            self.assertEqual(dialog.checked_videos(), ['alpha'])
            self.assertTrue(dialog.table.isRowHidden(0))
            self.assertFalse(dialog.table.isRowHidden(1))
            controls.sort.setCurrentIndex(0)
            self.assertEqual(dialog.videos, ['alpha', 'beta'])
            self.assertFalse(dialog.table.isRowHidden(0))
            self.assertTrue(dialog.table.isRowHidden(1))
            dialog.work_filters.presence.setCurrentIndex(dialog.work_filters.presence.findData('none'))
            self.assertEqual(dialog.checked_videos(), [])
            dialog.work_filters.presence.setCurrentIndex(0)
            controls.experiment.setCurrentIndex(0)
            dialog.resize(1450, 800)
            dialog.show()
            self.app.processEvents()
            dialog.grab().save('/tmp/bumblebox-experiment-assignment-browser.png')
            dialog.resize(920, 700)
            self.app.processEvents()
            self.assertGreaterEqual(dialog.table.columnWidth(0), 300)
            dialog.grab().save('/tmp/bumblebox-experiment-assignment-browser-small.png')
        finally:
            controls.stop()
            dialog.close()

    def test_no_date_scan_until_requested_and_project_change_discards_dates(self):
        controls = VideoBrowserControls()
        try:
            with patch('gui.video_browsing.local_modification_times', side_effect=AssertionError('No startup scan')):
                controls.set_context(self.source, ['alpha', 'beta'], {'alpha': '2021 bumblebox'})
                QTest.qWait(450)
            controls.experiment.setCurrentIndex(controls.experiment.findData('2021 bumblebox'))
            controls.set_context(self.base / 'other-project', ['gamma'], {})
            self.assertIsNone(controls.experiment.currentData())
            self.assertEqual(controls.modified, {})
            controls.sort.setCurrentIndex(controls.sort.findData('recent'))
            self.settle(controls)
            self.assertIsNone(controls.modified['gamma'])
        finally:
            controls.stop()
            controls.close()

    def test_destroying_panel_during_date_read_does_not_destroy_running_thread(self):
        controls = VideoBrowserControls()
        started, release = Event(), Event()
        def slow_read(project, videos, cancelled):
            started.set()
            release.wait(5)
            self.assertTrue(cancelled())
            raise InterruptedError('Cancelled')
        with patch('gui.video_browsing.local_modification_times', side_effect=slow_read):
            try:
                controls.set_context(self.source, ['alpha'], {})
                controls.refresh()
                worker = controls.worker
                self.assertTrue(started.wait(2))
                controls.deleteLater()
                QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
            finally:
                release.set()
                self.assertTrue(worker.wait(2000))
                self.app.processEvents()


if __name__ == '__main__':
    unittest.main()
