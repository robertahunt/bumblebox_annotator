"""Assignment filters, sidebar completion and navigation in disposable projects."""

import time
from copy import deepcopy
from threading import Event
import unittest
from unittest.mock import patch

from PyQt6.QtCore import Qt
from PyQt6.QtTest import QTest
from PyQt6.QtWidgets import QMessageBox, QDialog

from core.annotation_work import local_index, mark_finished, summarize
from core.assigned_work import catalog, latest_snapshot, read_assignments, save_assignments
from gui.assigned_work_dialog import AssignmentManagerDialog, BulkAssignmentDialog
from gui.annotation_work import WorkFilters
from gui.project_sync_dialog import SyncProgressDialog
from scripts import test_assigned_work_gui as fixtures


class AnnotationWorkGuiTests(unittest.TestCase):
    setUpClass = fixtures.AssignedWorkGuiTests.__dict__['setUpClass']
    setUp = fixtures.AssignedWorkGuiTests.setUp
    tearDown = fixtures.AssignedWorkGuiTests.tearDown
    make_window = fixtures.AssignedWorkGuiTests.make_window
    download = fixtures.AssignedWorkGuiTests.download
    view = fixtures.AssignedWorkGuiTests.view

    def settle(self, browser):
        end = time.monotonic() + 5
        while browser.timer.isActive() or browser.worker or browser.pending:
            self.app.processEvents()
            QTest.qWait(10)
            self.assertLess(time.monotonic(), end, 'Summary worker failed to settle')

    def test_initial_scan_shows_loading_and_keeps_pending_rows_visible(self):
        started, release = Event(), Event()
        def slow_index(*args, **kwargs):
            started.set()
            if not release.wait(5):
                raise TimeoutError('Test did not release the summary scan')
            return local_index(*args, **kwargs)
        with patch('gui.annotation_work.local_index', side_effect=slow_index):
            try:
                window = self.make_window(self.source)
                browser = window.annotation_work
                browser.filters.task.setCurrentIndex(browser.filters.task.findData('hive'))
                browser.filters.show.setCurrentIndex(browser.filters.show.findData('finished'))
                for i in range(window.video_list.count()):
                    item = window.video_list.item(i)
                    self.assertIn('Hive: Loading...', item.text())
                    self.assertNotIn('Unknown', item.text())
                    self.assertFalse(item.isHidden())
                self.assertIn('Loading...', window.frame_list.item(0).text())
                self.assertFalse(browser.finish.isEnabled())
                deadline = time.monotonic() + 5
                while not started.is_set():
                    QTest.qWait(10)
                    self.assertLess(time.monotonic(), deadline)
                self.assertIn('Checking local saved masks...', browser.note.text())
                window.resize(1500, 1050)
                window.show()
                self.app.processEvents()
                window.video_dock.grab().save('/tmp/bumblebox-mask-summary-loading.png')
            finally:
                release.set()
            self.settle(browser)
        self.assertIn('Hive: Shared mask | Not checked', window.video_list.item(0).text())
        self.assertTrue(browser.finish.isEnabled())
        self.assertTrue(window.video_list.item(1).isHidden())
        self.assertNotIn('Loading...', window.frame_list.item(0).text())

    def test_failed_scan_stays_unknown_until_successful_retry(self):
        with patch('gui.annotation_work.local_index', side_effect=OSError('Unreadable annotation metadata')):
            window = self.make_window(self.source)
            browser = window.annotation_work
            browser.filters.task.setCurrentIndex(browser.filters.task.findData('hive'))
            self.settle(browser)
        self.assertIn('Hive: Unknown | Unknown', window.video_list.item(0).text())
        self.assertIn('Unreadable annotation metadata', browser.note.text())
        self.assertIn('Unreadable annotation metadata', window.video_list.item(0).toolTip())
        self.assertFalse(browser.finish.isEnabled())
        browser.filters.show.setCurrentIndex(browser.filters.show.findData('not_checked'))
        self.assertIn('Unreadable annotation metadata', browser.note.text())
        browser.schedule()
        self.settle(browser)
        self.assertIn('Hive: Shared mask | Not checked', window.video_list.item(0).text())
        self.assertTrue(browser.finish.isEnabled())
        self.assertIsNone(browser.error)

    def test_assignment_filters_are_category_specific_and_clear_hidden_checks(self):
        dialog = AssignmentManagerDialog(self.source, self.configuration, self.coordinator)
        filters = dialog.work_filters
        self.assertTrue(dialog.table.alternatingRowColors())
        self.assertEqual([filters.show.itemText(i) for i in range(filters.show.count())],
                         ['All Work', 'Not Started', 'Not Checked', 'Finished'])
        filters.task.setCurrentIndex(filters.task.findData('hive'))
        self.assertEqual(dialog.table.item(0, 6).text(), 'Shared mask')
        self.assertEqual(dialog.table.item(0, 7).text(), 'Not checked')
        dialog.check_visible()
        self.assertEqual(len(dialog.checked_videos()), 2)
        filters.show.setCurrentIndex(filters.show.findData('not_started'))
        self.assertTrue(all(dialog.table.isRowHidden(i) for i in range(2)))
        self.assertEqual(dialog.checked_videos(), [])
        filters.show.setCurrentIndex(filters.show.findData('not_checked'))
        self.assertTrue(all(not dialog.table.isRowHidden(i) for i in range(2)))
        filters.show.setCurrentIndex(0)
        filters.presence.setCurrentIndex(filters.presence.findData('none'))
        self.assertEqual(dialog.checked_videos(), [])
        self.assertTrue(all(dialog.table.isRowHidden(i) for i in range(2)))
        filters.task.setCurrentIndex(filters.task.findData('pollen'))
        self.assertTrue(all(not dialog.table.isRowHidden(i) for i in range(2)))
        filters.show.setCurrentIndex(filters.show.findData('not_started'))
        self.assertTrue(all(not dialog.table.isRowHidden(i) for i in range(2)))
        filters.show.setCurrentIndex(filters.show.findData('not_checked'))
        self.assertTrue(all(dialog.table.isRowHidden(i) for i in range(2)))
        filters.show.setCurrentIndex(filters.show.findData('finished'))
        self.assertTrue(all(dialog.table.isRowHidden(i) for i in range(2)))
        filters.show.setCurrentIndex(0)
        filters.task.setCurrentIndex(filters.task.findData('hive'))
        filters.presence.setCurrentIndex(0)
        dialog.resize(1250, 760)
        dialog.show()
        self.app.processEvents()
        dialog.grab().save('/tmp/bumblebox-work-assignment-filters.png')
        self.assertFalse(filters.geometry().intersects(dialog.table.geometry()))
        filters.more.setChecked(True)
        dialog.resize(920, 650)
        self.app.processEvents()
        dialog.grab().save('/tmp/bumblebox-work-assignment-filters-small.png')
        self.assertFalse(filters.geometry().intersects(dialog.table.geometry()))
        dialog.close()

    def test_scanning_status_is_not_hidden_by_work_filters_but_still_respects_split(self):
        filters = WorkFilters()
        try:
            filters.show.setCurrentIndex(filters.show.findData('finished'))
            filters.presence.setCurrentIndex(filters.presence.findData('has'))
            filters.split.setCurrentIndex(filters.split.findData('val'))
            summary = {'state': 'loading', 'presence': 'loading'}
            self.assertTrue(filters.accepts(summary, 'val'))
            self.assertFalse(filters.accepts(summary, 'train'))
        finally:
            filters.close()

    def test_bulk_assignment_defaults_to_filtered_task(self):
        dialog = BulkAssignmentDialog(['alpha'], {'alpha': {'frames': [0, 1]}},
                                      {self.alex['id']: self.alex}, task='hive')
        dialog.contributor.setCurrentIndex(0)
        with patch('gui.assigned_work_dialog.TransferConfirmationDialog.exec', return_value=QDialog.DialogCode.Accepted):
            dialog.review()
        self.assertEqual(dialog.choices['alpha']['task'], 'hive')

    def test_old_publication_scan_is_explicit_and_preserves_assignment_edits(self):
        snapshot, _ = latest_snapshot(self.view())
        snapshot['manifest'].pop('annotation_work')
        available = catalog(snapshot, include_work=True)
        initial = read_assignments(self.view()), available, {}, snapshot
        with patch('core.annotation_work._unit', side_effect=AssertionError('Opening the table must not scan masks')):
            dialog = AssignmentManagerDialog(self.source, self.configuration, self.coordinator, initial=initial)
        self.assertIn('0/2', dialog.mask_status.text())
        dialog.work_filters.task.setCurrentIndex(dialog.work_filters.task.findData('hive'))
        self.assertEqual(dialog.table.item(0, 6).text(), 'Not scanned')
        self.assertEqual(dialog.table.item(0, 7).text(), 'Not scanned')
        self.assertIn('use Scan Mask Status', dialog.mask_status.text())
        self.assertIn('Scan Mask Status', dialog.table.item(0, 6).toolTip())
        frames = dialog.controls['alpha'][1]
        frames.setText('0-1')
        dialog.check_visible()
        original_run = SyncProgressDialog.run
        def inspect_scanning(progress):
            self.assertEqual(dialog.table.item(0, 6).text(), 'Scanning...')
            self.assertEqual(dialog.table.item(0, 7).text(), 'Scanning...')
            self.assertFalse(dialog.scan_masks.isEnabled())
            self.assertIn('Scanning published', dialog.mask_status.text())
            dialog.resize(1250, 760)
            dialog.show()
            self.app.processEvents()
            dialog.grab().save('/tmp/bumblebox-assignment-mask-scanning.png')
            return original_run(progress)
        with patch.object(SyncProgressDialog, 'run', autospec=True, side_effect=inspect_scanning):
            dialog.scan_mask_status()
        self.assertEqual(frames.text(), '0-1')
        self.assertEqual(dialog.checked_videos(), ['alpha', 'beta'])
        self.assertIn('2/2', dialog.mask_status.text())
        self.assertEqual(dialog.table.item(0, 6).text(), 'Shared mask')
        self.assertFalse(dialog.scan_masks.isEnabled())
        self.assertTrue(dialog.save_table.isEnabled())
        dialog.close()

    def test_cancelled_assignment_scan_restores_not_scanned_and_allows_retry(self):
        snapshot, _ = latest_snapshot(self.view())
        snapshot['manifest'].pop('annotation_work')
        available = catalog(snapshot, include_work=True)
        dialog = AssignmentManagerDialog(self.source, self.configuration, self.coordinator,
            initial=(read_assignments(self.view()), available, {}, snapshot))
        try:
            before = deepcopy(available)
            for exception in (InterruptedError('Cancelled'), OSError('Drive unavailable')):
                with self.subTest(error=str(exception)), patch.object(SyncProgressDialog, 'run', side_effect=exception):
                    dialog.scan_mask_status()
                self.assertEqual(dialog.available, before)
                self.assertFalse(dialog.scanning_masks)
                self.assertTrue(dialog.scan_masks.isEnabled())
                self.assertEqual(dialog.table.item(0, 6).text(), 'Not scanned')
                self.assertIn(str(exception), dialog.mask_status.text())
            dialog.scan_mask_status()
            self.assertEqual(dialog.table.item(0, 6).text(), 'Masks on 3/3 frames')
            self.assertNotIn('not completed', dialog.mask_status.text())
        finally:
            dialog.close()

    def test_unreadable_assignment_mask_shows_reason_and_can_be_rescanned(self):
        snapshot, _ = latest_snapshot(self.view())
        available = deepcopy(catalog(snapshot, include_work=True))
        hive = available['alpha']['annotation_work']['categories']['hive']
        hive['units']['video'] = {'count': None, 'signature': None, 'status': 'unknown',
                                  'error': 'Missing hive mask raster'}
        dialog = AssignmentManagerDialog(self.source, self.configuration, self.coordinator,
            initial=(read_assignments(self.view()), available, {}, snapshot))
        try:
            dialog.work_filters.task.setCurrentIndex(dialog.work_filters.task.findData('hive'))
            self.assertEqual(dialog.table.item(0, 6).text(), 'Unknown')
            self.assertIn('Missing hive mask raster', dialog.table.item(0, 6).toolTip())
            self.assertIn('1/2', dialog.mask_status.text())
            self.assertIn('1 unavailable', dialog.mask_status.text())
            self.assertTrue(dialog.scan_masks.isEnabled())
            dialog.scan_mask_status()
            self.assertEqual(dialog.table.item(0, 6).text(), 'Shared mask')
            self.assertFalse(dialog.scan_masks.isEnabled())
        finally:
            dialog.close()

    def test_assigned_task_selected_on_open_without_visibility_or_zoom_changes(self):
        data = read_assignments(self.view())
        self.choices['alpha']['task'] = 'hive'
        save_assignments(self.source, self.configuration, self.coordinator, self.choices, data['revision'])
        window = self.make_window(self.download())
        browser = window.annotation_work
        self.settle(browser)
        self.assertEqual(browser.filters.task.currentData(), 'hive')
        self.assertTrue(window.video_list.alternatingRowColors())
        self.assertIn('Shared mask', window.video_list.item(0).text())
        self.assertEqual(browser.index['videos']['alpha']['frames'], [0])
        transform = window.canvas.transform()
        annotations = window.canvas.get_annotations()
        browser.filters.show.setCurrentIndex(browser.filters.show.findData('finished'))
        self.assertFalse(window.video_list.item(0).isHidden())
        self.assertIn('kept visible', browser.note.text())
        self.assertEqual(transform, window.canvas.transform())
        self.assertEqual(len(annotations), len(window.canvas.get_annotations()))
        window.resize(1450, 940)
        window.show()
        self.app.processEvents()
        window.video_dock.grab().save('/tmp/bumblebox-work-sidebar.png')
        self.assertLess(browser.filters.geometry().bottom(), window.video_list.geometry().top())

    def test_completion_saves_and_reopens_without_submission(self):
        window = self.make_window(self.download())
        browser = window.annotation_work
        self.settle(browser)
        browser.filters.task.setCurrentIndex(browser.filters.task.findData('hive'))
        with patch.object(QMessageBox, 'question', return_value=QMessageBox.StandardButton.Yes):
            browser.finish.click()
        self.settle(browser)
        self.warnings.assert_not_called()
        self.assertEqual(browser.finish.text(), 'Reopen Work')
        self.assertEqual(summarize(local_index(window.project_path)['videos']['alpha'], 'hive')['state'], 'finished')
        with patch.object(QMessageBox, 'question', return_value=QMessageBox.StandardButton.Yes):
            browser.finish.click()
        self.settle(browser)
        self.assertEqual(browser.finish.text(), 'Mark Finished')

    def test_sampled_next_unfinished_uses_source_indices(self):
        data = read_assignments(self.view())
        self.choices['alpha'].update(task='bee', frames=[0, 2])
        save_assignments(self.source, self.configuration, self.coordinator, self.choices, data['revision'])
        local = self.download()
        mark_finished(local, 'alpha', 0, 'bee', self.alex)
        window = self.make_window(local)
        browser = window.annotation_work
        self.settle(browser)
        browser.next.click()
        self.assertEqual(window._get_frame_idx_in_video(window.current_frame_idx), 2)
        self.assertEqual(browser.finish.text(), 'Mark Finished')
        self.assertIn('Finished', window.frame_list.item(0).text())


if __name__ == '__main__':
    unittest.main()
