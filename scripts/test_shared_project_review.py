"""Shared-version provenance, pinned comparisons, and nonblocking UI checks."""

import os
from threading import Event
import time
import unittest
from unittest.mock import patch

from PyQt6.QtCore import QCoreApplication, QEvent
from PyQt6.QtTest import QTest
from PyQt6.QtWidgets import QDialog, QMessageBox

from core.assigned_work import latest_snapshot
from core.project_sync import _digest, _inventory, _write_json, sync_project
from core.project_update import PublishedProject
from core.shared_project_review import check_shared_version, compare_saved_annotations
from gui.assigned_work_dialog import AssignmentManagerDialog
from gui.publication_review_dialog import PublicationReviewDialog
from gui.shared_version import SharedVersionHeader
from scripts import test_assigned_work as fixtures
from scripts import test_assigned_work_gui as gui_fixtures


class SharedProjectReviewTests(unittest.TestCase):
    setUp = fixtures.AssignedWorkTests.setUp
    view = fixtures.AssignedWorkTests.view
    change = fixtures.AssignedWorkTests.change
    download = fixtures.AssignedWorkTests.download

    def test_freshness_reads_pointers_without_opening_manifests_or_masks(self):
        snapshot, _ = latest_snapshot(self.view())
        with patch.object(PublishedProject, 'snapshot', side_effect=AssertionError('No manifest scan')):
            self.assertFalse(check_shared_version(self.source, self.configuration, snapshot)['changed'])
        self.change(self.source)
        sync_project(self.source, self.configuration, self.coordinator)
        self.assertTrue(check_shared_version(self.source, self.configuration, snapshot)['changed'])

    def test_comparison_is_read_only_and_ignores_timestamp_only_changes(self):
        snapshot, _ = latest_snapshot(self.view())
        path = self.source / 'annotations/json/alpha/frame_000000.json'
        stamp = path.stat().st_mtime_ns + 10000000
        os.utime(path, ns=(stamp, stamp))
        before = _inventory(self.source, True, with_identity=True)
        with patch('core.project_sync._write_json', side_effect=AssertionError('No writes')):
            with patch('core.shared_project_review._digest', wraps=_digest) as digest:
                result = compare_saved_annotations(self.source, snapshot)
        self.assertEqual(result['rows'], [])
        self.assertGreater(result['unchanged'], 0)
        self.assertEqual(before, _inventory(self.source, True, with_identity=True))
        self.assertTrue(all('/frames/' not in str(call.args[0]) and '/input_data/' not in str(call.args[0])
                            for call in digest.call_args_list))

    def test_comparison_groups_file_differences_and_stays_on_displayed_version(self):
        snapshot, _ = latest_snapshot(self.view())
        self.change(self.source)
        sync_project(self.source, self.configuration, self.coordinator)
        newest, _ = latest_snapshot(self.view())
        result = compare_saved_annotations(self.source, snapshot)
        self.assertEqual(result['baseline']['revision'], snapshot['reference'][1])
        self.assertTrue(any(row['video'] == 'alpha' and row['action'] == 'Different' for row in result['rows']))
        self.assertEqual(compare_saved_annotations(self.source, newest)['rows'], [])
        _write_json(self.source / 'annotations/new_settings.json', {'new': True})
        (self.source / 'annotations/json/beta/frame_000000.json').unlink()
        (self.source / 'frames/alpha/frame_000000.jpg').write_bytes(b'not compared')
        rows = {row['path']: row for row in compare_saved_annotations(self.source, newest)['rows']}
        self.assertEqual(rows['annotations/new_settings.json']['action'], 'Only local')
        self.assertEqual(rows['annotations/json/beta/frame_000000.json']['action'], 'Only shared')
        self.assertNotIn('frames/alpha/frame_000000.jpg', rows)

    def test_partial_copy_does_not_label_undownloaded_annotations_deleted(self):
        snapshot, _ = latest_snapshot(self.view())
        result = compare_saved_annotations(self.download(), snapshot)
        self.assertTrue(any(row['video'] == 'beta' and row['action'] == 'Only shared' for row in result['rows']))
        self.assertFalse(any(row['action'] == 'Removed' for row in result['rows']))

    def test_wrong_identity_cancellation_and_changed_sources_fail_safely(self):
        snapshot, _ = latest_snapshot(self.view())
        with self.assertRaises(InterruptedError):
            compare_saved_annotations(self.source, snapshot, cancelled=lambda: True)
        with self.assertRaises(InterruptedError):
            check_shared_version(self.source, self.configuration, snapshot, cancelled=lambda: True)
        def change_during_scan(message):
            if message.startswith('Comparing saved'):
                _write_json(self.source / 'annotations/new_settings.json', {'changed': message})
        with self.assertRaisesRegex(ValueError, 'changed during comparison'):
            compare_saved_annotations(self.source, snapshot, progress=change_during_scan)
        snapshot['manifest_path'].write_bytes(snapshot['manifest_path'].read_bytes() + b'\n')
        with self.assertRaisesRegex(ValueError, 'shared version changed'):
            compare_saved_annotations(self.source, snapshot)
        _write_json(self.source / 'annotations/project_sync.json', {'format': 1, 'project_id': self.alex['id']})
        with self.assertRaisesRegex(ValueError, 'another project'):
            compare_saved_annotations(self.source, snapshot)


class SharedProjectReviewGuiTests(unittest.TestCase):
    setUpClass = gui_fixtures.AssignedWorkGuiTests.__dict__['setUpClass']
    setUp = gui_fixtures.AssignedWorkGuiTests.setUp
    tearDown = gui_fixtures.AssignedWorkGuiTests.tearDown
    view = gui_fixtures.AssignedWorkGuiTests.view
    make_window = gui_fixtures.AssignedWorkGuiTests.make_window
    change = fixtures.AssignedWorkTests.change

    def settle(self, header):
        end = time.monotonic() + 5
        while header.worker:
            QTest.qWait(10)
            self.assertLess(time.monotonic(), end, 'Version check did not finish')

    def test_source_header_and_checks_preserve_assignment_edits_and_displayed_version(self):
        dialog = AssignmentManagerDialog(self.source, self.configuration, self.coordinator)
        try:
            header = dialog.shared_version
            reference = dialog.snapshot['reference']
            self.assertIn('Shared version', header.title.text())
            self.assertIn('August', header.publication.text())
            self.assertIn('Published', header.publication.text())
            dialog.controls['alpha'][1].setText('0-1')
            dialog.check_visible()
            header.check_version()
            self.settle(header)
            self.assertIn('No newer shared', header.freshness.text())
            self.assertIn('Last checked', header.freshness.text())
            last_checked = header.last_checked
            with patch('gui.shared_version.check_shared_version', side_effect=OSError('Drive unavailable')):
                header.check_version()
                self.settle(header)
            self.assertIn('Unable to check', header.freshness.text())
            self.assertEqual(header.last_checked, last_checked)
            self.assertIn('Drive unavailable', header.freshness.toolTip())
            self.change(self.source)
            sync_project(self.source, self.configuration, self.coordinator)
            header.check_version()
            self.settle(header)
            self.assertIn('publications changed', header.freshness.text())
            self.assertFalse(header.reload.isHidden())
            self.assertEqual(dialog.snapshot['reference'], reference)
            self.assertEqual(dialog.choices()['alpha']['frames'], [0, 1])
            self.assertEqual(dialog.checked_videos(), ['alpha', 'beta'])
            with patch.object(QMessageBox, 'question', return_value=QMessageBox.StandardButton.No):
                header.reload.click()
            self.assertEqual(dialog.controls['alpha'][1].text(), '0-1')
            with patch('gui.shared_version.QDesktopServices.openUrl', return_value=True) as opened:
                header.location.click()
            self.assertEqual(opened.call_args.args[0].toLocalFile(), str(dialog.snapshot['manifest_path'].parent))
            dialog.resize(1250, 900)
            dialog.show()
            self.app.processEvents()
            dialog.grab().save('/tmp/bumblebox-shared-version-header.png')
            dialog.resize(920, 780)
            self.app.processEvents()
            dialog.grab().save('/tmp/bumblebox-shared-version-header-small.png')
        finally:
            dialog.shared_version.stop()
            dialog.close()

    def test_comparison_button_uses_displayed_version_and_neutral_review_labels(self):
        dialog = AssignmentManagerDialog(self.source, self.configuration, self.coordinator)
        try:
            reference = dialog.snapshot['reference']
            self.change(self.source)
            sync_project(self.source, self.configuration, self.coordinator)
            reviews = []
            def create_review(*args, **kwargs):
                reviews.append(PublicationReviewDialog(*args, **kwargs))
                return reviews[-1]
            with patch('gui.publication_review_dialog.PublicationReviewDialog', side_effect=create_review), \
                    patch.object(PublicationReviewDialog, 'exec', return_value=QDialog.DialogCode.Rejected):
                dialog.shared_version.compare.click()
            self.warnings.assert_not_called()
            review = reviews[0]
            self.assertEqual(review.review['baseline']['revision'], reference[1])
            self.assertEqual(review.action_names, ('Only local', 'Different', 'Only shared'))
            self.assertEqual(review.table.headerItem().text(3), 'Shared')
            self.assertEqual(review.table.headerItem().text(4), 'Local')
            self.assertIn('Different', review.actions.itemText(2))
            self.assertGreater(len(review.rows), 0)
            review.show()
            self.app.processEvents()
            review.grab().save('/tmp/bumblebox-local-shared-comparison.png')
            review.close()
            window = self.make_window(self.source)
            self.assertIn('Local working copy', window.annotation_work.source.text())
            self.assertEqual(window.annotation_work.source.toolTip(), str(self.source))
        finally:
            dialog.shared_version.stop()
            dialog.close()

    def test_disposed_header_cancels_worker_without_destroying_running_thread(self):
        snapshot, _ = latest_snapshot(self.view())
        header = SharedVersionHeader(self.source, self.configuration)
        header.set_snapshot(snapshot)
        started, release = Event(), Event()
        def delayed(*args, **kwargs):
            started.set()
            release.wait(5)
            self.assertTrue(kwargs['cancelled']())
            raise InterruptedError('Cancelled')
        with patch('gui.shared_version.check_shared_version', side_effect=delayed):
            try:
                header.check_version()
                worker = header.worker
                self.assertTrue(started.wait(2))
                header.deleteLater()
                QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
            finally:
                release.set()
                self.assertTrue(worker.wait(2000))
                self.app.processEvents()

    def test_result_for_old_version_cannot_label_refreshed_header_current(self):
        snapshot, _ = latest_snapshot(self.view())
        self.change(self.source)
        sync_project(self.source, self.configuration, self.coordinator)
        newest, _ = latest_snapshot(self.view())
        header = SharedVersionHeader(self.source, self.configuration)
        header.set_snapshot(snapshot)
        started, release = Event(), Event()
        def delayed(*args, **kwargs):
            started.set()
            release.wait(5)
            return {'changed': True, 'checked_at': '2026-10-27T12:00:00+00:00'}
        with patch('gui.shared_version.check_shared_version', side_effect=delayed):
            try:
                header.check_version()
                self.assertTrue(started.wait(2))
                header.set_snapshot(newest)
            finally:
                release.set()
                self.settle(header)
                header.stop()
                header.close()
        self.assertIsNone(header.last_checked)
        self.assertEqual(header.freshness.text(), 'Version freshness not checked')


if __name__ == '__main__':
    unittest.main()
