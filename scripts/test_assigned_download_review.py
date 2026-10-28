"""Download review/availability filters never silently request hidden or duplicate work."""

import unittest
from unittest.mock import Mock, patch

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QDialog

from core.assigned_work import accept_submission, publish_work, read_assignments, save_assignments
from core.assigned_workspace import assigned_download_overview
from core.project_sync import sync_project
from core.project_update import PreparedUpdate
from gui.assigned_work_dialog import AssignedDownloadDialog, managed_copy_key
from gui.project_sync_dialog import save_setting
from scripts import test_assigned_work as backend
from scripts import test_assigned_work_gui as gui


class DownloadOverviewTests(unittest.TestCase):
    setUp = backend.AssignedWorkTests.setUp
    view = backend.AssignedWorkTests.view
    download = backend.AssignedWorkTests.download
    change = backend.AssignedWorkTests.change
    status = backend.AssignedWorkTests.status

    def assignments(self):
        return {v: e for v, e in read_assignments(self.view())['videos'].items()
                if e['contributor']['id'] == self.alex['id']}

    def overview(self, **kwargs):
        return assigned_download_overview(self.configuration, self.project_id, self.alex,
                                          self.assignments(), **kwargs)

    def test_new_download_missing_frames_and_changed_assignment_are_distinct(self):
        self.assertEqual(self.overview()['local'], {'alpha': 'not_downloaded'})
        local = self.download()
        self.assertEqual(self.overview(saved_path=local)['local'], {'alpha': 'downloaded'})
        (local / 'frames/alpha/frame_000000.jpg').unlink()
        self.assertEqual(self.overview(saved_path=local)['local'], {'alpha': 'missing_frames'})
        self.choices['alpha']['frames'] = [1, 2]
        save_assignments(self.source, self.configuration, self.coordinator, self.choices,
                         read_assignments(self.view())['revision'])
        self.assertEqual(self.overview(saved_path=local)['local'], {'alpha': 'assignment_changed'})

    def test_new_publication_detected_from_metadata_without_reading_annotation_payloads(self):
        local = self.download()
        self.change(self.source, actor=self.coordinator)
        sync_project(self.source, self.configuration, self.coordinator)
        before = {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in local.rglob('*') if p.is_file()}
        with patch('core.assigned_workspace.file_record', side_effect=AssertionError('No payload checksums')), \
                patch('core.assigned_workspace._inventory', side_effect=AssertionError('No whole-tree scan')), \
                patch('core.assigned_workspace.snapshot_file', side_effect=AssertionError('No remote payload reads')):
            self.assertEqual(self.overview(saved_path=local)['local'], {'alpha': 'shared_updates'})
        self.assertEqual(before, {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in before})

    def test_saved_edits_are_not_confused_with_download_availability(self):
        local = self.download()
        self.change(local)
        self.assertEqual(self.overview(saved_path=local)['local'], {'alpha': 'downloaded'})
        publish_work(local, self.configuration, self.alex, ready=['alpha'])
        result = self.overview(saved_path=local)
        self.assertEqual(result['local']['alpha'], 'downloaded')
        self.assertEqual(result['statuses']['alpha'], 'Ready for Review')

    def test_missing_saved_copy_and_multiple_copies_are_not_new_downloads(self):
        missing = self.overview(saved_path=self.base / 'missing')
        self.assertEqual(missing['local'], {'alpha': 'unknown'})
        self.assertTrue(missing['local_error'])
        one, two = self.download(name='one'), self.download(name='two')
        multiple = self.overview(known_paths=[one, two])
        self.assertEqual(multiple['local'], {'alpha': 'unknown'})
        self.assertIn('Several older', multiple['local_error'])
        self.assertFalse(multiple['local_path'])
        preferred = self.overview(saved_path=one, known_paths=[one, two])
        self.assertEqual(preferred['local_path'], str(one))
        self.assertEqual(preferred['local']['alpha'], 'downloaded')

    def test_wrong_profile_copy_and_cancellation_are_not_reported_as_downloaded(self):
        other = self.download(self.sam, ['beta'], 'sam')
        self.assertEqual(self.overview(saved_path=other)['local']['alpha'], 'unknown')
        with self.assertRaises(InterruptedError):
            self.overview(cancelled=lambda: True)
        local = self.download()
        cancelled = [False]
        def progress(message):
            if 'frame availability' in message:
                cancelled[0] = True
        with self.assertRaises(InterruptedError):
            self.overview(saved_path=local, progress=progress, cancelled=lambda: cancelled[0])


class DownloadReviewGuiTests(unittest.TestCase):
    setUpClass = gui.AssignedWorkGuiTests.__dict__['setUpClass']
    setUp = gui.AssignedWorkGuiTests.setUp
    tearDown = gui.AssignedWorkGuiTests.tearDown
    view = backend.AssignedWorkTests.view
    download = backend.AssignedWorkTests.download
    status = backend.AssignedWorkTests.status

    def dialog(self):
        dialog = AssignedDownloadDialog(self.configuration, self.alex)
        self.addCleanup(dialog.close)
        return dialog

    def remember(self, local):
        save_setting('assigned_work/locations', {
            managed_copy_key(self.configuration, self.project_id, self.alex): str(local)})

    def visible(self, dialog):
        return [v for row, v in enumerate(dialog.videos) if not dialog.table.isRowHidden(row)]

    def mixed(self):
        self.choices['beta']['contributor'] = self.alex
        save_assignments(self.source, self.configuration, self.coordinator, self.choices,
                         read_assignments(self.view())['revision'])
        local = self.download()
        self.remember(local)
        publish_work(local, self.configuration, self.alex, ready=['alpha'])
        return local

    def test_defaults_skip_submitted_and_already_downloaded_work(self):
        self.mixed()
        dialog = self.dialog()
        self.assertEqual(dialog.review_view.currentData(), 'active')
        self.assertEqual(self.visible(dialog), ['beta'])
        self.assertEqual(dialog.selected_videos(), ['beta'])
        self.assertEqual(dialog.table.item(0, 1).text(), 'Ready for Review')
        self.assertEqual(dialog.table.item(0, 2).text(), 'Downloaded')
        self.assertEqual(dialog.table.item(1, 2).text(), 'Not downloaded')
        self.assertTrue(dialog.open_copy.isEnabled())
        dialog.review_view.setCurrentIndex(dialog.review_view.findData('all'))
        self.assertEqual(self.visible(dialog), ['alpha', 'beta'])
        self.assertEqual(dialog.selected_videos(), ['beta'])
        dialog.set_checked(True)
        dialog.local_filter.setCurrentIndex(dialog.local_filter.findData('needed'))
        self.assertEqual(self.visible(dialog), ['beta'])
        self.assertEqual(dialog.selected_videos(), ['beta'])
        dialog.local_filter.setCurrentIndex(0)
        self.assertEqual(dialog.selected_videos(), ['beta'])  # Hidden selections stay cleared.
        dialog.show()
        self.app.processEvents()
        dialog.grab().save('/tmp/bumblebox-download-review.png')

    def test_ready_to_accepted_refresh_counts_and_filters(self):
        self.mixed()
        dialog = self.dialog()
        dialog.review_view.setCurrentIndex(dialog.review_view.findData('review'))
        self.assertEqual(self.visible(dialog), ['alpha'])
        _, revision = self.status()
        PreparedUpdate(self.source, self.configuration, (self.alex['id'], revision)).apply()
        accept_submission(self.source, self.configuration, self.coordinator, 'alpha', revision)
        dialog.refresh_button.click()
        self.assertEqual(self.visible(dialog), [])
        dialog.review_view.setCurrentIndex(dialog.review_view.findData('accepted'))
        self.assertEqual(self.visible(dialog), ['alpha'])
        self.assertEqual(dialog.review_view.currentText(), 'Accepted (1)')
        self.assertEqual(dialog.selected_videos(), [])

    def test_open_working_copy_never_prepares_a_transfer(self):
        local = self.download()
        self.remember(local)
        dialog = self.dialog()
        self.assertEqual(dialog.selected_videos(), [])
        self.assertFalse(dialog.download.isEnabled())
        with patch('gui.assigned_work_dialog.PreparedAssignedDownload') as prepare:
            dialog.open_copy.click()
        prepare.assert_not_called()
        self.assertEqual(dialog.result(), QDialog.DialogCode.Accepted)
        self.assertEqual(dialog.download_result, {'path': str(local), 'updated': False, 'bytes': 0})

    def test_hidden_checked_rows_cannot_be_requested_as_new_downloads(self):
        local = self.mixed()
        dialog = self.dialog()
        dialog.table.item(0, dialog.DOWNLOAD_COLUMN).setCheckState(Qt.CheckState.Checked)
        plan = Mock(new_videos=['beta'], updated_videos=[], file_count=1, bytes=10,
                    before={}, kept_edits=[], retained=[])
        with patch('gui.assigned_work_dialog.PreparedAssignedDownload', return_value=plan) as prepare, \
                patch('gui.assigned_work_dialog.TransferConfirmationDialog.exec', return_value=QDialog.DialogCode.Rejected):
            dialog.download_selected()
        self.assertEqual(prepare.call_args.args[3], ['beta'])
        self.assertEqual(prepare.call_args.args[4], local)
        plan.apply.assert_not_called()

    def test_download_adds_new_work_without_replacing_hidden_existing_annotations(self):
        local = self.mixed()
        paths = list((local / 'annotations').glob('*/alpha/*'))
        paths += list((local / 'frames/alpha').glob('*.jpg'))
        before = {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in paths if p.is_file()}
        dialog = self.dialog()
        self.assertEqual(dialog.selected_videos(), ['beta'])
        with patch('gui.assigned_work_dialog.TransferConfirmationDialog.exec', return_value=QDialog.DialogCode.Accepted):
            dialog.download_selected()
        self.warnings.assert_not_called()
        self.assertEqual(dialog.download_result['path'], str(local))
        self.assertTrue((local / 'frames/beta/frame_000001.jpg').is_file())
        self.assertEqual(before, {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in before})

    def test_failed_check_disables_download_and_refresh_recovers(self):
        with patch('gui.assigned_work_dialog.assigned_download_overview', side_effect=OSError('Drive offline')):
            dialog = self.dialog()
        self.assertIn('Drive offline', dialog.status_note.text())
        dialog.set_checked(True)
        self.assertFalse(dialog.download.isEnabled())
        with patch('gui.assigned_work_dialog.PreparedAssignedDownload') as prepare:
            dialog.download_selected()
        prepare.assert_not_called()
        dialog.refresh_button.click()
        self.assertTrue(dialog.download.isEnabled())
        self.assertEqual(dialog.table.item(0, 2).text(), 'Not downloaded')

    def test_project_switch_resets_filters_checks_and_local_copy(self):
        self.mixed()
        dialog = self.dialog()
        dialog.review_view.setCurrentIndex(dialog.review_view.findData('review'))
        dialog.local_filter.setCurrentIndex(dialog.local_filter.findData('downloaded'))
        other_id = self.sam['id']
        dialog.available.append({'id': other_id, 'name': 'Second project',
                                 'videos': {'gamma': {'frames': [0]}}})
        dialog.projects.addItem('Second project', other_id)
        overview = {'statuses': {'gamma': 'Assigned'}, 'local': {'gamma': 'not_downloaded'},
                    'local_path': '', 'local_error': ''}
        with patch('gui.assigned_work_dialog.assigned_download_overview', return_value=overview) as read:
            dialog.projects.setCurrentIndex(1)
        self.assertEqual(read.call_args.args[1], other_id)
        self.assertEqual(dialog.review_view.currentData(), 'active')
        self.assertEqual(dialog.local_filter.currentData(), 'all')
        self.assertEqual(dialog.selected_videos(), ['gamma'])
        self.assertFalse(dialog.open_copy.isEnabled())


if __name__ == '__main__':
    unittest.main()
