"""Read-only outgoing review dialogs and publishing entry points."""

import unittest
from unittest.mock import patch

from PyQt6.QtCore import QTimer
from PyQt6.QtWidgets import QApplication, QDialogButtonBox, QMessageBox, QPushButton

from gui.publication_review_dialog import PublicationReviewDialog
from gui.project_sync_dialog import ProjectSyncDialog, confirm_publication
from scripts import test_compact_sync_gui as fixtures


def sample_estimate(count=1):
    rows = [{'path': f'annotations/png/colony_31/frame_{index:06d}.png', 'video': 'colony_31',
             'action': 'Changed', 'scope': 'Frame annotation', 'before_bytes': 1024, 'after_bytes': 2048}
            for index in range(count)]
    rows += [{'path': 'annotations/project.json', 'video': None, 'action': 'Changed',
              'scope': 'Project settings', 'before_bytes': 256, 'after_bytes': 512},
             {'path': 'annotations/png/colony_41/video_annotations_hive.png', 'video': 'colony_41',
              'action': 'Removed', 'scope': 'Shared across video', 'before_bytes': 8192, 'after_bytes': None},
             {'path': 'input_data/train/colony_41.mp4', 'video': 'colony_41', 'action': 'Excluded',
              'scope': 'Original video', 'before_bytes': 81920, 'after_bytes': None}]
    return {'project_name': 'Review project', 'files': 27868, 'logical_bytes': 33 * 1024**3,
            'new_bytes': 14 * 1024**2, 'free_bytes': 100 * 1024**3,
            'review': {'baseline': {'contributor_name': 'August', 'created_at': '2026-10-20T12:00:00Z',
                                    'revision': '20261020T120000-example'},
                       'rows': rows, 'counts': {'Changed': count + 1, 'Removed': 1, 'Excluded': 1},
                       'unchanged': 20000}}


class PublicationReviewDialogTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_groups_scopes_search_filters_and_read_only_details(self):
        with patch('core.project_sync._digest', side_effect=AssertionError('Review must not hash')):
            dialog = PublicationReviewDialog(sample_estimate())
            self.assertEqual(dialog.table.topLevelItemCount(), 3)
            self.assertEqual(dialog.table.topLevelItem(0).text(0), 'Project files')
            group = dialog.table.topLevelItem(2)
            self.assertEqual(group.text(0), 'colony_41')
            self.assertEqual(group.child(0).text(2), 'Shared across video')
            dialog.table.setCurrentItem(group.child(0))
            self.assertIn('video_annotations_hive.png', dialog.details.text())
            dialog.search.setText('colony_41')
            self.assertEqual(dialog.table.topLevelItemCount(), 1)
            dialog.actions.setCurrentText('Removed')
            self.assertEqual(dialog.table.topLevelItem(0).childCount(), 1)
            dialog.search.setText('missing video')
            self.assertEqual(dialog.table.topLevelItemCount(), 0)
            self.assertEqual(dialog.empty.text(), 'No changes match these filters.')
            dialog.search.clear()
            dialog.actions.setCurrentIndex(0)
            self.assertEqual(dialog.table.topLevelItemCount(), 3)
            self.assertFalse(any('Publish' in button.text() for button in dialog.findChildren(QPushButton)))
        dialog.show()
        self.app.processEvents()
        dialog.grab().save('/tmp/bumblebox-outgoing-review.png')
        dialog.resize(640, 520)
        self.app.processEvents()
        self.assertFalse(dialog.table.geometry().intersects(dialog.details.geometry()))
        dialog.grab().save('/tmp/bumblebox-outgoing-review-small.png')
        dialog.reject()

    def test_large_review_populates_children_only_when_expanded(self):
        dialog = PublicationReviewDialog(sample_estimate(28000))
        self.assertEqual(dialog.table.topLevelItemCount(), 3)
        self.assertTrue(all(dialog.table.topLevelItem(i).childCount() == 0 for i in range(3)))
        group = dialog.table.topLevelItem(2)
        group.setExpanded(True)
        self.assertEqual(group.childCount(), 2)
        self.assertEqual(dialog.table.topLevelItem(1).childCount(), 0)
        dialog.search.setText('frame_027999')
        self.assertEqual(dialog.table.topLevelItemCount(), 1)
        self.assertEqual(dialog.table.topLevelItem(0).childCount(), 1)

    def test_no_changes_and_first_publication_states(self):
        estimate = sample_estimate()
        estimate['review'].update(rows=[], counts={})
        dialog = PublicationReviewDialog(estimate)
        self.assertEqual(dialog.empty.text(), 'No file changes since this backup.')
        estimate['review']['baseline'] = None
        dialog = PublicationReviewDialog(estimate)
        self.assertIn('First backup for this contributor', dialog.baseline.text())

    def test_final_confirmation_review_does_not_accept_or_publish(self):
        estimate = sample_estimate()
        seen = []
        def review_and_cancel():
            box = self.app.activeModalWidget()
            with patch('gui.project_sync_dialog.show_publication_review') as review:
                next(button for button in box.buttons.buttons() if button.text() == 'Review Changes...').click()
                seen.append(review.call_args.args[0])
            self.assertTrue(box.isVisible())
            box.buttons.button(QDialogButtonBox.StandardButton.Cancel).click()
        QTimer.singleShot(0, review_and_cancel)
        self.assertFalse(confirm_publication(None, estimate, {}))
        self.assertIs(seen[0], estimate)


class PublicationReviewEntryTests(unittest.TestCase):
    setUpClass = classmethod(fixtures.StorageSettingsTests.setUpClass.__func__)
    setUp = fixtures.StorageSettingsTests.setUp
    tearDown = fixtures.StorageSettingsTests.tearDown
    make_window = fixtures.StorageSettingsTests.make_window
    linked_window = fixtures.StorageSettingsTests.linked_window

    def test_manage_review_reuses_estimate_and_does_not_publish(self):
        window = self.linked_window()
        self.assertFalse(window.review_backup_action.isEnabled())
        estimate = window.estimate_project_backup()
        self.assertTrue(window.review_backup_action.isEnabled())
        with patch('gui.main_window.show_publication_review') as review, \
                patch('gui.main_window.publication_estimate', side_effect=AssertionError('Do not estimate again')), \
                patch('core.project_sync._digest', side_effect=AssertionError('Do not hash again')), \
                patch('gui.main_window.sync_project') as publish:
            window.review_project_backup()
            review.assert_called_once_with(estimate, window)
            publish.assert_not_called()
        window._mark_current_annotations_dirty()
        self.assertFalse(window.review_backup_action.isEnabled())

    def test_review_blocks_external_local_changes_and_invalidates_buttons(self):
        window = self.linked_window()
        window.estimate_project_backup()
        (self.source / 'frames/new.png').write_bytes(b'changed')
        with patch('gui.main_window.show_publication_review') as review, patch.object(QMessageBox, 'information'):
            window.review_project_backup()
            review.assert_not_called()
        self.assertFalse(window.review_backup_action.isEnabled())
        self.assertFalse(window.sync_project_action.isEnabled())

    def test_settings_review_gate_and_option_invalidation(self):
        window = self.linked_window()
        dialog = ProjectSyncDialog(self.source, window)
        self.assertFalse(dialog.review_button.isEnabled())
        dialog.estimate_transfer()
        self.assertTrue(dialog.review_button.isEnabled())
        with patch('gui.project_sync_dialog.show_publication_review') as review:
            dialog.review_button.click()
            review.assert_called_once_with(dialog.transfer_estimate, dialog)
        self.assertFalse(dialog.sync_requested)
        dialog.videos.toggle()
        self.assertFalse(dialog.review_button.isEnabled())
        self.assertFalse(window.review_backup_action.isEnabled())


if __name__ == '__main__':
    unittest.main()
