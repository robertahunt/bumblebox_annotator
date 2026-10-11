"""Offscreen policy dialogs and saved, individual-frame completion workflow."""

import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
os.environ.setdefault('NO_ALBUMENTATIONS_UPDATE', '1')

from unittest.mock import patch
import unittest

from PyQt6.QtCore import QSettings
from PyQt6.QtWidgets import QApplication, QDialog, QMessageBox

from core.annotation_policy import category_policies, save_policy
from core.import_review import complete_review, review_status
from gui.annotation_policy_dialog import AnnotationPolicyDialog, ImportedFrameReviewDialog
from gui.dialogs import ProjectDialog
from gui.main_window import MainWindow
from scripts.test_annotation_policy import PolicyFixture


class AnnotationPolicyGuiTests(PolicyFixture):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def make_window(self):
        settings = QSettings(str(self.root / 'settings.ini'), QSettings.Format.IniFormat)
        mock = patch('gui.main_window.QSettings', lambda: settings)
        mock.start()
        self.addCleanup(mock.stop)
        with patch.object(MainWindow, 'load_settings'):
            window = MainWindow()
        def close():
            with patch.object(QMessageBox, 'question', return_value=QMessageBox.StandardButton.Discard):
                window.close()
            self.app.processEvents()
        self.addCleanup(close)
        window.set_contributor(self.contributor)
        with patch.object(QMessageBox, 'critical') as errors:
            window.load_project(self.dest)
            errors.assert_not_called()
        return window

    def test_new_project_policy_follows_hive_scope_choice(self):
        dialog = ProjectDialog(default_dir=self.root)
        self.addCleanup(dialog.close)
        self.assertEqual(dialog.get_project_info()['annotation_policy']['categories']['hive']['meaning'], 'visible')
        dialog.hive_scope_combo.setCurrentIndex(dialog.hive_scope_combo.findData('video'))
        self.assertEqual(dialog.get_project_info()['annotation_policy']['categories']['hive'],
                         {'scope': 'video', 'meaning': 'full_extent'})
        dialog.show()
        self.app.processEvents()
        dialog.grab().save('/tmp/annotation-new-project.png')

    def test_policy_save_requires_confirmation_and_reloads_manager(self):
        window = self.make_window()
        dialog = window._policy_dialog
        self.assertFalse(dialog.save_button.isEnabled())
        combo = dialog.policy.meanings['hive']
        combo.setCurrentIndex(combo.findData('visible'))
        dialog.confirm.setChecked(True)
        dialog.save()
        self.assertEqual(category_policies(window.annotation_manager.project_info)['hive']['meaning'], 'visible')
        window.canvas.current_annotation_type = 'hive'
        window._update_annotation_policy_label()
        self.assertIn('Visible surfaces', window.policy_label.text())

    def test_automatic_policy_reminder_only_once_per_policy_version(self):
        window = self.make_window()
        dialog = window._policy_dialog
        dialog.close()
        window.show_annotation_policy(automatic=True)
        self.assertIs(window._policy_dialog, dialog)
        self.assertFalse(dialog.isVisible())
        entries = category_policies(self.dest_info)
        entries['hive']['meaning'] = 'visible'
        save_policy(self.dest, entries)
        window.show_annotation_policy(automatic=True)
        self.assertIsNot(window._policy_dialog, dialog)
        self.assertTrue(window._policy_dialog.isVisible())
        self.app.processEvents()
        window._policy_dialog.grab().save('/tmp/annotation-policy.png')

    def test_individual_review_saves_completes_and_survives_unchanged_reload(self):
        self.import_hives()
        window = self.make_window()
        self.assertEqual(window._get_frame_idx_in_video(window.current_frame_idx), 2)
        self.assertTrue(window.frame_review_check.isEnabled())
        self.assertFalse(window.frame_review_check.isChecked())
        with patch.object(QMessageBox, 'question', return_value=QMessageBox.StandardButton.Yes), \
                patch.object(QMessageBox, 'warning') as warnings:
            window.frame_review_check.click()
            warnings.assert_not_called()
        self.assertTrue(window.frame_review_check.isChecked())
        self.assertEqual(review_status(self.dest, 'video', 2), 'complete')
        self.assertEqual(review_status(self.dest, 'video', 5), 'pending')
        window._save_annotation_sources()
        self.assertEqual(review_status(self.dest, 'video', 2), 'complete')
        window.load_frame(1)
        self.assertFalse(window.frame_review_check.isChecked())
        window.load_frame(0)
        self.assertTrue(window.frame_review_check.isChecked())
        window.on_annotation_changed()
        self.assertFalse(window.frame_review_check.isChecked())
        self.assertIn('review again', window.import_review_label.text())
        window.resize(1360, 900)
        window._policy_dialog.close()
        window.show()
        self.app.processEvents()
        window.grab().save('/tmp/annotation-review-main.png')

    def test_cancel_completion_leaves_pending_and_uncheck_reopens(self):
        self.import_hives()
        window = self.make_window()
        with patch.object(QMessageBox, 'question', return_value=QMessageBox.StandardButton.Cancel):
            window.frame_review_check.click()
        self.assertFalse(window.frame_review_check.isChecked())
        self.assertEqual(review_status(self.dest, 'video', 2), 'pending')
        complete_review(self.dest, 'video', 2)
        window._update_import_review_controls()
        window.frame_review_check.click()
        self.assertEqual(review_status(self.dest, 'video', 2), 'pending')

    def test_save_failure_never_completes_review(self):
        self.import_hives()
        window = self.make_window()
        with patch.object(window, '_save_annotation_sources', side_effect=OSError('disk full')), \
                patch.object(QMessageBox, 'warning') as warning:
            window.frame_review_check.click()
        warning.assert_called_once()
        self.assertFalse(window.frame_review_check.isChecked())
        self.assertEqual(review_status(self.dest, 'video', 2), 'pending')

    def test_queue_opens_pending_frame_and_shows_policy_conflicts(self):
        self.import_hives()
        complete_review(self.dest, 'video', 2)
        dialog = ImportedFrameReviewDialog(self.dest)
        self.addCleanup(dialog.close)
        self.assertEqual(dialog.table.currentRow(), 1)
        self.assertIn('Visible surfaces', dialog.table.item(1, 3).text())
        self.assertIn('Full object extent', dialog.table.item(1, 3).text())
        dialog.show()
        self.app.processEvents()
        dialog.grab().save('/tmp/annotation-review-queue.png')
        dialog.open_frame()
        self.assertEqual(dialog.target, ('video', 5))
        window = self.make_window()
        def choose_frame(queue):
            queue.target = ('video', 5)
            return QDialog.DialogCode.Accepted
        with patch.object(ImportedFrameReviewDialog, 'exec', choose_frame):
            window.show_import_reviews()
        self.assertEqual(window._get_frame_idx_in_video(window.current_frame_idx), 5)

    def test_frames_without_imports_do_not_need_completion(self):
        window = self.make_window()
        self.assertFalse(window.frame_review_check.isEnabled())
        self.assertEqual(window.import_review_label.text(), '')


if __name__ == '__main__':
    unittest.main()
