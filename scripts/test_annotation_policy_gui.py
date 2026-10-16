"""Offscreen policy dialogs and saved, individual-frame completion workflow."""

import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
os.environ.setdefault('NO_ALBUMENTATIONS_UPDATE', '1')

from unittest.mock import patch
import unittest

from PyQt6.QtCore import QSettings, Qt
from PyQt6.QtWidgets import QApplication, QComboBox, QDialog, QMessageBox

from core.annotation_policy import category_policies, policy_signature, save_policy
from core.categories import CATEGORIES, BROOD_CATEGORIES
from core.import_review import complete_review, review_status
from core.project_sync import _read_json, _write_json
from gui.annotation_policy_dialog import AnnotationPolicyDialog, ImportedFrameReviewDialog, PolicyTable
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
        dialog.begin_editing()
        combo = dialog.policy.meanings['hive']
        combo.setCurrentIndex(combo.findData('visible'))
        dialog.confirm.setChecked(True)
        dialog.save()
        self.assertEqual(category_policies(window.annotation_manager.project_info)['hive']['meaning'], 'visible')
        window.canvas.current_annotation_type = 'hive'
        window._update_annotation_policy_label()
        self.assertIn('Visible surfaces', window.policy_label.text())

    def test_new_project_groups_all_brood_stages_without_changing_labels(self):
        dialog = ProjectDialog(default_dir=self.root)
        self.addCleanup(dialog.close)
        table = dialog.policy_table
        self.assertEqual(table.rows, ['bee', 'hive', 'chamber', 'pollen', 'brood', 'nectar'])
        self.assertEqual(table.table.rowCount(), 6)
        combo = table.meanings['brood']
        combo.setCurrentIndex(combo.findData('full_extent'))
        entries = dialog.get_project_info()['annotation_policy']['categories']
        self.assertEqual(set(entries), set(CATEGORIES))
        for category in BROOD_CATEGORIES:
            self.assertEqual(entries[category], {'scope': 'frame', 'meaning': 'full_extent'})
        self.assertEqual(entries['nectar']['meaning'], 'visible')
        self.assertIn('queen brood', table.help.text())

    def test_grouped_brood_meaning_saves_and_reopens_for_every_stage(self):
        dialog = AnnotationPolicyDialog(self.dest, self.contributor)
        self.addCleanup(dialog.close)
        dialog.begin_editing()
        combo = dialog.policy.meanings['brood']
        combo.setCurrentIndex(combo.findData('full_extent'))
        dialog.confirm.setChecked(True)
        with patch.object(QMessageBox, 'warning') as warning:
            dialog.save()
            warning.assert_not_called()
        info = _read_json(self.dest / 'annotations/project.json')
        entries = category_policies(info)
        for category in BROOD_CATEGORIES:
            self.assertEqual(entries[category], {'scope': 'frame', 'meaning': 'full_extent'})
        self.assertEqual(entries['nectar'], category_policies(self.dest_info)['nectar'])
        reopened = PolicyTable(info)
        self.addCleanup(reopened.close)
        self.assertEqual(reopened.meanings['brood'].currentData(), 'full_extent')

    def test_mixed_brood_meanings_are_preserved_until_explicitly_unified(self):
        entries = category_policies(self.dest_info)
        entries['brood_early']['meaning'] = 'full_extent'
        entries['queen_brood_late']['meaning'] = 'unconfirmed'
        save_policy(self.dest, entries)
        before = (self.dest / 'annotations/project.json').read_bytes()
        dialog = AnnotationPolicyDialog(self.dest, self.contributor)
        self.addCleanup(dialog.close)
        combo = dialog.policy.meanings['brood']
        self.assertIsNone(combo.currentData())
        self.assertIn('Mixed', combo.currentText())
        dialog.policy.update_guidance('brood')
        self.assertIn('remain unchanged', dialog.policy.help.text())
        self.assertEqual(dialog.policy.entries(), entries)
        self.assertEqual((self.dest / 'annotations/project.json').read_bytes(), before)
        dialog.confirm.setChecked(True)
        dialog.save()
        self.assertEqual(category_policies(_read_json(self.dest / 'annotations/project.json')), entries)
        combo.setCurrentIndex(combo.findData('visible'))
        dialog.save()
        saved = category_policies(_read_json(self.dest / 'annotations/project.json'))
        self.assertTrue(all(saved[c]['meaning'] == 'visible' for c in BROOD_CATEGORIES))

    def test_nectar_video_scope_explains_unavailable_and_restores_saved_scope(self):
        for new in (True, False):
            with self.subTest(new=new):
                table = PolicyTable(self.dest_info, new=new)
                self.addCleanup(table.close)
                before = table.entries()
                combo = table.scopes['nectar']
                with patch.object(QMessageBox, 'information') as notice:
                    combo.setCurrentIndex(combo.findData('video'))
                    notice.assert_called_once()
                    self.assertIn('not been implemented', notice.call_args.args[2])
                    self.assertEqual(combo.currentData(), 'frame')
                    self.assertEqual(table.entries(), before)
                    combo.setCurrentIndex(combo.findData('video'))
                    self.assertEqual(notice.call_count, 2)
                self.assertEqual(table.entries(), before)

    def test_all_fixed_scope_categories_explain_and_restore_in_new_and_existing_projects(self):
        for new in (True, False):
            table = PolicyTable(self.dest_info, new=new)
            self.addCleanup(table.close)
            self.assertEqual(set(table.scopes), set(table.rows))
            before = table.entries()
            for row, group in enumerate(table.rows):
                self.assertIsInstance(table.table.cellWidget(row, 1), QComboBox)
                if group == 'hive':
                    continue
                with self.subTest(new=new, category=group):
                    saved = before[table.groups[group][0]]['scope']
                    requested = 'video' if saved == 'frame' else 'frame'
                    combo = table.scopes[group]
                    self.assertEqual(combo.itemData(combo.findData(requested), Qt.ItemDataRole.ToolTipRole),
                                     'Not implemented yet')
                    with patch.object(QMessageBox, 'information') as notice:
                        for _ in range(2):
                            combo.setCurrentIndex(combo.findData(requested))
                            self.assertEqual(combo.currentData(), saved)
                            self.assertEqual(table.entries(), before)
                        self.assertEqual(notice.call_count, 2)
                        self.assertIn('not been implemented', notice.call_args.args[2])
                        self.assertIn(table.table.item(row, 0).text(), notice.call_args.args[2])

    def test_existing_hive_scope_requires_conversion_and_preserves_saved_annotations(self):
        self.import_hives()
        for scope in ('frame', 'video'):
            with self.subTest(scope=scope):
                project = self.source if scope == 'frame' else self.dest
                before = {p.relative_to(project): p.read_bytes() for p in (project / 'annotations').rglob('*')
                          if p.is_file()}
                dialog = AnnotationPolicyDialog(project, self.contributor)
                self.addCleanup(dialog.close)
                dialog.begin_editing()
                entries = dialog.policy.entries()
                combo = dialog.policy.scopes['hive']
                requested = 'video' if scope == 'frame' else 'frame'
                with patch.object(QMessageBox, 'information') as notice:
                    combo.setCurrentIndex(combo.findData(requested))
                notice.assert_called_once()
                self.assertIn('Conversion Required', notice.call_args.args[1])
                self.assertIn('Import From Project', notice.call_args.args[2])
                self.assertNotIn('not been implemented', notice.call_args.args[2])
                self.assertEqual(combo.currentData(), scope)
                self.assertEqual(dialog.policy.entries(), entries)
                self.assertEqual(before, {p.relative_to(project): p.read_bytes()
                                         for p in (project / 'annotations').rglob('*') if p.is_file()})
                dialog.confirm.setChecked(True)
                dialog.save()
                self.assertEqual(category_policies(_read_json(project / 'annotations/project.json')), entries)

    def test_new_project_hive_scope_controls_stay_synchronized_in_both_directions(self):
        dialog = ProjectDialog(default_dir=self.root)
        self.addCleanup(dialog.close)
        combo = dialog.policy_table.scopes['hive']
        with patch.object(QMessageBox, 'information') as notice:
            for control, scope in ((combo, 'video'), (dialog.hive_scope_combo, 'frame'),
                                   (dialog.hive_scope_combo, 'video'), (combo, 'frame')):
                control.setCurrentIndex(control.findData(scope))
                self.assertEqual(combo.currentData(), scope)
                self.assertEqual(dialog.hive_scope_combo.currentData(), scope)
                info = dialog.get_project_info()
                self.assertEqual(info['hive_annotation_scope'], scope)
                self.assertEqual(info['annotation_policy']['categories']['hive'],
                                 {'scope': scope, 'meaning': 'visible' if scope == 'frame' else 'full_extent'})
            notice.assert_not_called()
        meaning = dialog.policy_table.meanings['hive']
        meaning.setCurrentIndex(meaning.findData('full_extent'))
        dialog.policy_table.set_hive_scope('frame')
        self.assertEqual(meaning.currentData(), 'full_extent')

    def test_legacy_brood_group_does_not_confirm_meanings_on_open(self):
        info = dict(self.dest_info)
        info.pop('annotation_policy', None)
        _write_json(self.dest / 'annotations/project.json', info)
        dialog = AnnotationPolicyDialog(self.dest, self.contributor)
        self.addCleanup(dialog.close)
        self.assertEqual(dialog.policy.meanings['brood'].currentData(), 'unconfirmed')
        self.assertTrue(all(entry['meaning'] == 'unconfirmed' for entry in dialog.policy.entries().values()))

    def test_automatic_policy_reminder_only_once_per_policy_version(self):
        window = self.make_window()
        dialog = window._policy_dialog
        dialog.defer_button.click()
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

    def test_unset_policy_returns_after_dismissal_and_application_restart(self):
        info = dict(self.dest_info)
        info.pop('annotation_policy', None)
        path = self.dest / 'annotations/project.json'
        _write_json(path, info)
        before = path.read_bytes()
        window = self.make_window()
        dialog = window._policy_dialog
        self.assertTrue(dialog.reminder_label.isVisible())
        self.assertEqual(dialog.defer_button.text(), 'Remind Me Later')
        # Simulate the old app having marked an unconfirmed policy as seen.
        settings = QSettings(str(self.root / 'settings.ini'), QSettings.Format.IniFormat)
        settings.setValue(window._annotation_policy_seen_key(), policy_signature(info))
        combo = dialog.policy.meanings['hive']
        combo.setCurrentIndex(combo.findData('visible'))
        dialog.defer_button.click()
        self.assertEqual(path.read_bytes(), before)
        window.load_project(self.dest)
        reminder = window._policy_dialog
        self.assertIsNot(reminder, dialog)
        self.assertTrue(reminder.isVisible())
        self.assertEqual(reminder.policy.meanings['hive'].currentData(), 'unconfirmed')
        self.app.processEvents()
        reminder.grab().save('/tmp/annotation-policy-reminder.png')
        reminder.close()
        with patch.object(QMessageBox, 'question', return_value=QMessageBox.StandardButton.Yes):
            window.close()
        reopened = self.make_window()
        self.assertTrue(reopened._policy_dialog.isVisible())
        self.assertTrue(reopened._policy_dialog.reminder_label.isVisible())
        self.assertEqual(path.read_bytes(), before)

    def test_partial_policy_save_keeps_reminding_until_fully_confirmed(self):
        info = dict(self.dest_info)
        info.pop('annotation_policy', None)
        _write_json(self.dest / 'annotations/project.json', info)
        window = self.make_window()
        dialog = window._policy_dialog
        combo = dialog.policy.meanings['hive']
        combo.setCurrentIndex(combo.findData('visible'))
        dialog.confirm.setChecked(True)
        dialog.save()
        window.load_project(self.dest)
        reminder = window._policy_dialog
        self.assertTrue(reminder.isVisible())
        self.assertEqual(reminder.policy.meanings['hive'].currentData(), 'visible')
        for combo in reminder.policy.meanings.values():
            combo.setCurrentIndex(combo.findData('visible'))
        reminder.confirm.setChecked(True)
        reminder.save()
        window.load_project(self.dest)
        self.assertIs(window._policy_dialog, reminder)
        self.assertFalse(reminder.isVisible())
        with patch.object(QMessageBox, 'question', return_value=QMessageBox.StandardButton.Yes):
            window.close()
        reopened = self.make_window()
        self.assertIsNone(getattr(reopened, '_policy_dialog', None))
        reopened.show_annotation_policy()
        self.assertTrue(reopened._policy_dialog.isVisible())
        self.assertTrue(reopened._policy_dialog.reminder_label.isHidden())

    def test_failed_policy_save_does_not_silence_reminder(self):
        entries = category_policies(self.dest_info)
        entries['nectar']['meaning'] = 'unconfirmed'
        save_policy(self.dest, entries)
        window = self.make_window()
        dialog = window._policy_dialog
        dialog.policy.meanings['nectar'].setCurrentIndex(dialog.policy.meanings['nectar'].findData('visible'))
        dialog.confirm.setChecked(True)
        with patch('gui.annotation_policy_dialog.save_policy', side_effect=OSError('disk full')), \
                patch.object(QMessageBox, 'warning') as warning:
            dialog.save()
        warning.assert_called_once()
        self.assertTrue(dialog.isVisible())
        dialog.close()
        window.load_project(self.dest)
        self.assertTrue(window._policy_dialog.isVisible())
        self.assertEqual(window._policy_dialog.policy.meanings['nectar'].currentData(), 'unconfirmed')

    def test_switching_to_seen_project_closes_previous_unfinished_reminder(self):
        info = dict(self.dest_info)
        info.pop('annotation_policy', None)
        _write_json(self.dest / 'annotations/project.json', info)
        window = self.make_window()
        window.load_project(self.source)
        window._policy_dialog.defer_button.click()
        window.load_project(self.dest)
        reminder = window._policy_dialog
        self.assertTrue(reminder.isVisible())
        window.load_project(self.source)
        self.assertFalse(reminder.isVisible())
        self.assertIs(window._policy_dialog, reminder)

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

    def test_complete_policy_review_never_saves_and_requires_explicit_edit(self):
        window = self.make_window()
        dialog = window._policy_dialog
        path = self.dest / 'annotations/project.json'
        before = path.read_bytes()
        self.assertFalse(dialog.editing)
        self.assertEqual(dialog.windowTitle(), 'Review Saved Annotation Policy')
        self.assertTrue(dialog.save_button.isHidden())
        self.assertTrue(all(not combo.isEnabled() for combo in dialog.policy.meanings.values()))
        dialog.confirm.setChecked(True)
        dialog.save()
        self.assertEqual(path.read_bytes(), before)
        dialog.defer_button.click()
        self.assertEqual(path.read_bytes(), before)
        window.show_annotation_policy(automatic=True)
        self.assertIs(window._policy_dialog, dialog)
        window.show_annotation_policy()
        dialog = window._policy_dialog
        dialog.edit_button.click()
        self.assertTrue(dialog.editing)
        self.assertTrue(dialog.policy.meanings['hive'].isEnabled())
        dialog.policy.meanings['hive'].setCurrentIndex(dialog.policy.meanings['hive'].findData('visible'))
        dialog.defer_button.click()
        self.assertEqual(path.read_bytes(), before)

    def test_closing_review_without_continue_does_not_acknowledge_it(self):
        window = self.make_window()
        first = window._policy_dialog
        first.close()
        window.show_annotation_policy(automatic=True)
        self.assertIsNot(window._policy_dialog, first)

    def test_policy_changed_elsewhere_cannot_be_overwritten_by_open_dialog(self):
        dialog = AnnotationPolicyDialog(self.dest, self.contributor)
        self.addCleanup(dialog.close)
        dialog.begin_editing()
        entries = category_policies(self.dest_info)
        entries['nectar']['meaning'] = 'full_extent'
        save_policy(self.dest, entries)
        dialog.confirm.setChecked(True)
        with patch.object(QMessageBox, 'warning') as warning:
            dialog.save()
        self.assertIn('changed while this dialog', warning.call_args.args[2])
        self.assertEqual(category_policies(_read_json(self.dest / 'annotations/project.json')), entries)


if __name__ == '__main__':
    unittest.main()
