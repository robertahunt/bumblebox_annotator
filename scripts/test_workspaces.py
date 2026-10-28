"""Workspace journeys and read-only shared status, using disposable projects."""

from contextlib import ExitStack
from pathlib import Path
import unittest
from unittest.mock import Mock, patch

import numpy as np
from PyQt6.QtCore import QSettings, QTimer, Qt
from PyQt6.QtGui import QCloseEvent, QColor, QPalette
from PyQt6.QtTest import QTest
from PyQt6.QtWidgets import (QApplication, QDialog, QMessageBox, QPushButton, QStyle,
                            QStyleOptionButton, QToolButton)

from core.assigned_work import publish_work, read_assignments, save_assignments, transfer_director
from core.project_activity import project_activity
from core.project_sync import sync_project
from gui.main_window import MainWindow
from gui.project_sync_dialog import save_setting
from scripts.test_assigned_work import AssignedWorkTests


class SharedActivityTests(unittest.TestCase):
    setUp = AssignedWorkTests.setUp
    view = AssignedWorkTests.view
    download = AssignedWorkTests.download
    change = AssignedWorkTests.change

    def test_full_project_is_current_and_local_edits_are_distinct(self):
        result = project_activity(self.source, self.configuration)
        self.assertEqual(result['status'], 'current')
        self.assertFalse(result['changed'])
        self.change(self.source, actor=self.coordinator)
        result = project_activity(self.source, self.configuration)
        self.assertEqual(result['status'], 'current')
        self.assertTrue(result['changed'])

    def test_unrelated_assignment_upload_does_not_request_update(self):
        local = self.download()
        other = self.download(self.sam, ['beta'], 'sam')
        self.change(other, 'beta', 1, self.sam)
        publish_work(other, self.configuration, self.sam)
        result = project_activity(local, self.configuration)
        self.assertEqual(result['status'], 'current')
        self.assertFalse(result['changed'])
        self.assertEqual(project_activity(self.source, self.configuration)['status'], 'update')

    def test_remote_and_local_edits_do_not_offer_blind_update(self):
        local = self.download()
        self.change(local)
        publish_work(local, self.configuration, self.alex)
        self.change(self.source, 'beta', 1, self.coordinator)
        result = project_activity(self.source, self.configuration)
        self.assertEqual(result['status'], 'update')
        self.assertTrue(result['changed'])
        self.assertIn('review before', result['message'])

    def test_same_video_and_reassignment_need_attention(self):
        local = self.download()
        other = self.download(name='second-computer')
        self.change(other)
        publish_work(other, self.configuration, self.alex)
        self.assertEqual(project_activity(local, self.configuration)['status'], 'attention')
        current = read_assignments(self.view())
        self.choices['alpha']['contributor'] = self.sam
        save_assignments(self.source, self.configuration, self.coordinator, self.choices, current['revision'])
        self.assertEqual(project_activity(local, self.configuration)['status'], 'attention')

    def test_missing_drive_and_wrong_root_never_report_current(self):
        local = self.download()
        self.assertEqual(project_activity(local, {})['status'], 'attention')
        with self.assertRaises((OSError, ValueError)):
            project_activity(local, dict(self.configuration, root=str(self.base / 'missing')))
        result = project_activity(local, dict(self.configuration, root_id=self.coordinator['id']))
        self.assertEqual(result['status'], 'attention')

    def test_check_does_not_write_annotations_or_remote_files(self):
        paths = [p for root in (self.source, self.remote) for p in root.rglob('*') if p.is_file()]
        before = {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in paths}
        project_activity(self.source, self.configuration)
        self.assertEqual(before, {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in paths})

    def test_director_transfer_preserves_assignments_and_removes_old_authority(self):
        before = read_assignments(self.view())
        after = transfer_director(self.source, self.configuration, self.coordinator,
                                  self.sam, before['revision'])
        self.assertEqual(after['videos'], before['videos'])
        self.assertEqual(after['coordinator']['id'], self.sam['id'])
        with self.assertRaisesRegex(ValueError, 'Only the coordinator'):
            save_assignments(self.source, self.configuration, self.coordinator, self.choices, after['revision'])
        with self.assertRaisesRegex(ValueError, 'current project director'):
            transfer_director(self.source, self.configuration, self.coordinator, self.alex, after['revision'])

    def test_transfer_rejects_stale_revision_and_assigned_copy(self):
        before = read_assignments(self.view())
        with self.assertRaisesRegex(ValueError, 'Refresh'):
            transfer_director(self.source, self.configuration, self.coordinator, self.sam, None)
        self.assertEqual(read_assignments(self.view()), before)
        local = self.download()
        before = read_assignments(self.view())
        with self.assertRaisesRegex(ValueError, 'full project'):
            transfer_director(local, self.configuration, self.alex, self.sam, before['revision'])
        self.assertEqual(read_assignments(self.view()), before)


class WorkspaceGuiTests(unittest.TestCase):
    view = AssignedWorkTests.view
    download = AssignedWorkTests.download

    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        AssignedWorkTests.setUp(self)
        self.settings = QSettings(str(self.base / 'settings.ini'), QSettings.Format.IniFormat)
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(patch('gui.main_window.QSettings', lambda: self.settings))
        self.stack.enter_context(patch('gui.project_sync_dialog.QSettings', lambda: self.settings))
        self.stack.enter_context(patch.object(MainWindow, 'show_annotation_policy'))
        self.stack.enter_context(patch.object(QMessageBox, 'warning'))
        self.stack.enter_context(patch.object(QMessageBox, 'information'))
        save_setting('project_sync/destination', self.configuration)
        self.window = MainWindow()
        self.window.set_contributor(self.coordinator)

    def tearDown(self):
        self.window.workflow.stop_check()
        self.app.processEvents()
        with patch.object(self.window.workflow, 'confirm_upload_on_close', return_value=True):
            self.window.save_worker.failed_tasks.clear()
            self.window.close()
        self.app.processEvents()

    def load(self, assigned=False):
        if assigned:
            self.window.set_contributor(self.alex)
        path = self.download() if assigned else self.source
        self.window.load_project(path)
        return self.window.workflow

    def test_home_has_no_annotation_chrome_and_can_resume(self):
        flow = self.window.workflow
        self.assertEqual(flow.current, 'Home')
        self.assertTrue(all(dock.isHidden() for dock in flow.docks))
        self.assertFalse(flow.home.open_selected.isEnabled())
        self.assertTrue(flow.home.recent_heading.isHidden())
        self.assertTrue(flow.home.recovery.isHidden())
        self.assertTrue(flow.home.context.isHidden())
        self.assertEqual(flow.home.context.text(), '')
        self.load()
        flow.show_home()
        self.assertTrue(flow.home.open_selected.isEnabled())
        self.assertEqual(flow.home.open_selected.text(), 'Return to Project')
        self.assertFalse(flow.home.context.isHidden())
        flow.home.open_selected.click()
        self.assertEqual(flow.current, 'Annotate')
        self.assertFalse(self.window.canvas.isHidden())

    def test_home_button_has_visible_label_large_house_icon_and_returns_home(self):
        flow = self.load()
        self.window.show()
        self.app.processEvents()
        button = flow.home_button
        self.assertEqual(button.text(), 'Home')
        self.assertEqual(button.toolButtonStyle(), Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.assertEqual(button.accessibleName(), 'Home / My Work')
        self.assertGreaterEqual(button.height(), 44)
        self.assertGreaterEqual(button.width(), 108)
        self.assertEqual(button.iconSize().width(), 28)
        icon = button.icon().pixmap(button.iconSize()).toImage()
        self.assertFalse(icon.isNull())
        white_pixels = sum(icon.pixelColor(x, y).alpha() > 128 and icon.pixelColor(x, y).lightness() > 220
                           for x in range(icon.width()) for y in range(icon.height()))
        self.assertGreater(white_pixels, 50)
        before = self.app.palette()
        try:
            for theme in ('light', 'dark'):
                palette = QPalette(before)
                if theme == 'dark':
                    palette.setColor(QPalette.ColorRole.Window, QColor('#202124'))
                    palette.setColor(QPalette.ColorRole.WindowText, QColor('#eeeeee'))
                    palette.setColor(QPalette.ColorRole.Button, QColor('#202124'))
                    palette.setColor(QPalette.ColorRole.ButtonText, QColor('#eeeeee'))
                self.app.setPalette(palette)
                self.app.processEvents()
                flow.navigation.grab().save(f'/tmp/bumblebox-home-navigation-{theme}.png')
                flow.activate('Annotate')
                QTest.mouseClick(button, Qt.MouseButton.LeftButton)
                self.assertEqual(flow.current, 'Home')
                self.assertTrue(all(dock.isHidden() for dock in flow.docks))
        finally:
            self.app.setPalette(before)

    def test_recent_project_open_button_precedes_other_open_commands(self):
        home = self.window.workflow.home
        self.window.show()
        self.app.processEvents()
        layout = home.recent.parentWidget().layout()
        self.assertEqual(layout.indexOf(home.open_selected), layout.indexOf(home.recent) + 1)
        local = next(button for button in home.findChildren(QPushButton) if button.text() == 'Open Local Project...')
        self.window.recent_projects = [str(self.source)]
        self.settings.setValue('workflow/recent_projects/' + self.coordinator['id'], [str(self.source)])
        home.refresh()
        self.app.processEvents()
        self.assertGreater(local.geometry().top(), home.open_selected.geometry().bottom())
        self.assertEqual(home.recent.currentRow(), 0)
        self.assertTrue(home.open_selected.isEnabled())
        with patch.object(self.window, 'open_recent_project') as open_project:
            home.open_selected.click()
            open_project.assert_called_once_with(str(self.source))
        home.grab().save('/tmp/bumblebox-project-home-layout.png')

    def test_home_is_light_under_dark_theme_and_recovery_is_secondary(self):
        before = self.app.palette()
        dark = QPalette(before)
        dark.setColor(QPalette.ColorRole.Window, QColor('#202124'))
        dark.setColor(QPalette.ColorRole.WindowText, QColor('#eeeeee'))
        self.app.setPalette(dark)
        try:
            home = self.window.workflow.home
            self.window.recent_projects = [str(self.source), '/a/very/long/path/' * 8 + 'second-project']
            self.settings.setValue('workflow/recent_projects/' + self.coordinator['id'], self.window.recent_projects)
            home.refresh()
            self.window.resize(1000, 800)
            self.window.show()
            self.app.processEvents()
            self.assertGreater(home.palette().color(QPalette.ColorRole.Window).lightness(), 220)
            self.assertTrue(home.recovery.isHidden())
            home.recovery_toggle.click()
            self.assertFalse(home.recovery.isHidden())
            self.assertEqual(home.recovery_toggle.arrowType(), Qt.ArrowType.DownArrow)
            home.recent.setCurrentRow(1)
            self.app.processEvents()
            self.assertLessEqual(home.minimumSizeHint().width(), 640)
            home.grab().save('/tmp/bumblebox-project-home-dark-system.png')
            # Exercise the page itself at laptop-sized dimensions independently of editor minimums.
            from gui.workspaces import ProjectHome
            compact = ProjectHome(self.window.workflow)
            compact.refresh()
            compact.resize(640, 600)
            compact.show()
            self.app.processEvents()
            rendered = compact.grab()
            rendered.save('/tmp/bumblebox-project-home-small.png')
            self.assertGreater(rendered.toImage().pixelColor(10, 10).lightness(), 220)
            self.assertLessEqual(compact.minimumSizeHint().width(), 640)
            compact.close()
        finally:
            self.app.setPalette(before)

    def test_startup_checkbox_is_legible_and_clickable_in_both_themes(self):
        before = self.app.palette()
        home = self.window.workflow.home
        checkbox = home.auto_resume
        self.window.show()
        try:
            for theme in ('light', 'dark'):
                with self.subTest(theme=theme):
                    palette = QPalette(before)
                    if theme == 'dark':
                        for role in (QPalette.ColorRole.Window, QPalette.ColorRole.Base,
                                     QPalette.ColorRole.Button, QPalette.ColorRole.Dark):
                            palette.setColor(role, QColor('#202124'))
                        for role in (QPalette.ColorRole.WindowText, QPalette.ColorRole.Text,
                                     QPalette.ColorRole.ButtonText, QPalette.ColorRole.Light):
                            palette.setColor(role, QColor('#eeeeee'))
                    self.app.setPalette(palette)
                    checkbox.setChecked(False)
                    self.app.processEvents()
                    option = QStyleOptionButton()
                    checkbox.initStyleOption(option)
                    rect = checkbox.style().subElementRect(QStyle.SubElement.SE_CheckBoxIndicator, option, checkbox)
                    self.assertGreaterEqual(rect.width(), 20)
                    self.assertGreaterEqual(rect.height(), 20)
                    unchecked = checkbox.grab(rect).toImage()
                    QTest.mouseClick(checkbox, Qt.MouseButton.LeftButton, pos=rect.center())
                    self.app.processEvents()
                    checked = checkbox.grab(rect).toImage()
                    dark_tick_pixels = sum(
                        unchecked.pixelColor(x, y).lightness() - checked.pixelColor(x, y).lightness() > 80
                        and checked.pixelColor(x, y).lightness() < 100
                        for x in range(checked.width()) for y in range(checked.height()))
                    self.assertGreater(dark_tick_pixels, 15)
                    self.assertTrue(self.settings.value('workflow/auto_resume', False, type=bool))
                    checkbox.grab().save(f'/tmp/bumblebox-startup-checkbox-{theme}.png')
                    QTest.mouseClick(checkbox, Qt.MouseButton.LeftButton, pos=rect.center())
                    self.assertFalse(checkbox.isChecked())
                    self.assertFalse(self.settings.value('workflow/auto_resume', True, type=bool))
        finally:
            self.app.setPalette(before)

    def test_help_opens_without_performing_transfer_and_works_when_disabled(self):
        flow = self.window.workflow
        button = next(button for button in flow.manage_page.findChildren(QToolButton)
                      if button.accessibleName() == 'Help: Back Up Local Project')
        self.assertFalse(self.window.sync_project_action.isEnabled())
        self.assertTrue(button.isEnabled())
        with patch('gui.workflow_help.QMessageBox.exec') as show, \
                patch.object(self.window, 'sync_current_project') as publish:
            button.click()
            show.assert_called_once()
            publish.assert_not_called()

    def test_recovery_cancel_and_invalid_folder_never_load_project(self):
        with patch('gui.main_window.TransferConfirmationDialog.exec', return_value=QDialog.DialogCode.Rejected), \
                patch('gui.main_window.QFileDialog.getExistingDirectory') as choose, \
                patch.object(self.window, 'load_project') as load:
            self.window.restore_project_backup()
            choose.assert_not_called()
            load.assert_not_called()
        with patch('gui.main_window.TransferConfirmationDialog.exec', return_value=QDialog.DialogCode.Accepted), \
                patch('gui.main_window.QFileDialog.getExistingDirectory', return_value=str(self.source)), \
                patch.object(self.window, 'load_project') as load:
            self.window.restore_project_backup()
            load.assert_not_called()

    def test_recovery_uses_existing_verified_restore_gate(self):
        snapshot = self.view().snapshot(next(iter(self.view().heads().items())))
        path = str(snapshot['manifest_path'].parent)
        with patch('gui.main_window.TransferConfirmationDialog.exec', return_value=QDialog.DialogCode.Accepted), \
                patch('gui.main_window.QFileDialog.getExistingDirectory', return_value=path), \
                patch.object(self.window, 'load_project') as load:
            self.window.restore_project_backup()
            load.assert_called_once_with(path)

    def test_workspace_switches_preserve_masks_tool_and_view(self):
        flow = self.load()
        self.window.show()
        self.app.processEvents()
        canvas = self.window.canvas
        self.window.toolbar.set_tool('brush')
        canvas.zoom_in()
        transform = canvas.transform()
        masks = [ann['mask'].copy() for ann in canvas.get_annotations()]
        flow.activate('Train')
        self.assertTrue(all(dock.isHidden() for dock in flow.docks))
        flow.activate('Review')
        self.app.processEvents()
        self.assertEqual(canvas.transform(), transform)
        self.assertEqual(canvas.current_tool, 'brush')
        self.assertTrue(all(np.array_equal(a, b['mask']) for a, b in zip(masks, canvas.get_annotations())))
        self.assertFalse(self.window.import_review_controls.isHidden())
        flow.activate('Annotate')
        self.assertTrue(self.window.import_review_controls.isHidden())

    def test_assigned_copy_cannot_enter_training_management_or_review(self):
        flow = self.load(assigned=True)
        for target in ('Train', 'Manage', 'Review'):
            flow.activate(target)
            self.assertEqual(flow.current, 'Annotate')
        self.assertTrue(all(not self.window.workflow.tabs.isTabVisible(i) for i in (1, 2, 3)))
        self.assertTrue(all(not action.isEnabled() for action in self.window.full_project_actions))
        self.assertTrue(self.window.tracking_sequences_dock.isHidden())

    def test_model_assistance_is_explicit_and_does_not_change_visibility(self):
        flow = self.load()
        canvas = self.window.canvas
        before = (canvas.show_segmentations, canvas.show_bboxes)
        self.assertFalse(flow.assist_toggle.isChecked())
        self.assertTrue(self.window.sam2_toolbar.isHidden())
        self.window.yolo_toolbar_action.trigger()
        self.assertTrue(flow.assist_toggle.isChecked())
        self.assertFalse(self.window.yolo_toolbar.isHidden())
        self.assertTrue(self.window.sam2_toolbar.isHidden())
        flow.assist_toggle.setChecked(False)
        self.assertTrue(all(widget.isHidden() for _, widget, _ in self.window._optional_top_toolbar_entries()))
        self.assertEqual(before, (canvas.show_segmentations, canvas.show_bboxes))

    def test_idle_save_keeps_active_edit_and_undo_history(self):
        flow = self.load()
        canvas = self.window.canvas
        canvas.start_editing_instance(1, 'bee')
        canvas.editing_mask[0, 0] = 255
        self.window._mark_current_annotations_dirty()
        with patch.object(canvas, 'commit_editing') as commit, patch.object(canvas, '_clear_brush_history') as clear:
            flow.autosave()
            self.window.save_worker.wait_until_idle()
            commit.assert_not_called()
            clear.assert_not_called()
        self.assertEqual(canvas.editing_instance_id, 1)
        self.assertTrue(flow.pending())
        saved = self.window.annotation_manager.load_frame_annotations(self.source, 'alpha', 0)
        self.assertEqual(saved[0]['mask'][0, 0], 255)

    def test_autosave_waits_until_stroke_finishes(self):
        flow = self.load()
        self.window._mark_current_annotations_dirty()
        self.window.canvas.is_drawing = True
        with patch.object(self.window, '_save_annotation_sources') as save:
            flow.autosave()
            save.assert_not_called()
        self.window.canvas.is_drawing = False

    def test_failed_save_blocks_close(self):
        self.load()
        self.window._mark_current_annotations_dirty()
        with patch.object(self.window, '_save_annotation_sources', side_effect=OSError('disk full')):
            event = QCloseEvent()
            self.window.closeEvent(event)
            self.assertFalse(event.isAccepted())
        self.assertTrue(self.window.save_worker.isRunning())

    def test_close_upload_cancel_failure_and_local_only(self):
        flow = self.load(assigned=True)
        self.window.canvas.start_editing_instance(1, 'bee')
        self.window.canvas.editing_mask[0, 0] = 255
        self.window._mark_current_annotations_dirty()
        flow.autosave()
        self.window.save_worker.wait_until_idle()
        def choose(text):
            def click():
                box = self.app.activeModalWidget()
                next(button for button in box.buttons() if text in button.text()).click()
            QTimer.singleShot(0, click)
        choose('Cancel')
        self.assertFalse(flow.confirm_upload_on_close())
        choose('Upload and Close')
        with patch.object(self.window, 'sync_current_project', return_value=False):
            self.assertFalse(flow.confirm_upload_on_close())
        choose('Close with Work')
        self.assertTrue(flow.confirm_upload_on_close())
        self.assertTrue(flow.pending())

    def test_stale_background_check_cannot_clear_new_edits(self):
        flow = self.load()
        generation = flow.generation
        flow.saved()
        worker = Mock(error=None, result={'status': 'current', 'message': 'No newer version', 'changed': False})
        flow.worker = worker
        flow.checked(worker, str(self.source), generation)
        self.assertTrue(flow.pending())

    def test_preferences_are_separate_for_each_contributor_and_project(self):
        flow = self.load()
        first = flow.key('workspace')
        flow.activate('Train')
        flow.show_home()
        flow.resume()
        self.assertEqual(flow.current, 'Train')
        self.window.set_contributor(self.alex)
        self.assertNotEqual(flow.key('workspace'), first)
        self.assertEqual(self.settings.value(flow.key('workspace'), 'Annotate'), 'Annotate')

    def test_idle_save_backpressure_does_not_queue_more_masks(self):
        flow = self.load()
        self.window._mark_current_annotations_dirty()
        with patch.object(self.window.save_worker, 'save_queue') as queue, \
                patch.object(self.window, '_save_annotation_sources') as save:
            queue.unfinished_tasks = 1
            flow.autosave()
            save.assert_not_called()

    def test_home_and_workspaces_fit_desktop_sizes(self):
        flow = self.load()
        self.window.show()
        for width, height in ((1400, 900), (1024, 768)):
            self.window.resize(width, height)
            for name in ('Home', 'Annotate', 'Review', 'Train', 'Manage'):
                flow.activate(name)
                self.app.processEvents()
                self.assertLessEqual(flow.stack.width(), self.window.width())
                self.assertGreater(flow.stack.height(), 100)
                self.window.grab().save(f'/tmp/bumblebox-{name.lower()}-{width}.png')


if __name__ == '__main__':
    unittest.main()
