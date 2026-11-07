"""Project-first Home journeys on disposable projects, without model inference."""

from pathlib import Path
import shutil
import unittest
from unittest.mock import Mock, patch
from uuid import uuid4

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QApplication, QBoxLayout, QScrollArea

from gui.lab_startup import set_working_locally
from gui.workspaces import ProjectHome
from scripts import test_assignment_workflow as fixture


class ProjectHomeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    setUp = fixture.AssignmentOnboardingGuiTests.setUp
    tearDown = fixture.AssignmentOnboardingGuiTests.tearDown
    view = fixture.AssignmentOnboardingGuiTests.view
    download = fixture.AssignmentOnboardingGuiTests.download
    load = fixture.AssignmentOnboardingGuiTests.load
    settle = fixture.AssignmentOnboardingGuiTests.settle

    def shared(self, name='Second Project', video='beta', **extra):
        return dict(id=str(uuid4()), name=name, role='contributor', count=1,
                    videos={video: {'id': str(uuid4()), 'frames': [0, 2], 'task': 'hive'}}, **extra)

    def test_local_and_shared_project_are_one_entry_without_merging_same_names(self):
        self.load()
        inbox = self.settle()
        first = inbox.shared_projects[0]
        inbox.show_projects([first, self.shared(name=first['name'])])
        self.assertEqual(inbox.projects.count(), 2)
        self.assertEqual(inbox.copies.currentData(), str(self.source))
        self.assertTrue(inbox.open.isEnabled())
        inbox.projects.setCurrentRow(1)
        self.assertEqual(inbox.copies.count(), 0)
        self.assertEqual(inbox.open.text(), 'Download My Assigned Work...')

    def test_selecting_project_only_previews_its_assignments(self):
        inbox = self.settle()
        a, b = self.shared(video='alpha'), self.shared(video='beta')
        with patch.object(self.window, 'open_assigned_work') as download, \
                patch.object(self.window, 'open_recent_project') as open_project:
            inbox.show_projects([a, b])
            inbox.projects.setCurrentRow(1)
            self.assertEqual(inbox.videos.topLevelItemCount(), 1)
            row = inbox.videos.topLevelItem(0)
            self.assertEqual([row.text(i) for i in range(3)], ['beta', 'Hive', '2'])
            download.assert_not_called()
            open_project.assert_not_called()
            inbox.open.click()
            download.assert_called_once_with(project_id=b['id'])

    def test_refresh_keeps_selection_and_does_not_open_or_download(self):
        inbox = self.settle()
        a, b = self.shared(), self.shared()
        inbox.show_projects([a, b])
        inbox.projects.setCurrentRow(1)
        with patch.object(self.window, 'open_assigned_work') as download:
            inbox.show_projects([b, a])
            self.assertEqual(inbox.projects.currentItem().data(Qt.ItemDataRole.UserRole)['id'], b['id'])
            download.assert_not_called()

    def test_local_selection_survives_arrival_of_shared_project(self):
        self.load()
        inbox = self.settle()
        remote = inbox.shared_projects[0]
        inbox.show_projects([])
        self.assertEqual(inbox.copies.currentData(), str(self.source))
        inbox.show_projects([self.shared(), remote])
        self.assertEqual(inbox.copies.currentData(), str(self.source))
        self.assertEqual(inbox.projects.currentItem().data(Qt.ItemDataRole.UserRole)['id'], remote['id'])

    def test_separate_local_copies_stay_explicit_and_open_chosen_path(self):
        self.load()
        inbox = self.settle()
        other = self.base / 'second-local-copy'
        shutil.copytree(self.source, other)
        self.settings.setValue('workflow/recent_projects/' + self.coordinator['id'],
                               [str(self.source), str(other)])
        inbox.rebuild_projects()
        self.assertEqual(inbox.projects.count(), 1)
        self.assertEqual(inbox.copies.count(), 2)
        self.assertFalse(inbox.copies.isHidden())
        inbox.copies.setCurrentIndex(inbox.copies.findData(str(other)))
        inbox.rebuild_projects()
        self.assertEqual(inbox.copies.currentData(), str(other))
        with patch.object(self.window, 'open_recent_project') as open_project:
            inbox.open.click()
            open_project.assert_called_once_with(str(other))

    def test_offline_assignments_are_downloaded_snapshot_not_verified_shared_status(self):
        self.load(assigned=True)
        inbox = self.settle()
        set_working_locally(True)
        try:
            with patch('gui.assignment_workspace.assignment_presence') as discover:
                inbox.refresh()
                self.assertIn('Shared status not checked', inbox.assignment_summary.text())
                self.assertEqual(inbox.videos.topLevelItemCount(), 1)
                self.assertEqual(inbox.open.text(), 'Open My Assigned Work')
                self.assertTrue(inbox.download_more.isHidden())
                inbox.open.click()
                self.assertEqual(self.window.workflow.current, 'Annotate')
                discover.assert_not_called()
        finally:
            set_working_locally(False)

    def test_missing_local_folder_never_offers_download_as_though_shared(self):
        inbox = self.settle()
        missing = str(self.base / 'unavailable-project')
        self.settings.setValue('workflow/recent_projects/' + self.coordinator['id'], [missing])
        inbox.show_projects([])
        self.assertEqual(inbox.projects.count(), 1)
        self.assertFalse(inbox.open.isEnabled())
        self.assertEqual(inbox.open.text(), 'Open Project')
        self.assertEqual(inbox.project_status.text(), 'Local folder unavailable')

    def test_recent_archive_path_is_not_scanned_as_a_local_project(self):
        inbox = self.settle()
        self.settings.setValue('workflow/recent_projects/' + self.coordinator['id'],
                               [str(self.remote / 'project' / 'backup')])
        with patch('gui.assignment_workspace.require_contributor') as read_local:
            inbox.show_projects([])
            read_local.assert_not_called()
        self.assertEqual(inbox.projects.count(), 0)

    def test_unavailable_offline_work_cannot_download_on_item_activation(self):
        self.load(assigned=True)
        inbox = self.settle()
        (self.window.project_path / 'annotations' / 'project.json').unlink()
        set_working_locally(True)
        try:
            inbox.refresh()
            self.assertFalse(inbox.open.isEnabled())
            with patch.object(self.window, 'open_assigned_work') as download:
                inbox.open_selected()
                inbox.download_selected()
                download.assert_not_called()
        finally:
            set_working_locally(False)

    def test_late_shared_result_cannot_override_continue_locally(self):
        self.load(assigned=True)
        inbox = self.settle()
        worker = Mock(error=None, result={'status': 'assigned', 'count': 1,
                                         'projects': [self.shared()]})
        inbox.worker = worker
        set_working_locally(True)
        try:
            inbox.finished(worker, inbox.context(), self.alex)
            self.assertIsNone(inbox.result)
            self.assertEqual(inbox.shared_projects, [])
            self.assertIn('Shared status not checked', inbox.assignment_summary.text())
        finally:
            set_working_locally(False)

    def test_director_action_targets_selected_project_not_previous_open_project(self):
        self.load()
        inbox = self.settle()
        other = self.base / 'other-copy'
        shutil.copytree(self.source, other)
        self.settings.setValue('workflow/recent_projects/' + self.coordinator['id'], [str(other)])
        inbox.rebuild_projects()
        inbox.copies.setCurrentIndex(inbox.copies.findData(str(other)))
        with patch.object(self.window, 'open_recent_project',
                          side_effect=lambda path: setattr(self.window, 'project_path', Path(path))) as open_project, \
                patch.object(self.window, 'manage_assignments') as manage:
            inbox.manage.click()
            open_project.assert_called_once_with(str(other))
            manage.assert_called_once()

    def test_project_page_layout_at_desktop_and_narrow_widths(self):
        self.settle()
        page = ProjectHome(self.window.workflow)
        try:
            page.refresh()
            page.inbox.show_projects([self.shared(name='2026_CV4E'), self.shared(name='Hive masks')])
            page.inbox.summary.setText('Shared drive connected')
            page.inbox.share.show()
            for width, height in ((1100, 700), (640, 650)):
                page.resize(width, height)
                page.show()
                self.app.processEvents()
                self.app.processEvents()
                page.grab().save(f'/tmp/bumblebox-project-first-{width}.png')
                scroll = page.findChild(QScrollArea)
                self.assertEqual(scroll.horizontalScrollBar().maximum(), 0)
                expected = QBoxLayout.Direction.LeftToRight if width > 720 else QBoxLayout.Direction.TopToBottom
                self.assertEqual(page.inbox.body.direction(), expected)
                self.assertEqual(page.inbox.share.text(), 'Shared Drive Settings...')
        finally:
            page.inbox.stop()
            page.close()


if __name__ == '__main__':
    unittest.main()
