"""Lab onboarding, project membership, and terminal login on disposable data."""

from pathlib import Path
import sys
import unittest
from unittest.mock import Mock, patch

from PyQt6.QtCore import QSettings, QTimer, Qt
from PyQt6.QtTest import QTest
from PyQt6.QtWidgets import QApplication, QDialog, QMessageBox, QPushButton

from core.assigned_work import (add_project_member, assignment_presence, project_members,
                               read_assignments, save_assignments, transfer_director)
from core.contributors import new_session
from core.drive_login import find_kinit, login_command, start_login, terminal_login
from core.project_sync import (LOCATION_MARKERS, _write_json, check_destination, test_destination as probe)
from core.shared_profiles import sync_profiles
from gui import project_sync_dialog as preferences
from gui.drive_login import DriveLoginButton
from gui.project_people import AddContributorDialog
from gui.shared_profiles import LabConnectionDialog
from scripts import test_assigned_work as backend
from scripts import test_assignment_workflow as onboarding
from scripts import test_shared_profiles as sharing


class DriveLoginTests(unittest.TestCase):
    def which(self, name):
        return {'kinit': '/usr/bin/kinit', 'gnome-terminal': '/usr/bin/gnome-terminal'}.get(name)

    def test_direct_arguments_and_no_password_or_output_capture(self):
        with patch('core.drive_login.sys.platform', 'linux'), \
                patch('core.drive_login.shutil.which', side_effect=self.which), \
                patch('core.drive_login.subprocess.Popen') as launch:
            start_login('august@EXAMPLE.EDU')
        args, kwargs = launch.call_args
        self.assertEqual(args[0][:4], ['/usr/bin/gnome-terminal', '--wait', '--', sys.executable])
        self.assertEqual(Path(args[0][4]).name, 'drive_login.py')
        self.assertEqual(args[0][5:], ['--terminal-login', '/usr/bin/kinit', 'august@EXAMPLE.EDU'])
        self.assertNotIn('shell', kwargs)
        self.assertNotIn('input', kwargs)
        self.assertEqual(kwargs['stdin'], kwargs['stdout'])
        self.assertEqual(kwargs['stdout'], kwargs['stderr'])

    def test_principal_is_never_a_shell_command_or_option(self):
        for principal in ('', '-R', 'kinit august', 'august\npassword', 'a' * 255):
            with self.subTest(principal=principal), self.assertRaises(ValueError):
                login_command(principal)
        with patch('core.drive_login.sys.platform', 'linux'), \
                patch('core.drive_login.shutil.which', side_effect=self.which):
            self.assertEqual(login_command('$(touch;/tmp/unwanted)')[-1], '$(touch;/tmp/unwanted)')

    def test_missing_programs_and_platform_report_manual_fallback(self):
        with patch('core.drive_login.sys.platform', 'linux'), \
                patch('core.drive_login.shutil.which', return_value=None), \
                patch('core.drive_login.Path.is_file', return_value=False), self.assertRaisesRegex(ValueError, 'kinit'):
            login_command('august')
        with patch('core.drive_login.sys.platform', 'linux'), \
                patch('core.drive_login.shutil.which', side_effect=lambda name: '/bin/kinit' if name == 'kinit' else None), \
                self.assertRaisesRegex(ValueError, 'own terminal'):
            login_command('august')
        with patch('core.drive_login.sys.platform', 'win32'), self.assertRaisesRegex(ValueError, 'Linux'):
            login_command('august')

    def test_xterm_fallback_waits_for_command(self):
        with patch('core.drive_login.sys.platform', 'linux'), \
                patch('core.drive_login.shutil.which', side_effect=lambda name: '/bin/' + name if name in ('kinit', 'xterm') else None):
            command = login_command('august')
            self.assertEqual(command[:3], ['/bin/xterm', '-e', sys.executable])
            self.assertEqual(command[-3:], ['--terminal-login', '/bin/kinit', 'august'])

    def test_conda_base_is_found_without_being_on_path(self):
        expected = Path('/opt/test-conda/bin/kinit')
        with patch('core.drive_login.shutil.which', return_value=None), \
                patch('core.drive_login.sys.prefix', '/opt/test-conda/envs/bee_annotator'), \
                patch('core.drive_login.sys.base_prefix', '/opt/test-conda/envs/bee_annotator'), \
                patch.dict('core.drive_login.os.environ', {}, clear=True), \
                patch('core.drive_login.Path.is_file', lambda path: path == expected), \
                patch('core.drive_login.os.access', return_value=True):
            self.assertEqual(find_kinit(), str(expected))

    def test_conda_executable_hint_supports_external_environments(self):
        expected = Path('/opt/test-conda/bin/kinit')
        with patch('core.drive_login.shutil.which', return_value=None), \
                patch('core.drive_login.sys.prefix', '/some/external/environment'), \
                patch('core.drive_login.sys.base_prefix', '/some/external/environment'), \
                patch.dict('core.drive_login.os.environ', {'CONDA_EXE': '/opt/test-conda/bin/conda'}, clear=True), \
                patch('core.drive_login.Path.is_file', lambda path: path == expected), \
                patch('core.drive_login.os.access', return_value=True):
            self.assertEqual(find_kinit(), str(expected))

    def test_nonexecutable_fallback_is_not_used(self):
        with patch('core.drive_login.shutil.which', return_value=None), \
                patch('core.drive_login.Path.is_file', return_value=True), \
                patch('core.drive_login.os.access', return_value=False), self.assertRaisesRegex(ValueError, 'not found'):
            find_kinit()

    def test_terminal_retains_success_and_error_without_capturing_credentials(self):
        for code in (0, 1):
            with self.subTest(code=code), patch('core.drive_login.subprocess.run', return_value=Mock(returncode=code)) as run, \
                    patch('builtins.input', return_value='') as acknowledge, patch('builtins.print'):
                self.assertEqual(terminal_login('/bin/kinit', 'august'), code)
                run.assert_called_once_with(['/bin/kinit', 'august'])
                acknowledge.assert_called_once()

    def test_terminal_cancellation_and_exec_error_remain_visible(self):
        for error, expected in ((KeyboardInterrupt(), 130), (OSError('missing library'), 127)):
            with patch('core.drive_login.subprocess.run', side_effect=error), \
                    patch('builtins.input', return_value='') as acknowledge, patch('builtins.print'):
                self.assertEqual(terminal_login('/bin/kinit', 'august'), expected)
                acknowledge.assert_called_once()


class ProjectMembershipTests(unittest.TestCase):
    setUp = backend.AssignedWorkTests.setUp
    view = backend.AssignedWorkTests.view
    download = backend.AssignedWorkTests.download

    def test_legacy_roster_is_derived_without_writing(self):
        path = self.view().home / 'assignments.json'
        data = read_assignments(self.view())
        data.pop('members')
        _write_json(path, data)
        before = path.read_bytes(), path.stat().st_mtime_ns
        members = project_members(read_assignments(self.view()))
        self.assertEqual(set(members), {self.coordinator['id'], self.alex['id'], self.sam['id']})
        self.assertEqual((path.read_bytes(), path.stat().st_mtime_ns), before)

    def test_member_without_assignments_is_discoverable_with_no_download(self):
        kevin = new_session('Kevin')
        before = read_assignments(self.view())
        updated = add_project_member(self.source, self.configuration, self.coordinator, kevin, before['revision'])
        self.assertEqual(updated['videos'], before['videos'])
        self.assertEqual(updated['coordinator'], before['coordinator'])
        self.assertNotIn('session_id', updated['members'][kevin['id']])
        presence = assignment_presence(self.configuration, kevin)
        self.assertEqual(presence['status'], 'empty')
        self.assertEqual(presence['projects'], [{'id': self.project_id, 'name': self.source.name,
                                               'role': 'contributor', 'count': 0, 'videos': {}}])
        self.assertEqual(assignment_presence(self.configuration, new_session('Kevin'))['projects'], [])

    def test_enrollment_is_director_only_revision_checked_and_idempotent(self):
        kevin = new_session('Kevin')
        current = read_assignments(self.view())
        with self.assertRaisesRegex(ValueError, 'Only the project director'):
            add_project_member(self.source, self.configuration, self.alex, kevin, current['revision'])
        added = add_project_member(self.source, self.configuration, self.coordinator, kevin, current['revision'])
        with self.assertRaisesRegex(ValueError, 'Refresh'):
            add_project_member(self.source, self.configuration, self.coordinator, self.alex, current['revision'])
        same = add_project_member(self.source, self.configuration, self.coordinator, kevin, added['revision'])
        self.assertEqual(same['revision'], added['revision'])
        with self.assertRaisesRegex(ValueError, 'different name'):
            add_project_member(self.source, self.configuration, self.coordinator, dict(kevin, name='Other'), added['revision'])
        with self.assertRaisesRegex(ValueError, 'full project'):
            add_project_member(self.download(), self.configuration, self.alex, kevin, added['revision'])

    def test_membership_survives_reassignments_and_director_transfer(self):
        before = read_assignments(self.view())
        data = save_assignments(self.source, self.configuration, self.coordinator, {}, before['revision'])
        self.assertIn(self.alex['id'], project_members(data))
        new = new_session('New Director')
        data = transfer_director(self.source, self.configuration, self.coordinator, new, data['revision'])
        self.assertIn(new['id'], project_members(data))
        self.assertIn(self.coordinator['id'], project_members(data))
        self.assertEqual(assignment_presence(self.configuration, new)['projects'][0]['role'], 'director')
        self.assertEqual(assignment_presence(self.configuration, self.coordinator)['projects'][0]['role'], 'contributor')

    def test_same_name_people_are_separate_and_corrupt_roster_is_rejected(self):
        second = new_session(self.alex['name'])
        current = read_assignments(self.view())
        data = add_project_member(self.source, self.configuration, self.coordinator, second, current['revision'])
        self.assertIn(self.alex['id'], data['members'])
        self.assertIn(second['id'], data['members'])
        data['members'][second['id']]['id'] = self.alex['id']
        _write_json(self.view().home / 'assignments.json', data)
        with self.assertRaises(ValueError):
            read_assignments(self.view())


class LabConnectionGuiTests(unittest.TestCase):
    setUpClass = sharing.SharedProfileGuiTests.__dict__['setUpClass']
    setUp = sharing.SharedProfileGuiTests.setUp
    tearDown = sharing.SharedProfileGuiTests.tearDown
    sync = sharing.SharedProfileGuiTests.sync

    def connect(self):
        dialog = LabConnectionDialog()
        dialog.folder.setText(str(self.root))
        dialog.accept()
        self.assertEqual(dialog.result(), QDialog.DialogCode.Accepted)
        self.sync()
        return dialog

    def test_one_connection_preserves_backup_settings_and_publishes_no_project(self):
        preferences.remember_contributor_profile(self.alex)
        self.settings.setValue('contributors/last_id', self.alex['id'])
        destination = dict(self.configuration, include_videos=False, storage_mode='full', rollback_root='/local/rollback')
        preferences.save_setting('project_sync/destination', destination)
        preferences.save_setting('project_sync/projects', {'/project': {'enabled': False}})
        dialog = self.connect()
        saved = preferences.setting_json('project_sync/destination', {})
        self.assertEqual(saved, destination)
        self.assertEqual(preferences.setting_json('project_sync/projects', {}), {'/project': {'enabled': False}})
        profiles = preferences.setting_json('contributors/shared', {})
        self.assertEqual(profiles['root_id'], saved['root_id'])
        self.assertEqual(profiles['publish_ids'], [self.alex['id']])
        self.assertTrue(profiles['share_new'])
        self.assertFalse(any(p.name == 'manifest.json' for p in self.root.rglob('*')))
        self.assertEqual(sync_profiles(saved, [])['profiles'][self.alex['id']], self.alex)
        dialog.close()

    def test_cancel_and_failed_connection_do_not_change_settings(self):
        before = self.settings.allKeys()
        dialog = LabConnectionDialog()
        dialog.folder.setText(str(self.base / 'not-mounted'))
        dialog.accept()
        self.assertEqual(dialog.result(), QDialog.DialogCode.Rejected)
        self.assertEqual(self.settings.allKeys(), before)
        self.assertFalse((self.base / 'not-mounted').exists())
        dialog.reject()

    def test_different_existing_root_needs_confirmation(self):
        previous = self.base / 'previous'
        previous.mkdir()
        destination = probe(previous)
        preferences.save_setting('project_sync/destination', destination)
        preferences.save_setting('contributors/shared', self.configuration)
        dialog = LabConnectionDialog()
        with patch('gui.shared_profiles.QMessageBox.question', return_value=QMessageBox.StandardButton.Cancel):
            dialog.accept()
        self.assertEqual(preferences.setting_json('project_sync/destination', {}), destination)
        self.assertEqual(dialog.result(), QDialog.DialogCode.Rejected)
        with patch('gui.shared_profiles.QMessageBox.question', return_value=QMessageBox.StandardButton.Yes):
            dialog.accept()
        self.assertEqual(preferences.setting_json('project_sync/destination', {})['root_id'], self.configuration['root_id'])
        dialog.close()

    def test_new_profile_auto_discovery_on_second_computer(self):
        dialog = self.connect()
        kevin = sharing.profile('Kevin')
        preferences.remember_contributor_profile(kevin, newly_created=True)
        self.sync()
        self.assertIn(kevin['id'], sync_profiles(self.configuration, [])['profiles'])
        self.settings = QSettings(str(self.base / 'computer-b.ini'), QSettings.Format.IniFormat)
        second = self.connect()
        self.assertIn(kevin['id'], preferences.load_contributor_profiles())
        self.assertIsNone(self.settings.value('contributors/last_id'))
        dialog.close()
        second.close()

    def test_disconnect_works_offline_without_deleting_anything(self):
        dialog = self.connect()
        dialog.close()
        self.root.rename(self.base / 'offline')
        dialog = LabConnectionDialog()
        with patch('gui.shared_profiles.QMessageBox.question', return_value=QMessageBox.StandardButton.Yes):
            dialog.disconnect_lab()
        self.assertFalse(preferences.setting_json('contributors/shared', {})['enabled'])
        self.assertNotIn('root_id', preferences.setting_json('project_sync/destination', {}))
        self.assertTrue((self.base / 'offline').exists())
        dialog.close()

    def test_login_explicit_cancel_failure_and_actual_retry(self):
        button = DriveLoginButton()
        retry = Mock()
        button.completed.connect(retry)
        with patch('gui.drive_login.start_login') as launch, \
                patch('gui.drive_login.QInputDialog.getText', return_value=('', False)):
            button.click()
            launch.assert_not_called()
        process = Mock(returncode=0)
        process.poll.return_value = None
        with patch('gui.drive_login.start_login', return_value=process) as launch, \
                patch('gui.drive_login.QInputDialog.getText', return_value=('kerberos-user', True)):
            button.click()
            button.sign_in()
            launch.assert_called_once_with('kerberos-user')
        self.assertFalse(button.isEnabled())
        button.poll()
        retry.assert_not_called()
        process.poll.return_value = 0
        button.poll()
        retry.assert_called_once()
        self.assertTrue(button.isEnabled())
        self.assertEqual(preferences.setting_json('lab/kerberos_principal', ''), 'kerberos-user')
        button.close()

    def test_background_failure_never_spawns_login(self):
        preferences.save_setting('contributors/shared', dict(self.configuration, root=str(self.base / 'missing')))
        with patch('gui.drive_login.start_login') as launch:
            self.sync()
            launch.assert_not_called()
        self.assertIn('unavailable', self.service.status)

    def test_new_person_editor_does_not_change_active_identity_or_save_on_cancel(self):
        preferences.remember_contributor_profile(self.alex)
        self.settings.setValue('contributors/last_id', self.alex['id'])
        dialog = AddContributorDialog({self.alex['id']: self.alex})
        dialog.tabs.setCurrentIndex(1)
        dialog.name.setText('Kevin')
        dialog.accept()
        self.assertEqual(dialog.profile['name'], 'Kevin')
        self.assertEqual(self.settings.value('contributors/last_id'), self.alex['id'])
        self.assertNotIn(dialog.profile['id'], preferences.load_contributor_profiles())
        cancelled = AddContributorDialog({})
        cancelled.tabs.setCurrentIndex(1)
        cancelled.name.setText('Unwanted')
        cancelled.reject()
        self.assertIsNone(cancelled.profile)
        self.assertEqual(len(preferences.load_contributor_profiles()), 1)

    def test_duplicate_name_does_not_replace_existing_profile(self):
        preferences.remember_contributor_profile(self.alex)
        dialog = AddContributorDialog({})
        dialog.tabs.setCurrentIndex(1)
        dialog.name.setText(self.alex['name'])
        with patch('gui.project_people.QMessageBox.question', return_value=QMessageBox.StandardButton.No):
            dialog.accept()
        self.assertIsNone(dialog.profile)
        with patch('gui.project_people.QMessageBox.question', return_value=QMessageBox.StandardButton.Yes):
            dialog.accept()
        self.assertNotEqual(dialog.profile['id'], self.alex['id'])
        self.assertEqual(preferences.load_contributor_profiles(), {self.alex['id']: self.alex})

    def test_connection_and_people_layouts_at_small_sizes(self):
        preferences.remember_contributor_profile(self.alex)
        for dialog, filename in ((LabConnectionDialog(), 'connect'), (AddContributorDialog({}), 'add-person')):
            dialog.resize(640, 600)
            dialog.show()
            self.app.processEvents()
            self.assertLessEqual(dialog.minimumSizeHint().width(), 640)
            dialog.grab().save('/tmp/bumblebox-lab-' + filename + '.png')
            dialog.reject()

    def test_explicit_connection_result_remains_until_acknowledged(self):
        for valid in (True, False):
            dialog = LabConnectionDialog()
            folder = self.root if valid else self.base / 'not-mounted'
            dialog.folder.setText(str(folder))
            observed = {}
            def dismiss_result():
                box = QApplication.activeModalWidget()
                if not isinstance(box, QMessageBox):
                    QTimer.singleShot(5, dismiss_result)
                    return
                observed.update(title=box.windowTitle(), text=box.text(), visible=box.isVisible())
                box.grab().save('/tmp/bumblebox-connection-result-' + str(valid) + '.png')
                box.accept()
            QTimer.singleShot(0, dismiss_result)
            next(button for button in dialog.findChildren(QPushButton) if button.text() == 'Test Connection').click()
            self.assertTrue(observed['visible'])
            self.assertEqual(observed['title'], 'Connection Test Passed' if valid else 'Connection Test Failed')
            self.assertIn(str(folder), observed['text'])
            self.assertEqual(dialog.status.text(), observed['text'])
            self.assertEqual(dialog.result(), QDialog.DialogCode.Rejected)
            dialog.reject()

    def test_login_completion_requests_visible_result(self):
        dialog = LabConnectionDialog()
        with patch.object(dialog, 'test_connection') as check:
            dialog.login.completed.emit()
        check.assert_called_once_with(report=True)
        dialog.reject()

    def test_project_subfolder_is_rejected_without_writes_even_with_existing_root_marker(self):
        child = self.root / 'project--id'
        child.mkdir()
        nested_configuration = probe(child)
        _write_json(child / LOCATION_MARKERS['project'], {'name': 'project', 'id': 'existing'})
        before = {p: p.read_bytes() for p in child.iterdir()}
        for operation in (lambda: probe(child), lambda: check_destination(child, nested_configuration['root_id'])):
            with self.assertRaisesRegex(ValueError, 'published project'):
                operation()
        self.assertEqual(before, {p: p.read_bytes() for p in child.iterdir()})
        preferences.save_setting('project_sync/destination', self.configuration)
        preferences.save_setting('contributors/shared', dict(nested_configuration, enabled=True))
        dialog = LabConnectionDialog()
        self.assertFalse(dialog.test_connection())
        self.assertIn(str(self.root), dialog.status.text())
        dialog.use_project_folder.click()
        self.assertEqual(dialog.folder.text(), str(self.root))
        self.assertTrue(dialog.test_connection())
        # Choosing/testing is not consent to change either saved connection.
        self.assertEqual(preferences.setting_json('contributors/shared', {})['root'], str(child))
        dialog.reject()


class ProjectPeopleGuiTests(unittest.TestCase):
    setUpClass = onboarding.AssignmentOnboardingGuiTests.__dict__['setUpClass']
    setUp = onboarding.AssignmentOnboardingGuiTests.setUp
    settle = onboarding.AssignmentOnboardingGuiTests.settle
    settle_assignments = onboarding.AssignmentOnboardingGuiTests.settle_assignments
    load = onboarding.AssignmentOnboardingGuiTests.load
    view = backend.AssignedWorkTests.view
    download = backend.AssignedWorkTests.download

    def tearDown(self):
        from gui.shared_profiles import stop_profile_sync
        stop_profile_sync()
        onboarding.AssignmentOnboardingGuiTests.tearDown(self)

    def test_director_adds_kevin_and_second_computer_discovers_without_switching(self):
        preferences.save_setting('contributors/shared', dict(self.configuration, enabled=True, share_new=True, publish_ids=[]))
        flow = self.load()
        self.settle()
        flow.activate('Assignments')
        self.settle_assignments()
        panel = flow.assignments_page.panel
        session = dict(self.window.contributor_session)
        kevin = new_session('Kevin')
        before = read_assignments(self.view())
        panel.enroll(kevin)
        from gui.shared_profiles import profile_sync_service
        service = profile_sync_service()
        for _ in range(500):
            QTest.qWait(10)
            if service.worker is None and not service.again:
                break
        self.assertIsNone(service.worker)
        self.assertEqual(self.window.contributor_session, session)
        self.assertEqual(read_assignments(self.view())['videos'], before['videos'])
        self.assertIn(kevin['id'], sync_profiles(self.configuration, [])['profiles'])
        actor, _ = panel.controls['alpha']
        self.assertGreaterEqual(actor.findData(kevin['id']), 0)
        self.window.show()
        self.app.processEvents()
        flow.assignments_page.grab().save('/tmp/bumblebox-lab-people.png')

    def test_home_lists_member_projects_and_keeps_other_peoples_projects_out(self):
        kevin = new_session('Kevin')
        data = read_assignments(self.view())
        add_project_member(self.source, self.configuration, self.coordinator, kevin, data['revision'])
        self.window.set_contributor(kevin)
        inbox = self.settle()
        self.assertEqual(inbox.projects.count(), 1)
        self.assertFalse(inbox.open.isEnabled())
        if inbox.notice:
            inbox.notice.close()
        self.window.show()
        self.app.processEvents()
        self.window.workflow.home.grab().save('/tmp/bumblebox-lab-my-work.png')
        self.window.set_contributor(new_session('Kevin'))
        inbox = self.settle()
        self.assertEqual(inbox.projects.count(), 0)

    def test_recent_projects_are_per_person_and_existing_local_copy_opens(self):
        flow = self.load()
        inbox = self.settle()
        self.assertEqual(inbox.projects.count(), 1)
        self.assertEqual(inbox.open.text(), 'Open Local Project')
        with patch.object(flow, 'activate') as activate:
            inbox.open_selected()
            activate.assert_called_once_with('Annotate')
        self.window.project_path = None
        self.window.set_contributor(self.alex)
        self.settle()
        flow.home.refresh()
        self.assertEqual(flow.home.recent.count(), 0)
        self.window.set_contributor(self.coordinator)
        self.settle()
        flow.home.refresh()
        self.assertEqual(flow.home.recent.count(), 1)

    def test_open_my_work_in_full_project_clears_filters_and_selects_assigned_frame(self):
        choices = dict(self.choices)
        choices['beta'] = {'contributor': self.coordinator, 'frames': [1, 2], 'task': 'hive'}
        save_assignments(self.source, self.configuration, self.coordinator, choices,
                         read_assignments(self.view())['revision'])
        flow = self.load()
        inbox = self.settle()
        w = self.window
        w.load_video_frames('alpha')
        w.video_browser.experiment.addItem('Unrelated experiment', 'other')
        w.video_browser.experiment.setCurrentIndex(2)
        w.video_split_filter_combo.setCurrentText('Validation Videos')
        w.split_filter_combo.setCurrentText('Validation Frames')
        w.annotation_work.filters.show.setCurrentIndex(3)
        w.annotation_work.filters.presence.setCurrentIndex(2)
        w.tracking_filter_checkbox.setChecked(True)
        w.video_dock.hide()
        w.frame_dock.hide()
        flow.show_home()
        self.settle()
        self.assertEqual(inbox.open.text(), 'Open My Assigned Work')
        with patch.object(w, 'open_assigned_work') as download:
            inbox.open_selected()
            download.assert_not_called()
        self.assertEqual(flow.current, 'Annotate')
        self.assertEqual(w.current_video_id, 'beta')
        self.assertEqual(w._get_frame_idx_in_video(w.current_frame_idx), 1)
        self.assertFalse(w.video_dock.isHidden())
        self.assertFalse(w.frame_dock.isHidden())
        self.assertTrue(w.video_browser.only_mine.isChecked())
        visible = lambda: {w.video_list.item(i).data(Qt.ItemDataRole.UserRole)
                           for i in range(w.video_list.count()) if not w.video_list.item(i).isHidden()}
        self.assertEqual(visible(), {'beta'})
        self.assertEqual(w.annotation_work.filters.task.currentData(), 'hive')
        self.assertEqual(w.annotation_work.filters.show.currentData(), 'all')
        self.assertEqual(w.video_list.currentItem().data(Qt.ItemDataRole.UserRole), 'beta')
        from core.assigned_work import workspace
        self.assertIsNone(workspace(self.source))  # A view filter is not an assignment download.
        w.video_browser.only_mine.setChecked(False)
        self.assertEqual(visible(), {'alpha', 'beta'})
        w.video_browser.only_mine.setChecked(True)
        w.show()
        self.app.processEvents()
        w.grab().save('/tmp/bumblebox-my-assigned-work.png')
        flow.show_home()
        self.settle()
        flow.home.grab().save('/tmp/bumblebox-my-work-home.png')

    def test_assigned_copy_opens_annotate_even_when_last_workspace_is_home_or_train(self):
        flow = self.load(assigned=True)
        inbox = self.settle()
        flow.show_home()
        self.settle()
        self.settings.setValue(flow.key('workspace'), 'Train')
        inbox.open_selected()
        self.assertEqual(flow.current, 'Annotate')
        self.assertEqual(self.window.current_video_id, 'alpha')
        self.assertTrue(self.window.video_browser.only_mine.isChecked())

    def test_missing_or_wrong_person_cached_copy_offers_download(self):
        local = self.download()
        self.window.set_contributor(self.sam)
        inbox = self.settle()
        key = f'workflow/shared_projects/{self.sam["id"]}/{self.configuration["root_id"]}/{self.project_id}'
        for path in (str(self.base / 'missing'), str(local)):
            self.settings.setValue(key, path)
            inbox.selection_changed()
            self.assertEqual(inbox.open.text(), 'Download Assigned Work...')
            with patch.object(self.window, 'open_assigned_work') as download:
                inbox.open_selected()
                download.assert_called_once_with(project_id=self.project_id)

    def test_new_assignments_not_in_selective_copy_keep_existing_work_and_offer_download(self):
        flow = self.load(assigned=True)
        choices = dict(self.choices)
        choices['beta'] = {'contributor': self.alex, 'frames': [1], 'task': 'hive'}
        save_assignments(self.source, self.configuration, self.coordinator, choices,
                         read_assignments(self.view())['revision'])
        flow.show_home()
        inbox = self.settle()
        before = {p: p.read_bytes() for p in self.window.project_path.glob('annotations/png/*/*.png')}
        with patch.object(QMessageBox, 'information') as notice:
            inbox.open_selected()
        self.assertIn('Some Assigned Work', notice.call_args.args[1])
        self.assertEqual(self.window.current_video_id, 'alpha')
        self.assertEqual(before, {p: p.read_bytes() for p in before})
        self.assertFalse(inbox.download_more.isHidden())
        with patch.object(self.window, 'open_assigned_work') as download:
            inbox.download_selected()
            download.assert_called_once_with(project_id=self.project_id)

    def test_reassigned_local_copy_does_not_silently_open_stale_work(self):
        flow = self.load(assigned=True)
        choices = dict(self.choices)
        choices['alpha'] = {'contributor': self.alex, 'frames': [2], 'task': 'hive'}
        save_assignments(self.source, self.configuration, self.coordinator, choices,
                         read_assignments(self.view())['revision'])
        flow.show_home()
        inbox = self.settle()
        with patch.object(QMessageBox, 'information') as notice:
            inbox.open_selected()
        self.assertEqual(flow.current, 'Home')
        self.assertEqual(notice.call_args.args[1], 'Assigned Work Not Available Locally')

    def test_switching_profile_clears_my_work_view(self):
        self.load()
        browser = self.window.video_browser
        browser.set_my_work(self.source, {'alpha': {}}, enabled=True)
        self.window.set_contributor(self.sam)
        self.settle()
        self.assertFalse(browser.only_mine.isChecked())
        self.assertEqual(browser.my_work, {})

    def test_local_read_error_is_reported_without_crashing_the_menu(self):
        self.load(assigned=True)
        inbox = self.settle()
        project = inbox.projects.currentItem().data(Qt.ItemDataRole.UserRole)
        with patch('gui.assignment_workspace.require_contributor', side_effect=OSError('Disk unavailable')), \
                patch.object(QMessageBox, 'warning') as warning:
            inbox.show_assigned_work(project)
        self.assertEqual(warning.call_args.args[1:], ('Assigned Work Unavailable', 'Disk unavailable'))

    def test_unreadable_other_projects_dont_hide_current_project_director(self):
        self.load()
        inbox = self.settle()
        def probe(configuration, actor, project=None, cancelled=lambda: False):
            if project is None:
                raise OSError('Other archive is unreadable')
            return assignment_presence(configuration, actor, project, cancelled)
        with patch('gui.assignment_workspace.assignment_presence', side_effect=probe):
            inbox.refresh()
            self.settle()
        self.assertEqual(inbox.result['status'], 'director')
        self.assertIn('Other shared projects', inbox.summary.text())
        self.assertFalse(inbox.manage.isHidden())
        self.assertIsNone(inbox.notice)


if __name__ == '__main__':
    unittest.main()
