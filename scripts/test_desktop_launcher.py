"""Desktop setup and process supervision, using disposable fake applications."""

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

from scripts import desktop_launcher as launcher


class DesktopLauncherTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix='annotator launcher ')
        self.addCleanup(self.temporary.cleanup)
        self.base = Path(self.temporary.name)
        self.data = self.base / 'app data'
        self.state = self.base / 'state'
        self.desktop = self.base / 'Desktop'
        self.repo = self.base / 'checkout with spaces'
        (self.repo / 'scripts').mkdir(parents=True)
        self.helper = self.repo / 'scripts/desktop_launcher.py'
        self.helper.write_text(Path(launcher.__file__).read_text(encoding='utf-8'), encoding='utf-8')
        (self.repo / 'main.py').write_text('print("APP STARTED")\n', encoding='utf-8')
        self.configuration = {'format': 1, 'repo': str(self.repo), 'python': sys.executable}
        self.config_path = self.base / 'launcher.json'
        self.config_path.write_text(json.dumps(self.configuration), encoding='utf-8')
        self.addCleanup(patch.stopall)
        patch.dict(os.environ, {'XDG_DATA_HOME': str(self.data), 'XDG_STATE_HOME': str(self.state)}).start()

    def install(self, **kwargs):
        with patch.object(launcher, '__file__', str(self.helper)), \
                patch.object(launcher, 'check_environment') as check, \
                patch.object(launcher, 'desktop_directory', return_value=self.desktop):
            paths = launcher.install(**kwargs)
            check.assert_called_once()
            return paths

    @property
    def log(self):
        return self.state / 'bee-annotator/launcher.log'

    def test_install_and_reinstall_use_one_shortcut_and_saved_checkout(self):
        targets = self.install(desktop=True)
        self.assertEqual(targets, self.install(desktop=True))
        self.assertEqual(len(list((self.data / 'applications').glob('*.desktop'))), 1)
        self.assertEqual(targets[0].read_bytes(), targets[1].read_bytes())
        config = json.loads((self.data / 'bee-annotator/launcher/launcher.json').read_text())
        self.assertEqual(config, self.configuration)
        self.assertIn('Terminal=false', targets[0].read_text())
        self.assertIn('/usr/bin/python3', targets[0].read_text())
        self.assertEqual(targets[0].stat().st_mode & 0o777, 0o755)
        validator = shutil.which('desktop-file-validate')
        if validator:
            result = subprocess.run([validator, str(targets[0])], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_python_symlink_is_not_resolved_out_of_virtual_environment(self):
        python = self.base / 'venv/bin/python'
        python.parent.mkdir(parents=True)
        python.symlink_to(sys.executable)
        self.install(python=str(python))
        config = json.loads((self.data / 'bee-annotator/launcher/launcher.json').read_text())
        self.assertEqual(config['python'], str(python))

    def test_desktop_service_executes_quoted_paths_literally(self):
        try:
            import gi
            gi.require_version('Gio', '2.0')
            from gi.repository import Gio
        except ImportError:
            self.skipTest('System Gio bindings are not installed')
        special = self.base / 'app %f $HOME "quote" \\ backtick` apostrophe\''
        with patch.dict(os.environ, {'XDG_DATA_HOME': str(special)}):
            entry = self.install()[0]
            app = Gio.DesktopAppInfo.new_from_filename(str(entry))
            self.assertIsNotNone(app)
            self.assertTrue(app.launch([], None))
            deadline = time.monotonic() + 5
            while time.monotonic() < deadline:
                if self.log.exists() and 'Application exit status: 0' in self.log.read_text():
                    break
                time.sleep(0.02)
            else:
                self.fail('Desktop launcher did not finish the fake app')
            self.assertIn('APP STARTED', self.log.read_text())

    def test_does_not_replace_unrelated_launcher(self):
        target = self.data / 'applications' / (launcher.APP_ID + '.desktop')
        target.parent.mkdir(parents=True)
        target.write_text('Unrelated shortcut')
        with self.assertRaisesRegex(ValueError, 'unrelated shortcut'):
            self.install()
        self.assertEqual(target.read_text(), 'Unrelated shortcut')
        self.assertFalse((self.data / 'bee-annotator/launcher/launcher.json').exists())

    def test_missing_dependencies_do_not_install_shortcut(self):
        with patch.object(launcher, '__file__', str(self.helper)), \
                patch.object(launcher, 'check_environment', side_effect=ValueError('Missing PyQt6')):
            with self.assertRaisesRegex(ValueError, 'Missing PyQt6'):
                launcher.install()
        self.assertFalse((self.data / 'applications').exists())

    def test_environment_check_does_not_run_cuda_or_import_models(self):
        with patch.object(launcher.subprocess, 'run') as run:
            run.return_value.returncode = 0
            launcher.check_environment(sys.executable)
        command = run.call_args.args[0]
        self.assertNotIn('cuda', command[-1])
        self.assertNotIn('import torch', command[-1])
        self.assertEqual(command[0], sys.executable)

    def test_desktop_paths_are_escaped_without_a_shell(self):
        self.assertEqual(launcher.desktop_argument('a b'), '"a b"')
        self.assertEqual(launcher.desktop_argument('%f'), '"%%f"')
        self.assertEqual(launcher.desktop_argument('$x'), '"\\\\$x"')
        self.assertEqual(launcher.desktop_argument('a\\b'), '"a\\\\\\\\b"')
        self.assertEqual(launcher.desktop_argument('a"b'), '"a\\\\"b"')
        with self.assertRaises(ValueError):
            launcher.desktop_argument('bad\npath')

    def test_relative_xdg_path_is_ignored(self):
        with patch.dict(os.environ, {'XDG_DATA_HOME': 'relative'}):
            self.assertEqual(launcher.xdg_directory('XDG_DATA_HOME', '.local/share'),
                             Path.home() / '.local/share')

    def test_launch_uses_saved_interpreter_and_cwd_not_terminal_environment(self):
        (self.repo / 'main.py').write_text(
            'import os,sys,json\nfrom pathlib import Path\n'
            'Path("observed.json").write_text(json.dumps({"cwd": os.getcwd(), '
            '"python": sys.executable, "path": os.environ["PATH"], '
            '"injected": os.environ.get("PYTHONPATH"), "args": sys.argv[1:]}))\n'
            'print("APP STARTED")\n', encoding='utf-8')
        with patch.dict(os.environ, {'PYTHONPATH': '/wrong', 'PYTHONHOME': '/wrong'}), \
                patch.object(launcher, 'notify') as notify:
            self.assertEqual(launcher.launch(self.config_path), 0)
            notify.assert_not_called()
        observed = json.loads((self.repo / 'observed.json').read_text())
        self.assertEqual(observed['cwd'], str(self.repo))
        self.assertEqual(observed['python'], sys.executable)
        self.assertEqual(observed['path'].split(os.pathsep)[0], str(Path(sys.executable).parent))
        self.assertIsNone(observed['injected'])
        self.assertEqual(observed['args'], [])  # No models, projects or identities auto-selected.
        self.assertIn('APP STARTED', self.log.read_text())

    def test_process_failure_keeps_traceback_and_names_log_in_dialog(self):
        (self.repo / 'main.py').write_text('raise RuntimeError("Synthetic startup failure")\n')
        with patch.object(launcher, 'notify') as notify:
            self.assertEqual(launcher.launch(self.config_path), 1)
        self.assertIn(str(self.log), notify.call_args.args[0])
        self.assertIn('Synthetic startup failure', self.log.read_text())

    def test_missing_environment_and_moved_checkout_have_actionable_errors(self):
        for field, value, message in (
                ('python', str(self.base / 'missing/bin/python'), 'environment is unavailable'),
                ('repo', str(self.base / 'moved-checkout'), 'folder has moved')):
            with self.subTest(field=field):
                self.config_path.write_text(json.dumps(dict(self.configuration, **{field: value})))
                with patch.object(launcher, 'notify') as notify:
                    self.assertEqual(launcher.launch(self.config_path), 1)
                self.assertIn(message, notify.call_args.args[0])

    def test_bad_configuration_reports_error(self):
        for content in ('{broken', '[]', '{"format":99}'):
            self.config_path.write_text(content)
            with patch.object(launcher, 'notify') as notify:
                self.assertEqual(launcher.launch(self.config_path), 1)
            self.assertIn('Could not open', notify.call_args.args[0])

    def test_second_launch_does_not_start_another_process(self):
        state = self.state / 'bee-annotator'
        state.mkdir(parents=True)
        with launcher.launch_lock(state), patch.object(launcher, 'notify') as notify, \
                patch.object(launcher.subprocess, 'Popen') as popen:
            self.assertEqual(launcher.launch(self.config_path), 0)
            popen.assert_not_called()
            self.assertIn('already starting or open', notify.call_args.args[0])
        with launcher.launch_lock(state) as acquired:
            self.assertTrue(acquired)

    def test_logs_are_bounded(self):
        (self.repo / 'main.py').write_text('print("x" * (18 * 1024 * 1024))\n')
        with patch.object(launcher, 'notify') as notify:
            self.assertEqual(launcher.launch(self.config_path), 0)
            notify.assert_not_called()
        logs = list(self.log.parent.glob('launcher.log*'))
        self.assertEqual(len(logs), 3)
        self.assertLessEqual(sum(path.stat().st_size for path in logs), 15 * 1024 * 1024)

    def test_unwritable_state_location_is_not_silent(self):
        self.state.write_text('not a directory')
        with patch.object(launcher, 'notify') as notify:
            self.assertEqual(launcher.launch(self.config_path), 1)
        self.assertIn('Could not open', notify.call_args.args[0])


if __name__ == '__main__':
    unittest.main()
