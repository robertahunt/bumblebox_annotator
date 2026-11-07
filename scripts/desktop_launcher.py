#!/usr/bin/env python3
"""Install and run the Linux desktop entry without depending on shell activation."""

import argparse
from contextlib import contextmanager
from datetime import datetime
import json
import logging
from logging.handlers import RotatingFileHandler
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile


APP_ID = 'org.beewhere.Annotator'
NAME = 'Bumblebox Annotator'
CONFIG_NAME = 'launcher.json'
MANAGED_ENTRY = 'X-BeeWhere-Launcher=true'


def xdg_directory(variable, fallback):
    value = os.environ.get(variable, '')
    return Path(value) if value and Path(value).is_absolute() else Path.home() / fallback


def application_environment(python):
    environment = os.environ.copy()
    environment['PATH'] = str(Path(python).parent) + os.pathsep + environment.get('PATH', os.defpath)
    environment['PYTHONUNBUFFERED'] = '1'
    environment.setdefault('NO_ALBUMENTATIONS_UPDATE', '1')
    # A desktop launched from another activated environment must use its saved
    # interpreter, not modules injected by that environment.
    environment.pop('PYTHONHOME', None)
    environment.pop('PYTHONPATH', None)
    return environment


def desktop_string(value):
    return str(value).replace('\\', '\\\\').replace('\n', '\\n').replace('\r', '\\r').replace('\t', '\\t')


def desktop_argument(value):
    # Exec quoting is not shell quoting: desktop string escapes are decoded first.
    value = str(value)
    if any(character in value for character in '\n\r\t'):
        raise ValueError('Launcher paths cannot contain tabs or newlines.')
    value = value.replace('%', '%%')
    for character in ('\\', '"', '`', '$'):
        value = value.replace(character, '\\' + character)
    return desktop_string('"' + value + '"')


def desktop_entry(runtime, helper):
    command = ' '.join(desktop_argument(value) for value in (runtime, helper, 'launch'))
    return (
        '[Desktop Entry]\nType=Application\nVersion=1.0\n'
        f'Name={NAME}\nComment=Annotate assigned videos and review masks\n'
        f'Exec={command}\nIcon=applications-graphics\nTerminal=false\n'
        'Categories=Graphics;\n'
        'Keywords=BeeWhere;bee;annotation;segmentation;masks;\n'
        'StartupNotify=false\n' + MANAGED_ENTRY + '\n'
    )


def atomic_write(path, content, mode):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=path.parent,
                                     prefix='.' + path.name, delete=False) as stream:
        temporary = Path(stream.name)
        try:
            stream.write(content)
            stream.flush()
            os.fchmod(stream.fileno(), mode)
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)


def validate_configuration(configuration):
    if configuration.get('format') != 1:
        raise ValueError('The saved launcher settings are unsupported. Reinstall the launcher.')
    for key in ('repo', 'python'):
        value = configuration.get(key)
        if not isinstance(value, str) or not Path(value).is_absolute():
            raise ValueError(f'The saved {key} path is invalid. Reinstall the launcher.')
    repo, python = Path(configuration['repo']), Path(configuration['python'])
    if not (repo / 'main.py').is_file():
        raise ValueError(f'The application folder has moved or is unavailable:\n{repo}\n\n'
                         'Ask the person who set up this computer to reinstall the launcher '
                         'from the correct checkout. Your annotations have not been removed.')
    if not python.is_file() or not os.access(python, os.X_OK):
        raise ValueError(f'The saved Python environment is unavailable:\n{python}\n\n'
                         'Ask the person who set up this computer to restore the environment '
                         'or reinstall the launcher. Your annotations have not been removed.')


def check_environment(python):
    code = (
        'import sys, importlib.util; '
        'assert sys.version_info >= (3, 10), "Python 3.10 or newer is required"; '
        'from PyQt6.QtWidgets import QApplication; '
        'names = ("numpy", "cv2", "torch", "ultralytics"); '
        'missing = [name for name in names if importlib.util.find_spec(name) is None]; '
        'assert not missing, "Missing packages: " + ", ".join(missing); '
        'print(sys.executable)'
    )
    result = subprocess.run([str(python), '-c', code], capture_output=True, text=True,
                            timeout=30, env=application_environment(python))
    if result.returncode:
        raise ValueError('This Python environment is not ready for annotation. Activate the '
                         'configured environment and follow docs/setup.md before installing '
                         'the launcher.\n' + (result.stderr or result.stdout)[-4000:])


def desktop_directory():
    executable = shutil.which('xdg-user-dir')
    if executable:
        result = subprocess.run([executable, 'DESKTOP'], capture_output=True, text=True, timeout=5)
        value = result.stdout.strip()
        if result.returncode == 0 and value and Path(value).is_absolute():
            if Path(value) == Path.home():
                raise ValueError('Desktop icons are disabled for this account. Install without --desktop.')
            return Path(value)
    return Path.home() / 'Desktop'


def install(python=None, desktop=False):
    if not sys.platform.startswith('linux'):
        raise ValueError('This launcher installer currently supports Linux only.')
    repo = Path(__file__).resolve().parent.parent
    # Do not resolve Python symlinks: a venv's bin/python may point at system Python.
    python = Path(os.path.abspath(os.path.expanduser(python or sys.executable)))
    configuration = {'format': 1, 'repo': str(repo), 'python': str(python)}
    validate_configuration(configuration)
    check_environment(python)
    runtime = Path('/usr/bin/python3')
    if not runtime.is_file():
        raise ValueError('Install the operating system Python 3 package first (/usr/bin/python3).')
    data = xdg_directory('XDG_DATA_HOME', '.local/share')
    support = data / 'bee-annotator' / 'launcher'
    helper = support / 'desktop_launcher.py'
    targets = [data / 'applications' / (APP_ID + '.desktop')]
    if desktop:
        targets.append(desktop_directory() / (APP_ID + '.desktop'))
    for target in targets:
        if target.exists() and MANAGED_ENTRY not in target.read_text(encoding='utf-8'):
            raise ValueError(f'Refusing to replace an unrelated shortcut: {target}')
    entry = desktop_entry(runtime, helper)
    # Keep the small supervisor outside the checkout, so a moved checkout or
    # missing Conda environment can still produce an actionable error dialog.
    atomic_write(helper, Path(__file__).read_text(encoding='utf-8'), 0o644)
    atomic_write(support / CONFIG_NAME, json.dumps(configuration, indent=2) + '\n', 0o600)
    for target in targets:
        atomic_write(target, entry, 0o755)
    print(f'Installed {NAME}\nApplication: {repo}\nPython: {python}')
    for target in targets:
        print(f'Shortcut: {target}')
    print('Search for Bumblebox Annotator in the applications menu, then pin it to your dock.')
    if desktop:
        print('If the desktop icon is shown as a text file, right-click it and choose Allow Launching.')
    return targets


def notify(message, error=True):
    print(message, file=sys.stderr, flush=True)
    if not (os.environ.get('DISPLAY') or os.environ.get('WAYLAND_DISPLAY')):
        return
    options = [
        ('zenity', ['--error' if error else '--info', '--no-markup', '--width=620',
                    '--title=' + NAME, '--text=' + message]),
        ('kdialog', ['--title', NAME, '--error' if error else '--msgbox', message]),
        ('xmessage', ['-center', '-title', NAME, '-buttons', 'OK:0', message]),
        ('notify-send', [NAME, message]),
    ]
    for executable, arguments in options:
        program = shutil.which(executable)
        if program:
            try:
                result = subprocess.run([program, *arguments], stdout=subprocess.DEVNULL,
                                        stderr=subprocess.DEVNULL)
                if result.returncode == 0:
                    return
            except OSError:
                continue


@contextmanager
def launch_lock(state):
    import fcntl
    with (state / 'launcher.lock').open('a') as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            yield False
            return
        try:
            yield True
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def launch(configuration_path):
    log_path = None
    handler = None
    try:
        state = xdg_directory('XDG_STATE_HOME', '.local/state') / 'bee-annotator'
        state.mkdir(parents=True, exist_ok=True, mode=0o700)
        with launch_lock(state) as acquired:
            if not acquired:
                notify('Bumblebox Annotator is already starting or open. '
                       'Check the taskbar for its window.', error=False)
                return 0
            log_path = state / 'launcher.log'
            handler = RotatingFileHandler(log_path, maxBytes=5 * 1024 * 1024,
                                          backupCount=2, encoding='utf-8')
            handler.setFormatter(logging.Formatter('%(message)s'))

            def record(message):
                handler.emit(logging.LogRecord('launcher', logging.INFO, '', 0, message, (), None))

            configuration = json.loads(Path(configuration_path).read_text(encoding='utf-8'))
            validate_configuration(configuration)
            repo, python = configuration['repo'], configuration['python']
            record(f'\nStarting {NAME}: {datetime.now().isoformat()}\nApplication: {repo}\nPython: {python}')
            with subprocess.Popen([python, '-u', str(Path(repo) / 'main.py')], cwd=repo,
                                  env=application_environment(python), stdin=subprocess.DEVNULL,
                                  stdout=subprocess.PIPE, stderr=subprocess.STDOUT) as process:
                while chunk := process.stdout.read1(64 * 1024):
                    record(chunk.decode('utf-8', errors='replace'))
                code = process.wait()
            record(f'Application exit status: {code}')
            if code != 0:
                notify(f'{NAME} could not start or closed unexpectedly (exit {code}).\n\n'
                       'Send the log below to the person who set up this computer. '
                       'Do not reinstall or delete your project.\n\nLog: ' + str(log_path))
            return code
    except (OSError, ValueError, TypeError, AttributeError) as exc:
        message = 'Could not open ' + NAME + '\n\n' + str(exc)
        if handler:
            handler.emit(logging.LogRecord('launcher', logging.ERROR, '', 0, message, (), None))
            message += '\n\nLog: ' + str(log_path)
        notify(message)
        return 1
    finally:
        if handler:
            handler.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    setup = commands.add_parser('install', help='Install for this Linux login account')
    setup.add_argument('--python', help='Annotation environment Python (defaults to this interpreter)')
    setup.add_argument('--desktop', action='store_true', help='Also create a desktop shortcut')
    commands.add_parser('launch', help='Used by the installed shortcut')
    args = parser.parse_args()
    if args.command == 'launch':
        return launch(Path(__file__).parent / CONFIG_NAME)
    try:
        install(args.python, args.desktop)
        return 0
    except (OSError, ValueError, subprocess.TimeoutExpired) as exc:
        print('Launcher installation failed: ' + str(exc), file=sys.stderr)
        return 1


if __name__ == '__main__':
    sys.exit(main())
