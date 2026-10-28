"""Launch Kerberos in an external terminal; never handle credentials in the app."""

import os
from pathlib import Path
import shutil
import subprocess
import sys


def find_kinit():
    """Desktop launchers may omit both system bins and the parent Conda install."""
    executable = shutil.which('kinit')
    if executable:
        return executable
    candidates = [Path('/usr/bin/kinit'), Path('/usr/local/bin/kinit'), Path('/bin/kinit')]
    prefixes = [Path(sys.prefix), Path(sys.base_prefix)]
    conda = os.environ.get('CONDA_EXE')
    if conda and Path(conda).is_absolute():
        prefixes.append(Path(conda).parent.parent)
    for prefix in prefixes:
        candidates.append(prefix / 'bin/kinit')
        if prefix.parent.name == 'envs':
            candidates.append(prefix.parent.parent / 'bin/kinit')
    for candidate in dict.fromkeys(candidates):
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return str(candidate)
    raise ValueError('kinit was not found on PATH, in standard system locations, or in this '
                     'Python/Conda installation. You can run your working kinit command in a '
                     'separate terminal, then use Test Connection. No reinstall is needed if '
                     'that command already works.')


def login_command(principal):
    principal = principal.strip()
    if (not principal or principal.startswith('-') or len(principal) > 254 or
            any(character.isspace() or ord(character) < 32 for character in principal)):
        raise ValueError('Enter your Kerberos username, optionally including @REALM, not a command.')
    if not sys.platform.startswith('linux'):
        raise ValueError('Terminal sign-in is available on Linux. Sign in using your system tools, then retry.')
    kinit = find_kinit()
    command = [sys.executable, str(Path(__file__).resolve()), '--terminal-login', kinit, principal]
    terminal = shutil.which('gnome-terminal')
    if terminal:
        return [terminal, '--wait', '--', *command]
    terminal = shutil.which('xterm')
    if terminal:
        return [terminal, '-e', *command]
    raise ValueError('No supported terminal found (GNOME Terminal or xterm). Run kinit in your own terminal, then retry.')


def start_login(principal):
    # The terminal creates its own TTY. No password, ticket, or terminal output
    # passes through our process, and the username is never interpreted by a shell.
    return subprocess.Popen(login_command(principal), stdin=subprocess.DEVNULL,
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                            start_new_session=True)


def terminal_login(executable, principal):
    """Runs inside the new terminal, inheriting its TTY, never piping credentials."""
    print('BeeWhere research drive sign-in', flush=True)
    print('Kerberos program: ' + executable, flush=True)
    try:
        code = subprocess.run([executable, principal]).returncode
    except KeyboardInterrupt:
        code = 130
        print('\nSign-in cancelled.', flush=True)
    except OSError as exc:
        code = 127
        print('Could not start Kerberos: ' + str(exc), flush=True)
    if code == 0:
        print('\nKerberos sign-in completed. Drive access still needs to be checked.', flush=True)
    else:
        print(f'\nKerberos did not complete (exit {code}). Read the error above before closing.', flush=True)
    try:
        input('Press Enter to close this terminal and recheck the drive in BeeWhere... ')
    except (EOFError, KeyboardInterrupt):
        pass
    return code


if __name__ == '__main__':
    if len(sys.argv) != 4 or sys.argv[1] != '--terminal-login':
        raise SystemExit('Use Sign In to Drive in BeeWhere to open this helper.')
    raise SystemExit(terminal_login(sys.argv[2], sys.argv[3]))
