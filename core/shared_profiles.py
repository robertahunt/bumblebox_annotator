"""Small shared attribution identities, independent of projects and permissions."""

import os
import stat
from uuid import UUID

from core.contributors import contributor_profile
from core.project_sync import _read_json, _write_json, check_destination


DIRECTORY = 'contributor_profiles'


def portable_profile(value):
    return contributor_profile(dict(value, format=1, kind='bumblebox-contributor'))


def _read_profile(path):
    if path.is_symlink() or not path.is_file() or path.stat().st_size > 4096:
        raise ValueError('Profile must be a small regular JSON file.')
    profile = contributor_profile(_read_json(path))
    if path.stem != profile['id']:
        raise ValueError('Profile filename does not match its contributor ID.')
    return profile


def _create_directory(directory, root):
    if directory.exists():
        return
    # New shared metadata must not inherit a private per-user umask. Preserve
    # the archive's group and access policy; never chmod an existing directory.
    directory.mkdir()
    try:
        parent = root.stat()
        mode = stat.S_IMODE(parent.st_mode) & 0o777
        if directory.stat().st_gid != parent.st_gid:
            os.chown(directory, -1, parent.st_gid)
        directory.chmod(mode | stat.S_ISGID)
    except OSError:
        directory.rmdir()
        raise


def sync_profiles(configuration, profiles, cancelled=lambda: False):
    """Discover identities and publish only explicitly selected ones, atomically.

    Existing IDs are never overwritten or deleted. Names are not unique. Invalid
    records are reported individually so they do not hide other contributors.
    """
    outgoing = {profile['id']: profile for profile in map(portable_profile, profiles)}
    root = check_destination(configuration['root'], configuration['root_id'])
    directory = root / DIRECTORY
    if directory.is_symlink() or (directory.exists() and not directory.is_dir()):
        raise ValueError('The shared profile directory must be a regular directory.')
    warnings, published, discovered = [], [], {}
    if outgoing:
        lock = root / '.contributor-profiles.lock'
        try:
            handle = lock.open('x')
        except FileExistsError:
            warnings.append('Profile sharing is busy or was interrupted. Retry later; do not remove '
                            '.contributor-profiles.lock unless no other profile sync is running.')
        except OSError as exc:
            warnings.append(f'Profiles could not be published: {exc}')
        else:
            try:
                with handle:
                    _create_directory(directory, root)
                    for identity, profile in outgoing.items():
                        if cancelled():
                            raise InterruptedError('Profile sync cancelled.')
                        path = directory / (identity + '.json')
                        try:
                            if path.exists() or path.is_symlink():
                                if _read_profile(path) != profile:
                                    raise ValueError('This ID already has a different shared name; '
                                                     'neither identity was overwritten.')
                            else:
                                mode = 0o600 | (stat.S_IMODE(directory.stat().st_mode) & 0o044)
                                _write_json(path, profile, mode=mode)
                            published.append(identity)
                        except (OSError, ValueError, KeyError, TypeError) as exc:
                            warnings.append(f"{profile['name']} ({identity[:8]}): {exc}")
            finally:
                lock.unlink(missing_ok=True)
    if directory.exists():
        for path in sorted(directory.glob('*.json')):
            if cancelled():
                raise InterruptedError('Profile sync cancelled.')
            try:
                UUID(path.stem)
                profile = _read_profile(path)
                discovered[profile['id']] = profile
            except (OSError, ValueError, KeyError, TypeError) as exc:
                warnings.append(f'{path.name}: {exc}')
    return {'profiles': discovered, 'published': published, 'warnings': warnings}
