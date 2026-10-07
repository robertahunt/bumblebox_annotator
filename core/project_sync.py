"""Optional one-way publishing of complete, versioned annotation projects.

No authentication or mounting is performed here. The destination must already be
accessible. A completed revision is published only after all files are verified.
"""

from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import tempfile
import unicodedata
from uuid import UUID, uuid4


ROOT_MARKER = '.bumblebox-sync-root.json'
PROJECT_MARKER = 'project_sync.json'
LOCATION_MARKERS = {'project': '.bumblebox-project.json',
                    'contributor': '.bumblebox-contributor.json'}


def _read_json(path):
    with Path(path).open() as handle:
        return json.load(handle)


def _write_json(path, value):
    path = Path(path)
    with tempfile.NamedTemporaryFile(mode='w', dir=path.parent, prefix='.tmp-',
                                     delete=False) as handle:
        temporary = Path(handle.name)
        try:
            json.dump(value, handle, indent=2)
            handle.flush()
            os.fsync(handle.fileno())
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _uuid(value):
    return str(UUID(value))


def check_destination(folder, expected_id):
    root = Path(folder).expanduser()
    if not root.is_dir():
        raise ValueError('Sync folder is unavailable. Connect or mount the drive, then try again.')
    marker = root / ROOT_MARKER
    if marker.is_symlink() or not marker.is_file():
        raise ValueError('Sync destination marker is missing. Reconnect the configured drive; '
                         'no replacement folder was created.')
    if _read_json(marker).get('id') != expected_id:
        raise ValueError('This is not the configured sync destination. Test it again in Settings.')
    return root.resolve()


def test_destination(folder, expected_id=None):
    """Test only a uniquely named probe; never remove existing user files."""
    if not str(folder).strip():
        raise ValueError('Choose a destination folder first.')
    root = Path(folder).expanduser()
    if not root.is_dir():
        raise ValueError('Choose an existing, accessible folder. Network drives must be mounted first.')
    root = root.resolve()
    if expected_id:
        check_destination(root, expected_id)
    payload = os.urandom(64)
    probe = None
    try:
        with tempfile.NamedTemporaryFile(dir=root, prefix='.bumblebox-test-', delete=False) as handle:
            probe = Path(handle.name)
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        if probe.read_bytes() != payload:
            raise OSError('The sync destination failed its read-back test.')
    finally:
        if probe is not None:
            probe.unlink()
    marker = root / ROOT_MARKER
    if marker.is_symlink():
        raise ValueError('The destination marker must not be a symbolic link.')
    if not marker.exists():
        try:
            with marker.open('x') as handle:
                json.dump({'id': str(uuid4()), 'format': 1}, handle)
        except FileExistsError:
            pass
    identity = _uuid(_read_json(marker)['id'])
    return {'root': str(root), 'root_id': identity}


def ensure_project_identity(project):
    project = Path(project).resolve()
    if not (project / 'annotations/project.json').is_file():
        raise ValueError('The source is not an annotation project.')
    marker = project / 'annotations' / PROJECT_MARKER
    if marker.is_symlink():
        raise ValueError('Project identity must not be a symbolic link.')
    if not marker.exists():
        try:
            with marker.open('x') as handle:
                json.dump({'project_id': str(uuid4()), 'format': 1}, handle, indent=2)
        except FileExistsError:
            pass
    return _uuid(_read_json(marker)['project_id'])


def _safe_child(root, relative, mkdir=False):
    relative = Path(relative)
    if relative.is_absolute() or '..' in relative.parts:
        raise ValueError('Unsafe path in sync metadata.')
    path = root
    for part in relative.parts:
        path = path / part
        if path.is_symlink():
            raise ValueError(f'Sync refuses symbolic links: {path}')
        if mkdir and not path.exists():
            path.mkdir(exist_ok=True)
        if mkdir and (path.is_symlink() or not path.is_dir()):
            raise ValueError(f'Sync requires an ordinary directory: {path}')
    return path


def _folder_label(name, fallback):
    text = unicodedata.normalize('NFKD', str(name)).encode('ascii', 'ignore').decode()
    text = re.sub(r'[^A-Za-z0-9._-]+', '-', text).strip('._-')[:64].rstrip('._-')
    return text or fallback


def _identity_directory(parent, name, identity, kind):
    """Find by full ID, not display name; migrate UUID-only directories in place."""
    marker_name = LOCATION_MARKERS[kind]
    legacy = _safe_child(parent, identity)
    matches = []
    for path in parent.iterdir():
        if path.is_symlink() or not path.is_dir():
            continue
        marker = _safe_child(path, marker_name)
        if marker.exists():
            stored_id = _uuid(_read_json(marker)['id'])
            if path == legacy and stored_id != identity:
                raise ValueError('The legacy sync folder belongs to a different identity.')
            if stored_id == identity:
                matches.append(path)
        elif path == legacy:
            matches.append(path)
    if len(matches) > 1:
        raise ValueError(f'Multiple sync folders have the same {kind} ID. '
                         'Resolve the duplicate folders before syncing; nothing was merged.')
    label = _folder_label(name, kind)
    if matches and matches[0] != legacy:
        return matches[0]
    target = _safe_child(parent, f'{label}--{identity[:8]}')
    if target.exists():
        target = _safe_child(parent, f'{label}--{identity}')
    if target.exists():
        raise ValueError(f'The requested sync folder already exists: {target}')
    if matches:
        locks = (list(legacy.glob('contributors/*/.sync-lock')) if kind == 'project'
                 else [legacy / '.sync-lock'] if (legacy / '.sync-lock').exists() else [])
        if locks:
            raise ValueError('The legacy sync copy is locked. Close other sync sessions '
                             'before migrating its folder names.')
    else:
        _safe_child(parent, identity, mkdir=True)
    # A failed/interrupted rename leaves a recognizable UUID directory; write
    # the full identity first so readable folders never depend on truncated IDs.
    _write_json(legacy / marker_name, {'format': 1, 'id': identity, 'name': str(name)})
    legacy.rename(target)
    return target


@contextmanager
def _sync_location(project, configuration, contributor):
    root = check_destination(configuration['root'], configuration['root_id'])
    if project == root or project in root.parents or root in project.parents:
        raise ValueError('The source and sync destination must be separate, non-nested folders.')
    project_id = ensure_project_identity(project)
    contributor_id = _uuid(contributor['id'])
    # The lock's path is independent of display names and folder migration.
    lock = _safe_child(root, f'.project-{project_id}.sync-lock')
    try:
        handle = lock.open('x')
    except FileExistsError:
        raise ValueError('This project is locked by another or interrupted sync. '
                         'Confirm no sync is running before removing its project lock.')
    try:
        with handle:
            json.dump({'pid': os.getpid(), 'contributor': contributor['name']}, handle)
        metadata = _read_json(project / 'annotations/project.json')
        name = metadata.get('name') or project.name
        location = _identity_directory(root, name, project_id, 'project')
        contributors = _safe_child(location, 'contributors', mkdir=True)
        home = _identity_directory(contributors, contributor['name'], contributor_id, 'contributor')
        yield home
    finally:
        try:
            check_destination(root, configuration['root_id'])
            lock.unlink(missing_ok=True)
        except (OSError, ValueError):
            pass


def prepare_sync_location(project, configuration, contributor):
    """Create/migrate readable folders without copying or changing any revision."""
    with _sync_location(Path(project).resolve(), configuration, contributor) as home:
        return home


def _inventory(project, include_videos):
    files = {}
    sequences = project / 'tracking_sequences.json'
    if sequences.is_symlink():
        raise ValueError(f'Sync does not follow symbolic links: {sequences}')
    if sequences.exists():
        stat = sequences.stat()
        files[sequences.name] = (stat.st_size, stat.st_mtime_ns)
    for name in ('annotations', 'frames', 'input_data') if include_videos else ('annotations', 'frames'):
        base = project / name
        if base.is_symlink():
            raise ValueError(f'Sync does not follow symbolic links: {base}')
        if not base.exists():
            continue
        for directory, directories, filenames in os.walk(base, followlinks=False):
            directory = Path(directory)
            for child in directories:
                if (directory / child).is_symlink():
                    raise ValueError(f'Sync does not follow symbolic links: {directory / child}')
            # COCO is a regenerable export, not the editable annotation source.
            if directory == project / 'annotations':
                directories[:] = [item for item in directories if item != 'coco']
            for filename in filenames:
                if filename.startswith('.'):
                    continue
                path = directory / filename
                if path.is_symlink() or not path.is_file():
                    raise ValueError(f'Sync requires ordinary files: {path}')
                stat = path.stat()
                files[path.relative_to(project).as_posix()] = (stat.st_size, stat.st_mtime_ns)
    return files


def _digest(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _clone_or_copy(source, destination):
    """Use copy-on-write when supported, never hard links between editable copies."""
    try:
        import fcntl
        with source.open('rb') as src, destination.open('xb') as dst:
            fcntl.ioctl(dst.fileno(), 0x40049409, src.fileno())  # Linux FICLONE
        shutil.copystat(source, destination)
    except (ImportError, OSError):
        destination.unlink(missing_ok=True)
        shutil.copy2(source, destination)


def sync_project(project, configuration, contributor, progress=None, cancelled=None):
    """Publish one revision. Call only after saves finish and while edits are paused."""
    project = Path(project).resolve()
    with _sync_location(project, configuration, contributor) as home:
        return _sync_revision(project, configuration, contributor, home, progress, cancelled)


def _sync_revision(project, configuration, contributor, home, progress, cancelled):
    progress = progress or (lambda message: None)
    cancelled = cancelled or (lambda: False)
    project = Path(project).resolve()
    root = check_destination(configuration['root'], configuration['root_id'])
    if project == root or project in root.parents or root in project.parents:
        raise ValueError('The source and sync destination must be separate, non-nested folders.')
    project_id = ensure_project_identity(project)
    contributor_id = _uuid(contributor['id'])
    lock = _safe_child(home, '.sync-lock')
    try:
        lock_handle = lock.open('x')
    except FileExistsError:
        raise ValueError('This contributor copy is locked by another or interrupted sync. '
                         'Confirm no sync is running before removing its .sync-lock file.')
    staging = None
    try:
        with lock_handle:
            json.dump({'pid': os.getpid(), 'contributor': contributor['name']}, lock_handle)
        include_videos = configuration.get('include_videos', True)
        inventory = _inventory(project, include_videos)
        previous = {}
        previous_project = None
        latest = _safe_child(home, 'latest.json')
        if latest.exists():
            last = _read_json(latest)
            revision = _safe_child(home, Path('revisions') / last['revision'])
            manifest = _read_json(_safe_child(revision, 'manifest.json'))
            if manifest['project_id'] != project_id or manifest['contributor']['id'] != contributor_id:
                raise ValueError('The destination belongs to a different project or contributor.')
            previous = manifest['files']
            previous_project = _safe_child(revision, 'project')
        files = {}
        for index, relative in enumerate(sorted(inventory), 1):
            if cancelled():
                raise InterruptedError('Sync cancelled. No new revision was published.')
            progress(f'Checking {index}/{len(inventory)}: {relative}')
            files[relative] = {'sha256': _digest(project / relative), 'size': inventory[relative][0]}
        changed = {relative for relative in files if files[relative] != previous.get(relative)}
        # Verify the last published copy, including unchanged files, before reusing it.
        for relative in files.keys() & previous.keys():
            if cancelled():
                raise InterruptedError('Sync cancelled. No new revision was published.')
            if relative not in changed:
                existing = _safe_child(previous_project, relative)
                if not existing.is_file() or _digest(existing) != files[relative]['sha256']:
                    raise ValueError('The last synced copy was edited or damaged. '
                                     'Use a separate destination to preserve both versions.')
        if not changed and files == previous:
            if inventory != _inventory(project, include_videos):
                raise ValueError('Project files changed during sync. Save and try again.')
            return {'path': str(previous_project), 'copied_files': 0, 'unchanged': True}

        revision_name = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S') + '-' + uuid4().hex[:8]
        revisions = _safe_child(home, 'revisions', mkdir=True)
        staging = _safe_child(revisions, '.pending-' + revision_name, mkdir=True)
        staged_project = _safe_child(staging, 'project', mkdir=True)
        for index, relative in enumerate(sorted(files), 1):
            if cancelled():
                raise InterruptedError('Sync cancelled. No new revision was published.')
            check_destination(root, configuration['root_id'])
            progress(f'Copying {index}/{len(files)}: {relative}')
            target = staged_project / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            if relative in changed:
                shutil.copy2(project / relative, target)
            else:
                _clone_or_copy(_safe_child(previous_project, relative), target)
            if _digest(target) != files[relative]['sha256']:
                raise OSError(f'File verification failed: {relative}')
        if inventory != _inventory(project, include_videos):
            raise ValueError('Project files changed during sync. Save and try again.')
        if cancelled():
            raise InterruptedError('Sync cancelled. No new revision was published.')
        check_destination(root, configuration['root_id'])
        _write_json(staging / 'manifest.json', {
            'format': 1, 'project_id': project_id, 'project_name': project.name,
            'contributor': {'id': contributor_id, 'name': contributor['name']},
            'created_at': datetime.now(timezone.utc).isoformat(),
            'include_videos': include_videos, 'files': files,
        })
        published = revisions / revision_name
        staging.rename(published)
        staging = None
        _write_json(latest, {'revision': revision_name})
        return {'path': str(published / 'project'), 'copied_files': len(changed), 'unchanged': False}
    finally:
        # Only this invocation's unpublished files are eligible for cleanup.
        if staging is not None:
            try:
                check_destination(root, configuration['root_id'])
                shutil.rmtree(staging)
            except (OSError, ValueError):
                pass
        try:
            check_destination(root, configuration['root_id'])
            lock.unlink(missing_ok=True)
        except (OSError, ValueError):
            pass
