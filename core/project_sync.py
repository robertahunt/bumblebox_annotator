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
LOCAL_STATE = '.project-sync-state.json'
WORK_STATE = '.assigned-work.json'
OBJECTS_DIR = 'objects/sha256'
LOCATION_MARKERS = {'project': '.bumblebox-project.json',
                    'contributor': '.bumblebox-contributor.json'}


class SyncConflictError(RuntimeError):
    """Publishing would advance a history this working copy has not incorporated."""


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


def backup_location(path):
    """Recognize managed archives before the GUI creates any project files."""
    path = Path(path).expanduser().resolve()
    ancestors = (path, *path.parents)
    if any(p.name.startswith('.bumblebox-update-stage-') for p in ancestors):
        return {'kind': 'pending'}
    managed = any((p / ROOT_MARKER).exists() or
                  any((p / marker).exists() for marker in LOCATION_MARKERS.values())
                  for p in ancestors)
    if managed and any(p.name.startswith('.pending-') for p in ancestors):
        return {'kind': 'pending'}
    for parent in ancestors:
        manifest_path = parent / 'manifest.json'
        # Also recognize a standalone copied revision, without its archive root.
        if manifest_path.is_file():
            manifest = _read_json(manifest_path)
            if isinstance(manifest, dict) and manifest.get('kind') == 'local-update-delta':
                return {'kind': 'rollback', 'manifest': manifest, 'manifest_path': manifest_path}
            if isinstance(manifest, dict) and 'project_id' in manifest and 'files' in manifest:
                if manifest.get('format') != 3 and not (parent / 'project').is_dir():
                    continue
                if parent.name.startswith('.pending-'):
                    return {'kind': 'pending'}
                return {'kind': 'published', 'project': parent if manifest.get('format') == 3 else parent / 'project',
                        'manifest': manifest, 'manifest_path': manifest_path}
    return {'kind': 'container'} if managed else None


def require_working_project(path):
    if backup_location(path):
        raise ValueError('Published backups and sync folders cannot be edited or republished in place. '
                         'Create a local working copy first.')


def update_journal_path(project):
    project = Path(project).expanduser().resolve()
    key = hashlib.sha256(str(project).encode()).hexdigest()[:16]
    return project.parent / f'.bumblebox-update-{key}.json'


def require_no_update(project):
    journal = update_journal_path(project)
    if journal.exists() or journal.is_symlink():
        raise ValueError('This project has an active or interrupted local update. '
                         'Finish or recover that update before opening or publishing it.\n' + str(journal))


def _read_local_state(project, project_id):
    path = _safe_child(project, Path('annotations') / LOCAL_STATE)
    state = _read_json(path) if path.exists() else {'format': 1, 'project_id': project_id, 'destinations': {}}
    if state.get('project_id') != project_id or not isinstance(state.get('destinations'), dict):
        raise ValueError('Local sync ancestry does not match this project. Nothing was published.')
    return state


def _save_local_heads(project, project_id, root_id, heads):
    state = _read_local_state(project, project_id)
    state['destinations'][root_id] = {'heads': heads}
    _write_json(project / 'annotations' / LOCAL_STATE, state)


def _published_heads(home, project_id):
    heads = {}
    for contributor_home in home.parent.iterdir():
        if contributor_home.is_symlink():
            raise ValueError('Sync refuses symbolic links in contributor histories.')
        if not contributor_home.is_dir():
            continue
        latest = _safe_child(contributor_home, 'latest.json')
        if not latest.exists():
            continue
        revision = _read_json(latest)['revision']
        if not isinstance(revision, str) or Path(revision).name != revision or revision.startswith('.'):
            raise ValueError('Invalid published revision name.')
        manifest = _read_json(_safe_child(contributor_home, Path('revisions') / revision / 'manifest.json'))
        if manifest['project_id'] != project_id:
            raise ValueError('A published history belongs to a different project.')
        identity = _uuid(manifest['contributor']['id'])
        if identity in heads:
            raise ValueError('Multiple sync folders have the same contributor ID.')
        heads[identity] = revision
    return heads


def _require_current_heads(heads, base_heads):
    conflicting = sorted(identity for identity in heads.keys() | base_heads.keys()
                         if heads.get(identity) != base_heads.get(identity))
    if conflicting:
        details = '\n'.join(f'{identity[:8]}: local base {base_heads.get(identity, "unknown")} '
                            f'-> published {heads.get(identity, "missing")}' for identity in conflicting)
        raise SyncConflictError(
            'Published versions differ from this working copy\'s recorded starting point. '
            'It may be outdated or independently edited; no merge was attempted.\n\n' + details +
            '\n\nUse Update Local Copy to preview a newer publication. If both copies have edits, '
            'create a local copy of the desired published version and review/import your work, '
            'or save a separate backup without replacing any latest pointer.')


def restore_project(backup, destination, progress=None, cancelled=None):
    """Verify a published snapshot into a new working directory, never overwrite."""
    location = backup_location(backup)
    if not location or location['kind'] != 'published':
        raise ValueError('Choose a completed backup, not a sync container or pending transfer.')
    source = location['project']
    manifest_path = location['manifest_path']
    if source.is_symlink() or manifest_path.is_symlink():
        raise ValueError('Backup project and manifest must not be symbolic links.')
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    manifest_hash = hashlib.sha256(manifest_bytes).hexdigest()
    project_id = _uuid(manifest['project_id'])
    destination = Path(destination).expanduser()
    if destination.exists() or destination.is_symlink():
        raise ValueError('Choose a new local folder; existing folders are never overwritten.')
    destination = destination.resolve()
    require_working_project(destination)
    require_no_update(destination)
    if not destination.parent.is_dir():
        raise ValueError('The parent folder must already exist.')
    files = manifest['files']
    if not isinstance(files, dict) or not {'annotations/project.json', 'annotations/' + PROJECT_MARKER} <= files.keys():
        raise ValueError('The backup manifest is missing required project metadata.')
    progress = progress or (lambda message: None)
    cancelled = cancelled or (lambda: False)
    staging = Path(tempfile.mkdtemp(prefix='.restore-', dir=destination.parent))
    reserved = False
    try:
        for index, (relative, expected) in enumerate(sorted(files.items()), 1):
            if cancelled():
                raise InterruptedError('Local copy cancelled. No working copy was created.')
            original = snapshot_file(manifest_path, manifest, relative)
            if not original.is_file():
                raise ValueError(f'Backup file is missing or is not an ordinary file: {relative}')
            target = _safe_child(staging, relative)
            target.parent.mkdir(parents=True, exist_ok=True)
            progress(f'Copying and verifying {index}/{len(files)}: {relative}')
            _clone_or_copy(original, target)
            if target.stat().st_size != expected['size'] or _digest(target) != expected['sha256']:
                raise ValueError(f'Backup file is missing, edited, or damaged: {relative}')
        if _read_json(staging / 'annotations' / PROJECT_MARKER)['project_id'] != project_id:
            raise ValueError('Backup project identity does not match its manifest.')
        if _digest(manifest_path) != manifest_hash:
            raise ValueError('The backup manifest changed while copying. Try again.')
        root_id = manifest.get('root_id')
        for parent in manifest_path.parents:
            marker = parent / ROOT_MARKER
            if marker.is_file():
                actual = _uuid(_read_json(marker)['id'])
                if root_id and root_id != actual:
                    raise ValueError('Backup destination identity does not match its manifest.')
                root_id = actual
                break
        if root_id:
            heads = dict(manifest.get('base_heads', {}))
            if manifest.get('publication') != 'separate':
                heads[_uuid(manifest['contributor']['id'])] = manifest_path.parent.name
            _save_local_heads(staging, project_id, _uuid(root_id), heads)
        if cancelled():
            raise InterruptedError('Local copy cancelled. No working copy was created.')
        # Reserve the requested name exclusively before promoting the verified tree.
        destination.mkdir()
        reserved = True
        staging.replace(destination)
        reserved = False
        return {'path': str(destination), 'include_videos': manifest.get('include_videos', True)}
    finally:
        if staging.exists():
            shutil.rmtree(staging)
        if reserved:
            destination.rmdir()


def _folder_label(name, fallback):
    text = unicodedata.normalize('NFKD', str(name)).encode('ascii', 'ignore').decode()
    text = re.sub(r'[^A-Za-z0-9._-]+', '-', text).strip('._-')[:64].rstrip('._-')
    return text or fallback


def _identity_directory(parent, name, identity, kind, *, rename_existing=False):
    """Find by full ID and rename project folders without duplicating history."""
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
    current = matches[0] if matches else legacy
    short_name, full_name = f'{label}--{identity[:8]}', f'{label}--{identity}'
    if matches and current != legacy and (not rename_existing or kind != 'project' or
                                         current.name in (short_name, full_name)):
        return current
    target = _safe_child(parent, short_name)
    if target.exists():
        target = _safe_child(parent, full_name)
    if target.exists():
        raise ValueError(f'The requested sync folder already exists: {target}')
    if matches:
        locks = (list(current.glob('contributors/*/.sync-lock')) if kind == 'project'
                 else [current / '.sync-lock'] if (current / '.sync-lock').exists() else [])
        if locks:
            raise ValueError('The sync copy is locked. Close other sync sessions '
                             'before renaming its folders.')
    else:
        _safe_child(parent, identity, mkdir=True)
    # Write identity before renaming so interruption leaves a recognizable folder.
    _write_json(current / marker_name, {'format': 1, 'id': identity, 'name': str(name)})
    current.rename(target)
    return target


@contextmanager
def _sync_location(project, configuration, contributor):
    root = check_destination(configuration['root'], configuration['root_id'])
    if project == root or project in root.parents or root in project.parents:
        raise ValueError('The source and sync destination must be separate, non-nested folders.')
    require_working_project(project)
    require_no_update(project)
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
        location = _identity_directory(root, project.name, project_id, 'project')
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


def object_path(home, digest):
    if not isinstance(digest, str) or not re.fullmatch('[0-9a-f]{64}', digest):
        raise ValueError('Invalid stored file checksum.')
    return _safe_child(home, Path(OBJECTS_DIR) / digest[:2] / digest)


def snapshot_file(manifest_path, manifest, name):
    """Resolve both independent legacy snapshots and compact shared-object snapshots."""
    manifest_path = Path(manifest_path)
    _safe_child(manifest_path.parent, name)
    if manifest.get('format') in (1, 2):
        return _safe_child(manifest_path.parent / 'project', name)
    if manifest.get('format') != 3 or manifest.get('storage') != 'objects-v1':
        raise ValueError('Unsupported publication storage format. Update the annotator.')
    for home in manifest_path.parents:
        marker = _safe_child(home, LOCATION_MARKERS['project'])
        if marker.is_file():
            if _read_json(marker).get('id') != manifest['project_id']:
                raise ValueError('Publication belongs to a different shared file store.')
            check_destination(home.parent, manifest['root_id'])
            return object_path(home, manifest['files'][name]['sha256'])
    raise ValueError('The shared file store is missing. Keep compact publications inside their sync project folder.')


def publication_estimate(project, configuration, contributor, progress=None, cancelled=None):
    """Read-only payload estimate; the actual publish rechecks content and free space."""
    project = Path(project).resolve()
    require_no_update(project)
    root = check_destination(configuration['root'], configuration['root_id'])
    inventory = _inventory(project, configuration.get('include_videos', True))
    logical = sum(stat[0] for stat in inventory.values())
    home = None
    identity = project / 'annotations' / PROJECT_MARKER
    if identity.exists():
        project_id = _read_json(identity)['project_id']
        for candidate in root.iterdir():
            if candidate.is_symlink() or not candidate.is_dir():
                continue
            marker = _safe_child(candidate, LOCATION_MARKERS['project'])
            if marker.is_file() and _read_json(marker).get('id') == project_id:
                if home is not None:
                    raise ValueError('Duplicate published project identity.')
                home = candidate
    missing, seen = 0, set()
    if configuration.get('storage_mode', 'deduplicated') == 'full' or home is None:
        missing = logical
    else:
        for index, (name, stat) in enumerate(inventory.items(), 1):
            if cancelled and cancelled():
                raise InterruptedError('Publication preview cancelled.')
            if progress:
                progress(f'Estimating new storage {index}/{len(inventory)}: {name}')
            digest = _digest(project / name)
            if digest not in seen and not object_path(home, digest).is_file():
                missing += stat[0]
            seen.add(digest)
    label = _folder_label(project.name, 'project')
    rename_from = (home.name if home and home.name not in
                   (f'{label}--{project_id[:8]}', f'{label}--{project_id}') else None)
    return {'project_name': project.name, 'rename_from': rename_from,
            'files': len(inventory), 'logical_bytes': logical, 'new_bytes': missing + 65536 + len(inventory) * 512,
            'free_bytes': shutil.disk_usage(root).free}


def sync_project(project, configuration, contributor, progress=None, cancelled=None, *, separate_backup=False):
    """Publish one revision. Call only after saves finish and while edits are paused."""
    project = Path(project).resolve()
    if _safe_child(project, 'annotations/' + WORK_STATE).exists():
        raise ValueError('This is assigned work, not a complete project. Use Publish Work to preserve undownloaded files.')
    with _sync_location(project, configuration, contributor) as home:
        result = _sync_revision(project, configuration, contributor, home, progress, cancelled, separate_backup)
        # Only an accepted publication may rename an existing readable archive.
        # Keep the project lock, but release the contributor lock before moving it.
        if not separate_backup:
            previous = home.parent.parent
            try:
                renamed = _identity_directory(previous.parent, project.name, ensure_project_identity(project),
                                              'project', rename_existing=True)
            except (OSError, ValueError) as exc:
                warning = 'Backup verified, but its archive folder could not be renamed: ' + str(exc)
                result['warning'] = '\n'.join(filter(None, (result.get('warning'), warning)))
            else:
                result['path'] = str(renamed / Path(result['path']).relative_to(previous))
        return result


def _sync_revision(project, configuration, contributor, home, progress, cancelled, separate_backup):
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
        if configuration.get('storage_mode', 'deduplicated') not in ('deduplicated', 'full'):
            raise ValueError('Unknown backup storage mode.')
        compact = configuration.get('storage_mode', 'deduplicated') == 'deduplicated'
        heads = _published_heads(home, project_id)
        state = _read_local_state(project, project_id)
        known = state['destinations'].get(configuration['root_id'])
        base_heads = dict(known['heads']) if known else {}
        inventory = _inventory(project, include_videos)
        previous = {}
        previous_project = None
        previous_manifest = None
        previous_manifest_path = None
        latest = _safe_child(home, 'latest.json')
        if latest.exists():
            last = _read_json(latest)
            revision = _safe_child(home, Path('revisions') / last['revision'])
            manifest = _read_json(_safe_child(revision, 'manifest.json'))
            previous_manifest, previous_manifest_path = manifest, revision / 'manifest.json'
            if manifest['project_id'] != project_id or manifest['contributor']['id'] != contributor_id:
                raise ValueError('The destination belongs to a different project or contributor.')
            previous = manifest['files']
            previous_project = revision if manifest.get('format') == 3 else _safe_child(revision, 'project')
        if not separate_backup and (known is not None or not previous):
            _require_current_heads(heads, base_heads)
        files = {}
        for index, relative in enumerate(sorted(inventory), 1):
            if cancelled():
                raise InterruptedError('Sync cancelled. No new revision was published.')
            progress(f'Checking {index}/{len(inventory)}: {relative}')
            files[relative] = {'sha256': _digest(project / relative), 'size': inventory[relative][0]}
        # Existing installations can establish ancestry from an exact, sole latest
        # snapshot. A changed legacy copy must not silently claim a shared baseline.
        if known is None and files == previous and set(heads) == {contributor_id}:
            base_heads = dict(heads)
        if not separate_backup:
            _require_current_heads(heads, base_heads)
        changed = {relative for relative in files if files[relative] != previous.get(relative)}
        # Verify the last published copy, including unchanged files, before reusing it.
        verified_paths = set()
        for relative in files.keys() & previous.keys():
            if cancelled():
                raise InterruptedError('Sync cancelled. No new revision was published.')
            if relative not in changed:
                existing = snapshot_file(previous_manifest_path, previous_manifest, relative)
                if existing not in verified_paths:
                    if (not existing.is_file() or existing.stat().st_size != files[relative]['size'] or
                            _digest(existing) != files[relative]['sha256']):
                        raise ValueError('The last synced copy was edited or damaged. '
                                         'Use a separate destination to preserve both versions.')
                    verified_paths.add(existing)
        if (not changed and files == previous and not separate_backup and
                bool(previous_manifest and previous_manifest.get('format') == 3) == compact):
            if inventory != _inventory(project, include_videos):
                raise ValueError('Project files changed during sync. Save and try again.')
            return _publication_result(project, project_id, configuration['root_id'], heads,
                                       previous_project, 0, True, False)

        revision_name = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S') + '-' + uuid4().hex[:8]
        project_home = home.parent.parent
        missing_objects, verified = {}, set()
        if compact:
            for relative, expected in files.items():
                if cancelled():
                    raise InterruptedError('Sync cancelled. No new revision was published.')
                digest = expected['sha256']
                if digest in verified:
                    continue
                verified.add(digest)
                existing = object_path(project_home, digest)
                if existing.exists():
                    if existing not in verified_paths and (not existing.is_file() or
                            existing.stat().st_size != expected['size'] or _digest(existing) != digest):
                        raise ValueError('A shared backup file is damaged. No version was published: ' + str(existing))
                else:
                    missing_objects[digest] = relative
            required = sum(files[name]['size'] for name in missing_objects.values())
        else:
            required = sum(entry['size'] for entry in files.values())
        required += 65536 + len(files) * 512
        if shutil.disk_usage(root).free < required:
            raise OSError(f'Insufficient destination space; allow approximately {required / 1024**3:.2f} GiB. '
                          'A server quota may impose an additional limit.')
        revisions = _safe_child(home, 'revisions', mkdir=True)
        staging = _safe_child(revisions, '.pending-' + revision_name, mkdir=True)
        staged_project = _safe_child(staging, 'project', mkdir=True) if not compact else None
        to_copy = sorted(missing_objects.values()) if compact else sorted(files)
        for index, relative in enumerate(to_copy, 1):
            if cancelled():
                raise InterruptedError('Sync cancelled. No new revision was published.')
            check_destination(root, configuration['root_id'])
            progress(f'Copying {index}/{len(to_copy)}: {relative}')
            target = (object_path(project_home, files[relative]['sha256']) if compact
                      else staged_project / relative)
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_name('.pending-' + uuid4().hex) if compact else target
            try:
                if compact or relative in changed:
                    shutil.copy2(project / relative, temporary)
                else:
                    _clone_or_copy(snapshot_file(previous_manifest_path, previous_manifest, relative), temporary)
                if _digest(temporary) != files[relative]['sha256']:
                    raise OSError(f'File verification failed: {relative}')
                if compact:
                    temporary.rename(target)
            finally:
                if compact:
                    temporary.unlink(missing_ok=True)
        if inventory != _inventory(project, include_videos):
            raise ValueError('Project files changed during sync. Save and try again.')
        if cancelled():
            raise InterruptedError('Sync cancelled. No new revision was published.')
        check_destination(root, configuration['root_id'])
        _write_json(staging / 'manifest.json', {
            'format': 3 if compact else 2, 'storage': 'objects-v1' if compact else 'full',
            'project_id': project_id, 'project_name': project.name,
            'contributor': {'id': contributor_id, 'name': contributor['name']},
            'device': contributor.get('device'),
            'created_at': datetime.now(timezone.utc).isoformat(),
            'include_videos': include_videos, 'files': files,
            'root_id': configuration['root_id'], 'base_heads': base_heads,
            'publication': 'separate' if separate_backup else 'latest',
        })
        published = revisions / revision_name
        staging.rename(published)
        staging = None
        if not separate_backup:
            _write_json(latest, {'revision': revision_name})
            heads[contributor_id] = revision_name
        return _publication_result(project, project_id, configuration['root_id'], heads,
                                   published if compact else published / 'project',
                                   len(to_copy) if compact else len(changed), False, separate_backup)
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


def _publication_result(project, project_id, root_id, heads, path, copied, unchanged, separate):
    result = {'path': str(path), 'copied_files': copied, 'unchanged': unchanged, 'separate_backup': separate}
    if not separate:
        try:
            _save_local_heads(project, project_id, root_id, heads)
        except (OSError, ValueError) as exc:
            result['warning'] = 'Backup completed, but local ancestry could not be saved: ' + str(exc)
    return result
