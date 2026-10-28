"""Journalled file-level updates and rollback bundles; unchanged data stays in place."""

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
from uuid import uuid4

from core.project_sync import (LOCAL_STATE, WORK_STATE, PROJECT_MARKER, _clone_or_copy, _digest,
    _inventory, _read_json, _safe_child, _write_json, require_no_update,
    require_working_project, update_journal_path)


def default_rollback_root():
    return Path(os.environ.get('XDG_DATA_HOME', Path.home() / '.local/share')) / 'BeeWhere/project-rollbacks'


def rollback_root(configuration, project, create=False):
    configured = configuration.get('rollback_root')
    root = Path(configured).expanduser() if configured else default_rollback_root()
    root = root.resolve()
    project = Path(project).resolve()
    remote = Path(configuration['root']).resolve() if configuration.get('root') else None
    if root == project or project in root.parents or (remote and (root == remote or remote in root.parents)):
        raise ValueError('Choose a rollback folder outside the working project and shared sync archive.')
    require_working_project(root)
    if create and not configured:
        root.mkdir(parents=True, exist_ok=True)
    if configured and not root.is_dir():
        raise ValueError('The configured rollback folder is unavailable. Reconnect it or choose another folder.')
    return root


def file_record(path):
    path = Path(path)
    if path.is_symlink():
        raise ValueError('Update files must not be symbolic links: ' + str(path))
    if not path.exists():
        return None
    if not path.is_file():
        raise ValueError('Update requires an ordinary file: ' + str(path))
    return {'size': path.stat().st_size, 'sha256': _digest(path)}


def bytes_record(data):
    return {'size': len(data), 'sha256': hashlib.sha256(data).hexdigest()}


def _matches(path, expected):
    return file_record(path) == expected


def _copy_verified(source, target, expected):
    target.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(source, bytes):
        with target.open('xb') as handle:
            handle.write(source)
            handle.flush()
            os.fsync(handle.fileno())
    else:
        if not Path(source).is_file():
            raise ValueError('Published or rollback file is missing: ' + str(source))
        _clone_or_copy(source, target)
    if not _matches(target, expected):
        raise ValueError('File failed verification: ' + str(source))


def _check_names(before, after):
    if before.keys() != after.keys():
        raise ValueError('Incomplete update recovery record.')
    for name in before:
        path = Path(name)
        if (not path.parts or path.is_absolute() or '..' in path.parts or path.as_posix() != name or
                (path.parts[0] not in ('annotations', 'frames', 'input_data') and
                 name not in ('tracking_sequences.json', 'dataset_info.json'))):
            raise ValueError('Unsafe file in update recovery record.')


def _space_parent(path):
    while not path.exists():
        path = path.parent
    return path


def check_space(project, root, before, after):
    backup_bytes = sum(v['size'] for v in before.values() if v)
    stage_bytes = sum(v['size'] for v in after.values() if v)
    recovery_bytes = max((v['size'] for v in before.values() if v), default=0)
    local = Path(project).parent
    storage = _space_parent(root)
    overhead = 1024 * 1024 + len(before) * 2048
    if local.stat().st_dev == storage.stat().st_dev:
        if shutil.disk_usage(local).free < backup_bytes + stage_bytes + recovery_bytes + overhead:
            raise OSError('Insufficient local disk space for changed files and their rollback bundle.')
    elif (shutil.disk_usage(local).free < stage_bytes + recovery_bytes + overhead or
          shutil.disk_usage(storage).free < backup_bytes + overhead):
        raise OSError('Insufficient space on the project or rollback drive.')


def apply_delta(project, configuration, before, after, sources, project_id,
                post_files, include_videos, check_unchanged, progress, cancelled):
    """Stage replacements and old affected files before any live file is changed."""
    project = Path(project).resolve()
    require_no_update(project)
    _check_names(before, after)
    root = rollback_root(configuration, project, create=True)
    check_space(project, root, before, after)
    token = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S') + '-' + uuid4().hex[:8]
    stage = project.parent / ('.bumblebox-update-stage-' + token)
    backup = root / ('bumblebox-before-update-' + token)
    journal = update_journal_path(project)
    record = {'format': 2, 'project': str(project), 'project_id': project_id,
              'stage': stage.name, 'backup': str(backup), 'phase': 'staging',
              'before': before, 'after': after}
    with journal.open('x') as handle:
        json.dump(record, handle)
        handle.flush()
        os.fsync(handle.fileno())
    try:
        stage.mkdir()
        backup.mkdir()
        manifest = {
            'format': 1, 'kind': 'local-update-delta', 'project': str(project),
            'project_id': project_id, 'created_at': datetime.now(timezone.utc).isoformat(),
            'before': before, 'after': after, 'post_files': post_files,
            'include_videos': include_videos, 'status': 'staging'}
        _write_json(backup / 'manifest.json', manifest)
        for index, name in enumerate(sorted(before), 1):
            if cancelled():
                raise InterruptedError('Update cancelled; no local files were replaced.')
            progress(f'Preparing changed file {index}/{len(before)}: {name}')
            original = _safe_child(project, name)
            if before[name]:
                _copy_verified(original, _safe_child(backup / 'before', name), before[name])
            elif original.exists():
                raise ValueError('Local files changed after preview.')
            if after[name]:
                _copy_verified(sources[name], _safe_child(stage, name), after[name])
        check_unchanged()
        if cancelled():
            raise InterruptedError('Update cancelled; no local files were replaced.')
        manifest['status'] = 'prepared'
        _write_json(backup / 'manifest.json', manifest)
        record['phase'] = 'applying'
        _write_json(journal, record)
        # Cancellation stops here: finish all replacements or roll back all of them.
        for name in sorted(after):
            target = _safe_child(project, name)
            if not _matches(target, before[name]):
                raise ValueError('A local file changed while applying the update: ' + name)
            if after[name]:
                target.parent.mkdir(parents=True, exist_ok=True)
                os.replace(_safe_child(stage, name), target)
            else:
                target.unlink(missing_ok=True)
        manifest = _read_json(backup / 'manifest.json')
        manifest['status'] = 'complete'
        _write_json(backup / 'manifest.json', manifest)
        record['phase'] = 'committed'
        _write_json(journal, record)
        result = {'updated': True, 'path': str(project), 'backup': str(backup)}
        shutil.rmtree(stage)
        try:
            journal.unlink()
        except OSError:
            result['warning'] = 'Update completed; its recovery marker remains. Reopen the project to verify it.'
        return result
    except BaseException:
        if record['phase'] in ('applying', 'committed'):
            try:
                recover_delta(project, _read_json(journal), progress)
            except Exception as exc:
                raise RuntimeError('Update requires recovery before editing. The old affected files are safe at '
                                   + str(backup)) from exc
        else:
            if stage.exists():
                shutil.rmtree(stage)
            if backup.exists():
                shutil.rmtree(backup)
            journal.unlink(missing_ok=True)
        raise


def recover_delta(project, record, progress=lambda text: None):
    project = Path(project).resolve()
    if record.get('format') != 2 or record.get('project') != str(project):
        raise ValueError('Update journal belongs to another project.')
    if record.get('phase') not in ('staging', 'applying', 'committed'):
        raise ValueError('Unknown update recovery phase. Keep the recovery files for manual review.')
    stage = _safe_child(project.parent, record['stage'])
    backup = Path(record['backup'])
    if (stage.parent != project.parent or not stage.name.startswith('.bumblebox-update-stage-') or
            not backup.is_absolute() or not backup.name.startswith('bumblebox-before-update-') or
            backup.resolve() != backup or
            stage.name.removeprefix('.bumblebox-update-stage-') !=
            backup.name.removeprefix('bumblebox-before-update-')):
        raise ValueError('Unsafe recovery folder.')
    before, after = record['before'], record['after']
    _check_names(before, after)
    if _read_json(_safe_child(project, 'annotations/' + PROJECT_MARKER)).get('project_id') != record['project_id']:
        raise ValueError('The working project identity changed after interruption.')
    if backup.exists() or record['phase'] != 'staging':
        manifest = _read_json(_safe_child(backup, 'manifest.json'))
        if (manifest.get('kind') != 'local-update-delta' or manifest.get('project') != str(project) or
                manifest.get('project_id') != record['project_id'] or
                manifest.get('before') != before or manifest.get('after') != after):
            raise ValueError('Rollback bundle does not match the interrupted update.')
    if record['phase'] != 'staging':
        # Verify every affected file before recovery writes anything.
        for name in before:
            current = file_record(_safe_child(project, name))
            allowed = [after[name]] if record['phase'] == 'committed' else [before[name], after[name]]
            if current not in allowed:
                raise ValueError('A file changed after interruption. Keep the rollback bundle for manual review: ' + name)
            if record['phase'] != 'committed' and before[name]:
                if not _matches(_safe_child(backup / 'before', name), before[name]):
                    raise ValueError('Rollback file is missing or damaged: ' + name)
        if record['phase'] != 'committed':
            stage.mkdir(exist_ok=True)
            for index, name in enumerate(sorted(before)):
                progress('Restoring previous file: ' + name)
                target = _safe_child(project, name)
                if before[name] is None:
                    target.unlink(missing_ok=True)
                elif not _matches(target, before[name]):
                    temporary = stage / f'.recovery-{index}'
                    temporary.unlink(missing_ok=True)
                    _copy_verified(_safe_child(backup / 'before', name), temporary, before[name])
                    target.parent.mkdir(parents=True, exist_ok=True)
                    os.replace(temporary, target)
            manifest['status'] = 'rolled-back'
            _write_json(backup / 'manifest.json', manifest)
    if stage.exists():
        shutil.rmtree(stage)
    if record['phase'] == 'staging' and backup.exists():
        shutil.rmtree(backup)
    update_journal_path(project).unlink()
    return {'path': str(project), 'message': ('The completed update was verified; its rollback bundle remains available.'
            if record['phase'] == 'committed' else 'The original local files were restored.')}


def rollback_update(project, bundle, configuration, progress=None, cancelled=None):
    """Undo only when the managed project still matches the completed update."""
    project, bundle = Path(project).resolve(), Path(bundle).resolve()
    require_no_update(project)
    manifest = _read_json(_safe_child(bundle, 'manifest.json'))
    if (manifest.get('kind') != 'local-update-delta' or manifest.get('project') != str(project)
            or manifest.get('status') != 'complete'):
        raise ValueError('Choose a completed rollback bundle for this working copy.')
    def unchanged():
        current = {name: file_record(project / name)
                   for name in _inventory(project, manifest['include_videos'])}
        if current != manifest['post_files']:
            raise ValueError('This project has changed since the update. Rollback would overwrite newer work.')
        for name, expected in manifest['after'].items():
            if not _matches(_safe_child(project, name), expected):
                raise ValueError('An updated file changed since the update: ' + name)
    unchanged()
    before, after = manifest['after'], manifest['before']
    post = dict(manifest['post_files'])
    for name, entry in after.items():
        if name in ('annotations/' + LOCAL_STATE, 'annotations/' + WORK_STATE) or name.startswith('annotations/coco/') or name == 'dataset_info.json':
            continue
        if entry is None:
            post.pop(name, None)
        else:
            post[name] = entry
    return apply_delta(project, configuration, before, after,
        {name: _safe_child(bundle / 'before', name) for name, entry in after.items() if entry},
        manifest['project_id'], post, manifest['include_videos'], unchanged,
        progress or (lambda text: None), cancelled or (lambda: False))


def rollback_bundles(project, configuration):
    root = rollback_root(configuration, project)
    result = []
    if not root.exists():
        return result
    for path in root.glob('bumblebox-before-update-*'):
        if path.is_symlink() or not path.is_dir():
            continue
        manifest_path = _safe_child(path, 'manifest.json')
        if not manifest_path.is_file():
            continue  # Another project's interrupted staging is not completed history.
        info = _read_json(manifest_path)
        if info.get('kind') == 'local-update-delta' and info.get('project') == str(Path(project).resolve()):
            result.append({'path': path, 'manifest': info,
                           'bytes': sum(v['size'] for v in info['before'].values() if v)})
    return sorted(result, key=lambda item: item['manifest']['created_at'], reverse=True)


def delete_rollback_bundle(project, bundle, configuration):
    require_no_update(project)
    match = next((item for item in rollback_bundles(project, configuration)
                  if item['path'] == Path(bundle)), None)
    if not match or match['manifest']['status'] not in ('complete', 'rolled-back'):
        raise ValueError('Only completed, inactive rollback bundles can be removed.')
    shutil.rmtree(match['path'])
