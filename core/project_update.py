"""Explicit, previewed fast-forward updates of local projects. No annotation merge."""

from copy import deepcopy
import json
import os
from pathlib import Path
import shutil

from core.annotation_policy import category_policies, policy_text
from core.categories import category_label
from core.project_delta import apply_delta, bytes_record, file_record, recover_delta, rollback_root
from core.project_sync import (
    LOCAL_STATE, WORK_STATE, LOCATION_MARKERS, PROJECT_MARKER, SyncConflictError,
    _digest, _inventory, _read_json, _read_local_state,
    _safe_child, _uuid, check_destination,
    require_no_update, require_working_project, update_journal_path,
    snapshot_file,
)


def _cancelled(cancelled):
    if cancelled():
        raise InterruptedError('Update cancelled; the local project was not replaced.')


def _identity_folder(parent, identity, kind):
    matches = []
    for path in parent.iterdir():
        if path.is_symlink() or not path.is_dir():
            continue
        marker = _safe_child(path, LOCATION_MARKERS[kind])
        if (marker.is_file() and _read_json(marker).get('id') == identity) or (
                not marker.exists() and path.name == identity):
            matches.append(path)
    if len(matches) != 1:
        raise ValueError(f'Expected one published {kind} folder for {identity}; found {len(matches)}.')
    return matches[0]


def _reference(contributor, revision):
    contributor = _uuid(contributor)
    if (not isinstance(revision, str) or not revision or Path(revision).name != revision
            or revision.startswith('.')):
        raise ValueError('Invalid published version reference.')
    return contributor, revision


class PublishedProject:
    """Read-only view; selecting updates never creates/migrates remote folders."""

    def __init__(self, project, configuration):
        self.project = Path(project).resolve()
        require_working_project(self.project)
        self.configuration = dict(configuration)
        self.root = check_destination(configuration['root'], configuration['root_id'])
        marker = _safe_child(self.project, 'annotations/' + PROJECT_MARKER)
        if not marker.is_file():
            raise ValueError('This project has no shared identity. Publish it first, or create a local copy of a backup.')
        self.project_id = _uuid(_read_json(marker)['project_id'])
        self.home = _identity_folder(self.root, self.project_id, 'project')
        self.contributors = _safe_child(self.home, 'contributors')
        self.cache = {}

    @classmethod
    def from_archive(cls, configuration, project_id):
        view = cls.__new__(cls)
        view.project = None
        view.configuration = dict(configuration)
        view.root = check_destination(configuration['root'], configuration['root_id'])
        view.project_id = _uuid(project_id)
        view.home = _identity_folder(view.root, view.project_id, 'project')
        view.contributors = _safe_child(view.home, 'contributors')
        view.cache = {}
        return view

    def snapshot(self, reference, metadata_only=False):
        reference = _reference(*reference)
        cache_key = (reference, metadata_only)
        if cache_key in self.cache:
            return self.cache[cache_key]
        contributor, revision = reference
        home = _identity_folder(self.contributors, contributor, 'contributor')
        path = _safe_child(home, Path('revisions') / revision)
        manifest_path = _safe_child(path, 'manifest.json')
        fingerprint = _digest(manifest_path)
        manifest = _read_json(manifest_path)
        if _digest(manifest_path) != fingerprint:
            raise ValueError('A published manifest changed while reading it. Preview again.')
        if (manifest.get('format') not in (1, 2, 3) or manifest['project_id'] != self.project_id
                or manifest['contributor']['id'] != contributor
                or manifest.get('root_id', self.configuration['root_id']) != self.configuration['root_id']):
            raise ValueError('Published version identity or format is incompatible with this project.')
        files = manifest['files']
        if not isinstance(files, dict) or not {'annotations/project.json', 'annotations/' + PROJECT_MARKER} <= files.keys():
            raise ValueError('Published version is missing required project metadata.')
        for name, record in files.items():
            relative = Path(name)
            _safe_child(path / 'project', relative)
            if (relative.as_posix() != name or not relative.parts or
                    (relative.parts[0] not in ('annotations', 'frames', 'input_data') and name != 'tracking_sequences.json')
                    or any(part.startswith('.') for part in relative.parts)
                    or relative.parts[:2] == ('annotations', 'coco')
                    or not isinstance(record.get('size'), int) or record['size'] < 0
                    or not isinstance(record.get('sha256'), str) or len(record['sha256']) != 64):
                raise ValueError(f'Unsupported file in published manifest: {name}')
            if relative.parts[0] == 'input_data' and not manifest.get('include_videos', True):
                raise ValueError('Partial backup unexpectedly includes original videos.')
        parents = dict(_reference(k, v) for k, v in manifest.get('base_heads', {}).items())
        if metadata_only:
            del manifest['files']
            manifest.pop('annotation_work', None)
        result = {'reference': reference, 'path': path / 'project', 'manifest_path': manifest_path,
                  'manifest_hash': fingerprint, 'manifest': manifest, 'parents': parents}
        self.cache[cache_key] = result
        return result

    def heads(self, *, validate_snapshots=True):
        check_destination(self.root, self.configuration['root_id'])
        heads = {}
        for home in self.contributors.iterdir():
            if home.is_symlink():
                raise ValueError('Published histories must not be symbolic links.')
            if not home.is_dir():
                continue
            latest = _safe_child(home, 'latest.json')
            if not latest.exists():
                continue
            revision = _read_json(latest)['revision']
            marker = _safe_child(home, LOCATION_MARKERS['contributor'])
            identity = _uuid(_read_json(marker)['id'] if marker.exists() else home.name)
            if identity in heads:
                raise ValueError('Duplicate published contributor identity.')
            _reference(identity, revision)
            if validate_snapshots:
                self.snapshot((identity, revision), metadata_only=True)
            heads[identity] = revision
        return heads

    def descends_from(self, selected, base):
        queue, visited = [selected], set()
        while queue:
            reference = queue.pop()
            if reference == base:
                return True
            if reference in visited:
                continue
            visited.add(reference)
            if len(visited) > 10000:
                raise ValueError('Published history is too large to validate automatically.')
            queue.extend(self.snapshot(reference, metadata_only=True)['parents'].items())
        return False


def published_versions(project, configuration):
    require_no_update(project)
    published = PublishedProject(project, configuration)
    return [published.snapshot(ref, metadata_only=True) for ref in sorted(published.heads().items())]


def _base_snapshot(published, known):
    if not known or not known.get('heads'):
        raise ValueError('This copy has no recorded starting version. Create a verified local working copy '
                         'of a backup, then import/review any other local work. No files were changed.')
    heads = dict(_reference(k, v) for k, v in known['heads'].items())
    candidates = []
    for ref in heads.items():
        snapshot = published.snapshot(ref, metadata_only=True)
        covered = dict(snapshot['parents'])
        covered[ref[0]] = ref[1]
        if covered == heads:
            candidates.append(snapshot)
    if len(candidates) != 1:
        raise ValueError('The starting snapshot is ambiguous. Use a verified local copy; automatic updating is unsafe.')
    return published.snapshot(candidates[0]['reference'])


def _project_tree(project):
    """Include local-only models/outputs so the rollback copy preserves everything."""
    files = {}
    for directory, subdirs, names in os.walk(project, followlinks=False):
        directory = Path(directory)
        for name in subdirs + names:
            path = directory / name
            if path.is_symlink():
                raise ValueError(f'Local updates do not follow symbolic links: {path}')
            relative = path.relative_to(project).as_posix()
            stat = path.stat()
            if path.is_dir():
                files[relative + '/'] = None
            elif path.is_file():
                files[relative] = (stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns, stat.st_ino)
            else:
                raise ValueError(f'Local updates require ordinary files: {path}')
    return files


def _projection(files, include_videos):
    return {name: value for name, value in files.items() if include_videos or not name.startswith('input_data/')}


def _changes(before, after):
    return {name: ('Add' if name not in before else 'Delete' if name not in after else 'Replace')
            for name in before.keys() | after.keys() if before.get(name) != after.get(name)}


def _verified_info(snapshot):
    path = snapshot_file(snapshot['manifest_path'], snapshot['manifest'], 'annotations/project.json')
    if _digest(path) != snapshot['manifest']['files']['annotations/project.json']['sha256']:
        raise ValueError('Published project settings failed verification.')
    return _read_json(path)


def _invalidated_files(project):
    coco = _safe_child(project, 'annotations/coco')
    names = [p.relative_to(project).as_posix() for p in coco.rglob('*') if p.is_file() or p.is_symlink()]
    if (project / 'dataset_info.json').exists():
        names.append('dataset_info.json')
    return names


def _update_tree(project, include_videos):
    names = set(_inventory(project, include_videos)) | set(_invalidated_files(project))
    state = 'annotations/' + LOCAL_STATE
    if (project / state).exists():
        names.add(state)
    result = {}
    for name in names:
        path = _safe_child(project, name)
        stat = path.stat()
        result[name] = (stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns, stat.st_ino)
    return result


class PreparedUpdate:
    def __init__(self, project, configuration, reference, progress=None, cancelled=None):
        progress = progress or (lambda message: None)
        cancelled = cancelled or (lambda: False)
        self.project = Path(project).resolve()
        if _safe_child(self.project, 'annotations/' + WORK_STATE).exists():
            raise ValueError('Assigned work cannot use a full-project update. Publish it, then download a new working set.')
        require_no_update(self.project)
        self.published = PublishedProject(self.project, configuration)
        self.heads = self.published.heads()
        reference = _reference(*reference)
        if self.heads.get(reference[0]) != reference[1]:
            raise ValueError('The selected publication changed. Refresh the version list and preview again.')
        self.selected = self.published.snapshot(reference)
        self.local_state = _read_local_state(self.project, self.published.project_id)
        known = self.local_state['destinations'].get(configuration['root_id'])
        self.base = _base_snapshot(self.published, known)
        self.blocked = ''
        if not self.published.descends_from(reference, self.base['reference']):
            self.blocked = ('The selected version is older or comes from a different history. '
                            'Restore it into a new working copy and review/import the differences instead.')
        self.include_videos = self.selected['manifest'].get('include_videos', True)
        self.tree = _update_tree(self.project, self.include_videos)
        inventory = _inventory(self.project, self.include_videos)
        self.local_files = {}
        for index, (name, stat) in enumerate(sorted(inventory.items()), 1):
            _cancelled(cancelled)
            progress(f'Comparing local files {index}/{len(inventory)}: {name}')
            self.local_files[name] = {'size': stat[0], 'sha256': _digest(self.project / name)}
        baseline = _projection(self.base['manifest']['files'], self.include_videos)
        self.incoming = self.selected['manifest']['files']
        self.local_changes = _changes(baseline, self.local_files)
        self.remote_changes = _changes(baseline, self.incoming)
        self.changes = _changes(self.local_files, self.incoming)
        if self.local_changes and self.remote_changes and self.changes:
            self.blocked = ('Both local and published annotations/data changed since the starting version. '
                            'No automatic merging is performed, even across different frames or videos. '
                            'Publish a separate backup, then review/import the contributions in a new working copy.')
        self.local_only = bool(self.local_changes and not self.remote_changes)
        self.noop = self.local_only or (not self.changes and reference == self.base['reference'])
        local_policy = category_policies(_read_json(self.project / 'annotations/project.json'))
        remote_policy = category_policies(_verified_info(self.selected))
        self.policy_changes = [f'{category_label(c)}: {policy_text(local_policy[c])} -> {policy_text(remote_policy[c])}'
                               for c in local_policy if local_policy[c] != remote_policy[c]]
        self.split_changes = [name for name in self.changes if name.startswith('input_data/')
                              or (name.startswith('frames/') and name.endswith('/video_metadata.json'))]
        self.invalidated = _invalidated_files(self.project)
        affected = set(self.changes) | set(self.invalidated) | {'annotations/' + LOCAL_STATE}
        self.before = {name: file_record(_safe_child(self.project, name)) for name in affected}
        self.after = {name: self.incoming.get(name) for name in affected}
        state = deepcopy(self.local_state)
        heads = dict(self.selected['parents'])
        heads[reference[0]] = reference[1]
        state['destinations'][configuration['root_id']] = {'heads': heads}
        self.state_bytes = json.dumps(state, indent=2).encode()
        self.after['annotations/' + LOCAL_STATE] = bytes_record(self.state_bytes)
        self.rollback_bytes = sum(v['size'] for v in self.before.values() if v)
        self.stage_bytes = sum(v['size'] for v in self.after.values() if v)
        self.additional_bytes = self.rollback_bytes + self.stage_bytes
        self.rollback_root = rollback_root(configuration, self.project)
        self._check_unchanged()

    def _check_unchanged(self):
        if self.tree != _update_tree(self.project, self.include_videos):
            raise ValueError('Local files changed after preview. Preview again before applying.')
        if self.published.heads() != self.heads:
            raise ValueError('Published versions changed after preview. Refresh and preview again.')
        for snapshot in self.published.cache.values():
            if _digest(snapshot['manifest_path']) != snapshot['manifest_hash']:
                raise ValueError('A published manifest changed after preview. Nothing was replaced.')

    def apply(self, progress=None, cancelled=None):
        if self.blocked:
            raise SyncConflictError(self.blocked)
        if self.noop:
            return {'updated': False, 'path': str(self.project)}
        progress = progress or (lambda message: None)
        cancelled = cancelled or (lambda: False)
        require_no_update(self.project)
        self._check_unchanged()
        _cancelled(cancelled)
        return self._apply_transaction(progress, cancelled)

    def _apply_transaction(self, progress, cancelled):
        sources = {name: snapshot_file(self.selected['manifest_path'], self.selected['manifest'], name)
                   for name in self.changes if name in self.incoming}
        sources['annotations/' + LOCAL_STATE] = self.state_bytes
        result = apply_delta(
            self.project, self.published.configuration, self.before, self.after, sources,
            self.published.project_id, self.incoming, self.include_videos,
            self._check_unchanged, progress, cancelled)
        result['version'] = self.selected['reference'][1]
        return result


def recover_update(project, progress=None, cancelled=None):
    """Recover a stopped process only; callers must confirm no update is running."""
    project = Path(project).resolve()
    journal = update_journal_path(project)
    if journal.is_symlink():
        raise ValueError('Recovery journal must not be a symbolic link.')
    record = _read_json(journal)
    if record.get('format') == 2:
        if cancelled and cancelled():
            raise InterruptedError('Recovery cancelled; no files were changed.')
        return recover_delta(project, record, progress or (lambda text: None))
    if record.get('format') != 1 or record.get('project') != str(project):
        raise ValueError('Recovery journal belongs to another project.')
    stage = _safe_child(project.parent, record['stage'])
    backup = _safe_child(project.parent, record['backup'])
    if (stage.parent != project.parent or not stage.name.startswith('.bumblebox-update-stage-')
            or backup.parent != project.parent or not backup.name.startswith('bumblebox-before-update-')):
        raise ValueError('Unsafe recovery directory.')
    progress = progress or (lambda message: None)
    cancelled = cancelled or (lambda: False)
    _cancelled(cancelled)
    old = _safe_child(backup, 'project')
    if project.exists() and old.exists():
        expected = record.get('after_files')
        if not expected or set(k for k, v in _project_tree(project).items() if v) != set(expected):
            raise ValueError('The working copy changed after interruption. Preserve both folders for manual review.')
        for name, value in expected.items():
            _cancelled(cancelled)
            progress('Verifying recovered project: ' + name)
            if _digest(_safe_child(project, name)) != value['sha256']:
                raise ValueError('The working copy changed after interruption. Preserve both folders for manual review.')
        _cancelled(cancelled)
        status = 'The completed update was verified; its rollback backup remains available.'
    elif not project.exists() and old.exists():
        old.rename(project)
        status = 'The original local project was restored.'
    elif project.is_dir() and not old.exists():
        status = 'The original local project was never replaced.'
    else:
        raise ValueError('Neither a complete working copy nor its original is available. Manual recovery is required.')
    if stage.exists():
        shutil.rmtree(stage)
    if backup.exists() and not old.exists():
        shutil.rmtree(backup)
    journal.unlink()
    return {'path': str(project), 'message': status}
