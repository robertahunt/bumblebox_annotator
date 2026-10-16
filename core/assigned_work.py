"""Video assignments and small local workspaces with guarded annotation-only publishing.

Assignments are workflow controls, not authenticated access control. Publications
remain complete manifests: omitted local files are never interpreted as deletions.
"""

from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import shutil
import tempfile
from uuid import uuid4

from core.contributors import contributor_profile, utc_now
from core.project_delta import bytes_record, file_record
from core.project_sync import (
    WORK_STATE, PROJECT_MARKER, LOCATION_MARKERS, SyncConflictError,
    _read_json, _write_json, _safe_child, _uuid, _digest, _inventory,
    _clone_or_copy, _sync_location, object_path, snapshot_file,
    require_no_update, require_working_project,
)
from core.project_update import PublishedProject, _base_snapshot

ASSIGNMENTS = 'assignments.json'
ANNOTATION_KINDS = ('json', 'png', 'bbox')


def _video(value):
    if not isinstance(value, str) or not value or Path(value).name != value or value.startswith('.') or '\\' in value:
        raise ValueError('Invalid video identifier.')
    return value


def workspace(project):
    if not project:
        return None
    path = _safe_child(Path(project), 'annotations/' + WORK_STATE)
    if not path.exists():
        return None
    state = _read_json(path)
    if state.get('format') != 1 or state.get('kind') != 'assigned-work':
        raise ValueError('Unsupported assigned-work record. Keep this copy and update the app.')
    _uuid(state['project_id'])
    _uuid(state['contributor']['id'])
    _uuid(state['workspace_id'])
    identity = _read_json(_safe_child(Path(project), 'annotations/' + PROJECT_MARKER))
    if identity.get('project_id') != state['project_id'] or not state['assignments']:
        raise ValueError('Assigned-work identity does not match this project.')
    for video, assignment in state['assignments'].items():
        _video(video)
        _uuid(assignment['id'])
        if not assignment['frames'] or any(type(i) is not int or i < 0 for i in assignment['frames']):
            raise ValueError('Invalid assigned frames.')
    return state


def require_contributor(project, contributor):
    state = workspace(project)
    if state and (not contributor or state['contributor']['id'] != contributor['id']):
        raise ValueError('This working copy belongs to ' + state['contributor']['name'] +
                         '. Select/import that contributor profile or download your own assignments.')
    return state


def require_full_project(project):
    if workspace(project):
        raise ValueError('Project setup, imports, training, and split changes belong in the coordinator\'s full project.')


def require_assigned_frame(project, contributor, video, frame=None):
    state = require_contributor(project, contributor)
    if state and (video not in state['assignments'] or
                  (frame is not None and frame not in state['assignments'][video]['frames'])):
        raise ValueError('This frame is outside your assigned work.')


def _annotation_video(name):
    parts = Path(name).parts
    if len(parts) == 4 and parts[0] == 'annotations' and parts[1] in ANNOTATION_KINDS:
        return parts[2]
    return None


def _frame(name):
    match = re.fullmatch(r'frame_(\d+)(?:_[^.]+)?\.(jpg|png|json)', Path(name).name)
    return int(match[1]) if match else None


def _editable(name, assignments):
    video = _annotation_video(name)
    if video not in assignments:
        return False
    filename = Path(name).name
    return (filename == 'video_annotations.json' or
            bool(re.fullmatch(r'video_annotations(?:_[A-Za-z0-9_]+)?\.png', filename)) or
            _frame(name) in assignments[video]['frames'])


def _video_files(files, video):
    return {name: record for name, record in files.items()
            if _annotation_video(name) == video or name.startswith('frames/' + video + '/') or
            (name.startswith('input_data/') and Path(name).stem == video)}


def _global_files(files):
    return {name: record for name, record in files.items()
            if name.startswith('annotations/') and _annotation_video(name) is None}


def _verified_json(snapshot, name):
    path = snapshot_file(snapshot['manifest_path'], snapshot['manifest'], name)
    if file_record(path) != snapshot['manifest']['files'][name]:
        raise ValueError('Published metadata is missing or damaged: ' + name)
    return _read_json(path)


def latest_snapshot(view):
    heads = view.heads()
    return _base_snapshot(view, {'heads': heads}), heads


def catalog(snapshot):
    files = snapshot['manifest']['files']
    if any(name.startswith(('annotations/pkl/', 'annotations/frames/')) for name in files):
        raise ValueError('Assigned work requires current PNG/JSON annotations, not legacy pickle files.')
    result = {}
    for name, record in files.items():
        parts = Path(name).parts
        if len(parts) == 3 and parts[0] == 'frames' and re.fullmatch(r'frame_\d+\.jpg', parts[2]):
            video = _video(parts[1])
            entry = result.setdefault(video, {'frames': [], 'bytes': 0})
            entry['frames'].append(_frame(name))
            entry['bytes'] += record['size']
    for video, entry in result.items():
        entry['frames'].sort()
        meta_name = f'frames/{video}/video_metadata.json'
        meta = _verified_json(snapshot, meta_name) if meta_name in files else {}
        splits = {Path(name).parts[1] for name in files if name.startswith('input_data/') and Path(name).stem == video}
        split = next(iter(splits)) if len(splits) == 1 else meta.get('split')
        if len(splits) > 1 or split not in ('train', 'val', 'test', 'inference'):
            raise ValueError('Publish valid split metadata for video ' + video + ' before assigning it.')
        entry['split'] = split
        entry['metadata'] = meta
    return result


def parse_frames(text, available):
    if text.strip().lower() in ('all', ''):
        return sorted(available)
    result = set()
    for part in text.split(','):
        match = re.fullmatch(r'\s*(\d+)(?:\s*-\s*(\d+))?\s*', part)
        if not match:
            raise ValueError('Use All, frame numbers, or ranges such as 0,5-10.')
        start, end = int(match[1]), int(match[2] or match[1])
        if start > end or end - start > 1000000:
            raise ValueError('Invalid frame range.')
        result.update(range(start, end + 1))
    if not result or not result <= set(available):
        raise ValueError('Some requested frames are not in the published project.')
    return sorted(result)


def read_assignments(view):
    path = _safe_child(view.home, ASSIGNMENTS)
    if not path.exists():
        return {'format': 1, 'project_id': view.project_id, 'revision': None, 'coordinator': None, 'videos': {}}
    data = _read_json(path)
    if data.get('format') != 1 or data.get('project_id') != view.project_id:
        raise ValueError('Assignment metadata does not match the project.')
    for video, entry in data['videos'].items():
        _video(video)
        _uuid(entry['id'])
        _uuid(entry['contributor']['id'])
        if not entry['frames'] or any(type(i) is not int or i < 0 for i in entry['frames']):
            raise ValueError('Invalid assignment frame list.')
    return data


def assignment_status(view, entry):
    revision = view.heads().get(entry['contributor']['id'])
    visited = set()
    while revision:
        if revision in visited or len(visited) >= 1000:
            raise ValueError('Assignment publication history needs coordinator review.')
        visited.add(revision)
        snapshot = view.snapshot((entry['contributor']['id'], revision), metadata_only=True)
        submission = snapshot['manifest'].get('assigned_work', {})
        video = entry['video']
        if submission.get('assignments', {}).get(video) == entry['id']:
            submitted = view.snapshot(snapshot['reference'])
            current, _ = latest_snapshot(view)
            if _video_files(submitted['manifest']['files'], video) != _video_files(current['manifest']['files'], video):
                return 'Changed Since Submission', revision
            if video in submission.get('ready', []):
                return ('Accepted' if entry.get('accepted_revision') == revision else 'Ready for Review'), revision
            return 'In Progress', revision
        revision = snapshot['parents'].get(entry['contributor']['id'])
    return ('In Progress' if entry.get('started_at') else 'Assigned'), None


def save_assignments(project, configuration, contributor, videos, expected_revision):
    require_full_project(project)
    with _sync_location(Path(project).resolve(), configuration, contributor):
        view = PublishedProject(project, configuration)
        current = read_assignments(view)
        if current['coordinator'] and current['coordinator']['id'] != contributor['id']:
            raise ValueError('Only the coordinator profile can change assignments.')
        if current['revision'] != expected_revision:
            raise ValueError('Assignments changed. Refresh before saving.')
        snapshot, _ = latest_snapshot(view)
        available = catalog(snapshot)
        updated = {}
        for video, choice in videos.items():
            if video not in available:
                raise ValueError('Publish this video before assigning it: ' + video)
            actor = contributor_profile(dict(choice['contributor'], format=1, kind='bumblebox-contributor'))
            actor = {key: actor[key] for key in ('id', 'name')}
            if any(type(i) is not int or i < 0 for i in choice['frames']):
                raise ValueError('Invalid assigned frames for ' + video)
            frames = sorted(set(choice['frames']))
            if not frames or not set(frames) <= set(available[video]['frames']):
                raise ValueError('Invalid assigned frames for ' + video)
            old = current['videos'].get(video)
            if old and old['contributor']['id'] == actor['id'] and old['frames'] == frames:
                updated[video] = dict(old, contributor=actor)
            else:
                updated[video] = {'id': str(uuid4()), 'video': video, 'contributor': actor,
                                  'frames': frames, 'assigned_at': utc_now()}
        data = dict(current, revision=str(uuid4()), videos=updated,
                    coordinator={key: contributor[key] for key in ('id', 'name')})
        _write_json(view.home / ASSIGNMENTS, data)
        return data


def accept_submission(project, configuration, contributor, video, expected_revision):
    require_full_project(project)
    with _sync_location(Path(project).resolve(), configuration, contributor):
        view = PublishedProject(project, configuration)
        data = read_assignments(view)
        if not data['coordinator'] or data['coordinator']['id'] != contributor['id']:
            raise ValueError('Only the coordinator can accept work.')
        entry = data['videos'][video]
        status, revision = assignment_status(view, entry)
        if status != 'Ready for Review' or revision != expected_revision:
            raise ValueError('The submission changed or is not ready for review. Refresh first.')
        submitted = view.snapshot((entry['contributor']['id'], revision))['manifest']['files']
        expected = {name: record for name, record in submitted.items() if _annotation_video(name) == video}
        actual = {name: file_record(Path(project) / name) for name in _inventory(Path(project), False)
                  if _annotation_video(name) == video}
        if actual != expected:
            raise ValueError('Update Local Copy and review this video before accepting its submission.')
        entry['accepted_revision'] = revision
        data['revision'] = str(uuid4())
        _write_json(view.home / ASSIGNMENTS, data)


def my_projects(configuration, contributor):
    from core.project_sync import check_destination
    root = check_destination(configuration['root'], configuration['root_id'])
    projects = []
    for folder in sorted(root.iterdir()):
        if folder.is_symlink() or not folder.is_dir():
            continue
        marker = _safe_child(folder, LOCATION_MARKERS['project'])
        if not marker.is_file():
            continue
        identity = _read_json(marker)
        view = PublishedProject.from_archive(configuration, identity['id'])
        data = read_assignments(view)
        owned = {video: entry for video, entry in data['videos'].items()
                 if entry['contributor']['id'] == contributor['id']}
        if owned:
            projects.append({'id': view.project_id, 'name': identity['name'], 'videos': owned})
    return projects


def _check_owned(data, selected, contributor):
    for video, entry in selected.items():
        current = data['videos'].get(video)
        if (not current or current['id'] != entry['id'] or current['frames'] != entry['frames'] or
                current['contributor']['id'] != contributor['id']):
            raise SyncConflictError('Assignment changed for ' + video +
                                    '. Your local work is preserved; ask the coordinator to review it.')


def download_plan(configuration, project_id, contributor, videos):
    view = PublishedProject.from_archive(configuration, project_id)
    data = read_assignments(view)
    selected = {video: deepcopy(data['videos'][video]) for video in videos if video in data['videos']}
    if not selected or len(selected) != len(set(videos)):
        raise ValueError('Select assigned videos.')
    _check_owned(data, selected, contributor)
    snapshot, heads = latest_snapshot(view)
    if snapshot['manifest'].get('format') != 3:
        raise ValueError('The coordinator must publish a deduplicated backup before assigned work can be downloaded.')
    available = catalog(snapshot)
    from core.annotation_policy import policy_needs_confirmation
    if policy_needs_confirmation(_verified_json(snapshot, 'annotations/project.json')):
        raise ValueError('The coordinator must finish and publish the annotation policy before assigning work.')
    names = set()
    files = snapshot['manifest']['files']
    globals_ = _global_files(files)
    for name in files:
        parts = Path(name).parts
        if name in globals_:
            names.add(name)
        elif _annotation_video(name) in selected:
            names.add(name)
        elif len(parts) == 3 and parts[0] == 'frames' and parts[1] in selected:
            if parts[2] == 'video_metadata.json' or _frame(name) in selected[parts[1]]['frames']:
                names.add(name)
    for video, entry in selected.items():
        if video not in available or not set(entry['frames']) <= set(available[video]['frames']):
            raise ValueError('Assigned frames are no longer published: ' + video)
    return {'view': view, 'snapshot': snapshot, 'heads': heads, 'assignments': selected,
            'names': names, 'catalog': available,
            'bytes': sum(files[name]['size'] for name in names)}


def download_work(configuration, project_id, contributor, videos, destination, progress=None, cancelled=None):
    progress, cancelled = progress or (lambda text: None), cancelled or (lambda: False)
    plan = download_plan(configuration, project_id, contributor, videos)
    destination = Path(destination).expanduser().absolute()
    if destination.exists() or destination.is_symlink() or not destination.parent.is_dir():
        raise ValueError('Choose a new local folder with an existing parent.')
    require_working_project(destination)
    require_no_update(destination)
    root = plan['view'].root.resolve()
    resolved_destination = destination.resolve()
    if (resolved_destination == root or root in resolved_destination.parents or
            resolved_destination in root.parents):
        raise ValueError('Download into a local working folder outside the shared backup root.')
    if shutil.disk_usage(destination.parent).free < plan['bytes'] + 1024**2:
        raise OSError('Insufficient local storage for these assignments.')
    stage = Path(tempfile.mkdtemp(prefix='.assigned-download-', dir=destination.parent))
    reserved = False
    try:
        snapshot = plan['snapshot']
        for index, name in enumerate(sorted(plan['names']), 1):
            if cancelled():
                raise InterruptedError('Download cancelled; no working copy was created.')
            progress(f'Downloading {index}/{len(plan["names"])}: {name}')
            source = snapshot_file(snapshot['manifest_path'], snapshot['manifest'], name)
            target = _safe_child(stage, name)
            target.parent.mkdir(parents=True, exist_ok=True)
            _clone_or_copy(source, target)
            if file_record(target) != snapshot['manifest']['files'][name]:
                raise ValueError('Downloaded file failed verification: ' + name)
        for video, entry in plan['assignments'].items():
            meta = dict(plan['catalog'][video]['metadata'], project_import=True,
                        split=plan['catalog'][video]['split'], selected_frames=entry['frames'])
            _write_json(stage / 'frames' / video / 'video_metadata.json', meta)
        state = {'format': 1, 'kind': 'assigned-work', 'project_id': project_id,
                 'workspace_id': str(uuid4()), 'root_id': configuration['root_id'],
                 'contributor': {key: contributor[key] for key in ('id', 'name')},
                 'assignments': plan['assignments'], 'base_reference': list(snapshot['reference']),
                 'local_files': {name: file_record(stage / name) for name in _inventory(stage, False)},
                 'metadata': {video: _read_json(stage / 'frames' / video / 'video_metadata.json')
                              for video in plan['assignments']}}
        if _read_json(stage / 'annotations' / PROJECT_MARKER)['project_id'] != project_id:
            raise ValueError('Downloaded project identity does not match the assignment.')
        _write_json(stage / 'annotations' / WORK_STATE, state)
        with _sync_location(stage, configuration, contributor):
            view = PublishedProject.from_archive(configuration, project_id)
            data = read_assignments(view)
            _check_owned(data, plan['assignments'], contributor)
            if view.heads() != plan['heads'] or _digest(snapshot['manifest_path']) != snapshot['manifest_hash']:
                raise ValueError('The shared project changed during download. Download again.')
            if cancelled():
                raise InterruptedError('Download cancelled; no working copy was created.')
            destination.mkdir()
            reserved = True
            stage.replace(destination)
            reserved = False
            try:
                for video in plan['assignments']:
                    data['videos'][video]['started_at'] = utc_now()
                data['revision'] = str(uuid4())
                _write_json(view.home / ASSIGNMENTS, data)
            except OSError:
                pass  # The verified local copy is usable even if progress cannot be recorded.
        return {'path': str(destination), 'bytes': plan['bytes']}
    finally:
        if stage.exists():
            shutil.rmtree(stage)
        if reserved:
            destination.rmdir()


def _recover_receipt(project, state, view):
    pending = state.pop('pending_publication', None)
    if pending is None:
        return state
    actor, expected = pending['base_reference']
    if actor != state['contributor']['id']:
        raise ValueError('Invalid pending publication receipt.')
    revision = view.heads().get(actor)
    visited = set()
    while revision:
        if revision in visited or len(visited) >= 1000:
            raise ValueError('Publication receipt needs coordinator review.')
        visited.add(revision)
        snapshot = view.snapshot((actor, revision), metadata_only=True)
        if revision == expected:
            if (snapshot['manifest_hash'] != pending['manifest_hash'] or
                    snapshot['manifest'].get('assigned_work', {}).get('workspace_id') != state['workspace_id']):
                raise ValueError('Published receipt verification failed.')
            state.update({key: pending[key] for key in ('base_reference', 'local_files', 'metadata')})
            break
        revision = snapshot['parents'].get(actor)
    _write_json(project / 'annotations' / WORK_STATE, state)
    return state


def publish_work(project, configuration, contributor, ready=(), progress=None, cancelled=None):
    project = Path(project).resolve()
    state = require_contributor(project, contributor)
    if not state or state['root_id'] != configuration['root_id']:
        raise ValueError('Use the shared destination from which this work was downloaded.')
    progress, cancelled = progress or (lambda text: None), cancelled or (lambda: False)
    assignments = state['assignments']
    if not set(ready) <= assignments.keys():
        raise ValueError('Only assigned videos can be submitted for review.')
    with _sync_location(project, configuration, contributor) as home:
        view = PublishedProject(project, configuration)
        state = _recover_receipt(project, state, view)
        data = read_assignments(view)
        _check_owned(data, assignments, contributor)
        current, heads = latest_snapshot(view)
        base = view.snapshot(tuple(state['base_reference']))
        before, incoming = base['manifest']['files'], current['manifest']['files']
        if _global_files(before) != _global_files(incoming):
            raise SyncConflictError('Project policy or shared metadata changed. Keep this work for coordinator review.')
        for video in assignments:
            if _video_files(before, video) != _video_files(incoming, video):
                raise SyncConflictError('Published work changed in video ' + video +
                                        '. No annotations were overwritten; ask the coordinator to reconcile both copies.')
        inventory = _inventory(project, True)
        local = {}
        for index, name in enumerate(sorted(inventory), 1):
            if cancelled():
                raise InterruptedError('Publishing cancelled; local work remains saved.')
            progress(f'Checking assigned work {index}/{len(inventory)}: {name}')
            local[name] = file_record(_safe_child(project, name))
        files = dict(incoming)
        sources = {}
        changes = set()
        for name in state['local_files'].keys() | local.keys():
            if local.get(name) == state['local_files'].get(name):
                continue
            if _editable(name, assignments):
                changes.add(_annotation_video(name))
                if name in local:
                    files[name], sources[name] = local[name], project / name
                else:
                    files.pop(name, None)
                continue
            parts = Path(name).parts
            if len(parts) == 3 and parts[0] == 'frames' and parts[1] in assignments and parts[2] == 'video_metadata.json':
                video = parts[1]
                meta = _read_json(_safe_child(project, name))
                original = state['metadata'][video]
                clean = lambda value: {k: v for k, v in value.items() if k != 'max_mask_id'}
                if clean(meta) != clean(original) or type(meta.get('max_mask_id', 0)) is not int or meta.get('max_mask_id', 0) < 0:
                    raise ValueError('Assigned work cannot change video scope, calibration, frames, or splits.')
                remote_meta = _verified_json(current, name) if name in incoming else {}
                remote_meta['max_mask_id'] = meta.get('max_mask_id', 0)
                payload = json.dumps(remote_meta, indent=2).encode()
                if bytes_record(payload) != incoming.get(name):
                    files[name], sources[name] = bytes_record(payload), payload
                continue
            raise ValueError('A protected file changed outside the assigned annotations: ' + name)
        previous_ready = set()
        for video, entry in assignments.items():
            status, _ = assignment_status(view, entry)
            if status in ('Ready for Review', 'Accepted') and video not in changes:
                previous_ready.add(video)
        ready = sorted(previous_ready | set(ready))
        if files == incoming and set(ready) == previous_ready:
            statuses = [assignment_status(view, entry)[1] for entry in assignments.values()]
            if all(statuses):
                return {'path': str(current['manifest_path'].parent), 'copied_files': 0, 'ready': ready,
                        'unchanged': True}
        missing = {}
        for name, source in sources.items():
            digest = files[name]['sha256']
            target = object_path(view.home, digest)
            if target.exists():
                if file_record(target) != files[name]:
                    raise ValueError('A shared backup object is damaged. Publishing stopped.')
            else:
                missing[digest] = (name, source)
        # Legacy bases must be populated into the shared store before publishing a compact snapshot.
        if current['manifest'].get('format') != 3:
            raise ValueError('The coordinator must publish a deduplicated backup before students can publish work.')
        required = sum(files[name]['size'] for name, _ in missing.values()) + 65536 + len(files) * 512
        if shutil.disk_usage(view.root).free < required:
            raise OSError('Insufficient shared storage for the changed annotations.')
        for digest, (name, source) in missing.items():
            if cancelled():
                raise InterruptedError('Publishing cancelled; local work remains saved.')
            progress('Uploading changed annotation: ' + name)
            target = object_path(view.home, digest)
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_name('.pending-' + uuid4().hex)
            try:
                if isinstance(source, bytes):
                    temporary.write_bytes(source)
                else:
                    shutil.copy2(source, temporary)
                if file_record(temporary) != files[name]:
                    raise ValueError('Changed annotation failed verification: ' + name)
                temporary.replace(target)
            finally:
                temporary.unlink(missing_ok=True)
        if inventory != _inventory(project, True):
            raise ValueError('Local files changed while publishing. Save and retry.')
        _check_owned(read_assignments(view), assignments, contributor)
        if (_digest(current['manifest_path']) != current['manifest_hash'] or
                _digest(base['manifest_path']) != base['manifest_hash']):
            raise ValueError('A published manifest changed during upload. No latest pointer was changed.')
        if view.heads() != heads or cancelled():
            raise InterruptedError('Publishing stopped before completion. Local work remains saved.')
        revision = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S') + '-' + uuid4().hex[:8]
        revisions = _safe_child(home, 'revisions', mkdir=True)
        stage = _safe_child(revisions, '.pending-' + revision, mkdir=True)
        try:
            manifest = dict(current['manifest'], files=files, base_heads=heads,
                            contributor={key: contributor[key] for key in ('id', 'name')},
                            created_at=utc_now(), device=contributor.get('device'), publication='latest',
                            assigned_work={'workspace_id': state['workspace_id'],
                                           'assignments': {v: a['id'] for v, a in assignments.items()}, 'ready': ready})
            _write_json(stage / 'manifest.json', manifest)
            # Persist a recovery receipt before advancing the shared latest pointer.
            # A disconnect after publication must not turn a safe retry into a conflict.
            receipt = {'base_reference': [contributor['id'], revision], 'local_files': local,
                       'metadata': {v: _read_json(project / 'frames' / v / 'video_metadata.json')
                                    for v in assignments}, 'manifest_hash': _digest(stage / 'manifest.json')}
            _write_json(project / 'annotations' / WORK_STATE, dict(state, pending_publication=receipt))
            stage.rename(revisions / revision)
            _write_json(home / 'latest.json', {'revision': revision})
        finally:
            if stage.exists():
                shutil.rmtree(stage)
        state.update({key: receipt[key] for key in ('base_reference', 'local_files', 'metadata')})
        result = {'path': str(revisions / revision), 'copied_files': len(missing), 'ready': ready}
        try:
            _write_json(project / 'annotations' / WORK_STATE, state)
        except OSError as exc:
            result['warning'] = 'Work was published. Its local receipt will be recovered on the next Publish Work: ' + str(exc)
        return result
