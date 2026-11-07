"""Pinned, video-scoped director review with atomic approval publications."""

from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
from uuid import uuid4

from core.annotation_policy import policy_signature
from core.assigned_work import (ANNOTATION_KINDS, _editable, _verified_json, assignment_status,
    latest_snapshot, read_assignments, require_full_project, video_digest)
from core.contributors import utc_now
from core.project_delta import apply_delta, bytes_record, file_record
from core.project_sync import (SyncConflictError, _digest, _read_json, _read_local_state,
    _safe_child, _sync_location, _write_json, object_path, require_no_update, snapshot_file)
from core.project_update import PublishedProject, _base_snapshot


def _director(view, contributor):
    data = read_assignments(view)
    if (data.get('coordinator') or {}).get('id') != contributor.get('id'):
        raise ValueError('Only the project director can review submitted work.')
    return data


def review_queue(project, configuration, contributor, progress=lambda text: None, cancelled=lambda: False):
    require_full_project(project)
    view = PublishedProject(project, configuration)
    data = _director(view, contributor)
    current, heads = latest_snapshot(view)
    # Discover ready receipts once per contributor, not once per assigned video.
    # These manifests contain metadata; no annotation/image payloads are scanned.
    progress('Finding submitted work in published assignment records')
    candidates = set()
    for actor, revision in heads.items():
        pending = {v: e for v, e in data['videos'].items() if e['contributor']['id'] == actor}
        visited = set()
        while pending and revision:
            if cancelled():
                raise InterruptedError('Submission check cancelled.')
            if revision in visited or len(visited) >= 1000:
                raise ValueError('Assignment publication history needs coordinator review.')
            visited.add(revision)
            snapshot = view.snapshot((actor, revision), metadata_only=True)
            work = snapshot['manifest'].get('assigned_work', {})
            for video, entry in list(pending.items()):
                if work.get('assignments', {}).get(video) == entry['id']:
                    if video in work.get('ready', []):
                        candidates.add(video)
                    del pending[video]
            revision = snapshot['parents'].get(actor)
    result = []
    for index, video in enumerate(sorted(candidates), 1):
        entry = data['videos'][video]
        progress(f'Checking submitted video {index}/{len(candidates)}: {video}')
        status, revision = assignment_status(view, entry, heads=heads, current=current, cancelled=cancelled)
        if status == 'Ready for Review':
            result.append(dict(deepcopy(entry), revision=revision))
    if view.heads(validate_snapshots=False) != heads or read_assignments(view)['revision'] != data['revision']:
        raise ValueError('Submissions changed during the check. Refresh the review queue.')
    return result


def _selected(project, configuration, contributor, video, revision, assignment_id, *, allow_accepted=False):
    require_full_project(project)
    require_no_update(project)
    view = PublishedProject(project, configuration)
    data = _director(view, contributor)
    entry = data['videos'].get(video)
    if not entry or entry['id'] != assignment_id:
        raise ValueError('The assignment changed. Refresh the review queue.')
    current, heads = latest_snapshot(view)
    status, actual = assignment_status(view, entry, heads=heads, current=current)
    if actual != revision or status not in (('Ready for Review', 'Accepted') if allow_accepted else ('Ready for Review',)):
        raise ValueError('The submission changed or is no longer ready. Refresh the review queue; local edits are kept.')
    original = view.snapshot((entry['contributor']['id'], revision))
    signature = policy_signature(_verified_json(original, 'annotations/project.json'))
    if (policy_signature(_read_json(Path(project) / 'annotations/project.json')) != signature or
            policy_signature(_verified_json(current, 'annotations/project.json')) != signature):
        raise SyncConflictError('Annotation policy differs from the submission. Reconcile project policies before reviewing.')
    return view, entry, original, current, heads, signature, status


def _local_annotations(project, entry):
    records = {}
    for kind in ANNOTATION_KINDS:
        folder = _safe_child(project, f'annotations/{kind}/{entry["video"]}')
        if not folder.exists():
            continue
        for path in folder.iterdir():
            name = path.relative_to(project).as_posix()
            if _editable(name, {entry['video']: entry}):
                records[name] = file_record(_safe_child(project, name))
    return records


def _state_name(video):
    return 'annotations/.submission-review-' + bytes_record(video.encode())['sha256'][:16] + '.json'


def _state(project, video):
    path = _safe_child(project, _state_name(video))
    return _read_json(path) if path.exists() else {}


def _same_session(state, view, entry, contributor, revision):
    return (state.get('format') == 1 and state.get('project_id') == view.project_id and
            state.get('root_id') == view.configuration['root_id'] and
            state.get('director_id') == contributor['id'] and state.get('revision') == revision and
            all(state.get('assignment', {}).get(k) == entry.get(k)
                for k in ('id', 'video', 'contributor', 'frames', 'task')) and not state.get('approved'))


def _verify_frames(project, entry, snapshot, *, missing_ok=False):
    records = {}
    for frame in entry['frames']:
        name = f'frames/{entry["video"]}/frame_{frame:06d}.jpg'
        expected = snapshot['manifest']['files'].get(name)
        actual = file_record(_safe_child(project, name))
        if expected is None or (actual != expected and not (missing_ok and actual is None)):
            raise SyncConflictError('A review frame is missing or differs from the submitted image: ' + name)
        records[name] = actual
    return records


class PreparedSubmissionReview:
    """Pin both versions; conflicting local work requires an explicit choice."""

    def __init__(self, project, configuration, contributor, video, revision, assignment_id,
                 progress=lambda text: None, cancelled=lambda: False):
        if cancelled():
            raise InterruptedError('Review opening cancelled.')
        progress('Checking the submitted version and local annotations')
        self.project, self.configuration = Path(project).resolve(), dict(configuration)
        self.contributor, self.video, self.revision = dict(contributor), video, revision
        self.selected = (video, revision, assignment_id)
        view, entry, original, current, heads, signature, _ = _selected(
            self.project, configuration, contributor, *self.selected)
        self.view, self.entry, self.original = view, deepcopy(entry), original
        self.local = _local_annotations(self.project, entry)
        self.frames = _verify_frames(self.project, entry, original, missing_ok=True)
        state = _state(self.project, video)
        self.resuming = _same_session(state, view, entry, contributor, revision)
        self.conflicts = []
        self.before, self.after, self.sources = {}, {}, {}
        if self.resuming:
            if state.get('manifest_hash') != original['manifest_hash']:
                raise ValueError('The pinned submission manifest changed. Reconcile the shared backup before reviewing.')
            if any(record is None for record in self.frames.values()):
                raise ValueError('A frame was removed during review. Restore it before continuing.')
            self.state = state
            return
        unfinished = bool(state and not state.get('approved'))
        incoming = {n: r for n, r in original['manifest']['files'].items() if _editable(n, {video: entry})}
        self.incoming = incoming
        known = _read_local_state(self.project, view.project_id)['destinations'].get(configuration['root_id'])
        base = _base_snapshot(view, known)
        baseline = base['manifest']['files']
        if state.get('approved') and state.get('project_id') == view.project_id:
            baseline = state['local_baseline']
        for name in self.local.keys() | incoming.keys():
            if cancelled():
                raise InterruptedError('Review opening cancelled.')
            local, remote = self.local.get(name), incoming.get(name)
            if local == remote:
                continue
            if unfinished or local != baseline.get(name):
                self.conflicts.append(name)
            self.before[name], self.after[name] = local, remote
            if remote:
                self.sources[name] = snapshot_file(original['manifest_path'], original['manifest'], name)
        for name, record in self.frames.items():
            if cancelled():
                raise InterruptedError('Review opening cancelled.')
            if record is None:
                self.before[name] = None
                self.after[name] = original['manifest']['files'][name]
                self.sources[name] = snapshot_file(original['manifest_path'], original['manifest'], name)
        self.metadata_name = f'frames/{video}/video_metadata.json'
        metadata_path = _safe_child(self.project, self.metadata_name)
        remote_meta = _verified_json(original, self.metadata_name)
        local_meta = _read_json(metadata_path) if metadata_path.exists() else remote_meta
        if local_meta.get('split') != remote_meta.get('split'):
            raise SyncConflictError('The local and submitted video splits differ. Resolve the split before review.')
        metadata = dict(local_meta, max_mask_id=max(local_meta.get('max_mask_id', 0), remote_meta.get('max_mask_id', 0)))
        self.state = {'format': 1, 'project_id': view.project_id, 'root_id': configuration['root_id'],
                      'director_id': contributor['id'], 'revision': revision, 'assignment': deepcopy(entry),
                      'session_id': str(uuid4()), 'opened_at': utc_now(), 'policy_signature': signature,
                      'manifest_hash': original['manifest_hash']}
        self.conflicts.sort()
        for name, value in ((self.metadata_name, metadata), (_state_name(video), self.state)):
            payload = json.dumps(value, indent=2).encode()
            before, after = file_record(_safe_child(self.project, name)), bytes_record(payload)
            if before != after:
                self.before[name], self.after[name], self.sources[name] = before, after, payload

    def apply(self, progress=lambda text: None, cancelled=lambda: False, *, conflict_choice=None):
        if self.conflicts:
            if conflict_choice not in ('local', 'submitted'):
                raise SyncConflictError('Choose local or submitted annotations in the comparison before opening this review.')
            # Select a complete annotation set, including completion records and
            # shared masks. Never mix a JSON index with the other version's PNGs.
            for name in self.local.keys() | self.incoming.keys():
                local = self.local.get(name)
                target = local if conflict_choice == 'local' else self.incoming.get(name)
                self.before[name], self.after[name] = local, target
                if target:
                    self.sources[name] = (_safe_child(self.project, name) if conflict_choice == 'local' else
                        snapshot_file(self.original['manifest_path'], self.original['manifest'], name))
                else:
                    self.sources.pop(name, None)
            self.state['conflict_resolution'] = {'choice': conflict_choice, 'files': self.conflicts,
                                                'resolved_by': self.contributor['id'], 'resolved_at': utc_now()}
            payload = json.dumps(self.state, indent=2).encode()
            self.after[_state_name(self.video)], self.sources[_state_name(self.video)] = bytes_record(payload), payload
        elif conflict_choice is not None:
            raise ValueError('This review has no local annotation conflict to resolve.')
        def check():
            view = PublishedProject(self.project, self.configuration)
            entry = _director(view, self.contributor)['videos'].get(self.video)
            if entry != self.entry or assignment_status(view, entry) != ('Ready for Review', self.revision):
                raise ValueError('Submission changed after preview. Refresh the queue.')
            if self.local != _local_annotations(self.project, self.entry):
                raise ValueError('Local annotations changed after preview. Save and retry.')
            if self.resuming and _state(self.project, self.video) != self.state:
                raise ValueError('The local review session changed after preview. Reopen the submission.')
            if self.frames != _verify_frames(self.project, self.entry, self.original, missing_ok=True):
                raise ValueError('Local review images changed after preview.')
            current, _ = latest_snapshot(view)
            if (policy_signature(_verified_json(current, 'annotations/project.json')) != self.state['policy_signature'] or
                    policy_signature(_read_json(self.project / 'annotations/project.json')) != self.state['policy_signature'] or
                    _digest(self.original['manifest_path']) != self.original['manifest_hash']):
                raise ValueError('Project policy or submission changed after preview.')
        require_no_update(self.project)
        check()
        if cancelled():
            raise InterruptedError('Review opening cancelled.')
        if self.resuming:
            return {'updated': False, 'resuming': True, 'entry': self.entry}
        result = apply_delta(self.project, self.configuration, self.before, self.after, self.sources,
            self.view.project_id, {n: r for n, r in self.after.items() if r}, False,
            check, progress, cancelled, scoped=True)
        return dict(result, entry=self.entry)


def approve_review(project, configuration, contributor, video, revision, assignment_id,
                   progress=lambda text: None, cancelled=lambda: False):
    project = Path(project).resolve()
    with _sync_location(project, configuration, contributor) as home:
        view, entry, original, current, heads, signature, status = _selected(
            project, configuration, contributor, video, revision, assignment_id, allow_accepted=True)
        state = _state(project, video)
        local = _local_annotations(project, entry)
        if status == 'Accepted':
            receipt = current['manifest'].get('submission_reviews', {}).get(video, {})
            expected = {n: r for n, r in current['manifest']['files'].items() if _editable(n, {video: entry})}
            if receipt.get('session_id') == state.get('session_id') and local == expected:
                result = {'unchanged': True, 'video': video, 'path': str(current['manifest_path'].parent)}
                try:
                    _write_json(project / _state_name(video), dict(state, approved=True, local_baseline=local))
                except OSError as exc:
                    result['warning'] = 'Approval was published; local review receipt could not be updated: ' + str(exc)
                return result
            raise ValueError('This submission was already accepted. Refresh the review queue.')
        if not _same_session(state, view, entry, contributor, revision):
            raise ValueError('Open this exact submission for review before approving it.')
        if state.get('manifest_hash') != original['manifest_hash']:
            raise ValueError('The pinned submission manifest changed. Reconcile the shared backup before approving.')
        _verify_frames(project, entry, original)
        files = dict(current['manifest']['files'])
        scope = {n for n in files if _editable(n, {video: entry})} | local.keys()
        sources = {}
        for name in scope:
            if name not in local:
                files.pop(name, None)
            else:
                if files.get(name) != local[name]:
                    sources[name] = _safe_child(project, name)
                files[name] = local[name]
        metadata_name = f'frames/{video}/video_metadata.json'
        metadata_record = file_record(_safe_child(project, metadata_name))
        local_meta = _read_json(project / metadata_name)
        metadata = _verified_json(current, metadata_name)
        value = local_meta.get('max_mask_id', 0)
        if type(value) is not int or value < 0:
            raise ValueError('Invalid local maximum instance ID.')
        metadata['max_mask_id'] = max(metadata.get('max_mask_id', 0), value)
        payload = json.dumps(metadata, indent=2).encode()
        if bytes_record(payload) != files.get(metadata_name):
            files[metadata_name], sources[metadata_name] = bytes_record(payload), payload
        if current['manifest'].get('format') != 3:
            raise ValueError('Publish a deduplicated project backup before approving submissions.')
        from core.backup_integrity import require_healthy_references
        require_healthy_references(view.home, configuration, view.project_id, files)
        required = sum(files[n]['size'] for n in sources) + 65536 + len(files) * 512
        if shutil.disk_usage(view.root).free < required:
            raise OSError('Insufficient shared storage for the reviewed annotations.')
        for name, source in sources.items():
            if cancelled():
                raise InterruptedError('Approval cancelled. Corrections remain saved locally.')
            progress('Uploading reviewed annotation: ' + name)
            target = object_path(view.home, files[name]['sha256'])
            if target.exists():
                if file_record(target) != files[name]:
                    raise ValueError('A shared annotation object is damaged: ' + name)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_name('.pending-' + uuid4().hex)
            try:
                if isinstance(source, bytes):
                    temporary.write_bytes(source)
                else:
                    shutil.copy2(source, temporary)
                if file_record(temporary) != files[name]:
                    raise ValueError('Reviewed annotation failed verification: ' + name)
                temporary.replace(target)
            finally:
                temporary.unlink(missing_ok=True)
        from core.annotation_work import build_index
        annotation_work = build_index(files, lambda name: object_path(view.home, files[name]['sha256']),
            previous=current['manifest'].get('annotation_work'), cancelled=cancelled)
        fresh = _selected(project, configuration, contributor, video, revision, assignment_id)
        if (fresh[4] != heads or local != _local_annotations(project, entry) or
                state != _state(project, video) or
                metadata_record != file_record(project / metadata_name) or
                _digest(original['manifest_path']) != original['manifest_hash'] or
                _digest(current['manifest_path']) != current['manifest_hash']):
            raise ValueError('Local or shared work changed during approval. Save and retry.')
        _verify_frames(project, entry, original)
        if cancelled():
            raise InterruptedError('Approval cancelled. Corrections remain saved locally.')
        receipts = dict(current['manifest'].get('submission_reviews', {}))
        receipts[video] = {'assignment_id': entry['id'], 'submission': list(original['reference']),
            'submitted_digest': video_digest(original['manifest']['files'], video),
            'reviewed_digest': video_digest(files, video), 'session_id': state['session_id'],
            'reviewed_at': utc_now(), 'reviewer': {k: contributor[k] for k in ('id', 'name')}}
        if state.get('conflict_resolution'):
            receipts[video]['conflict_resolution'] = state['conflict_resolution']
        manifest = dict(current['manifest'], files=files, base_heads=heads, annotation_work=annotation_work,
            contributor={k: contributor[k] for k in ('id', 'name')}, device=contributor.get('device'),
            created_at=utc_now(), publication='latest', submission_reviews=receipts)
        manifest.pop('assigned_work', None)
        token = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S') + '-' + uuid4().hex[:8]
        revisions = _safe_child(home, 'revisions', mkdir=True)
        stage = _safe_child(revisions, '.pending-' + token, mkdir=True)
        try:
            _write_json(stage / 'manifest.json', manifest)
            stage.rename(revisions / token)
            # This single pointer publishes both the corrections and the approval.
            _write_json(home / 'latest.json', {'revision': token})
        finally:
            if stage.exists():
                shutil.rmtree(stage)
        result = {'video': video, 'path': str(revisions / token), 'copied_files': len(sources)}
        try:
            _write_json(project / _state_name(video), dict(state, approved=True, local_baseline=local))
        except OSError as exc:
            result['warning'] = 'Approval was published; local review receipt could not be updated: ' + str(exc)
        return result
