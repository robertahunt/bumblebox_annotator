"""Persistent assigned copies and conservative, transactional incremental downloads."""

from copy import deepcopy
import json
import os
from pathlib import Path
from uuid import uuid4

from core.assigned_work import (ASSIGNMENTS, _check_owned, _global_files, _video_files,
    download_plan, download_work, read_assignments, require_contributor)
from core.contributors import utc_now
from core.project_delta import apply_delta, bytes_record, file_record
from core.project_sync import (WORK_STATE, SyncConflictError, _digest, _inventory,
    _read_json, _write_json, _safe_child, _sync_location, _uuid, require_no_update,
    require_working_project, snapshot_file)
from core.project_update import PublishedProject


def managed_directory(configuration, project_id, contributor):
    """Stable across project/profile renames and repository moves; never a cache."""
    root = Path(os.environ.get('XDG_DATA_HOME', Path.home() / '.local/share'))
    if not root.is_absolute():
        root = Path.home() / '.local/share'
    return root / 'BeeWhere/assigned-work' / _uuid(configuration['root_id']) / _uuid(project_id) / _uuid(contributor['id'])


def matching_copy(path, configuration, project_id, contributor):
    state = require_contributor(path, contributor)
    if not state or state['root_id'] != configuration['root_id'] or state['project_id'] != project_id:
        raise ValueError('This is not an assigned working copy for this project and contributor.')
    require_working_project(path)
    return state


def existing_copies(paths, configuration, project_id, contributor):
    """Inspect known paths only. Never pick the newest of several annotation copies."""
    found = []
    for value in paths:
        if not value:
            continue
        path = Path(value).expanduser().resolve()
        if path in found:
            continue
        try:
            matching_copy(path, configuration, project_id, contributor)
        except (ValueError, OSError, KeyError):
            continue
        found.append(path)
    return found


def _payload(value):
    return json.dumps(value, indent=2).encode()


class PreparedAssignedDownload:
    """Keep local edits; replace clean videos only; never merge mask rasters piecemeal.

    Every existing active video participates, even when only new videos were checked.
    This keeps one coherent remote baseline for future annotation-only uploads.
    """

    def __init__(self, configuration, project_id, contributor, videos, destination,
                 progress=lambda message: None, cancelled=lambda: False):
        self.configuration, self.project_id, self.contributor = configuration, project_id, contributor
        self.destination = Path(destination).expanduser().absolute()
        require_working_project(self.destination)
        require_no_update(self.destination)
        if self.destination.is_symlink():
            raise ValueError('Working copies must not be symbolic links.')
        root = Path(configuration['root']).resolve()
        resolved = self.destination.resolve()
        if resolved == root or root in resolved.parents or resolved in root.parents:
            raise ValueError('Working copies must be outside the shared backup root.')
        self.state = matching_copy(self.destination, configuration, project_id, contributor) if self.destination.exists() else None
        self.before, self.after, self.sources = {}, {}, {}
        self.kept_edits, self.retained = [], []
        self.new_videos, self.updated_videos = [], []
        view = PublishedProject.from_archive(configuration, project_id)
        assigned = read_assignments(view)['videos']
        selected = set(videos)
        if self.state:
            if self.state.get('pending_publication'):
                raise ValueError('Retry Upload Progress first to recover the previous upload receipt.')
            selected.update(v for v in self.state['assignments']
                            if v in assigned and assigned[v]['contributor']['id'] == contributor['id'])
        self.plan = download_plan(configuration, project_id, contributor, sorted(selected))
        self.assignment_record = file_record(view.home / ASSIGNMENTS)
        self.state_name = 'annotations/' + WORK_STATE
        if not self.state:
            self.new_videos = sorted(selected)
            self.bytes = self.plan['bytes']
            self.file_count = len(self.plan['names'])
            return

        progress('Checking the saved working copy; local edits will be preserved')
        self.state_record = file_record(self.destination / self.state_name)
        if _read_json(self.destination / self.state_name) != self.state:
            raise ValueError('The working copy changed while preparing the download. Check again.')
        self.inventory = _inventory(self.destination, True)
        self.local = {}
        for index, name in enumerate(self.inventory, 1):
            if cancelled():
                raise InterruptedError('Download check cancelled; no files were changed.')
            progress(f'Checking local file {index}/{len(self.inventory)}: {name}')
            self.local[name] = file_record(_safe_child(self.destination, name))
        self.base = view.snapshot(tuple(self.state['base_reference']))
        current = self.plan['snapshot']
        if not view.descends_from(current['reference'], self.base['reference']):
            raise SyncConflictError('Shared history differs from this working copy. Keep it for director review.')
        before, incoming = self.base['manifest']['files'], current['manifest']['files']
        if _global_files(before) != _global_files(incoming):
            raise SyncConflictError('Project policy or shared settings changed. Keep this copy for director review before updating.')
        baseline = self.state['local_files']
        dirty = {name for name in baseline.keys() | self.local.keys()
                 if baseline.get(name) != self.local.get(name)}
        if _global_files({name: None for name in dirty}):
            raise SyncConflictError('Local project settings changed; review them before downloading assigned work.')
        new_state = deepcopy(self.state)
        new_state['format'] = 2
        new_state['retained_assignments'] = dict(self.state.get('retained_assignments', {}))
        for video, assignment in self.state['assignments'].items():
            if video not in self.plan['assignments']:
                new_state['retained_assignments'][video] = assignment
                self.retained.append(video)

        for video, entry in self.plan['assignments'].items():
            old = self.state['assignments'].get(video)
            video_dirty = bool(_video_files({name: None for name in dirty}, video))
            if video in new_state['retained_assignments']:
                raise SyncConflictError('Previously retained work exists for ' + video +
                    '. Ask the director to reconcile it before downloading a new assignment for that video.')
            if video_dirty and (not old or old['id'] != entry['id'] or
                               _video_files(before, video) != _video_files(incoming, video)):
                raise SyncConflictError('Both the local work and the published work or assignment changed for ' + video +
                    '. Nothing was overwritten. Keep this copy and ask the director to reconcile the annotations.')
            if video_dirty:
                self.kept_edits.append(video)
            if old is None:
                self.new_videos.append(video)
            elif _video_files(before, video) != _video_files(incoming, video) or old['id'] != entry['id']:
                self.updated_videos.append(video)

            names = set(_video_files({n: None for n in self.plan['names']}, video))
            # Retain old frame images, but remove obsolete annotations only when the video is clean.
            old_annotations = {n for n in _video_files(baseline, video) if n.startswith('annotations/')}
            for name in names | old_annotations:
                if name in dirty:
                    continue
                record = incoming.get(name) if name in names else None
                if name.endswith('/video_metadata.json'):
                    meta = dict(self.plan['catalog'][video]['metadata'], project_import=True,
                                split=self.plan['catalog'][video]['split'], selected_frames=entry['frames'])
                    source = _payload(meta)
                    record = bytes_record(source)
                    new_state['metadata'][video] = meta
                else:
                    source = snapshot_file(current['manifest_path'], current['manifest'], name) if record else None
                if record:
                    new_state['local_files'][name] = record
                else:
                    new_state['local_files'].pop(name, None)
                if self.local.get(name) != record:
                    self.before[name], self.after[name], self.sources[name] = self.local.get(name), record, source

        new_state['assignments'] = self.plan['assignments']
        new_state['base_reference'] = list(current['reference'])
        # Ignore mutable remote progress timestamps when deciding whether the copy needs updating.
        for video, entry in new_state['assignments'].items():
            old = self.state['assignments'].get(video)
            if old and old['id'] == entry['id']:
                new_state['assignments'][video] = old
        self.new_state = new_state
        self.bytes = sum(r['size'] for r in self.after.values() if r)
        self.file_count = len(self.after)
        if new_state != self.state:
            payload = _payload(new_state)
            self.before[self.state_name] = self.state_record
            self.after[self.state_name], self.sources[self.state_name] = bytes_record(payload), payload

    def apply(self, progress=lambda message: None, cancelled=lambda: False):
        if not self.state:
            self.destination.parent.mkdir(parents=True, exist_ok=True)
            return download_work(self.configuration, self.project_id, self.contributor,
                                 self.plan['assignments'], self.destination, progress, cancelled, plan=self.plan)
        with _sync_location(self.destination, self.configuration, self.contributor):
            def check():
                view = PublishedProject.from_archive(self.configuration, self.project_id)
                _check_owned(read_assignments(view), self.plan['assignments'], self.contributor)
                if (view.heads() != self.plan['heads'] or file_record(view.home / ASSIGNMENTS) != self.assignment_record or
                        _digest(self.plan['snapshot']['manifest_path']) != self.plan['snapshot']['manifest_hash'] or
                        _digest(self.base['manifest_path']) != self.base['manifest_hash']):
                    raise ValueError('The shared project changed after preview. Check the download again.')
                if (file_record(self.destination / self.state_name) != self.state_record or
                        _inventory(self.destination, True) != self.inventory or
                        any(file_record(self.destination / name) != record for name, record in self.local.items())):
                    raise ValueError('Local files changed after preview. Save and check the download again.')
            check()
            if not self.after:
                return {'path': str(self.destination), 'updated': False, 'bytes': 0}
            post = dict(self.local)
            for name, record in self.after.items():
                if name == self.state_name:
                    continue  # The delta guards this hidden file separately from the public inventory.
                if record:
                    post[name] = record
                else:
                    post.pop(name, None)
            result = apply_delta(self.destination, self.configuration, self.before, self.after, self.sources,
                                 self.project_id, post, True, check, progress, cancelled)
            result['bytes'] = self.bytes
            if self.new_videos or self.updated_videos:
                try:
                    view = PublishedProject.from_archive(self.configuration, self.project_id)
                    data = read_assignments(view)
                    for video in self.new_videos + self.updated_videos:
                        data['videos'][video]['started_at'] = utc_now()
                    data['revision'] = str(uuid4())
                    _write_json(view.home / ASSIGNMENTS, data)
                except (OSError, ValueError, KeyError) as exc:
                    # Never report a committed local update as failed or leave a stale canvas open.
                    result['warning'] = 'Work was downloaded, but shared progress could not be recorded: ' + str(exc)
            return result
