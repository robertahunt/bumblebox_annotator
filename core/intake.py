"""Take contributors' latest published work on their assigned videos into the director's project.

Without this, assigned work reaches the director's project only when a video is
submitted and opened for review. Intake copies each contributor's latest
publication for their assigned frames and video-wide masks, finished or not.

Guardrails:
  - only the files of an assignment are touched (the same set a review loads);
  - annotations changed here since their last version was taken in are never
    written over: the video is skipped and reported instead;
  - every write goes through the verified, journalled update that reviews use,
    which keeps the replaced files in a rollback bundle;
  - what intake brings in is held: left out of COCO export and training until
    the director accepts it.
"""

from copy import deepcopy
import json
from pathlib import Path
from uuid import uuid4

import cv2
import numpy as np

from core.annotation_policy import policy_signature
from core.assigned_work import (ASSIGNMENTS, _editable, _frame, _verified_json, assignment_status, latest_snapshot,
                                read_assignments, require_full_project)
from core.contributors import utc_now
from core.project_delta import apply_delta, bytes_record, delete_rollback_bundle, file_record, rollback_bundles
from core.project_sync import (SyncConflictError, _read_json, _read_local_state, _safe_child, _sync_location,
                               _write_json, require_no_update, snapshot_file)
from core.project_update import PublishedProject, _base_snapshot

# A dot file: inventories skip it, so it is never published, exported or downloaded.
STATE = 'annotations/.intake.json'
KEPT_BUNDLES = 30
CHANGED_HERE = 'This video was changed here after their last version was taken in.'
REVIEW_OPEN = 'A review of this video is open here. Approve it or exit the review first.'


def read_state(project):
    path = _safe_child(Path(project), STATE)
    if not path.exists():
        return {'format': 1, 'videos': {}}
    state = _read_json(path)
    if state.get('format') != 1 or not isinstance(state.get('videos'), dict):
        raise ValueError('Unsupported intake record: ' + str(path))
    return state


def _write_state(project, state):
    require_no_update(project)
    _write_json(_safe_child(Path(project), STATE), state)


SHARED_CATEGORIES = ('chamber', 'hive', 'pollen')  # the masks a video can carry for all of its frames


def held_work(project):
    """{video: {'frames': [...], 'shared': [categories]}}: contributor work here that is not accepted yet.

    'frames' are frames whose own annotations a contributor changed; 'shared' are the
    video-wide masks they changed. Export leaves both out, and nothing else.
    """
    return {video: {'frames': list(record['held'].get('frames', [])), 'shared': list(record['held'].get('shared', []))}
            for video, record in read_state(project)['videos'].items() if record.get('held')}


def is_held(held, video, frame):
    return frame in held.get(video, {}).get('frames', ())


def held_categories(held, video):
    return set(held.get(video, {}).get('shared', ()))


def _hold(record, frames, shared):
    if frames or shared:
        held = record.get('held') or {}
        record['held'] = {'frames': sorted(set(held.get('frames', [])) | set(frames)),
                          'shared': sorted(set(held.get('shared', [])) | set(shared))}


def _rows(data, *, by_category=False, masks=True):
    """A saved annotation list without its bookkeeping, so that a re-save alone does not count as a change."""
    value = json.loads(data) if data else []
    rows = value.get('annotations', []) if isinstance(value, dict) else value
    flat, grouped = [], {}
    for row in rows:
        if row.get('from_mask'):
            continue  # a box computed from a mask; the mask image itself is compared
        # Newer saves add who-made-it records, a category name beside the category number, and for a
        # mask (`masks`: a mask list, not a box list) its outline box and a second encoding. Nobody draws these.
        text = json.dumps({key: item for key, item in row.items()
                           if key not in ('provenance', 'mask_coco_rle') and
                           not (key == 'category' and 'category_id' in row) and
                           not (key == 'bbox' and masks and not row.get('bbox_only'))}, sort_keys=True)
        flat.append(text)
        grouped.setdefault(row.get('category'), []).append(text)
    return {category: sorted(items) for category, items in grouped.items()} if by_category else sorted(flat)


def _same_pixels(first, second):
    arrays = []
    for data in (first, second):
        array = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_UNCHANGED) if data else None
        if data and array is None:
            raise ValueError('Unreadable mask image')
        arrays.append(array)
    if all(array is None or not array.any() for array in arrays):
        return True  # no mask on either side
    return arrays[0] is not None and arrays[1] is not None and np.array_equal(arrays[0], arrays[1])


def real_changes(changed, read_local, read_theirs):
    """(frames, video-wide categories) whose annotations truly differ, among files whose bytes differ.

    Opening a frame in a working copy can create an empty file or re-save one with added
    bookkeeping. Those are copied like everything else but are not work to hold back.
    """
    frames, shared = set(), set()
    for name in changed:
        kind, leaf = name.split('/')[1], name.split('/')[-1]
        if kind == 'work':
            continue  # a finished mark, not annotation content
        mine, theirs = read_local(name), read_theirs(name)
        try:
            if leaf == 'video_annotations.json':
                a, b = _rows(mine, by_category=True), _rows(theirs, by_category=True)
                shared |= {category for category in a.keys() | b.keys() if a.get(category) != b.get(category)}
            elif leaf.startswith('video_'):
                if not _same_pixels(mine, theirs):
                    category = leaf[len('video_annotations_'):-len('.png')]
                    shared |= {category} if category in SHARED_CATEGORIES else set(SHARED_CATEGORIES)
            elif not (_same_pixels(mine, theirs) if leaf.endswith('.png') else
                      _rows(mine, masks=kind == 'json') == _rows(theirs, masks=kind == 'json')):
                frames.add(_frame(name))
        except (ValueError, TypeError, AttributeError, KeyError):
            # Unreadable on either side: treat it as changed.
            if leaf.startswith('video_'):
                shared |= set(SHARED_CATEGORIES)
            else:
                frames.add(_frame(name))
    return frames - {None}, {category for category in shared if category}


def _assignment_files(files, entry):
    return {name: record for name, record in files.items() if _editable(name, {entry['video']: entry})}


def mark_accepted(project, entry, revision, files):
    """Record an acceptance: nothing of this video is held, and `files` is the last version seen from them."""
    state = read_state(project)
    old = state['videos'].get(entry['video']) or {}
    state['videos'][entry['video']] = {
        'assignment_id': entry['id'], 'contributor': dict(entry['contributor']), 'frames': list(entry['frames']),
        'task': entry.get('task'), 'revision': revision, 'files': files, 'taken_at': old.get('taken_at'),
        'held': None, 'skipped': None, 'kept_local': None, 'accepted_at': utc_now()}
    _write_state(project, state)


def release(project, videos):
    """Stop holding these videos without accepting them. For work whose assignment is gone, for example."""
    state = read_state(project)
    for video in videos:
        if video in state['videos']:
            state['videos'][video].update(held=None, released_at=utc_now())
    _write_state(project, state)


def fingerprint(project, configuration):
    """Changes whenever anyone publishes or assignments change: a cheap reason to plan an intake."""
    view = PublishedProject(project, configuration)
    return read_assignments(view)['revision'], tuple(sorted(view.heads(validate_snapshots=False).items()))


class PreparedIntake:
    """Decide, without writing anything, what the contributors' latest publications change here."""

    def __init__(self, project, configuration, contributor, *, replace=(), keep=(),
                 progress=lambda text: None, cancelled=lambda: False):
        from core.submission_review import _director
        self.project, self.configuration = Path(project).resolve(), dict(configuration)
        self.contributor, self.replace, self.keep = dict(contributor), set(replace), set(keep)
        require_full_project(self.project)
        require_no_update(self.project)
        progress('Checking published work on assigned videos')
        self.view = view = PublishedProject(self.project, configuration)
        data = _director(view, contributor)
        self.current, self.heads = latest_snapshot(view)
        self.assignments_revision = data['revision']
        self.old_state = read_state(self.project)
        self.state = deepcopy(self.old_state)
        self.before, self.after, self.sources = {}, {}, {}
        self.taken, self.skipped, self.entries, self.local, self.frames = [], {}, {}, {}, {}
        self._base = self._policies = None
        for video, entry in sorted(data['videos'].items()):
            if cancelled():
                raise InterruptedError('Intake cancelled.')
            actor = entry['contributor']['id']
            if actor == contributor['id'] or actor not in self.heads:
                continue  # the director works in this project directly; others may not have published yet
            head = self.heads[actor]
            record = self._record(video, entry)
            try:
                snapshot = view.snapshot((actor, head))
                if snapshot['manifest'].get('assigned_work', {}).get('assignments', {}).get(video) != entry['id']:
                    continue  # nothing published under this assignment yet
                progress(f'Checking {video}')
                self._plan(video, entry, record, snapshot, head, cancelled)
            except (SyncConflictError, ValueError, OSError) as exc:
                self._skip(video, record, head, str(exc))
        if self.taken:
            payload = json.dumps(self.state, indent=2).encode()
            self.before[STATE] = file_record(_safe_child(self.project, STATE))
            self.after[STATE], self.sources[STATE] = bytes_record(payload), payload

    def _record(self, video, entry):
        old = self.old_state['videos'].get(video) or {}
        # A changed assignment starts afresh, but work already held stays held until it is accepted.
        record = deepcopy(old) if old.get('assignment_id') == entry['id'] else {'held': deepcopy(old.get('held'))}
        record.update(assignment_id=entry['id'], contributor=dict(entry['contributor']),
                      frames=list(entry['frames']), task=entry.get('task'))
        return record

    def _keep(self, video, record):
        self.state['videos'][video] = record

    def _skip(self, video, record, head, reason):
        record['skipped'] = {'revision': head, 'reason': reason,
                             'at': (record.get('skipped') or {}).get('at') or utc_now()}
        self.skipped[video] = reason
        self._keep(video, record)

    def _same_policy(self, snapshot):
        if self._policies is None:
            self._policies = {
                'local': policy_signature(_read_json(self.project / 'annotations/project.json')),
                'current': policy_signature(_verified_json(self.current, 'annotations/project.json'))}
        theirs = policy_signature(_verified_json(snapshot, 'annotations/project.json'))
        return self._policies['local'] == theirs == self._policies['current']

    def _baselines(self, video, record, review):
        """Versions of this video's files that hold nothing made only here."""
        baselines = []
        mine = self.heads.get(self.contributor['id'])
        if mine:
            # What the director published, corrections included: a contributor can only upload a later
            # version of the video after downloading this one, so replacing it loses nothing.
            baselines.append(self.view.snapshot((self.contributor['id'], mine))['manifest']['files'])
        if record.get('files') is not None:
            return baselines + [record['files']]
        if self._base is None:
            try:
                known = _read_local_state(self.project, self.view.project_id)['destinations'].get(
                    self.configuration['root_id'])
                self._base = _base_snapshot(self.view, known)['manifest']['files']
            except ValueError:
                self._base = False  # this copy's starting version is unknown
        if self._base is not False:
            baselines.append(self._base)
        if (review.get('approved') and review.get('project_id') == self.view.project_id and
                isinstance(review.get('local_baseline'), dict)):
            # An approved version counts only if the director accepted it uncorrected; corrections are local work.
            submitted = self.view.snapshot((review['assignment']['contributor']['id'], review['revision']))
            if review['local_baseline'] == _assignment_files(submitted['manifest']['files'], review['assignment']):
                baselines.append(review['local_baseline'])
        return baselines

    def _plan(self, video, entry, record, snapshot, head, cancelled):
        from core.submission_review import _local_annotations, _state, _verify_frames
        incoming = _assignment_files(snapshot['manifest']['files'], entry)
        nothing_new = record.get('files') == incoming and video not in self.replace
        if nothing_new and not record.get('held'):
            # The usual case, kept cheap: nothing from them since last time, so local files are not even read.
            record.update(revision=head)
            return self._keep(video, record)
        status, _ = assignment_status(self.view, entry, heads=self.heads, current=self.current, cancelled=cancelled)
        if status == 'Accepted':
            # The accepted version, with any corrections made here, is the one that counts.
            record.update(revision=head, files=incoming, held=None, skipped=None, kept_local=None)
            return self._keep(video, record)
        if nothing_new:
            record.update(revision=head)  # still held, still not accepted
            return self._keep(video, record)
        local = _local_annotations(self.project, entry)
        review = _state(self.project, video)
        reviewing = bool(review) and not review.get('approved')
        if local == incoming:
            if record.get('files') is None and reviewing:
                # It came in through a review that is not approved yet; what it changed is not known here.
                _hold(record, entry['frames'], SHARED_CATEGORIES)
            record.update(revision=head, files=incoming, skipped=None, kept_local=None)
            return self._keep(video, record)
        if reviewing:
            return self._skip(video, record, head, REVIEW_OPEN)
        if video in self.keep or (record.get('kept_local') == head and video not in self.replace):
            record.update(kept_local=head, skipped=None)
            return self._keep(video, record)
        if not self._same_policy(snapshot):
            raise SyncConflictError('The annotation policy differs between this project and their copy.')
        changed = sorted(name for name in local.keys() | incoming.keys() if local.get(name) != incoming.get(name))
        if video not in self.replace:
            baselines = self._baselines(video, record, review)
            if any(not any(local.get(name) == base.get(name) for base in baselines) for name in changed):
                return self._skip(video, record, head, CHANGED_HERE)
        frames = _verify_frames(self.project, entry, snapshot, missing_ok=True)
        metadata_name = f'frames/{video}/video_metadata.json'
        metadata_path = _safe_child(self.project, metadata_name)
        remote_meta = _verified_json(snapshot, metadata_name)
        local_meta = _read_json(metadata_path) if metadata_path.exists() else None
        if local_meta is not None and local_meta.get('split') != remote_meta.get('split'):
            raise SyncConflictError('This video is in a different split here and in their copy.')
        metadata = dict(local_meta or remote_meta, max_mask_id=max(
            (local_meta or remote_meta).get('max_mask_id', 0), remote_meta.get('max_mask_id', 0)))
        manifest = snapshot['manifest']
        for name in changed:
            self.before[name], self.after[name] = local.get(name), incoming.get(name)
            if incoming.get(name):
                self.sources[name] = snapshot_file(snapshot['manifest_path'], manifest, name)
        for name, present in frames.items():
            if present is None:  # an assigned frame image this project does not have yet
                self.before[name], self.after[name] = None, manifest['files'][name]
                self.sources[name] = snapshot_file(snapshot['manifest_path'], manifest, name)
        if metadata != local_meta:
            payload = json.dumps(metadata, indent=2).encode()
            self.before[metadata_name] = file_record(metadata_path)
            self.after[metadata_name], self.sources[metadata_name] = bytes_record(payload), payload
        record.update(revision=head, files=incoming, taken_at=utc_now(), skipped=None, kept_local=None)
        _hold(record, *real_changes(
            changed,
            lambda name: (self.project / name).read_bytes() if local.get(name) else None,
            lambda name: Path(self.sources[name]).read_bytes() if incoming.get(name) else None))
        self.taken.append(video)
        self.entries[video], self.local[video], self.frames[video] = deepcopy(entry), local, (snapshot, frames)
        self._keep(video, record)

    def summary(self):
        records = self.state['videos']
        return {'taken': [{'video': video, 'contributor': records[video]['contributor']['name']}
                          for video in self.taken],
                'skipped': dict(self.skipped)}

    def apply(self, progress=lambda text: None, cancelled=lambda: False):
        from core.submission_review import _director, _local_annotations, _verify_frames
        if not self.taken:
            if self.state != self.old_state:
                _write_state(self.project, self.state)
            return dict(self.summary(), updated=False)

        def check():
            view = PublishedProject(self.project, self.configuration)
            if (_director(view, self.contributor)['revision'] != self.assignments_revision or
                    view.heads(validate_snapshots=False) != self.heads):
                raise ValueError('Assignments or published work changed during intake. It will be tried again.')
            if read_state(self.project) != self.old_state:
                raise ValueError('The intake record changed during intake. It will be tried again.')
            for video in self.taken:
                snapshot, frames = self.frames[video]
                if (_local_annotations(self.project, self.entries[video]) != self.local[video] or
                        _verify_frames(self.project, self.entries[video], snapshot, missing_ok=True) != frames):
                    raise ValueError('Annotations changed here during intake. It will be tried again.')

        check()
        if cancelled():
            raise InterruptedError('Intake cancelled.')
        result = apply_delta(self.project, self.configuration, self.before, self.after, self.sources,
                             self.view.project_id, {name: record for name, record in self.after.items() if record},
                             False, check, progress, cancelled, scoped=True)
        prune_bundles(self.project, self.configuration)
        return dict(result, **self.summary())


def prune_bundles(project, configuration, keep=KEPT_BUNDLES):
    """Keep the newest rollback bundles that intake made. Each holds an older published version of their work."""
    try:
        mine = [bundle for bundle in rollback_bundles(project, configuration)
                if STATE in bundle['manifest'].get('scope', []) and bundle['manifest'].get('status') == 'complete']
        for bundle in mine[keep:]:
            delete_rollback_bundle(project, bundle['path'], configuration)
    except (OSError, ValueError):
        pass  # housekeeping must never fail an intake that has already been applied


def accept_work(project, configuration, contributor, videos):
    """Accept submitted videos whose assigned annotations here are exactly what was submitted.

    Returns (accepted videos, {video: why it was not accepted}). Anything else, such as a
    video corrected here, goes through Review Submitted Work, which publishes the corrections.
    """
    from core.submission_review import _director, _local_annotations
    project = Path(project).resolve()
    require_full_project(project)
    with _sync_location(project, configuration, contributor):
        view = PublishedProject(project, configuration)
        data = _director(view, contributor)
        current, heads = latest_snapshot(view)
        accepted, refused = [], {}
        for video in videos:
            entry = data['videos'].get(video)
            if not entry:
                refused[video] = 'It is no longer assigned.'
                continue
            status, revision = assignment_status(view, entry, heads=heads, current=current)
            if status != 'Ready for Review':
                refused[video] = 'It is not waiting for review.'
                continue
            submitted = view.snapshot((entry['contributor']['id'], revision))
            files = _assignment_files(submitted['manifest']['files'], entry)
            if _local_annotations(project, entry) != files:
                refused[video] = 'It differs here from what was submitted. Open it in Review Submitted Work.'
                continue
            entry['accepted_revision'] = revision
            accepted.append((entry, revision, files))
        if accepted:
            data['revision'] = str(uuid4())
            _write_json(view.home / ASSIGNMENTS, data)
            for entry, revision, files in accepted:
                mark_accepted(project, entry, revision, files)
        return [entry['video'] for entry, _, _ in accepted], refused
