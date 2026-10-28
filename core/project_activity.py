"""Read-only collaboration status for the workspace UI.

Status is advisory. Publishing and updating still run their own locked checks.
Only annotation/configuration files are compared locally, not cached source media.
"""

from pathlib import Path

from core.assigned_work import (workspace, read_assignments, latest_snapshot,
                                _check_owned, _global_files, _video_files)
from core.project_delta import file_record
from core.project_sync import PROJECT_MARKER, SyncConflictError, _read_local_state, _inventory
from core.project_update import PublishedProject, _base_snapshot


def annotation_files(files):
    return {name: value for name, value in files.items()
            if name.startswith('annotations/') or name == 'tracking_sequences.json'
            or (name.startswith('frames/') and name.endswith('/video_metadata.json'))}


def project_activity(project, configuration, cancelled=lambda: False):
    project = Path(project)
    state = workspace(project)
    if not configuration.get('root_id'):
        if state:
            return {'status': 'attention', 'message': 'Choose the shared folder used for this assignment',
                    'changed': None}
        return {'status': 'local', 'message': 'Local project', 'changed': None}
    if state and state['root_id'] != configuration['root_id']:
        return {'status': 'attention', 'message': 'Choose the shared folder used for this assignment', 'changed': None}
    if not (project / 'annotations' / PROJECT_MARKER).exists():
        return {'status': 'unpublished', 'message': 'No project backup published yet', 'changed': True}
    view = PublishedProject(project, configuration)
    assignments = read_assignments(view)
    director = assignments.get('coordinator')
    current, heads = latest_snapshot(view)
    if cancelled():
        raise InterruptedError('Status check cancelled')
    if state:
        try:
            _check_owned(assignments, state['assignments'], state['contributor'])
        except (ValueError, SyncConflictError) as exc:
            return {'status': 'attention', 'message': str(exc), 'changed': None, 'director': director}
        if state.get('pending_publication'):
            return {'status': 'attention', 'message': 'Retry Upload Progress to recover the previous upload receipt',
                    'changed': None, 'director': director}
        base = view.snapshot(tuple(state['base_reference']))
        before, incoming = base['manifest']['files'], current['manifest']['files']
        relevant_change = (_global_files(before) != _global_files(incoming) or any(
            _video_files(before, video) != _video_files(incoming, video) for video in state['assignments']))
        status = 'attention' if relevant_change else 'current'
        message = ('Shared annotations or policy changed; keep local work for director review'
                   if relevant_change else 'No shared changes affecting your assignments')
        baseline = state['local_files']
    else:
        known = _read_local_state(project, view.project_id)['destinations'].get(configuration['root_id'])
        if not known or not known.get('heads'):
            return {'status': 'attention', 'message': 'No recorded starting version; review shared history',
                    'changed': None, 'director': director}
        base = _base_snapshot(view, known)
        if heads == known['heads']:
            status, message = 'current', 'No newer shared version'
        elif view.descends_from(current['reference'], base['reference']):
            status, message = 'update', 'A newer shared version is available'
        else:
            status, message = 'attention', 'Shared history differs; review before updating'
        baseline = base['manifest']['files']
    baseline = annotation_files(baseline)
    local = annotation_files(_inventory(project, False))
    if state:
        for video in state.get('retained_assignments', {}):
            for name in _video_files({**baseline, **local}, video):
                baseline.pop(name, None)
                local.pop(name, None)
    changed = local.keys() != baseline.keys()
    if not changed:
        for name in local:
            if cancelled():
                raise InterruptedError('Status check cancelled')
            if file_record(project / name) != baseline[name]:
                changed = True
                break
    if changed and status == 'update':
        message = 'Local annotation edits and a newer shared version; review before updating'
    return {'status': status, 'message': message, 'changed': changed, 'director': director}
