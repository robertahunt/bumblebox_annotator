"""Read-only freshness checks and comparisons against one displayed shared version."""

from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from core.project_sync import (PROJECT_MARKER, _digest, _inventory, _read_json, _safe_child,
                               require_no_update, require_working_project)
from core.project_update import PublishedProject
from core.publication_review import file_context


def _cancelled(cancelled):
    if cancelled():
        raise InterruptedError('Check cancelled; no files were changed.')


def check_shared_version(project, configuration, snapshot, *, cancelled=lambda: False):
    _cancelled(cancelled)
    view = PublishedProject(project, configuration)
    if view.project_id != snapshot['manifest']['project_id']:
        raise ValueError('The displayed shared version belongs to another project.')
    # Freshness is a pointer check, not a scan of mask payloads or a merge decision.
    heads = view.heads(validate_snapshots=False)
    _cancelled(cancelled)
    displayed = dict(snapshot['parents'])
    displayed[snapshot['reference'][0]] = snapshot['reference'][1]
    return {'changed': heads != displayed, 'checked_at': datetime.now(timezone.utc).isoformat()}


def compare_saved_annotations(project, snapshot, *, progress=lambda message: None, cancelled=lambda: False):
    """Compare saved annotation files, not unsaved edits, images, or original videos.

    Differences are deliberately nondirectional: this is not an upload plan and
    does not establish which copy is newer or whether either edit should win.
    """
    project = Path(project).resolve()
    require_working_project(project)
    require_no_update(project)
    manifest = snapshot['manifest']
    identity = _read_json(_safe_child(project, 'annotations/' + PROJECT_MARKER))
    if identity['project_id'] != manifest['project_id']:
        raise ValueError('The displayed shared version belongs to another project.')
    _cancelled(cancelled)
    progress('Verifying the displayed shared version...')
    if _digest(snapshot['manifest_path']) != snapshot['manifest_hash']:
        raise ValueError('The displayed shared version changed. Refresh Assignments before comparing.')

    def inventory():
        return {name: record for name, record in _inventory(project, False, with_identity=True).items()
                if name.startswith('annotations/')}

    progress('Listing local saved annotations...')
    local = inventory()
    shared = {name: record for name, record in manifest['files'].items()
              if name.startswith('annotations/') and not name.startswith('annotations/coco/')}
    rows, unchanged = [], 0
    names = sorted(local.keys() | shared.keys())
    for position, name in enumerate(names, 1):
        _cancelled(cancelled)
        if position == 1 or position % 25 == 0:
            progress(f'Comparing saved annotation files: {position}/{len(names)}\n{name}')
        before, after = shared.get(name), local.get(name)
        if after is None:
            action = 'Only shared'
        elif before is None:
            action = 'Only local'
        elif before['size'] == after[0] and before['sha256'] == _digest(_safe_child(project, name)):
            unchanged += 1
            continue
        else:
            action = 'Different'
        video, scope = file_context(name)
        rows.append({'path': name, 'video': video, 'scope': scope, 'action': action,
                     'before_bytes': before['size'] if before else None,
                     'after_bytes': after[0] if after else None})
    _cancelled(cancelled)
    if inventory() != local:
        raise ValueError('Local annotation files changed during comparison. Save and compare again.')
    if _digest(snapshot['manifest_path']) != snapshot['manifest_hash']:
        raise ValueError('The displayed shared version changed during comparison. Refresh and compare again.')
    _cancelled(cancelled)
    return {'comparison': 'local_vs_shared', 'local_project': str(project),
            'checked_at': datetime.now(timezone.utc).isoformat(),
            'baseline': {'contributor_name': manifest['contributor']['name'],
                         'created_at': manifest.get('created_at', ''),
                         'revision': snapshot['reference'][1],
                         'manifest_path': str(snapshot['manifest_path'])},
            'rows': rows, 'counts': dict(Counter(row['action'] for row in rows)), 'unchanged': unchanged}
