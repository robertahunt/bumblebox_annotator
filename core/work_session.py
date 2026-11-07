"""Small local upload outbox and last-report records, independent of mask storage.

Presence reports are advisory, self-declared metadata, never locks or proof that
a computer is online. One bounded record per contributor/device/working copy is
replaced atomically; no shared append file or growing per-heartbeat history.
"""

from copy import deepcopy
from datetime import datetime
from pathlib import Path
from uuid import uuid4

from core.contributors import utc_now
from core.project_sync import PROJECT_MARKER, _read_json, _write_json, _safe_child, _uuid
from core.project_update import PublishedProject


class WorkSession:
    def __init__(self, project, actor):
        self.project = Path(project).resolve()
        self.actor = deepcopy(actor)
        identity = _uuid(actor['id'])
        device = _uuid(actor['device']['id'])
        self.path = _safe_child(self.project, f'annotations/.work-session-{identity}-{device}.json')
        self.data = _read_json(self.path) if self.path.exists() else {
            'format': 1, 'copy_id': str(uuid4()), 'pending': False}
        if self.data.get('format') != 1:
            raise ValueError('Unsupported local work-session record; update the app.')
        _uuid(self.data['copy_id'])

    def save(self):
        _write_json(self.path, self.data)

    def start(self):
        if self.data.get('state') == 'closed':
            self.data['previous_session_ended_at'] = self.data.get('session_ended_at')
        self.data.update(session_id=self.actor['session_id'], state='open', session_started_at=utc_now())
        self.save()

    def changed(self, video=None, *, queue=False, completed=False):
        self.data.update(last_edited_at=utc_now(), last_video=video)
        if queue:
            self.data['pending'] = True
        if completed:
            self.data['last_completion_at'] = utc_now()
        self.save()

    def published(self, result):
        self.data.update(pending=False, last_uploaded_at=utc_now(),
                         ready=result.get('ready', []), upload_error='')
        self.save()

    def failed(self, error):
        self.data['upload_error'] = str(error)
        self.save()

    def end(self):
        self.data.update(state='closed', session_ended_at=utc_now())
        self.save()

    def report(self):
        marker = _read_json(_safe_child(self.project, 'annotations/' + PROJECT_MARKER))
        return dict(deepcopy(self.data), project_id=_uuid(marker['project_id']),
                    contributor={k: self.actor[k] for k in ('id', 'name')},
                    device=deepcopy(self.actor['device']), report_created_at=utc_now())


def publish_report(configuration, report, cancelled=lambda: False):
    """Report a successful write now, separately from any earlier local event time."""
    from core.assigned_work import read_assignments, project_members
    if cancelled():
        raise InterruptedError('Activity report cancelled')
    view = PublishedProject.from_archive(configuration, _uuid(report['project_id']))
    if report['contributor']['id'] not in project_members(read_assignments(view)):
        raise ValueError('Only project members can report activity.')
    name = '-'.join(_uuid(v) for v in (report['contributor']['id'], report['device']['id'], report['copy_id']))
    folder = _safe_child(view.home, 'activity', mkdir=True)
    if cancelled():
        raise InterruptedError('Activity report cancelled')
    # Never publish local paths or diagnostic errors, which may contain paths.
    value = {k: v for k, v in report.items() if k != 'upload_error'}
    value['reported_at'] = utc_now()
    _write_json(_safe_child(folder, name + '.json'), value)
    return value


def read_reports(view, contributor=None, cancelled=lambda: False):
    folder = _safe_child(view.home, 'activity')
    if not folder.exists():
        return []
    reports = []
    for path in folder.glob('*.json'):
        if cancelled():
            raise InterruptedError('Activity check cancelled')
        try:
            item = _read_json(_safe_child(folder, path.name))
            if (item.get('format') == 1 and item.get('project_id') == view.project_id
                    and (contributor is None or item['contributor']['id'] == contributor)):
                for field in ('device', 'contributor'):
                    _uuid(item[field]['id'])
                    if not isinstance(item[field]['name'], str):
                        raise ValueError('Invalid activity name')
                if (item.get('state') not in ('open', 'closed') or
                        datetime.fromisoformat(item['reported_at']).tzinfo is None):
                    raise ValueError('Invalid activity timestamp or state')
                reports.append(item)
        except (OSError, ValueError, KeyError, TypeError, AttributeError):
            continue  # A damaged advisory report must not hide verified progress.
    return sorted(reports, key=lambda r: r.get('reported_at', ''), reverse=True)
