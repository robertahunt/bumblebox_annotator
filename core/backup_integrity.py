"""Version-scoped, non-repairing backup checks and shared integrity reports."""

from datetime import datetime, timezone
import hashlib
import stat
import time
from uuid import uuid4

from core.project_sync import (_digest, _read_json, _safe_child, _uuid, _write_json,
                               check_destination, project_storage_lock, snapshot_file)
from core.project_update import PublishedProject, _identity_folder


class StoredFileProblem(ValueError):
    def __init__(self, path, reason, actual=None):
        super().__init__(f'Stored backup file is missing, edited or damaged ({reason}): {path}')
        self.reason, self.actual = reason, actual


def check_stored_file(path, expected):
    """Cheap structural check. Same-size corruption requires content verification."""
    try:
        info = path.lstat()
    except FileNotFoundError:
        raise StoredFileProblem(path, 'missing') from None
    if not stat.S_ISREG(info.st_mode):
        raise StoredFileProblem(path, 'not an ordinary file')
    if info.st_size != expected['size']:
        raise StoredFileProblem(path, 'size mismatch', info.st_size)
    return info


def _key(reference, name, record, compact):
    if compact:
        return 'sha256:' + record['sha256']
    return 'file:' + '/'.join(reference) + '/' + name + ':' + record['sha256']


def integrity_state(home, configuration, project_id):
    path = _safe_child(home, 'verification/state.json')
    if not path.exists():
        return {'format': 1, 'root_id': configuration['root_id'], 'project_id': project_id,
                'versions': {}, 'problems': {}}
    data = _read_json(path)
    if (not isinstance(data, dict) or data.get('format') != 1 or data.get('root_id') != configuration['root_id'] or
            data.get('project_id') != project_id or not isinstance(data.get('versions'), dict) or
            not isinstance(data.get('problems'), dict)):
        raise ValueError('Shared integrity records are invalid. No integrity status can be assumed.')
    for entry in data['versions'].values():
        if not isinstance(entry, dict):
            raise ValueError('Shared integrity version record is invalid.')
        for kind, summary in entry.items():
            if (kind not in ('last_check', 'last_pass') or not isinstance(summary, dict) or
                    summary.get('status') not in ('passed', 'problems', 'cancelled', 'unable') or
                    (kind == 'last_pass' and summary['status'] != 'passed') or
                    not all(isinstance(summary.get(k), str) for k in
                            ('finished_at', 'manifest_hash', 'report_id'))):
                raise ValueError('Shared integrity check summary is invalid.')
    return data


def require_healthy_references(home, configuration, project_id, files, manifest_path=None):
    """A routine publish must not silently reuse content already known to be bad."""
    problems = integrity_state(home, configuration, project_id)['problems']
    if not problems:
        return
    legacy_reference = None
    if manifest_path is not None:
        contributor_home = manifest_path.parents[2]
        marker = _safe_child(contributor_home, '.bumblebox-contributor.json')
        identity = _read_json(marker)['id'] if marker.exists() else contributor_home.name
        legacy_reference = (_uuid(identity), manifest_path.parent.name)
    for name, record in files.items():
        keys = ['sha256:' + record['sha256']]
        if legacy_reference is not None:
            keys.append(_key(legacy_reference, name, record, False))
        if any(key in problems for key in keys):
            raise ValueError('Known backup damage affects ' + name +
                             '. Resolve it and rerun Check Backup Integrity before publishing. '
                             'No stored content was repaired or replaced.')


def integrity_versions(project, configuration, cancelled=lambda: False):
    """List all completed versions, including historical and separate backups."""
    view = PublishedProject(project, configuration)
    versions = []
    for home in view.contributors.iterdir():
        if cancelled():
            raise InterruptedError('Reading backup versions cancelled.')
        if home.is_symlink():
            raise ValueError('Contributor folders must not be symbolic links.')
        if not home.is_dir():
            continue
        marker = _safe_child(home, '.bumblebox-contributor.json')
        contributor = _read_json(marker)['id'] if marker.exists() else home.name
        revisions = _safe_child(home, 'revisions')
        if not revisions.is_dir():
            continue
        for revision in revisions.iterdir():
            if cancelled():
                raise InterruptedError('Reading backup versions cancelled.')
            if revision.name.startswith('.'):
                continue
            if revision.is_symlink():
                raise ValueError('Backup versions must not be symbolic links.')
            if revision.is_dir():
                versions.append(view.snapshot((contributor, revision.name), metadata_only=True))
    return {'versions': sorted(versions, key=lambda s: s['manifest'].get('created_at', ''), reverse=True),
            'state': integrity_state(view.home, configuration, view.project_id)}


def read_integrity_report(project, configuration, reference, report_id):
    view = PublishedProject(project, configuration)
    path = _safe_child(view.home, 'verification/reports/' + _uuid(report_id) + '.json')
    report = _read_json(path)
    if (report.get('format') != 1 or report.get('root_id') != configuration['root_id'] or
            report.get('project_id') != view.project_id or report.get('reference') != list(reference) or
            report.get('report_id') != report_id):
        raise ValueError('The integrity report does not belong to the selected project and version.')
    return dict(report, report_path=str(path))


def _signature(info):
    return info.st_size, info.st_mtime_ns, info.st_ctime_ns, info.st_ino, info.st_dev


def _hash_file(path, cancelled, progress):
    digest, count, last = hashlib.sha256(), 0, 0.0
    with path.open('rb') as handle:
        while True:
            if cancelled():
                raise InterruptedError('Integrity check cancelled.')
            block = handle.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
            count += len(block)
            if time.monotonic() - last >= 0.2:
                progress(count)
                last = time.monotonic()
    progress(count)
    return digest.hexdigest()


def _save_report(view, report, checked, failures):
    check_destination(view.root, view.configuration['root_id'])
    # A stale pass is not authority to clear errors; only files read in this check count.
    state = integrity_state(view.home, view.configuration, view.project_id)
    for key in checked:
        state['problems'].pop(key, None)
    state['problems'].update(failures)
    version_key = '/'.join(report['reference'])
    entry = state['versions'].setdefault(version_key, {})
    summary = {key: report[key] for key in ('status', 'finished_at', 'manifest_hash', 'report_id')}
    entry['last_check'] = summary
    if report['status'] == 'passed':
        entry['last_pass'] = summary
    folder = _safe_child(view.home, 'verification', mkdir=True)
    reports = _safe_child(folder, 'reports', mkdir=True)
    path = reports / (report['report_id'] + '.json')
    _write_json(path, report)
    _write_json(folder / 'state.json', state)
    report['report_path'] = str(path)


def check_backup_integrity(project, configuration, reference, expected_manifest_hash,
                           progress=None, cancelled=None):
    """Hash one pinned version under the publication lock; never repair any files."""
    progress = progress or (lambda message: None)
    cancelled = cancelled or (lambda: False)
    initial = PublishedProject(project, configuration)
    with project_storage_lock(configuration, initial.project_id, 'Backup integrity check'):
        view = PublishedProject(project, configuration)
        snapshot = view.snapshot(reference)
        if snapshot['manifest_hash'] != expected_manifest_hash:
            raise ValueError('The selected manifest changed. Refresh backup versions before checking.')
        manifest, manifest_path = snapshot['manifest'], snapshot['manifest_path']
        groups = {}
        for name, record in manifest['files'].items():
            key = _key(snapshot['reference'], name, record, manifest.get('format') == 3)
            if key in groups and groups[key]['record'] != record:
                raise ValueError('Inconsistent sizes for identical stored content in this manifest.')
            group = groups.setdefault(key, {'record': record, 'names': []})
            group['names'].append(name)
        now = lambda: datetime.now(timezone.utc).isoformat()
        report = {'format': 1, 'report_id': str(uuid4()), 'root_id': configuration['root_id'],
                  'project_id': view.project_id, 'reference': list(snapshot['reference']),
                  'manifest_hash': snapshot['manifest_hash'], 'contributor': manifest['contributor'],
                  'started_at': now(), 'status': 'unable', 'files': len(manifest['files']),
                  'unique_contents': len(groups), 'checked_contents': 0,
                  'total_bytes': sum(g['record']['size'] for g in groups.values()),
                  'bytes_read': 0, 'issues': []}
        checked, failures = set(), {}
        try:
            for index, (key, group) in enumerate(groups.items(), 1):
                if cancelled():
                    raise InterruptedError('Integrity check cancelled.')
                check_destination(view.root, configuration['root_id'])
                name, expected = group['names'][0], group['record']
                path = snapshot_file(manifest_path, manifest, name)
                before_bytes = report['bytes_read']
                def update(count):
                    report['bytes_read'] = before_bytes + count
                    progress(f"Checking backup content {index}/{len(groups)} | "
                             f"{report['bytes_read'] / 1024**2:.1f}/{report['total_bytes'] / 1024**2:.1f} MiB\n{name}")
                update(0)
                try:
                    before = check_stored_file(path, expected)
                    actual = _hash_file(path, cancelled, update)
                    after = path.stat()
                    if _signature(before) != _signature(after):
                        raise OSError('Stored content changed during verification: ' + name)
                    if actual != expected['sha256']:
                        raise StoredFileProblem(path, 'checksum mismatch', actual)
                    checked.add(key)
                except StoredFileProblem as exc:
                    # A lost mount is an incomplete operation, not thousands of missing files.
                    check_destination(view.root, configuration['root_id'])
                    if _identity_folder(view.root, view.project_id, 'project') != view.home:
                        raise OSError('The shared project moved during verification.')
                    issue = {'key': key, 'paths': group['names'], 'reason': exc.reason,
                             'expected': expected, 'actual': exc.actual}
                    report['issues'].append(issue)
                    failures[key] = dict(issue, detected_at=now())
                report['checked_contents'] += 1
            if cancelled():
                raise InterruptedError('Integrity check cancelled.')
            check_destination(view.root, configuration['root_id'])
            if _digest(manifest_path) != snapshot['manifest_hash']:
                raise OSError('The manifest changed during verification. Results cannot certify this version.')
            report['status'] = 'problems' if failures else 'passed'
        except InterruptedError as exc:
            report.update(status='cancelled', error=str(exc))
        except (OSError, ValueError) as exc:
            report.update(status='unable', error=str(exc))
        report['finished_at'] = now()
        # Incomplete checks may record new failures, but never clear an earlier warning.
        try:
            _save_report(view, report, checked if report['status'] in ('passed', 'problems') else set(), failures)
        except (OSError, ValueError) as exc:
            report['report_warning'] = 'Could not save the shared integrity report: ' + str(exc)
        return report
