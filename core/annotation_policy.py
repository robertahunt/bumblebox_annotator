"""Explicit label intent, without changing the existing category storage rules."""

from copy import deepcopy
import hashlib
import json
from pathlib import Path

from core.annotation_scope import frame_categories, read_project_info
from core.categories import CATEGORIES, BROOD_CATEGORIES, category_label
from core.contributors import utc_now
from core.project_sync import _write_json


MEANINGS = {'visible': 'Visible surfaces', 'full_extent': 'Full object extent',
            'unconfirmed': 'Not yet confirmed'}
SCOPES = {'frame': 'This frame', 'video': 'Shared across video'}


def category_policies(info):
    info = info.get('project_info', info)
    policy = info.get('annotation_policy', {})
    if policy and policy.get('version') != 1:
        raise ValueError('Unsupported annotation policy version. Update the app before editing this project.')
    local = frame_categories(info)
    result = {}
    for category in CATEGORIES:
        scope = 'frame' if category in local else 'video'
        entry = policy.get('categories', {}).get(category, {})
        if entry.get('scope', scope) != scope:
            raise ValueError(f'{category_label(category)} policy conflicts with stored annotation scope. '
                             'Use project import for scope conversion, not a metadata change.')
        meaning = entry.get('meaning', 'unconfirmed')
        if meaning not in MEANINGS:
            raise ValueError(f'Unknown label meaning for {category}: {meaning}')
        result[category] = {'scope': scope, 'meaning': meaning}
    return result


def default_policy(info):
    entries = category_policies(info)
    for entry in entries.values():
        entry['meaning'] = 'visible' if entry['scope'] == 'frame' else 'full_extent'
    return {'version': 1, 'categories': entries}


def policy_text(entry):
    return f"{SCOPES[entry['scope']]} / {MEANINGS[entry['meaning']]}"


def guidance(entry):
    if entry['meaning'] == 'unconfirmed':
        return 'Existing label meaning is unknown. Confirm the project policy before completing import reviews.'
    text = ('Label only visible object pixels; exclude areas covered by bees or other objects.'
            if entry['meaning'] == 'visible' else
            'Use clear views to establish the full object extent. Do not guess unknown hidden boundaries.')
    if entry['scope'] == 'video':
        text += ' One mask applies to every frame; verify position, shape, and visibility where relevant.'
    return text


def temporal_support(category):
    if category == 'hive':
        return 'Optional batch history; configured separately'
    if category in BROOD_CATEGORIES:
        return 'Experimental batch history; configured separately'
    if category == 'nectar':
        return 'Planned: batch inference and history (unavailable)'
    if category == 'pollen':
        return 'No temporal reconstruction; summaries only'
    return 'Alignment input, not reconstructed' if category == 'chamber' else 'Identity tracking, not a surface map'


def policy_signature(info):
    return hashlib.sha256(json.dumps(category_policies(info), sort_keys=True).encode()).hexdigest()


def save_policy(project, entries, contributor=None):
    """Confirm intent only. Scope conversion is deliberately not a settings operation."""
    info = read_project_info(project)
    updated = deepcopy(info)
    body = updated.get('project_info', updated)
    previous = category_policies(info)
    body['annotation_policy'] = {'version': 1, 'categories': deepcopy(entries)}
    current = category_policies(updated)
    if set(entries) != set(CATEGORIES) or any(current[c]['scope'] != previous[c]['scope'] for c in CATEGORIES):
        raise ValueError('Policy settings cannot change annotation storage scope.')
    history = info.get('project_info', info).get('annotation_policy', {}).get('history', [])
    body['annotation_policy']['history'] = history + [{
        'confirmed_at': utc_now(),
        'confirmed_by': ({'id': contributor['id'], 'name': contributor['name']} if contributor else None),
        'previous': previous, 'categories': current,
    }]
    _write_json(Path(project) / 'annotations/project.json', updated)
    return updated
