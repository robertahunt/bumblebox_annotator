"""Optional per-instance keypoints (pose landmarks such as head, thorax, abdomen tip).

Keypoints are off unless a project turns them on (see ``KeypointConfig`` and
``project.json`` key ``"keypoints"``). When used, each annotated instance may carry

    ann["keypoints"] = {"head": [x, y, 2], "thorax": [x, y, 1], ...}

``x, y`` are full-resolution image pixels (floats). The third value follows the COCO
convention: 2 = visible, 1 = labelled but not visible (occluded), 0 = not labelled.
Unlabelled points are normally omitted. Names not in the current configuration are
kept unchanged, so frames survive a later change of keypoint list.

This module has no Qt dependency; the canvas layer and exporters build on it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

VISIBLE = 2
OCCLUDED = 1
UNLABELLED = 0

PROJECT_KEY = 'keypoints'
INSTANCE_KEY = 'keypoints'
DEFAULT_KEYPOINT_NAMES = ('head', 'thorax', 'abdomen_tip')
DEFAULT_EDGES = (('head', 'thorax'), ('thorax', 'abdomen_tip'))
CONFIG_VERSION = 1


def _clean_name(name):
    text = str(name).strip().lower().replace(' ', '_').replace('-', '_')
    if not text or not all(ch.isalnum() or ch == '_' for ch in text):
        raise ValueError(f'Keypoint names may use letters, digits and underscores only: {name!r}')
    return text


@dataclass(frozen=True)
class KeypointConfig:
    """Project-level keypoint definition."""

    names: tuple = DEFAULT_KEYPOINT_NAMES
    edges: tuple = DEFAULT_EDGES
    category: str = 'bee'
    enabled: bool = True
    # Keypoint filled from an instance's ArUco marker centre on request (or None).
    marker_keypoint: str | None = 'thorax'
    extra: dict = field(default_factory=dict, compare=False)

    def __post_init__(self):
        names = tuple(_clean_name(n) for n in self.names)
        if not names:
            raise ValueError('A keypoint configuration needs at least one keypoint.')
        if len(set(names)) != len(names):
            raise ValueError('Keypoint names must be unique.')
        edges = []
        for edge in self.edges:
            a, b = (_clean_name(n) for n in edge)
            if a not in names or b not in names:
                raise ValueError(f'Skeleton edge {a}-{b} uses a keypoint that is not defined.')
            if a == b:
                raise ValueError(f'Skeleton edge {a}-{b} connects a keypoint to itself.')
            if (a, b) not in edges and (b, a) not in edges:
                edges.append((a, b))
        marker = self.marker_keypoint
        if marker is not None:
            marker = _clean_name(marker)
            if marker not in names:
                marker = None
        object.__setattr__(self, 'names', names)
        object.__setattr__(self, 'edges', tuple(edges))
        object.__setattr__(self, 'marker_keypoint', marker)

    def to_dict(self):
        data = dict(self.extra)
        data.update(
            version=CONFIG_VERSION,
            enabled=bool(self.enabled),
            category=self.category,
            names=list(self.names),
            edges=[list(e) for e in self.edges],
            marker_keypoint=self.marker_keypoint,
        )
        return data

    @classmethod
    def from_dict(cls, data):
        if not isinstance(data, dict):
            raise ValueError('Keypoint settings must be a JSON object.')
        known = {'version', 'enabled', 'category', 'names', 'edges', 'marker_keypoint'}
        return cls(
            names=tuple(data.get('names') or DEFAULT_KEYPOINT_NAMES),
            edges=tuple(tuple(e) for e in (data.get('edges') or ())),
            category=str(data.get('category') or 'bee'),
            enabled=bool(data.get('enabled', True)),
            marker_keypoint=data.get('marker_keypoint', 'thorax'),
            extra={k: v for k, v in data.items() if k not in known},
        )

    def coco_category_fields(self):
        """COCO ``keypoints`` and 1-based ``skeleton`` for a category entry."""
        index = {name: i + 1 for i, name in enumerate(self.names)}
        return {
            'keypoints': list(self.names),
            'skeleton': [[index[a], index[b]] for a, b in self.edges],
        }


def default_config():
    return KeypointConfig()


def config_from_project_info(project_info):
    """Return the enabled ``KeypointConfig`` stored in project info, else None.

    Malformed settings are treated as "off" rather than raising, so a bad edit
    can never stop a project from opening.
    """
    if not isinstance(project_info, dict):
        return None
    data = project_info.get(PROJECT_KEY)
    if not data:
        return None
    try:
        config = KeypointConfig.from_dict(data)
    except (TypeError, ValueError):
        return None
    return config if config.enabled else None


def stored_project_config(project_info):
    """The stored configuration even when turned off (for the settings dialog), or None."""
    if not isinstance(project_info, dict):
        return None
    data = project_info.get('project_info', project_info).get(PROJECT_KEY)
    if not data:
        return None
    try:
        return KeypointConfig.from_dict(data)
    except (TypeError, ValueError):
        return None


def read_project_config(project_path):
    """Enabled keypoint configuration of the project at ``project_path``, else None."""
    from core.annotation_scope import read_project_info
    info = read_project_info(project_path)
    return config_from_project_info(info.get('project_info', info))


def save_project_config(project_path, config, enabled=True):
    """Write the project's keypoint settings to ``annotations/project.json``.

    Like the annotation policy, this is a project-setup change: refused in published
    backups/sync folders and in contributors' assigned workspaces. Turning keypoints
    off keeps the stored names and leaves keypoints already saved in frames untouched.
    """
    from copy import deepcopy
    from pathlib import Path

    from core.annotation_scope import read_project_info
    from core.assigned_work import require_full_project
    from core.project_sync import _write_json, require_working_project

    require_working_project(project_path)
    require_full_project(project_path)
    info = read_project_info(project_path)
    updated = deepcopy(info)
    body = updated.get('project_info', updated)
    data = config.to_dict()
    data['enabled'] = bool(enabled)
    body[PROJECT_KEY] = data
    _write_json(Path(project_path) / 'annotations/project.json', updated)
    return updated


def parse_names(text):
    """Keypoint names from text: one per line or comma-separated, in order."""
    parts = [p for chunk in str(text).replace(',', '\n').splitlines() for p in [chunk.strip()] if p]
    return tuple(_clean_name(p) for p in parts)


def parse_edges(text):
    """Skeleton edges from text like ``head-thorax, thorax-abdomen_tip``."""
    edges = []
    for chunk in str(text).replace(';', ',').replace('\n', ',').split(','):
        chunk = chunk.strip()
        if not chunk:
            continue
        for sep in ('->', '--', '-', ':'):
            if sep in chunk:
                a, b = chunk.split(sep, 1)
                edges.append((_clean_name(a), _clean_name(b)))
                break
        else:
            raise ValueError(f'Write each edge as name-name: {chunk!r}')
    return tuple(edges)


def format_edges(edges):
    return ', '.join(f'{a}-{b}' for a, b in edges)


def normalize_point(value):
    """``[x, y, v]`` with float x, y and int v in {1, 2}; None if missing/invalid/unlabelled."""
    if value is None:
        return None
    try:
        if isinstance(value, dict):
            x, y = value['x'], value['y']
            v = value.get('v', value.get('visibility', VISIBLE))
        else:
            x, y = value[0], value[1]
            v = value[2] if len(value) > 2 else VISIBLE
        x, y, v = float(x), float(y), int(v)
    except (KeyError, IndexError, TypeError, ValueError):
        return None
    if not (math.isfinite(x) and math.isfinite(y)) or v not in (OCCLUDED, VISIBLE):
        return None
    return [x, y, v]


def get_keypoints(metadata):
    """Clean copy of an instance's labelled keypoints (``{}`` when none)."""
    raw = (metadata or {}).get(INSTANCE_KEY)
    if not isinstance(raw, dict):
        return {}
    out = {}
    for name, value in raw.items():
        point = normalize_point(value)
        if point is not None:
            out[str(name)] = point
    return out


def set_keypoints(metadata, keypoints):
    """Store keypoints on instance metadata; removes the key when empty."""
    clean = {name: normalize_point(p) for name, p in (keypoints or {}).items()}
    clean = {name: p for name, p in clean.items() if p is not None}
    if clean:
        metadata[INSTANCE_KEY] = clean
    else:
        metadata.pop(INSTANCE_KEY, None)
    return clean


def with_point(keypoints, name, x, y, visibility=VISIBLE):
    out = dict(keypoints or {})
    out[name] = [float(x), float(y), int(visibility)]
    return out


def without_point(keypoints, name):
    out = dict(keypoints or {})
    out.pop(name, None)
    return out


def with_toggled_visibility(keypoints, name):
    out = dict(keypoints or {})
    if name in out:
        x, y, v = out[name]
        out[name] = [x, y, OCCLUDED if v == VISIBLE else VISIBLE]
    return out


def next_unlabelled(keypoints, names, after=None):
    """First name in ``names`` without a point, searching after ``after`` and wrapping."""
    names = list(names)
    if not names:
        return None
    start = names.index(after) + 1 if after in names else 0
    for offset in range(len(names)):
        name = names[(start + offset) % len(names)]
        if name not in (keypoints or {}):
            return name
    return None


def nearest_keypoint(instances, x, y, max_distance):
    """Closest labelled point to (x, y) within ``max_distance`` image pixels.

    ``instances`` maps instance id -> keypoints dict. Returns
    ``(instance_id, name, distance)`` or None.
    """
    best = None
    for instance_id, keypoints in instances.items():
        for name, point in (keypoints or {}).items():
            d = math.hypot(point[0] - x, point[1] - y)
            if d <= max_distance and (best is None or d < best[2]):
                best = (instance_id, name, d)
    return best


def merge_keypoints(target, source):
    """Keypoints for an instance merged into another: target's points win."""
    out = dict(source or {})
    out.update(target or {})
    return out


def marker_center(metadata):
    """(x, y) of an instance's ArUco marker centre, or None."""
    marker = (metadata or {}).get('marker')
    if not isinstance(marker, dict):
        return None
    center = marker.get('center')
    try:
        x, y = float(center[0]), float(center[1])
    except (TypeError, IndexError, ValueError):
        return None
    if not (math.isfinite(x) and math.isfinite(y)):
        return None
    return x, y


def coco_keypoints(keypoints, names):
    """COCO ``keypoints`` flat list and ``num_keypoints`` for one instance."""
    flat, count = [], 0
    for name in names:
        point = normalize_point((keypoints or {}).get(name))
        if point is None:
            flat.extend([0, 0, UNLABELLED])
        else:
            flat.extend([round(point[0], 2), round(point[1], 2), point[2]])
            count += 1
    return flat, count


def coco_add_category_fields(categories, config):
    """Add ``keypoints``/``skeleton`` to the configured category entry (in place)."""
    if config is None:
        return categories
    for category in categories:
        if category.get('name') == config.category:
            category.update(config.coco_category_fields())
    return categories


def coco_add_instance_keypoints(coco_ann, ann, config):
    """Add COCO ``keypoints``/``num_keypoints`` for instances of the configured category.

    Every such instance gets the fields (``num_keypoints`` 0 when unlabelled), following
    COCO; other categories and projects without keypoints are left unchanged.
    """
    if config is None or ann.get('category', 'bee') != config.category:
        return coco_ann
    flat, count = coco_keypoints(get_keypoints(ann), config.names)
    coco_ann['keypoints'] = flat
    coco_ann['num_keypoints'] = count
    return coco_ann


def summary_text(keypoints, names):
    """Short status like ``head ✓  thorax (hidden)  abdomen_tip –``."""
    parts = []
    for name in names:
        point = (keypoints or {}).get(name)
        if point is None:
            parts.append(f'{name} –')
        elif point[2] == OCCLUDED:
            parts.append(f'{name} (hidden)')
        else:
            parts.append(f'{name} ✓')
    return '  '.join(parts)
