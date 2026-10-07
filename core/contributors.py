"""Session attribution without retaining additional mask arrays."""

from collections import OrderedDict
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from uuid import uuid4

import numpy as np


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def new_session(name, contributor_id=None):
    name = name.strip()
    if not name:
        raise ValueError('A contributor name is required.')
    return {'id': contributor_id or str(uuid4()), 'name': name,
            'session_id': str(uuid4()), 'started_at': utc_now()}


def annotation_digest(annotation):
    """Compare scientific content, ignoring derived/display/attribution fields."""
    content = {key: annotation[key] for key in
               ('category', 'marker', 'aruco_id', 'track_id', 'source', 'confidence')
               if key in annotation}
    content.setdefault('category', 'bee')
    mask = annotation.get('mask')
    digest = hashlib.sha256()
    if mask is not None:
        binary = np.asarray(mask) > 0
        content['shape'] = list(binary.shape)
        digest.update(np.packbits(binary).tobytes())
    else:
        content['bbox'] = [float(value) for value in annotation.get('bbox', [])]
    def json_value(value):
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, np.ndarray):
            return value.tolist()
        return str(value)
    digest.update(json.dumps(content, sort_keys=True, default=json_value).encode())
    return digest.hexdigest()


def annotation_id(annotation):
    return int(annotation.get('mask_id', annotation.get('instance_id', 0)))


def provenance_records(annotations):
    return [{'mask_id': annotation_id(ann), 'category': ann.get('category', 'bee'),
             'provenance': deepcopy(ann['provenance'])}
            for ann in annotations if ann.get('provenance')]


def summarize(annotations):
    return {(annotation_id(ann), ann.get('category', 'bee')):
            (annotation_digest(ann), deepcopy(ann.get('provenance', {})))
            for ann in annotations}


class AnnotationAttributor:
    """Small save-time baselines. Commit a baseline only after a successful write."""

    def __init__(self, max_sources=16):
        self.baselines = OrderedDict()
        self.max_sources = max_sources

    def prepare(self, key, annotations, load_previous, session):
        if not session:
            return annotations
        previous = self.baselines.get(key)
        if previous is None:
            previous = summarize(load_previous())
        actor = {'id': session['id'], 'name': session['name']}
        timestamp = utc_now()
        result = []
        for annotation in annotations:
            ann = dict(annotation)
            key = (annotation_id(ann), ann.get('category', 'bee'))
            prior = previous.get(key)
            if prior is None:
                same_id = [value for old_key, value in previous.items() if old_key[0] == key[0]]
                if len(same_id) == 1:
                    prior = same_id[0]
            provenance = deepcopy(prior[1] if prior else ann.get('provenance', {}))
            changed = prior is None or prior[0] != annotation_digest(ann)
            if changed:
                if not provenance:
                    # A legacy instance's original creator cannot be inferred.
                    provenance = {'created_by': None, 'created_at': None}
                    if prior is None:
                        provenance.update(created_by=actor, created_at=timestamp)
                provenance.update(last_modified_by=actor, last_modified_at=timestamp,
                                  session_id=session['session_id'])
                provenance.setdefault('origin', ann.get('source', 'manual_or_unspecified'))
            if provenance:
                ann['provenance'] = provenance
            result.append(ann)
        return result

    def committed(self, key, annotations):
        self.baselines[key] = summarize(annotations)
        self.baselines.move_to_end(key)
        while len(self.baselines) > self.max_sources:
            self.baselines.popitem(last=False)
