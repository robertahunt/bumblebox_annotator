"""Experimental, visible-evidence brood maps, independent of hive occupancy."""

from dataclasses import dataclass
import math

import numpy as np

from core.categories import BROOD_CATEGORIES, BROOD_MAP_LABELS, BROOD_UNRESOLVED_LABEL
from core.temporal_hive_prior import TemporalHivePrior


@dataclass
class _BroodState:
    counts: np.ndarray
    weight: np.ndarray
    last_seen: np.ndarray
    time: float


class TemporalBroodMap:
    """Bounded recent evidence per chamber pixel, not a biological age model.

    Input labels: 0=background, 1..N=appearance classes, 255=unobserved.
    Output labels: 0=unknown, 1=background, class values from BROOD_MAP_LABELS,
    7=brood/stage unresolved (preserved for compatibility with older maps).
    Uncertain appearance classes are ordinary classes, not missing observations.
    """

    def __init__(self, resolution=(256, 256), window_seconds=8 * 3600,
                 max_weight=8.0, min_weight=1.5, stage_threshold=0.6):
        if (len(resolution) != 2 or any(int(n) != n or n < 1 for n in resolution)
                or not math.isfinite(window_seconds) or window_seconds <= 0
                or not 0 < min_weight <= max_weight or max_weight <= 1
                or not 0.5 < stage_threshold <= 1):
            raise ValueError('Invalid temporal brood map settings')
        self.resolution = tuple(int(n) for n in resolution)
        self.window_seconds = float(window_seconds)
        self.max_weight = float(max_weight)
        self.min_weight = float(min_weight)
        self.stage_threshold = float(stage_threshold)
        self.geometry = TemporalHivePrior(resolution=self.resolution)
        self.states = {}

    def update(self, context, chamber_id, chamber, labels, bees, frame_shape, time_seconds):
        if time_seconds is None or not math.isfinite(time_seconds):
            raise ValueError('Temporal brood maps require a valid frame time')
        if labels is not None:
            if labels.shape != tuple(frame_shape) or not np.isin(labels, [*range(len(BROOD_CATEGORIES) + 1), 255]).all():
                raise ValueError(f'Brood evidence must be a full-frame label image (0..{len(BROOD_CATEGORIES)} or 255)')
        key = (str(context), int(chamber_id))
        shape = self.resolution[::-1]
        state = self.states.get(key)
        if state is None:
            state = _BroodState(np.zeros((len(BROOD_CATEGORIES) + 1, *shape), np.float32),
                                np.zeros(shape, np.float32),
                                np.full(shape, -np.inf), float(time_seconds))
            self.states[key] = state
        elapsed = float(time_seconds) - state.time
        if elapsed < -1e-6:
            raise ValueError('Brood history received out-of-order frames; use chronological videos')
        decay = math.exp(-max(0, elapsed) / self.window_seconds)
        state.counts *= decay
        state.weight *= decay
        state.time = float(time_seconds)
        if labels is None or chamber.get('temporal_unavailable', False):
            return self.snapshot(context, chamber_id, chamber, frame_shape)

        normalize = lambda mask: self.geometry._normalize_mask(mask, chamber, frame_shape)
        valid = self.geometry._normalize_chamber_mask(chamber, frame_shape)
        valid &= ~self.geometry._normalize_bee_occlusion(bees, chamber, frame_shape)
        valid &= normalize(labels != 255)
        # Cap support so thousands of earlier frames cannot freeze a stage forever.
        factor = np.ones(shape, np.float32)
        np.divide(self.max_weight - 1, state.weight, out=factor, where=state.weight > 0)
        factor = np.where(valid, np.minimum(factor, 1), 1)
        state.counts *= factor
        state.weight *= factor
        for index in range(len(BROOD_CATEGORIES) + 1):
            state.counts[index] += normalize(labels == index) & valid
        state.weight[valid] += 1
        state.last_seen[valid] = float(time_seconds)
        return self.snapshot(context, chamber_id, chamber, frame_shape)

    def snapshot(self, context, chamber_id, chamber, frame_shape):
        state = self.states[(str(context), int(chamber_id))]
        probability = np.zeros_like(state.counts)
        np.divide(state.counts, state.weight[None], out=probability,
                  where=state.weight[None] > 0)
        known = ((state.weight >= self.min_weight)
                 & (state.time - state.last_seen <= self.window_seconds)
                 & self.geometry._normalize_chamber_mask(chamber, frame_shape))
        if chamber.get('temporal_unavailable', False):
            known[:] = False
        presence = 1 - probability[0]
        stage_confidence = np.zeros_like(state.weight)
        np.divide(probability[1:].max(axis=0), presence, out=stage_confidence, where=presence > 0)
        labels = np.zeros(state.weight.shape, np.uint8)
        labels[known & (presence < 0.5)] = 1
        brood = known & (presence >= 0.5)
        labels[brood] = BROOD_UNRESOLVED_LABEL
        resolved = brood & (stage_confidence >= self.stage_threshold)
        class_labels = np.array([BROOD_MAP_LABELS[cat] for cat in BROOD_CATEGORIES], np.uint8)
        stage = class_labels[probability[1:].argmax(axis=0)]
        labels[resolved] = stage[resolved]
        return dict(labels=labels, probability=probability, weight=state.weight.copy(),
                    last_seen=state.last_seen.copy(), chamber_id=int(chamber_id),
                    bbox=self.geometry._chamber_bbox(chamber, frame_shape), time_seconds=state.time)

    def bee_overlap(self, snapshot, bee, chamber, frame_shape):
        mask = self.geometry._normalize_detection(
            getattr(bee, 'mask', None), getattr(bee, 'bbox', None), chamber, frame_shape)
        labels = snapshot['labels']
        known = mask & (labels != 0)
        n_bee, n_known = int(mask.sum()), int(known.sum())
        values = {'known_fraction': n_known / n_bee if n_bee else None}
        values['brood_fraction'] = float((known & (labels >= 2)).sum() / n_known) if n_known else None
        for category, index in BROOD_MAP_LABELS.items():
            values[category + '_fraction'] = float((known & (labels == index)).sum() / n_known) if n_known else None
        values['unresolved_fraction'] = float((known & (labels == BROOD_UNRESOLVED_LABEL)).sum() / n_known) if n_known else None
        return values
