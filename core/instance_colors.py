"""Deterministic, spatially contrasting display colors; no annotation mutations."""

import colorsys
import math

import cv2
import numpy as np


BEE_PALETTE = (
    (0, 170, 255), (255, 80, 90), (80, 225, 95), (210, 95, 245),
    (255, 205, 40), (0, 220, 200), (255, 135, 35), (110, 125, 255),
    (245, 120, 195), (180, 235, 45), (35, 165, 125), (255, 0, 187),
)
# Frames are often greyscale, so bee colours must never be grey, white or near-black.
MIN_SATURATION = 0.45
MIN_VALUE = 0.55
_LAB = cv2.cvtColor(
    np.asarray([BEE_PALETTE], dtype=np.float32) / 255, cv2.COLOR_RGB2Lab
)[0]
_DISTANCES = np.linalg.norm(_LAB[:, None] - _LAB[None, :], axis=2)


def random_vivid_color(seed=None):
    """A saturated, bright RGB colour. Uses its own generator, never the global one."""
    rng = np.random.default_rng(None if seed is None else abs(int(seed)))
    hue = rng.random()
    saturation = rng.uniform(0.7, 1.0)
    value = rng.uniform(0.85, 1.0)
    return tuple(int(round(c * 255)) for c in colorsys.hsv_to_rgb(hue, saturation, value))


def vivid_color(color):
    """Return ``color`` unchanged if it stands out on a grey frame, else a vivid version.

    A weakly tinted colour keeps its hue. A true grey gets a hue derived from its value,
    so the same stored colour always maps to the same display colour.
    """
    r, g, b = (min(255, max(0, int(c))) for c in color[:3])
    hue, saturation, value = colorsys.rgb_to_hsv(r / 255, g / 255, b / 255)
    if saturation >= MIN_SATURATION and value >= MIN_VALUE:
        return (r, g, b)
    if saturation < 0.15:
        hue = ((r * 7919 + g * 104729 + b * 1299709) % 360) / 360
    return tuple(int(round(c * 255)) for c in colorsys.hsv_to_rgb(hue, 0.85, 0.95))


def neighboring_boxes(boxes):
    """Connect touching/overlapping boxes and boxes within half a typical bee size.

    Bounding boxes conservatively include touching long bodies even when their
    centroids are far apart, without retaining or dilating full-resolution masks.
    """
    neighbors = {instance_id: set() for instance_id in boxes}
    ids = sorted(boxes)
    for index, first in enumerate(ids):
        x, y, w, h = boxes[first]
        for second in ids[index + 1:]:
            x2, y2, w2, h2 = boxes[second]
            dx = max(x - (x2 + w2), x2 - (x + w), 0)
            dy = max(y - (y2 + h2), y2 - (y + h), 0)
            padding = 0.25 * (math.sqrt(w * h) + math.sqrt(w2 * h2))
            if math.hypot(dx, dy) <= padding:
                neighbors[first].add(second)
                neighbors[second].add(first)
    return neighbors


def contrasting_instance_colors(boxes, *, preferred=None, fixed=None):
    """Greedy graph coloring using perceptual distance in CIELAB.

    Fixed colors stay put when adding an instance. Preferred colors from a prior
    frame are retained when their local contrast is within 15% of the best choice.
    Ties use IDs, never dictionary order or the global random-number generator.
    """
    boxes = {int(i): tuple(float(v) for v in box) for i, box in boxes.items()
             if box is not None and len(box) == 4
             and all(math.isfinite(v) for v in box) and box[2] > 0 and box[3] > 0}
    neighbors = neighboring_boxes(boxes)
    palette_index = {color: index for index, color in enumerate(BEE_PALETTE)}
    assigned = {i: palette_index[tuple(color)] for i, color in (fixed or {}).items()
                if i in boxes and tuple(color) in palette_index}
    pending = set(boxes) - assigned.keys()
    while pending:
        instance_id = max(pending, key=lambda i: (
            len({assigned[n] for n in neighbors[i] if n in assigned}),
            len(neighbors[i]), -i))
        used = [assigned[n] for n in sorted(neighbors[instance_id]) if n in assigned]
        previous = palette_index.get(tuple((preferred or {}).get(instance_id, ())))
        if used:
            scores = _DISTANCES[:, used].min(axis=1)
            best = int(np.argmax(scores))
            chosen = previous if previous is not None and scores[previous] >= 0.85 * scores[best] else best
        else:
            chosen = previous if previous is not None else (instance_id - 1) % len(BEE_PALETTE)
        assigned[instance_id] = chosen
        pending.remove(instance_id)
    return {i: BEE_PALETTE[index] for i, index in assigned.items()}
