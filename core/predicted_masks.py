"""Tidying a model's predicted mask for a single instance."""

import cv2
import numpy as np

# A separate piece at least this fraction of the largest piece's area is kept as
# part of the instance (a bee crossed by a wire). Smaller stray pieces are noise.
MIN_PIECE_FRACTION = 0.1


def instance_pieces(mask, reference=None, min_fraction=MIN_PIECE_FRACTION):
    """The pieces of a predicted mask that belong to the instance, as a 0/255 array.

    The largest piece is always kept. Another piece is kept when it is at least
    ``min_fraction`` of the largest piece's area, or when it overlaps
    ``reference``: the instance's existing mask, where there is one. Anything
    else is dropped as a stray fragment.

    Each kept piece is drawn filled from its outline, so holes inside a piece
    are closed. Keeping only the largest piece cut a bee in two wherever
    something crossed it.
    """
    binary = (np.asarray(mask) > 0).astype(np.uint8)
    if reference is not None:
        reference = np.asarray(reference) > 0
        if reference.shape != binary.shape:
            raise ValueError('The reference mask must match the predicted mask')
    kept = np.zeros_like(binary)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return kept
    areas = [cv2.contourArea(contour) for contour in contours]
    largest = max(range(len(contours)), key=areas.__getitem__)
    for index, contour in enumerate(contours):
        large_enough = areas[largest] > 0 and areas[index] >= min_fraction * areas[largest]
        if index != largest and not large_enough:
            if reference is None:
                continue
            piece = np.zeros_like(binary)
            cv2.drawContours(piece, [contour], 0, 1, -1)
            if not np.any(reference[piece > 0]):
                continue
        cv2.drawContours(kept, [contour], 0, 255, -1)
    return kept
