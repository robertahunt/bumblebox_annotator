"""Image-space tag measurements and bounded-memory video browsing."""

from pathlib import Path

import cv2
import numpy as np


def measure_tag(points, frame_width, frame_height):
    """Measure four cyclically ordered outer tag corners in source pixels."""
    corners = np.asarray(points, dtype=np.float64)
    if corners.shape != (4, 2) or not np.isfinite(corners).all():
        raise ValueError("Mark four finite corner positions around the tag.")
    if frame_width <= 0 or frame_height <= 0:
        raise ValueError("The frame dimensions must be positive.")
    if (corners < 0).any() or (corners[:, 0] >= frame_width).any() or (corners[:, 1] >= frame_height).any():
        raise ValueError("Tag corners must be inside the original frame.")
    contour = corners.astype(np.float32).reshape(-1, 1, 2)
    if not cv2.isContourConvex(contour) or cv2.contourArea(contour) <= 0:
        raise ValueError("Mark the four corners in order around the tag, without crossing sides.")
    perimeter = float(np.linalg.norm(corners - np.roll(corners, -1, axis=0), axis=1).sum())
    return {"perimeter_px": perimeter, "perimeter_rate": perimeter / max(frame_width, frame_height)}


def tag_perimeter_bounds(smallest_rate, largest_rate, margin_percent=10.0):
    """Add symmetric tolerance to measured rates without camera-specific caps."""
    if not np.isfinite([smallest_rate, largest_rate, margin_percent]).all():
        raise ValueError("Measurements and margin must be finite.")
    if smallest_rate <= 0 or largest_rate < smallest_rate:
        raise ValueError("The smallest tag must not have a larger perimeter rate than the largest tag.")
    if not 0 <= margin_percent < 100:
        raise ValueError("The margin must be between 0 and 100 percent.")
    lower = round(smallest_rate * (1 - margin_percent / 100), 6)
    upper = round(largest_rate * (1 + margin_percent / 100), 6)
    if not 0 < lower < upper:
        raise ValueError("The measurements must produce distinct positive lower and upper bounds.")
    return lower, upper


class MeasurementVideoReader:
    """Keep only the current frame; MJPEG files need not report a frame count."""

    def __init__(self, path):
        self.path = Path(path)
        self.cap = None
        self._open()
        count = self.cap.get(cv2.CAP_PROP_FRAME_COUNT)
        self.frame_count = int(count) if np.isfinite(count) and 0 < count < 2**31 else None
        self.current_index = None
        self.current_frame = None

    def _open(self):
        self.close()
        self.cap = cv2.VideoCapture(str(self.path))
        self.next_index = 0
        if not self.cap.isOpened():
            self.close()
            raise ValueError(f"Could not open video: {self.path}")

    def read(self, index):
        if index < 0:
            raise ValueError("Frame number must not be negative.")
        if index == self.current_index:
            return self.current_frame
        if index != self.next_index:
            # Unknown-length streams often claim to seek successfully without moving.
            # Reopen and advance sequentially instead of trusting their frame position.
            if self.frame_count is None or not self.cap.set(cv2.CAP_PROP_POS_FRAMES, index):
                self._open()
            elif abs(self.cap.get(cv2.CAP_PROP_POS_FRAMES) - index) < 0.5:
                self.next_index = index
            else:
                self._open()
        while self.next_index < index:
            if not self.cap.grab():
                raise ValueError(f"Could not read frame {index + 1}; the video may have ended.")
            self.next_index += 1
        ok, frame = self.cap.read()
        if not ok or frame is None:
            raise ValueError(f"Could not read frame {index + 1}; the video may have ended.")
        self.next_index = index + 1
        self.current_index, self.current_frame = index, frame
        return frame

    def close(self):
        if self.cap is not None:
            self.cap.release()
            self.cap = None
