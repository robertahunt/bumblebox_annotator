"""
Chamber-aligned temporal hive prior for batch video inference.

The prior keeps a rolling, normalized map of where hive pixels usually appear
inside each chamber/context. By default, current usable evidence is incorporated
before scoring bee contact; past-only scoring remains available. Bee-occluded
pixels do not contribute new evidence in either mode.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import json
import math
import os
from pathlib import Path

import cv2
import numpy as np


@dataclass
class TemporalHiveOverlap:
    """Bee overlap with the chamber-normalized temporal hive prior."""

    on_hive: Optional[bool] = None
    overlap_fraction: Optional[float] = None
    mean_probability: Optional[float] = None
    known_fraction: Optional[float] = None
    overlap_pixels: Optional[int] = None
    known_pixels: Optional[int] = None
    bee_pixels: Optional[int] = None
    mean_prior_weight: Optional[float] = None
    sum_prior_weight: Optional[float] = None


@dataclass
class TemporalHiveSnapshot:
    """Map snapshot: 0 unknown, 1 supported non-hive, 2 supported hive."""

    chamber_id: int
    bbox: Tuple[int, int, int, int]
    labels: np.ndarray


@dataclass
class _PriorState:
    hive_sum: np.ndarray
    weight_sum: np.ndarray
    last_time_seconds: Optional[float] = None
    observation_count: int = 0


class TemporalChamberStabilizer:
    """Causal, per-video smoothing of small chamber-box changes, not image registration."""

    alpha = 0.2
    reset_fraction = 0.02

    def __init__(self):
        self.boxes = {}
        self.frame_shape = None

    def reset(self):
        self.boxes.clear()
        self.frame_shape = None

    def prepare(self, chambers, frame_shape):
        shape = tuple(frame_shape[:2])
        if self.frame_shape != shape or set(self.boxes) != set(chambers):
            self.reset()
        self.frame_shape = shape
        prepared = {}
        for chamber_id, info in chambers.items():
            raw = np.array(TemporalHivePrior._chamber_bbox({'bbox': info.get('bbox')}, shape), dtype=float)
            previous = self.boxes.get(chamber_id)
            if previous is not None:
                size = np.array([previous[2] - previous[0], previous[3] - previous[1]] * 2)
                # Follow large moves immediately instead of dragging a stale box across the image.
                if np.max(np.abs(raw - previous) / np.maximum(size, 1)) <= self.reset_fraction:
                    raw = self.alpha * raw + (1 - self.alpha) * previous
            self.boxes[chamber_id] = raw
            prepared[chamber_id] = dict(info, temporal_bbox=tuple(raw))
        return prepared


class TemporalHivePrior:
    """Rolling hive probability map in normalized chamber coordinates."""

    def __init__(
        self,
        window_seconds: float = 8 * 60 * 60,
        resolution: Tuple[int, int] = (256, 256),
        hive_probability_threshold: float = 0.5,
        bee_overlap_threshold: float = 0.10,
        min_prior_weight: float = 2.0,
        cleanup_kernel_size: int = 3,
        min_component_pixels: int = 12,
        stabilize_chambers: bool = False,
        scoring_mode: str = 'updated',
    ):
        if scoring_mode not in ('updated', 'prior'):
            raise ValueError(f'Unknown temporal hive scoring mode: {scoring_mode}')
        self.window_seconds = max(1.0, float(window_seconds))
        self.resolution = (int(resolution[0]), int(resolution[1]))
        self.hive_probability_threshold = float(hive_probability_threshold)
        self.bee_overlap_threshold = float(bee_overlap_threshold)
        self.min_prior_weight = float(min_prior_weight)
        self.cleanup_kernel_size = max(0, int(cleanup_kernel_size))
        self.min_component_pixels = max(0, int(min_component_pixels))
        self.stabilize_chambers = bool(stabilize_chambers)
        self.scoring_mode = scoring_mode
        self._states: Dict[Tuple[str, int], _PriorState] = {}

    def query_bee_overlap(
        self,
        context_id: str,
        chamber_id: int,
        chamber_info: Dict,
        bee_mask: Optional[np.ndarray],
        bee_bbox,
        frame_shape: Tuple[int, int],
        observation_time_seconds: Optional[float],
    ) -> TemporalHiveOverlap:
        """Evaluate the current map state; the caller controls update/scoring order."""
        if chamber_info.get('temporal_unavailable', False):
            return self._empty_overlap()
        state = self._states.get((context_id, chamber_id))
        if state is None:
            return self._empty_overlap()

        self._decay_state(state, observation_time_seconds)

        bee_norm = self._normalize_detection(bee_mask, bee_bbox, chamber_info, frame_shape)
        bee_pixels = bee_norm > 0
        bee_pixel_count = int(np.sum(bee_pixels))
        if bee_pixel_count == 0:
            return self._empty_overlap()

        probability = self._probability_map(state)
        bee_prior_weights = state.weight_sum[bee_pixels]
        mean_prior_weight = float(np.mean(bee_prior_weights)) if len(bee_prior_weights) else None
        sum_prior_weight = float(np.sum(bee_prior_weights)) if len(bee_prior_weights) else None
        known_pixels = state.weight_sum >= self.min_prior_weight
        bee_known = bee_pixels & known_pixels
        known_count = int(np.sum(bee_known))
        known_fraction = known_count / bee_pixel_count

        if known_count == 0:
            return TemporalHiveOverlap(
                on_hive=None,
                overlap_fraction=None,
                mean_probability=None,
                known_fraction=float(known_fraction),
                overlap_pixels=0,
                known_pixels=known_count,
                bee_pixels=bee_pixel_count,
                mean_prior_weight=mean_prior_weight,
                sum_prior_weight=sum_prior_weight,
            )

        stable_hive = probability >= self.hive_probability_threshold
        overlap_pixels = int(np.sum(bee_known & stable_hive))
        overlap_fraction = float(overlap_pixels / known_count)
        mean_probability = float(np.mean(probability[bee_known]))

        return TemporalHiveOverlap(
            on_hive=overlap_fraction >= self.bee_overlap_threshold,
            overlap_fraction=overlap_fraction,
            mean_probability=mean_probability,
            known_fraction=float(known_fraction),
            overlap_pixels=overlap_pixels,
            known_pixels=known_count,
            bee_pixels=bee_pixel_count,
            mean_prior_weight=mean_prior_weight,
            sum_prior_weight=sum_prior_weight,
        )

    def visualization_snapshot(
        self, context_id: str, chamber_id: int, chamber_info: Dict,
        frame_shape: Tuple[int, int], observation_time_seconds: Optional[float],
    ) -> TemporalHiveSnapshot:
        """Copy the map at the time of capture without mutating its values or clock."""
        labels = np.zeros(self.resolution[::-1], dtype=np.uint8)
        state = (None if chamber_info.get('temporal_unavailable', False)
                 else self._states.get((context_id, chamber_id)))
        if state is not None:
            snapshot = _PriorState(state.hive_sum.copy(), state.weight_sum.copy(),
                                   state.last_time_seconds, state.observation_count)
            self._decay_state(snapshot, observation_time_seconds)
            known = snapshot.weight_sum >= self.min_prior_weight
            labels[known] = 1
            labels[known & (self._probability_map(snapshot) >= self.hive_probability_threshold)] = 2
        return TemporalHiveSnapshot(
            chamber_id, self._chamber_bbox(chamber_info, frame_shape), labels,
        )

    def update(
        self,
        context_id: str,
        chamber_id: int,
        chamber_info: Dict,
        hive_mask: Optional[np.ndarray],
        bee_detections: List,
        frame_shape: Tuple[int, int],
        observation_time_seconds: Optional[float],
    ):
        """Update the prior with current hive evidence, excluding bee-occluded pixels."""
        if chamber_info.get('temporal_unavailable', False) or hive_mask is None or not np.any(hive_mask > 0):
            return

        key = (context_id, chamber_id)
        state = self._states.get(key)
        if state is None:
            state = _PriorState(
                hive_sum=np.zeros(self.resolution[::-1], dtype=np.float32),
                weight_sum=np.zeros(self.resolution[::-1], dtype=np.float32),
            )
            self._states[key] = state

        self._decay_state(state, observation_time_seconds)

        hive_norm = self._clean_hive_evidence(
            self._normalize_mask(hive_mask, chamber_info, frame_shape)
        )
        chamber_valid = self._normalize_chamber_mask(chamber_info, frame_shape)
        bee_occlusion = self._normalize_bee_occlusion(bee_detections, chamber_info, frame_shape)
        valid_pixels = chamber_valid & ~bee_occlusion

        if not np.any(valid_pixels):
            return

        hive_pixels = (hive_norm > 0) & valid_pixels
        state.hive_sum[valid_pixels] += hive_pixels[valid_pixels].astype(np.float32)
        state.weight_sum[valid_pixels] += 1.0
        state.observation_count += 1

    def summaries(self) -> List[Dict]:
        """Return stable prior maps as export-ready dictionaries."""
        rows = []
        width, height = self.resolution

        for (context_id, chamber_id), state in sorted(self._states.items()):
            probability = self._probability_map(state)
            known_pixels = state.weight_sum >= self.min_prior_weight
            stable_hive = (probability >= self.hive_probability_threshold) & known_pixels
            hive_pixels = int(np.sum(stable_hive))

            if hive_pixels > 0:
                y_coords, x_coords = np.where(stable_hive)
                centroid_x = float(np.mean(x_coords) / max(1, width - 1))
                centroid_y = float(np.mean(y_coords) / max(1, height - 1))
                mean_weight = float(np.mean(state.weight_sum[stable_hive]))
            else:
                centroid_x = None
                centroid_y = None
                mean_weight = 0.0

            rows.append({
                "context_id": context_id,
                "chamber_id": chamber_id,
                "prior_hive_pixels_norm": hive_pixels,
                "centroid_x_norm": centroid_x,
                "centroid_y_norm": centroid_y,
                "mean_prior_weight": mean_weight,
                "max_prior_weight": float(np.max(state.weight_sum)) if state.weight_sum.size else 0.0,
                "observation_count": state.observation_count,
                "prior_polygon_norm": self._mask_to_polygon_string(stable_hive),
                "resolution_width": width,
                "resolution_height": height,
            })

        return rows

    def save_checkpoint(self, path: str | Path, metadata: Optional[Dict] = None) -> Path:
        """Persist the exact rolling prior state for fast batch resume."""
        checkpoint_path = Path(path)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

        metadata = dict(metadata or {})
        metadata["prior_settings"] = {
            "window_seconds": self.window_seconds,
            "resolution": list(self.resolution),
            "hive_probability_threshold": self.hive_probability_threshold,
            "bee_overlap_threshold": self.bee_overlap_threshold,
            "min_prior_weight": self.min_prior_weight,
            "cleanup_kernel_size": self.cleanup_kernel_size,
            "min_component_pixels": self.min_component_pixels,
            "stabilize_chambers": self.stabilize_chambers,
            "scoring_mode": self.scoring_mode,
        }

        state_metadata = []
        payload = {}
        for state_index, ((context_id, chamber_id), state) in enumerate(sorted(self._states.items())):
            hive_key = f"hive_sum_{state_index}"
            weight_key = f"weight_sum_{state_index}"
            state_metadata.append({
                "context_id": context_id,
                "chamber_id": int(chamber_id),
                "hive_key": hive_key,
                "weight_key": weight_key,
                "last_time_seconds": state.last_time_seconds,
                "observation_count": int(state.observation_count),
            })
            payload[hive_key] = state.hive_sum.astype(np.float32, copy=False)
            payload[weight_key] = state.weight_sum.astype(np.float32, copy=False)

        metadata["states"] = state_metadata
        payload["metadata"] = np.array(json.dumps(metadata), dtype=np.str_)

        tmp_path = checkpoint_path.with_name(f"{checkpoint_path.name}.tmp")
        try:
            with tmp_path.open("wb") as handle:
                np.savez_compressed(handle, **payload)
            os.replace(tmp_path, checkpoint_path)
        finally:
            if tmp_path.exists():
                try:
                    tmp_path.unlink()
                except OSError:
                    pass

        return checkpoint_path

    @classmethod
    def load_checkpoint(cls, path: str | Path) -> Tuple["TemporalHivePrior", Dict]:
        """Load an exact rolling prior state saved by save_checkpoint."""
        checkpoint_path = Path(path)
        with np.load(checkpoint_path, allow_pickle=False) as data:
            metadata = json.loads(str(data["metadata"].item()))
            settings = metadata.get("prior_settings", {})
            prior = cls(
                window_seconds=float(settings.get("window_seconds", 8 * 60 * 60)),
                resolution=tuple(settings.get("resolution", (256, 256))),
                hive_probability_threshold=float(settings.get("hive_probability_threshold", 0.5)),
                bee_overlap_threshold=float(settings.get("bee_overlap_threshold", 0.10)),
                min_prior_weight=float(settings.get("min_prior_weight", 2.0)),
                cleanup_kernel_size=int(settings.get("cleanup_kernel_size", 3)),
                min_component_pixels=int(settings.get("min_component_pixels", 12)),
                stabilize_chambers=bool(settings.get("stabilize_chambers", False)),
                scoring_mode=settings.get("scoring_mode", "prior"),
            )

            for state_info in metadata.get("states", []):
                context_id = str(state_info["context_id"])
                chamber_id = int(state_info["chamber_id"])
                prior._states[(context_id, chamber_id)] = _PriorState(
                    hive_sum=data[state_info["hive_key"]].astype(np.float32),
                    weight_sum=data[state_info["weight_key"]].astype(np.float32),
                    last_time_seconds=state_info.get("last_time_seconds"),
                    observation_count=int(state_info.get("observation_count") or 0),
                )

        return prior, metadata

    def _decay_state(self, state: _PriorState, observation_time_seconds: Optional[float]):
        """Apply exponential time decay when timestamps are available."""
        if observation_time_seconds is None:
            return

        if state.last_time_seconds is None:
            state.last_time_seconds = observation_time_seconds
            return

        elapsed = observation_time_seconds - state.last_time_seconds
        if elapsed <= 0:
            state.last_time_seconds = observation_time_seconds
            return

        decay = math.exp(-elapsed / self.window_seconds)
        state.hive_sum *= decay
        state.weight_sum *= decay
        state.last_time_seconds = observation_time_seconds

    def _probability_map(self, state: _PriorState) -> np.ndarray:
        probability = np.zeros_like(state.hive_sum, dtype=np.float32)
        np.divide(
            state.hive_sum,
            state.weight_sum,
            out=probability,
            where=state.weight_sum > 0,
        )
        return probability

    def _clean_hive_evidence(self, mask: np.ndarray) -> np.ndarray:
        """Suppress single-frame edge noise before adding hive evidence to the prior."""
        binary = (mask > 0).astype(np.uint8)
        if not np.any(binary):
            return binary.astype(bool)

        if self.cleanup_kernel_size >= 2:
            kernel_size = self.cleanup_kernel_size
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
            binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)
            binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)

        if self.min_component_pixels > 1:
            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
            cleaned = np.zeros_like(binary)
            for label in range(1, num_labels):
                if int(stats[label, cv2.CC_STAT_AREA]) >= self.min_component_pixels:
                    cleaned[labels == label] = 1
            binary = cleaned

        return binary > 0

    def _normalize_detection(
        self,
        mask: Optional[np.ndarray],
        bbox,
        chamber_info: Dict,
        frame_shape: Tuple[int, int],
    ) -> np.ndarray:
        if mask is not None:
            return self._normalize_mask(mask, chamber_info, frame_shape)
        return self._normalize_bbox(bbox, chamber_info, frame_shape)

    def _normalize_mask(
        self,
        mask: np.ndarray,
        chamber_info: Dict,
        frame_shape: Tuple[int, int],
    ) -> np.ndarray:
        x1, y1, x2, y2 = self._chamber_bbox(chamber_info, frame_shape)
        cropped = mask[y1:y2, x1:x2]
        if cropped.size == 0:
            return np.zeros(self.resolution[::-1], dtype=bool)
        resized = cv2.resize(
            (cropped > 0).astype(np.uint8),
            self.resolution,
            interpolation=cv2.INTER_NEAREST,
        )
        return resized > 0

    def _normalize_chamber_mask(self, chamber_info: Dict, frame_shape: Tuple[int, int]) -> np.ndarray:
        chamber_mask = chamber_info.get("mask") if chamber_info else None
        if chamber_mask is None:
            return np.ones(self.resolution[::-1], dtype=bool)
        return self._normalize_mask(chamber_mask, chamber_info, frame_shape)

    def _normalize_bee_occlusion(
        self,
        bee_detections: List,
        chamber_info: Dict,
        frame_shape: Tuple[int, int],
    ) -> np.ndarray:
        occlusion = np.zeros(self.resolution[::-1], dtype=bool)
        for bee in bee_detections:
            mask = getattr(bee, "mask", None)
            bbox = getattr(bee, "bbox", None)
            if mask is not None:
                occlusion |= self._normalize_mask(mask, chamber_info, frame_shape)
            elif bbox is not None:
                occlusion |= self._normalize_bbox(bbox, chamber_info, frame_shape)
        return occlusion

    def _normalize_bbox(self, bbox, chamber_info: Dict, frame_shape: Tuple[int, int]) -> np.ndarray:
        normalized = np.zeros(self.resolution[::-1], dtype=bool)
        if bbox is None:
            return normalized

        x1, y1, x2, y2 = self._chamber_bbox(chamber_info, frame_shape)
        chamber_w = max(1, x2 - x1)
        chamber_h = max(1, y2 - y1)
        width, height = self.resolution

        bx1, by1, bx2, by2 = [float(v) for v in bbox]
        nx1 = int(np.floor((bx1 - x1) / chamber_w * width))
        ny1 = int(np.floor((by1 - y1) / chamber_h * height))
        nx2 = int(np.ceil((bx2 - x1) / chamber_w * width))
        ny2 = int(np.ceil((by2 - y1) / chamber_h * height))

        nx1 = max(0, min(width, nx1))
        nx2 = max(0, min(width, nx2))
        ny1 = max(0, min(height, ny1))
        ny2 = max(0, min(height, ny2))

        if nx2 > nx1 and ny2 > ny1:
            normalized[ny1:ny2, nx1:nx2] = True
        return normalized

    @staticmethod
    def _chamber_bbox(chamber_info: Dict, frame_shape: Tuple[int, int]) -> Tuple[int, int, int, int]:
        frame_h, frame_w = frame_shape[:2]
        bbox = chamber_info.get('temporal_bbox', chamber_info.get('bbox')) if chamber_info else None
        if bbox is None:
            return (0, 0, frame_w, frame_h)

        x1, y1, x2, y2 = [int(round(float(v))) for v in bbox]
        x1 = max(0, min(frame_w - 1, x1))
        y1 = max(0, min(frame_h - 1, y1))
        x2 = max(x1 + 1, min(frame_w, x2))
        y2 = max(y1 + 1, min(frame_h, y2))
        return (x1, y1, x2, y2)

    def _mask_to_polygon_string(self, mask: np.ndarray) -> str:
        if mask is None or not np.any(mask):
            return ""

        contours, _ = cv2.findContours(
            mask.astype(np.uint8),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        if not contours:
            return ""

        largest = max(contours, key=cv2.contourArea)
        epsilon = 0.02 * cv2.arcLength(largest, True)
        simplified = cv2.approxPolyDP(largest, epsilon, True).reshape(-1, 2)

        width, height = self.resolution
        coords = []
        for x, y in simplified:
            coords.append(f"{float(x) / max(1, width - 1):.6f}")
            coords.append(f"{float(y) / max(1, height - 1):.6f}")
        return " ".join(coords)

    @staticmethod
    def _empty_overlap() -> TemporalHiveOverlap:
        return TemporalHiveOverlap(
            on_hive=None,
            overlap_fraction=None,
            mean_probability=None,
            known_fraction=None,
            overlap_pixels=None,
            known_pixels=None,
            bee_pixels=None,
            mean_prior_weight=None,
            sum_prior_weight=None,
        )
