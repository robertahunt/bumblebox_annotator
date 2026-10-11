"""
Shared ArUco-based track identity correction.

This module contains the same ArUco assignment and re-identification rules used
by batch video inference, so other tracking workflows can use the identical
identity guidance behavior.
"""

from collections import defaultdict
from typing import Callable, Dict, List, Optional

import numpy as np

from core.instance_tracker import Detection
from core.marker_detector import MarkerDetector


class ArucoTrackingIdentityManager:
    """Assign ArUco codes to track IDs and merge re-identified tracks."""

    def __init__(self, log_callback=None, verbose_output: bool = False):
        self.log_callback = log_callback
        self.verbose_output = verbose_output
        self.marker_detector = MarkerDetector(
            aruco_dicts=['4x4_50', '4x4_100', '4x4_250', '4x4_1000'],
            enable_aruco=True,
            enable_qr=False,
            min_confidence=0.2,
            debug=False,
            debug_folder=None
        )
        self.bee_to_aruco: Dict[int, str] = {}
        self.aruco_to_bee: Dict[str, int] = {}
        self.bee_frames: Dict[int, set] = defaultdict(set)

    def _log(self, message: str):
        if self.log_callback:
            self.log_callback(message)
        else:
            print(message)

    def _log_verbose(self, message: str):
        if self.verbose_output:
            self._log(message)

    def detect_bee_aruco_codes(
        self,
        frame: np.ndarray,
        bee_detections: List[Detection],
        frame_count: int,
        merge_callback: Optional[Callable[[int, int, str], None]] = None
    ) -> Dict[int, str]:
        """
        Detect ArUco codes on individual bees using efficient full-frame detection.

        Strategy:
        1. Detect all ArUco codes in the full frame once.
        2. Match them to bee detections based on spatial overlap.
        3. Validate assignments with strict rules:
           - If ArUco in multiple boxes in this frame: reject.
           - If ArUco already assigned to another bee active in this frame: reject.
           - If ArUco not yet assigned and only in one bee's box: assign.
           - If ArUco was assigned before but that bee is inactive: re-identify.
        """
        self._record_active_bees(bee_detections, frame_count)

        if self.verbose_output and frame_count == 1:
            self._log("\n=== Frame 1 ArUco Diagnostic ===")
            self._log(f"Total detections: {len(bee_detections)}")

            dets_with_id = sum(1 for det in bee_detections if det.instance_id is not None)
            dets_with_mask = sum(1 for det in bee_detections if det.mask is not None)
            dets_with_both = sum(
                1 for det in bee_detections
                if det.instance_id is not None and det.mask is not None
            )

            self._log(f"  Detections with instance_id: {dets_with_id}")
            self._log(f"  Detections with mask: {dets_with_mask}")
            self._log(f"  Detections with BOTH (valid for ArUco): {dets_with_both}")

            sample = next((det for det in bee_detections if det.mask is not None), None)
            if sample:
                self._log(
                    f"  Sample mask: shape={sample.mask.shape}, dtype={sample.mask.dtype}, "
                    f"range=[{sample.mask.min()}, {sample.mask.max()}]"
                )
                self._log(
                    f"  Sample has instance_id: {sample.instance_id is not None} "
                    f"(id={sample.instance_id})"
                )
            self._log("=" * 35 + "\n")

        annotations = []
        for det in bee_detections:
            if det.instance_id is None or det.mask is None:
                continue

            x1, y1, x2, y2 = det.bbox
            bbox = [x1, y1, x2 - x1, y2 - y1]

            mask = det.mask
            if mask.max() > 1:
                mask = (mask > 127).astype(np.uint8)

            annotations.append({
                'instance_id': det.instance_id,
                'mask_id': det.instance_id,
                'category': 'bee',
                'bbox': bbox,
                'mask': mask
            })

        if not annotations:
            return {}

        detections = self.marker_detector.detect_aruco_in_bee_instances(
            image=frame,
            annotations=annotations,
            reject_multiple=True
        )

        marker_codes = {
            instance_id: self._marker_code(marker_result)
            for instance_id, marker_result in detections.items()
        }

        return self.apply_bee_aruco_codes(
            bee_detections=bee_detections,
            frame_count=frame_count,
            marker_codes=marker_codes,
            merge_callback=merge_callback
        )

    def apply_bee_aruco_codes(
        self,
        bee_detections: List[Detection],
        frame_count: int,
        marker_codes: Dict[int, str],
        merge_callback: Optional[Callable[[int, int, str], None]] = None
    ) -> Dict[int, str]:
        """Apply already-matched per-bee ArUco codes using the normal identity rules."""
        self._record_active_bees(bee_detections, frame_count)
        active_bee_ids = {det.instance_id for det in bee_detections if det.instance_id is not None}

        frame_marker_codes = dict(marker_codes)

        aruco_to_instances = {}
        for instance_id, aruco_code in marker_codes.items():
            aruco_to_instances.setdefault(str(aruco_code), []).append(instance_id)

        ambiguous_codes = set()
        for aruco_code, instance_list in aruco_to_instances.items():
            if len(instance_list) > 1:
                ambiguous_codes.add(aruco_code)
                if self.verbose_output and frame_count <= 10:
                    self._log(
                        f"  Frame {frame_count}: ArUco {aruco_code} in "
                        f"{len(instance_list)} boxes (rejected): {instance_list}"
                    )

        for instance_id, aruco_code in marker_codes.items():
            aruco_code = str(aruco_code)

            if aruco_code in ambiguous_codes:
                continue

            if aruco_code in self.aruco_to_bee:
                assigned_bee_id = self.aruco_to_bee[aruco_code]

                if assigned_bee_id in active_bee_ids:
                    if assigned_bee_id != instance_id:
                        if self.verbose_output and frame_count <= 10:
                            self._log(
                                f"  Frame {frame_count}: ArUco {aruco_code} already on "
                                f"active bee {assigned_bee_id}, rejecting for bee {instance_id}"
                            )
                        continue
                else:
                    if self._bees_coexisted(assigned_bee_id, instance_id):
                        if self.verbose_output and frame_count <= 10:
                            self._log(
                                f"  Frame {frame_count}: ArUco {aruco_code} conflict - "
                                f"bees {assigned_bee_id} and {instance_id} coexisted, rejecting"
                            )
                        continue

                    if self.verbose_output and frame_count <= 10:
                        self._log(
                            f"  Frame {frame_count}: Re-identification - merging bee "
                            f"{instance_id} into bee {assigned_bee_id} (ArUco {aruco_code})"
                        )
                    self._merge_bee_tracks(
                        source_id=instance_id,
                        target_id=assigned_bee_id,
                        aruco_code=aruco_code,
                        bee_detections=bee_detections,
                        merge_callback=merge_callback
                    )
                    frame_marker_codes[assigned_bee_id] = aruco_code
                    if instance_id in frame_marker_codes:
                        del frame_marker_codes[instance_id]
                    continue

            if instance_id in self.bee_to_aruco:
                if self.bee_to_aruco[instance_id] != aruco_code:
                    if self.verbose_output and frame_count <= 10:
                        self._log(
                            f"  Frame {frame_count}: Bee {instance_id} already has "
                            f"ArUco {self.bee_to_aruco[instance_id]}, rejecting new code {aruco_code}"
                        )
                    continue
            else:
                self.bee_to_aruco[instance_id] = aruco_code
                self.aruco_to_bee[aruco_code] = instance_id
                if self.verbose_output and frame_count <= 10:
                    self._log(f"  Frame {frame_count}: Assigned ArUco {aruco_code} to bee {instance_id}")

        return frame_marker_codes

    def _record_active_bees(self, bee_detections: List[Detection], frame_count: int):
        active_bee_ids = {det.instance_id for det in bee_detections if det.instance_id is not None}
        for bee_id in active_bee_ids:
            self.bee_frames.setdefault(bee_id, set()).add(frame_count)

    def finalize_detections(self, detections):
        """Update stored records that expose bee_id/aruco_code attributes."""
        for detection in detections:
            if detection.bee_id in self.bee_to_aruco:
                detection.aruco_code = self.bee_to_aruco[detection.bee_id]

    def finalize_trajectories(self, trajectories):
        """Update stored trajectory objects with final ArUco codes."""
        for bee_id, trajectory in trajectories.items():
            if bee_id in self.bee_to_aruco:
                trajectory.aruco_code = self.bee_to_aruco[bee_id]

    def _marker_code(self, marker_result) -> str:
        if marker_result.marker_type == 'aruco':
            return str(int(marker_result.marker_id))
        return str(marker_result.marker_id)

    def _bees_coexisted(self, bee_id_a: int, bee_id_b: int) -> bool:
        frames_a = self.bee_frames.get(bee_id_a, set())
        frames_b = self.bee_frames.get(bee_id_b, set())
        return len(frames_a & frames_b) > 0

    def _merge_bee_tracks(
        self,
        source_id: int,
        target_id: int,
        aruco_code: str,
        bee_detections: List[Detection],
        merge_callback: Optional[Callable[[int, int, str], None]]
    ):
        if merge_callback:
            merge_callback(source_id, target_id, aruco_code)

        for det in bee_detections:
            if det.instance_id == source_id:
                det.instance_id = target_id

        if source_id in self.bee_frames:
            self.bee_frames.setdefault(target_id, set()).update(self.bee_frames[source_id])
            del self.bee_frames[source_id]

        if source_id in self.bee_to_aruco:
            del self.bee_to_aruco[source_id]
        self.bee_to_aruco[target_id] = aruco_code
        self.aruco_to_bee[aruco_code] = target_id

        self._log_verbose(f"    Merged bee {source_id} into {target_id} (ArUco {aruco_code})")
