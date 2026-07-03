"""
Video-specific ArUco parameter bank optimization.

The optimizer samples frames from one video, evaluates a grid of OpenCV ArUco
DetectorParameters, then keeps a compact bank of whole parameter sets that
cover different sampled-frame conditions.
"""

from __future__ import annotations

import csv
import json
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import cv2
import numpy as np


ARUCO_DICTIONARIES = {
    "4x4_50": cv2.aruco.DICT_4X4_50,
    "4x4_100": cv2.aruco.DICT_4X4_100,
    "4x4_250": cv2.aruco.DICT_4X4_250,
    "4x4_1000": cv2.aruco.DICT_4X4_1000,
    "5x5_50": cv2.aruco.DICT_5X5_50,
    "5x5_100": cv2.aruco.DICT_5X5_100,
    "5x5_250": cv2.aruco.DICT_5X5_250,
    "5x5_1000": cv2.aruco.DICT_5X5_1000,
    "6x6_50": cv2.aruco.DICT_6X6_50,
    "6x6_100": cv2.aruco.DICT_6X6_100,
    "6x6_250": cv2.aruco.DICT_6X6_250,
    "6x6_1000": cv2.aruco.DICT_6X6_1000,
    "7x7_50": cv2.aruco.DICT_7X7_50,
    "7x7_100": cv2.aruco.DICT_7X7_100,
    "7x7_250": cv2.aruco.DICT_7X7_250,
    "7x7_1000": cv2.aruco.DICT_7X7_1000,
}

OPTIMIZED_ARUCO_PARAM_KEYS = (
    "minMarkerPerimeterRate",
    "maxMarkerPerimeterRate",
    "adaptiveThreshWinSizeMin",
    "adaptiveThreshWinSizeMax",
    "adaptiveThreshWinSizeStep",
    "polygonalApproxAccuracyRate",
    "adaptiveThreshConstant",
)

PROFILE_PARAMETER_SPACE = {
    "quick": {
        "minMarkerPerimeterRate": [0.015, 0.02, 0.03],
        "maxMarkerPerimeterRate": [4.0],
        "adaptiveThreshWinSizeMin": [3, 5],
        "adaptiveThreshWinSizeMax": [23, 31],
        "adaptiveThreshWinSizeStep": [2, 4],
        "polygonalApproxAccuracyRate": [0.05, 0.08],
        "adaptiveThreshConstant": [7],
    },
    "balanced": {
        "minMarkerPerimeterRate": [0.01, 0.015, 0.02, 0.03],
        "maxMarkerPerimeterRate": [4.0],
        "adaptiveThreshWinSizeMin": [3, 5, 7],
        "adaptiveThreshWinSizeMax": [21, 31, 41],
        "adaptiveThreshWinSizeStep": [2, 4],
        "polygonalApproxAccuracyRate": [0.04, 0.06, 0.08],
        "adaptiveThreshConstant": [7],
    },
    "deep": {
        "minMarkerPerimeterRate": [0.008, 0.012, 0.016, 0.02, 0.03],
        "maxMarkerPerimeterRate": [4.0],
        "adaptiveThreshWinSizeMin": [3, 5, 7],
        "adaptiveThreshWinSizeMax": [21, 31, 41],
        "adaptiveThreshWinSizeStep": [2, 4],
        "polygonalApproxAccuracyRate": [0.04, 0.05, 0.06, 0.08],
        "adaptiveThreshConstant": [5, 7],
    },
    "daily": {
        "minMarkerPerimeterRate": [0.019153],
        "maxMarkerPerimeterRate": [0.052808],
        "adaptiveThreshWinSizeMin": [3, 5],
        "adaptiveThreshWinSizeMax": [29, 36, 41, 57, 73, 81, 105, 127, 151],
        "adaptiveThreshWinSizeStep": [2, 3],
        "polygonalApproxAccuracyRate": [0.06, 0.08],
        "adaptiveThreshConstant": [1, 3, 5, 7, 9, 11],
    },
}

ProgressCallback = Callable[[int, int, Dict], None]
StopCallback = Callable[[], bool]


@dataclass
class ArucoCandidate:
    rank: int
    score: float
    params: Dict[str, float | int]
    mean_detected: float
    mean_decoded: float
    mean_filtered: float
    mean_rejected: float
    std_detected: float
    coverage_frames: int
    eval_fps: float
    runtime_seconds: float
    frame_detected_counts: List[int]


@dataclass
class ArucoOptimizationResult:
    created_at: str
    video_path: str
    dictionary: str
    profile: str
    sampled_frame_indices: List[int]
    parameter_combinations_total: int
    combinations_evaluated: int
    output_dir: str
    summary_json_path: str
    candidates_csv_path: str
    parameter_bank: List[Dict[str, float | int]]
    selected_candidates: List[ArucoCandidate]
    top_candidates: List[ArucoCandidate]

    def to_dict(self) -> Dict:
        payload = asdict(self)
        payload["selected_candidates"] = [asdict(item) for item in self.selected_candidates]
        payload["top_candidates"] = [asdict(item) for item in self.top_candidates]
        return payload


def normalize_dictionary_name(dictionary_name: str) -> str:
    raw = str(dictionary_name or "").strip()
    if raw.startswith("DICT_"):
        raw = raw[5:]
    key = raw.lower()
    if key not in ARUCO_DICTIONARIES:
        raise ValueError(f"Unsupported ArUco dictionary: {dictionary_name}")
    return key


def load_tag_ids(path: str | Path) -> Set[int]:
    tag_path = Path(path).expanduser()
    if not tag_path.exists():
        raise FileNotFoundError(f"Tag list does not exist: {tag_path}")

    text = tag_path.read_text().strip()
    if not text:
        return set()

    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        payload = None

    if isinstance(payload, list):
        return {int(item) for item in payload}
    if isinstance(payload, dict):
        for key in ("tag_ids", "allowed_tag_ids", "ids", "tags"):
            value = payload.get(key)
            if isinstance(value, list):
                return {int(item) for item in value}

    out = set()
    for token in text.replace(",", "\n").replace("\t", "\n").splitlines():
        token = token.strip()
        if not token:
            continue
        try:
            out.add(int(token))
        except ValueError:
            continue
    return out


def sample_video_frames(video_path: Path, sample_count: int) -> Tuple[List[np.ndarray], List[int], int]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video for ArUco optimization: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if total_frames > 0:
        indices = np.linspace(0, max(0, total_frames - 1), max(1, sample_count), dtype=int)
        frame_indices = sorted(set(int(idx) for idx in indices))
    else:
        frame_indices = list(range(max(1, sample_count)))

    frames = []
    used_indices = []
    for frame_index in frame_indices:
        if total_frames > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = cap.read()
        if not ok or frame is None:
            continue
        frames.append(frame)
        used_indices.append(frame_index)
        if total_frames <= 0 and len(frames) >= sample_count:
            break

    cap.release()

    if not frames:
        raise RuntimeError(f"No readable frames sampled from: {video_path}")
    return frames, used_indices, total_frames


def build_parameter_grid(
    profile: str,
    sweep_overrides: Optional[Dict[str, Sequence[float | int]]] = None,
    max_combinations: Optional[int] = None,
) -> List[Dict[str, float | int]]:
    profile_key = str(profile or "daily").strip().lower()
    if profile_key not in PROFILE_PARAMETER_SPACE:
        raise ValueError(f"profile must be one of {sorted(PROFILE_PARAMETER_SPACE)}")

    space = {key: list(values) for key, values in PROFILE_PARAMETER_SPACE[profile_key].items()}
    for key, values in (sweep_overrides or {}).items():
        if key not in OPTIMIZED_ARUCO_PARAM_KEYS:
            raise ValueError(f"Unsupported ArUco sweep key: {key}")
        parsed = []
        for value in values:
            if key in {"minMarkerPerimeterRate", "maxMarkerPerimeterRate", "polygonalApproxAccuracyRate"}:
                parsed.append(round(float(value), 6))
            else:
                parsed.append(int(value))
        if parsed:
            space[key] = sorted(set(parsed))

    keys = list(space.keys())
    grid = []

    def walk(index: int, current: Dict[str, float | int]) -> None:
        if index >= len(keys):
            params = dict(current)
            if float(params["maxMarkerPerimeterRate"]) <= float(params["minMarkerPerimeterRate"]):
                return
            if int(params["adaptiveThreshWinSizeMin"]) >= int(params["adaptiveThreshWinSizeMax"]):
                return
            if int(params["adaptiveThreshWinSizeStep"]) <= 0:
                return
            grid.append(params)
            return
        key = keys[index]
        for value in space[key]:
            current[key] = value
            walk(index + 1, current)

    walk(0, {})

    if not grid:
        raise RuntimeError("ArUco parameter grid is empty after validation.")

    if max_combinations is not None and int(max_combinations) > 0 and len(grid) > int(max_combinations):
        grid = grid[: int(max_combinations)]
    return grid


def _detector_params_from_dict(params: Dict[str, float | int]) -> cv2.aruco.DetectorParameters:
    detector_params = cv2.aruco.DetectorParameters()
    for key, value in params.items():
        if hasattr(detector_params, key):
            setattr(detector_params, key, value)
    return detector_params


def _corner_perimeter_rate(corner: np.ndarray, frame_width: int, frame_height: int) -> float:
    points = np.asarray(corner, dtype=np.float32).reshape(-1, 2)
    if len(points) < 4:
        return 0.0
    perimeter = float(cv2.arcLength(points, True))
    return perimeter / float(max(1, frame_width + frame_height))


def _evaluate_candidate(
    params: Dict[str, float | int],
    frames: Sequence[np.ndarray],
    dictionary_key: str,
    allowed_tag_ids: Optional[Set[int]],
    expected_tags: Optional[float],
) -> ArucoCandidate:
    detector_params = _detector_params_from_dict(params)
    dictionary = cv2.aruco.getPredefinedDictionary(ARUCO_DICTIONARIES[dictionary_key])
    detector = cv2.aruco.ArucoDetector(dictionary, detector_params)

    detected_counts = []
    decoded_counts = []
    filtered_counts = []
    rejected_counts = []
    start = time.perf_counter()

    for frame in frames:
        gray = frame if frame.ndim == 2 else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_height, frame_width = gray.shape[:2]
        corners, ids, rejected = detector.detectMarkers(gray)

        decoded = 0
        filtered = 0
        valid_ids = set()
        if ids is not None and len(ids) > 0:
            for corner, raw_marker_id in zip(corners, ids.flatten().tolist()):
                marker_id = int(raw_marker_id)
                decoded += 1
                if allowed_tag_ids is not None and marker_id not in allowed_tag_ids:
                    filtered += 1
                    continue

                min_rate = float(params.get("minMarkerPerimeterRate", 0.0) or 0.0)
                max_rate = float(params.get("maxMarkerPerimeterRate", 4.0) or 4.0)
                perimeter_rate = _corner_perimeter_rate(corner, frame_width, frame_height)
                if perimeter_rate < min_rate or perimeter_rate > max_rate:
                    filtered += 1
                    continue
                valid_ids.add(marker_id)

        detected_counts.append(len(valid_ids))
        decoded_counts.append(decoded)
        filtered_counts.append(filtered)
        rejected_counts.append(len(rejected) if rejected is not None else 0)

    runtime = max(1e-9, time.perf_counter() - start)
    mean_detected = float(np.mean(detected_counts)) if detected_counts else 0.0
    mean_decoded = float(np.mean(decoded_counts)) if decoded_counts else 0.0
    mean_filtered = float(np.mean(filtered_counts)) if filtered_counts else 0.0
    mean_rejected = float(np.mean(rejected_counts)) if rejected_counts else 0.0
    std_detected = float(np.std(detected_counts)) if detected_counts else 0.0
    coverage_target = 1.0 if not expected_tags else max(1.0, min(float(expected_tags), mean_detected or float(expected_tags)))
    coverage_frames = sum(1 for count in detected_counts if count >= coverage_target)
    eval_fps = len(frames) / runtime

    score = mean_detected
    score += 0.15 * coverage_frames
    score += 0.05 * min(eval_fps, 90.0) / 90.0
    score -= 0.01 * std_detected
    score -= 0.005 * mean_rejected
    score -= 0.10 * mean_filtered
    if expected_tags:
        score -= 0.10 * abs(mean_detected - float(expected_tags)) / float(expected_tags)

    return ArucoCandidate(
        rank=0,
        score=float(score),
        params=dict(params),
        mean_detected=mean_detected,
        mean_decoded=mean_decoded,
        mean_filtered=mean_filtered,
        mean_rejected=mean_rejected,
        std_detected=std_detected,
        coverage_frames=int(coverage_frames),
        eval_fps=float(eval_fps),
        runtime_seconds=float(runtime),
        frame_detected_counts=[int(count) for count in detected_counts],
    )


def _params_key(params: Dict[str, float | int]) -> Tuple[Tuple[str, float | int], ...]:
    return tuple(sorted(params.items()))


def _select_parameter_bank(candidates: List[ArucoCandidate], bank_size: int) -> List[ArucoCandidate]:
    if not candidates:
        return []

    target_counts = [
        max(candidate.frame_detected_counts[index] for candidate in candidates)
        for index in range(len(candidates[0].frame_detected_counts))
    ]
    selected = []
    selected_keys = set()
    current_counts = [0 for _ in target_counts]

    ranked = sorted(
        candidates,
        key=lambda item: (item.score, item.mean_detected, item.coverage_frames, -item.mean_rejected, item.eval_fps),
        reverse=True,
    )

    while len(selected) < max(1, bank_size):
        best_candidate = None
        best_gain = float("-inf")
        for candidate in ranked:
            key = _params_key(candidate.params)
            if key in selected_keys:
                continue
            gain = 0.0
            for index, count in enumerate(candidate.frame_detected_counts):
                target = target_counts[index]
                if target <= 0:
                    target = max(count, 1)
                gain += max(0, min(count, target) - min(current_counts[index], target))
            gain += 0.01 * candidate.score
            if gain > best_gain:
                best_gain = gain
                best_candidate = candidate

        if best_candidate is None:
            break
        selected.append(best_candidate)
        selected_keys.add(_params_key(best_candidate.params))
        current_counts = [
            max(current_counts[index], count)
            for index, count in enumerate(best_candidate.frame_detected_counts)
        ]

        if best_gain <= 0 and len(selected) >= 1:
            break

    return selected


def optimize_aruco_parameter_bank(
    video_path: str | Path,
    output_dir: str | Path,
    dictionary_name: str = "4x4_100",
    profile: str = "daily",
    sample_frames: int = 12,
    max_combinations: int = 750,
    bank_size: int = 5,
    expected_tags: Optional[float] = None,
    allowed_tag_ids: Optional[Iterable[int]] = None,
    sweep_overrides: Optional[Dict[str, Sequence[float | int]]] = None,
    progress_callback: Optional[ProgressCallback] = None,
    stop_requested: Optional[StopCallback] = None,
) -> ArucoOptimizationResult:
    resolved_video = Path(video_path).expanduser().resolve()
    dictionary_key = normalize_dictionary_name(dictionary_name)
    frames, frame_indices, _total_frames = sample_video_frames(resolved_video, max(1, int(sample_frames)))
    grid = build_parameter_grid(profile, sweep_overrides=sweep_overrides, max_combinations=max_combinations)
    allowed_ids = {int(item) for item in allowed_tag_ids} if allowed_tag_ids else None

    evaluated = []
    total = len(grid)
    for index, params in enumerate(grid, start=1):
        if stop_requested and stop_requested():
            break
        candidate = _evaluate_candidate(
            params=params,
            frames=frames,
            dictionary_key=dictionary_key,
            allowed_tag_ids=allowed_ids,
            expected_tags=expected_tags,
        )
        evaluated.append(candidate)
        if progress_callback and (index == 1 or index == total or index % 25 == 0):
            progress_callback(index, total, asdict(candidate))

    if not evaluated:
        raise RuntimeError("ArUco optimization stopped before any candidates completed.")

    evaluated.sort(
        key=lambda item: (item.score, item.mean_detected, item.coverage_frames, -item.mean_rejected, item.eval_fps),
        reverse=True,
    )
    for rank, candidate in enumerate(evaluated, start=1):
        candidate.rank = rank

    selected_candidates = _select_parameter_bank(evaluated, bank_size=bank_size)
    run_dir = Path(output_dir).expanduser().resolve() / resolved_video.stem / "aruco_parameter_bank"
    run_dir.mkdir(parents=True, exist_ok=True)

    candidates_csv = run_dir / "candidate_scores.csv"
    with candidates_csv.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "rank",
                "score",
                "mean_detected",
                "mean_decoded",
                "mean_filtered",
                "mean_rejected",
                "std_detected",
                "coverage_frames",
                "eval_fps",
                "runtime_seconds",
                "frame_detected_counts",
                "params_json",
            ],
        )
        writer.writeheader()
        for candidate in evaluated:
            writer.writerow(
                {
                    "rank": candidate.rank,
                    "score": f"{candidate.score:.6f}",
                    "mean_detected": f"{candidate.mean_detected:.6f}",
                    "mean_decoded": f"{candidate.mean_decoded:.6f}",
                    "mean_filtered": f"{candidate.mean_filtered:.6f}",
                    "mean_rejected": f"{candidate.mean_rejected:.6f}",
                    "std_detected": f"{candidate.std_detected:.6f}",
                    "coverage_frames": candidate.coverage_frames,
                    "eval_fps": f"{candidate.eval_fps:.6f}",
                    "runtime_seconds": f"{candidate.runtime_seconds:.6f}",
                    "frame_detected_counts": json.dumps(candidate.frame_detected_counts),
                    "params_json": json.dumps(candidate.params, sort_keys=True),
                }
            )

    result = ArucoOptimizationResult(
        created_at=datetime.now().isoformat(timespec="seconds"),
        video_path=str(resolved_video),
        dictionary=dictionary_key,
        profile=str(profile).lower(),
        sampled_frame_indices=frame_indices,
        parameter_combinations_total=total,
        combinations_evaluated=len(evaluated),
        output_dir=str(run_dir),
        summary_json_path=str(run_dir / "optimization_summary.json"),
        candidates_csv_path=str(candidates_csv),
        parameter_bank=[dict(candidate.params) for candidate in selected_candidates],
        selected_candidates=selected_candidates,
        top_candidates=evaluated[: min(20, len(evaluated))],
    )

    with Path(result.summary_json_path).open("w") as f:
        json.dump(result.to_dict(), f, indent=2)

    return result
