#!/usr/bin/env python3
"""Refresh hive-derived metrics for an existing BumbleBox inference run.

This tool intentionally treats bee detections, tracks, ArUco identities, and
interaction events as fixed. It replays the videos in manifest order, reruns
chamber/hive/pollen segmentation, rebuilds the temporal hive prior, and writes a
new versioned bee detections CSV with only hive/pollen-derived columns updated.
"""

import argparse
import csv
import gc
import json
import re
import sys
import time
from collections import defaultdict
from contextlib import nullcontext
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np
from scipy.spatial import cKDTree
from ultralytics import YOLO

try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    torch = None
    TORCH_AVAILABLE = False

from core.batch_video_processor import BatchVideoProcessor
from core.temporal_hive_prior import TemporalHivePrior
from core.temporal_hive_visualization import TemporalHiveOverlayWriter
from core.video_inference_exporter import VideoInferenceExporter


PRIOR_CHECKPOINT_FILENAME = "hive_refresh_temporal_prior_checkpoint.npz"

VIDEO_TIME_RE = re.compile(
    r"(?P<year>\d{4})-(?P<month>\d{2})-(?P<day>\d{2})_"
    r"(?P<hour>\d{2})_(?P<minute>\d{2})_(?P<second>\d{2})"
)


@dataclass
class RefreshBee:
    """Minimal detection object compatible with BatchVideoProcessor helpers."""

    row: Dict[str, str]
    bbox: np.ndarray
    mask: Optional[np.ndarray]
    confidence: float
    instance_id: Optional[int]
    chamber_id: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Rerun hive/chamber/pollen/temporal-prior metrics for existing "
            "BumbleBox bee detections without retracking bees or ArUco IDs."
        )
    )
    parser.add_argument(
        "--source-output-folder",
        type=Path,
        required=True,
        help="Existing inference output folder containing bee_detections.csv.",
    )
    parser.add_argument(
        "--file-list",
        type=Path,
        required=True,
        help="Ordered text file of video paths to replay.",
    )
    parser.add_argument(
        "--output-folder",
        type=Path,
        required=True,
        help="Folder for refreshed, versioned outputs.",
    )
    parser.add_argument("--hive-model", type=Path, required=True)
    parser.add_argument("--chamber-model", type=Path, required=True)
    pollen = parser.add_mutually_exclusive_group(required=True)
    pollen.add_argument("--pollen-model", type=Path)
    pollen.add_argument("--no-pollen-model", action="store_true")
    parser.add_argument(
        "--keep-pollen-in-hive",
        action="store_true",
        help="Keep pollen pixels in hive masks. Default removes pollen from hive masks.",
    )
    parser.add_argument("--confidence", type=float, default=0.5)
    parser.add_argument("--nms-iou", type=float, default=0.45)
    parser.add_argument("--pixel-size-mm", type=float, default=0.0666)
    parser.add_argument("--temporal-window-hours", type=float, default=8.0)
    parser.add_argument('--stabilize-temporal-hive', action='store_true',
                        help='Stabilize chamber placement for temporal evidence, scores and overlays.')
    parser.add_argument('--temporal-hive-scoring', choices=['updated', 'prior'], default='updated',
                        help='Score contact on history plus current evidence (default), or the past-only map.')
    parser.add_argument(
        "--temporal-resolution",
        default="800x1500",
        help="Temporal hive prior resolution as WIDTHxHEIGHT, e.g. 800x1500.",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument('--save-temporal-overlays', action='store_true',
                        help='Cache every refreshed frame for model-free temporal hive video rendering.')
    parser.add_argument('--temporal-overlay-timing', choices=['scored', 'prior', 'updated'], default='scored',
                        help='Default follows contact scoring; optionally cache a different map for comparison.')
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Optional maximum number of listed videos to process, for testing.",
    )
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def parse_resolution(raw: str) -> Tuple[int, int]:
    text = str(raw).lower().replace(",", "x").strip()
    parts = [part.strip() for part in text.split("x") if part.strip()]
    if len(parts) != 2:
        raise ValueError(f"Expected temporal resolution WIDTHxHEIGHT, got {raw!r}")
    width, height = int(parts[0]), int(parts[1])
    if width <= 0 or height <= 0:
        raise ValueError(f"Temporal resolution must be positive, got {raw!r}")
    return width, height


def load_video_list(path: Path) -> List[Path]:
    videos = []
    with open(path, newline="") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            videos.append(Path(stripped))
    return videos


def video_id_for_path(path: Path) -> str:
    return path.stem


def context_id_for_path(path: Path) -> str:
    parts = [part for part in path.parts if part.startswith("MCs-")]
    if parts:
        return parts[-1]
    return path.parent.name or "default"


def timestamp_seconds_from_filename(path: Path) -> Optional[float]:
    match = VIDEO_TIME_RE.search(path.name)
    if not match:
        return None
    pieces = {key: int(value) for key, value in match.groupdict().items()}
    dt = datetime(
        pieces["year"],
        pieces["month"],
        pieces["day"],
        pieces["hour"],
        pieces["minute"],
        pieces["second"],
    )
    return dt.timestamp()


def int_value(value, default: int = 0) -> int:
    try:
        if value is None or value == "":
            return default
        return int(float(value))
    except (TypeError, ValueError):
        return default


def float_value(value, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def optional_float(value) -> Optional[float]:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def format_optional_float(value: Optional[float], decimals: int = 4) -> str:
    if value is None:
        return ""
    return f"{float(value):.{decimals}f}"


def format_optional_int(value: Optional[int]) -> str:
    if value is None:
        return ""
    return str(int(value))


def format_optional_bool(value: Optional[bool]) -> str:
    if value is None:
        return ""
    return "True" if value else "False"


def polygon_to_mask(polygon_text: str, frame_shape: Tuple[int, int]) -> Optional[np.ndarray]:
    text = (polygon_text or "").strip()
    if not text:
        return None

    numbers = []
    for token in re.split(r"[\s,]+", text):
        if not token:
            continue
        try:
            numbers.append(float(token))
        except ValueError:
            return None

    if len(numbers) < 6 or len(numbers) % 2:
        return None

    points = np.asarray(numbers, dtype=np.float32).reshape(-1, 2)
    points[:, 0] = np.clip(points[:, 0], 0, frame_shape[1] - 1)
    points[:, 1] = np.clip(points[:, 1], 0, frame_shape[0] - 1)
    points = np.rint(points).astype(np.int32)

    mask = np.zeros(frame_shape, dtype=np.uint8)
    cv2.fillPoly(mask, [points], 255)
    return mask if np.any(mask > 0) else None


def row_to_bee(row: Dict[str, str], frame_shape: Tuple[int, int]) -> RefreshBee:
    x = float_value(row.get("bbox_x"))
    y = float_value(row.get("bbox_y"))
    w = float_value(row.get("bbox_width"))
    h = float_value(row.get("bbox_height"))
    bbox = np.asarray([x, y, x + w, y + h], dtype=np.float32)

    mask = polygon_to_mask(row.get("pred_polygon", ""), frame_shape)
    if mask is None:
        mask = np.zeros(frame_shape, dtype=np.uint8)
        x1 = max(0, min(frame_shape[1], int(round(bbox[0]))))
        y1 = max(0, min(frame_shape[0], int(round(bbox[1]))))
        x2 = max(0, min(frame_shape[1], int(round(bbox[2]))))
        y2 = max(0, min(frame_shape[0], int(round(bbox[3]))))
        if x2 > x1 and y2 > y1:
            mask[y1:y2, x1:x2] = 255
        else:
            mask = None

    return RefreshBee(
        row=row,
        bbox=bbox,
        mask=mask,
        confidence=float_value(row.get("confidence")),
        instance_id=int_value(row.get("bee_id"), default=-1),
        chamber_id=int_value(row.get("chamber_id"), default=0),
    )


def build_hive_kdtrees(hive_masks_by_chamber: Dict[int, Optional[np.ndarray]]) -> Dict[int, Optional[cKDTree]]:
    kdtrees = {}
    for chamber_id, hive_mask in hive_masks_by_chamber.items():
        if hive_mask is None or not np.any(hive_mask > 0):
            kdtrees[chamber_id] = None
            continue
        coords_yx = np.argwhere(hive_mask > 0)
        kdtrees[chamber_id] = cKDTree(coords_yx[:, [1, 0]])
    return kdtrees


def build_pollen_indexes(
    pollen_by_chamber: Dict[int, List[Dict]],
    frame_shape: Tuple[int, int],
) -> Tuple[Dict[int, Optional[cKDTree]], Dict[int, Optional[np.ndarray]]]:
    kdtrees = {}
    masks = {}
    for chamber_id, pollen_balls in pollen_by_chamber.items():
        coords = []
        combined_mask = None
        for pollen in pollen_balls:
            mask = pollen.get("mask")
            if mask is None or mask.shape[:2] != frame_shape or not np.any(mask > 0):
                continue
            binary_mask = (mask > 0).astype(np.uint8, copy=False)
            combined_mask = binary_mask.copy() if combined_mask is None else np.maximum(combined_mask, binary_mask)
            coords_yx = np.argwhere(mask > 0)
            if len(coords_yx):
                coords.append(coords_yx[:, [1, 0]])
        masks[chamber_id] = combined_mask
        kdtrees[chamber_id] = cKDTree(np.vstack(coords)) if coords else None
    return kdtrees, masks


def pollen_overlap_metrics(
    bee: RefreshBee,
    pollen_mask: Optional[np.ndarray],
    pollen_model_available: bool,
) -> Dict[str, Optional[float]]:
    empty = {
        "on_pollen_ball": None if pollen_model_available else None,
        "pollen_overlap_pixels": None if pollen_model_available else None,
        "pollen_overlap_fraction": None if pollen_model_available else None,
    }
    if not pollen_model_available:
        return empty
    if bee.mask is None or pollen_mask is None or bee.mask.shape[:2] != pollen_mask.shape[:2]:
        return {
            "on_pollen_ball": False,
            "pollen_overlap_pixels": 0,
            "pollen_overlap_fraction": 0.0,
        }

    bee_pixels = bee.mask > 0
    bee_pixel_count = int(bee_pixels.sum())
    if bee_pixel_count <= 0:
        return {
            "on_pollen_ball": False,
            "pollen_overlap_pixels": 0,
            "pollen_overlap_fraction": 0.0,
        }

    overlap_pixels = int(np.logical_and(bee_pixels, pollen_mask > 0).sum())
    return {
        "on_pollen_ball": overlap_pixels > 0,
        "pollen_overlap_pixels": overlap_pixels,
        "pollen_overlap_fraction": float(overlap_pixels / bee_pixel_count),
    }


def load_source_bee_rows(
    source_csv: Path,
    target_video_ids: Iterable[str],
) -> Tuple[List[str], Dict[str, Dict[int, List[Dict[str, str]]]], int]:
    target_set = set(target_video_ids)
    rows_by_video_frame: Dict[str, Dict[int, List[Dict[str, str]]]] = defaultdict(lambda: defaultdict(list))
    row_count = 0

    with open(source_csv, newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        for row in reader:
            video_id = row.get("video_id", "")
            if video_id not in target_set:
                continue
            frame_number = int_value(row.get("frame_number"), default=0)
            rows_by_video_frame[video_id][frame_number].append(row)
            row_count += 1

    return fieldnames, rows_by_video_frame, row_count


def read_completed_status(status_path: Path) -> set:
    if not status_path.exists():
        return set()
    completed = set()
    with open(status_path, newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if row.get("status") == "complete":
                completed.add(row.get("video_path", ""))
    return completed


def completed_prefix_count(videos: List[Path], completed_paths: set) -> int:
    count = 0
    for video in videos:
        if str(video) not in completed_paths:
            break
        count += 1
    return count


def initialize_temporal_prior(
    args: argparse.Namespace,
    videos: List[Path],
    resolution: Tuple[int, int],
    completed_prefix: int,
) -> TemporalHivePrior:
    checkpoint_path = args.output_folder / PRIOR_CHECKPOINT_FILENAME
    window_seconds = float(args.temporal_window_hours) * 3600.0

    if args.resume and completed_prefix > 0:
        if not checkpoint_path.exists():
            raise RuntimeError(
                f"Resume found {completed_prefix} completed leading video(s), but no "
                f"{PRIOR_CHECKPOINT_FILENAME}. Restart without --resume, or rerun into a fresh output folder."
            )

        prior, metadata = TemporalHivePrior.load_checkpoint(checkpoint_path)
        checkpoint_index = int(metadata.get("video_index", 0))
        checkpoint_video = str(metadata.get("video_path", ""))
        expected_video = str(videos[completed_prefix - 1])
        settings = metadata.get("settings", {})
        checkpoint_resolution = tuple(settings.get("resolution", []))
        checkpoint_window = float(settings.get("window_seconds", 0.0))

        if checkpoint_index != completed_prefix or checkpoint_video != expected_video:
            raise RuntimeError(
                "Temporal-prior checkpoint does not match the completed video prefix. "
                "Use a fresh refresh output folder for this run."
            )
        if (checkpoint_resolution != tuple(resolution) or abs(checkpoint_window - window_seconds) > 1e-6
                or prior.stabilize_chambers != args.stabilize_temporal_hive
                or prior.scoring_mode != args.temporal_hive_scoring):
            raise RuntimeError(
                "Temporal-prior checkpoint settings do not match this refresh command. "
                "Use the same temporal window/resolution/stabilization/scoring mode, or start a fresh output folder."
            )
        return prior

    return TemporalHivePrior(
        window_seconds=window_seconds,
        resolution=resolution,
        stabilize_chambers=args.stabilize_temporal_hive,
        scoring_mode=args.temporal_hive_scoring,
    )


def save_temporal_prior_checkpoint(
    output_folder: Path,
    temporal_prior: TemporalHivePrior,
    video_path: Path,
    video_index: int,
    total_videos: int,
    args: argparse.Namespace,
    resolution: Tuple[int, int],
):
    metadata = {
        "checkpoint_version": 1,
        "video_index": int(video_index),
        "total_videos": int(total_videos),
        "video_path": str(video_path),
        "video_id": video_id_for_path(video_path),
        "saved_at": datetime.now().isoformat(timespec="seconds"),
        "settings": {
            "window_seconds": float(args.temporal_window_hours) * 3600.0,
            "resolution": [int(resolution[0]), int(resolution[1])],
            "exclude_pollen_from_hive": not args.keep_pollen_in_hive,
        },
    }
    temporal_prior.save_checkpoint(output_folder / PRIOR_CHECKPOINT_FILENAME, metadata=metadata)


def ensure_fresh_outputs(output_folder: Path, resume: bool):
    output_folder.mkdir(parents=True, exist_ok=True)
    if resume:
        return

    for filename in [
        "bee_detections_hive_refreshed.csv",
        "hive_detections_hive_refreshed.csv",
        "pollen_detections_hive_refreshed.csv",
        "temporal_hive_priors_hive_refreshed.csv",
        "hive_refresh_status.csv",
    ]:
        path = output_folder / filename
        if path.exists():
            path.unlink()


def write_header_if_needed(path: Path, fieldnames: List[str], append: bool):
    if append and path.exists() and path.stat().st_size > 0:
        return
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()


def append_rows(path: Path, fieldnames: List[str], rows: List[Dict]):
    if not rows:
        return
    with open(path, "a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writerows(rows)


def append_status(output_folder: Path, row: Dict[str, str]):
    path = output_folder / "hive_refresh_status.csv"
    fieldnames = [
        "video_path",
        "video_id",
        "status",
        "frame_count",
        "refreshed_bee_rows",
        "message",
        "elapsed_seconds",
        "completed_at",
    ]
    write_header_if_needed(path, fieldnames, append=True)
    append_rows(path, fieldnames, [row])


def write_config(args: argparse.Namespace, videos: List[Path], resolution: Tuple[int, int]):
    config = {
        "source_output_folder": str(args.source_output_folder),
        "file_list": str(args.file_list),
        "output_folder": str(args.output_folder),
        "hive_model": str(args.hive_model),
        "chamber_model": str(args.chamber_model),
        "pollen_model": "" if args.no_pollen_model else str(args.pollen_model),
        "exclude_pollen_from_hive": not args.keep_pollen_in_hive,
        "confidence": args.confidence,
        "nms_iou": args.nms_iou,
        "pixel_size_mm": args.pixel_size_mm,
        "temporal_window_hours": args.temporal_window_hours,
        "temporal_resolution": {"width": resolution[0], "height": resolution[1]},
        "save_temporal_overlays": args.save_temporal_overlays,
        "temporal_overlay_timing": args.temporal_overlay_timing,
        "stabilize_temporal_hive": args.stabilize_temporal_hive,
        "temporal_hive_scoring": args.temporal_hive_scoring,
        "video_count": len(videos),
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    args.output_folder.mkdir(parents=True, exist_ok=True)
    with open(args.output_folder / "hive_refresh_config.json", "w") as handle:
        json.dump(config, handle, indent=2)


def run_hive_model(processor: BatchVideoProcessor, hive_model, frame: np.ndarray):
    if hive_model is None:
        return None
    if TORCH_AVAILABLE:
        with torch.inference_mode():
            results = hive_model(
                frame,
                conf=processor.confidence_threshold,
                iou=processor.nms_iou_threshold,
                retina_masks=processor.high_quality_masks,
                half=torch.cuda.is_available(),
                verbose=False,
            )
    else:
        results = hive_model(
            frame,
            conf=processor.confidence_threshold,
            iou=processor.nms_iou_threshold,
            retina_masks=processor.high_quality_masks,
            verbose=False,
        )
    processor._sync_cuda()
    return results


def load_models(args: argparse.Namespace):
    print("Loading refresh models...")
    hive_model = YOLO(str(args.hive_model))
    print(f"✓ Loaded hive model: {args.hive_model}")
    chamber_model = YOLO(str(args.chamber_model)) if args.chamber_model else None
    if chamber_model is not None:
        print(f"✓ Loaded chamber model: {args.chamber_model}")
    pollen_model = None
    if not args.no_pollen_model and args.pollen_model:
        pollen_model = YOLO(str(args.pollen_model))
        print(f"✓ Loaded pollen model: {args.pollen_model}")
    return hive_model, chamber_model, pollen_model


def write_temporal_overlay_frame(writer, prior, context_id, chambers, frame_number, frame_shape, frame_time):
    snapshots = [
        prior.visualization_snapshot(context_id, chamber_id, chamber, frame_shape, frame_time)
        for chamber_id, chamber in sorted(chambers.items())
    ]
    writer.write(frame_number, frame_shape, snapshots)


def refresh_video(
    video_path: Path,
    rows_by_frame: Dict[int, List[Dict[str, str]]],
    bee_fieldnames: List[str],
    output_folder: Path,
    hive_model,
    chamber_model,
    pollen_model,
    temporal_prior: TemporalHivePrior,
    args: argparse.Namespace,
    overlay_writer=None,
) -> Tuple[int, int]:
    video_id = video_id_for_path(video_path)
    context_id = context_id_for_path(video_path)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    start_time_seconds = timestamp_seconds_from_filename(video_path)

    processor = BatchVideoProcessor(
        video_path=video_path,
        video_id=video_id,
        bee_model=None,
        hive_model=hive_model,
        chamber_model=chamber_model,
        pollen_model=pollen_model,
        tracker=None,
        confidence_threshold=args.confidence,
        nms_iou_threshold=args.nms_iou,
        enable_aruco=False,
        distance_method="centroid",
        bee_model_type="segmentation",
        compute_spatial_metrics=True,
        verbose_output=args.verbose,
        high_quality_masks=False,
        temporal_hive_prior=temporal_prior,
        temporal_hive_context_id=context_id,
        video_start_time_seconds=start_time_seconds,
        pixel_size_mm=args.pixel_size_mm,
        exclude_pollen_from_hive=not args.keep_pollen_in_hive,
        prior_only=True,
    )
    processor.video_fps = fps if fps > 0 else None
    if overlay_writer is not None:
        overlay_writer.metadata['fps'] = processor.video_fps

    bee_output_rows = []
    hive_rows = []
    pollen_rows = []
    frame_number = 0
    refreshed_bee_rows = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_number += 1
        frame_shape = frame.shape[:2]
        frame_time_seconds = processor._frame_time_seconds(frame_number)

        source_rows = rows_by_frame.get(frame_number, [])
        bees = [row_to_bee(row, frame_shape) for row in source_rows]

        chambers_detected = processor._detect_chambers(frame)
        chambers_detected = processor._prepare_temporal_chambers(chambers_detected, frame_shape)
        hive_results = run_hive_model(processor, hive_model, frame)
        hive_result = hive_results[0] if hive_results else None
        hive_masks_by_chamber = processor._extract_hive_masks(hive_result, chambers_detected)

        pollen_balls = processor._detect_pollen_balls(frame)
        pollen_by_chamber = processor._assign_pollen_to_chambers(pollen_balls, chambers_detected)
        if not args.keep_pollen_in_hive:
            hive_masks_by_chamber = processor._exclude_pollen_from_hive_masks(
                hive_masks_by_chamber,
                pollen_by_chamber,
            )

        hive_kdtrees = build_hive_kdtrees(hive_masks_by_chamber)
        pollen_kdtrees, pollen_masks = build_pollen_indexes(pollen_by_chamber, frame_shape)

        if overlay_writer is not None and overlay_writer.metadata['timing'] == 'before_current_frame_update':
            write_temporal_overlay_frame(overlay_writer, temporal_prior, context_id, chambers_detected,
                                         frame_number, frame_shape, frame_time_seconds)

        if temporal_prior.scoring_mode == 'updated':
            processor._update_temporal_hive_prior(
                chambers_detected, hive_masks_by_chamber, bees, frame_shape, frame_time_seconds,
            )

        for chamber_id, hive_mask in hive_masks_by_chamber.items():
            hive_pixels = int(np.sum(hive_mask > 0)) if hive_mask is not None else 0
            centroid_x = ""
            centroid_y = ""
            if hive_mask is not None and hive_pixels > 0:
                ys, xs = np.where(hive_mask > 0)
                centroid_x = format_optional_float(float(np.mean(xs)), decimals=2)
                centroid_y = format_optional_float(float(np.mean(ys)), decimals=2)
            hive_rows.append({
                "video_id": video_id,
                "chamber_id": chamber_id,
                "frame_number": frame_number,
                "hive_pixels": hive_pixels,
                "centroid_x": centroid_x,
                "centroid_y": centroid_y,
            })

        chamber_ids = set(chambers_detected.keys()) | set(pollen_by_chamber.keys())
        for chamber_id in sorted(chamber_ids):
            balls = pollen_by_chamber.get(chamber_id, [])
            pollen_pixels = int(sum(int(ball.get("pixels") or 0) for ball in balls))
            pollen_rows.append({
                "video_id": video_id,
                "chamber_id": chamber_id,
                "frame_number": frame_number,
                "pollen_count": len(balls),
                "pollen_pixels": pollen_pixels,
                "pollen_area_mm2": format_optional_float(
                    processor._area_pixels_to_mm2(pollen_pixels),
                    decimals=4,
                ),
            })

        for bee in bees:
            row = dict(bee.row)
            chamber_id = bee.chamber_id
            if chamber_id not in chambers_detected:
                chamber_id = processor._assign_bee_to_chamber(bee, chambers_detected)
                row["chamber_id"] = str(chamber_id)

            centroid = processor._get_centroid(bee.bbox, bee.mask)
            temporal_overlap = processor._query_temporal_hive_overlap(
                chamber_id,
                chambers_detected,
                bee,
                frame_shape,
                frame_time_seconds,
            )

            hive_kdtree = hive_kdtrees.get(chamber_id)
            distance_to_hive = (
                processor._calculate_distance_to_hive_fast(centroid, hive_kdtree)
                if hive_kdtree is not None
                else None
            )

            pollen_kdtree = pollen_kdtrees.get(chamber_id)
            distance_to_pollen = (
                processor._calculate_distance_to_hive_fast(centroid, pollen_kdtree)
                if pollen_kdtree is not None
                else None
            )
            pollen_overlap = pollen_overlap_metrics(
                bee,
                pollen_masks.get(chamber_id),
                pollen_model_available=pollen_model is not None,
            )

            row["distance_to_hive_pixels"] = format_optional_float(distance_to_hive, decimals=2)
            row["distance_to_hive_mm"] = format_optional_float(
                processor._pixels_to_mm(distance_to_hive),
                decimals=4,
            )
            row["distance_to_nearest_pollen_pixels"] = format_optional_float(distance_to_pollen, decimals=2)
            row["distance_to_nearest_pollen_mm"] = format_optional_float(
                processor._pixels_to_mm(distance_to_pollen),
                decimals=4,
            )
            row["pollen_count_in_chamber"] = format_optional_int(len(pollen_by_chamber.get(chamber_id, [])))
            row["on_pollen_ball"] = format_optional_bool(pollen_overlap["on_pollen_ball"])
            row["pollen_overlap_pixels"] = format_optional_int(pollen_overlap["pollen_overlap_pixels"])
            row["pollen_overlap_fraction"] = format_optional_float(
                optional_float(pollen_overlap["pollen_overlap_fraction"]),
                decimals=4,
            )
            row["on_temporal_hive"] = format_optional_bool(temporal_overlap.on_hive)
            row["temporal_hive_overlap_fraction"] = format_optional_float(
                temporal_overlap.overlap_fraction,
                decimals=4,
            )
            row["temporal_hive_mean_probability"] = format_optional_float(
                temporal_overlap.mean_probability,
                decimals=4,
            )
            row["temporal_hive_known_fraction"] = format_optional_float(
                temporal_overlap.known_fraction,
                decimals=4,
            )
            row["temporal_hive_overlap_pixels_norm"] = format_optional_int(temporal_overlap.overlap_pixels)
            row["temporal_hive_known_pixels_norm"] = format_optional_int(temporal_overlap.known_pixels)
            row["temporal_hive_bee_pixels_norm"] = format_optional_int(temporal_overlap.bee_pixels)
            row["temporal_hive_prior_weight_mean"] = format_optional_float(
                temporal_overlap.mean_prior_weight,
                decimals=4,
            )
            row["temporal_hive_prior_weight_sum"] = format_optional_float(
                temporal_overlap.sum_prior_weight,
                decimals=2,
            )
            bee_output_rows.append(row)
            refreshed_bee_rows += 1

        if temporal_prior.scoring_mode == 'prior':
            processor._update_temporal_hive_prior(
                chambers_detected, hive_masks_by_chamber, bees, frame_shape, frame_time_seconds,
            )

        if overlay_writer is not None and overlay_writer.metadata['timing'] == 'after_current_frame_update':
            write_temporal_overlay_frame(overlay_writer, temporal_prior, context_id, chambers_detected,
                                         frame_number, frame_shape, frame_time_seconds)

        if hive_results is not None:
            del hive_results
        del frame
        if frame_number % 25 == 0:
            gc.collect()
            if TORCH_AVAILABLE and torch.cuda.is_available():
                torch.cuda.empty_cache()

    cap.release()

    append_rows(output_folder / "bee_detections_hive_refreshed.csv", bee_fieldnames, bee_output_rows)
    append_rows(
        output_folder / "hive_detections_hive_refreshed.csv",
        ["video_id", "chamber_id", "frame_number", "hive_pixels", "centroid_x", "centroid_y"],
        hive_rows,
    )
    append_rows(
        output_folder / "pollen_detections_hive_refreshed.csv",
        ["video_id", "chamber_id", "frame_number", "pollen_count", "pollen_pixels", "pollen_area_mm2"],
        pollen_rows,
    )

    return frame_number, refreshed_bee_rows


def write_temporal_prior_summary(output_folder: Path, temporal_prior: TemporalHivePrior):
    summaries = temporal_prior.summaries()
    exporter = VideoInferenceExporter(output_folder)
    path = output_folder / "temporal_hive_priors_hive_refreshed.csv"
    fieldnames = [
        "context_id",
        "chamber_id",
        "prior_hive_pixels_norm",
        "centroid_x_norm",
        "centroid_y_norm",
        "mean_prior_weight",
        "max_prior_weight",
        "observation_count",
        "prior_polygon_norm",
        "resolution_width",
        "resolution_height",
    ]
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in summaries:
            writer.writerow({
                "context_id": row.get("context_id", ""),
                "chamber_id": row.get("chamber_id", ""),
                "prior_hive_pixels_norm": row.get("prior_hive_pixels_norm", 0),
                "centroid_x_norm": exporter._format_optional_float(row.get("centroid_x_norm"), decimals=6),
                "centroid_y_norm": exporter._format_optional_float(row.get("centroid_y_norm"), decimals=6),
                "mean_prior_weight": exporter._format_optional_float(row.get("mean_prior_weight"), decimals=2),
                "max_prior_weight": exporter._format_optional_float(row.get("max_prior_weight"), decimals=2),
                "observation_count": row.get("observation_count", 0),
                "prior_polygon_norm": row.get("prior_polygon_norm", ""),
                "resolution_width": row.get("resolution_width", ""),
                "resolution_height": row.get("resolution_height", ""),
            })


def main() -> int:
    args = parse_args()
    try:
        resolution = parse_resolution(args.temporal_resolution)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    videos = load_video_list(args.file_list)
    if args.limit and args.limit > 0:
        videos = videos[:args.limit]
    if not videos:
        print("ERROR: file list contained no videos", file=sys.stderr)
        return 2

    source_csv = args.source_output_folder / "bee_detections.csv"
    if not source_csv.exists():
        print(f"ERROR: missing source bee detections CSV: {source_csv}", file=sys.stderr)
        return 2

    ensure_fresh_outputs(args.output_folder, resume=args.resume)

    status_path = args.output_folder / "hive_refresh_status.csv"
    completed_paths = read_completed_status(status_path) if args.resume else set()
    completed_prefix = completed_prefix_count(videos, completed_paths) if args.resume else 0
    videos_to_process = videos[completed_prefix:]
    non_prefix_completed = len(completed_paths) - completed_prefix

    try:
        temporal_prior = initialize_temporal_prior(args, videos, resolution, completed_prefix)
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    write_config(args, videos, resolution)

    target_video_ids = [video_id_for_path(video) for video in videos_to_process]
    print(f"Loading source bee detections for {len(videos_to_process)}/{len(videos)} video(s)...")
    bee_fieldnames, rows_by_video_frame, row_count = load_source_bee_rows(source_csv, target_video_ids)
    if not bee_fieldnames:
        print(f"ERROR: could not read fieldnames from {source_csv}", file=sys.stderr)
        return 2

    required_columns = [
        "distance_to_hive_pixels",
        "distance_to_hive_mm",
        "distance_to_nearest_pollen_pixels",
        "distance_to_nearest_pollen_mm",
        "pollen_count_in_chamber",
        "on_pollen_ball",
        "pollen_overlap_pixels",
        "pollen_overlap_fraction",
        "on_temporal_hive",
        "temporal_hive_overlap_fraction",
        "temporal_hive_mean_probability",
        "temporal_hive_known_fraction",
        "temporal_hive_overlap_pixels_norm",
        "temporal_hive_known_pixels_norm",
        "temporal_hive_bee_pixels_norm",
        "temporal_hive_prior_weight_mean",
        "temporal_hive_prior_weight_sum",
    ]
    for column in required_columns:
        if column not in bee_fieldnames:
            bee_fieldnames.append(column)

    write_header_if_needed(args.output_folder / "bee_detections_hive_refreshed.csv", bee_fieldnames, append=args.resume)
    write_header_if_needed(
        args.output_folder / "hive_detections_hive_refreshed.csv",
        ["video_id", "chamber_id", "frame_number", "hive_pixels", "centroid_x", "centroid_y"],
        append=args.resume,
    )
    write_header_if_needed(
        args.output_folder / "pollen_detections_hive_refreshed.csv",
        ["video_id", "chamber_id", "frame_number", "pollen_count", "pollen_pixels", "pollen_area_mm2"],
        append=args.resume,
    )

    print(f"✓ Loaded {row_count} bee rows to refresh")
    print(f"Temporal hive prior: {args.temporal_window_hours:.1f}h window, {resolution[0]}x{resolution[1]} map")
    print(f"Temporal hive contact scoring: {temporal_prior.scoring_mode}")
    print(f"Pollen excluded from hive masks: {'yes' if not args.keep_pollen_in_hive else 'no'}")

    hive_model, chamber_model, pollen_model = load_models(args)

    processed = 0
    skipped = completed_prefix
    if skipped:
        print(f"Resume enabled: skipping {skipped} already refreshed leading video(s)")
    if non_prefix_completed > 0:
        print(
            f"WARNING: ignoring {non_prefix_completed} non-prefix completed status row(s); "
            "temporal prior refresh must resume in manifest order."
        )

    total = len(videos)
    remaining = len(videos_to_process)
    for idx, video_path in enumerate(videos_to_process, start=1):
        video_id = video_id_for_path(video_path)
        global_idx = videos.index(video_path) + 1
        rows_by_frame = rows_by_video_frame.get(video_id, {})
        print(f"[{global_idx}/{total}] Refreshing {video_path.name} ({sum(len(v) for v in rows_by_frame.values())} bee rows)")
        start = time.perf_counter()
        try:
            overlay_timing = (temporal_prior.scoring_mode if args.temporal_overlay_timing == 'scored'
                              else args.temporal_overlay_timing)
            updated_overlay = overlay_timing == 'updated'
            cache_name = f'{video_id}_updated.zip' if updated_overlay else f'{video_id}.zip'
            overlay_context = (
                TemporalHiveOverlayWriter(
                    args.output_folder / 'temporal_hive_overlays' / cache_name,
                    video_path, context_id_for_path(video_path),
                    provenance='reconstructed_from_saved_bee_detections',
                    prior=temporal_prior,
                    timing=('after_current_frame_update' if updated_overlay else 'before_current_frame_update'),
                ) if args.save_temporal_overlays else nullcontext(None)
            )
            with overlay_context as overlay_writer:
                frame_count, refreshed_rows = refresh_video(
                    video_path=video_path,
                    rows_by_frame=rows_by_frame,
                    bee_fieldnames=bee_fieldnames,
                    output_folder=args.output_folder,
                    hive_model=hive_model,
                    chamber_model=chamber_model,
                    pollen_model=pollen_model,
                    temporal_prior=temporal_prior,
                    args=args,
                    overlay_writer=overlay_writer,
                )
        except Exception as exc:
            elapsed = time.perf_counter() - start
            append_status(args.output_folder, {
                "video_path": str(video_path),
                "video_id": video_id,
                "status": "failed",
                "frame_count": "",
                "refreshed_bee_rows": "",
                "message": str(exc),
                "elapsed_seconds": f"{elapsed:.2f}",
                "completed_at": datetime.now().isoformat(timespec="seconds"),
            })
            print(f"ERROR: refresh failed for {video_path}: {exc}", file=sys.stderr)
            write_temporal_prior_summary(args.output_folder, temporal_prior)
            return 1

        elapsed = time.perf_counter() - start
        append_status(args.output_folder, {
            "video_path": str(video_path),
            "video_id": video_id,
            "status": "complete",
            "frame_count": str(frame_count),
            "refreshed_bee_rows": str(refreshed_rows),
            "message": "",
            "elapsed_seconds": f"{elapsed:.2f}",
            "completed_at": datetime.now().isoformat(timespec="seconds"),
        })
        save_temporal_prior_checkpoint(
            args.output_folder,
            temporal_prior,
            video_path,
            global_idx,
            total,
            args,
            resolution,
        )
        processed += 1
        print(
            f"  ✓ {video_path.name}: {frame_count} frame(s), {refreshed_rows} bee rows, "
            f"{elapsed:.1f}s ({idx}/{remaining} remaining batch videos)"
        )

    write_temporal_prior_summary(args.output_folder, temporal_prior)
    print(f"Done. Refreshed {processed} video(s); skipped {skipped}.")
    print(f"Outputs: {args.output_folder}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
