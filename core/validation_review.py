"""Helpers for locating editable ground-truth validation masks."""

from __future__ import annotations

import json
from pathlib import Path


def selected_frame_indices(project_path: Path | str, video_id: str) -> list[int]:
    """Return the frame indices selected for a video's dataset split."""
    metadata_path = Path(project_path) / "frames" / video_id / "video_metadata.json"
    try:
        with metadata_path.open() as metadata_file:
            metadata = json.load(metadata_file)
    except (OSError, json.JSONDecodeError, TypeError):
        return []

    selected = metadata.get("selected_frames", [])
    if not isinstance(selected, list):
        return []

    indices = []
    for value in selected:
        try:
            indices.append(int(value))
        except (TypeError, ValueError):
            continue
    return indices


def frame_has_bee_mask_annotation(
    project_path: Path | str,
    video_id: str,
    frame_idx: int,
) -> bool:
    """Return whether a frame has at least one saved, editable bee mask."""
    project_path = Path(project_path)
    stem = f"frame_{int(frame_idx):06d}"
    json_path = project_path / "annotations" / "json" / video_id / f"{stem}.json"
    png_path = project_path / "annotations" / "png" / video_id / f"{stem}.png"

    if not json_path.is_file() or not png_path.is_file():
        return False

    try:
        with json_path.open() as annotation_file:
            annotations = json.load(annotation_file)
    except (OSError, json.JSONDecodeError, TypeError):
        return False

    if not isinstance(annotations, list):
        return False

    for annotation in annotations:
        if not isinstance(annotation, dict):
            continue
        if annotation.get("category", "bee") != "bee":
            continue
        if annotation.get("bbox_only", False):
            continue
        try:
            mask_id = int(
                annotation.get("mask_id", annotation.get("instance_id", 0)) or 0
            )
        except (TypeError, ValueError):
            continue
        if mask_id > 0:
            return True
    return False


def video_has_selected_bee_masks(project_path: Path | str, video_id: str) -> bool:
    """Return whether any selected frame in a video has an editable bee mask."""
    return any(
        frame_has_bee_mask_annotation(project_path, video_id, frame_idx)
        for frame_idx in selected_frame_indices(project_path, video_id)
    )
