import json
from pathlib import Path

from core.validation_review import (
    frame_has_bee_mask_annotation,
    selected_frame_indices,
    video_has_selected_bee_masks,
)


def _write_frame(project: Path, video_id: str, frame_idx: int, annotations):
    stem = f"frame_{frame_idx:06d}"
    json_dir = project / "annotations" / "json" / video_id
    png_dir = project / "annotations" / "png" / video_id
    json_dir.mkdir(parents=True, exist_ok=True)
    png_dir.mkdir(parents=True, exist_ok=True)
    (json_dir / f"{stem}.json").write_text(json.dumps(annotations))
    (png_dir / f"{stem}.png").write_bytes(b"mask")


def test_selected_frame_indices_uses_video_metadata(tmp_path):
    metadata_dir = tmp_path / "frames" / "video-a"
    metadata_dir.mkdir(parents=True)
    (metadata_dir / "video_metadata.json").write_text(
        json.dumps({"split": "val", "selected_frames": [0, "20", 40]})
    )

    assert selected_frame_indices(tmp_path, "video-a") == [0, 20, 40]


def test_frame_has_bee_mask_requires_png_and_non_bbox_bee_metadata(tmp_path):
    bee_mask = {"mask_id": 1, "category": "bee", "bbox_only": False}
    _write_frame(tmp_path, "video-a", 20, [bee_mask])

    assert frame_has_bee_mask_annotation(tmp_path, "video-a", 20)

    _write_frame(tmp_path, "video-a", 40, [dict(bee_mask, bbox_only=True)])
    assert not frame_has_bee_mask_annotation(tmp_path, "video-a", 40)

    _write_frame(tmp_path, "video-a", 50, [dict(bee_mask, mask_id="invalid")])
    assert not frame_has_bee_mask_annotation(tmp_path, "video-a", 50)

    assert not frame_has_bee_mask_annotation(tmp_path, "video-a", 60)


def test_video_has_selected_bee_masks_ignores_unselected_masks(tmp_path):
    metadata_dir = tmp_path / "frames" / "video-a"
    metadata_dir.mkdir(parents=True)
    (metadata_dir / "video_metadata.json").write_text(
        json.dumps({"split": "val", "selected_frames": [0, 20]})
    )

    annotation = {"mask_id": 1, "category": "bee", "bbox_only": False}
    _write_frame(tmp_path, "video-a", 10, [annotation])
    assert not video_has_selected_bee_masks(tmp_path, "video-a")

    _write_frame(tmp_path, "video-a", 20, [annotation])
    assert video_has_selected_bee_masks(tmp_path, "video-a")
