"""
Project overview dialog with dataset, annotation, tracking, and carbon stats.
"""

import csv
import json
import re
from pathlib import Path
from typing import Dict, Iterable, List, Set, Tuple

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)


def _read_json(path: Path, default):
    try:
        if path.exists():
            with open(path, "r") as f:
                return json.load(f)
    except Exception:
        pass
    return default


def _fmt_int(value) -> str:
    try:
        return f"{int(value):,}"
    except (TypeError, ValueError):
        return "0"


def _fmt_float(value, digits: int = 4) -> str:
    try:
        return f"{float(value):,.{digits}f}"
    except (TypeError, ValueError):
        return "0"


def _frame_index(path: Path):
    try:
        return int(path.stem.split("_")[1])
    except (IndexError, ValueError):
        return None


def _is_bee_annotation(ann: Dict) -> bool:
    category = ann.get("category")
    category_id = ann.get("category_id")
    if category is not None:
        return category == "bee"
    return category_id in (None, 1)


def _has_segmentation(ann: Dict) -> bool:
    return _is_bee_annotation(ann) and not ann.get("bbox_only", False)


def _has_bbox(ann: Dict) -> bool:
    bbox = ann.get("bbox")
    return _is_bee_annotation(ann) and isinstance(bbox, list) and len(bbox) >= 4


def _annotation_identity(ann: Dict, fallback_prefix: str, fallback_idx: int) -> str:
    instance_id = ann.get("instance_id", ann.get("mask_id"))
    if instance_id is not None:
        return str(instance_id)
    bbox = ann.get("bbox")
    if isinstance(bbox, list):
        return f"bbox:{bbox}"
    return f"{fallback_prefix}:{fallback_idx}"


class ProjectOverviewDialog(QDialog):
    """Shows high-level statistics for the current project."""

    def __init__(self, project_path: Path, project_manager=None, parent=None):
        super().__init__(parent)
        self.project_path = Path(project_path)
        self.project_manager = project_manager
        self.setWindowTitle("Project Overview")
        self.setMinimumSize(720, 680)

        self.content_layout = QVBoxLayout()
        self._init_ui()
        self.refresh()

    def _init_ui(self):
        layout = QVBoxLayout(self)

        header_layout = QHBoxLayout()
        self.title_label = QLabel(self.project_path.name)
        self.title_label.setStyleSheet("font-size: 18px; font-weight: 600;")
        header_layout.addWidget(self.title_label)
        header_layout.addStretch()

        refresh_btn = QPushButton("Refresh")
        refresh_btn.clicked.connect(self.refresh)
        header_layout.addWidget(refresh_btn)

        export_btn = QPushButton("Export Video CSV...")
        export_btn.clicked.connect(self.export_video_csv)
        header_layout.addWidget(export_btn)
        layout.addLayout(header_layout)

        self.subtitle_label = QLabel(str(self.project_path))
        self.subtitle_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.subtitle_label.setStyleSheet("color: #555;")
        layout.addWidget(self.subtitle_label)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll_widget = QWidget()
        scroll_widget.setLayout(self.content_layout)
        scroll.setWidget(scroll_widget)
        layout.addWidget(scroll)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def refresh(self):
        self._clear_layout()
        stats = self._collect_stats()

        self._add_section("Dataset", [
            (
                "Training videos",
                stats["dataset"]["train_videos"],
                "Number of video files currently assigned to the train split.",
            ),
            (
                "Validation videos",
                stats["dataset"]["val_videos"],
                "Number of video files currently assigned to the validation split.",
            ),
            (
                "Training frames",
                stats["dataset"]["train_frames"],
                "Number of selected frame images used for training across videos in the train split.",
            ),
            (
                "Validation frames",
                stats["dataset"]["val_frames"],
                "Number of selected frame images used for validation across videos in the validation split.",
            ),
            (
                "Training annotated frames",
                stats["dataset"]["train_annotated_frames"],
                "Number of selected train frames with at least one bee bounding box or bee segmentation annotation.",
            ),
            (
                "Validation annotated frames",
                stats["dataset"]["val_annotated_frames"],
                "Number of selected validation frames with at least one bee bounding box or bee segmentation annotation.",
            ),
        ])

        self._add_section("Bee Annotations", [
            (
                "Training bee bounding boxes",
                stats["annotations"]["train_bboxes"],
                "Number of bee bounding boxes on train frames. This includes bbox-only annotations "
                "and explicit or implicit bounding boxes from bee segmentations.",
            ),
            (
                "Training bee segmentations",
                stats["annotations"]["train_segmentations"],
                "Number of bee mask/segmentation annotations on train frames. Bbox-only bees are not counted here.",
            ),
            (
                "Training frames with bee boxes",
                stats["annotations"]["train_bbox_frames"],
                "Number of selected train frames that contain at least one bee bounding box. "
                "Bee segmentations count here because every mask has an implied bounding box.",
            ),
            (
                "Training frames with bee segmentations",
                stats["annotations"]["train_segmentation_frames"],
                "Number of selected train frames that contain at least one bee segmentation mask.",
            ),
            (
                "Validation bee bounding boxes",
                stats["annotations"]["val_bboxes"],
                "Number of bee bounding boxes on validation frames. This includes bbox-only annotations "
                "and explicit or implicit bounding boxes from bee segmentations.",
            ),
            (
                "Validation bee segmentations",
                stats["annotations"]["val_segmentations"],
                "Number of bee mask/segmentation annotations on validation frames. Bbox-only bees are not counted here.",
            ),
            (
                "Validation frames with bee boxes",
                stats["annotations"]["val_bbox_frames"],
                "Number of selected validation frames that contain at least one bee bounding box. "
                "Bee segmentations count here because every mask has an implied bounding box.",
            ),
            (
                "Validation frames with bee segmentations",
                stats["annotations"]["val_segmentation_frames"],
                "Number of selected validation frames that contain at least one bee segmentation mask.",
            ),
        ])

        self._add_section("Hive, Pollen & Chamber Annotations", [
            (
                "Training chamber annotations",
                self._format_video_level_counts(stats["video_level"]["train"]["chamber"]),
                "Number of chamber annotations on videos in the train split. "
                "These are video-level annotations shared across frames, not per-frame bee annotations.",
            ),
            (
                "Training hive annotations",
                self._format_video_level_counts(stats["video_level"]["train"]["hive"]),
                "Number of hive annotations on videos in the train split. "
                "The value shows total annotations, with mask-backed and bbox-only counts in parentheses.",
            ),
            (
                "Training pollen ball annotations",
                self._format_video_level_counts(stats["video_level"]["train"]["pollen"]),
                "Number of pollen ball annotations on videos in the train split. "
                "The value shows total annotations, with mask-backed and bbox-only counts in parentheses.",
            ),
            (
                "Validation chamber annotations",
                self._format_video_level_counts(stats["video_level"]["val"]["chamber"]),
                "Number of chamber annotations on videos in the validation split. "
                "These are video-level annotations shared across frames, not per-frame bee annotations.",
            ),
            (
                "Validation hive annotations",
                self._format_video_level_counts(stats["video_level"]["val"]["hive"]),
                "Number of hive annotations on videos in the validation split. "
                "The value shows total annotations, with mask-backed and bbox-only counts in parentheses.",
            ),
            (
                "Validation pollen ball annotations",
                self._format_video_level_counts(stats["video_level"]["val"]["pollen"]),
                "Number of pollen ball annotations on videos in the validation split. "
                "The value shows total annotations, with mask-backed and bbox-only counts in parentheses.",
            ),
            (
                "Training videos with any of these",
                stats["video_level"]["train"]["videos_with_any"],
                "Number of train videos that have at least one chamber, hive, or pollen ball annotation.",
            ),
            (
                "Validation videos with any of these",
                stats["video_level"]["val"]["videos_with_any"],
                "Number of validation videos that have at least one chamber, hive, or pollen ball annotation.",
            ),
        ])

        self._add_section("Tracking Validation", [
            (
                "Sequences",
                stats["tracking"]["total_sequences"],
                "Total number of tracking validation sequences saved in tracking_sequences.json.",
            ),
            (
                "Enabled sequences",
                stats["tracking"]["enabled_sequences"],
                "Number of tracking validation sequences currently marked enabled.",
            ),
            (
                "Frames in enabled sequences",
                stats["tracking"]["sequence_frames"],
                "Total inclusive frame span across enabled tracking sequences.",
            ),
            (
                "Sequence frames with bee boxes",
                stats["tracking"]["bbox_frames"],
                "Number of frames inside enabled tracking sequences that contain at least one bee bounding box.",
            ),
            (
                "Sequence frames with bee segmentations",
                stats["tracking"]["segmentation_frames"],
                "Number of frames inside enabled tracking sequences that contain at least one bee segmentation mask.",
            ),
            (
                "Bee boxes in enabled sequences",
                stats["tracking"]["bboxes"],
                "Total bee bounding boxes on frames inside enabled tracking sequences, including implicit boxes from segmentations.",
            ),
            (
                "Bee segmentations in enabled sequences",
                stats["tracking"]["segmentations"],
                "Total bee segmentation masks on frames inside enabled tracking sequences.",
            ),
        ])

        self._add_section("ArUco Identifications", [
            (
                "Video-level ArUco IDs",
                stats["aruco"]["tracked_aruco_ids"],
                "Number of distinct ArUco IDs saved in each video's video-level tracking map. "
                "This is the long-lived identity table that maps marker ID to bee instance ID.",
            ),
            (
                "Video-level identified instances",
                stats["aruco"]["tracked_instances"],
                "Number of bee instances that currently have a video-level ArUco identity assignment. "
                "One instance is counted per video/instance pair, even if the same marker appears in many frames.",
            ),
            (
                "Frame marker detections",
                stats["aruco"]["frame_marker_detections"],
                "Total ArUco detections stored directly on frame annotations. "
                "This counts every detected marker occurrence across frames, so the same marker can contribute multiple times.",
            ),
            (
                "Unique frame marker IDs",
                stats["aruco"]["unique_frame_marker_ids"],
                "Number of distinct ArUco marker IDs found in frame-level annotation metadata.",
            ),
            (
                "Frame instances with markers",
                stats["aruco"]["frame_instances_with_markers"],
                "Number of annotated bee instances that have marker metadata on a specific frame. "
                "The same bee identity on different frames is counted once per frame.",
            ),
        ])

        self._add_section("Carbon Tracking", [
            (
                "Training runs in ledger",
                stats["carbon"]["runs"],
                "Number of model training runs recorded in the project's carbon_tracking/carbon_usage.json ledger.",
            ),
            (
                "Runs with measured carbon",
                stats["carbon"]["tracked_runs"],
                "Number of ledger runs with parsed electricity or CO2 measurements from carbontracker.",
            ),
            (
                "Electricity",
                f"{_fmt_float(stats['carbon']['energy_kwh'], 6)} kWh",
                "Total measured electricity use across recorded training runs, as reported by carbontracker.",
            ),
            (
                "CO2eq",
                f"{_fmt_float(stats['carbon']['co2eq_g'], 3)} g",
                "Total carbon dioxide equivalent emissions across recorded training runs, as reported by carbontracker.",
            ),
            (
                "Tracked duration",
                self._format_duration(stats["carbon"]["duration_s"]),
                "Total measured training duration represented in the carbon ledger.",
            ),
        ])

        self._add_section("Models", [
            (
                "Model checkpoint files",
                stats["models"]["checkpoints"],
                "Number of .pt and .pth checkpoint files currently under the project's models folder.",
            ),
            (
                "YOLO run folders",
                stats["models"]["yolo_run_dirs"],
                "Number of YOLO training run directories under yolo_runs, yolo_bbox_runs, and yolo_instance_focused_runs.",
            ),
            (
                "SAM2 fine-tuned checkpoints",
                stats["models"]["sam2_checkpoints"],
                "Number of .pt and .pth checkpoint files in models/sam2_finetuned.",
            ),
        ])

        self.content_layout.addStretch()

    def _clear_layout(self):
        while self.content_layout.count():
            item = self.content_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()

    def _add_section(self, title: str, rows: List[Tuple]):
        group = QGroupBox(title)
        form = QFormLayout()
        form.setLabelAlignment(Qt.AlignmentFlag.AlignLeft)
        form.setFormAlignment(Qt.AlignmentFlag.AlignTop)

        for row in rows:
            label, value = row[:2]
            tooltip = row[2] if len(row) > 2 else None
            value_text = value if isinstance(value, str) else _fmt_int(value)
            value_label = QLabel(value_text)
            value_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
            form.addRow(self._label_widget(label, tooltip), value_label)

        group.setLayout(form)
        self.content_layout.addWidget(group)

    def _label_widget(self, label: str, tooltip: str = None) -> QWidget:
        if not tooltip:
            plain_label = QLabel(f"{label}:")
            plain_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
            return plain_label

        wrapper = QWidget()
        layout = QHBoxLayout(wrapper)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(5)

        label_widget = QLabel(f"{label}:")
        label_widget.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(label_widget)

        help_label = QLabel("?")
        help_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        help_label.setToolTip(tooltip)
        help_label.setFixedSize(16, 16)
        help_label.setStyleSheet(
            "QLabel { border: 1px solid #888; border-radius: 8px; color: #555; font-weight: 600; }"
        )
        layout.addWidget(help_label)
        layout.addStretch()
        return wrapper

    def _collect_stats(self) -> Dict:
        train_videos = self._videos_by_split("train")
        val_videos = self._videos_by_split("val")
        dataset_stats = self._dataset_stats(train_videos, val_videos)
        annotation_stats = self._annotation_stats(train_videos, val_videos)

        return {
            "dataset": dataset_stats,
            "annotations": annotation_stats,
            "video_level": {
                "train": self._video_level_annotation_counts(train_videos),
                "val": self._video_level_annotation_counts(val_videos),
            },
            "tracking": self._tracking_stats(),
            "aruco": self._aruco_stats(),
            "carbon": self._carbon_stats(),
            "models": self._model_stats(),
        }

    def _videos_by_split(self, split: str) -> List[str]:
        if self.project_manager is not None:
            try:
                return self.project_manager.get_videos_by_split(split)
            except Exception:
                pass

        split_dir = self.project_path / "input_data" / split
        if not split_dir.exists():
            return []
        return sorted(
            path.stem
            for path in split_dir.iterdir()
            if path.suffix.lower() in {".mp4", ".avi", ".mov", ".mkv", ".mjpeg"}
        )

    def _dataset_stats(self, train_videos: List[str], val_videos: List[str]) -> Dict:
        if self.project_manager is not None:
            try:
                stats = self.project_manager.get_dataset_statistics()
                train = stats.get("train", {})
                val = stats.get("val", {})
                return {
                    "train_videos": train.get("videos", len(train_videos)),
                    "val_videos": val.get("videos", len(val_videos)),
                    "train_frames": self._count_selected_frames(train_videos, "train"),
                    "val_frames": self._count_selected_frames(val_videos, "val"),
                    "train_annotated_frames": self._count_annotated_frames(train_videos, "train"),
                    "val_annotated_frames": self._count_annotated_frames(val_videos, "val"),
                }
            except Exception:
                pass

        return {
            "train_videos": len(train_videos),
            "val_videos": len(val_videos),
            "train_frames": self._count_selected_frames(train_videos, "train"),
            "val_frames": self._count_selected_frames(val_videos, "val"),
            "train_annotated_frames": self._count_annotated_frames(train_videos, "train"),
            "val_annotated_frames": self._count_annotated_frames(val_videos, "val"),
        }

    def _count_selected_frames(self, video_ids: Iterable[str], split: str) -> int:
        total = 0
        for video_id in video_ids:
            total += len(self._selected_frame_ids(video_id, split))
        return total

    def _count_annotated_frames(self, video_ids: Iterable[str], split: str) -> int:
        total = 0
        for video_id in video_ids:
            for frame_idx in self._selected_frame_ids(video_id, split):
                counts = self._frame_annotation_counts(video_id, frame_idx)
                if counts["bboxes"] or counts["segmentations"]:
                    total += 1
        return total

    def export_video_csv(self):
        """Export one row per train/validation video with annotation frame counts."""
        default_path = self.project_path / "project_video_annotation_summary.csv"
        output_path, _ = QFileDialog.getSaveFileName(
            self,
            "Export Video-Level CSV",
            str(default_path),
            "CSV files (*.csv);;All files (*)",
        )
        if not output_path:
            return

        rows = self._video_csv_rows()
        fieldnames = [
            "dataset",
            "video_id",
            "year",
            "selected_frames_with_bounding_box_annotations",
            "selected_frames_with_segmentations",
        ]

        try:
            with open(output_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)
        except Exception as exc:
            QMessageBox.critical(self, "Export Failed", f"Could not write CSV:\n\n{exc}")
            return

        QMessageBox.information(
            self,
            "Export Complete",
            f"Exported {len(rows):,} video rows to:\n{output_path}",
        )

    def _video_csv_rows(self) -> List[Dict]:
        rows = []
        for dataset in ("train", "val"):
            label = "training" if dataset == "train" else "validation"
            for video_id in self._videos_by_split(dataset):
                selected_frame_ids = self._selected_frame_ids(video_id, dataset)
                counts = self._video_annotation_frame_counts(video_id, selected_frame_ids)
                rows.append({
                    "dataset": label,
                    "video_id": video_id,
                    "year": self._extract_year(video_id) or "",
                    "selected_frames_with_bounding_box_annotations": counts["bbox_frames"],
                    "selected_frames_with_segmentations": counts["segmentation_frames"],
                })
        return rows

    def _video_annotation_frame_counts(self, video_id: str, frame_ids: Iterable[int] = None) -> Dict:
        counts = {"bbox_frames": 0, "segmentation_frames": 0}
        if frame_ids is None:
            frame_ids = self._annotation_frame_ids(video_id)
        for frame_idx in frame_ids:
            frame_counts = self._frame_annotation_counts(video_id, frame_idx)
            if frame_counts["bboxes"]:
                counts["bbox_frames"] += 1
            if frame_counts["segmentations"]:
                counts["segmentation_frames"] += 1
        return counts

    def _selected_frame_ids(self, video_id: str, expected_split: str) -> Set[int]:
        metadata = _read_json(
            self.project_path / "frames" / video_id / "video_metadata.json",
            {},
        )
        if not isinstance(metadata, dict):
            return set()

        split = metadata.get("split")
        if split and split != expected_split:
            return set()

        selected = metadata.get("selected_frames", [])
        if not isinstance(selected, list):
            return set()

        frame_ids = set()
        for frame_idx in selected:
            try:
                frame_ids.add(int(frame_idx))
            except (TypeError, ValueError):
                continue
        return frame_ids

    @staticmethod
    def _extract_year(video_id: str):
        match = re.search(r"(?<!\d)(19\d{2}|20\d{2})(?!\d)", video_id)
        return match.group(1) if match else None

    def _annotation_stats(self, train_videos: List[str], val_videos: List[str]) -> Dict:
        train = self._annotation_counts_for_videos(train_videos, "train")
        val = self._annotation_counts_for_videos(val_videos, "val")
        return {
            "train_bboxes": train["bboxes"],
            "train_segmentations": train["segmentations"],
            "train_bbox_frames": train["bbox_frames"],
            "train_segmentation_frames": train["segmentation_frames"],
            "val_bboxes": val["bboxes"],
            "val_segmentations": val["segmentations"],
            "val_bbox_frames": val["bbox_frames"],
            "val_segmentation_frames": val["segmentation_frames"],
        }

    def _annotation_counts_for_videos(self, video_ids: Iterable[str], split: str = None) -> Dict:
        counts = {"bboxes": 0, "segmentations": 0, "bbox_frames": 0, "segmentation_frames": 0}
        for video_id in video_ids:
            frame_ids = self._selected_frame_ids(video_id, split) if split else self._annotation_frame_ids(video_id)
            for frame_idx in frame_ids:
                frame_counts = self._frame_annotation_counts(video_id, frame_idx)
                counts["bboxes"] += frame_counts["bboxes"]
                counts["segmentations"] += frame_counts["segmentations"]
                if frame_counts["bboxes"]:
                    counts["bbox_frames"] += 1
                if frame_counts["segmentations"]:
                    counts["segmentation_frames"] += 1
        return counts

    def _annotation_frame_ids(self, video_id: str) -> Set[int]:
        frame_ids = set()
        for base in ("json", "bbox"):
            ann_dir = self.project_path / "annotations" / base / video_id
            if ann_dir.exists():
                frame_ids.update(
                    idx
                    for idx in (_frame_index(path) for path in ann_dir.glob("frame_*.json"))
                    if idx is not None
                )
        return frame_ids

    def _frame_annotation_counts(self, video_id: str, frame_idx: int) -> Dict:
        json_anns = _read_json(
            self.project_path / "annotations" / "json" / video_id / f"frame_{frame_idx:06d}.json",
            [],
        )
        bbox_anns = _read_json(
            self.project_path / "annotations" / "bbox" / video_id / f"frame_{frame_idx:06d}.json",
            [],
        )

        segmentations = sum(1 for ann in json_anns if _has_segmentation(ann))
        bbox_instance_ids = set()
        for idx, ann in enumerate(bbox_anns):
            if _has_bbox(ann):
                bbox_instance_ids.add(_annotation_identity(ann, "bbox", idx))
        for idx, ann in enumerate(json_anns):
            if _has_bbox(ann):
                bbox_instance_ids.add(_annotation_identity(ann, "json", idx))
            elif _has_segmentation(ann):
                # Every segmentation has an implicit bounding box, even if older
                # annotation metadata did not store bbox coordinates.
                bbox_instance_ids.add(_annotation_identity(ann, "segmentation", idx))
        bboxes = len(bbox_instance_ids)

        return {"bboxes": bboxes, "segmentations": segmentations}

    def _video_level_annotation_counts(self, video_ids: Iterable[str]) -> Dict:
        categories = ("chamber", "hive", "pollen")
        counts = {
            category: {"total": 0, "masks": 0, "bbox_only": 0, "videos": 0}
            for category in categories
        }
        counts["videos_with_any"] = 0

        for video_id in video_ids:
            anns = self._video_level_annotations(video_id)
            if not anns:
                continue

            video_had_any = False
            video_categories = set()
            for ann in anns:
                category = ann.get("category", "chamber")
                if category not in categories:
                    continue

                video_had_any = True
                video_categories.add(category)
                counts[category]["total"] += 1

                if ann.get("bbox_only", False) or ann.get("mask_id", 0) in (0, None):
                    counts[category]["bbox_only"] += 1
                else:
                    counts[category]["masks"] += 1

            if video_had_any:
                counts["videos_with_any"] += 1
            for category in video_categories:
                counts[category]["videos"] += 1

        return counts

    def _video_level_annotations(self, video_id: str) -> List[Dict]:
        data = _read_json(
            self.project_path / "annotations" / "json" / video_id / "video_annotations.json",
            {},
        )
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            anns = data.get("annotations", [])
            return anns if isinstance(anns, list) else []
        return []

    def _tracking_stats(self) -> Dict:
        data = _read_json(self.project_path / "tracking_sequences.json", {"sequences": []})
        sequences = data.get("sequences", []) if isinstance(data, dict) else []
        enabled = [seq for seq in sequences if seq.get("enabled", True)]

        stats = {
            "total_sequences": len(sequences),
            "enabled_sequences": len(enabled),
            "sequence_frames": 0,
            "bbox_frames": 0,
            "segmentation_frames": 0,
            "bboxes": 0,
            "segmentations": 0,
        }

        for seq in enabled:
            video_id = seq.get("video_id")
            if not video_id:
                continue
            try:
                start = int(seq.get("start_frame", 0))
                end = int(seq.get("end_frame", start))
            except (TypeError, ValueError):
                continue

            if end < start:
                start, end = end, start

            for frame_idx in range(start, end + 1):
                stats["sequence_frames"] += 1
                frame_counts = self._frame_annotation_counts(video_id, frame_idx)
                stats["bboxes"] += frame_counts["bboxes"]
                stats["segmentations"] += frame_counts["segmentations"]
                if frame_counts["bboxes"]:
                    stats["bbox_frames"] += 1
                if frame_counts["segmentations"]:
                    stats["segmentation_frames"] += 1

        return stats

    def _aruco_stats(self) -> Dict:
        tracked_aruco_ids = set()
        tracked_instances = set()
        frame_marker_ids = set()
        frame_marker_instances = set()
        frame_marker_detections = 0

        video_jsons = self.project_path.glob("annotations/json/*/video_annotations.json")
        for path in video_jsons:
            data = _read_json(path, {})
            if isinstance(data, dict):
                aruco_tracking = data.get("aruco_tracking", {})
                for aruco_id, instance_id in aruco_tracking.items():
                    tracked_aruco_ids.add(str(aruco_id))
                    tracked_instances.add((path.parent.name, str(instance_id)))

        for path in self.project_path.glob("annotations/json/*/frame_*.json"):
            anns = _read_json(path, [])
            if not isinstance(anns, list):
                continue
            video_id = path.parent.name
            frame_idx = _frame_index(path)
            for ann in anns:
                marker = ann.get("marker")
                if not marker or marker.get("type") != "aruco":
                    continue
                marker_id = marker.get("id")
                instance_id = ann.get("instance_id", ann.get("mask_id"))
                frame_marker_detections += 1
                frame_marker_ids.add(str(marker_id))
                frame_marker_instances.add((video_id, frame_idx, str(instance_id)))

        return {
            "tracked_aruco_ids": len(tracked_aruco_ids),
            "tracked_instances": len(tracked_instances),
            "frame_marker_detections": frame_marker_detections,
            "unique_frame_marker_ids": len(frame_marker_ids),
            "frame_instances_with_markers": len(frame_marker_instances),
        }

    def _carbon_stats(self) -> Dict:
        ledger = _read_json(self.project_path / "carbon_tracking" / "carbon_usage.json", {})
        totals = ledger.get("totals", {}) if isinstance(ledger, dict) else {}
        return {
            "runs": totals.get("runs", len(ledger.get("runs", [])) if isinstance(ledger, dict) else 0),
            "tracked_runs": totals.get("tracked_runs", 0),
            "duration_s": totals.get("duration_s", 0.0),
            "energy_kwh": totals.get("energy_kwh", 0.0),
            "co2eq_g": totals.get("co2eq_g", 0.0),
        }

    def _model_stats(self) -> Dict:
        models_dir = self.project_path / "models"
        checkpoints = len(list(models_dir.rglob("*.pt"))) + len(list(models_dir.rglob("*.pth"))) if models_dir.exists() else 0
        yolo_run_dirs = 0
        for dirname in ("yolo_runs", "yolo_bbox_runs", "yolo_instance_focused_runs"):
            run_root = models_dir / dirname
            if run_root.exists():
                yolo_run_dirs += sum(1 for path in run_root.iterdir() if path.is_dir())
        sam2_dir = models_dir / "sam2_finetuned"
        sam2_checkpoints = len(list(sam2_dir.glob("*.pt"))) + len(list(sam2_dir.glob("*.pth"))) if sam2_dir.exists() else 0
        return {
            "checkpoints": checkpoints,
            "yolo_run_dirs": yolo_run_dirs,
            "sam2_checkpoints": sam2_checkpoints,
        }

    @staticmethod
    def _format_duration(seconds) -> str:
        try:
            seconds = int(float(seconds))
        except (TypeError, ValueError):
            seconds = 0
        hours, rem = divmod(seconds, 3600)
        minutes, secs = divmod(rem, 60)
        if hours:
            return f"{hours:,}h {minutes:02d}m {secs:02d}s"
        if minutes:
            return f"{minutes:,}m {secs:02d}s"
        return f"{secs}s"

    @staticmethod
    def _format_video_level_counts(counts: Dict) -> str:
        total = counts.get("total", 0)
        masks = counts.get("masks", 0)
        bbox_only = counts.get("bbox_only", 0)
        videos = counts.get("videos", 0)
        return (
            f"{_fmt_int(total)} "
            f"({_fmt_int(masks)} masks, {_fmt_int(bbox_only)} bbox-only, "
            f"{_fmt_int(videos)} videos)"
        )
