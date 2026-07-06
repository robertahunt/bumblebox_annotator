#!/usr/bin/env python3
"""Command-line batch video inference runner."""

import argparse
import hashlib
import os
import sys
from pathlib import Path

from PyQt6.QtCore import QCoreApplication, QSettings

from gui.batch_video_inference_worker import BatchVideoInferenceWorker


DEFAULT_POLLEN_MODEL_PATH = Path(
    "/home/august/Dropbox/bee_annotator/projects/test_august_june25/models/pollen_segmentation_jun23/weights/best.pt"
)

DEFAULT_SWEEP_OVERRIDES = {
    "minMarkerPerimeterRate": [0.019153],
    "maxMarkerPerimeterRate": [0.052808],
    "adaptiveThreshWinSizeMin": [3],
    "adaptiveThreshWinSizeMax": [30, 50, 70, 90, 110, 130, 150],
    "adaptiveThreshWinSizeStep": [3],
    "polygonalApproxAccuracyRate": [0.08],
    "adaptiveThreshConstant": list(range(3, 20)),
}


def load_saved_defaults():
    """Load the last GUI batch-inference settings as CLI defaults."""
    settings = QSettings("BumbleBoxAnnotator", "BatchVideoInference")

    def path_value(key):
        value = settings.value(key, "", type=str)
        value = str(value).strip() if value is not None else ""
        return Path(value) if value else None

    def text_value(key, default):
        value = settings.value(key, default, type=str)
        value = str(value).strip() if value is not None else ""
        return value or default

    def int_value(key, default):
        try:
            return int(float(settings.value(key, default)))
        except (TypeError, ValueError):
            return default

    def float_value(key, default):
        try:
            return float(settings.value(key, default))
        except (TypeError, ValueError):
            return default

    def list_value(key, default, cast):
        raw = settings.value(key, "", type=str)
        raw = str(raw).strip().strip('"') if raw is not None else ""
        if not raw:
            return list(default)

        values = []
        for token in raw.split(","):
            token = token.strip()
            if not token:
                continue
            try:
                if cast is int:
                    value = float(token)
                    if not value.is_integer():
                        raise ValueError
                    values.append(int(value))
                else:
                    values.append(float(token))
            except ValueError:
                return list(default)
        return values or list(default)

    aruco_sweep_overrides = {
        "minMarkerPerimeterRate": [float_value(
            "aruco/sweep_min_perimeter",
            DEFAULT_SWEEP_OVERRIDES["minMarkerPerimeterRate"][0],
        )],
        "maxMarkerPerimeterRate": [float_value(
            "aruco/sweep_max_perimeter",
            DEFAULT_SWEEP_OVERRIDES["maxMarkerPerimeterRate"][0],
        )],
        "adaptiveThreshWinSizeMin": [int_value(
            "aruco/sweep_win_min",
            DEFAULT_SWEEP_OVERRIDES["adaptiveThreshWinSizeMin"][0],
        )],
        "adaptiveThreshWinSizeMax": list_value(
            "aruco/sweep_win_max",
            DEFAULT_SWEEP_OVERRIDES["adaptiveThreshWinSizeMax"],
            int,
        ),
        "adaptiveThreshWinSizeStep": [int_value(
            "aruco/sweep_win_step",
            DEFAULT_SWEEP_OVERRIDES["adaptiveThreshWinSizeStep"][0],
        )],
        "polygonalApproxAccuracyRate": [float_value(
            "aruco/sweep_polygon",
            DEFAULT_SWEEP_OVERRIDES["polygonalApproxAccuracyRate"][0],
        )],
        "adaptiveThreshConstant": list_value(
            "aruco/sweep_constant",
            DEFAULT_SWEEP_OVERRIDES["adaptiveThreshConstant"],
            int,
        ),
    }

    return {
        "bee_model": path_value("models/bee_model_path"),
        "hive_model": path_value("models/hive_model_path"),
        "pollen_model": path_value("models/pollen_model_path"),
        "chamber_model": path_value("models/chamber_model_path"),
        "tag_map": path_value("aruco/tag_list_path"),
        "exclude_tags": path_value("aruco/exclude_tag_list_path"),
        "aruco_dictionary": text_value("aruco/dictionary", "4x4_100"),
        "aruco_profile": text_value("aruco/profile", "daily"),
        "aruco_sample_frames": int_value("aruco/sample_frames", 12),
        "aruco_max_combinations": int_value("aruco/max_combinations", 45),
        "aruco_bank_size": int_value("aruco/bank_size", 6),
        "aruco_workers": int_value("aruco/workers", max(1, min(15, (os.cpu_count() or 2) - 1))),
        "aruco_expected_tags": float_value("aruco/expected_tags", 0.0),
        "aruco_sweep_overrides": aruco_sweep_overrides,
    }


def parse_args():
    saved = load_saved_defaults()
    saved_sweep = saved["aruco_sweep_overrides"]
    parser = argparse.ArgumentParser(
        description="Run BumbleBox batch video inference without opening the GUI."
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--video", type=Path, help="Single video to process")
    source.add_argument("--file-list", type=Path, help="Text file containing ordered video paths")

    parser.add_argument("--output-folder", type=Path, required=True)
    parser.add_argument("--bee-model", type=Path, default=saved["bee_model"] or Path("best.pt"))
    parser.add_argument("--hive-model", type=Path, default=saved["hive_model"])
    parser.add_argument("--pollen-model", type=Path, default=saved["pollen_model"] or DEFAULT_POLLEN_MODEL_PATH)
    parser.add_argument("--chamber-model", type=Path, default=saved["chamber_model"])
    parser.add_argument("--tag-map", type=Path, default=saved["tag_map"], help="Simple allowlist or per-MC-pair CSV")
    parser.add_argument("--exclude-tags", type=Path, default=saved["exclude_tags"])

    parser.add_argument("--confidence", type=float, default=0.5)
    parser.add_argument("--nms-iou", type=float, default=0.45)
    parser.add_argument("--distance-method", default="centroid")
    parser.add_argument("--pixel-size-mm", type=float, default=0.0666)
    parser.add_argument("--temporal-window-hours", type=float, default=8.0)

    parser.add_argument("--tracking", choices=["centroid", "simple_iou", "bytetrack"], default="centroid")
    parser.add_argument("--centroid-max-distance", type=int, default=200)
    parser.add_argument("--centroid-max-missing", type=int, default=1)

    parser.add_argument("--aruco-dictionary", default=saved["aruco_dictionary"])
    parser.add_argument("--no-aruco", action="store_true")
    parser.add_argument("--no-optimize-aruco", action="store_true")
    parser.add_argument("--aruco-profile", default=saved["aruco_profile"])
    parser.add_argument("--aruco-sample-frames", type=int, default=saved["aruco_sample_frames"])
    parser.add_argument("--aruco-max-combinations", type=int, default=saved["aruco_max_combinations"])
    parser.add_argument("--aruco-bank-size", type=int, default=saved["aruco_bank_size"])
    parser.add_argument("--aruco-workers", type=int, default=saved["aruco_workers"])
    parser.add_argument("--expected-tags", type=float, default=saved["aruco_expected_tags"])
    parser.add_argument(
        "--aruco-sweep-min-perimeter",
        type=float,
        default=saved_sweep["minMarkerPerimeterRate"][0],
    )
    parser.add_argument(
        "--aruco-sweep-max-perimeter",
        type=float,
        default=saved_sweep["maxMarkerPerimeterRate"][0],
    )
    parser.add_argument(
        "--aruco-sweep-win-min",
        type=int,
        default=saved_sweep["adaptiveThreshWinSizeMin"][0],
    )
    parser.add_argument(
        "--aruco-sweep-win-max",
        default=",".join(str(value) for value in saved_sweep["adaptiveThreshWinSizeMax"]),
        help="Comma-separated adaptiveThreshWinSizeMax values",
    )
    parser.add_argument(
        "--aruco-sweep-win-step",
        type=int,
        default=saved_sweep["adaptiveThreshWinSizeStep"][0],
    )
    parser.add_argument(
        "--aruco-sweep-polygon",
        type=float,
        default=saved_sweep["polygonalApproxAccuracyRate"][0],
    )
    parser.add_argument(
        "--aruco-sweep-constants",
        default=",".join(str(value) for value in saved_sweep["adaptiveThreshConstant"]),
        help="Comma-separated adaptiveThreshConstant values",
    )

    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--ignore-config-mismatch", action="store_true")
    parser.add_argument("--save-visualizations", action="store_true")
    parser.add_argument("--visualization-format", choices=["video", "frames"], default="video")
    parser.add_argument("--visualize-every", type=int, default=1)
    parser.add_argument(
        "--visualize-indices",
        default="",
        help="Comma-separated 1-based video positions to visualize instead of --visualize-every",
    )
    parser.add_argument(
        "--visualization-max-frames",
        type=int,
        default=0,
        help="0 means all frames. Streaming MP4 output can safely use all frames.",
    )
    parser.add_argument(
        "--keep-pollen-in-hive",
        action="store_true",
        help="Do not subtract pollen segmentation masks from hive masks.",
    )
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--limit", type=int, default=None, help="Optional first N videos from the selected source")
    parser.add_argument("--analysis-max-frames", type=int, default=None, help="Optional first N frames per video for quick CLI previews")
    return parser.parse_args()


def read_video_list(path: Path):
    videos = []
    for line in path.expanduser().read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        videos.append(str(Path(line).expanduser()))
    return videos


def parse_int_list(raw: str):
    if not raw:
        return []

    values = []
    for token in str(raw).replace(";", ",").split(","):
        token = token.strip()
        if token:
            values.append(int(token))
    return values


def build_aruco_sweep_overrides(args):
    return {
        "minMarkerPerimeterRate": [float(args.aruco_sweep_min_perimeter)],
        "maxMarkerPerimeterRate": [float(args.aruco_sweep_max_perimeter)],
        "adaptiveThreshWinSizeMin": [int(args.aruco_sweep_win_min)],
        "adaptiveThreshWinSizeMax": parse_int_list(args.aruco_sweep_win_max),
        "adaptiveThreshWinSizeStep": [int(args.aruco_sweep_win_step)],
        "polygonalApproxAccuracyRate": [float(args.aruco_sweep_polygon)],
        "adaptiveThreshConstant": parse_int_list(args.aruco_sweep_constants),
    }


def build_tracking_config(args):
    if args.tracking == "centroid":
        return {
            "algorithm": "centroid",
            "max_distance": args.centroid_max_distance,
            "max_frames_missing": args.centroid_max_missing,
        }
    if args.tracking == "simple_iou":
        return {
            "algorithm": "simple_iou",
            "iou_threshold": 0.5,
            "use_mask_iou": True,
        }
    return {
        "algorithm": "bytetrack",
        "high_conf_threshold": 0.5,
        "high_iou_threshold": 0.6,
        "low_iou_threshold": 0.3,
        "max_frames_lost": 10,
        "use_mask_iou": True,
    }


def validate_path(path: Path, label: str, required: bool = True):
    if path is None:
        return None
    resolved = path.expanduser()
    if required and not resolved.exists():
        raise FileNotFoundError(f"{label} does not exist: {resolved}")
    return resolved


def build_config(args):
    if args.video:
        video_source = [str(validate_path(args.video, "Video"))]
    else:
        video_source = read_video_list(validate_path(args.file_list, "File list"))

    if args.limit:
        video_source = video_source[: max(0, args.limit)]

    if not video_source:
        raise ValueError("No videos selected")

    order_payload = "\n".join(video_source)
    output_folder = args.output_folder.expanduser()

    return {
        "video_source": video_source,
        "folder_mode": False,
        "preserve_file_order": True,
        "selected_file_order_signature": hashlib.sha256(order_payload.encode("utf-8")).hexdigest()[:16],
        "bee_model_path": str(validate_path(args.bee_model, "Bee model")),
        "bee_model_type": "segmentation",
        "distance_method": args.distance_method,
        "hive_model_path": str(validate_path(args.hive_model, "Hive model", required=False)) if args.hive_model else None,
        "pollen_model_path": str(validate_path(args.pollen_model, "Pollen model", required=False)) if args.pollen_model else None,
        "chamber_model_path": str(validate_path(args.chamber_model, "Chamber model", required=False)) if args.chamber_model else None,
        "use_temporal_hive_prior": bool(args.hive_model),
        "temporal_hive_window_hours": args.temporal_window_hours,
        "temporal_hive_resolution": 256,
        "temporal_context_include_date": True,
        "exclude_pollen_from_hive": not args.keep_pollen_in_hive,
        "tracking_config": build_tracking_config(args),
        "confidence_threshold": args.confidence,
        "nms_iou_threshold": args.nms_iou,
        "max_frames": args.analysis_max_frames if args.analysis_max_frames and args.analysis_max_frames > 0 else None,
        "compute_spatial_metrics": True,
        "pixel_size_mm": args.pixel_size_mm if args.pixel_size_mm > 0 else None,
        "enable_aruco": not args.no_aruco,
        "aruco_dictionary": args.aruco_dictionary,
        "aruco_dictionary_mode": "single",
        "tag_list_path": str(args.tag_map.expanduser()) if args.tag_map else None,
        "exclude_tag_list_path": str(args.exclude_tags.expanduser()) if args.exclude_tags else None,
        "allowed_tag_ids": [],
        "excluded_tag_ids": [],
        "aruco_optimization": {
            "enabled": (not args.no_aruco) and (not args.no_optimize_aruco),
            "dictionary": args.aruco_dictionary,
            "profile": args.aruco_profile,
            "sample_frames": args.aruco_sample_frames,
            "max_combinations": args.aruco_max_combinations,
            "workers": args.aruco_workers,
            "bank_size": args.aruco_bank_size,
            "expected_tags": args.expected_tags if args.expected_tags > 0 else None,
            "sweep_overrides": build_aruco_sweep_overrides(args),
        },
        "output_folder": str(output_folder),
        "resume_completed_videos": args.resume,
        "resume_ignore_config_mismatch": args.ignore_config_mismatch,
        "save_visualizations": args.save_visualizations,
        "visualization_format": args.visualization_format,
        "visualization_interval": max(1, args.visualize_every),
        "visualization_video_indices": parse_int_list(args.visualize_indices),
        "visualization_max_frames": max(0, args.visualization_max_frames),
        "skip_completed_visualizations": True,
        "verbose_output": args.verbose,
    }


def main():
    args = parse_args()
    app = QCoreApplication(sys.argv)
    _ = app

    config = build_config(args)
    worker = BatchVideoInferenceWorker(config)

    worker.log_message.connect(print)
    worker.status_updated.connect(lambda message: print(f"[status] {message}"))
    worker.progress_updated.connect(lambda current, total: print(f"[progress] {current}/{total}"))

    failed = {"message": None}
    worker.inference_failed.connect(lambda message: failed.__setitem__("message", message))

    worker.run()
    if failed["message"]:
        print(f"ERROR: {failed['message']}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
