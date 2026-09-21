#!/usr/bin/env python3
"""Merge two BumbleBox batch inference output folders without double-counting videos."""

import argparse
import csv
import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple


VIDEOLESS_COPY_FROM_SECONDARY = {
    "temporal_hive_priors.csv",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Create a merged CSV output folder from two batch inference folders. "
            "Rows from the primary folder are kept for overlapping video IDs; "
            "rows from the secondary folder are added only for new video IDs."
        )
    )
    parser.add_argument("--primary-folder", type=Path, required=True)
    parser.add_argument("--secondary-folder", type=Path, required=True)
    parser.add_argument("--output-folder", type=Path, required=True)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow writing into an existing output folder. Existing merged CSVs may be replaced.",
    )
    return parser.parse_args()


def row_video_id(row: Dict[str, str]) -> str:
    video_id = (row.get("video_id") or "").strip()
    if video_id:
        return video_id
    video_path = (row.get("video_path") or "").strip()
    if video_path:
        return Path(video_path).stem
    return ""


def csv_header(path: Path) -> List[str]:
    with open(path, newline="") as handle:
        reader = csv.reader(handle)
        return next(reader, [])


def collect_video_ids(path: Path) -> Set[str]:
    if not path.exists():
        return set()
    video_ids = set()
    with open(path, newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            video_id = row_video_id(row)
            if video_id:
                video_ids.add(video_id)
    return video_ids


def count_rows_and_videos(path: Path) -> Tuple[int, int]:
    if not path.exists():
        return 0, 0
    rows = 0
    video_ids = set()
    with open(path, newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows += 1
            video_id = row_video_id(row)
            if video_id:
                video_ids.add(video_id)
    return rows, len(video_ids)


def copy_csv(source: Path, output: Path) -> Tuple[int, int, int]:
    output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, output)
    rows, videos = count_rows_and_videos(output)
    return rows, rows, videos


def merge_csv(
    primary_path: Path,
    secondary_path: Path,
    output_path: Path,
    primary_video_ids: Set[str],
) -> Dict[str, int]:
    primary_header = csv_header(primary_path)
    secondary_header = csv_header(secondary_path)
    if primary_header != secondary_header:
        raise RuntimeError(
            f"CSV headers differ for {primary_path.name}; refusing to merge ambiguous schemas."
        )

    rows_from_primary = 0
    rows_from_secondary = 0
    dropped_secondary_overlap = 0
    output_video_ids = set()

    with open(output_path, "w", newline="") as out_handle:
        writer = csv.DictWriter(out_handle, fieldnames=primary_header)
        writer.writeheader()

        with open(primary_path, newline="") as primary_handle:
            reader = csv.DictReader(primary_handle)
            for row in reader:
                writer.writerow(row)
                rows_from_primary += 1
                video_id = row_video_id(row)
                if video_id:
                    output_video_ids.add(video_id)

        with open(secondary_path, newline="") as secondary_handle:
            reader = csv.DictReader(secondary_handle)
            for row in reader:
                video_id = row_video_id(row)
                if video_id and video_id in primary_video_ids:
                    dropped_secondary_overlap += 1
                    continue
                writer.writerow(row)
                rows_from_secondary += 1
                if video_id:
                    output_video_ids.add(video_id)

    return {
        "rows_from_primary": rows_from_primary,
        "rows_from_secondary": rows_from_secondary,
        "dropped_secondary_overlap_rows": dropped_secondary_overlap,
        "merged_rows": rows_from_primary + rows_from_secondary,
        "merged_video_count": len(output_video_ids),
    }


def write_readme(output_folder: Path, summary: Dict):
    headline_stats = compute_headline_stats(output_folder)
    text_path = output_folder / "MERGE_README.txt"
    lines = [
        "BumbleBox merged batch inference outputs",
        "",
        f"Created at: {summary['created_at']}",
        f"Primary folder: {summary['primary_folder']}",
        f"Secondary folder: {summary['secondary_folder']}",
        "",
        "Merge rule:",
        "- Keep all rows from the primary folder.",
        "- Add rows from the secondary folder only when their video_id was not present in the primary folder.",
        "- Copy temporal_hive_priors.csv from the secondary folder, because it should reflect the full resumed timeline.",
        "- Do not copy aruco_optimization/ or visualizations/ subfolders.",
        "",
        "Headline analysis totals:",
    ]
    lines.extend(headline_stats)
    text_path.write_text("\n".join(lines) + "\n")


def truthy(value: str) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def int_value(value, default: int = 0) -> int:
    try:
        if value in (None, ""):
            return default
        return int(float(value))
    except (TypeError, ValueError):
        return default


def compute_headline_stats(output_folder: Path) -> List[str]:
    stats = {
        "videos_analyzed": 0,
        "frames": 0,
        "bee_detections": 0,
        "aruco_observations": 0,
        "accepted_aruco_observations": 0,
        "bee_contact_events": 0,
        "bee_detections_on_pollen_balls": 0,
        "bee_detections_on_temporal_hive_prior": 0,
    }

    status_path = output_folder / "batch_video_status.csv"
    if status_path.exists():
        video_ids = set()
        with open(status_path, newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                video_id = row_video_id(row)
                if video_id:
                    video_ids.add(video_id)
                stats["frames"] += int_value(row.get("frame_count"))
        stats["videos_analyzed"] = len(video_ids)

    bee_path = output_folder / "bee_detections.csv"
    if bee_path.exists():
        with open(bee_path, newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                stats["bee_detections"] += 1
                if truthy(row.get("on_pollen_ball")):
                    stats["bee_detections_on_pollen_balls"] += 1
                if truthy(row.get("on_temporal_hive")):
                    stats["bee_detections_on_temporal_hive_prior"] += 1

    aruco_path = output_folder / "aruco_observations.csv"
    if aruco_path.exists():
        with open(aruco_path, newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                stats["aruco_observations"] += 1
                if truthy(row.get("accepted")):
                    stats["accepted_aruco_observations"] += 1

    interaction_path = output_folder / "bee_interactions.csv"
    if interaction_path.exists():
        with open(interaction_path, newline="") as handle:
            reader = csv.DictReader(handle)
            for _row in reader:
                stats["bee_contact_events"] += 1

    return [
        f"videos analyzed: {stats['videos_analyzed']:,}",
        f"total analyzed frames: {stats['frames']:,}",
        f"yolo bee detections: {stats['bee_detections']:,}",
        f"aruco observations decoded/matched: {stats['aruco_observations']:,}",
        f"accepted aruco observations: {stats['accepted_aruco_observations']:,}",
        f"bee contact events: {stats['bee_contact_events']:,}",
        f"bee detections on pollen balls: {stats['bee_detections_on_pollen_balls']:,}",
        f"bee detections on temporal hive prior: {stats['bee_detections_on_temporal_hive_prior']:,}",
    ]


def main() -> int:
    args = parse_args()
    primary = args.primary_folder
    secondary = args.secondary_folder
    output = args.output_folder

    if output.resolve() in (primary.resolve(), secondary.resolve()):
        raise SystemExit("Output must be a separate folder from both input folders.")

    if not primary.exists():
        raise SystemExit(f"Primary folder does not exist: {primary}")
    if not secondary.exists():
        raise SystemExit(f"Secondary folder does not exist: {secondary}")
    if output.exists() and any(output.iterdir()) and not args.overwrite:
        raise SystemExit(
            f"Output folder exists and is not empty: {output}\n"
            "Use --overwrite if this is intentional."
        )

    output.mkdir(parents=True, exist_ok=True)

    primary_csvs = {path.name for path in primary.glob("*.csv")}
    secondary_csvs = {path.name for path in secondary.glob("*.csv")}
    csv_names = sorted(primary_csvs | secondary_csvs)
    primary_video_ids = collect_video_ids(primary / "batch_video_status.csv")
    if not primary_video_ids:
        primary_video_ids = collect_video_ids(primary / "bee_detections.csv")

    summary = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "primary_folder": str(primary),
        "secondary_folder": str(secondary),
        "output_folder": str(output),
        "primary_video_count": len(primary_video_ids),
        "csv_files": {},
    }

    for name in csv_names:
        primary_path = primary / name
        secondary_path = secondary / name
        output_path = output / name

        if name in VIDEOLESS_COPY_FROM_SECONDARY and secondary_path.exists():
            copied_rows, copied_source_rows, copied_videos = copy_csv(secondary_path, output_path)
            summary["csv_files"][name] = {
                "strategy": "copied_from_secondary",
                "copied_rows": copied_rows,
                "copied_source_rows": copied_source_rows,
                "copied_video_count": copied_videos,
            }
            continue

        if primary_path.exists() and secondary_path.exists():
            summary["csv_files"][name] = {
                "strategy": "primary_plus_secondary_nonoverlap",
                **merge_csv(primary_path, secondary_path, output_path, primary_video_ids),
            }
        elif primary_path.exists():
            copied_rows, copied_source_rows, copied_videos = copy_csv(primary_path, output_path)
            summary["csv_files"][name] = {
                "strategy": "copied_from_primary",
                "copied_rows": copied_rows,
                "copied_source_rows": copied_source_rows,
                "copied_video_count": copied_videos,
            }
        elif secondary_path.exists():
            copied_rows, copied_source_rows, copied_videos = copy_csv(secondary_path, output_path)
            summary["csv_files"][name] = {
                "strategy": "copied_from_secondary",
                "copied_rows": copied_rows,
                "copied_source_rows": copied_source_rows,
                "copied_video_count": copied_videos,
            }

    with open(output / "merge_summary.json", "w") as handle:
        json.dump(summary, handle, indent=2)
    write_readme(output, summary)

    print(f"Merged outputs written to: {output}")
    print(f"Primary video IDs protected from duplication: {len(primary_video_ids)}")
    for name, data in sorted(summary["csv_files"].items()):
        print(
            f"{name}: rows={data.get('merged_rows', data.get('copied_rows', 0)):,}, "
            f"videos={data.get('merged_video_count', data.get('copied_video_count', 0)):,}, "
            f"strategy={data['strategy']}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
