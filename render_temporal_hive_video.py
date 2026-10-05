#!/usr/bin/env python3
"""Render cached hive maps with their recorded timing, without model inference."""

import argparse
import math
import os
from pathlib import Path
import tempfile

import cv2

from core.temporal_hive_visualization import (
    TemporalHiveOverlayReader, draw_temporal_hive_overlay, draw_hive_overlay_label,
)


def render_video(cache_path, output_path, video_path=None, fps=None, overwrite=False):
    cache_path, output_path = Path(cache_path), Path(output_path)
    if output_path.suffix.lower() != '.mp4':
        raise ValueError('Output must have an .mp4 extension')
    if output_path.exists() and not overwrite:
        raise FileExistsError(f'Output exists; choose a new path or use --overwrite: {output_path}')

    with TemporalHiveOverlayReader(cache_path) as reader:
        metadata = reader.metadata
        source = Path(video_path or metadata['video_path']).expanduser()
        if source.name != Path(metadata['video_path']).name:
            raise ValueError('Source filename does not match this video cache')
        if output_path.resolve() in (source.resolve(), cache_path.resolve()):
            raise ValueError('Output must not overwrite the source video or cache')
        count = int(metadata['frame_count'])
        if count < 1:
            raise ValueError('The archive contains no visualized frames')
        capture = cv2.VideoCapture(str(source))
        writer = None
        temporary_path = None
        try:
            if not capture.isOpened():
                raise ValueError(f'Could not open source video: {source}')
            output_fps = float(fps if fps is not None else
                               (metadata.get('fps') or capture.get(cv2.CAP_PROP_FPS)))
            if not math.isfinite(output_fps) or output_fps <= 0:
                raise ValueError('Source FPS is unavailable; supply --fps explicitly')
            output_path.parent.mkdir(parents=True, exist_ok=True)
            fd, name = tempfile.mkstemp(suffix='.mp4', prefix='.temporal-', dir=output_path.parent)
            os.close(fd)
            temporary_path = Path(name)
            height, width = metadata['frame_shape']
            writer = cv2.VideoWriter(str(temporary_path), cv2.VideoWriter_fourcc(*'mp4v'),
                                     output_fps, (width, height))
            if not writer.isOpened():
                raise RuntimeError('Could not initialize MP4 output')
            for frame_number in range(1, count + 1):
                ok, frame = capture.read()
                if not ok:
                    raise ValueError(f'Source ended before cached frame {frame_number}')
                if list(frame.shape[:2]) != metadata['frame_shape']:
                    raise ValueError('Source dimensions do not match the temporal cache')
                supported = draw_temporal_hive_overlay(frame, reader.read(frame_number))
                draw_hive_overlay_label(frame, reader.overlay_mode, supported)
                writer.write(frame)
            writer.release()
            writer = None
            temporary_path.replace(output_path)
            temporary_path = None
        finally:
            capture.release()
            if writer is not None:
                writer.release()
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)
    return count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--video', type=Path, help='Relocated original video; defaults to the cached source path')
    parser.add_argument('--fps', type=float, help='Override output playback FPS')
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()
    try:
        count = render_video(args.cache, args.output, args.video, args.fps, args.overwrite)
    except (ValueError, OSError, RuntimeError, KeyError) as exc:
        parser.exit(1, f'ERROR: {exc}\n')
    print(f'Rendered {count} cached frames without model inference: {args.output}')


if __name__ == '__main__':
    main()
