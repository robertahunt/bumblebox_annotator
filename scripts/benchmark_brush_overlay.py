"""Synthetic CPU-side stroke/overlay benchmark; no projects or files are modified.

Run from the repository with the annotator Python environment. This excludes
screen presentation, tablet input, saving, and model inference.
"""

import argparse
import os
from pathlib import Path
import statistics
import sys
import time

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from PyQt6.QtCore import QPointF
from PyQt6.QtWidgets import QApplication
from gui.canvas import ImageCanvas


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--width', type=int, default=4096)
    parser.add_argument('--height', type=int, default=3072)
    parser.add_argument('--steps', type=int, default=60)
    args = parser.parse_args()
    if min(args.width, args.height) < 100 or args.steps < 1:
        parser.error('Use dimensions >= 100 and at least one step.')
    app = QApplication.instance() or QApplication([])
    canvas = ImageCanvas()
    canvas.current_image = np.zeros((args.height, args.width, 3), np.uint8)
    canvas.editing_mask = np.zeros((args.height, args.width), np.uint8)
    canvas.editing_mask[args.height // 4:3 * args.height // 4, args.width // 4:3 * args.width // 4] = 255
    canvas.editing_instance_id = canvas.selected_mask_idx = 1
    canvas.editing_instance_category = canvas.selected_instance_category = canvas.current_annotation_type = 'bee'
    canvas.show_bboxes = True
    canvas._update_editing_visualization()
    elapsed = []
    for index in range(args.steps + 5):
        canvas.current_tool = 'brush' if index % 2 else 'eraser'
        x, y = args.width // 2 + index % 20, args.height // 2
        start = time.perf_counter()
        canvas.draw_on_mask(QPointF(x, y), QPointF(x + 2, y + 1))
        canvas._viz_update_timer.stop()
        canvas._flush_pending_viz_update()
        duration = (time.perf_counter() - start) * 1000
        if index >= 5:
            elapsed.append(duration)
    print(f'{args.width} x {args.height}, {args.steps} updates, brush + eraser, bounding boxes on')
    print(f'Median: {statistics.median(elapsed):.2f} ms | p95: {np.percentile(elapsed, 95):.2f} ms')
    print('CPU-side stroke + overlay update only; not end-to-end tablet latency.')
    canvas.close()
    app.processEvents()


if __name__ == '__main__':
    main()
