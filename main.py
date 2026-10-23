#!/usr/bin/env python3
"""
Bee Annotator - Video Instance Segmentation with Human-in-the-Loop Training
Main application entry point
"""

import os
import sys
import argparse

# Set PyTorch CUDA memory allocator config BEFORE any CUDA/PyTorch imports
# This prevents memory fragmentation during training
os.environ['PYTORCH_ALLOC_CONF'] = 'expandable_segments:True'

from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import Qt

from gui.main_window import MainWindow
from gui.project_sync_dialog import choose_contributor
from gui.shared_profiles import stop_profile_sync


def parse_args():
    parser = argparse.ArgumentParser(description='Bee Annotator - Video Instance Segmentation Tool')
    parser.add_argument('--video', type=str, help='Path to video file to load on startup')
    parser.add_argument('--project', type=str, help='Path to project directory to load on startup')
    parser.add_argument('--sam2-checkpoint', type=str, help='Path to SAM2 checkpoint (.pt) to load on startup')
    parser.add_argument('--coarse-yolo-checkpoint', type=str, help='Path to coarse-grained YOLO checkpoint (.pt) to load on startup')
    parser.add_argument('--bbox-checkpoint', type=str, help='Path to YOLO BBox detection checkpoint (.pt) to load on startup')
    parser.add_argument('--instance-focused-checkpoint', type=str, help='Path to instance-focused YOLO checkpoint (.pt) to load on startup')
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Enable high DPI scaling
    QApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )
    
    app = QApplication(sys.argv)
    app.setApplicationName("Bee Annotator")
    app.setOrganizationName("BeeWhere")

    session = choose_contributor()
    if session is None:
        stop_profile_sync()
        return
    
    try:
        # Create main window with optional checkpoints
        window = MainWindow(
            sam2_checkpoint=args.sam2_checkpoint,
            coarse_yolo_checkpoint=args.coarse_yolo_checkpoint,
            bbox_checkpoint=args.bbox_checkpoint,
            instance_focused_checkpoint=args.instance_focused_checkpoint
        )
        window.set_contributor(session)

        # Load project if specified
        if args.project:
            window.load_project(args.project)
        elif not args.video and window.workflow.home.auto_resume.isChecked():
            window.workflow.resume()

        # A command-line video still goes through the existing import workflow.
        if args.video:
            window.load_video(args.video)

        window.show()
        sys.exit(app.exec())
    finally:
        # Startup may fail before QApplication.exec(), when aboutToQuit cannot run.
        stop_profile_sync()


if __name__ == '__main__':
    main()
