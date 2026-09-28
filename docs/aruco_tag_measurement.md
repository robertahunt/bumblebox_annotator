# ArUco Tag Size Measurement

Open **Model > Batch Video Inference with Tracking**, choose a specific ArUco
dictionary, then click **Measure tag bounds...** next to the perimeter sweeps.
With one video selected, that video opens directly. For a folder or multiple
videos, choose a representative video in the file picker.

1. Select **Smallest tag** and click its four outer black-border corners in
   clockwise or counterclockwise order. Do not measure its internal grid or the
   surrounding white margin.
2. Select **Largest tag** and repeat. The two measurements may use different
   frames in the same video. The frame number shown in the dialog is one-based.
3. Scroll to zoom; right-drag or middle-drag to pan. Drag a corner to adjust it.
   The back-arrow beside the measurement selectors removes the last corner;
   the trash button clears that measurement. The top arrows navigate frames.
4. Check the perimeter rates and the margin, then click **Apply to batch**.
   Cancel leaves the batch configuration unchanged.

The perimeter is the sum of the four sides in original image pixels, divided
by the longest original frame dimension. Zoom does not affect the measurement.
With the default 10% margin:

```text
minMarkerPerimeterRate = smallest measured rate * 0.90
maxMarkerPerimeterRate = largest measured rate * 1.10
```

Crossed/degenerate quadrilaterals and reversed smallest/largest measurements
cannot be applied. The margin can be adjusted before applying.

Applying replaces the minimum and maximum perimeter sweep fields and enables
parameter-bank optimization. It does not change the other sweep fields, start
inference, or modify the source video. The bounds use the existing batch config
and settings persistence when the batch is started.

**Scope:** these are batch-wide bounds, not automatic per-video calibration.
Use separate batches or remeasure when the imaging setup or relative tag size
changes. OpenCV candidate filtering and the optimizer's decoded-marker size
check now use the same longest-dimension normalization as the measurement tool.
Tag allowlists and exclusions remain separate settings.

The viewer retains only the current decoded frame. For MJPEG streams without
a usable frame count, next-frame navigation reads sequentially; earlier frames
are reread from the beginning. The frame-number control can revisit frames
already reached without scanning the entire video on open.

## Implementation

- `gui/aruco_measurement_dialog.py`: Qt measurement view and dialog.
- `core/aruco_measurement.py`: image-space geometry and video reader.
- `gui/batch_video_inference_dialog.py`: batch settings integration.

This is independent of the separate BumbleBox checkout. The original Tk tool
is `BumbleBox/bumblebox_v2/gui_app.py::_open_tag_perimeter_measurement_dialog`;
its suggested rates are calculated in `bumblebox_v2/tracking_optimizer.py`.
The annotator retains the 10% tolerance convention but does not copy the
original helper's fixed camera-specific rate caps.

Run synthetic checks with the annotator environment:

```bash
python scripts/test_aruco_measurement.py
```
