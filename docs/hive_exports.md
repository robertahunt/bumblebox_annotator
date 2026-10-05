# Hive Masks in Batch Inference

The batch pipeline has three distinct representations of hive. With temporal
maps enabled, new runs default to scoring on history plus current evidence and
displaying that same map. Raw detections and past-only scoring remain available.

| Representation | Meaning |
| --- | --- |
| Current-frame hive mask | The current frame's YOLO hive mask, with detected pollen removed when pollen exclusion is enabled. This can fluctuate between frames. |
| `hive_detections.csv` | One majority-vote mask summary per video/chamber, reporting area in original-image pixels and the resulting centroid. A pixel must appear in more than half of the accumulated masks. |
| Temporal hive map | A rolling chamber-normalized estimate combining history and current usable evidence. Bee contact is evaluated after the current update by default, or before it in legacy past-only mode. It is not a fixed mask for the whole video. |

`bee_detections.csv` contains the frame-specific temporal overlap/status fields.
`temporal_hive_priors.csv` contains the final map summaries per context/chamber,
not a snapshot for every frame. Current-frame hive distance and historical hive
overlap remain separate measurements.

## Choosing Contact Scoring

Under the temporal hive controls, choose **Hive contact scoring**:

- **History + current evidence** (default): detect the current hive and bees,
  incorporate usable current hive evidence into the rolling map, then calculate
  `on_temporal_hive` and `temporal_hive_*` overlap/support fields from that updated
  map. The default overlay shows the same map used for these scores.
- **Past-only map (legacy)**: calculate those scores before adding the current
  evidence, preserving the previous scoring order.

Both full batch inference and hive-only refresh expose
`--temporal-hive-scoring updated|prior`, defaulting to `updated`. Each frame
contributes exactly once regardless of scoring mode, preview settings or replay.
Only past and current evidence is used, never future frames.

Bee-covered pixels remain excluded from current evidence. Their estimates come
from history, so many bee overlap scores may stay the same when switching modes;
the updated map is not allowed to learn apparent hive absence from a covering
bee. Support thresholds and unknown results remain in effect. This is a weighted
estimate, not ground truth or a simple union of current and historical masks.

The CSV column names are unchanged; scoring mode is recorded in run configuration,
checkpoints and new overlay archives. Older checkpoints without this setting are
treated as past-only, never silently upgraded. Updated scoring changes the
analysis signature and rejects incompatible checkpoints. Use a **fresh output
folder** for existing past-only datasets; do not force a resume mismatch override.
Raw `distance_to_hive_*` fields and majority-mask exports retain their existing
current-detection semantics, rather than being silently relabeled as revised-map
measurements.

## Choosing an Overlay

In **Batch Video Inference with Tracking**, provide a hive model and enable the
temporal hive prior. Under **Output Options**, enable **Generate annotated
visualizations**, then choose **Hive overlay**:

- **Match contact scoring** (default): displays the temporal map selected for
  scoring; falls back to raw detections only when temporal maps are disabled.
- **Current frame detections**: the existing YOLO overlay, available for inspection.
- **History + current evidence**: a light teal fill and outline of the updated
  estimate after incorporating the current frame's usable evidence. This is not
  ground truth, and it is not simply a union of every past detection.
- **Temporal hive prior (past only)**: a light teal fill and thin outline of historically
  supported hive. The map is projected from normalized chamber coordinates into
  the current frame's chamber bounding box.
- **Compare prior and current**: the same historical overlay with a magenta
  outline of the current-frame hive detection.

The past-only/comparison overlay is captured **before updating with the current
frame**. The history-plus-current overlay is captured **after updating**. Both
use the same probability and evidence-weight thresholds as temporal bee scoring;
neither uses future frames or the final batch map to paint earlier frames. The
current frame's bee-occluded pixels are still excluded from new evidence, and
evidence must still meet the support threshold. Missing/empty hive detections
retain the existing update rules. Unsupported regions remain clear, with an
insufficient-history/evidence footer when nothing is supported.
Clear pixels can mean either known non-hive or insufficient
evidence, not necessarily "no hive". Current-frame hive pixel counts in the
diagnostic panel remain current-frame measurements and are labeled accordingly.
Any separate `hive_detection_summary.png` also remains a current-frame image.

Selecting a different overlay does not change the scoring mode. An explicitly
chosen comparison or past-only overlay can therefore differ from the scoring
map; **Match contact scoring** keeps them aligned. Bee/pollen overlays and
tracking are unchanged. Previously saved display choices are preserved, so
select **Match contact scoring** when reopening an existing GUI configuration.

For the batch CLI, append these options to an otherwise complete command:

```bash
--temporal-hive-scoring updated --save-visualizations --hive-overlay scored --visualization-max-frames 0
```

Use `--hive-overlay temporal` for past-only or `compare` for comparison. Explicit
history views require a hive model; `scored` can also run without temporal maps. In the GUI,
set **Frames per visualization** to **All frames** for a complete video; the
existing GUI default is only 25 preview frames. Temporal and comparison MP4s use
`annotated_videos/<video_id>_temporal_annotated.mp4` and
`annotated_videos/<video_id>_compare_annotated.mp4`, respectively. Updated videos
use `<video_id>_updated_annotated.mp4`. PNG sequences use
`visualizations/<video_id>_temporal/` or `visualizations/<video_id>_compare/`.
The updated mode uses `visualizations/<video_id>_updated/`.

## Stabilizing Temporal Measurements

Enable **Stabilize temporal hive placement and measurements** beside the temporal
prior controls, or add `--stabilize-temporal-hive` to either batch CLI. This is
opt-in; existing settings and results are unchanged by default.

The coordinate transform into each chamber grid is smoothed, not just the drawn
outline. The same stabilized box is used for new hive evidence, bee-occlusion
exclusion, temporal bee overlap scoring, and overlay projection. Raw YOLO boxes,
masks, chamber assignment and current-frame hive distance/area metrics retain
their existing semantics.

The first implementation uses a causal exponential average of small bounding-box
changes: 20% current box and 80% preceding smoothed box. It resets at each video,
changed image dimensions or chamber set, and follows a displacement exceeding
2% of chamber width/height immediately. When a chamber model falls back to the
whole image after failing detection, stabilized temporal scoring/updating is
skipped and its overlay is unknown. Without a chamber model, the fixed full-frame
coordinate system remains supported.

This addresses detector-placement jitter on stationary-camera recordings. It is
not optical image registration, cannot guarantee correct chamber identities after
detection failures, and can lag small genuine camera/chamber motion. The binary
map can still change as evidence crosses thresholds; coarse-grid stair steps
are not removed. Validate a representative preview before a full analysis run.

Stabilization **changes temporal measurements**, unlike selecting a different
overlay timing. It is included in analysis signatures and checkpoint settings;
mismatched checkpoint stabilization is rejected. Start a fresh output folder,
and do not force a resume mismatch override to mix old and stabilized results.
Per-video smoothing resets mean no geometry history needs carrying across a
video-boundary checkpoint.

## Re-rendering Without Models

When temporal priors and visualizations are enabled, new batch runs also save
`temporal_hive_overlays/<video_id>.zip`, even when viewing current detections.
The updated mode saves `<video_id>_updated.zip` separately.
There is one compressed archive per visualized video, not thousands of loose
files. Only frames selected for visualization are cached. Disabling previews or
skipping a video does not create a cache. Analysis still uses all analyzed frames.

The archive contains categorical grids (unknown / supported non-hive /
supported hive), chamber bounding boxes, source dimensions/path/FPS, prior
settings, scoring mode, explicit before/after-update timing and provenance. It does not contain
source images or model checkpoints. The renderer reads timing from the archive:
old past-only caches are never silently relabeled as history-plus-current.
Streaming writes each frame immediately, so RAM does not grow with video length.
Disk use depends on grid resolution, number of chambers and cached frames.

From the repository root, with the annotator environment activated:

```bash
python render_temporal_hive_video.py \
  --cache "/path/to/output/temporal_hive_overlays/VIDEO_ID_updated.zip" \
  --output "/path/to/output/VIDEO_ID_temporal_replay.mp4"
```

These are placeholder paths. This command reads the original video and saved
maps using OpenCV/NumPy, without loading YOLO, SAM2, ArUco or a GPU. It draws the
historical hive overlay only, not bee boxes, IDs or trails. Supply `--video` if
the original video moved (same filename and dimensions), `--fps` to override
playback speed, or `--overwrite` to explicitly replace an existing render. A
cache finalized after interruption can render its recorded prefix. It cannot
recover uncached frames or repair an archive cut off by a hard process crash.

Changing only the visualization mode does not change scientific results. Resume still
skips completed videos; choosing a new overlay does not automatically rebuild
their previews. Use the cache renderer instead.

## Rebuilding Older History

Older runs did not save per-frame historical maps. `temporal_hive_priors.csv`
contains only final simplified polygons, and a final prior checkpoint is not a
time series either. Neither can faithfully show the history used on each earlier
frame. A static final-map overlay would be retrospective, not a past-only prior.

To rebuild video history without repeating bee tracking or ArUco, the existing
`batch_hive_refresh_cli.py` now accepts `--save-temporal-overlays`. It reruns
chamber/hive/pollen inference and replays saved bee detections for occlusion,
writing the same replay archives. Use an **ordered video list**, the necessary
earlier videos within each context, the original settings/models, and a **new
output folder**. Starting with only a later video loses its earlier context.

For example (replace all paths):

```bash
python batch_hive_refresh_cli.py \
  --source-output-folder "/path/to/existing/results" \
  --file-list "/path/to/ordered_videos.txt" \
  --output-folder "/path/to/new/hive_refresh" \
  --hive-model "/path/to/hive.pt" \
  --chamber-model "/path/to/chamber.pt" \
  --pollen-model "/path/to/pollen.pt" \
  --temporal-window-hours 8 \
  --temporal-resolution 256x256 \
  --save-temporal-overlays \
  --temporal-hive-scoring updated \
  --temporal-overlay-timing scored \
  --stabilize-temporal-hive
```

The 8-hour, 256x256 settings match the older pesticide batch's temporal grid;
they are not universal recommendations. In particular the refresh CLI's default
resolution is different, so specify it explicitly. If the original run had no
pollen model, use `--no-pollen-model` instead. Then run the cache renderer above.
For an unstabilized, past-only reconstruction, omit `--stabilize-temporal-hive`
and use `--temporal-hive-scoring prior --temporal-overlay-timing scored`.
An explicit `--temporal-overlay-timing prior|updated` can cache a comparison
view without changing scoring. Stabilized or updated maps cannot be
recovered exactly from an old categorical past-only cache: reconstruct them
from source evidence instead.

This is a reconstruction, not guaranteed bit-for-bit reproduction: saved bee
polygons can be simplified compared with the original full masks, and changed
models/settings change the evidence. Original videos must remain accessible.
Resume skips completed refresh videos too; it will not backfill caches for old
completed rows. No old research outputs are modified by adding this feature.

## Export Accumulation

Hive and chamber export counts are accumulated during inference, independently
of visualization settings. Disabling previews, limiting annotated frames,
streaming annotated output, or clearing visualization caches does not remove
these counts. Stopping a video retains summaries for the frames already
processed. Prior-only resume replays do not add duplicate export counts.

The processor keeps one full-resolution unsigned 32-bit count array per
video/chamber/mask type, not every frame's mask. Memory therefore does not grow
with the number of analyzed frames. Counts are released by the batch worker
after a successful CSV flush. The chamber CSV also averages detected centroids.

Existing mask semantics are unchanged: a missing mask (`None`) is skipped;
an available all-zero mask contributes a frame with no positive pixels. If no
usable masks are available, a header-only export is still possible. These
majority summaries are not occlusion-corrected by the temporal map.

## Older Outputs

Earlier batch code accumulated these summaries from visualization caches. This
could leave hive/chamber CSVs header-only when previews were disabled or
streamed, or summarize only the stored preview frames. Temporal-prior metrics
were calculated separately and did not depend on those caches.

The fix applies to newly processed videos. Resuming a run still skips videos
marked complete; it does not repair their earlier CSV rows. Preserve old results
and reprocess into a separate output folder to regenerate complete summaries.
Do not substitute the final temporal prior for a per-video majority mask: they
describe different data and coordinate systems.

Synthetic regression checks:

```bash
python scripts/test_batch_mask_exports.py
python scripts/test_temporal_hive_visualization.py
```
