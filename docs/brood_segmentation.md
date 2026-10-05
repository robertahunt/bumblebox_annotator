# Experimental Brood Segmentation

Labels describe **visible brood-cell surfaces by appearance**. They do not claim
to observe closed-cell contents, establish biological age, or identify successive
cohorts. Temporal maps are derived outputs, never manual training annotations.

## Labeling

Use **New Instance** (toolbar or canvas context menu) and choose a brood category.
The editor switches to Brush and enables that category. The **Brood** visibility
menu contains category switches; sidebar instance switches and the global overlay
toggle also apply. Brush, eraser, fill, undo, category changes, and counts include
these new instances.

| Category | Appearance estimate | Stable machine name |
| --- | --- | --- |
| Early brood: eggs | Eggs | `brood_early` |
| Uncertain: eggs to larvae | Late eggs to early larvae | `brood_early_middle` |
| Middle brood: larvae | Larvae | `brood_middle` |
| Uncertain: larvae to pupae | Late larvae to early pupae | `brood_middle_late` |
| Late brood: pupae | Pupae | `brood_late` |

- Paint visible pixels only; exclude bee-covered or otherwise hidden surfaces.
- Hive and brood masks can overlap independently, with different instance IDs.
- One visible surface should have one brood stage. Training rejects overlapping
  different brood-stage labels, but permits overlap with hive annotations.
- Brood is always frame-specific, including legacy video-wide-hive projects.
  Existing category IDs are unchanged; the new categories are appended.
- After copying a mask to another training frame, review that frame and remove
  pixels that are now hidden. Copying is not temporal reconstruction.

## Training

1. Select training and validation frames, preferably from independent videos.
2. Review **all visible brood stages** in each selected frame. Exclude unfinished
   frames; unlabeled visible brood otherwise becomes training background.
3. Open **Train YOLO Model**, select **Brood (5 appearance classes)**, confirm the
   review checkbox, and leave **Export COCO Annotations** enabled.
4. Train a separate brood checkpoint alongside the existing independent models.

The five-class model uses class indices 0..4 in table order. Source JSON, COCO,
and raster training labels preserve holes and disconnected visible pieces.
Images without any brood segmentation are currently skipped, consistent with
existing segmentation training. The review checkbox is an attestation, not an
automated completeness check or per-pixel ignore system. Include examples of
every stage in train and validation before interpreting per-class scores.

## Batch Inference

In **Batch Video Inference**, choose a trained checkpoint under **Brood Model
(experimental)**. The normal bee model supplies occlusion masks, with conservative
bounding-box fallback. A chamber model supplies normalized chamber coordinates.
Use an **empty output folder**, with **Resume disabled**. Brood replay/resume is
not implemented; existing output is refused rather than silently skipping history.
Runs without a brood model keep their existing resume behavior.

The equivalent optional CLI argument, appended to a
`batch_video_inference_cli.py` command using a fresh output folder, is:

```bash
--brood-model "/path/to/brood_training/weights/best.pt"
```

The existing **Temporal prior window** controls evidence expiry (default 8 hours).
The GUI uses a 256 x 256 grid per chamber; CLI `--temporal-resolution` changes it.
Grid cells are not original-image pixels or square millimeters.

### History Rules

- Current usable evidence updates history **before** bee-overlap measurements.
- Six channels store background and the five classes; class indices are never
  averaged as ages. Evidence decays with elapsed time and support is capped at
  eight observations so long recordings cannot permanently freeze a stage.
- A cell needs 1.5 effective observations to become known. Unseen, stale,
  unsupported and unavailable-chamber regions are unknown. Observations older
  than the time window cease to count as known even with residual weight.
- Known cells with brood probability at least 0.5 are brood. Assigning a stage
  needs conditional support of at least 0.6; otherwise it is **stage unresolved**.
  This algorithmic uncertainty is distinct from the two transitional labels.
- Bee-covered pixels do not add absence evidence. A successful empty prediction
  does add negative evidence in unoccluded pixels; missing inference does not.
  Contradictory overlapping stage predictions withhold evidence at those pixels.

Videos are sorted chronologically unless selected-file ordering was explicitly
requested. Backwards time within a shared context raises an error. Recognized
`bumblebox-N` contexts with parsed timestamps share history across videos; other
names or missing timestamps keep video-local history. Not all legacy filename
formats have timestamps recognized by the existing parser. Limit each batch to
one consistent recording study/setup.

Alignment uses chamber boxes/masks, not full image registration. If existing hive
stabilization is enabled, brood uses those stabilized coordinates. Missing chamber
detections withhold evidence; changing chamber counts within a video resets the
context history. Camera rotation, large moves, and changed chamber identities
still require review or separate runs. The thresholds are engineering starting
points, not biologically calibrated transition rates. A change to an earlier
stage could mean new brood above old brood, not reversal of the same cohort.

### Outputs

Files go under the batch output's `brood/` folder:

- `<video>_brood_maps.zip`: one NPZ record per frame/chamber plus `metadata.json`.
  Stores observed labels, updated labels, six-channel probabilities, evidence
  weight, last observation time, chamber box and frame time. Frames are one-based
  like batch CSVs. Metadata marks incomplete versus completed runs.
- `<video>_brood_overlap.csv`: emitted when tracked bees occur in chamber boxes.
  Fractions use **known grid cells within the normalized bee footprint** as
  denominator. `known_fraction` reports coverage; no coverage gives blank scores,
  not zero contact. These are 2-D overlap estimates, not verified interactions.
  `tracker_bee_id` is the online ID before later ArUco reassignment, not necessarily
  the final biological ID.
- `<video>_brood_annotated.mp4`: for videos selected for annotated output, respecting
  the preview frame limit. It displays the same updated map used for overlap.
  Gray is unresolved stage; unknown/background is untinted. This separate preview
  is an MP4 even when the main visualization format is PNG frames.

Observed label values: 0 background, 1..5 stages, 255 unobserved.
Updated label values: 0 unknown, 1 background, 2..6 stages, 7 unresolved stage.
Probability channels: background, then the five stages in table order.
Archives avoid thousands of loose files but may still be large for long videos.

## Current Limits

Synthetic tests cover editing/visibility, source round-trips, overlapping hive
masks, five-class training/checkpoint reload, occlusion, stage changes, expiry,
ordering, and map/preview export. No real brood checkpoint or biological ground
truth has been evaluated. The bee/pollen validation GUI has not gained a brood
dashboard. Dated cohorts, chronological-age prediction, and stacked-generation
identities remain future experimental work.
