# Arena Review and Image Registration (WIP)

Status: shared registration and mask helpers extracted; annotation branches
combined on `integration/annotation-batch`. Full review-tool integration remains WIP.
This folder documents the work rather than providing a new application
entrypoint. Existing local review tools continue to run in their current location
and now delegate registration math to `core/registration.py` and enclosed-region
selection to `core/mask_editing.py`.

## Implemented

- `core/registration.py`: labeled coordinate validation, shared visible-point
  selection, perspective fitting with optional three-point affine fallback,
  robust outlier handling, and per-landmark error diagnostics.
- Landmark labels and groups are caller-defined; the shared module has no
  experiment paths, arena names, model dependencies, or private-package imports.
- `tests/test_registration.py`: small synthetic fixtures covering perspective
  motion, independent groups, obscured points, bad clicks, degenerate geometry,
  and error reports. They run with standard-library unittest, NumPy, and OpenCV.
- The local review workflow imports these helpers without changing saved
  reference formats, landmark conventions, or its command-line entrypoints.
- The batch CLI now parses temporal-map resolution in `build_config`, fixing the
  undefined-variable error when building inference settings.
- `core/mask_editing.py`: enclosed fill/subtract selection shared by the main
  annotation canvas and the local arena editor. Protection zones cannot seal an
  open outline and their pixels are never changed by Fill or Subtract.
- The combined annotation branch retains No Draw Zones, measurement/calibration,
  imaging setups, bbox editing, validation review, and newer batch workflows.
  See [branch reconciliation](branch_reconciliation.md) for tests and launch steps.

Run the shared tests from the repository root:

```bash
python -m unittest discover -s tests -p test_registration.py -v
```

## Purpose

Make reusable review and editing tools available to the annotation application:

1. Identify intervals when a chamber is stationary, moved, or absent.
2. Review representative images and correct the usable arena boundaries.
3. Mark corresponding physical landmarks to align images across positions.
4. Aggregate model masks in aligned coordinates and review the resulting masks.
5. Record which inputs and corrected masks were approved for downstream use.

These tools should accept user-selected input, output, and model paths. They
must work without experiment-specific analysis code or private datasets.

## Candidates for Extraction

| Component | Reusable behavior | Changes needed before integration |
| --- | --- | --- |
| Chamber-position review | Image contact sheets, proposed movement/removal intervals, editable reference times, review status | Configurable image discovery, date selection, and arena roles; manual review of automatic proposals |
| Arena reference editor | Polygon replacement, brush/eraser corrections, separate arena masks, original model masks retained | Share annotation controls and allow configurable arena labels |
| Landmark registration | Corresponding landmarks, obscured-point handling, separate transforms for independently moving arenas, alignment diagnostics | Configure landmark names and groups rather than assuming one chamber design |
| Registered mask review | Combine evidence across images, subtract an exclusion mask, adjust support thresholds, edit and approve masks | Configurable model classes and support settings; address pose-specific limitations below |
| Review and resume state | Atomic metadata saves, input/model/mask fingerprints, approval invalidation after edits, compatible-result skipping | Document versioned formats and reusable validation rules |

Keep generic image loading, hashing, timestamps, and mask operations in shared
helpers. Extracted modules must not import an ignored local analysis package.
Existing local workflows can later import the shared implementation so fixes
do not need to be maintained twice.

## Existing Annotation Work

The `augusts-code-development` branch supplied fill/subtract, mask-edit
undo/redo, brush cursor previews, mask/bbox visibility fixes, one-pixel brush
controls, and other instance-editing improvements. The current
`augs-conference-rush` branch does not contain all those changes. They are now
combined in the separate integration worktree; the original worktrees remain
available without branch switches.

The GUI worktree's local No Draw Zones, physical measurement/calibration,
imaging-setup controls, bounding-box editing, and reusable validation-review
module were preserved in the integration branch. Fill/Subtract now respects
protection, with synthetic pixel and undo/redo tests.

See [branch reconciliation](branch_reconciliation.md) for the integration plan.

Potential additions to the shared annotation canvas include:

- High-resolution scrolling and native pinch zoom, plus pen-accessible zoom
  controls.
- Reliable initial brush dabs, including when a model supplied an empty mask.
- Polygon-based replacement of an arena boundary.
- Review panels whose image geometry stays fixed when overlays are toggled.

Check behavior against the development branch before selecting individual
changes. A single-mask arena editor should not replace the main application's
multi-instance selection, category, visibility, or bounding-box rules.

## Other General Application Updates

The integration branch also includes these existing local updates:

- Temporal hive-state checkpoints to resume batches without replaying all
  previously completed videos, with configuration and video-order checks.
- Rectangular temporal-map resolutions; the CLI configuration reference is now
  fixed. This sets the width and height of the rolling hive probability map, not
  the YOLO inference size or the registration reference image size.
- Refreshing hive/pollen metrics from existing detections while preserving
  tracking and identity results.
- Combining batch result folders with explicit precedence for overlapping
  video IDs.

The refresh CLI requires explicit hive/chamber models and either `--pollen-model`
or `--no-pollen-model`. The batch CLI uses saved/user-provided model paths instead
of a hard-coded personal pollen checkpoint. CSV merge precedence and checkpoint
contracts have synthetic coverage; real-data review is still needed before release.

## Deferred Limitation

Pooling aligned images from several chamber positions into one daily background
can blur boundaries when alignment is imperfect or nest material moves.
Production use needs backgrounds for each pose and the ability to override the
mask for an individual pose. Approval and timestamp-to-mask selection must
include those overrides. Automatic movement proposals also require review.

## Suggested Integration Order

1. Extract independent registration helpers with synthetic tests (completed).
2. Reconcile existing annotation controls and local GUI updates (completed).
3. Extract the arena editor and landmark controls; shared mask operations are
   completed.
4. Add chamber-position and registered-mask review as an optional workflow.
5. Resolve the pose-specific limitation before integrating final spatial
   classifications into batch analysis.

Verification should cover protected pixels, empty-mask editing, undo/redo,
view preservation, obscured/bad landmarks, independently moving arenas, approval
invalidation, and resume compatibility. Use small synthetic fixtures rather than
research images or trained weights.

## Local Research Files

Keep contact-response analyses, statistical models, experimental results,
notebooks, images/videos, trained weights, and experiment-specific manifests out
of this WIP folder. The local research package, its dedicated tests, and its
generated outputs remain ignored. Generic tests may be extracted separately.

Ignore rules do not remove files already tracked in Git. Existing published
assets or experiment manifests require a separate, explicit cleanup if needed.
