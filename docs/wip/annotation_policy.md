# Annotation Policy and Temporal Support (WIP)

Status: the initial policy interface and per-frame import review are implemented
on `integration/annotation-batch` (2026-10-11). Remaining unchecked tasks are
planned, not functional. Mainline through `56c9a18` is now incorporated into the
integration worktree; see the compatibility checklist below. This is not yet a
user-validated production release.

Project creation shows scope and meaning for every category. Existing projects
show their policy on first opening (and after it changes), with **Project >
Annotation Policy...** available anytime. An active-category summary appears
above the image. Import warnings and the **Imported Frame Review...** queue are
described in [the import workflow](../project_import.md).

## Working Today

| Category | Saved manual annotation scope | Temporal batch support |
| --- | --- | --- |
| Bee | Frame-specific | Identity tracking, not a reconstructed stationary surface map |
| Hive / nest | Frame-specific or shared across video, selected per project | Optional history-informed map; updated-map overlap scoring and matching visualization |
| Brood and queen brood | Frame-specific only | Experimental history-informed stage maps and bee-overlap outputs |
| Nectar source | Frame-specific only | Not implemented; annotation, training, and current-frame inference work |
| Pollen | Shared across video | No history-informed reconstruction; batch predictions and summaries are separate outputs |
| Chamber | Shared across video | Provides alignment coordinates for hive/brood history; not its own reconstructed surface map |

The current scope rules live in [annotation_scope.py](../../core/annotation_scope.py).
New projects default to frame-specific hive labels. Older projects without the
`hive_annotation_scope` setting retain video-wide hives. There is no universal
per-category scope selector. Versioned `annotation_policy` metadata now records
each category's scope and intended label meaning in `annotations/project.json`.
Legacy meanings are unconfirmed, not inferred from their scope or pixels.

History is calculated from model predictions during batch analysis, not while
the user draws or changes annotation frames. Temporal outputs do not replace
manual training masks. A fixed shared mask, a video-level prediction summary,
and an evolving history-informed map are three different representations.

Brood support remains experimental: synthetic coverage does not establish
biological accuracy. Brood replay/resume and a dedicated validation dashboard
are not implemented. See the current [brood workflow](../brood_segmentation.md),
[nectar limits](../nectar_segmentation.md), and [hive outputs](../hive_exports.md).

## Agreed Design

Record two independent properties for each category:

- **Scope:** one frame or shared across the video.
- **Label meaning:** visible surfaces, full estimated object extent, or legacy
  meaning not yet confirmed.

Scope alone cannot prove what an annotator intended. A frame-specific mask can
also describe an estimated full object; a shared mask is not proof that hidden
boundaries are known. Initial controls should offer only combinations that the
storage, editing, export, and training paths actually support.

New-project defaults and guidance:

| Preset | Intended guidance |
| --- | --- |
| Frame-specific: visible surfaces | Label only object pixels visible in this frame. Exclude pixels covered by bees or other objects. |
| Shared across video: full object extent | Use clear views from the video to establish the full extent. The same mask applies to every frame; use only when position and shape remain sufficiently stable. |

Do not encourage users to invent boundaries that cannot be determined. For
visible-surface labels, a fully hidden object has no visible pixels to draw;
that is not evidence of physical absence for temporal reconstruction. Whether
a frame is finished/reviewed is a separate question from its mask being empty.

For hive/brood visible-surface projects, explain: "Label visible evidence here;
temporal reconstruction happens later during analysis." Do not promise this
workflow for nectar or pollen until their temporal pipelines exist.

## Implementation Checklist

All unchecked items below are pending, not available functionality.

### Policy and Guidance

- [x] Define versioned, per-category `annotation_policy` metadata, separating scope
  from label meaning and recording confirmation/change provenance. Use the UI,
  not hand-edited metadata, to confirm meanings.
- [x] Add a policy table to project creation, showing every category's supported
  scope and meaning. Show unavailable features as "Planned - not available" or
  explanatory text, never as usable switches.
- [x] Show the saved policy on first project opening and after policy changes;
  keep it accessible through a Project > Annotation Policy action. Avoid a
  mandatory repeated dialog on every ordinary reopen.
- [x] Show the active category's policy near its controls, for example
  "Hive | This frame | Visible surfaces", with concise contextual guidance.
- [x] Distinguish available temporal analysis from whether it is actually enabled
  in a batch. A frame-specific labeling policy must not imply it is running.

### Existing Data and Transfers

- [x] Derive legacy scope from existing settings, but mark label meaning as
  unconfirmed. Ask the user rather than inferring intent from pixels or scope.
- [x] Keep policy confirmation distinct from data conversion. Never silently
  move masks, fill hidden regions, or relabel existing annotations when a
  project opens or a policy is confirmed.
- [x] Require a preview and explicit conversion for scope changes affecting
  saved masks, preserving source data and backups. Reuse the existing project
  importer where appropriate; do not treat editing a setting as migration.
- [x] Preserve policy provenance on imports and exports. Check source and
  destination meanings as well as frame/video scope, including mixed legacy
  data. Do not reinterpret old masks merely because a default changes.
- [x] Persist per-frame review requirements for new imports. Completion records
  include reviewer/time and saved source fingerprints; edits reopen review.
  Shared imports require separate approval for every affected extracted frame.

### Training and Verification

- [x] Block selected pending import reviews at COCO export and validate cached
  COCO in GUI training workers. Include policy and review metadata in exports.
- [ ] Extend policy auditing to all historical annotations/imports and external
  training entry points. Confirming a project-wide intended meaning does not
  automatically certify old pixels or find already-mixed training targets.
- [x] Keep exact mask pixels, category IDs, creator provenance, and existing
  training behavior unchanged unless the user explicitly requests conversion.
- [x] Test new and legacy projects, save/reopen, changed policies, imports in
  both directions, unknown meanings, incompatible exports, and unavailable
  controls using disposable fixtures.
- [x] Exercise creation/opening/review dialogs offscreen and inspect screenshots;
  test save, completion, cancellation, reload, and save-failure behavior using
  disposable fixtures. Tests: `scripts/test_annotation_policy.py` and
  `scripts/test_annotation_policy_gui.py`, plus importer regression coverage.
- [ ] User acceptance testing on a disposable copy of a representative real
  project, especially large queues and shared-mask correction across frames.

## Deferred Extensions

These are recorded gaps/candidates, **not approved universal scope switches or
guarantees of scientific validity**. Each needs its own design and tests after
the policy layer accurately describes today's behavior.

- [ ] Consider broader per-category scope choices, including frame-specific
  pollen and shared nectar where appropriate. Review whether shared brood
  stages are meaningful before offering them; stages and visibility can change.
- [ ] Add nectar batch inference and measurement support before considering a
  temporal nectar map. Current-frame inference is not batch integration.
- [ ] Evaluate temporal pollen reconstruction separately from mainline's
  per-instance video summaries; those summaries are not already temporal maps.
- [ ] Complete brood resume/replay and validation support before relying on it
  for resumable production analyses.
- [ ] Keep dated brood cohorts, continuous-age prediction, and stacked-generation
  identities separate from appearance labels. None is implemented by the
  current temporal stage maps.

## Mainline Compatibility

The October 11 integration addresses these compatibility boundaries separately
from any training-memory optimization:

- [x] Route mainline's ArUco CSV blocking-save helper through the scope-aware
  save path; preserve brood/queen/nectar, frame-specific hive, attribution, and
  queued-save ordering.
- [x] Teach the hive-refresh polygon reader to accept mainline's pipe-separated
  disconnected contours without falling back silently to bounding boxes.
- [x] Preserve mainline's pollen output naming, but protect old output folders
  from reset/append across incompatible schemas. Keep old per-frame summaries
  distinct from new per-instance results, and correct the batch dialog wording.
- [x] Combine wrapping toolbars with the newer category and grouped-visibility
  controls, then verify the merged application on disposable projects.
- [x] Extend Project Overview to count frame-specific hive, brood, queen brood,
  and nectar labels on selected frames without double-counting derived boxes.

Regression coverage is in `scripts/test_mainline_compatibility.py`, in addition
to the existing policy/import/editing suites. Offscreen GUI checks do not replace
the user acceptance testing still listed above.

Mark tasks complete only with implementation and verification evidence. Keep
private datasets, model checkpoints, and experimental analysis outputs out of
this roadmap and its tests.
