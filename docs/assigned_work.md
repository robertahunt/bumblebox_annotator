# Assigned Work

Assigned Work is the small-local-copy workflow for contributors who do not have
room for the entire project. It is also a simpler annotation view: only assigned
videos and downloaded frames appear. It is not a separate undergraduate identity
or a security role. Use the same updated app version on every computer.

## Coordinator Setup

1. Open the **full local project**, finish its annotation policy, and save labels.
2. In **Settings > Project Sync**, test the shared folder and use
   **Deduplicated versions**. **Include original videos** may be unchecked;
   extracted frame images are still published. Publish a completed project backup.
3. Open the **Assignments** workspace. On first setup, choose **Set Project
   Director...** and explicitly confirm your active profile. Existing directors
   are preserved. The app no longer assigns the role implicitly when saving work.
   Missing connection, published policy, deduplicated backup, or frame prerequisites
   appear here with actions to resolve them. Check Setup Again after making changes.
   This check runs in the Assignments page, not a blocking startup dialog. You can
   switch to Annotate or cancel the check while the shared drive responds.
4. Use [Connect to Lab](shared_profiles.md) once on each computer using the same
   shared folder. In **Assignments > People > Add Contributor...**, select an
   existing lab identity or create a new one. This adds project membership without
   changing your active person; new profiles appear automatically on connected
   computers. **Import Profile...** inside People remains a fallback.
5. Search/filter the video table and check several videos, or **Select Visible**.
   Choose **Assign Selected Videos...**, choose a contributor, and choose **All
   extracted frames** or **Specific frames**, such as `0,5-10`. Review the list and
   confirm **Assign Work**. Frame ranges must exist in every selected video.
Assignments are saved immediately; no downloads are started on another computer.
   Changing filters clears hidden checkbox selections. Table edits remain available
   separately through **Save Table Edits**; save or discard them before bulk assigning.
   Unsaved table choices require confirmation before reload, project/profile changes,
   or closing the app; annotation autosave does not save assignment choices.

Select the same shared director profile on another computer, or import its exported
profile; entering the same name does not recover its ID. One video has one assignee. Assignment choices do
not change train/validation splits or training-frame selections.

The director can use **Transfer Director Role...** to hand over to another
project member. This preserves existing assignments and requires confirmation;
it does not grant a second director or change annotation attribution.

Publish newly imported videos before assigning them. A completed annotation
policy is required before assigning or downloading work. Legacy projects need current
PNG/JSON annotation storage and valid video split metadata. Legacy full-folder
backups must first be published using deduplicated storage.

## Select Work by Mask Type

The source header identifies the **Shared version**, who published it, and its
publication time. The folder icon opens that exact version's directory; the
information icon shows its version ID, manifest location, and the local project
path. **Last checked** is the time of a freshness check, not the publication time.

While this page is visible, a background check reads small publication pointers
once per minute, without opening masks. The refresh icon in the header checks
immediately. Changed publications show **Refresh Assignments**; the displayed
version and unsaved assignment choices are not replaced automatically. Refresh
retains the existing discard-confirmation step. A failed check is shown as a
failure, never as "up to date". Divergent shared histories still require resolution
through the existing guarded workflow; the check never chooses a winning branch.

In Assignments, choose **Work on: Hive**, then **Show: Not Started** for untouched
work or **Show: Not Checked** for started/unverified work, and **Unassigned**.
Use **Select Visible > Assign Selected Videos...** to assign the
matching videos. **More filters** includes Has masks, No masks, Some frames lack
masks, Unknown availability, and the train/validation/test/inference split.
Bee annotations have no effect when filtering hive work. Brood groups select all
their stages; individual stages are also available.

The Show filter offers **All Work**, **Not Started**, **Not Checked**, and
**Finished** in both Assignments and the annotation sidebar. Not Started means no
saved masks or completion records for the selected task. Not Checked includes
partially finished work, changed masks, pending import reviews, and unknown/unscanned status;
the specific status is still shown on each row. Finished includes work explicitly
checked as having no objects present. Alternating row backgrounds separate videos.

**Experiment / setup** filters by the project's existing named imaging setups,
such as `2021 bumblebox` or `2024 rig build`. Choose All experiments to remove this
filter, or No experiment assigned to find ungrouped videos. This is only a view
filter: it does not reassign videos or change calibration. It combines with the
mask, split, search and contributor filters; hidden checkbox selections are cleared.
The assignment table uses the published setup assignments.

**Sort > Recently modified** orders videos newest-first using annotation-file and
annotation-folder modification times in your local working project. The
**Modified (local)** column shows those dates; unavailable dates sort last. It
includes shared masks, frame masks, bounding boxes, completion records and folder
changes from deletions. It excludes original-video dates, extracted-frame dates,
COCO exports and calibration settings. Imports/downloads can change file dates,
so this is not a cross-computer edit audit trail. Dates are read in the background
only when requested, without loading images or masks. Table edits and selections
stay attached to their videos when sorting.

The Masks and Annotation work columns describe the **published annotations**,
over all published frame images. Publish local changes before expecting them in
this table. The assignment confirmation carries the selected mask type as its
task, alongside the contributor and frame range. Opening an assigned video
selects that task in the annotation sidebar. It is guidance, not a restriction
on editing other categories. One contributor still owns the video; this does not
enable concurrent category-specific assignments to the same video.

Mask availability is automatic; completion is not inferred from the existence of
an annotation. Empty instances and bounding-box-only annotations do not count as
masks. Missing/corrupt or unsupported legacy data appear as Unknown, not No masks.
Existing masks initially appear as Not checked. Shared masks count once per video;
frame-specific progress counts the available or assigned frames, not missing
frames from a selective download.

New backups include a sparse summary in the publication manifest. Unchanged video
summaries are reused by source fingerprint. Opening Assignments checks these saved
summaries but does not scan annotation payloads. Missing or stale summaries appear
as **Not scanned**, not Unknown or No masks; this is not a background scan in
progress. Choose **Scan Mask Status** to inspect older backups
from annotation metadata (and legacy mask rasters where needed), without downloading
frame images. Unresolved rows show **Scanning...** during that operation and remain
visible under work filters until results arrive; experiment, split, search, and
assignment filters still apply. **Unknown** after scanning means data could not be
read or interpreted; row tooltips show the reason, and scanning can be retried.
Cancellation or failure restores the previous summaries, never leaves Scanning
displayed, and does not claim the masks are absent.
The explicit scan reports video/frame progress and can be cancelled
after the current file operation; assignment table edits are preserved. Subsequent
filter changes use the in-memory index. Publishing a backup with this app includes
the summary for later sessions; merely inspecting a backup does not modify it.

## Compare Local and Shared Work

**Review Local vs Shared Changes...** compares saved annotation files and project
annotation settings against the **exact shared version displayed in Assignments**.
It does not switch to a newer publication halfway through the comparison. Results
are grouped by video, with project-level files in their own group; filters search
by video/file and distinguish **Different**, **Only local**, and **Only shared**.
An undownloaded file is Only shared, never automatically considered deleted.

This is a read-only, file-level comparison, not an upload plan or a mask-pixel
viewer. It does not say which copy is newer, infer conflicts, merge, upload, or
approve annotations. Unsaved canvas edits, extracted image files, original videos,
models, and regenerated COCO exports are excluded. Save local annotation edits
before comparing them. If annotation files change while checking, the comparison
asks for a retry rather than presenting an inconsistent result. The operation is
cancellable after the current file read.

The upload workflow's existing Review Changes still compares with that contributor's
previous publication. Its baseline remains explicitly labeled; it is separate from
the Assignments comparison. Per-mask visual comparisons, per-class local/shared
completion columns, and persistent scan caching for older backups remain future work.

## Contributor Workflow

1. Select your contributor profile. On another computer, use Connect to Lab
   and select the existing shared identity, or import the same exported profile.
   Do not create a second identity with the same name.
2. Authenticate the drive if needed using **Sign In to Drive...** (Linux Kerberos)
   or your system tools. Connect to the same archive root. The path may differ between
   computers, but it must point to the same shared folder identity.
3. Select your project in **My Work**, or open **File > My Assigned Work...**. Choose assigned videos, then
   **Download Assigned Work...**. Review new/updated videos and the incremental
   transfer size. The app chooses a persistent local working folder automatically.
   Later downloads reuse it instead of making numbered copies.
4. The verified copy opens in the **Annotate** workspace in **Assigned Work** mode.
   Edits autosave locally after a brief idle period; **File > Save** is also
   available. Work can continue while the drive is disconnected.
5. Reconnect and choose **Upload Assigned Annotations** to upload progress. When the current
   video's assigned frames are finished, choose **Submit Current Video for Review**.

Assignment status is checked in the background after choosing a profile or opening
a project, and on returning Home. A verified empty result opens a non-modal **No
Work Assigned Yet** notice once per contributor session/shared root, offering
Connect to Lab, Check Again, and Continue Locally. The status remains on Home.
Disconnected/unreadable drives say **Unable to check assignments**, never "no work".
An unconfigured connection or unpublished project has its own setup state. Results
from an older profile/project/drive check are discarded. Directors do not get the
empty-assignment notice for their own project. Roles are per project, not global
properties of the exported profile.

Reopen this local copy through **My Work** or **Recent Projects** on subsequent
sessions. Use **Download Assigned Work** again to add assignments or refresh clean
local videos. Already downloaded, unchanged files are not copied again. A
different active contributor cannot edit or publish this copy.

On Linux, new copies live under `$XDG_DATA_HOME/BeeWhere/assigned-work`, defaulting
to `~/.local/share/BeeWhere/assigned-work`, keyed by shared-root, project, and
contributor IDs. This is persistent data, not a disposable cache, and is outside
the application checkout. The download preview shows the actual folder. Project
or contributor renames do not create another copy. Each computer has its own copy.
Images and annotations still need local storage; this is not remote streaming.

If older downloaded copies are known in Recent Projects, the app asks which one
to keep using and remembers that choice. It does **not** choose by timestamp,
merge competing masks, move, or delete the other folders. If work is split across
copies, retain all of them and ask the director to compare/import their changes.
An older copy not listed in Recents can be opened first, then adopted on download.
A remembered copy that is unavailable is an error, not permission to silently
create another folder. Reconnect/restore that folder before continuing.

Project Home offers **Open Selected Project** and **Download Assigned Work**. See
[workspaces.md](workspaces.md) for startup preferences and save/upload indicators.

## Storage and Scope

- Only selected frame images from assigned videos are downloaded. No original
  videos, models, training outputs, or tracking sequences are downloaded.
- The annotation policy and other project settings are copied unchanged.
  Existing annotations for the selected videos are included, even for their
  undownloaded frames, so shared masks and instance-ID allocation remain intact.
  Their size is included in the estimate; selecting very long, densely annotated
  videos can still consume substantial space.
- Only assigned frame annotations and the selected videos' shared masks can be
  published. **A shared-video mask still applies to the entire video**, including
  undownloaded frames. The normal category policy guidance remains visible.
- Setup, imports, calibration, split changes, and training/export are coordinator
  tasks. Policy can be reviewed but not edited in an assigned copy.
- Publishing uploads new/changed annotation objects and a complete manifest
  referencing existing remote files. Files absent from the local working copy
  are preserved remotely, not treated as deletions. Explicit deletions of editable
  local annotation files do propagate in the new version; earlier versions remain.
- Original train/validation selections and other full-video metadata are retained.
  Only the instance-ID counter can be updated by assigned work.
- This does not provide streaming images, automatic cache eviction, or remote
  garbage collection. Downloaded media remain on the local disk. Publish and
  verify work with the coordinator before manually retiring an old working copy.

## Review

The coordinator opens the **Assignments** workspace (also reachable through
**Project > Video Assignments**) to see:

| Status | Meaning |
| --- | --- |
| Assigned | Not yet downloaded or published under this assignment. |
| In Progress | Downloaded, or saved work has been published without submission. |
| Ready for Review | The contributor submitted the assigned frames. |
| Accepted | The coordinator accepted that exact submission. |
| Changed Since Submission | Later published changes affected this video. Review again. |

Local unsent edits cannot be seen remotely. These statuses describe published
work, except the initial download's In Progress indication.

Use **File > Review Shared Updates** in the coordinator's full project to get the
latest combined publication, review the assigned frames and shared masks, then
select the video and **Accept Reviewed Submission...**. Acceptance checks that
the local annotations actually match the submission. After two students publish
in sequence, select the later publication, which includes both contributions.
Students publishing different assigned videos do not need to download each
other's frames.

An unchanged repeat publication retains Accepted status. Further annotation edits
return the video to In Progress unless explicitly resubmitted. Acceptance is a
collaboration status, not a substitute for pending imported-frame reviews or a
new automatic training-eligibility rule; the existing training/export checks still
apply in the full project.

## Conflicts and Recovery

One contributor owns each video assignment, even if only some frames are assigned.
Two people should not edit the same video's shared masks concurrently.

Reassigning a video or changing its assigned frame range or mask task retires the old assignment.
Its local files remain intact. Downloading can refresh a changed assignment if
the local video has no unpublished changes. If that video has edits, review is
required before refreshing its task/range. Finish/publish current work before
expanding an assignment when possible.

Incremental downloads include existing active assignments as well as the new
selection. They preserve local edits when the published video is unchanged, and
update clean videos when shared annotations change. If both sides changed any
part of the same video, the whole update stops: mask PNGs and annotation JSONs
are not merged file by file. Changed project-wide policy/settings also require
director review. A simultaneous change after the preview requires a fresh check.
Reassigned videos are retained on disk, removed from active work, and excluded
from uploads when downloading other current assignments. They are never deleted
automatically; unpublished retained work still needs director review.

Publishing also stops if someone else changed any part of the assigned video, or
changed project-wide settings after the download. This includes two computers
using the same contributor profile on the same video. It does not choose a winner
or merge masks automatically. Keep both working copies and ask the coordinator to
compare/import the intended annotations into the full project, then issue fresh
assignments. **Do not delete `.assigned-work.json` to force a full backup.**

New downloads are verified in staging before opening. Updates of existing copies
stage and verify changed files, retain a rollback bundle, and use the normal local
update journal. Cancellation before replacement leaves the old copy unchanged;
an interrupted replacement must be recovered on reopening before editing. The
GUI saves pending edits and pauses preloading/autosaving during the operation,
then discards old mask caches before reopening the same path. Use **File > Local
Rollback History** to undo an update only if no newer edits have been made.
Updated copies use assigned-work record format 2; older apps must be upgraded
before opening them.

Upload cancellation
leaves local edits saved. If an upload finished but its final local receipt could
not be written, the next Upload Assigned Annotations recovers that receipt before checking new
edits. After a process crash, the existing sync lock may still require the manual
checks described in [project sync recovery](project_sync.md#connection-failures-and-recovery).

Contributor IDs and assignment checks are workflow protection, not authentication
or filesystem permissions. Anyone with write access to the shared folder could
alter its files. Keep independent backups and appropriate shared-drive permissions.

## Testing

`scripts/test_annotation_work.py` and `scripts/test_annotation_work_gui.py` cover
category filters, exact mask completion signatures, reviewed absence, shared masks,
import-review independence, sparse-frame navigation, and completion publication
without removing undownloaded files.

`scripts/test_assigned_work.py` and `scripts/test_assigned_work_gui.py` exercise
selective download, normal GUI saves, identity and scope guards, independent-video
publications, stale/same-video conflicts, review acceptance, cancellation, storage
failure, damaged downloads, and interrupted publication receipts using temporary
projects. No real research-drive projects are modified by these tests.
