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
3. Ask each contributor to export their profile using
   **Settings > Change Contributor > Export Profile...**. Import it in
   **Project > Video Assignments > Import Contributor Profile...**. This adds an
   assignee; it does not switch your active contributor.
4. Assign each video to one contributor. Leave **Assigned frames** as `All`, or
   enter actual extracted frame numbers, such as `0,5-10`. Only published frames
   can be assigned. These choices do not change train/validation splits or the
   full project's training-frame selections.
5. Choose **Save Assignments**. The first profile to save assignments becomes
   this project's coordinator. Keep/export that profile for use on another
   computer; entering the same name there does not recover its ID.

Publish newly imported videos before assigning them. A completed annotation
policy is required before downloading assignments. Legacy projects need current
PNG/JSON annotation storage and valid video split metadata. Legacy full-folder
backups must first be published using deduplicated storage.

## Contributor Workflow

1. Select your contributor profile. On another computer, import the same exported
   profile instead of creating a second profile with the same name.
2. Mount/authenticate the shared drive outside the app. Configure and test the
   same archive root in **Settings > Project Sync**. The path may differ between
   computers, but it must point to the same shared folder identity.
3. Open **File > My Assigned Work...**, choose assigned videos, and choose
   **Download Assigned Work...**. Confirm the frame count and estimated download
   size, then select a local parent folder with enough free space.
4. The verified copy opens in **Assigned Work** mode. Annotate normally and save
   locally with **File > Save**. Work can continue while the drive is disconnected.
5. Reconnect and choose **Publish Work** to upload progress. When the current
   video's assigned frames are finished, choose **Submit Current Video for Review**.

Reopen this local copy through **Recent Projects** on subsequent sessions. Do not
download it again each day. Downloading always creates a new folder; it never
silently replaces existing work. A different active contributor cannot edit or
publish this copy.

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

The coordinator opens **Project > Video Assignments** to see:

| Status | Meaning |
| --- | --- |
| Assigned | Not yet downloaded or published under this assignment. |
| In Progress | Downloaded, or saved work has been published without submission. |
| Ready for Review | The contributor submitted the assigned frames. |
| Accepted | The coordinator accepted that exact submission. |
| Changed Since Submission | Later published changes affected this video. Review again. |

Local unsent edits cannot be seen remotely. These statuses describe published
work, except the initial download's In Progress indication.

Use **File > Update Local Copy** in the coordinator's full project to get the
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

Reassigning a video or changing its assigned frame range retires the old assignment.
Its local files remain intact, but publishing is blocked until the coordinator
reconciles the work. Finish/publish current work before expanding an assignment;
then download a fresh working set. There is no in-place assignment refresh yet.

Publishing also stops if someone else changed any part of the assigned video, or
changed project-wide settings after the download. This includes two computers
using the same contributor profile on the same video. It does not choose a winner
or merge masks automatically. Keep both working copies and ask the coordinator to
compare/import the intended annotations into the full project, then issue fresh
assignments. **Do not delete `.assigned-work.json` to force a full backup.**

Interrupted downloads are not opened as completed projects. Upload cancellation
leaves local edits saved. If an upload finished but its final local receipt could
not be written, the next Publish Work recovers that receipt before checking new
edits. After a process crash, the existing sync lock may still require the manual
checks described in [project sync recovery](project_sync.md#connection-failures-and-recovery).

Contributor IDs and assignment checks are workflow protection, not authentication
or filesystem permissions. Anyone with write access to the shared folder could
alter its files. Keep independent backups and appropriate shared-drive permissions.

## Testing

`scripts/test_assigned_work.py` and `scripts/test_assigned_work_gui.py` exercise
selective download, normal GUI saves, identity and scope guards, independent-video
publications, stale/same-video conflicts, review acceptance, cancellation, storage
failure, damaged downloads, and interrupted publication receipts using temporary
projects. No real research-drive projects are modified by these tests.
