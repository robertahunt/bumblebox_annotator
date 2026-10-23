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
4. Enable [Profile Sharing](shared_profiles.md) on the director's and contributors'
   computers using the same shared folder. Select existing identities to share;
   new identities can be published automatically. Discovered profiles appear in
   Assignments without changing your active contributor or any existing roles.
   **Add Contributor: Import Profile...** remains an offline fallback.
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
imported profile. This preserves existing assignments and requires confirmation;
it does not grant a second director or change annotation attribution.

Publish newly imported videos before assigning them. A completed annotation
policy is required before assigning or downloading work. Legacy projects need current
PNG/JSON annotation storage and valid video split metadata. Legacy full-folder
backups must first be published using deduplicated storage.

## Contributor Workflow

1. Select your contributor profile. On another computer, enable Profile Sharing
   and select the existing shared identity, or import the same exported profile.
   Do not create a second identity with the same name.
2. Mount/authenticate the shared drive outside the app. Configure and test the
   same archive root in **Settings > Project Sync**. The path may differ between
   computers, but it must point to the same shared folder identity.
3. Open **File > My Assigned Work...**, choose assigned videos, and choose
   **Download Assigned Work...**. Confirm the frame count and estimated download
   size, then select a local parent folder with enough free space.
4. The verified copy opens in the **Annotate** workspace in **Assigned Work** mode.
   Edits autosave locally after a brief idle period; **File > Save** is also
   available. Work can continue while the drive is disconnected.
5. Reconnect and choose **Upload Assigned Annotations** to upload progress. When the current
   video's assigned frames are finished, choose **Submit Current Video for Review**.

Assignment status is checked in the background after choosing a profile or opening
a project, and on returning Home. A verified empty result opens a non-modal **No
Work Assigned Yet** notice once per contributor session/shared root, offering
Profile Sharing, Check Again, and Continue Locally. The status remains on Home.
Disconnected/unreadable drives say **Unable to check assignments**, never "no work".
An unconfigured connection or unpublished project has its own setup state. Results
from an older profile/project/drive check are discarded. Directors do not get the
empty-assignment notice for their own project. Roles are per project, not global
properties of the exported profile.

Reopen this local copy through **Recent Projects** on subsequent sessions. Do not
download it again each day. Downloading always creates a new folder; it never
silently replaces existing work. A different active contributor cannot edit or
publish this copy.

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
not be written, the next Upload Assigned Annotations recovers that receipt before checking new
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
