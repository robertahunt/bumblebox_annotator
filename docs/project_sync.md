# Contributors and Optional Project Sync

## Contributor Sessions

Starting `python main.py` shows contributor profile tiles. Select your identity each session, particularly on a
shared workstation. Cancel exits without starting an editing session.

The active name appears in the status bar. **Settings > Change Contributor** saves
pending work under the previous contributor before starting another session.
Background saves retain the identity with which they were queued.

Contributor names and stable IDs are stored in local application settings, not
inherited from an opened project. A new name gets a new ID; selecting a remembered
name reuses its ID but creates a new session ID. Names are self-declared, not
authenticated accounts. The same name first entered on another installation gets
a different ID unless you select the existing shared profile or import it.

Use **Connect to Lab...** to [share identities automatically](shared_profiles.md)
and configure project discovery through one shared folder. Review the existing
profiles selected for publication; new profiles are then shared automatically.
Discovered identities are cached locally; sharing
does not synchronize active sessions, device settings, or project roles.

As an offline fallback, use **Export Profile...** in the contributor dialog (also available through
**Settings > Change Contributor**) on the first computer, then **Import Profile...**
on the second. Import asks for confirmation and does not reattribute old labels.
The profile contains only your display name and contributor UUID, not credentials.
Cancelling the session does not select a new active person; explicitly imported or
created profiles remain saved. Each installation keeps a
separate device UUID and hostname, recorded with subsequent edits and backups;
profiles do not transfer that device identity. There is no account service or
authentication of claimed authorship.

Annotation JSON records a `provenance` object with creator, latest editor, UTC
timestamps, session ID, and origin. Attribution is recorded at successful save:

- New annotations get the current contributor as creator and editor.
- Changed pixels, categories, bbox-only coordinates, and supported identity/model
  metadata update the latest editor, preserving the creator.
- Opening, selecting, hiding, or saving unchanged annotations does not claim them.
- Existing annotations without provenance retain an unknown creator, even after
  someone edits them. Existing model source metadata is preserved.
- Frame-local and video-wide masks and bbox-only annotations are supported.

Hover over a saved instance in the sidebar to see its creator and latest editor.
This is instance-level attribution, not a per-pixel edit log. Deletions are
represented by absence in the next saved revision. CLI/batch jobs without a
contributor session do not invent human authorship.

### Retrospective Attribution

Legacy annotations can be explicitly credited after saving and closing the app:

```bash
python -m scripts.attribute_project_annotations \
  --project /path/to/project \
  --contributor-name "Your name" \
  --contributor-id YOUR_EXISTING_PROFILE_UUID
```

This previews the affected records. Add `--apply --confirm-app-closed` to apply.
Use the UUID belonging to your existing local contributor profile, not a new one.
Original JSON is backed up under `project/attribution_backups/` before any changes.
Both frame/video masks and matching bbox records are updated, but pixels, model
source, existing edit history, and original timestamps remain unchanged. Unknown
creation dates stay unknown; a separate attribution-history entry records when
authorship was assigned. Existing credit to someone else is not overwritten.
Repeating the command for the same contributor is a no-op. Reopen the app after
applying, then sync to publish the updated attribution in a **new** revision;
historical backup contents are not rewritten.

## Setup

For selective downloads and a contributor-only video list, use
[Assigned Work](assigned_work.md). The full-project backup/update workflow below
remains available to coordinators; assigned copies use **Upload Assigned Annotations** instead.

The first app launch offers optional sync setup. **Not Now** leaves local editing
and saving fully available. Setup is also available at **Settings > Project Sync**.

1. Choose an existing local, USB, or mounted network folder.
2. Select **Test Connection**. The app writes, reads, and removes a uniquely named
   probe, then records a small destination identity marker.
3. With a project open, select **Enable sync for this project**, then **Save**.
4. Choose **Estimate Transfer Size** in Manage Project, the File menu, or the
   settings dialog. Pending annotations are saved locally, then file count,
   logical size, estimated additional storage, and free space are displayed.
   This does not upload project files. A connection test may write a small probe
   and destination identity marker.
5. **Back Up Local Project** becomes available after a successful estimate.
   **Review Changes...**, next to Publish in Manage Project and Project Sync settings,
   opens the outgoing comparison. It is also available in the final publish confirmation.
   Edits or changed backup options invalidate it. Publishing reuses the local
   checksums calculated by the estimate and checks file metadata to reject stale
   results, without rereading those local contents. The final confirmation still allows
   cancellation. Publishing does not download or merge others' changes.
6. To receive a newer publication, use **File > Review Shared Updates** (also available
   in Project Sync settings). Updating is a separate, explicit action.

The destination can be configured before opening a project. Enable each project
separately when ready. Uncheck its enable option to disable sync; changing the
destination requires another successful test. Neither action deletes remote data.

Authentication and mounting remain outside the app. For a Kerberos-protected
drive, authenticate and mount it using your normal lab procedure first. No
passwords, tickets, hostnames, or lab-specific paths are embedded in the app.
`smb://` and `sftp://` URLs are not supported directly: use an accessible filesystem
path. Destination configuration and enablement remain local to each installation.

## Included Files

Each published revision contains:

- `annotations/`, including exact masks, JSON, bbox files, project settings, and
  imaging setup metadata, but excluding regenerable `annotations/coco/` exports.
- `frames/`, including extracted images and `video_metadata.json`.
- `tracking_sequences.json`, when present.
- `input_data/` when **Include original videos** is checked (the default).

Models, checkpoints, training runs, debug images, and unrelated project-root
folders are excluded. With original videos included, a restored working copy can
be opened as a normal project. Without them, it is a partial backup and may require restoring the
videos before all workflows work. Symbolic links in included data are rejected
rather than silently copying files from outside the project.

## Revisions and Collaboration

On the first sync the source gets `annotations/project_sync.json`, containing a
stable project ID. Copies made afterward keep that shared identity. Each
contributor publishes into a separate directory:

```text
chosen-destination/
  .bumblebox-sync-root.json
  hive-test-v1--6918cd50/
    .bumblebox-project.json
    objects/sha256/<prefix>/<checksum>
    contributors/August--0a0b573d/
      .bumblebox-contributor.json
      latest.json
      revisions/<UTC-timestamp-and-id>/
        manifest.json
```

New publications default to **Deduplicated versions** in Settings > Project Sync.
Each manifest lists the exact files in that version. Their contents are stored
once by SHA-256 checksum in the project's shared `objects/` folder, across all
contributors and versions. This is file-level deduplication, not lossy compression
or annotation merging. Images, masks, IDs, and attribution remain unchanged.
**Keep the entire shared project folder together.** A manifest copied elsewhere
without its object store is not a usable backup. Use this updated app on every
computer that publishes or restores compact versions.

**Full project folders (legacy)** remains available: each revision has its own
`project/annotations`, `project/frames`, and optional `project/input_data` tree.
Existing full-folder versions remain readable and are neither converted nor
deleted automatically. The first compact publication of an existing archive
needs space for the initial shared file store in addition to those old backups.

Folder names combine a filesystem-safe name with the first eight characters of
the ID. Identity markers retain the **full UUID**; shortened IDs are only labels.
Name collisions fall back to the full ID. The published project folder name comes
from the **current local project folder name**, not the potentially older `name`
in `annotations/project.json`. For example, a local `2026_CV4E` folder is published
as `2026_CV4E--<short-id>`, even if its internal name is still `test`.

When publishing finds that project's existing archive under an older name, it
shows the proposed rename in the publish preview and, after successful normal
publication, renames the directory in place under the project lock. A blocked or
cancelled publication, or a separate conflict backup, does not rename an existing
readable archive. If renaming fails, the published backup remains at the old path
and the app reports a warning; publishing again can retry the rename. Project identity, all
contributors' histories, stored files, and revision contents are preserved; no
second archive is created. Existing absolute paths/bookmarks need updating. Use
the same project folder name on collaborating computers to avoid renaming the
archive back and forth, and do not run older publishers at the same time. Local
project settings and annotation contents are not changed by this naming rule.
Contributor display-name changes still reuse the existing contributor directory.

Existing UUID-only project and contributor folders are renamed automatically on
their next sync. This moves directories, not their contents: revisions, manifests,
and checksums are unchanged, but old absolute paths must be updated. Other
contributors' histories move with the project; their own folder names migrate
when they next sync. Close older app versions before the first migration, and
use the updated app on all collaborating installations. Duplicate folders with
the same identity are reported, never silently merged or overwritten.

The completion dialog gives the exact backup path. The manifest includes human
project/contributor names and file checksums; `latest.json` points to that
contributor's latest accepted publication. Separate conflict backups deliberately
do not advance it. There is no single project-wide authoritative latest version.
Settings show the latest successful backup time or failure.

Publishing is manual and one-way. Local annotations are saved and queued writes are
drained before a modal progress dialog pauses editing. Newly copied files are
verified in an unpublished staging directory. Source changes during copying
abort publishing; do not edit the local project from another process during sync.
Only complete revisions are published, and prior revisions are not automatically
deleted. A stable project-ID lock serializes syncs to that project, including
folder migration; a contributor-specific lock also protects each copy. Sync never
modifies another contributor's revision contents.

Unchanged projects reuse their existing revision, unless the storage mode changes.
For compact publications, only missing file contents are uploaded. Repeated media
do not require additional data files or filesystem reflink support. Unchanged
referenced files are checked for presence, ordinary-file status, and expected size;
their contents are not rehashed during routine publication. Newly referenced
contents already present in the shared store are fully hashed before reuse.
Shared contents are never edited in place. Detected damage stops publication.
A storage preview shows logical size, file count, estimated new bytes,
and destination free space; publishing rechecks space. Server quotas can be lower
than filesystem free space. Estimates still read local data and publication still
performs remote metadata operations, but unchanged remote media are not reread.

Estimates keep their local checksums in memory until publication, project edits,
or a settings change. Publication checks sizes, modification/change times and
file identities before reuse, and checks the inventory again before completing.
Existing destination files get **Checking stored file information** checks.
Newly copied files get **Verifying uploaded content** checks before their version
is published. This does not certify all unchanged stored content. A first
backup estimate may not need to hash local files; those files are read once during
publication instead. Full-folder estimates hash files when a previous publication
exists so the outgoing review can compare contents, then reuse those checksums at
publication. No persistent checksum cache is trusted
across application sessions.

Assigned-work uploads use a narrower routine check: they inherit unchanged
published references without per-object presence/size checks, while verifying
uploaded content and refusing known damage. They also reuse stable local
checksums within the current app session. See
[Incremental Upload Checks](assigned_work.md#incremental-upload-checks) for the
scope and limitations; the full-project backup checks above are unchanged.

### Full Integrity Checks

Use **File > Check Backup Integrity...**, also available in Project Sync, to select
an exact contributor/version and run a full content check. Historical and separate
backups are included; a newer date does not imply a version includes everyone's
work. No full check runs automatically when the dialog opens. The dialog shows
the latest attempt and last successful full check for the selected manifest.
An informational reminder appears when no successful check is recorded or the
last pass is older than the configurable interval (default 30 days). This is not
a scheduler and does not prevent routine publication.

The checker reads all content referenced by that version, including extracted
frames and original videos if included. Identical compact objects are read once.
Progress reports unique contents and bytes read. Only manifest-referenced data
are checked, not other versions or unreferenced objects. Legacy full-folder
backups are supported without conversion. A changed manifest invalidates the
check. The checker takes the existing project-wide publication lock, so other
computers can annotate locally but cannot publish this project until it ends.
Use an appropriate quiet period for large checks. Cancellation is checked between
read chunks; an operating-system network read may still take time to return.

Outcomes are **Passed**, **Problems found**, **Cancelled**, or **Unable to complete**.
Disconnected storage or unreadable files do not become a false pass or a blanket
corruption report. Cancelled/incomplete checks preserve earlier pass records and
never clear known damage. Reports live outside snapshots in
`verification/reports/<id>.json`; `verification/state.json` indexes per-version
results and known damaged content. **View Last Report** displays details and
**Save Report...** exports a local JSON copy, including when the drive could not
store the report. Report persistence failures are shown explicitly.

Known damaged content blocks both full-project and assigned-work publications
that would reference it. There is no automatic repair, deletion, or merge. After
an explicit repair, rerun a check that covers the affected content to clear the
warning. Clearing unrelated checks cannot clear the error. Reports cannot be
treated as authenticated evidence against a malicious drive administrator.

Routine size checks cannot detect same-size damage in unchanged remote files.
Full checks and restores detect that damage by hashing contents. Every downloaded
file is still verified during restore, assigned-work download, and incoming
updates. A successful upload is not a full integrity check, and a successful
check is only evidence about the selected version at the time of the check.
Use updated clients across collaborating computers. Old clients cannot enforce
the new shared known-damage records. No existing backup format is rewritten.

This separation follows established backup patterns: [Restic structural versus
data checks](https://restic.readthedocs.io/en/stable/045_working_with_repos.html#checking-integrity-and-consistency),
[Borg archive/data verification](https://borgbackup.readthedocs.io/en/stable/usage/check.html),
and [Kopia snapshot verification](https://kopia.io/docs/reference/command-line/common/snapshot-verify/).
Resumable/subset checks, repair tooling, automatic schedules, and concurrent
publication during checks are not implemented.

The estimate's inventory count and logical total describe the whole project,
not the number or size of files to upload. **Estimated additional storage** includes
new contents and a conservative allowance for the version manifest. After publication, the version's
`manifest.json` records every included path, size and checksum.
**Review Shared Updates > Preview Changes** provides a file-level review for incoming changes; assigned
work also has submission/review status.

### Review Outgoing Changes

After **Estimate Transfer Size**, choose **Review Changes...**. This read-only table
uses the estimate's cached comparison, without hashing local files again or uploading.
Files are grouped by video; shared project metadata has a separate **Project files**
group. Search and change-type filters narrow the list. Large groups load their rows
only when expanded. Each row shows the project-relative path, change type, annotation
scope/file type, and previous/new sizes. Full paths and exact byte counts are available
in tooltips; selecting a file also shows its full path below the table.

The baseline is explicitly **this contributor's latest published backup**, not an
arbitrary collaborator's version. If this contributor has never published, all
included files are **Added**, even when their contents can be reused from another
contributor's stored objects. The table describes changes to the snapshot's files,
not physical upload counts or counts of edited segmentation instances.

- **Added**: a new path in this contributor's version.
- **Changed**: the file's contents differ, not merely its timestamp.
- **Removed**: the path will not appear in the new version. Older backups remain intact.
- **Excluded**: an original video from the previous version is omitted because
  **Include original videos** is unchecked. It is not deleted from older backups.

Unchanged files are counted but not listed. Shared-video annotation files are
labeled **Shared across video**; frame masks/JSON/bboxes are **Frame annotation**.
Project settings include policy and split metadata; this is a file-level review,
not a field-by-field policy comparison or a visual before/after mask viewer.
Generated version manifests and initial project identity metadata are separate
from the existing local file inventory shown here.

Opening or closing the review never publishes and does not add a new mandatory
approval step. The normal publish confirmation remains required. Local edits or
changed settings invalidate the estimate and review. If the contributor's remote
baseline changes after estimation, publication requires a fresh estimate/review;
the existing collaborator-history conflict checks also remain active. Assigned
working sets still use their separate **Upload Assigned Annotations** flow; this table currently
applies to **Back Up Local Project** for full local projects only.

The legacy full-folder mode still copies complete trees, using copy-on-write
where available. Its estimates conservatively allow full ordinary copies.

### Opening a Backup on Another Computer

Open the completed revision folder using **File > Open Project**, or the
`project/` subfolder for an older full-folder version.
The app offers **Create Local Working Copy**, showing the project, contributor,
and publication date. Choose a **new** folder on local storage. Existing folders
are never overwritten. Files are copied into staging and verified against the
manifest before the working folder becomes available. Cancellation, missing
files, or checksum failure do not leave a loadable partial project.

The app refuses to open sync containers or `.pending-*` transfers for editing.
Opening published backups through Open, Recent Projects, or `--project` goes
through the same guard, before any managers or debug folders are initialized.
This is an application safeguard, **not filesystem read-only permissions**:
external programs and older app versions can still edit archive files. Use the
updated app on all computers and leave published files untouched.

Policy metadata is copied exactly. A complete policy opens in read-only review
with **Continue** and **Edit Policy**. Review acknowledgment stays on that computer;
it does not modify project metadata. An absent or partial policy opens **Finish
Configuring Annotation Policy** and keeps reminding until confirmed. Changing a
meaning remains an explicit project-wide edit, not a conversion of mask pixels.

### Outdated or Independently Edited Copies

Working copies record their known published starting points in the local-only
`annotations/.project-sync-state.json`. This is excluded from backups and restored
from the snapshot's manifest instead. Do not delete it to dismiss a conflict.
New manifests record the destination identity, starting versions, and device.

Before publishing, the app compares those starting points with **all contributors'
latest pointers** under a project lock. If another copy published in the meantime,
or a legacy copy has no verifiable starting point, publication stops with a warning.
An unchanged legacy copy matching its contributor's sole latest snapshot can
establish its baseline automatically. A changed legacy copy cannot safely do so.

Choose **Cancel**, or **Save Separate Backup** to preserve your work without
changing any latest pointer. Separate backups use the same routine verification;
their location is shown in the completion dialog and saved in sync settings.
They do not acknowledge, incorporate, or merge the other work. Further publishes
will continue to warn until you work from an appropriate verified version and
reconcile the changes. Even an unedited older copy cannot roll back a newer one.

For example, two computers restore version 5. Computer A publishes version 6.
Computer B's edits based on version 5 can be preserved as a separate backup, but
cannot silently replace version 6. Create a new local copy of the version you
intend to continue, then review/import the other contributions. Keep both originals.

This check is deliberately project-wide: even different frames can share masks,
instance identities, and policy. **No automatic annotation merging,
conflict-resolution editor, or shared live editing is implemented.** The explicit
local update described below can advance an unedited copy. Publishing locks
serialize transfers, not editing sessions. Older clients do not implement the
ancestry check; use the updated version everywhere.

### Review Shared Updates

This provides controlled updates in the other direction, not continuous two-way
folder mirroring. Continue to edit on local storage and publish completed versions
to the shared drive. Do not run two annotators against the same local folder.

1. Mount the shared drive and open your **local working copy**.
2. Choose **File > Review Shared Updates...**. Pending edits are saved locally and
   background writes/reads finish before comparison. No changes are published.
3. Choose a contributor's latest publication. If there are multiple contributors,
   none is selected automatically. There is no universal latest project version.
4. Select **Preview Changes**. The table compares local and published files with
   their recorded common starting snapshot. It lists additions, replacements and
   deletions, with separate notices for policy changes and video/split metadata.
5. Review and acknowledge the changes, then choose **Back Up and Update**.

An update is allowed only when the selected publication descends from this copy's
recorded starting snapshot and there are no divergent local edits. If both copies
changed, even on different frames/videos, applying is blocked. If only the local
copy changed, it is kept with nothing to download. Identical content on both sides
can advance the recorded history. Unknown, ambiguous, older or unrelated starting
histories cannot be force-overwritten: restore a new local copy and review/import
the other work instead. Changing the selection or refreshing invalidates the preview.

Before applying, the app stages only incoming changed files, verifies their
checksums, and preserves the previous versions of affected files. Unchanged media,
models, and generated results are **not copied or moved**. The app rechecks local
files and publication history before replacing anything. A persistent journal
blocks editing until an interrupted multi-file update is completed or recovered.

Rollback bundles default to `$XDG_DATA_HOME/BeeWhere/project-rollbacks`, or
`~/.local/share/BeeWhere/project-rollbacks` when XDG_DATA_HOME is unset. This is
outside the working project and normally outside Dropbox. **Settings > Project
Sync > Local rollbacks** can select another existing folder; blank restores the
default. Do not select a folder inside the working project or sync archive. An
unavailable custom rollback drive stops the update instead of silently falling
back to another location. Each bundle has:

```text
bumblebox-before-update-<UTC-timestamp-and-id>/
  manifest.json
  before/<previous versions of affected files>
```

The working folder keeps its original name. Models, training outputs, and other
local-only files are preserved; incoming publications do not update them. When a
publication excludes original videos, local originals are kept. Existing COCO
exports and `dataset_info.json` are invalidated, but remain in the rollback bundle.
**Re-export COCO before training.** All caches and the displayed
frame are reloaded, without reattributing imported annotations to the local user.

The preview reports replacement and rollback bytes separately. Disk checks also
allow recovery workspace and metadata, checking both drives if rollback storage
is elsewhere. An annotation-only edit no longer stages the entire media dataset.
Moving/deleting a large media file still needs space for that affected file.
Managed local files are still hashed for safe comparison; historical manifest
traversal retains only history metadata, not every old file inventory in RAM.

Use **File > Local Rollback History** to inspect bundle sizes, undo a selected
update, or select older bundles for deletion while retaining the newest N (default
3). Nothing is automatically deleted: cleanup needs explicit confirmation.
Undo requires the managed project to still match that update, so newer annotations
are not overwritten; it creates another rollback bundle for the reverse change.
Unrelated models/results can change without preventing undo. These partial bundles
cannot be opened as standalone projects. Full rollback copies created by the older
updater still restore through Open Project, but are not automatically cleaned up.

To test on two computers, publish version A, restore it locally on the second
computer, then edit and publish version B on the first. Leave the second copy
unedited and use Review Shared Updates to receive B. As a separate test on disposable
copies, edit both before publishing: the second copy's update should show the
conflict and disable applying. Use the new app version on **both** computers.

## Connection Failures and Recovery

A missing destination or identity marker stops sync; the app does not create a
replacement folder at a disconnected mount point. Local saving still works.
Reconnect the drive and retry. Cancellation waits for the current filesystem
operation, which can take time if the operating system is waiting on a network
mount. Authentication failures and disk-full errors are reported, not treated as
successful syncs.

After an application or computer crash, an unpublished `.pending-*` folder or
`.sync-lock` may remain, along with a destination-root
`.project-<full-project-id>.sync-lock`. Confirm that no sync or integrity check is
running before removing stale locks and retrying. Do not remove completed revisions or
an active contributor's lock. Unpublished files are not referenced by `latest.json`.
There is no automatic history pruning or merge wizard yet.

Do not delete shared objects or old revision manifests by hand. Manifests are
needed for ancestry checks and objects may be referenced by many publications.
Cancellation can leave verified, unreferenced objects; subsequent publishes can
reuse them. Remote garbage collection and reviewed legacy-archive compaction are
not implemented. Local Rollback History cleanup never deletes published history.

For a local update, cancellation or a failed verification leaves the old working
copy in place. Replacement failures attempt to restore it immediately. A crash
can leave a sibling `.bumblebox-update-<id>.json` recovery marker and staging folder.
Opening or publishing that project is blocked until recovery. After confirming
that no other annotator is still updating that local copy, reopen it through
**File > Recent Projects** (even if the working folder is temporarily missing).
The app offers recovery: restore the original if replacement did not finish, or
verify the installed update if it did. Unexpected post-interruption changes stop
recovery for manual review. Do not delete the marker, staging tree, or backup to
bypass this check. It is interruption recovery, not a guarantee against physical
disk failure; retain independent backups.

## Next Steps (Not Implemented)

- A conflict-review UI for same-video edits, deletions, policy and split changes.
  Ordinary full-project updates remain conservative. The separate Assigned Work
  workflow can combine changes to different assigned videos with explicit scope
  and ancestry checks; it does not merge edits to the same video.
- A policy-level **Not used in this project** state, with safeguards for already
  labeled categories and required setup when an unused category is activated.
- Remote garbage collection/retention and legacy archive compaction, preserving
  manifests and all referenced contents. Local rollback cleanup is already available.

Regression tests use disposable projects only: `scripts/test_backup_safety.py`,
`scripts/test_project_sync.py`, `scripts/test_contributor_sync_gui.py`,
`scripts/test_project_update.py`, `scripts/test_project_update_gui.py`,
`scripts/test_compact_sync.py`, `scripts/test_compact_sync_gui.py`,
`scripts/test_backup_integrity.py`, `scripts/test_backup_integrity_gui.py`, and the
annotation-policy suites. No real lab-drive content is modified by these tests.
