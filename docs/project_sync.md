# Contributors and Optional Project Sync

## Contributor Sessions

Starting `python main.py` asks for a contributor name, with previously used names
available in the dropdown. Confirm the name each session, particularly on a
shared workstation. Cancel exits without starting an editing session.

The active name appears in the status bar. **Settings > Change Contributor** saves
pending work under the previous contributor before starting another session.
Background saves retain the identity with which they were queued.

Contributor names and stable IDs are stored in local application settings, not
inherited from an opened project. A new name gets a new ID; selecting a remembered
name reuses its ID but creates a new session ID. Names are self-declared, not
authenticated accounts. The same name first entered on another installation gets
a different ID; there is no cross-machine profile/account service yet.

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

The first app launch offers optional sync setup. **Not Now** leaves local editing
and saving fully available. Setup is also available at **Settings > Project Sync**.

1. Choose an existing local, USB, or mounted network folder.
2. Select **Test Connection**. The app writes, reads, and removes a uniquely named
   probe, then records a small destination identity marker.
3. With a project open, select **Enable sync for this project**, then **Save**.
4. Use **File > Sync Project Now** or **Sync Now** in the settings dialog.

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
folders are excluded. With original videos included, the copy can be opened as a
normal project. Without them, it is a partial backup and may require restoring the
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
    contributors/August--0a0b573d/
      .bumblebox-contributor.json
      latest.json
      revisions/<UTC-timestamp-and-id>/
        manifest.json
        project/
          annotations/
          frames/
          input_data/
```

Folder names combine a filesystem-safe name with the first eight characters of
the ID. Identity markers retain the **full UUID**; shortened IDs are only labels.
Name collisions fall back to the full ID. Changing a display name later reuses
the existing folder, preserving links and history instead of creating another
copy. The source project name comes from `annotations/project.json`.

Existing UUID-only project and contributor folders are renamed automatically on
their next sync. This moves directories, not their contents: revisions, manifests,
and checksums are unchanged, but old absolute paths must be updated. Other
contributors' histories move with the project; their own folder names migrate
when they next sync. Close older app versions before the first migration, and
use the updated app on all collaborating installations. Duplicate folders with
the same identity are reported, never silently merged or overwritten.

The completion dialog gives the exact project path. The manifest includes human
project/contributor names and file checksums; `latest.json` points to the most
recent completed revision. Settings show the latest success time or failure.

Sync is manual and one-way. Local annotations are saved and queued writes are
drained before a modal progress dialog pauses editing. Files are checked and
verified in an unpublished staging directory. Source changes during copying
abort publishing; do not edit the local project from another process during sync.
Only complete revisions are published, and prior revisions are not automatically
deleted. A stable project-ID lock serializes syncs to that project, including
folder migration; a contributor-specific lock also protects each copy. Sync never
modifies another contributor's revision contents.

Unchanged projects reuse their existing revision. For changed projects, changed
files are copied from the source; unchanged files are reused from the previous
destination revision using copy-on-write where supported, or ordinary copying
otherwise. **On drives without copy-on-write, each changed revision can require
another full project's worth of disk space and substantial network I/O**, including
unchanged videos. Checksums also require reading files. This first version favors
independent, reopenable revisions over a compact incremental archive. It is not a
guarantee that only changed bytes cross the network. Linux copy-on-write clones
keep writes independent between revisions ([FICLONE documentation](https://man7.org/linux/man-pages/man2/FICLONE.2const.html)).

Treat published revisions as backups: copy a revision's `project/` directory to
a local working location before editing it. The sync checks unchanged files in the
prior revision before reuse and refuses damaged or externally edited copies.
Collaborators can annotate the same videos in their own local projects, but there
is **no automatic annotation merging, conflict resolution, or shared live editing**.
Review and combine their contributions separately.

## Connection Failures and Recovery

A missing destination or identity marker stops sync; the app does not create a
replacement folder at a disconnected mount point. Local saving still works.
Reconnect the drive and retry. Cancellation waits for the current filesystem
operation, which can take time if the operating system is waiting on a network
mount. Authentication failures and disk-full errors are reported, not treated as
successful syncs.

After an application or computer crash, an unpublished `.pending-*` folder or
`.sync-lock` may remain, along with a destination-root
`.project-<full-project-id>.sync-lock`. Confirm that no sync process is running
before removing stale locks and retrying. Do not remove completed revisions or
an active contributor's lock. Unpublished files are not referenced by `latest.json`.
There is no automatic history pruning or restore/merge wizard yet.
