# Shared Contributor Profiles

Profile sharing makes the same contributor identities available on connected lab
computers, without exporting and sending JSON files. It is optional and independent
of project backups, annotation transfers, assignments, and project-director roles.

## First-Time Setup

On **each computer**, mount the shared drive, then:

1. Open **Profile Sharing...** in the contributor chooser. It is also available in
   **Settings > Project Sync** and the Home assignment notice.
2. Enable **Share contributor profiles** and choose the shared archive root, for
   example `/mnt/researchdrive/crall2/Crall_Lab/bumblebox_annotator_projects`, not a
   particular project's folder. Paths may differ between computers, provided they
   point to the same shared folder.
3. Under **Existing profiles to share**, check only the identities you intend to
   make available. Existing profiles are not automatically selected on first setup.
4. Leave **Automatically share new profiles created on this computer** checked to
   publish new identities automatically. Uncheck it for discovery-only use or to
   select each identity explicitly later.
5. Test the connection and Save. A first-time connection must pass the write/read
   test. No project, frames, models, or annotations are uploaded by this action.

The app suggests the configured project-sync folder, but profile sharing has its
own saved connection. Changing a project backup destination does not silently
publish your profiles to a new location. Changing the profile-sharing folder is an
explicit setup action and does not redirect project backups.

## On Another Computer

Open the app and choose your tile after the shared-profile check. You keep the same
contributor ID, but get a new editing session and that computer's own device ID.
Do not create another profile merely because you are using a different computer.

Profiles refresh when the chooser opens, through its refresh button, and every
minute while the app is running. New identities appear in the chooser and the
director's assignment choices without replacing an active session or unsaved
assignment choices. The last-selected person, recent projects, device identity,
and application preferences are never shared.

The chooser keeps a local cache. If the drive is disconnected, cached profiles and
local annotation work remain available; newly created profiles wait for a later
successful refresh. The app reports unavailable storage rather than treating it
as an empty directory or creating a replacement mount folder. An already configured
connection can be disabled or its publishing choices adjusted while offline.

**Hide Profile** hides a tile only on this computer. **Show hidden profiles** exposes
hidden tiles so you can select one and use **Show Profile**. Hiding, disabling sync,
or unchecking publication never deletes a shared identity or reattributes labels.

## Identity and Safety

- Shared files contain only format information, name, and stable contributor UUID.
  They do not contain credentials, device/session details, or project roles.
- Names are not unique. Same-name profiles remain separate, with a short ID shown
  on their tiles. Creating another person with an existing name requires confirmation.
- Existing IDs are never silently renamed or overwritten. Conflicting names for
  the same ID are reported for manual reconciliation; existing attribution is kept.
- Publication uses a shared lock and atomic JSON replacement. A busy/interrupted
  lock leaves uploads pending, while discovery of valid profiles remains available.
  Only remove `.contributor-profiles.lock` after confirming no sync is running.
- Invalid, oversized, or symlinked profile files are reported and skipped. The
  directory itself must not be a symlink. Valid neighbors remain discoverable.
- Export/Import Profile remains a fallback for computers without shared storage.

This is **not authentication**. Anyone who can access the directory can discover
and select its profiles. Drive permissions are the security boundary. Project
membership requests, approval, passwords, and account recovery are not implemented
by profile sharing. Directors still assign work explicitly; discovering a profile
does not grant a role or download a project.

## Storage

The archive root contains a small `contributor_profiles/<UUID>.json` for each shared
identity, separate from the versioned project directories. Files are keyed by UUID
so simultaneous creation of different same-name people cannot overwrite either.
Unchanged profiles are not rewritten on subsequent checks. Local preferences use
an ID-keyed cache, migrating the previous name-keyed list without changing IDs.
New profile-directory permissions inherit the archive root's access and group;
JSON files retain its group/other read access instead of becoming private to the
creating account. Existing directory permissions are never changed. The shared
folder must permit contributors to create files; profile sharing does not grant
additional drive access or repair administrator-controlled permissions.

## Verification

`scripts/test_shared_profiles.py` tests isolated temporary drives and computer
settings, including two-computer discovery, offline queuing, duplicate names,
opt-in publication, cancelled setup, stale results, corrupted files, locks, and
local hiding. Real research projects and shared-drive contents are not used.
