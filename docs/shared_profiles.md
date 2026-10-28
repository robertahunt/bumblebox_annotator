# Shared Contributor Profiles

Connect to Lab makes the same contributor identities available on connected lab
computers without exporting JSON files. One identity represents one person across
projects. Membership, assignments, and the single director role belong to each
project, not to the person's profile. Connecting is optional.

## First-Time Setup

On **each computer**, configure access to the shared drive once, then:

1. Open **Connect to Lab...** in the contributor chooser or **My Work**. It is also
   available in **Settings > Project Sync** as **Connect Profiles and Projects to Lab...**.
2. Choose the shared archive root, for
   example `/mnt/researchdrive/crall2/Crall_Lab/bumblebox_annotator_projects`, not a
   particular project's folder. Paths may differ between computers, provided they
   point to the same shared folder.
3. Review **Existing profiles to share**. The current/last-selected person is
   preselected on first setup; other existing private profiles are not. Confirming
   the connection consents to sharing the checked names/IDs and future new profiles.
4. Test the connection and choose **Connect to Lab**. It must pass the write/read
   test. No project, frames, models, or annotations are uploaded by this action.

This configures both profile sharing and the default project connection, preserving
backup options and per-project opt-in. If previously separate connections point to
different folders, replacing them requires confirmation. Changing an advanced
backup destination alone does not silently publish profiles to that destination.
If the two saved folders differ, **Use Project Sync Folder** copies the project
destination into the form without saving it. Choose the parent archive root, not
a folder such as `2026_CV4E--d10f4771`. Published-project subfolders are rejected,
even if an older setup accidentally created a root marker inside one. Existing
files and markers are not moved or deleted. An explicit **Test Connection** shows
a result that remains open until acknowledged; testing does not save the connection.

## Adding Kevin Without Becoming Kevin

In your project's **Assignments > People**, choose **Add Contributor...**. Select
Kevin's existing lab identity, or choose **New Profile** and enter his name. Adding
him shares project membership immediately; a connected lab also queues his profile
for automatic discovery. Your active profile remains unchanged. Assign videos as
a separate action. Merely discovering Kevin in the lab directory does not enroll
him in all projects, grant a director role, or download any data.

On the other connected computer, Kevin selects his tile, then sees his projects
in **My Work**, including projects where he is a member but has no assignments yet.
He downloads assigned work explicitly after seeing the size estimate. Existing
directors and assignees are recognized as members without reconfiguration.

## On Another Computer

Open the app and choose your tile after the shared-profile check. You keep the same
contributor ID, but get a new editing session and that computer's own device ID.
Do not create another profile merely because you are using a different computer.

Profiles refresh when the chooser opens, through its refresh button, and every
minute while the app is running. New identities appear in the chooser and the
director's **Add Contributor** picker without replacing an active session or unsaved
assignment choices. The last-selected person, recent local projects, device identity,
and application preferences are never shared.

The chooser keeps a local cache. If the drive is disconnected, cached profiles and
local annotation work remain available; newly created profiles wait for a later
successful refresh. The app reports unavailable storage rather than treating it
as an empty directory or creating a replacement mount folder. **Disconnect This
Computer** works offline, stopping automatic profile sharing and clearing the
default project connection without deleting any local or shared data.

## Kerberos Drive Login

On Linux, **Sign In to Drive...** opens GNOME Terminal or xterm and runs
`kinit <username>`. Enter the actual Kerberos username, which need not match your
contributor display name. Enter your password only in the terminal. The app never
reads, stores, or logs passwords or tickets; only the username is remembered
locally. It does not invoke a shell, sudo, mount commands, or alter drive settings.
The executable lookup checks PATH, standard system locations, and the current
Python/Conda installation, including its base installation. A Conda environment
does not need its own copy of kinit if the base installation provides it.

The terminal keeps the outcome and any Kerberos error visible. Press Enter there
after reading it to close the terminal and retry the connection or assignment
check. A successful kinit exit does not by itself count as successful drive access.
For the lab's automounted research drive, valid tickets should make access possible;
other machines may still need administrator-configured mounting or permissions.
Terminal login is always an explicit click, never a recurring background popup.
Without kinit or a supported terminal, sign in manually and retry. Closing the app
or its connection dialog does not close an already-open login terminal.

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
join requests, approval queues, passwords, and account recovery are not implemented
by profile sharing. Directors add project members and assign work explicitly; discovering a profile
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
local hiding. `scripts/test_lab_connection.py` covers unified setup, enrollment,
unchanged active identities, and mocked terminal launches. Real research projects,
shared-drive contents, and Kerberos credentials are not used in these tests.
