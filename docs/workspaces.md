# Project Home and Workspaces

The application opens on Project Home instead of an empty editor with model
toolbars. Every launch first shows saved contributor profiles as side-by-side
tiles with initials and names. Click your tile to start your session; additional
profiles wrap into rows and scroll. The last-used profile is highlighted, not
automatically opened. Arrow keys and Enter, or Continue, also select a profile.
New Profile and Import Profile are separate actions; Export Profile exports the
highlighted identity. Cancel exits without opening a project or changing the saved
selection. Each confirmed profile starts a new session ID before Project Home or
automatic project reopening. The name remains visible in the header; select it
to switch contributors or import/export your own profile. Profiles are attribution
identities, not authenticated accounts.

**Profile Sharing...** enables optional [shared identities](shared_profiles.md).
Connected computers discover the same profiles automatically; selecting the active
person stays explicit. Same-name identities show distinct IDs. Hiding a profile is
local to the computer and does not delete it from the shared folder.

## Everyday Flow

- **Choose a project** shows recent projects with one primary action directly below
  the list: **Open Selected Project**, or **Return to Project** for the project
  already open. The current contributor's last project is preselected. Paths appear
  below the action and in tooltips, not as competing project names.
- **Open Local Project**, **Download Assigned Work**, and **Create New Project**
  are separate choices below the recent-project section. With no recent projects,
  that empty section is omitted. The page uses a light palette without changing
  the annotation editor's theme and scrolls on smaller screens.
- **Reopen my last project on startup** is an optional device preference. Project
  workspace, model-panel selection, video/frame position, and saved zoom/position
  are remembered separately for each contributor/project on that device.
- **Annotate** keeps editing and frame navigation visible. Class/global visibility
  controls sit above the instance list. Model assistance is an optional panel,
  showing one model toolbar at a time. Additional tools reveals calibration,
  ArUco operations, protected-area clearing, and bulk deletion. Hiding a model
  panel does not unload its model or change annotation visibility.
- **Review** adds imported-frame review controls and tracking sequences to the
  same editor. The annotation policy remains visible in both editing workspaces.
  It is still possible to correct labels during review.
- **Train** opens the existing model training, validation, and batch inference
  dialogs. Their algorithms, export eligibility, train/validation splits, and
  parameter defaults are unchanged.
- **Manage** opens project imports, annotation policy, video assignments,
  publishing, update previews, and rollback history.
- **Assignments** is a dedicated full-project workspace for director setup,
  contributor profiles, searchable/bulk video assignment, and review status.
  It is hidden for confirmed non-directors and assigned-only copies. Before a
  director is established, it exposes the explicit setup action and prerequisites.

For full-project backups, choose **Estimate Transfer Size** in Manage, the File
menu, or Project Sync settings first. **Back Up Local Project** stays disabled
until the estimate succeeds. The estimate saves pending local annotations but
does not upload project files. It reports the file count, logical size, estimated
additional storage, and free destination space. Edits, profile/project changes,
or changes to destination, included videos, or storage mode require a fresh
estimate. Files are checked again before uploading; an external change blocks
publishing. Estimates are not remembered across app restarts. Assigned working
copies retain their separate **Upload Assigned Annotations** workflow.

Question-mark buttons beside transfer actions open keyboard-accessible help,
including when the action is disabled. **Back Up Local Project** stores managed
files present locally, not missing files from another published version.
**Upload Assigned Annotations** preserves undownloaded shared files and uploads
permitted annotation changes. It saves progress without marking a video ready.

**Recovery > Restore from Backup** on Home (also in File) is for recovery or
full-project setup, not daily submission of work. Select a completed published
version, check its contributor/date, then choose a new local destination. The
existing verified restore never overwrites a working folder. It downloads the
whole selected version, not only changed files. Newest timestamp does not imply
all contributors' changes have been combined; ask the director which version to
restore. There is deliberately no automatic "most up-to-date" restore or merge.

The Home icon or File > Project Home returns to project selection. Existing
menus and shortcuts remain available. Model assistance and workspace selection
do not silently change masks, classes, or the selected editing tool.

## Local Saving and Shared Work

Annotation edits are saved locally after about 2.5 seconds of inactivity, after
the stroke finishes. Autosave snapshots the active mask without ending the edit
or clearing brush undo. It waits for a pending frame save before queuing another
snapshot. Local saves may still take time for large video-wide masks. Manual
Save remains available. Unsaved changes, Saving locally, Local save failed,
and Changes to upload are distinct states.

Closing saves pending local annotation edits and waits for outstanding writes.
There is no discard-on-exit workflow: use Undo to revert an unwanted edit before
closing. A failed local save keeps the application open. When a linked project
has unpublished annotation changes, the closing dialog offers Upload and Close,
Close with Work Saved Locally, or Cancel. A cancelled/failed upload, or a separate
conflict backup, does not count as a successful update of shared work.

Opening a linked project performs a background, read-only status check. The
refresh icon checks again. Full copies use recorded publication ancestry; they
do not decide freshness from wall-clock timestamps. Review Shared Updates opens the
existing preview and conflict checks; it never automatically replaces files.
Assigned copies ignore changes to other assignees' videos, but flag changes to
their own videos, assignments, or shared project metadata. Offline/unreadable
storage is never displayed as up to date.

The local change check compares annotation/configuration files against their
baseline. It does not hash source images or original videos and is not a complete
backup-integrity audit. The publishing operation still performs its full checks.
Status describes the last check, not a continuously monitored remote folder.

## Assignments and Director

Download Assigned Work configures/tests a mounted shared folder when necessary,
then opens the existing selective assignment downloader. Assigned copies show
only Annotate; training, project administration, and full-project updates remain
blocked. Upload Assigned Annotations publishes drafts; Submit Current Video for Review is a
separate action. Downloaded source frames remain on disk for offline work.

The existing single coordinator is presented as the project director. Existing
roles remain unchanged. New directors must explicitly confirm **Set Project
Director** in Assignments; saving assignments without a director is rejected.
Profiles can direct one project and contribute to another. In Video Assignments,
Transfer Director Role transfers it to an imported contributor profile after
explicit confirmation and an optimistic revision check. It leaves assignments,
submission records, and annotations intact. Only the current director can transfer
the role. Reuse an existing shared identity or import its exported profile rather
than creating another profile with the same display name.

These are collaboration guards, not an access-control system: a full local copy
still provides local administration, and drive permissions remain the security
boundary. No new role is silently assigned to an existing project during startup.

Home checks assignments automatically after profile/project changes and when
reopened, without downloading or modifying project files. Confirmed empty work
shows a non-blocking notice once per session/shared folder; inaccessible storage
and missing setup are distinct states. **Profile Sharing** connects the profile
directory; it does not approve membership, assign work, or upload annotations.

## Deliberate Limits

- Assigned copies still require director reconciliation and a fresh working copy
  after incompatible updates or reassignment. In-place assignment refresh is not
  implemented; the UI does not offer a misleading generic Sync action for it.
- There is no streaming image cache, automatic eviction, cloud login, or automatic
  merging of masks. Only assigned material is downloaded; local storage cannot
  consist solely of changed labels because source frames are required to edit.
- Pending annotation-policy setup and changed-policy review remain explicit.
  Remembering startup preferences does not bypass these checks.

## Testing

`scripts/test_workspaces.py` covers Home/resume, assigned-work restrictions,
workspace transitions, local autosave/undo, offline and conflicting shared status,
director transfer, and closing with unpublished work. Screenshots are generated
at 1400x900 and 1024x768 using temporary synthetic projects, not research data.
Initialize QApplication before loading GUI tests, as shown in [setup.md](setup.md).
