# Project Home and Workspaces

The application opens on Project Home instead of an empty editor with model
toolbars. Every launch first shows saved contributor profiles as side-by-side
tiles with initials and names. Click your tile to start your session; additional
profiles wrap into rows and scroll. The last-used profile is highlighted, not
automatically opened. Arrow keys and Enter also select a profile. **Manage
Profiles...** contains profile creation, local removal/restoration, and an Advanced
section for import/export. There are no connection or transfer buttons in the
chooser. Closing it exits without opening a project or changing the saved
selection. Each confirmed profile starts a new session ID before Project Home or
automatic project reopening. The name remains visible in the header; select it
to switch contributors or import/export your own profile. Profiles are attribution
identities, not authenticated accounts.

A new computer first offers **Connect to a Lab...** or **Use Locally**, once per
Linux login account. Existing installations keep their saved setup. Later changes
are available in My Work or Settings, outside the profile chooser.

On returning launches, the sequence is **choose profile, check saved drive access,
open Home or the last local project**. An available drive needs no extra click.
If it is unavailable, a separate screen offers **Sign In to Drive...**, retry,
and **Continue Locally**. Sign-in uses the existing Kerberos terminal, not the
contributor name or an in-app password form. Successful sign-in is followed by
another access check; it is not assumed to prove drive availability.

The read-only startup check verifies the saved lab root's identity and directory
access. It does not upload, download, merge, or verify every project's permissions.
It runs in a separate process with a 15-second limit. Continue Locally is available
even while checking. This pauses background profile, assignment, shared-update,
and review-status checks for the session; local annotation and saving remain
available. Reconnect using Sign In to Drive or the check button on Home. The saved
lab configuration is preserved and checked again on the next launch. Downloads
and uploads still require working drive access and their own permissions checks.

See [shared identities and project discovery](shared_profiles.md). Selecting the
active person stays explicit. Same-name identities show distinct IDs. Removing a
profile from the chooser is local, reversible, and does not delete it from the
shared folder or alter annotation ownership.

## Everyday Flow

- **My Work** starts with one **Projects** picker combining shared membership and
  recent local projects for the selected contributor. Selecting a project previews
  its assigned videos, mask tasks, frame counts, role, and local availability; it
  never downloads or opens a project automatically. Shared and local entries with
  the same project identity are combined, not projects that merely share a name.
  If several local copies exist, a **Local working copy** selector keeps those
  folders explicit and does not combine or move their annotations.
  **Open My Assigned Work** opens available assigned
  videos in Annotate, shows the video/frame sidebars, clears conflicting filters,
  and selects an assigned frame. This also works in a full local project: the
  **Only my assigned videos** checkbox limits the view without converting or
  deleting project data. Uncheck it to browse the other local videos. Assignments
  reflect the last Home check, not a live access-control rule.
  **Download My Assigned Work...** is the primary action when no local copy exists;
  **Download / Update Assigned Work...** remains separate when a copy exists but
  new assignments or frames may be missing. The existing managed-download workflow
  reuses the contributor's working copy and protects local edits. Stale, missing, wrong-project,
  or wrong-contributor shortcuts do not silently open unrelated work.
  Projects without personal assignments offer **Open Project** when a local copy exists.
  Directors can use **Manage Assignments...** for the selected project.
  Merely creating a lab identity does not enroll it in projects.
- The assignment table's **Shared Status** column checks the selected project's
  published review state automatically, using the same Home refresh and periodic
  checks as project discovery. It also shows completed assigned-frame counts when
  a current published summary exists. Shared-video tasks are labeled **Video task
  finished/unfinished**, not misleading frame counts. The check time is shown;
  failed or offline checks say **Unavailable** or **Not checked**, never zero
  progress. Old or stale summaries remain unknown without scanning mask pixels.
  This works before downloading a local copy and does not change any annotations.
  **In Progress** can mean that assignments were downloaded; it is not proof of
  drawing or completion. Edits and Mark Finished records saved only on another
  computer are invisible until uploaded. In downloaded assigned working copies,
  Mark Finished queues a verified upload and automatic review submission when the
  complete assigned task is finished. Full-project backups remain manual.
- The picker retains the selection across background refreshes. **Shared Drive
  Settings...** changes the connection for profiles, assignments, and backups;
  connection status and sign-in remain separate from project actions. Offline,
  downloaded assignments remain available and are explicitly marked as not checked
  against the shared drive, rather than being described as current remote assignments.
- **Project Options** contains **Open Local Project...**, **Create New Project...**,
  and full-backup recovery. They no longer compete with the selected project's
  primary action. The page has a light palette, project/detail columns on larger
  screens, and a vertically stacked layout on narrower screens. The editor's theme
  and all data-transfer rules remain unchanged.
- **Reopen my last project on startup** is an optional device preference. When
  checked, startup skips Home; uncheck it on **Home / My Work** to start there.
  Explicit `--project` or `--video` launch arguments also bypass Home. Project
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
copies upload their saved changes automatically, using a persistent local queue.

Question-mark buttons beside transfer actions open keyboard-accessible help,
including when the action is disabled. **Back Up Local Project** stores managed
files present locally, not missing files from another published version.
**Upload Assigned Annotations** preserves undownloaded shared files and uploads
permitted annotation changes. This File-menu command remains as a manual fallback;
the everyday assigned-work UI offers **Retry Upload** instead of separate upload
and submit buttons.

### Automatic Assigned Work and Computer Reports

In a downloaded assigned copy, contributors only need **Mark Finished** for the
assigned mask task. Saved edits queue an upload after 30 idle seconds (checked
every 10 seconds); completion requests an earlier attempt. Attempts are at least
one minute apart. Transfers briefly pause editing in a cancellable progress dialog
to protect the saved snapshot. They never start during a stroke or while saves
are outstanding. Only changed annotation content is transferred, with existing
conflict checks and upload verification. No automatic merging or downloading is
introduced. Full-project/director backups still use the explicit backup workflow.

After verification, all required assigned frames must have matching completion
records for the assigned task before the video is made ready for review. A
shared-video task needs one video-level completion, not bee masks on every frame.
Unknown tasks, missing evidence, changed masks, or pending import reviews cannot
be auto-submitted. A director must set a specific task on older "All mask types"
assignments. Editing completed geometry invalidates completion; the next upload
withdraws the old ready state until the task is finished again. Manual submission
remains available in File for exceptional workflows, including older clients.

Offline work stays in a hidden local outbox and resumes when that working copy is
reopened and connected. **Continue Locally** pauses attempts until reconnecting.
Conflicts stop automatic attempts and require director attention; cancellation
pauses attempts until Retry Upload or the next launch. Closing or switching away
from a project attempts the remaining upload and session-end report; an unavailable
drive never turns local completion into a verified submission. Keep the local copy
until it reports a verified upload.

The selected project's Home detail and the header show a persistent **ready for
review** count for directors, refreshed approximately once a minute while the app
is open and again on launch. It is derived from shared submissions, not transient
notifications; an unchanged accepted video does not become new work just because
another video was uploaded. Acceptance remains a director action. There is no
email or system notification service.

With a shared project open, each computer attempts a tiny activity report on open,
every 30 minutes, after an automatic upload, and at session end. The shared project's
`activity/` folder stores one atomically replaced JSON record per profile, device,
and working copy, not an ever-growing file per heartbeat. These records contain
display names, stable IDs, timestamps, the last edited video, and queued/verified
upload state. They contain no passwords, masks, images, or local paths. Local
outboxes are excluded from project backups.

Home shows **last heard from**, last local edit activity, and session end where
reported. After 65 minutes without a heartbeat from an open session, it says
**connection status unknown**. An idle timestamp is not proof a machine is offline:
the app may be closed, asleep, or disconnected. Offline session-end timestamps
remain local until that copy reports again. Activity is advisory and never grants
ownership, resolves conflicts, or establishes that annotations were uploaded;
the separate Shared Status column uses verified publications.

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

## Video Browsing

Annotate labels its video list **Local working copy**, with the project name and
folder path in the tooltip. Assignments labels its source **Shared version**, with
publisher and publication time. These are deliberately different data sources.
Its separate last-checked time and newer-publication notice never imply local and
shared annotations match. The source header can show the exact archive location
and open a read-only, video-grouped comparison of local saved annotations against
the displayed shared version. See
[assigned_work.md](assigned_work.md#compare-local-and-shared-work).

The Annotate video sidebar and the Assignments table have an **Experiment / setup**
filter and a **Sort** menu. Experiments use the existing imaging setup names, not
new project folders. Filtering does not change assignments, calibration, annotations,
or the current canvas. The experiment filter is exact: the active video's list item
can be hidden while its image remains open, with a notice below the list. Hidden
videos are removed from batch selections. All experiments restores the full list.

**Recently modified** sorts newest-first across splits using local annotation-file
and annotation-folder dates, including shared masks and completion records. Dates
are checked in the background without opening masks, and refreshed after saves;
the sidebar refresh button also rechecks them. Unknown dates sort last, with name
as the tie-breaker. **Default order** restores the original split/setup grouping
in Annotate, or alphabetical order in Assignments. Copying/importing/downloading
files can affect dates; this is not a shared contributor edit history. See
[assigned_work.md](assigned_work.md#select-work-by-mask-type) for details.

## Local Saving and Shared Work

### Mask Work in the Annotation Sidebar

The video list has **Work on**, **Show**, and **More filters** controls, matching
the assignment table. Each video shows mask availability and completion for the
selected type; each frame shows its work status. These are **local saved** results,
not the published backup. Unsaved edits are identified separately. The summary
refreshes in the background after saves; the refresh icon also checks external
changes. Before the initial scan finishes, videos and frames show **Loading...**
rather than Unknown. Pending videos stay visible under work filters until their
status is available; experiment filtering still applies. **Unknown** after the
scan means the saved status could not be determined, not that masks are absent.
Assignments instead uses published summaries: **Not scanned** there means an
explicit **Scan Mask Status** is needed, and **Scanning...** appears while it runs.
The current video remains visible even if it no longer matches a work
filter (the experiment filter remains exact). Alternating backgrounds separate
the visible video rows, including after filtering or sorting.
Filtering never changes mask visibility, the drawing category, zoom, or training
selection.

**Show** offers All Work, Not Started, Not Checked, and Finished. Not Started is
untouched work for the selected mask type. Not Checked includes existing masks
without full completion, changed work, pending import reviews, and unknown status;
each row still shows its specific status. Finished also includes checked absence.

Choose a mask type and use **Mark Finished** after checking the current frame.
For a shared-video category, the confirmation explicitly applies to the shared
mask once for the whole video. With no mask, confirmation records **Checked: none
present**, rather than treating a checked negative example as permanently unfinished.
**Reopen Work** removes that completion. The arrow button advances to the next
unfinished frame in the existing frame list, wrapping within the current video;
sampled frame numbers remain the original source indices.

Completion is saved in `annotations/work/<video>/frame_<index>.json` or
`video_work.json`, with contributor identity, timestamp, and a category-specific
mask signature. Editing that category's geometry or changing its policy requires
rechecking; unrelated category edits do not invalidate its completion. Existing
labels are never automatically marked finished. Unsupported/missing pixels and
unconfirmed annotation policies must be resolved first.

This is distinct from **Submit Current Video for Review**, director acceptance,
and imported-frame review. Marking a shared mask finished does not approve all
imported frames. Existing import-review and training/export checks remain in force.
Completion records are included in backups and assigned-annotation uploads.

Annotation edits are saved locally after about 2.5 seconds of inactivity, after
the stroke finishes. Autosave snapshots the active mask without ending the edit
or clearing brush undo. It waits for a pending frame save before queuing another
snapshot. Local saves may still take time for large video-wide masks. Manual
Save remains available. Unsaved changes, Saving locally, Local save failed,
and Changes to upload are distinct states.

Closing saves pending local annotation edits and waits for outstanding writes.
There is no discard-on-exit workflow: use Undo to revert an unwanted edit before
closing. A failed local save keeps the application open. When a linked full project
has unpublished annotation changes, the closing dialog offers Upload and Close,
Close with Work Saved Locally, or Cancel. A cancelled/failed upload, or a separate
conflict backup, does not count as a successful update of shared work. Assigned
copies instead finish the automatic queue, or retain it locally when offline or
cancelled, without a separate upload/submit choice.

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
blocked. Saved assigned changes upload automatically; completed assigned tasks
are submitted after verification. File-menu manual upload/submission commands
remain available for exceptional workflows. Downloaded source frames remain on
disk for offline work.

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
and missing setup are distinct states. **Connect to Lab** connects the profile
directory and project discovery; it does not add membership, assign work, or upload
annotations. The director adds people explicitly in **Assignments > People**.

## Deliberate Limits

- **Download Assigned Work** refreshes the persistent local assigned copy and adds
  new assignments. Unsent edits are kept; competing same-video changes, changed
  project settings, and changed assignments with unsent edits require director
  reconciliation. Older scattered copies are not merged automatically.
- There is no streaming image cache, automatic eviction, cloud login, or automatic
  merging of masks. Only assigned material is downloaded; local storage cannot
  consist solely of changed labels because source frames are required to edit.
- Pending annotation-policy setup and changed-policy review remain explicit.
  Remembering startup preferences does not bypass these checks.

## Brush Performance Check

Manual brush/eraser editing uses CPU-side mask arrays and Qt painting, not CUDA
model inference. Small strokes now accumulate a bounding rectangle and update
only that region of the RGBA overlay. Mask pixels, stroke-start guards, protected
regions, and undo semantics are unchanged. Qt still receives a full pixmap, so
this does not eliminate all image-size-dependent costs.

The scrollable canvas is fixed to the image bounds. Cursor previews, bounding
boxes, and labels crossing an image edge must not enlarge it or shift the view.
Brush-cursor motion is coalesced to the latest position at 16 ms intervals;
every drawing/erasing movement still reaches the stroke handler. Cursor geometry
and colors are reused until the brush size or tool changes. Empty or
horizontal-only scroll events do not change zoom.

Run `python scripts/benchmark_brush_overlay.py` from the annotator environment to
compare machines using the same synthetic 4096 x 3072 frame. It writes no project
files and reports median/p95 CPU-side stroke + overlay times, with boxes enabled.
It does not measure tablet input, screen presentation, autosave, or frame loading.
Two same-machine before/after runs measured approximately 35 ms before and
11-17 ms after this change; these are not a guarantee of tablet latency on other
hardware. Test both a mouse and pen on the second computer with locally stored
frames before attributing remaining stalls to its GPU.

`scripts/test_canvas_view_stability.py` checks view stability with edge overlays,
mask toggles, cursor bursts, frame changes, and empty wheel events. These tests
use synthetic input; physical tablet/driver behavior still needs testing on the
affected computer.

## Testing

`scripts/test_workspaces.py` covers Home/resume, assigned-work restrictions,
workspace transitions, local autosave/undo, offline and conflicting shared status,
director transfer, and closing with unpublished work. Screenshots are generated
at 1400x900 and 1024x768 using temporary synthetic projects, not research data.
Initialize QApplication before loading GUI tests, as shown in [setup.md](setup.md).
