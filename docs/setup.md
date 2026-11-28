# Setting Up Another Computer

This guide targets **Ubuntu 24.04 / Linux x86_64, Python 3.10.19**, with an NVIDIA
GPU for inference/training. The pinned Qt wheel requires glibc 2.34 or newer;
older Linux installations need a separately tested package set. It transfers the `integration/annotation-batch` development
branch, not `master`. No worktree is necessary on the new computer.

Use a new environment; do not alter a collaborator's existing environment.
Windows/macOS and other architectures need separate dependency and GPU checks;
the Linux constraints are not a validated cross-platform lockfile. An Apple
Silicon Mac has its own recipe under [Apple Silicon Mac](#apple-silicon-mac-untested),
which has only partly been tried on a Mac. The old
`install.sh` and `install.ps1` are legacy installers, not this setup procedure.

## Reference Versions

| Component | Version |
| --- | --- |
| Python | 3.10.19 |
| PyTorch / torchvision | 2.10.0 / 0.25.0 |
| PyTorch CUDA runtime | 12.8 (`cu128`) |
| Ultralytics | 8.4.16 (headless OpenCV distribution) |
| OpenCV | 4.13.0.92 (headless) |
| NumPy | 2.2.6 |
| PyQt6 / Qt | 6.10.2 |
| Albumentations | 1.4.24 |
| Optional SAM2 | `2b90b9f5ceec907a1c18123530e92e794ad901a4` |

[The constraints file](../constraints-linux-py310.txt) pins supporting packages
as well. It records package versions, not OS libraries, model weights, Git
history, or wheel hashes. Keep the application Git commit with your run records;
do not replace these pins with the latest versions during replication.

The reference environment previously had both OpenCV distributions installed.
This recipe intentionally uses only `opencv-python-headless` and
`ultralytics-opencv-headless`: the latter provides the same `ultralytics` Python
API without pulling in a second OpenCV/Qt installation. PyQt6 still supplies the
application's GUI. Do not install ordinary `ultralytics`, `opencv-python`, or a
contrib OpenCV package on top. See the [Ultralytics installation guide](https://docs.ultralytics.com/quickstart).

Pandas and carbontracker were declared dependencies but absent from the reference
environment. Their pinned versions and dependencies are included here rather
than claiming they were used in the original GPU preview. Carbon measurement
remains best-effort; an import check does not validate energy measurements.

## 1. Clone This Branch

Install Git and Miniconda first. From a folder where you want the checkout:

```bash
git clone --branch integration/annotation-batch --single-branch --depth 1 \
  https://github.com/robertahunt/bumblebox_annotator.git
cd bumblebox_annotator
git branch --show-current
git rev-parse HEAD
```

The shallow clone avoids downloading obsolete large checkpoints in older Git
history. Current checkpoints and research data are not part of this checkout.
Use a different destination name if `bumblebox_annotator` already exists; do not
overwrite an existing checkout or project. If working on code history later,
`git fetch --unshallow` retrieves that older history, including old large blobs.

## 2. Create the Environment

```bash
conda create -n bee_annotator python=3.10.19 pip -y
conda activate bee_annotator
python -m pip install pip==26.0.1 setuptools==80.10.2 wheel==0.46.3 \
  -c constraints-linux-py310.txt
```

If that environment name exists, choose another name and activate that name in
later commands. These commands do not install or change the NVIDIA driver.

## 3. Choose GPU or CPU

### NVIDIA GPU (Training and Batch Inference)

Run `nvidia-smi` first. It must identify the GPU successfully. The operating
system needs an NVIDIA driver compatible with CUDA 12.8; install it using your
OS/vendor's instructions and reboot if required. The CUDA number shown by
`nvidia-smi` need not equal PyTorch's bundled runtime version.

```bash
nvidia-smi
python -m pip install torch==2.10.0 torchvision==0.25.0 \
  --index-url https://download.pytorch.org/whl/cu128 \
  -c constraints-linux-py310.txt
```

This is the matching pair documented by [PyTorch](https://pytorch.org/get-started/previous-versions/).
Ordinary YOLO inference/training does not require installing the separate CUDA
development toolkit. Optional SAM2 extension compilation is a separate matter.

### CPU Only (Annotation and Limited Inference)

Use this **instead of** the preceding GPU installation:

```bash
python -m pip install torch==2.10.0 torchvision==0.25.0 \
  --index-url https://download.pytorch.org/whl/cpu \
  -c constraints-linux-py310.txt
```

CPU annotation does not require a GPU. This application's training workflows
require an NVIDIA card or an Apple GPU; CPU-only setup is not a workaround for
that requirement. CPU
inference is slower, and some model-specific workflows still require a GPU.

### Apple Silicon Mac (Untested)

**This section was written on the Linux reference computer, and only part of it
has been seen to work on a Mac.** On one Mac Studio (M5 Ultra, 96 GB) the app
started in an environment with the reference versions, found the Apple GPU and
trained on it; see [what the first training run on a Mac showed](#what-the-first-training-run-on-a-mac-showed).
The numbered checks further down have not been reported back.
What was checked from Linux: every pinned package has an
Apple Silicon build for Python 3.10, except the NVIDIA-only packages (never
installed on a Mac) and two small pure-Python SAM2 dependencies that build from
source; and the code that does not depend on CUDA was exercised on the CPU.
Treat work on a Mac as a test, not as production work, until those checks pass.

Requirements: an Apple Silicon Mac (M1 or later), macOS 13 or newer, and the
**arm64** Miniconda or Miniforge installer. An Intel Python running under
Rosetta cannot see the Apple GPU. Ultralytics carries workarounds for an Apple
GPU bug in macOS 14, so prefer a newer macOS if you can choose.

Follow steps 1 and 2 above, then use this **instead of** the NVIDIA or CPU
installation. The ordinary package index already holds the Apple GPU build:

```bash
python -c "import platform; print(platform.machine())"   # must print arm64
python -m pip install torch==2.10.0 torchvision==0.25.0 \
  -c constraints-linux-py310.txt
```

Continue with step 4 unchanged. The constraints file is named for Linux, but
its NVIDIA lines only apply to packages that something asks for, and nothing
does on a Mac. Once an installation works, save `python -m pip freeze` as the
Mac's own record; the Linux file does not prove a Mac resolves the same way.

In step 5, replace the CUDA check with this one:

```bash
python - <<'PY'
import platform
import torch
from torchvision.ops import nms
assert platform.machine() == 'arm64', 'Intel Python under Rosetta cannot use the Apple GPU'
assert torch.backends.mps.is_available(), 'Apple GPU unavailable'
x = torch.ones((128, 128), device='mps')
assert (x @ x).mean().item() == 128
boxes = torch.tensor([[0., 0., 10., 10.], [1., 1., 9., 9.]], device='mps')
scores = torch.tensor([0.9, 0.8], device='mps')
assert nms(boxes, scores, 0.5).numel() == 1
torch.mps.synchronize()
print('Apple GPU operations passed')
PY
QT_QPA_PLATFORM=offscreen NO_ALBUMENTATIONS_UPDATE=1 python -m unittest scripts.test_gpu
```

The application picks its device by itself: an NVIDIA card if PyTorch can reach
one, otherwise the Apple GPU, otherwise the CPU ([core/gpu.py](../core/gpu.py)).
Training needs one of the two GPUs and never falls back to the CPU. If the
Apple GPU fails or gives odd masks, keep everything on the CPU while you
investigate:

```bash
BUMBLEBOX_FORCE_CPU=1 python main.py
```

Operations the Apple GPU lacks run on the CPU instead of failing
(`PYTORCH_ENABLE_MPS_FALLBACK`, set at startup); PyTorch prints a warning naming
each one. On an Apple GPU, Ultralytics turns mixed precision off, so training
runs at full precision: expect more memory per image and longer epochs than on
an NVIDIA card. SAM2 training also runs at full precision there.

Check these on the Mac, in this order, before relying on it:

1. Load a YOLO checkpoint and predict one frame; compare the masks with the same
   frame and checkpoint on the Linux computer.
2. With SAM2 loaded, draw a box prompt, then propagate masks to the next frame.
   SAM2's video model stores its memory as `bfloat16`, which an Apple GPU may not
   accept; SAM2's own authors call their Apple GPU support preliminary.
3. Train a few epochs on a small dataset and compare losses and validation
   numbers with the same short run on the NVIDIA computer. They will not match
   digit for digit; they should be close. Train and score all models that a
   paper compares on one computer.
4. Run batch video inference on a long video and watch the `GPU ... allocated`
   lines in the log for steady growth.
5. Open one video on both computers and confirm the frame count and a few
   numbered frames are identical, since annotations are stored by frame number.

Not available on a Mac: the desktop launcher below (start the app with
`python main.py`), the terminal sign-in for the research drive (sign in with
macOS, then use Test Connection), and `scripts/recover_segmentation_training.py`,
which still accepts only an NVIDIA card.

#### What the first training run on a Mac showed

In November 2026 a whole-frame bee mask run at **Full resolution** (4096 pixels,
batch size 1, 224 training frames) on the Mac Studio took 4 to 6.6 seconds a
step. PyTorch held 108 GB of GPU memory on a 96 GB computer, so macOS was
swapping to disk. The run stopped in its fourth epoch with `shape mismatch: value
tensor of shape [16133] cannot be broadcast to indexing result of shape [23132]`.
That error comes from the step where Ultralytics matches each labelled bee to
the model's predictions: the Apple GPU counted one mask two ways. What sets it
off is not known; the same crowded frames had passed in the three epochs before.
Run on its own on that Mac afterwards, selecting and writing back through a mask
at the same table sizes (up to 107 bees) gave correct results every time, so
table size alone does not cause it.

The app now does four things differently. With them, the next full-resolution run on
the Mac Studio was reported to run well at about 60 GB of memory.

- **Matching on the CPU.** When training on an Apple GPU, the matching step is
  computed on the CPU ([training/apple_gpu.py](../training/apple_gpu.py)). On the
  NVIDIA computer the CPU route gives the same matches as the GPU, checked at
  4096 pixels with up to 107 bees, and takes at most 0.44 seconds a step.
- **A memory ceiling.** PyTorch is kept to the GPU memory macOS recommends for
  one program, and starts handing unused memory back at 80% of it. PyTorch's own
  default is 1.7 times the recommended amount, which is more than the Mac has.
  If a run now stops with `MPS backend out of memory`, it needs more than the
  ceiling allows. To raise it, start the app with both values set, the lower one
  no higher than the upper one:

  ```bash
  PYTORCH_MPS_HIGH_WATERMARK_RATIO=1.2 PYTORCH_MPS_LOW_WATERMARK_RATIO=1.0 python main.py
  ```

- **Errors in the terminal.** A training failure is printed in the terminal as
  well as shown in the training window. Before, it reached only the window.
- **No password prompt.** Energy tracking reads a Mac's power draw with
  `sudo powermetrics`. It is now skipped, with one line saying so, unless that
  command runs without asking for a password. Before, the prompt sat unanswered
  in the terminal for the whole run and nothing was measured.

#### What the second run showed

That full-resolution run (`bee_segmentation4`) started from the standard
pretrained weights and stopped after 226 epochs (about 54 hours, 15 minutes an
epoch), when early stopping (patience 50) found no improvement since epoch 176.
Scored afterwards on the NVIDIA computer against the November 3072-pixel model
(`bee_recovery_fp32_20261106`), each at its own image size, on the same frames:

| Validation frames | Model | Mask mAP50 | Mask mAP50-95 | Box mAP50-95 |
|---|---|---|---|---|
| 3 Nov (37 frames, 719 bees) | November, 3072 | 0.927 | 0.711 | 0.740 |
| 3 Nov (37 frames, 719 bees) | Mac, 4096 | 0.904 | 0.663 | 0.703 |
| 25 Nov (40 frames, 846 bees) | November, 3072 | 0.926 | 0.696 | 0.728 |
| 25 Nov (40 frames, 846 bees) | Mac, 4096 | 0.911 | 0.650 | 0.701 |

The November model had about 630 epochs of training behind it, all on frames
that leave out the held-out colonies; the Mac run was still improving slowly
when it stopped. To see what full resolution adds, train at 4096 with **Start
From** set to the November model (see the README's training section).

## 4. Install Application Dependencies

```bash
python -m pip install -r requirements.txt -c constraints-linux-py310.txt
python -m pip check
```

Run both commands inside the new environment. Expect `No broken requirements
found.` The pins allow the already-installed CPU or CUDA build of the matching
PyTorch version; this step should not replace it with a different release.

On Linux, Qt also needs OS display libraries. If launch reports a missing
`xcb`/EGL library, install the corresponding OS package rather than another
OpenCV wheel. On Ubuntu 24.04, commonly needed packages include `libegl1`,
`libopengl0`, `libxcb-cursor0`, and `libxkbcommon-x11-0`. Run the GUI from a desktop
session, not a display-less SSH terminal. `QT_QPA_PLATFORM=offscreen` below is for
automated tests only, not interactive annotation.

## 5. Verify Before Opening Research Data

First verify core imports and the OpenCV distribution:

```bash
python - <<'PY'
import cv2
import torch
import torchvision
import ultralytics
from importlib.metadata import distributions
from PyQt6.QtCore import qVersion
names = {d.metadata['Name'].lower().replace('_', '-') for d in distributions()}
assert 'ultralytics-opencv-headless' in names
assert 'ultralytics' not in names, 'Use a clean environment, not both YOLO distributions'
assert not names.intersection({'opencv-python', 'opencv-contrib-python', 'opencv-contrib-python-headless'})
assert hasattr(cv2.aruco, 'ArucoDetector')
print('OpenCV:', cv2.__version__, 'Qt:', qVersion())
print('Torch:', torch.__version__, 'torchvision:', torchvision.__version__)
print('Ultralytics:', ultralytics.__version__, 'CUDA runtime:', torch.version.cuda)
PY
```

For the **GPU setup**, also run an actual CUDA operation and torchvision kernel.
A `True` availability flag alone is not a complete test:

```bash
python - <<'PY'
import torch
from torchvision.ops import nms
assert torch.cuda.is_available(), 'CUDA unavailable: check driver and active environment'
x = torch.ones((128, 128), device='cuda')
assert (x @ x).mean().item() == 128
boxes = torch.tensor([[0., 0., 10., 10.], [1., 1., 9., 9.]], device='cuda')
scores = torch.tensor([0.9, 0.8], device='cuda')
assert nms(boxes, scores, 0.5).numel() == 1
torch.cuda.synchronize()
print('CUDA operations passed:', torch.cuda.get_device_name(0))
PY
```

From the repository root, run the focused synthetic/offscreen checks:

```bash
QT_QPA_PLATFORM=offscreen NO_ALBUMENTATIONS_UPDATE=1 python - <<'PY'
import sys
import unittest
from PyQt6.QtWidgets import QApplication
# Initialize Qt before importing GUI modules that select a Matplotlib Qt backend.
app = QApplication([])
suite = unittest.defaultTestLoader.loadTestsFromNames([
    'scripts.test_annotation_policy',
    'scripts.test_annotation_policy_gui',
    'scripts.test_batch_mask_exports',
    'scripts.test_aruco_detection_workers',
    'scripts.test_visible_nest_training',
])
result = unittest.TextTestRunner(verbosity=1).run(suite)
sys.exit(not result.wasSuccessful())
PY
```

These tests use temporary data, not your annotations. They do not constitute a
GPU training benchmark or validate a research model's accuracy. After passing,
launch `python main.py` and try editing/saving a disposable project before
opening the real project. A first model load should be tested separately.

### Desktop Launcher

After verifying the environment, an experienced person can run this **once per
Linux login account** from the intended repository checkout:

```bash
conda activate bee_annotator
python scripts/desktop_launcher.py install --desktop
```

The installed application is named **Bumblebox Annotator**. Open it from the
applications menu and pin it to the dock. `--desktop` also creates a desktop
shortcut; some desktops require right-clicking that icon and choosing **Allow
Launching** the first time. Omit `--desktop` for an application-menu entry only.
Desktop environments without desktop icons can still use the applications menu.

Students then **open the app, select their profile, and open their assigned
work**. No terminal, environment activation, or choice of repository is needed.
The launcher does not select an identity, sign in to the lab drive, download
projects, load models, or require CUDA. Existing profile and project startup
preferences still apply. Several annotation profiles using the same Linux login
share this launcher; separate Linux logins need their own installation.

The installer records the absolute checkout and environment interpreter paths.
It checks Python/Qt and the presence of core dependencies without loading models
or initializing CUDA; it is not a replacement for the verification above. If
you use a differently named environment, activate it instead, or supply its
interpreter explicitly with `--python /absolute/path/to/env/bin/python`.
Unlike the old `run.sh`, the shortcut does not rely on Conda being on the desktop's
PATH and does not automatically load SAM2.

For a Git worktree, install from that worktree. The shortcut keeps using it even
if another terminal/editor is in a different checkout. Normal `git pull` updates
to that checkout are picked up next launch. **Do not switch its branch or remove
the checkout while the application or training is running.** After moving the
checkout/environment, or updating the launcher helper itself, rerun the install
command from the intended checkout. Reinstalling updates the same shortcut; it
does not create another project or change annotations. It does not update Git,
install packages, or switch branches for you.

Startup and application output is saved locally under
`~/.local/state/bee-annotator/launcher.log`, with two rotated backups capped at
about 5 MiB each. Errors display the log path instead of silently closing. Logs
can contain local project paths and annotation diagnostics; nothing is uploaded
automatically. If the app fails, send the log to the person who configured the
computer. A graphical error dialog uses `zenity` (normally available on Ubuntu),
`kdialog`, or `xmessage`, with a desktop-notification fallback. On a minimal Linux
desktop, have the administrator install one of those dialog tools. Without any
of them, failures are still recorded but cannot show a dialog.

Repeated shortcut clicks are blocked while an app started by this launcher is
starting or open. This does not detect apps started manually with `python main.py`;
close any manually launched annotator before using the shortcut. An already-open
window is not automatically brought to the foreground.

The application-menu entry is
`~/.local/share/applications/org.beewhere.Annotator.desktop`; its small launcher
helper and settings are in `~/.local/share/bee-annotator/launcher/`. XDG data/state
directory overrides are respected. Remove the menu/desktop entries to remove
the shortcuts; none of these files contain project annotations. Installation
requires no `sudo` and does not modify the research drive or the training environment.

## 6. Optional SAM2

Brush/eraser annotation and YOLO do not require SAM2 or its checkpoint. To enable
SAM2, install the pinned revision **after** the other dependencies:

```bash
SAM2_BUILD_CUDA=0 python -m pip install --no-build-isolation \
  -r requirements-sam2.txt -c constraints-linux-py310.txt
python -m pip check
python -c "from sam2.build_sam import build_sam2; from sam2.sam2_image_predictor import SAM2ImagePredictor; print('SAM2 imports passed')"
```

`SAM2_BUILD_CUDA=0` skips compilation of SAM2's custom postprocessing extension,
not GPU model inference. Small-hole/sprinkle postprocessing supplied by that
extension is unavailable. To require the extension instead, use
`SAM2_BUILD_CUDA=1 SAM2_BUILD_ALLOW_ERRORS=0` with the same install command and a
compatible CUDA toolkit/compiler. See [SAM2's installation notes](https://github.com/facebookresearch/sam2/blob/2b90b9f5ceec907a1c18123530e92e794ad901a4/INSTALL.md).
Neither recipe has been validated here for a new full SAM2 training run.

Copy the checkpoint separately and select it in the app. Match its SAM2 model
configuration; not every SAM2/SAM2.1 checkpoint is interchangeable. Detectron2
and other optional training backends are outside this setup's tested scope.

## 7. Transfer Projects and Models Separately

Save and close the annotator before copying a project. Transfer the **whole
project folder**, including `annotations`, `frames`, original videos if needed,
models, project identity/attribution files, and import-review metadata. A Git
clone contains none of these private files. Dropbox or a verified lab-drive
project copy can carry them; keep a backup and avoid editing the same local
project copy simultaneously from two computers. If using the app's lab-drive
sync, copy a published revision to a local working folder before editing it;
sync revisions omit models and regenerable COCO exports, so transfer models
separately and regenerate those exports as needed.

On the new computer:

1. Use the installed launcher, or activate the environment and run `python main.py`.
2. Complete the one-time computer setup with **Connect to a Lab...** or **Use
   Locally**, then select your contributor profile. Lab connections discover
   existing identities automatically; do not create another profile just because
   this is a new computer. Machine-local connection settings are not transferred
   by Git. For offline identity transfer, use **Manage Profiles > Advanced > Import
   Profile...**. Existing annotation attribution is preserved. See
   [shared profiles](shared_profiles.md) and [the sync guide](project_sync.md).
3. Open the copied project folder. Re-select checkpoint, video, and sync paths
   where the old machine's absolute paths no longer apply.
4. Confirm annotation policy and outstanding imported-frame reviews. Copying a
   project is not an approval of pending labels.
5. Check one frame, save/reopen it, then try a small inference or training run.

[BeeWhere segmentation models!](https://drive.google.com/drive/folders/1EDx2Gp3tX8OQCpdNEyMcNlSrJFp3Y0RC?usp=sharing)
provides the published `2026_CV4E` model folder. Private data, credentials, large
checkpoints, and local analysis outputs should remain outside Git.

## Verification Scope

On 2026-10-12, a clean Python 3.10.19 virtual environment was installed using the
CPU recipe on Ubuntu 24.04, including the optional pinned SAM2 install without
its CUDA extension. `pip check` and SAM2 imports passed. The headless Ultralytics
package's application files matched the reference installation byte for byte.
All **108 focused tests** above passed, including one synthetic training epoch,
checkpoint reload and standalone validation through the raster-mask trainer.

The matrix multiplication and torchvision CUDA smoke checks also passed on the
existing RTX A5000 environment with PyTorch 2.10.0+cu128. That environment had
already run the small GPU bee/hive preview. A complete new CUDA environment was
not installed for this check. Fresh-machine driver/display setup, SAM2 extension
compilation, and full GPU training still need verification on the destination
computer. Carbontracker can issue location/short-measurement warnings in the
synthetic test; those are not a validation of energy measurement accuracy.
