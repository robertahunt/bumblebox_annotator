# Setting Up Another Computer

This guide targets **Ubuntu 24.04 / Linux x86_64, Python 3.10.19**, with an NVIDIA
GPU for inference/training. The pinned Qt wheel requires glibc 2.34 or newer;
older Linux installations need a separately tested package set. It transfers the `integration/annotation-batch` development
branch, not `master`. No worktree is necessary on the new computer.

Use a new environment; do not alter a collaborator's existing environment.
Windows/macOS and other architectures need separate dependency and GPU checks;
the Linux constraints are not a validated cross-platform lockfile. The old
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

CPU annotation does not require a GPU. This application's YOLO training workflow
requires CUDA; CPU-only setup is not a workaround for that requirement. CPU
inference is slower, and some model-specific workflows still require a GPU.

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

1. Activate the environment and run `python main.py`.
2. Set your contributor identity and optionally configure external sync. These
   machine-local settings are not transferred by Git. Entering the same name on
   a new installation creates a new contributor ID; there is no automatic
   cross-machine account/profile migration. Existing annotation attribution is
   preserved. See [the sync guide](project_sync.md).
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
