"""What training on an Apple GPU needs that an NVIDIA card does not.

Written after the first full-resolution run on a Mac (November 2026) stopped with
"shape mismatch: value tensor of shape [16133] cannot be broadcast to indexing
result of shape [23132]": PyTorch's Apple GPU support had counted one mask two
ways inside Ultralytics' target assignment. Nothing here has been run on a Mac.
"""

from contextlib import contextmanager

CPU_ASSIGNMENT_DEVICES = ('mps',)


@contextmanager
def assign_targets_on_cpu(device_types=None):
    """While active, match targets to predictions on the CPU for models on these devices.

    The match builds a table with one entry per target per prediction position
    (344,064 positions in a 4096-pixel frame) and picks entries out of it with a
    mask, which is the operation the Apple GPU got wrong. Its inputs carry no
    gradient, so computing it elsewhere changes nothing else about training.
    Ultralytics takes the same route itself when an NVIDIA card runs out of memory.
    """
    from ultralytics.utils.tal import TaskAlignedAssigner
    device_types = CPU_ASSIGNMENT_DEVICES if device_types is None else device_types
    original = TaskAlignedAssigner.forward
    announced = []

    def forward(self, pd_scores, pd_bboxes, anc_points, gt_labels, gt_bboxes, mask_gt):
        tensors = (pd_scores, pd_bboxes, anc_points, gt_labels, gt_bboxes, mask_gt)
        device = gt_bboxes.device
        if device.type not in device_types:
            return original(self, *tensors)
        if not announced:  # once per training run, so the log shows the route is in use
            announced.append(True)
            print(f'Matching targets to predictions on the CPU instead of {device.type}.', flush=True)
        return tuple(result.to(device) for result in original(self, *(tensor.cpu() for tensor in tensors)))

    TaskAlignedAssigner.forward = forward
    try:
        yield
    finally:
        TaskAlignedAssigner.forward = original
