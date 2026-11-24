"""Which processor the models run on.

An NVIDIA card (CUDA) is used when PyTorch can reach one, otherwise an Apple
GPU (MPS), otherwise the CPU. Set BUMBLEBOX_FORCE_CPU=1 before starting the
app to stay on the CPU, for example if an Apple GPU fails or gives odd masks.

The Apple GPU path is written from PyTorch's documented interface on a Linux
computer; docs/setup.md says what has and has not been seen to work on a Mac.
"""

import os

FORCE_CPU_VARIABLE = 'BUMBLEBOX_FORCE_CPU'

# PyTorch's own names. Both are fractions of the GPU memory macOS recommends for one program.
APPLE_GPU_MEMORY_LIMITS = {
    'PYTORCH_MPS_HIGH_WATERMARK_RATIO': '1.0',  # never hold more than this
    'PYTORCH_MPS_LOW_WATERMARK_RATIO': '0.8',   # from here on, hand unused memory back
}


def cpu_forced():
    return os.environ.get(FORCE_CPU_VARIABLE, '').strip().lower() in ('1', 'true', 'yes')


def limit_apple_gpu_memory(environment=os.environ):
    """Keep PyTorch within the GPU memory macOS recommends. Call before PyTorch is imported.

    Left alone, PyTorch lets an Apple GPU hold 1.7 times the recommended amount,
    which is more memory than the Mac has: the first full-resolution training run
    on a 96 GB Mac held 108 GB and spent its time swapping. At the limit PyTorch
    first frees memory it is only keeping for reuse, and fails with an
    out-of-memory error only if that is not enough.

    PyTorch refuses to start its Apple GPU support when the lower value exceeds
    the upper one, so if either is already set, both are left as they are. The
    settings do nothing on a computer without an Apple GPU.
    """
    if not any(name in environment for name in APPLE_GPU_MEMORY_LIMITS):
        environment.update(APPLE_GPU_MEMORY_LIMITS)


def apple_gpu_available():
    import torch
    mps = getattr(torch.backends, 'mps', None)
    return bool(mps and mps.is_available())


def device_name():
    """Return 'cuda', 'mps' or 'cpu': where models should run on this computer."""
    import torch
    if cpu_forced():
        return 'cpu'
    if torch.cuda.is_available():
        return 'cuda'
    if apple_gpu_available():
        return 'mps'
    return 'cpu'


def place_model(model):
    """Put a loaded YOLO model on this computer's device and return it.

    Ultralytics finds an NVIDIA card by itself but uses an Apple GPU only when
    told to, so every other device is stated here. A model on an NVIDIA
    computer is returned exactly as it was loaded, and so is an exported model
    (ONNX and the like), which Ultralytics cannot move.
    """
    import torch
    name = device_name()
    if name != 'cuda' and isinstance(getattr(model, 'model', None), torch.nn.Module):
        model.to(name)
    return model


def release_gpu_memory():
    """Hand cached GPU memory back to the system. Does nothing without a GPU."""
    import torch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif apple_gpu_available():
        torch.mps.empty_cache()


def synchronize_gpu():
    """Wait for queued GPU work, so a timing lands on the step that caused it."""
    import torch
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elif apple_gpu_available():
        torch.mps.synchronize()


def gpu_memory_in_use():
    """Return (allocated, reserved) GPU memory in bytes, or None without a GPU."""
    import torch
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated(), torch.cuda.memory_reserved()
    if apple_gpu_available():
        return torch.mps.current_allocated_memory(), torch.mps.driver_allocated_memory()
    return None
