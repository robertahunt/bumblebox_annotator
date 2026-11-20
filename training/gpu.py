"""GPU helpers for training workers."""

import os
import sys

from core.gpu import FORCE_CPU_VARIABLE, apple_gpu_available, cpu_forced


def require_training_device(device_index=0):
    """Return (device, label) for an NVIDIA card or an Apple GPU, or raise a clear error.

    The device is what Ultralytics expects: a CUDA index, or 'mps' for an
    Apple GPU. Training never falls back to the CPU.
    """
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError(
            "GPU training is required, but PyTorch is not installed in this environment."
        ) from exc

    if cpu_forced():
        raise RuntimeError(
            f"GPU training is required, but {FORCE_CPU_VARIABLE} is set, which keeps the app "
            "on the CPU. Unset it and restart the app to train."
        )

    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        if device_count <= device_index:
            raise RuntimeError(
                f"GPU training is required, but CUDA device {device_index} is not available. "
                f"PyTorch sees {device_count} CUDA device(s)."
            )
        torch.cuda.set_device(device_index)
        device_name = torch.cuda.get_device_name(device_index)
        total_gb = torch.cuda.get_device_properties(device_index).total_memory / (1024 ** 3)
        return device_index, f"{device_name} ({total_gb:.1f} GB)"

    if apple_gpu_available():
        return 'mps', _apple_gpu_label()

    if sys.platform == 'darwin':
        raise RuntimeError(
            "GPU training is required, but PyTorch cannot access the Apple GPU.\n\n"
            "Start the app from the bee_annotator environment and verify this command first:\n"
            "  python -c \"import platform, torch; print(platform.machine(), "
            "torch.backends.mps.is_built(), torch.backends.mps.is_available())\"\n\n"
            "It should print: arm64 True True. Anything else usually means Python is the "
            "Intel build running under Rosetta, or macOS is older than PyTorch supports.\n\n"
            f"PyTorch version: {getattr(torch, '__version__', 'unknown')}"
        )

    raise RuntimeError(
        "GPU training is required, but PyTorch cannot access CUDA.\n\n"
        "Start the app from the bee_annotator environment and verify these commands first:\n"
        "  nvidia-smi\n"
        "  python -c \"import torch; print(torch.cuda.is_available()); "
        "print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')\"\n\n"
        f"PyTorch version: {getattr(torch, '__version__', 'unknown')}\n"
        f"PyTorch CUDA build: {getattr(torch.version, 'cuda', None)}"
    )


def _apple_gpu_label():
    # An Apple GPU has no memory of its own; it shares the computer's.
    try:
        total_gb = os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES') / (1024 ** 3)
    except (ValueError, OSError):
        return "Apple GPU"
    return f"Apple GPU ({total_gb:.0f} GB shared with the system)"
