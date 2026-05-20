"""GPU helpers for training workers."""


def require_cuda_device(device_index=0):
    """Return a CUDA device index, or raise a clear error if GPU training cannot run."""
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError(
            "GPU training is required, but PyTorch is not installed in this environment."
        ) from exc

    if not torch.cuda.is_available():
        raise RuntimeError(
            "GPU training is required, but PyTorch cannot access CUDA.\n\n"
            "Start the app from the bee_annotator environment and verify these commands first:\n"
            "  nvidia-smi\n"
            "  python -c \"import torch; print(torch.cuda.is_available()); "
            "print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')\"\n\n"
            f"PyTorch version: {getattr(torch, '__version__', 'unknown')}\n"
            f"PyTorch CUDA build: {getattr(torch.version, 'cuda', None)}"
        )

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
