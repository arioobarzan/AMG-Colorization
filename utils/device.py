"""Automatic device selection for PyTorch."""

import torch


def get_device(device_str: str = "auto") -> torch.device:
    """Resolve device string to a torch.device.

    Args:
        device_str: One of "auto", "cuda", "cpu", or a CUDA device index (e.g. "cuda:0").

    Returns:
        torch.device: The resolved device.
    """
    if device_str == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    if device_str == "cuda" and not torch.cuda.is_available():
        print("[WARNING] CUDA requested but not available. Falling back to CPU.")
        return torch.device("cpu")

    return torch.device(device_str)


def get_device_str(device: torch.device) -> str:
    """Return a human-readable device string."""
    if device.type == "cuda":
        return f"CUDA ({torch.cuda.get_device_name(device)})"
    return "CPU"
