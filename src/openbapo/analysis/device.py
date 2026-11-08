"""Compute device selection shared by the model adapters."""

AUTO = "auto"


def resolve_device(device: str = AUTO) -> str:
    """Map "auto" to "cuda" when a GPU is visible, else "cpu"; pass anything else through."""
    if device != AUTO:
        return device
    try:
        import torch
    except ImportError:
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"
