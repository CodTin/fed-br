import torch
from torch import nn
from torch.types import Device


def get_device(cuda_num: int = 0) -> Device:
    device = "cpu"

    if torch.mps.is_available():
        device = "mps"

    if torch.cuda.is_available():
        device = f"cuda:{cuda_num}"
    print(40 * "=")
    print(f"Using device: {device}")
    print(40 * "=")

    return device


def unwrap_state_dict(model: nn.Module) -> dict:
    # NOTE: this is to return plain or unwrapped state_dict even if Opacus wrapped the model.
    return (
        model._module.state_dict() if hasattr(model, "_module") else model.state_dict()
    )