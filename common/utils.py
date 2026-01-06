import torch
from torch.types import Device


def get_device(cuda_num: int = 0) -> Device:
    device = "cpu"

    if torch.mps.is_available():
        device = "mps"

    if torch.cuda.is_available():
        device = f"cuda:{cuda_num}"

    return device
