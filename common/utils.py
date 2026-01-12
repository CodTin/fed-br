from typing import cast

import torch
from torch import Tensor, nn


def get_device(cuda_num: int = 0) -> str:
    """自动检测并返回最优计算设备。

    优先检测 CUDA GPU,其次是 MPS (Apple Silicon),最后是 CPU。

    Args:
        cuda_num: CUDA 设备编号,默认为 0

    Returns:
        str: 计算设备字符串,可能的值包括:
            - "cuda:{n}": NVIDIA GPU (如 "cuda:0")
            - "mps": Apple Silicon GPU
            - "cpu": CPU

    Example:
        >>> device = get_device()
        >>> # 返回 "cuda:0"、"mps" 或 "cpu"
    """
    device = "cpu"

    if torch.mps.is_available():
        device = "mps"

    if torch.cuda.is_available():
        device = f"cuda:{cuda_num}"

    return device


def unwrap_state_dict(model: nn.Module) -> dict[str, Tensor]:
    """
    安全提取模型状态字典,处理 Opacus 包装情况。

    Opacus 的 PrivacyEngine.make_private() 会将原始模型包装为
    PrivacyEngineAwareModule,其状态字典保存在 _module 属性中。
    此函数统一处理两种情况,确保返回正确的状态字典。

    Args:
        model: PyTorch 模型(可能被 Opacus 包装)

    Returns:
        dict[str, Tensor]: 模型的状态字典

    Example:
        >>> model = LightweightCNN()
        >>> # 普通模型
        >>> state_dict = unwrap_state_dict(model)
        >>> # Opacus 包装后的模型
        >>> state_dict = unwrap_state_dict(wrapped_model)

    Note:
        如果模型有 _module 属性(Opacus 包装),返回 model._module.state_dict()
        否则返回 model.state_dict()
    """
    if hasattr(model, "_module"):
        module = cast("nn.Module", model._module)
        return module.state_dict()
    return model.state_dict()
