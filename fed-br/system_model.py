"""
联邦学习通信模型,用于计算传输速率、延迟和能耗。

该类封装了无线通信相关的计算,包括信道增益建模、
传输速率计算以及通信延迟与能耗估算。

Attributes:
    bandwidth: 通信带宽 (Hz)
    tx_power: 发射功率 (W)
    noise_psd: 噪声功率谱密度 (W/Hz)
    circuit_power: 电路功耗 (W)
"""
import math

import numpy as np
from torch import nn

BANDWIDTH = 1e6  #: 通信带宽 (Hz)
TX_POWER = 0.5  #: 发射功率 (W)
NOISE_PSD = 1e-9  #: 噪声功率谱密度 (W/Hz)
CIRCUIT_POWER = 0.1  #: 电路功耗 (W)


class Communication:
    @staticmethod
    def get_model_size_bits(model: nn.Module) -> int:
        """
        计算 PyTorch 模型的参数量(以比特为单位)。

        Args:
            model: PyTorch 模型实例

        Returns:
            int: 模型参数量(总参数量 x 32 bits)

        Example:
            >>> model = LightweightCNN()
            >>> size = Communication.get_model_size_bits(model)
            >>> print(f"Model size: {size} bits")  # doctest: +SKIP
        """
        total_params = sum(p.numel() for p in model.parameters())
        bits_per_param = 32
        m = total_params * bits_per_param
        return m

    @staticmethod
    def get_channel_gain(client_id: int) -> float:
        """
        根据客户端 ID 生成信道增益。

        使用确定性的随机种子确保同一客户端每次调用获得相同的信道增益,
        同时不同客户端之间具有独立的信道特性。

        Args:
            client_id: 客户端标识符,用于生成唯一的随机种子

        Returns:
            float: 信道增益值,最小值为 1e-4 以避免数值问题

        Note:
            信道增益服从指数分布: h_i ~ Exp(scale=1.0)
        """
        seed = 42 + client_id * 100
        rng = np.random.default_rng(seed)

        h_i = rng.exponential(scale=1.0)

        return max(h_i, 1e-4)

    @staticmethod
    def compute_transmission_rate(h_i: float) -> float:
        """
        根据信道增益计算传输速率。

        基于香农定理计算信道容量:
        r_i = B x log2(1 + SNR)

        Args:
            h_i: 信道增益 (linear scale)

        Returns:
            float: 传输速率 (bits/sec)

        Note:
            SNR = (h_i x TX_POWER) / (NOISE_PSD x BANDWIDTH)
        """
        snr = (h_i * TX_POWER) / (NOISE_PSD * BANDWIDTH)
        r_i = BANDWIDTH * math.log2(1 + snr)

        return r_i

    @staticmethod
    def compute_latency_and_energy(model_size: int, r_i: float) -> tuple[float, float]:
        """
        计算模型传输的延迟和能耗。

        Args:
            model_size: 模型参数量(以比特为单位)
            r_i: 传输速率 (bits/sec)

        Returns:
            tuple[float, float]: (延迟时间(秒), 能耗(焦耳))

        Note:
            延迟 t_comm = model_size / r_i
            能耗 e_comm = (TX_POWER + CIRCUIT_POWER) x t_comm

        Raises:
            ValueError: 当传输速率 r_i <= 0 时返回 (inf, inf)
        """
        if r_i <= 0:
            return float("inf"), float("inf")

        t_comm = model_size / r_i
        e_comm = (TX_POWER + CIRCUIT_POWER) * t_comm
        return t_comm, e_comm
