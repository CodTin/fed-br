"""
联邦学习通信与计算模型,用于计算传输速率、延迟和能耗。

该模块封装了:
- Communication: 无线通信相关的计算,包括信道增益建模、
  传输速率计算以及通信延迟与能耗估算
- Computation: 本地计算相关的计算,包括客户端硬件参数建模、
  本地训练延迟和能耗估算

Attributes:
    Communication: 通信模型类
    Computation: 计算模型类
"""

import math
from typing import Any

import numpy as np
from torch import nn
from torch.utils.data import DataLoader

from common.const import (
    CommunicationConstant,
    ComputationConstant,
    GlobalConvergenceConstant,
    PrivacyLeakageConstant,
)


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
        total_params: int = sum(p.numel() for p in model.parameters())
        bits_per_param: int = 32
        m: int = total_params * bits_per_param
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
        seed: int = 42 + client_id * 100
        rng = np.random.default_rng(seed)

        h_i: float = rng.exponential(scale=1.0)

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
        snr: float = (h_i * CommunicationConstant.TX_POWER.value) / (
            CommunicationConstant.NOISE_PSD.value
            * CommunicationConstant.BANDWIDTH.value
        )
        r_i: float = CommunicationConstant.BANDWIDTH.value * math.log2(1.0 + snr)

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

        t_comm: float = model_size / r_i
        e_comm: float = (
            CommunicationConstant.TX_POWER.value
            + CommunicationConstant.CIRCUIT_POWER.value
        ) * t_comm
        return t_comm, e_comm


class Computation:
    @staticmethod
    def get_client_hardware_params(client_id: int) -> tuple[float, float]:
        """
        获取客户端硬件参数(计算频率和能量系数)。

        根据客户端 ID 生成确定性的随机硬件参数,
        用于后续的本地计算延迟和能耗估算。

        Args:
            client_id: 客户端标识符,用于生成唯一的随机种子

        Returns:
            tuple: (计算频率 f_i (Hz), 能量系数 kappa_i (J/operation))

        Example:
            >>> f_i, kappa_i = Computation.get_client_hardware_params(0)
            >>> print(f"Frequency: {f_i:.2e} Hz")
            >>> print(f"Energy coefficient: {kappa_i:.2e} J")
        """
        seed: int = 2025 + client_id * 50
        rng = np.random.default_rng(seed)

        f_i: float = rng.uniform(1.0e9, 2.5e9)

        kappa_i: float = rng.uniform(1e-28, 5e-28)
        return (f_i, kappa_i)

    @staticmethod
    def compute_local_latency_and_energy(
        num_sample: int, epochs: float, f_i: float, kappa_i: float
    ) -> tuple[float, float]:
        """
        计算本地训练的延迟和能耗。

        Args:
            num_sample: 训练样本数量
            epochs: 本地训练轮数
            f_i: 客户端计算频率 (Hz)
            kappa_i: 客户端能量系数 (J/operation)

        Returns:
            tuple[float, float]: (本地计算延迟(秒), 本地计算能耗(焦耳))

        Note:
            计算公式:
            - cycles_per_epoch = CYCLES_PER_SAMPLE x num_sample
            - total_cycles = epochs x cycles_per_epoch
            - t_comp = total_cycles / f_i
            - e_local = kappa_i x f_i^2 x total_cycles

        Example:
            >>> t_comp, e_local = Computation.compute_local_latency_and_energy(
            ...     num_sample=100, epochs=1, f_i=2e9, kappa_i=1e-28
            ... )
            >>> print(f"Computation latency: {t_comp:.4f}s")
            >>> print(f"Computation energy: {e_local:.4f}J")
        """
        cycles_per_epoch: float = (
            ComputationConstant.CYCLES_PER_SAMPLE.value * num_sample
        )

        total_cycles: float = epochs * cycles_per_epoch

        t_comp: float = total_cycles / f_i

        e_local: float = kappa_i * (f_i**2) * total_cycles

        return (t_comp, e_local)

    @staticmethod
    def compute_server_energy(model_size: int, num_participating_client: int) -> float:
        """
        计算服务器聚合所有客户端模型更新的能耗。

        Args:
            model_size: 模型参数量(以比特为单位)
            num_participating_client: 参与聚合的客户端数量

        Returns:
            float: 服务器聚合能耗(焦耳)

        Note:
            能耗 = num_clients x model_size x (E_DEC + E_AGG)

            其中:

            - E_DEC: 解码能耗系数
            - E_AGG: 聚合能耗系数

        Example:
            >>> energy = Computation.compute_server_energy(
            ...     model_size=480000, num_participating_client=10
            ... )
            >>> print(f"Server aggregation energy: {energy:.4f}J")
        """
        e_per_client: float = model_size * (
            ComputationConstant.E_DEC.value + ComputationConstant.E_AGG.value
        )
        total_energy: float = num_participating_client * e_per_client
        return total_energy


class PrivacyLeakage:
    """
    隐私泄露模型。

    基于差分隐私理论计算隐私成本（epsilon 值）。
    用于评估联邦学习过程中各客户端的隐私泄露程度。

    Example:
        >>> cost = PrivacyLeakage.compute_privacy_cost(
        ...     num_samples=100, batch_size=32, epochs=1,
        ...     noise_multiplier=1.0, target_delta=1e-5
        ... )
        >>> print(f"Privacy cost: {cost:.2f}")
    """

    @staticmethod
    def compute_privacy_cost(
        num_samples: int,
        batch_size: int,
        epochs: int,
        noise_multiplier: float,
        target_delta: float,
    ) -> float:
        """
        计算差分隐私成本（epsilon 值）。

        基于 Moments Accountant 方法计算给定训练配置下的
        差分隐私预算消耗。

        Args:
            num_samples: 本地训练样本数
            batch_size: 批次大小
            epochs: 本地训练轮数
            noise_multiplier: 差分隐私噪声乘数
            target_delta: 差分隐私目标 delta 值

        Returns:
            float: 隐私成本（epsilon 值）。如果 noise_multiplier <= 0
                   或 num_samples <= 0，返回无穷大

        Note:
            计算公式基于 Moments Accountant 近似方法，
            隐私成本随 epochs 和 batch_size 增大而增加，
            随 noise_multiplier 增大而减小。

        Example:
            >>> epsilon = PrivacyLeakage.compute_privacy_cost(
            ...     500, 32, 1, 1.0, 1e-5
            ... )
            >>> print(f"Epsilon: {epsilon:.4f}")
        """
        if noise_multiplier <= 0 or num_samples <= 0:
            return float("inf")

        gamma_i = batch_size / num_samples

        term1 = (epochs * gamma_i * PrivacyLeakageConstant.ALPHA_0.value) / (
            2 * (noise_multiplier**2)
        )

        term2 = math.log(1.0 / target_delta) / (
            PrivacyLeakageConstant.ALPHA_0.value - 1
        )

        return term1 + term2


class GlobalConvergence:
    """
    全局收敛模型。

    用于估算数据质量因子和全局收敛误差。
    数据质量因子反映客户端数据分布与均匀分布的差异，
    收敛误差用于评估全局模型训练过程中的收敛程度。

    Example:
        >>> theta = GlobalConvergence.estimate_data_quality(test_loader)
        >>> error = GlobalConvergence.compute_global_error(100, 0.5)
    """

    @staticmethod
    def estimate_data_quality(dataloader: DataLoader[Any]) -> float:
        """
        估算客户端数据质量因子。

        通过分析数据集中各类别的分布，计算与均匀分布的
        差异程度来估算数据质量因子 theta_i。

        Args:
            dataloader: 包含标签信息的数据加载器

        Returns:
            float: 数据质量因子 theta_i，值越大表示数据分布越不均衡

        Note:
            计算步骤：
            1. 统计各类别样本数
            2. 计算与均匀分布的 L1 距离
            3. 将 L1 距离映射到 [0.5, 1.0] 范围内的 theta 值

        Example:
            >>> theta = GlobalConvergence.estimate_data_quality(train_loader)
            >>> print(f"Data quality factor: {theta:.4f}")
        """
        num_classes = 10
        label_counts = np.zeros(num_classes)

        dataset = dataloader.dataset

        try:
            if hasattr(dataset, "indices") and hasattr(dataset, "dataset"):
                targets = np.array(dataset.dataset.targets)[dataset.indices]
            elif hasattr(dataset, "targets"):
                targets = np.array(dataset.targets)
            else:
                targets_list = []
                for batch in dataloader:
                    labels = batch["label"]
                    targets_list.extend(labels.numpy())
                targets = np.array(targets_list)

            unique, counts = np.unique(targets, return_counts=True)
            for cls, count in zip(unique, counts, strict=False):
                label_counts[cls] = count
            total_samples = len(targets)

        except Exception:
            from loguru import logger

            logger.warning("Error occurred in `estimate_data_quality`", exc_info=True)
            return GlobalConvergenceConstant.THETA_BASE.value

        if total_samples == 0:
            return GlobalConvergenceConstant.THETA_BASE.value

        # 计算概率分布 p_k
        p_dist = label_counts / total_samples

        # 计算与均匀分布 u_k 的 L1 距离
        u_dist = np.full(num_classes, 1.0 / num_classes)
        l1_distance = np.sum(np.abs(p_dist - u_dist))

        # 映射到 theta_i
        theta_i = (
            GlobalConvergenceConstant.THETA_BASE.value
            + GlobalConvergenceConstant.THETA_PENALTY.value * (l1_distance / 1.8)
        )

        return float(theta_i)

    @staticmethod
    def compute_global_error(total_epochs: float, noise_impact_sum: float) -> float:
        """
        计算全局收敛误差。

        根据总训练轮数和噪声影响计算全局模型的收敛误差。

        Args:
            total_epochs: 所有客户端累计的训练轮数
            noise_impact_sum: 所有客户端噪声影响的加权和

        Returns:
            float: 全局收敛误差。如果 total_epochs <= 0，返回无穷大

        Note:
            误差公式: error = A / total_epochs + noise_impact_sum
            训练轮数越多，误差越小；噪声影响越大，误差越大。

        Example:
            >>> error = GlobalConvergence.compute_global_error(100, 0.5)
            >>> print(f"Global convergence error: {error:.4f}")
        """
        if total_epochs <= 0:
            return float("inf")

        term1 = GlobalConvergenceConstant.A.value / total_epochs
        term2 = noise_impact_sum

        return term1 + term2

    @staticmethod
    def compute_local_noise_impact(
        num_samples: int, total_samples: int, noise_multiplier: float, theta_i: float
    ) -> float:
        """
        计算单个客户端的噪声影响因子。

        根据客户端样本占比、噪声乘数和数据质量因子计算
        该客户端对全局收敛的噪声影响贡献。

        Args:
            num_samples: 本地样本数
            total_samples: 全局总样本数
            noise_multiplier: 差分隐私噪声乘数
            theta_i: 本地数据质量因子

        Returns:
            float: 客户端的噪声影响因子

        Note:
            计算公式: impact = q_i * (B + sigma^2 * theta_i)
            其中 q_i = num_samples / total_samples

        Example:
            >>> impact = GlobalConvergence.compute_local_noise_impact(
            ...     500, 5000, 1.0, 0.8
            ... )
            >>> print(f"Local noise impact: {impact:.4f}")
        """
        if total_samples == 0:
            return 0.0

        q_i = num_samples / total_samples

        c_sigma = noise_multiplier**2

        impact = q_i * (GlobalConvergenceConstant.B.value + c_sigma * theta_i)

        return impact
