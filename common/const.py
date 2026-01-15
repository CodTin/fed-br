"""
项目常量定义模块。

集中管理联邦学习系统中的所有常量配置，包括:
- 输出路径配置
- 通信模型参数（带宽、功率、噪声等）
- 计算模型参数（周期、能耗系数等）
- 隐私泄露模型参数
- 全局收敛模型参数
- 博弈论优化参数

Attributes:
    FINAL_MODEL_DIR: 模型输出目录
    FINAL_MODEL_NAME: 模型文件名
    FINAL_MODEL_PATH: 模型完整路径
    CommunicationConstant: 通信模型常量
    ComputationConstant: 计算模型常量
    PrivacyLeakageConstant: 隐私泄露模型常量
    GlobalConvergenceConstant: 全局收敛模型常量
    GameLogicConstants: 博弈论优化常量
"""

from enum import Enum, unique

FINAL_MODEL_DIR = "outputs"
FINAL_MODEL_NAME = "final_model.pt"
FINAL_MODEL_PATH = f"{FINAL_MODEL_DIR}/{FINAL_MODEL_NAME}"

CLIENT_METRICS_PREFIX = "client_metrics"
CLIENT_METRICS_PATTERN = f"{CLIENT_METRICS_PREFIX}_*.jsonl"
CLIENT_PLOTS_DIR = f"../{FINAL_MODEL_DIR}/client_plots"


@unique
class CommunicationConstant(Enum):
    """
    通信模型常量定义。

    定义无线通信相关的系统参数，包括带宽、发射功率、噪声功率谱密度
    和电路功耗等。这些参数用于计算模型传输的延迟和能耗。

    Attributes:
        BANDWIDTH: 通信带宽 (Hz)
        TX_POWER: 发射功率 (W)
        NOISE_PSD: 噪声功率谱密度 (W/Hz)
        CIRCUIT_POWER: 电路功耗 (W)

    Example:
        >>> bandwidth = CommunicationConstant.BANDWIDTH.value
        >>> print(f"Bandwidth: {bandwidth} Hz")
    """

    BANDWIDTH = 1e6  #: 通信带宽 (Hz)
    TX_POWER = 0.5  #: 发射功率 (W)
    NOISE_PSD = 1e-9  #: 噪声功率谱密度 (W/Hz)
    CIRCUIT_POWER = 0.1  #: 电路功耗 (W)


@unique
class ComputationConstant(Enum):
    """
    计算模型常量定义。

    定义本地计算相关的系统参数，包括每样本计算周期、解码能耗系数
    和聚合能耗系数等。这些参数用于计算本地训练的延迟和能耗。

    Attributes:
        CYCLES_PER_SAMPLE: 每个样本所需的计算周期数
        E_DEC: 解码能耗系数 (J/bit)
        E_AGG: 聚合能耗系数 (J/operation)

    Example:
        >>> cycles = ComputationConstant.CYCLES_PER_SAMPLE.value
        >>> print(f"Cycles per sample: {cycles}")
    """

    CYCLES_PER_SAMPLE = 2e6
    E_DEC = 1e-10  # 解码能耗 (J)
    E_AGG = 1e-11  # 聚合能耗 (j)


@unique
class PrivacyLeakageConstant(Enum):
    """
    隐私泄露模型常量定义。

    定义差分隐私相关的参数，用于计算隐私成本（epsilon 值）。

    Attributes:
        ALPHA_0: DP 理论常数，用于计算隐私预算消耗

    Note:
        该常数影响隐私预算的计算公式，与噪声乘数和训练轮数共同
        决定最终的差分隐私保证。

    Example:
        >>> alpha = PrivacyLeakageConstant.ALPHA_0.value
        >>> print(f"Alpha_0: {alpha}")
    """

    ALPHA_0 = [1 + x / 10.0 for x in range(1, 100)] + list(range(12, 64))


@unique
class GlobalConvergenceConstant(Enum):
    """
    全局收敛模型常量定义。

    定义全局模型收敛相关的参数，包括收敛常数、基础数据质量因子
    和异构性惩罚系数等。这些参数用于估算全局收敛误差。

    Attributes:
        A: 收敛常数 A，影响收敛速度
        B: 收敛常数 B，影响噪声对收敛的影响
        THETA_BASE: 基础数据质量因子
        THETA_PENALTY: 异构性惩罚系数，用于调整数据分布不均的影响

    Example:
        >>> a = GlobalConvergenceConstant.A.value
        >>> print(f"Convergence constant A: {a}")
    """

    A = 50.0
    B = 0.0

    THETA_BASE = 1.0  # 基础敏感度
    THETA_PENALTY = 0.5  # 异构性惩罚系数


# @unique
class GameLogicConstants(Enum):
    """
    博弈论优化常量定义。

    定义客户端最优响应计算中的离散选项和成本权重参数。
    这些常量用于在博弈论框架下计算最优的本地训练配置。

    Attributes:
        EPOCH_OPTIONS: 可选的本地训练轮数离散值
        NOISE_OPTIONS: 可选的差分隐私噪声乘数离散值
        ALPHA: 能耗在总成本中的权重系数
        BETA: 隐私成本在总成本中的权重系数
        GAMMA: 收敛误差在总成本中的权重系数

    Note:
        成本函数: Cost = ALPHA * 能耗 + BETA * 隐私成本 + GAMMA * 收敛误差

    Example:
        >>> epochs = GameLogicConstants.EPOCH_OPTIONS.value
        >>> print(f"Available epochs: {epochs}")
    """

    # EPOCH_OPTIONS = (1, 2, 3, 4, 5)  # 离散的 epochs 选项
    EPOCH_OPTIONS = tuple(i for i in range(1, 11))

    # NOISE_OPTIONS = (0.5, 0.7, 1.0, 1.2, 1.5)  # 离散的 noise multipliers 选项
    NOISE_OPTIONS = tuple(i / 10 for i in range(1, 16))

    ALPHA = 0.5
    BETA = 2.0
    GAMMA = 100.0
