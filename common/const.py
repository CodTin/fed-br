from enum import Enum, unique

FINAL_MODEL_DIR = "outputs"
FINAL_MODEL_NAME = "final_model.pt"
FINAL_MODEL_PATH = f"{FINAL_MODEL_DIR}/{FINAL_MODEL_NAME}"


@unique
class CommunicationConstant(Enum):
    BANDWIDTH = 1e6  #: 通信带宽 (Hz)
    TX_POWER = 0.5  #: 发射功率 (W)
    NOISE_PSD = 1e-9  #: 噪声功率谱密度 (W/Hz)
    CIRCUIT_POWER = 0.1  #: 电路功耗 (W)


@unique
class ComputationConstant(Enum):
    CYCLES_PER_SAMPLE = 2e6
    E_DEC = 1e-10  # 解码能耗 (J)
    E_AGG = 1e-11  # 聚合能耗 (j)


@unique
class PrivacyLeakageConstant(Enum):
    ALPHA_0 = 4.0
