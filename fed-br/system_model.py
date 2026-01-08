import math

import numpy as np
from torch import nn

BANDWIDTH = 1e6
TX_POWER = 0.5
NOISE_PSD = 1e-9
CIRCUIT_POWER = 0.1


class Communication:
    @staticmethod
    def get_model_size_bits(model: nn.Module) -> int:
        total_params = sum(p.numel() for p in model.parameters())
        bits_per_param = 32
        m = total_params * bits_per_param
        return m

    @staticmethod
    def get_channel_gain(client_id: int) -> float:
        seed = 42 + client_id * 100
        rng = np.random.default_rng(seed)

        h_i = rng.exponential(scale=1.0)

        return max(h_i, 1e-4)

    @staticmethod
    def compute_transmission_rate(h_i: float) -> float:
        snr = (h_i * TX_POWER) / (NOISE_PSD * BANDWIDTH)
        r_i = BANDWIDTH * math.log2(1 + snr)

        return r_i

    @staticmethod
    def compute_latency_and_energy(model_size: int, r_i: float) -> tuple[float, float]:
        if r_i <= 0:
            return float("inf"), float("inf")

        t_comm = model_size / r_i
        e_comm = (TX_POWER + CIRCUIT_POWER) * t_comm
        return t_comm, e_comm
