from common.const import GameLogicConstants, GlobalConvergenceConstant
from fed_br.system_model import (
    Communication,
    Computation,
    GlobalConvergence,
    PrivacyLeakage,
)


class GameLogic:
    """
    博弈论最优响应计算模块。

    基于博弈论框架计算客户端的最优本地训练策略。
    客户端作为博弈参与者，在给定全局状态和其他客户端策略的
    假设下，选择最小化总成本的本地训练参数（epoch 和噪声乘数）。

    Note:
        成本函数: Cost = ALPHA * 能耗 + BETA * 隐私成本 + GAMMA * 收敛误差

        其中：
        - 能耗 = 通信能耗 + 计算能耗
        - 隐私成本 = 差分隐私 epsilon 值
        - 收敛误差 = 预测的全局收敛误差变化

    Example:
        >>> best_e, best_sigma, min_cost = GameLogic.find_best_response(
        ...     global_t_total_old=100.0,
        ...     global_noise_impact_old=0.5,
        ...     prev_e=1,
        ...     prev_local_impact=0.1,
        ...     num_samples=500,
        ...     total_samples_global=5000,
        ...     batch_size=32,
        ...     target_delta=1e-5,
        ...     f_i=2e9,
        ...     kappa_i=1e-28,
        ...     theta_i=0.8,
        ...     h_i=0.5,
        ...     model_size_bites=480000,
        ... )
    """

    @staticmethod
    def find_best_response(
        global_t_total_old: float,
        global_noise_impact_old: float,
        prev_e: int,
        prev_local_impact: float,
        num_samples: int,
        total_samples_global: int,
        batch_size: int,
        target_delta: float,
        f_i: float,
        kappa_i: float,
        theta_i: float,
        h_i: float,
        model_size_bites: int,
    ) -> tuple[int, float, float]:
        """
        计算客户端的最优响应策略。

        在博弈论框架下，给定全局状态和客户端本地条件，
        通过遍历所有可选的 (epoch, noise_multiplier) 组合，
        找到总成本最小的配置作为最优响应。

        Args:
            global_t_total_old: 上一轮全局累计训练轮数
            global_noise_impact_old: 上一轮全局噪声影响
            prev_e: 上一轮本地使用的 epoch 数
            prev_local_impact: 上一轮本地噪声影响
            num_samples: 本地样本数
            total_samples_global: 全局总样本数
            batch_size: 批次大小
            target_delta: 差分隐私目标 delta 值
            f_i: 客户端计算频率 (Hz)
            kappa_i: 客户端能量系数 (J/operation)
            theta_i: 本地数据质量因子
            h_i: 客户端信道增益
            model_size_bites: 模型参数量（bits）

        Returns:
            tuple: (best_e, best_sigma, min_cost)
            - best_e: 最优的本地训练轮数
            - best_sigma: 最优的噪声乘数
            - min_cost: 对应的最小总成本

        Note:
            搜索空间由 GameLogicConstants.EPOCH_OPTIONS 和
            GameLogicConstants.NOISE_OPTIONS 定义。

        Example:
            >>> e, sigma, cost = GameLogic.find_best_response(100, 0.5, 1, 0.1,
            ...     500, 5000, 32, 1e-5, 2e9, 1e-28, 0.8, 0.5, 480000)
            >>> print(f"Best epochs: {e}, noise: {sigma}, cost: {cost:.4f}")
        """
        min_cost = float("inf")
        best_e = prev_e
        best_sigma = 1.0

        r_i = Communication.compute_transmission_rate(h_i)
        _, comm_energy = Communication.compute_latency_and_energy(model_size_bites, r_i)

        if total_samples_global > 0:
            q_i = num_samples / total_samples_global
        else:
            q_i = 0.0

        for e in GameLogicConstants.EPOCH_OPTIONS.value:
            for sigma in GameLogicConstants.NOISE_OPTIONS.value:
                _, comp_energy = Computation.compute_local_latency_and_energy(
                    num_samples, e, f_i, kappa_i
                )
                e_total = comm_energy + comp_energy

                privacy_cost = PrivacyLeakage.compute_privacy_cost(
                    num_samples, batch_size, e, sigma, target_delta
                )

                new_local_impact = (
                    GlobalConvergenceConstant.B.value + (sigma**2) * theta_i
                )

                my_old_contribution = q_i * prev_local_impact
                my_new_contribution = q_i * new_local_impact

                if global_t_total_old == 0:
                    est_global_noise_impact = my_new_contribution
                else:
                    est_global_noise_impact = (
                        global_noise_impact_old
                        - my_old_contribution
                        + my_new_contribution
                    )

                est_t_total = global_t_total_old - prev_e + e

                j_err = GlobalConvergence.compute_global_error(
                    est_t_total, est_global_noise_impact
                )

                current_cost = (
                    (GameLogicConstants.ALPHA.value * e_total)
                    + (GameLogicConstants.BETA.value * privacy_cost)
                    + (GameLogicConstants.GAMMA.value * j_err)
                )

                if current_cost < min_cost:
                    min_cost = current_cost
                    best_e = e
                    best_sigma = sigma

        return best_e, best_sigma, min_cost
