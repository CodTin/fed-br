from common.const import GameLogicConstants, GlobalConvergenceConstant
from fed_br.system_model import (
    Communication,
    Computation,
    GlobalConvergence,
    PrivacyLeakage,
)


class GameLogic:
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

                est_global_noise_impact = (
                    global_noise_impact_old - my_old_contribution + my_new_contribution
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
