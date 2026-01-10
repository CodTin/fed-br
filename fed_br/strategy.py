from collections.abc import Iterable

from flwr.common import ArrayRecord, ConfigRecord, Message, MetricRecord
from flwr.serverapp.strategy import FedAvg


class FedBr(FedAvg):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.current_t_total: float = 0.0
        self.current_global_noise_impact: float = 0.0
        self.total_samples: int = 0

    def configure_train(self, server_round: int, *args, **kwargs) -> Iterable[Message]:
        client_instructions = super().configure_train(server_round, *args, **kwargs)

        updated_instructions = []

        for msg in client_instructions:
            if "config" not in msg.content:
                msg.content["config"] = ConfigRecord({})

            config_record = msg.content["config"]

            config_record["global_t_total"] = self.current_t_total
            config_record["global_noise_impact"] = self.current_global_noise_impact
            config_record["global_total_samples"] = self.total_samples
            config_record["current_round"] = server_round

            updated_instructions.append(msg)

        return updated_instructions

    def aggregate_train(
        self,
        server_round: int,
        replies: Iterable[Message],
    ) -> tuple[ArrayRecord | None, MetricRecord | None]:
        # 1. 调用父类方法聚合模型参数
        aggregated_arrays, aggregated_metrics = super().aggregate_train(
            server_round, replies
        )

        if aggregated_arrays is None:
            return aggregated_arrays, aggregated_metrics

        # 2. 统计自定义指标
        epoch_sum = 0
        weighted_impact_sum = 0.0
        round_total_samples = 0

        # 遍历所有客户端的回复消息
        for msg in replies:
            # 获取 metrics 记录
            if "metrics" not in msg.content:
                continue

            metrics_record = msg.content["metrics"]

            # 获取样本数 (通常在 metrics 中会有 num-examples)
            num_examples = int(metrics_record.get("num-examples", 0))

            # 获取客户端上报的 local_epochs
            e_i = int(metrics_record.get("local_epochs", 1))

            # 获取客户端上报的 local_impact_factor
            local_impact = float(metrics_record.get("local_impact_factor", 0.0))

            round_total_samples += num_examples
            epoch_sum += e_i
            # 累加加权部分: n_i * impact
            weighted_impact_sum += num_examples * local_impact

        # 3. 更新全局状态
        self.total_samples = round_total_samples
        self.current_t_total = float(epoch_sum)

        if round_total_samples > 0:
            self.current_global_noise_impact = weighted_impact_sum / round_total_samples
        else:
            self.current_global_noise_impact = 0.0

        # 4. 将全局状态记录到 Aggregated Metrics 中
        if aggregated_metrics is None:
            aggregated_metrics = MetricRecord({})

        aggregated_metrics["global_t_total"] = self.current_t_total
        aggregated_metrics["global_noise_impact"] = self.current_global_noise_impact
        aggregated_metrics["total_participants"] = (
            len(list(replies)) if isinstance(replies, list) else 0
        )

        return aggregated_arrays, aggregated_metrics
