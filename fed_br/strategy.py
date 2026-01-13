from collections.abc import Iterable
from datetime import datetime
from pathlib import Path
from typing import Any

from flwr.common import ArrayRecord, ConfigRecord, Message, MetricRecord
from flwr.serverapp.strategy import FedAvg

from common.const import FINAL_MODEL_DIR, CLIENT_METRICS_PREFIX
from fed_br.metrics_logger import ClientMetricLogger


class FedBr(FedAvg):
    """
    FedBr 聚合策略（扩展 FedAvg）。

    基于博弈论优化的联邦学习聚合策略，扩展自 FedAvg。
    维护全局状态（总训练轮数、噪声影响、样本数），
    并将这些信息注入到客户端训练配置中，使客户端能够
    计算最优的本地训练参数（epoch 数和噪声乘数）。

    Attributes:
        current_t_total: 累计的全局训练轮数
        current_global_noise_impact: 加权平均的全局噪声影响
        total_samples: 当前轮的参与样本总数

    Example:
        >>> strategy = FedBr()
        >>> print(f"Initial t_total: {strategy.current_t_total}")
    """

    def __init__(self, **kwargs):
        """
        初始化 FedBr 策略。

        Args:
            **kwargs: 传递给父类 FedAvg 的参数
        """
        super().__init__(**kwargs)
        self.current_t_total: float = 0.0
        self.current_global_noise_impact: float = 0.0
        self.total_samples: int = 0
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        metrics_path = Path(FINAL_MODEL_DIR) / f"{CLIENT_METRICS_PREFIX}_{timestamp}.jsonl"
        self.metrics_logger = ClientMetricLogger(metrics_path)

    @staticmethod
    def _extract_client_metrics(metrics_record: MetricRecord) -> dict[str, float | int]:
        metrics: dict[str, float | int] = {}
        for key, value in metrics_record.items():
            if key == "client_id":
                continue
            if isinstance(value, bool):
                metrics[key] = int(value)
            elif isinstance(value, (int, float)):
                metrics[key] = value
        return metrics

    def configure_train(self, server_round: int, *args, **kwargs) -> Iterable[Message]:
        """
        配置客户端训练参数。

        重写父类方法，在每个训练轮次开始时，将全局状态信息
        注入到每个客户端的配置中。这些信息包括：
        - 全局累计训练轮数
        - 全局噪声影响
        - 全局样本数
        - 当前轮次

        Args:
            server_round: 当前训练轮次
            *args: 传递给父类的位置参数
            **kwargs: 传递给父类的关键字参数

        Returns:
            Iterable[Message]: 更新后的客户端指令消息列表

        Example:
            >>> messages = strategy.configure_train(server_round=1)
            >>> for msg in messages:
            ...     print(msg.content["config"]["global_t_total"])
        """
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
        """
        聚合客户端训练结果并更新全局状态。

        重写父类方法，从客户端回复中提取训练指标，
        更新全局状态（总训练轮数、噪声影响、样本数），
        并将这些状态信息记录到聚合指标中。

        Args:
            server_round: 当前训练轮次
            replies: 客户端返回的消息迭代器

        Returns:
            tuple: (聚合后的模型参数 ArrayRecord, 聚合后的指标 MetricRecord)

        Note:
            从每个客户端的 metrics 中提取:
            - num-examples: 样本数
            - local_epochs: 本地训练轮数
            - local_impact_factor: 本地噪声影响因子

        Example:
            >>> aggregated, metrics = strategy.aggregate_train(1, client_replies)
            >>> print(metrics["global_t_total"])
        """
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
            client_id = int(metrics_record.get("client_id", -1))

            if client_id >= 0:
                self.metrics_logger.log(
                    round_number=server_round,
                    phase="train",
                    client_id=client_id,
                    metrics=self._extract_client_metrics(metrics_record)
                )

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

    # def aggregate_evaluate(
    #     self,
    #     server_round: int,
    #     replies: Iterable[Message],
    # ) -> tuple[float | None, MetricRecord | None]:
    #     total_examples = 0
    #     weighted_loss_sum = 0.0
    #     weighted_acc_sum = 0.0
    #
    #     for msg in replies:
    #         if "metrics" not in msg.content:
    #             continue
    #
    #         metrics_record = msg.content["metrics"]
    #         client_id = int(metrics_record.get("client_id", -1))
    #         if client_id >= 0:
    #             self.metrics_logger.log(
    #                 round_number=server_round,
    #                 phase="evaluate",
    #                 client_id=client_id,
    #                 metrics=self._extract_client_metrics(metrics_record),
    #             )
    #
    #         num_examples = int(metrics_record.get("num-examples", 0))
    #         eval_loss = float(metrics_record.get("eval_loss", 0.0))
    #         eval_acc = float(metrics_record.get("eval_acc", 0.0))
    #         total_examples += num_examples
    #         weighted_loss_sum += eval_loss * num_examples
    #         weighted_acc_sum += eval_acc * num_examples
    #
    #     if total_examples == 0:
    #         return None, MetricRecord({})
    #
    #     avg_loss = weighted_loss_sum / total_examples
    #     avg_acc = weighted_acc_sum / total_examples
    #     aggregated_metrics = MetricRecord({"eval_loss": avg_loss, "eval_acc": avg_acc})
    #     return avg_loss, aggregated_metrics
