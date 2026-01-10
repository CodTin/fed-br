import warnings

import torch
from flwr.app import ArrayRecord, ConfigRecord, Context, MetricRecord
from flwr.serverapp import Grid, ServerApp

from common import get_device
from common.const import FINAL_MODEL_PATH
from common.logging import configure_flwr_logging

from .strategy import FedBr
from .task import Net, load_centralized_dataset, test

warnings.filterwarnings("ignore")

configure_flwr_logging()

app = ServerApp()


@app.main()
def main(grid: Grid, context: Context) -> None:
    """
    FedAvg 服务端主入口,启动联邦学习聚合流程。

    读取运行配置,加载全局模型,初始化 FedAvg 策略,
    并执行指定轮数的联邦学习聚合,最后保存最终模型到磁盘。

    Args:
        grid: Flower 网格实例,用于管理与客户端的连接和通信
        context: 运行上下文,包含运行时配置信息(如学习率、轮数等)

    Returns:
        None

    Side Effects:
        - 保存最终模型到 FINAL_MODEL_PATH

    Example:
        >>> from flwr.app import Grid
        >>> # 在 Flower 内部调用
        >>> # main(grid, context)
    """

    # Read run config
    fraction_evaluate: float = float(context.run_config["fraction-evaluate"])
    num_rounds: int = int(context.run_config["num-server-rounds"])
    lr: float = float(context.run_config["learning-rate"])

    # Load global model
    global_model = Net()
    arrays = ArrayRecord(global_model.state_dict())

    # Initialize FedAvg strategy
    strategy = FedBr(fraction_evaluate=fraction_evaluate)

    # Start strategy, run FedAvg for `num_rounds`
    result = strategy.start(
        grid=grid,
        initial_arrays=arrays,
        train_config=ConfigRecord({"lr": lr}),
        num_rounds=num_rounds,
        evaluate_fn=global_evaluate,
    )

    # Save final model to disk
    state_dict = result.arrays.to_torch_state_dict()
    torch.save(state_dict, FINAL_MODEL_PATH)


def global_evaluate(server_round: int, arrays: ArrayRecord) -> MetricRecord:
    """
    在中央测试集上评估全局模型。

    加载接收到的模型权重,在 CIFAR-10 测试集上进行推理,
    计算损失值和准确率并返回评估指标。

    Args:
        server_round: 当前联邦学习轮数(从 1 开始)
        arrays: 包含模型参数权重的 ArrayRecord 对象

    Returns:
        MetricRecord: 包含以下字段的评估指标字典:
            - accuracy (float): 测试集准确率 (0-1)
            - loss (float): 测试集平均损失值

    Example:
        >>> from flwr.app import ArrayRecord, MetricRecord
        >>> # 由 FedAvg 策略自动调用
        >>> # result = global_evaluate(1, arrays)
    """
    # Load the model and initialize it with the received weights
    model = Net()
    model.load_state_dict(arrays.to_torch_state_dict())
    device = get_device()
    model.to(device)

    # Load entire test set
    test_dataloader = load_centralized_dataset()

    # Evaluate the global model on the test set
    test_loss, test_acc = test(model, test_dataloader, get_device())

    # Return the evaluation metrics
    return MetricRecord({"accuracy": test_acc, "loss": test_loss})
