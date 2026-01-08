from typing import TYPE_CHECKING, Any, cast

from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp
from opacus import PrivacyEngine
from torch.optim import SGD
from torch.utils.data import DataLoader

from common import get_device, unwrap_state_dict

from .system_model import Communication, Computation, PrivacyLeakage
from .task import Net, load_data
from .task import test as test_fn
from .task import train as train_fn

if TYPE_CHECKING:
    from collections.abc import Sized

app = ClientApp()


def partition_loader(
    ctx: Context,
) -> tuple[int, float, DataLoader[Any], DataLoader[Any]]:
    """
    根据上下文配置加载客户端数据分区。

    从上下文中读取分区 ID、分区总数和批次大小,
    调用 load_data() 加载对应的 CIFAR-10 数据分区,
    并根据分区 ID 确定差分隐私噪声乘数(偶数分区 1.0,奇数分区 1.5)。

    Args:
        ctx: Flower 客户端上下文,包含节点配置和运行配置

    Returns:
        tuple: 包含以下元素的元组:
            - partition_id (int): 当前客户端的分区 ID
            - noise (float): 差分隐私噪声乘数
            - trainloader (DataLoader): 训练数据加载器
            - testloader (DataLoader): 测试数据加载器

    Example:
        >>> from flwr.app import Context
        >>> # 由 train() 和 evaluate() 调用
        >>> # pid, noise, trainloader, testloader = partition_loader(context)
    """
    partition_id = int(ctx.node_config["partition-id"])
    num_partitions = int(ctx.node_config["num-partitions"])
    batch_size = int(ctx.run_config["batch-size"])

    noise = 1.0 if partition_id % 2 == 0 else 1.5

    _train, _eval = load_data(partition_id, num_partitions, batch_size)

    return partition_id, noise, _train, _eval


@app.train()
def train(msg: Message, context: Context) -> Message:
    """
    联邦学习客户端训练入口,执行本地差分隐私训练。

    接收服务端发送的全局模型权重,加载本地数据分区,
    使用 Opacus PrivacyEngine 进行差分隐私 SGD 训练,
    计算通信模型的延迟和能耗,返回更新后的模型权重与训练指标。

    Args:
        msg: 服务端发送的消息,包含模型配置和学习率
        context: 客户端上下文,包含分区 ID、运行配置(批次大小、目标 delta 等)

    Returns:
        Message: 包含以下内容的回复消息:
            - arrays: 更新后的模型权重 (ArrayRecord)
            - metrics: 训练指标 (MetricRecord),包含:
                - train_loss: 训练损失
                - num-examples: 训练样本数
                - epsilon: 差分隐私预算
                - noise_multiplier: 噪声乘数
                - max_grad_norm: 梯度裁剪阈值
                - comm_latency: 通信延迟(秒)
                - comm_energy: 通信能耗(焦耳)
                - transmission_rate: 传输速率(bits/sec)

    Example:
        >>> from flwr.app import Message, Context
        >>> # 由 Flower 自动调用
        >>> # reply = train(msg, context)
    """
    # Load the model and initialize it with the received weights
    model = Net()

    target_delta = float(context.run_config["target-delta"])
    max_grad_norm = float(context.run_config["max-grad-norm"])
    lr = float(cast("float", cast("object", msg.content["config"]["lr"])))
    batch_size = int(context.run_config["batch-size"])
    local_epochs = int(context.run_config["local-epochs"])

    arrays = cast("ArrayRecord", msg.content["arrays"])
    model.load_state_dict(arrays.to_torch_state_dict())
    device = get_device()
    model.to(device)

    optim = SGD(params=model.parameters(), lr=lr, momentum=0.9)

    # Load the data
    pid, noise_multiplier, trainloader, _ = partition_loader(context)

    # 计算 Communication Model
    model_size = Communication.get_model_size_bits(model)
    h_i = Communication.get_channel_gain(pid)
    r_i = Communication.compute_transmission_rate(h_i)
    comm_latency, comm_energy = Communication.compute_latency_and_energy(
        model_size, r_i
    )

    # 计算 Computation Model
    num_samples = len(cast("Sized", cast("object", trainloader.dataset)))
    f_i, kappa_i = Computation.get_client_hardware_params(pid)
    comp_latency, comp_energy = Computation.compute_local_latency_and_energy(
        num_sample=num_samples, epochs=local_epochs, f_i=f_i, kappa_i=kappa_i
    )

    # 计算 PrivacyLeakage Model
    privacy_cost = PrivacyLeakage.compute_privacy_cost(
        num_samples=num_samples,
        batch_size=batch_size,
        epochs=local_epochs,
        noise_multiplier=noise_multiplier,
        target_delta=target_delta,
    )

    privacy_engine = PrivacyEngine(secure_mode=False)

    model, optim, trainloader = privacy_engine.make_private(
        module=model,
        optimizer=optim,
        data_loader=trainloader,
        noise_multiplier=noise_multiplier,
        max_grad_norm=max_grad_norm,
    )

    # Call the training function
    local_epochs_int = int(context.run_config["local-epochs"])
    train_loss, epsilon = train_fn(
        model,
        trainloader,
        local_epochs_int,
        device,
        target_delta=target_delta,
        privacy_engine=privacy_engine,
        optimizer=optim,
    )

    # Construct and return reply Message
    model_record = ArrayRecord(unwrap_state_dict(model))
    metrics: dict[str, int | float] = {
        "train_loss": train_loss,
        "num-examples": len(cast("Sized", trainloader.dataset)),
        "epsilon": float(epsilon),
        "target_delta": float(target_delta),
        "noise_multiplier": float(noise_multiplier),
        "max_grad_norm": float(max_grad_norm),
        # Communication Model Metrics
        "comm_latency": float(comm_latency),
        "comm_energy": float(comm_energy),
        "transmission_rate": float(r_i),
        # Computation Model Metrics
        "comp_latency": float(comp_latency),
        "comp_energy": float(comp_energy),
        # Total Metric
        "total_latency": float(comm_latency + comp_latency),
        "total_energy": float(comm_energy + comp_energy),
        # Privacy
        "privacy_cost": float(privacy_cost),
    }
    metric_record = MetricRecord(metrics)
    content = RecordDict({"arrays": model_record, "metrics": metric_record})
    return Message(content=content, reply_to=msg)


@app.evaluate()
def evaluate(msg: Message, context: Context) -> Message:
    """
    联邦学习客户端评估入口,在本地验证集上评估模型。

    接收服务端发送的全局模型权重,在本地数据分区的测试部分上
    进行推理,计算损失值和准确率并返回评估指标。

    Args:
        msg: 服务端发送的消息,包含模型权重
        context: 客户端上下文,包含分区 ID、运行配置

    Returns:
        Message: 包含评估指标的回复消息:
            - metrics: 评估指标 (MetricRecord),包含:
                - eval_loss: 验证集损失
                - eval_acc: 验证集准确率
                - num-examples: 验证样本数

    Example:
        >>> from flwr.app import Message, Context
        >>> # 由 Flower 自动调用
        >>> # reply = evaluate(msg, context)
    """
    # Load the model and initialize it with the received weights
    model = Net()
    arrays = cast("ArrayRecord", msg.content["arrays"])
    model.load_state_dict(arrays.to_torch_state_dict())
    device = get_device()
    model.to(device)

    # Load the data
    _, _, _, valloader = partition_loader(context)

    # Call the evaluation function
    device_str = get_device()
    eval_loss, eval_acc = test_fn(
        model,
        valloader,
        device_str,
    )

    # Construct and return reply Message
    metric_record = MetricRecord(
        {
            "eval_loss": eval_loss,
            "eval_acc": eval_acc,
            "num-examples": len(cast("Sized", cast("object", valloader.dataset))),
        }
    )
    content = RecordDict({"metrics": metric_record})
    return Message(content=content, reply_to=msg)
