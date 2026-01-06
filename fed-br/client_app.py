import torch.nn as nn
from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp
from opacus import PrivacyEngine
from torch.optim import SGD
from torch.utils.data import DataLoader

from common import get_device
from .task import Net, load_data
from .task import test as test_fn
from .task import train as train_fn

app = ClientApp()


def partition_loader(ctx: Context) -> tuple[int, float, DataLoader, DataLoader]:
    """
    load dataloader using user config
    """
    partition_id = ctx.node_config["partition-id"]
    num_partitions = ctx.node_config["num-partitions"]
    batch_size = ctx.run_config["batch-size"]

    noise = 1.0 if partition_id % 2 == 0 else 1.5

    _train, _eval = load_data(partition_id, num_partitions, batch_size)

    return partition_id, noise, _train, _eval


def _unwrap_state_dict(model: nn.Module) -> dict:
    # NOTE: this is to return plain or unwrapped state_dict even if Opacus wrapped the model.
    return (
        model._module.state_dict() if hasattr(model, "_module") else model.state_dict()
    )


@app.train()
def train(msg: Message, context: Context):
    """Train the model on local data."""

    # Load the model and initialize it with the received weights
    model = Net()

    target_delta = float(context.run_config["target-delta"])
    max_grad_norm = float(context.run_config["max-grad-norm"])
    lr = float(msg.content["config"]["lr"])

    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
    device = get_device()
    model.to(device)

    optim = SGD(params=model.parameters(), lr=lr, momentum=0.9)

    # Load the data
    pid, noise_multiplier, trainloader, _ = partition_loader(context)

    privacy_engine = PrivacyEngine(secure_mode=False)

    model, optim, trainloader = privacy_engine.make_private(
        module=model,
        optimizer=optim,
        data_loader=trainloader,
        noise_multiplier=noise_multiplier,
        max_grad_norm=max_grad_norm,
    )

    # Call the training function
    train_loss, epsilon = train_fn(
        model,
        trainloader,
        context.run_config["local-epochs"],
        device,
        target_delta=target_delta,
        privacy_engine=privacy_engine,
        optimizer=optim,
    )

    # Construct and return reply Message
    model_record = ArrayRecord(_unwrap_state_dict(model))
    metrics = {
        "train_loss": train_loss,
        "num-examples": len(trainloader.dataset),
        "epsilon": float(epsilon),
        "target_delta": float(target_delta),
        "noise_multiplier": float(noise_multiplier),
        "max_grad_norm": float(max_grad_norm),
    }
    metric_record = MetricRecord(metrics)
    content = RecordDict({"arrays": model_record, "metrics": metric_record})
    print(40 * "=")
    print(
        f"[client {pid}] epsilon(delta={target_delta})={epsilon:.2f}, noise={noise_multiplier}"
    )
    print(40 * "=")
    return Message(content=content, reply_to=msg)


@app.evaluate()
def evaluate(msg: Message, context: Context):
    """Evaluate the model on local data."""

    # Load the model and initialize it with the received weights
    model = Net()
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
    device = get_device()
    model.to(device)

    # Load the data
    _, _, _, valloader = partition_loader(context)

    # Call the evaluation function
    eval_loss, eval_acc = test_fn(
        model,
        valloader,
        device,
    )

    # Construct and return reply Message
    metrics = {
        "eval_loss": eval_loss,
        "eval_acc": eval_acc,
        "num-examples": len(valloader.dataset),
    }
    metric_record = MetricRecord(metrics)
    content = RecordDict({"metrics": metric_record})
    return Message(content=content, reply_to=msg)
