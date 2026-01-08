from typing import TYPE_CHECKING, Any, cast

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812
from datasets import load_dataset
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import IidPartitioner
from opacus import PrivacyEngine
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Normalize, ToTensor

if TYPE_CHECKING:
    from collections.abc import Sized


class LightweightCNN(nn.Module):
    """
    Lightweight CNN model for CIFAR-10 classification with dropout regularization.

    Layer-by-layer breakdown:
    - Conv2d(3, 32, kernel=3) -> ReLU -> MaxPool2d(2) -> Dropout2d(0.25)    # Feature extraction
    - Conv2d(32, 64, kernel=3) -> ReLU -> MaxPool2d(2) -> Dropout2d(0.25)   # Feature refinement
    - Conv2d(64, 64, kernel=3) -> ReLU                                       # High-level features
    - Flatten with automatic dimension inference                             # Transition to classification
    - Linear(auto_inferred, 64) -> ReLU -> Dropout(0.5)                    # Feature compression
    - Linear(64, 10) -> Output (10 classes)                                 # Final classification
    """

    def __init__(self) -> None:
        """
        初始化轻量级 CNN 模型。

        构建模型各层:
        - 3 个卷积层用于特征提取
        - 2 个全连接层用于分类
        - Dropout 层用于正则化

        Attributes:
            conv1: 第一个卷积层 (3 -> 32 channels)
            conv2: 第二个卷积层 (32 -> 64 channels)
            conv3: 第三个卷积层 (64 -> 64 channels)
            pool: 最大池化层 (2x2)
            dropout_conv: 卷积层 Dropout (p=0.25)
            dropout_fc: 全连接层 Dropout (p=0.5)
            fc1: 第一个全连接层 (1024 -> 64)
            fc2: 第二个全连接层 (64 -> 10)
        """
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout_conv = nn.Dropout2d(0.25)  # Dropout for convolutional layers
        self.dropout_fc = nn.Dropout(0.5)  # Dropout for fully connected layers

        self.fc1 = nn.Linear(4 * 4 * 64, 64)
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播函数。

        Args:
            x: 输入张量,形状为 (batch_size, 3, 32, 32)

        Returns:
            torch.Tensor: 输出 logits,形状为 (batch_size, 10)

        Example:
            >>> import torch
            >>> model = LightweightCNN()
            >>> x = torch.randn(32, 3, 32, 32)
            >>> y = model(x)
            >>> y.shape  # torch.Size([32, 10])
        """
        # Layer 1: Feature extraction with dropout
        x = self.pool(F.relu(self.conv1(x)))
        x = self.dropout_conv(x)

        # Layer 2: Feature refinement with dropout
        x = self.pool(F.relu(self.conv2(x)))
        x = self.dropout_conv(x)

        # Layer 3: High-level features
        x = F.relu(self.conv3(x))

        # Flatten and fully connected layers
        x = x.view(-1, 4 * 4 * 64)
        x = F.relu(self.fc1(x))
        x = self.dropout_fc(x)
        x = self.fc2(x)
        return x


Net = LightweightCNN


fds = None  # Cache FederatedDataset

pytorch_transforms = Compose([ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


def apply_transforms(batch: dict[str, list[Any]]) -> dict[str, list[Any]]:
    """
    对 FederatedDataset 分区数据应用图像变换。

    将原始图像转换为 PyTorch 张量并进行归一化处理。
    归一化使用均值和标准差 (0.5, 0.5, 0.5) 将像素值映射到 [-1, 1] 范围。

    Args:
        batch: 包含 'img' 键的字典,值为原始 PIL 图像列表

    Returns:
        dict: 包含 'img' 键的字典,值为变换后的 PyTorch 张量列表

    Example:
        >>> batch = {"img": [pil_image1, pil_image2]}
        >>> transformed = apply_transforms(batch)
        >>> transformed["img"][0].shape  # torch.Size([3, 32, 32])
    """
    batch["img"] = [pytorch_transforms(img) for img in batch["img"]]
    return batch


def load_data(
    partition_id: int, num_partitions: int, batch_size: int
) -> tuple[DataLoader[Any], DataLoader[Any]]:
    """
    加载 CIFAR-10 联邦数据集的指定分区。

    使用 IidPartitioner 将 CIFAR-10 训练集划分为多个均匀分区,
    每个分区进一步划分为 80% 训练数据和 20% 测试数据。

    Args:
        partition_id: 要加载的分区 ID (0 到 num_partitions-1)
        num_partitions: 总分区数
        batch_size: DataLoader 的批次大小

    Returns:
        tuple: (训练 DataLoader, 测试 DataLoader)

    Side Effects:
        首次调用时从 HuggingFace 加载数据集,会创建全局 FederatedDataset 缓存

    Example:
        >>> trainloader, testloader = load_data(0, 10, 32)
        >>> len(trainloader)  # 训练批次数
    """
    # Only initialize `FederatedDataset` once
    global fds
    if fds is None:
        partitioner = IidPartitioner(num_partitions=num_partitions)
        fds = FederatedDataset(
            dataset="uoft-cs/cifar10",
            partitioners={"train": partitioner},
        )
    partition = fds.load_partition(partition_id)
    # Divide data on each node: 80% train, 20% test
    partition_train_test = partition.train_test_split(test_size=0.2, seed=42)
    # Construct dataloaders
    partition_train_test = partition_train_test.with_transform(apply_transforms)
    trainloader = DataLoader(
        partition_train_test["train"], batch_size=batch_size, shuffle=True
    )
    testloader = DataLoader(partition_train_test["test"], batch_size=batch_size)
    return trainloader, testloader


def load_centralized_dataset() -> DataLoader[Any]:
    """
    加载 CIFAR-10 测试集作为集中式评估数据。

    从 HuggingFace 加载完整的 CIFAR-10 测试集,
    应用与训练集相同的图像变换,并返回 DataLoader。

    Returns:
        DataLoader: 包含完整测试集的 DataLoader,批次大小为 128

    Example:
        >>> testloader = load_centralized_dataset()
        >>> len(testloader)  # 79 (10000 / 128)
    """
    # Load entire test set
    test_dataset = load_dataset("uoft-cs/cifar10", split="test")
    dataset = test_dataset.with_format("torch").with_transform(apply_transforms)
    return DataLoader(dataset, batch_size=128)


def train(
    net: nn.Module,
    trainloader: DataLoader[Any],
    epochs: int,
    device: str,
    privacy_engine: PrivacyEngine,
    target_delta: float,
    optimizer: Optimizer,
) -> tuple[float, float]:
    """
    使用差分隐私在训练集上训练模型。

    执行指定轮数的本地训练,使用 DP-SGD 进行梯度裁剪和噪声添加,
    训练完成后计算并返回平均训练损失和差分隐私预算 epsilon。

    Args:
        net: PyTorch 模型
        trainloader: 训练数据 DataLoader
        epochs: 本地训练轮数
        device: 计算设备 ("cuda", "mps", 或 "cpu")
        privacy_engine: Opacus PrivacyEngine 实例
        target_delta: 差分隐私目标 delta 值
        optimizer: 优化器 (SGD with momentum)

    Returns:
        tuple: (平均训练损失, 差分隐私预算 epsilon)

    Example:
        >>> from opacus import PrivacyEngine
        >>> engine = PrivacyEngine(secure_mode=False)
        >>> loss, epsilon = train(model, trainloader, 1, "cpu", engine, 1e-5, optimizer)
        >>> print(f"Loss: {loss:.4f}, Epsilon: {epsilon:.2f}")
    """
    net.to(device)  # move model to GPU if available
    criterion = torch.nn.CrossEntropyLoss().to(device)
    net.train()
    running_loss = 0.0
    for _ in range(epochs):
        for batch in trainloader:
            images = batch["img"].to(device)
            labels = batch["label"].to(device)
            optimizer.zero_grad()
            loss = criterion(net(images), labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

    epsilon = privacy_engine.get_epsilon(delta=target_delta)
    avg_trainloss = running_loss / len(trainloader)

    return avg_trainloss, epsilon


def test(
    net: nn.Module, testloader: DataLoader[Any], device: str
) -> tuple[float, float]:
    """
    在测试集上评估模型性能。

    遍历测试集计算预测准确率和平均损失值。

    Args:
        net: PyTorch 模型
        testloader: 测试数据 DataLoader
        device: 计算设备 ("cuda", "mps", 或 "cpu")

    Returns:
        tuple: (平均损失值, 准确率 (0-1))

    Example:
        >>> loss, accuracy = test(model, testloader, "cpu")
        >>> print(f"Loss: {loss:.4f}, Accuracy: {accuracy:.2%}")
    """
    net.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    correct, loss = 0, 0.0
    with torch.no_grad():
        for batch in testloader:
            images = batch["img"].to(device)
            labels = batch["label"].to(device)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
    accuracy = correct / len(cast("Sized", cast("object", testloader.dataset)))
    loss = loss / len(testloader)
    return loss, accuracy
