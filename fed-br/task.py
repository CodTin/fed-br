import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import IidPartitioner
from opacus import PrivacyEngine
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Normalize, ToTensor


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

    def __init__(self):
        super(LightweightCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout_conv = nn.Dropout2d(0.25)  # Dropout for convolutional layers
        self.dropout_fc = nn.Dropout(0.5)  # Dropout for fully connected layers

        self.fc1 = nn.Linear(4 * 4 * 64, 64)
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x: torch.Tensor):
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


def apply_transforms(batch):
    """Apply transforms to the partition from FederatedDataset."""
    batch["img"] = [pytorch_transforms(img) for img in batch["img"]]
    return batch


def load_data(partition_id: int, num_partitions: int, batch_size: int):
    """Load partition CIFAR10 data."""
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


def load_centralized_dataset():
    """Load test set and return dataloader."""
    # Load entire test set
    test_dataset = load_dataset("uoft-cs/cifar10", split="test")
    dataset = test_dataset.with_format("torch").with_transform(apply_transforms)
    return DataLoader(dataset, batch_size=128)


def train(net: nn.Module, trainloader: DataLoader, epochs: int, lr: float, device: str, privacy_engine: PrivacyEngine):
    """Train the model on the training set."""
    net.to(device)  # move model to GPU if available
    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9)
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
    avg_trainloss = running_loss / len(trainloader)
    return avg_trainloss


def test(net: nn.Module, testloader: DataLoader, device: str):
    """Validate the model on the test set."""
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
    accuracy = correct / len(testloader.dataset)
    loss = loss / len(testloader)
    return loss, accuracy
