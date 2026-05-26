"""
MNIST MLP model and data utilities.

The model is compatible with torch.jit.script: all forward() arguments
and return types are annotated, only TorchScript-safe ops are used.
"""
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

DATA_DIR = os.path.expanduser("~/.fl_mnist_data")


class MNISTMLP(nn.Module):
    """Two-layer fully-connected network for MNIST classification (10 classes)."""

    def __init__(self, hidden: int = 256) -> None:
        super().__init__()
        self.fc1 = nn.Linear(784, hidden)
        self.fc2 = nn.Linear(hidden, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(-1, 784)
        x = torch.relu(self.fc1(x))
        return torch.log_softmax(self.fc2(x), dim=1)


def get_data_partition(
    client_id: int,
    num_clients: int,
    train: bool = True,
    batch_size: int = 64,
) -> DataLoader:
    """
    Return a DataLoader with a contiguous slice of the MNIST split.

    The dataset is divided into num_clients equal shards by index.
    This gives an IID data distribution across clients.
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    dataset = datasets.MNIST(DATA_DIR, train=train, download=True, transform=transform)

    total = len(dataset)
    shard_size = total // num_clients
    start = client_id * shard_size
    end = start + shard_size
    indices = list(range(start, end))

    return DataLoader(Subset(dataset, indices), batch_size=batch_size, shuffle=True)
