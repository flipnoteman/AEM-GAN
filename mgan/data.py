"""MNIST loader.

Images are returned as float tensors in [-1, 1] with shape [B, 1, 28, 28],
matching the generator's tanh output.  Labels are dropped: memoryGAN is
unsupervised.
"""

from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def mnist_loader(root: Path, batch_size: int, train: bool = True) -> DataLoader:
    """Build a shuffled MNIST loader, downloading to ``root`` on first use."""
    root.mkdir(parents=True, exist_ok=True)
    transform = transforms.Compose([
        transforms.ToTensor(),                  # [0, 1]
        transforms.Normalize((0.5,), (0.5,)),   # [-1, 1]
    ])
    dataset = datasets.MNIST(str(root), train=train, download=True, transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)


def images_only(batch) -> torch.Tensor:
    """Strip labels from a ``(images, labels)`` batch."""
    return batch[0]
