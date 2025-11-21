from typing import Any, Dict

import torch
from src.config import Config
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


class MNISTDataLoader:
    def __init__(self, config: Config, train: bool = True) -> None:
        self.config = config
        self.train = train

        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        )

        dataset = datasets.MNIST(
            root="./data", train=train, download=True, transform=transform
        )

        self.loader = DataLoader(
            dataset,
            batch_size=config.batch_size,
            shuffle=train,
            num_workers=2,  # CPU環境なので2 workerに設定
        )
