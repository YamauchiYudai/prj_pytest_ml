import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from src.config import Config
from src.data_loader import MNISTDataLoader
from src.model import SimpleCNN
from src.train import train, validate  # trainモジュールはまだ存在しない


def main(config: Config) -> None:
    """
    MNIST分類モデルの訓練と評価を実行するメイン関数。
    """
    # Reproducibility
    torch.manual_seed(0)
    random.seed(0)
    np.random.seed(0)

    device = torch.device("cpu")  # CPU環境

    print("--- Loading Data ---")
    train_loader = MNISTDataLoader(config, train=True).loader
    val_loader = MNISTDataLoader(config, train=False).loader

    print("--- Initializing Model ---")
    model = SimpleCNN()
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    criterion = nn.CrossEntropyLoss()

    print("--- Training started ---")
    best_accuracy = 0.0
    for epoch in range(1, config.epochs + 1):
        train_loss = train(
            model, device, train_loader, optimizer, criterion, epoch, config
        )
        val_loss, val_accuracy = validate(model, device, val_loader, criterion, config)

        # 最も良いモデルを保存
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            # モデル保存ディレクトリが存在しない場合は作成
            os.makedirs(os.path.dirname(config.save_path), exist_ok=True)
            torch.save(model.state_dict(), config.save_path)
            print(
                f"Model saved to {config.save_path} with accuracy: {best_accuracy:.2f}%"
            )

    print("--- Training finished ---")
    print(f"Best validation accuracy: {best_accuracy:.2f}%")


if __name__ == "__main__":
    config = Config()
    main(config)
