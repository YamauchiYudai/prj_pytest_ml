from typing import Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from src.config import Config
from src.model import SimpleCNN
from torch.utils.data import DataLoader


def train(
    model: SimpleCNN,
    device: torch.device,
    train_loader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    epoch: int,
    config: Config,
) -> float:
    """
    モデルの訓練ループを実行する。

    Args:
        model (SimpleCNN): 訓練するモデル。
        device (torch.device): 訓練に使用するデバイス (CPU)。
        train_loader (DataLoader): 訓練データローダー。
        optimizer (optim.Optimizer): オプティマイザ。
        criterion (nn.Module): 損失関数。
        epoch (int): 現在のエポック数。
        config (Config): 設定オブジェクト。

    Returns:
        float: 訓練損失。
    """
    model.train()
    total_loss = 0.0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        if batch_idx % config.log_interval == 0:
            print(
                f"Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} ({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}"
            )
    return total_loss / len(train_loader)


def validate(
    model: SimpleCNN,
    device: torch.device,
    val_loader: DataLoader,
    criterion: nn.Module,
    config: Config,
) -> Tuple[float, float]:
    """
    モデルの評価ループを実行する。

    Args:
        model (SimpleCNN): 評価するモデル。
        device (torch.device): 評価に使用するデバイス (CPU)。
        val_loader (DataLoader): 検証データローダー。
        criterion (nn.Module): 損失関数。
        config (Config): 設定オブジェクト。

    Returns:
        Tuple[float, float]: 検証損失と精度。
    """
    model.eval()
    val_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            val_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    val_loss /= len(val_loader)
    accuracy = 100.0 * correct / len(val_loader.dataset)

    print(
        f"\nValidation set: Average loss: {val_loss:.4f}, Accuracy: {correct}/{len(val_loader.dataset)} ({accuracy:.0f}%)\n"
    )
    return val_loss, accuracy


if __name__ == "__main__":
    # このスクリプトが直接実行された場合の簡易的なテスト（通常はmain.pyから呼び出される）
    import random

    import numpy as np
    from src.config import Config
    from src.data_loader import MNISTDataLoader

    # Reproducibility
    torch.manual_seed(0)
    random.seed(0)
    np.random.seed(0)

    config = Config()
    device = torch.device("cpu")  # CPU環境

    train_loader = MNISTDataLoader(config, train=True)
    val_loader = MNISTDataLoader(config, train=False)

    model = SimpleCNN()
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    criterion = nn.CrossEntropyLoss()

    print("--- Dummy Training Start ---")
    for epoch in range(1, config.epochs + 1):
        train_loss = train(
            model, device, train_loader.loader, optimizer, criterion, epoch, config
        )
        val_loss, val_accuracy = validate(
            model, device, val_loader.loader, criterion, config
        )
        print(
            f"Epoch {epoch}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}, Val Accuracy = {val_accuracy:.2f}%"
        )
    print("--- Dummy Training End ---")
