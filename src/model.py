import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleCNN(nn.Module):
    """
    MNIST分類のためのシンプルなCNNモデル。
    入力: (batch_size, 1, 28, 28)
    出力: (batch_size, 10)
    """

    def __init__(self) -> None:
        super().__init__()
        # 28x28の単一チャンネル画像入力
        # 1チャンネル入力、32個のフィルタ、カーネルサイズ3x3
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        # 2x2の最大プーリング
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        # 全結合層
        # 活性化後の特徴マップのサイズを計算:
        # conv1: 28x28 -> padding=1, kernel=3, stride=1 -> (28-3+2*1)/1 + 1 = 28x28
        # pool: 28x28 -> kernel=2, stride=2 -> 28/2 = 14x14
        # 32チャネル * 14 * 14 の特徴が平坦化される
        self.fc1 = nn.Linear(32 * 14 * 14, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not isinstance(x, torch.Tensor):
            raise TypeError("Input must be a torch.Tensor")

        # 畳み込み層 -> ReLU -> プーリング層
        x = self.pool(F.relu(self.conv1(x)))
        # テンソルをフラット化（バッチサイズを維持）
        x = x.view(-1, 32 * 14 * 14)
        # 全結合層
        x = self.fc1(x)
        return x
