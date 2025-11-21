from typing import Union

import torch
import torch.nn.functional as F
from src.config import Config
from src.model import SimpleCNN


def predict(model: SimpleCNN, device: torch.device, data: torch.Tensor) -> torch.Tensor:
    """
    与えられたデータに対してモデルの推論を実行する。

    Args:
        model (SimpleCNN): 推論に使用するモデル。
        device (torch.device): 推論に使用するデバイス (CPU)。
        data (torch.Tensor): 推論対象の入力データ。

    Returns:
        torch.Tensor: 各入力データに対する予測クラスのインデックス。
    """
    model.eval()  # 評価モードに設定
    with torch.no_grad():
        data = data.to(device)
        output = model(data)
        # 確率ではなく、最も確率の高いクラスのインデックスを返す
        pred = output.argmax(dim=1, keepdim=False)
    return pred
