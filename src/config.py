from dataclasses import dataclass


@dataclass
class Config:
    """
    ハイパーパラメータ管理用クラス
    """

    image_size: int = 28
    num_classes: int = 10
    channels: int = 1  # MNISTはグレースケールのため1チャンネル
    batch_size: int = 64
    epochs: int = 10
    learning_rate: float = 0.001
    log_interval: int = 100  # 何バッチごとにログを出力するか
    save_path: str = "models/mnist_model.pth"
