import pytest
import torch
import torch.nn as nn
from src.config import Config
from src.model import SimpleCNN
from src.predict import predict  # まだ存在しないため、このimportは失敗するはず


class TestPrediction:
    @pytest.fixture(scope="class")
    def config(self) -> Config:
        return Config(batch_size=4, image_size=28, channels=1, num_classes=10)

    @pytest.fixture(scope="class")
    def device(self) -> torch.device:
        return torch.device("cpu")

    @pytest.fixture(scope="class")
    def model(self, config: Config) -> SimpleCNN:
        # テストのためにダミーの学習済みモデルを作成
        model = SimpleCNN()
        # モデルのパラメータをランダムに初期化するだけなので、実際の学習は行わない
        return model

    @pytest.fixture(scope="class")
    def dummy_input(self, config: Config) -> torch.Tensor:
        # ダミーの入力データ (batch_size, channels, image_size, image_size)
        return torch.randn(
            config.batch_size, config.channels, config.image_size, config.image_size
        )

    def test_predict_function(
        self, model: SimpleCNN, device: torch.device, dummy_input: torch.Tensor
    ) -> None:
        """predict関数が正しく動作し、適切な形式の予測を返すことをテスト"""
        model.eval()  # 推論モードに設定
        output = predict(model, device, dummy_input)

        # 出力がtorch.Tensorであること
        assert isinstance(output, torch.Tensor)
        # 出力のバッチサイズが入力と同じであること
        assert output.shape[0] == dummy_input.shape[0]
        # 出力がクラス数に対応するロジットまたは確率であること
        # 最も確率の高いクラスのインデックスを返すこと（argmaxの結果を想定）
        assert output.dtype == torch.int64  # argmaxの結果はint64
        assert output.max() < 10  # クラスのインデックスは0-9の範囲内
        assert output.min() >= 0  # クラスのインデックスは0以上
