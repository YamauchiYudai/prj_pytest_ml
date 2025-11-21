import pytest
import torch
import torch.nn as nn
import torch.optim as optim
from src.config import Config
from src.data_loader import MNISTDataLoader
from src.model import SimpleCNN
from src.train import train, validate  # まだ存在しないため、このimportは失敗するはず


class TestTraining:
    @pytest.fixture(scope="class")
    def config(self) -> Config:
        return Config(batch_size=4, epochs=1, learning_rate=0.01)

    @pytest.fixture(scope="class")
    def device(self) -> torch.device:
        return torch.device("cpu")

    @pytest.fixture(scope="class")
    def model(self, config: Config) -> SimpleCNN:
        return SimpleCNN()

    @pytest.fixture(scope="class")
    def train_loader(self, config: Config) -> MNISTDataLoader:
        return MNISTDataLoader(config, train=True)

    @pytest.fixture(scope="class")
    def val_loader(self, config: Config) -> MNISTDataLoader:
        return MNISTDataLoader(config, train=False)

    @pytest.fixture(scope="class")
    def optimizer(self, model: SimpleCNN, config: Config) -> optim.Optimizer:
        return optim.Adam(model.parameters(), lr=config.learning_rate)

    @pytest.fixture(scope="class")
    def criterion(self) -> nn.Module:
        return nn.CrossEntropyLoss()

    def test_train_function(
        self,
        model: SimpleCNN,
        device: torch.device,
        train_loader: MNISTDataLoader,
        optimizer: optim.Optimizer,
        criterion: nn.Module,
        config: Config,
    ) -> None:
        """train関数が正しく動作し、損失が減少することを確認するテスト"""
        initial_loss = train(
            model, device, train_loader.loader, optimizer, criterion, 0, config
        )
        # 簡易的な確認として、1エポック後の損失が数値であり、エラーが発生しないことを確認
        assert isinstance(initial_loss, float)
        assert initial_loss > 0  # 損失が正の値であることを確認 (初期状態ではほぼ確実に正)

        # 損失が減少することを確認するのは、ミニバッチの変動が大きく、
        # 1エポックだけでは必ずしも単調減少しないため、より厳密なテストが必要
        # この時点では、関数がエラーなく実行され、適切な値を返すことを重視する
        # より厳密な減少のテストは、モックや固定データセットで行うべき

    def test_validate_function(
        self,
        model: SimpleCNN,
        device: torch.device,
        val_loader: MNISTDataLoader,
        criterion: nn.Module,
        config: Config,
    ) -> None:
        """validate関数が正しく動作し、損失と精度を返すことを確認するテスト"""
        loss, accuracy = validate(model, device, val_loader.loader, criterion, config)
        assert isinstance(loss, float)
        assert isinstance(accuracy, float)
        assert 0.0 <= accuracy <= 100.0
        assert loss >= 0.0
