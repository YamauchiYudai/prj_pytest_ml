import pytest
import torch
from src.config import Config
from src.data_loader import MNISTDataLoader  # まだ存在しないため、このimportは失敗するはず
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


class TestMNISTDataLoader:
    @pytest.fixture(scope="class")
    def config(self) -> Config:
        return Config(batch_size=4, image_size=28, channels=1)

    @pytest.fixture(scope="class")
    def data_loader(self, config: Config) -> MNISTDataLoader:
        return MNISTDataLoader(config, train=True)

    def test_dataloader_creation(self, data_loader: MNISTDataLoader) -> None:
        """データローダーが正しく作成されるかテスト"""
        assert isinstance(data_loader.loader, DataLoader)
        assert data_loader.loader.batch_size == data_loader.config.batch_size

    def test_data_shape_and_type(self, data_loader: MNISTDataLoader) -> None:
        """データローダーが返すデータのシェイプと型が正しいかテスト"""
        for data, target in data_loader.loader:
            assert data.shape == (
                data_loader.config.batch_size,
                data_loader.config.channels,
                data_loader.config.image_size,
                data_loader.config.image_size,
            )
            assert target.shape == (data_loader.config.batch_size,)
            assert data.dtype == torch.float32
            assert target.dtype == torch.int64
            break  # 最初のバッチのみ確認
