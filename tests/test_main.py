import os
from unittest.mock import MagicMock, patch

import pytest
import torch
from src.config import Config
from src.main import main  # まだ存在しないため、このimportは失敗するはず


class TestMain:
    @pytest.fixture(scope="class")
    def config(self) -> Config:
        # テスト用のConfigを作成。保存パスを一時的なものにする。
        return Config(
            epochs=1,
            batch_size=4,
            log_interval=1,
            save_path="models/test_mnist_model.pth",
        )

    @pytest.fixture(autouse=True)
    def setup_and_teardown(self, config: Config) -> None:
        # テスト実行前にmodelsディレクトリが存在することを確認し、
        # テスト実行後に作成されたモデルファイルを削除する
        models_dir = os.path.dirname(config.save_path)
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)
        yield
        if os.path.exists(config.save_path):
            os.remove(config.save_path)
        if os.path.exists(models_dir) and not os.listdir(models_dir):
            os.rmdir(models_dir)

    @patch("src.train.train")
    @patch("src.train.validate")
    @patch("torch.save")
    def test_main_function_runs(
        self,
        mock_torch_save: MagicMock,
        mock_validate: MagicMock,
        mock_train: MagicMock,
        config: Config,
    ) -> None:
        """main関数がエラーなく実行され、主要な関数が呼び出されることをテスト"""
        # trainとvalidateがモックされているため、実際の学習は行われない
        mock_train.return_value = 0.5
        mock_validate.return_value = (0.5, 90.0)

        main(config)

        # train関数が呼び出されたことを確認
        mock_train.assert_called_once()
        # validate関数が呼び出されたことを確認
        mock_validate.assert_called_once()
        # torch.saveが呼び出されたことを確認
        mock_torch_save.assert_called_once()

        # モデルファイルが実際に保存されたかを確認
        assert os.path.exists(config.save_path)

    @patch("src.train.train")
    @patch("src.train.validate")
    @patch("torch.save")
    def test_main_function_prints_output(
        self,
        mock_torch_save: MagicMock,
        mock_validate: MagicMock,
        mock_train: MagicMock,
        config: Config,
        capsys,
    ) -> None:
        """main関数が正しく情報を出力することを確認するテスト"""
        mock_train.return_value = 0.5
        mock_validate.return_value = (0.5, 90.0)

        main(config)
        captured = capsys.readouterr()

        # 訓練開始、終了、モデル保存などのメッセージが出力されることを確認
        assert "Training started" in captured.out
        assert "Training finished" in captured.out
        assert f"Model saved to {config.save_path}" in captured.out
        assert (
            "Validation set: Average loss: 0.5000, Accuracy: 90%" in captured.out
        )  # validateモックの出力
