from typing import Tuple

import pytest
import torch

# モデルが存在しないため、ここでは仮のパスを定義
# 実際のモデルはsrc/model.pyに実装されます
# from src.model import SimpleCNN


class TestModel:
    """
    モデルのテストクラス。
    入出力のシェイプを検証する。
    """

    def test_model_output_shape(self) -> None:
        """
        モデルの出力シェイプが期待通りであることを確認する。
        MNISTデータセットの入力 (1, 28, 28) と10クラスの出力 (10) に対応。
        """
        batch_size = 4
        input_shape = (batch_size, 1, 28, 28)
        expected_output_shape = (batch_size, 10)

        # src/model.pyにSimpleCNNが実装されていないため、このテストは失敗するはずです。
        # まずはモデルがインポートできないこと、またはモデルの __init__ が未定義であることを確認します。
        # 後のステップでSimpleCNNを実装し、このテストをパスさせます。
        try:
            from src.model import SimpleCNN

            model = SimpleCNN()
            dummy_input = torch.randn(input_shape)
            output = model(dummy_input)
            assert (
                output.shape == expected_output_shape
            ), f"Expected output shape {expected_output_shape}, but got {output.shape}"
        except ImportError:
            pytest.fail("SimpleCNNクラスがsrc/model.pyに存在しません。")
        except AttributeError:
            pytest.fail("SimpleCNNクラスが適切に定義されていません。")

    def test_model_input_type(self) -> None:
        """
        モデルがtorch.Tensor以外の入力を受け付けないことを確認する。
        """
        batch_size = 4
        # ダミー入力として、意図的に数値のリストを使用
        dummy_input = [[0.0] * 28 * 28] * batch_size

        try:
            from src.model import SimpleCNN

            model = SimpleCNN()
            with pytest.raises(TypeError):
                # 型ヒントに反する入力を与える
                model(dummy_input)  # type: ignore
        except ImportError:
            pytest.fail("SimpleCNNクラスがsrc/model.pyに存在しません。")
        except AttributeError:
            pytest.fail("SimpleCNNクラスが適切に定義されていません。")
