# MNIST Classification with PyTorch (CPU)

このプロジェクトは、PyTorch (CPU版) を使用してMNIST手書き数字データセットを分類するモデルを構築し、テスト駆動開発 (TDD) のアプローチで実装されています。

## Tech Stack
- **Language**: Python 3.9+
- **ML Framework**: PyTorch (CPU version)
- **Testing**: Pytest
- **Linting/Formatting**: Pysen (Black, Isort, Mypy)
- **Infrastructure**: Docker / Docker Compose

## Project Structure
```text
.
├── src/                # ソースコード
│   ├── config.py       # 全てのハイパーパラメータを定義
│   ├── model.py        # ニューラルネットワークモデルの定義 (SimpleCNN)
│   ├── data_loader.py  # データローディングと前処理のロジック
│   ├── train.py        # モデルの訓練および検証ループ
│   ├── predict.py      # 学習済みモデルによる推論機能
│   └── main.py         # プロジェクトのエントリポイント (訓練とモデル保存を統括)
├── tests/              # テストコード (srcディレクトリの構造をミラー)
│   ├── test_model.py       # model.pyのテスト
│   ├── test_data_loader.py # data_loader.pyのテスト
│   ├── test_train.py       # train.pyのテスト
│   └── test_predict.py     # predict.pyのテスト
├── Dockerfile
├── docker-compose.yml
├── pyproject.toml      # Pysen およびその他のツール設定
├── requirements.txt    # Pythonパッケージの依存関係
├── AGENTS.md           # エージェントの行動指針
└── CLAUDE.md           # プロジェクトのアーキテクチャとコマンド
```

## TDD (テスト駆動開発) プロセス
このプロジェクトは、以下のTDDサイクルに従って開発されました。
1.  **Red (Test First)**: 機能要件を満たすための「失敗するテスト」を `tests/` に作成し、テストが失敗することを確認します。これにより、必要な機能と設計が明確になります。
2.  **Green (Minimal Implementation)**: テストを通過させるための最小限の実装を `src/` に行います。この段階では、機能を満たすことに焦点を当て、複雑な設計や最適化は後回しにします。型ヒントは必ず記述します。
3.  **Refactor**: コードを整理し、可読性、保守性、効率性を高めます。必ず `pysen` を実行して静的解析を通し、コード品質を確保します。

## Setup & Installation (環境構築)

1.  **リポジトリをクローンします:**
    ```bash
    git clone https://github.com/your-repo/MNIST_withPytest.git
    cd MNIST_withPytest
    ```
2.  **Dockerイメージをビルドします:**
    ```bash
    docker compose build
    ```
    これにより、`Dockerfile`と`requirements.txt`に基づいて必要な依存関係がインストールされたDockerイメージが作成されます。

## Usage (使い方)

すべてのコマンドは、再現性を確保するためにDockerを介して実行する必要があります。

### 1. Run Tests (テストの実行)
開発中に頻繁に実行し、コードの正しさを確認します。
```bash
docker compose run --rm app pytest
```
特定のテストファイルのみを実行する場合は、パスを指定します (例: `docker compose run --rm app pytest tests/test_model.py`)。

### 2. Linting & Formatting (コードの整形と静的解析)
コードの品質を維持するために、コミット前などに実行します。
```bash
# コードのフォーマット（isortとblack）
docker compose run --rm app pysen run format
# 型チェックとリント（mypy）
docker compose run --rm app pysen run lint
```

### 3. Training (モデルの訓練)
MNIST分類モデルの訓練を開始します。訓練済みモデルは `models/mnist_model.pth` に保存されます。
```bash
docker compose run --rm app python src/main.py
```
訓練中にデータセットがダウンロードされ、訓練の進捗状況と検証精度が表示されます。

## Code Explanation (コードの説明)

### `src/config.py`
データクラス `Config` を使用して、バッチサイズ、エポック数、学習率、モデル保存パスなど、プロジェクト全体のハイパーパラメータを一元管理します。

### `src/model.py`
`SimpleCNN` クラスとして、MNIST画像を分類するためのシンプルな畳み込みニューラルネットワークを定義します。畳み込み層、ReLU活性化関数、最大プーリング層、全結合層で構成されます。

### `src/data_loader.py`
`MNISTDataLoader` クラスを提供し、`torchvision.datasets.MNIST` を使用してMNISTデータセットをロードします。データをテンソルに変換し、標準化する前処理 (`transforms.ToTensor()`, `transforms.Normalize()`) を適用し、`torch.utils.data.DataLoader` を介してバッチ処理されたデータを返します。

### `src/train.py`
`train` 関数と `validate` 関数を定義します。
- `train`: モデルを訓練モードに設定し、指定されたエポックに対して訓練データローダーからバッチを受け取り、損失計算、バックプロパゲーション、オプティマイザのステップを実行します。
- `validate`: モデルを評価モードに設定し、検証データローダーを使用してモデルの損失と精度を計算します。

### `src/predict.py`
`predict` 関数を定義します。学習済みモデルと入力データを受け取り、モデルを評価モードで実行し、各入力に対する予測クラスのインデックスを返します。

### `src/main.py`
プロジェクトのエントリポイントです。
`main` 関数内で `Config` の初期化、データローダーの準備、モデルとオプティマイザ、損失関数の初期化、訓練ループの実行、そして最も性能の良いモデルの保存を行います。再現性のためのシード設定もここで行われます。
