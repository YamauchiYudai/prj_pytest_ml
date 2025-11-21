# CLAUDE.md
# Project Architecture & Commands

## Tech Stack
- **Language**: Python 3.9+
- **ML Framework**: PyTorch (CPU version)
- **Testing**: Pytest
- **Linting/Formatting**: Pysen (Black, Isort, Mypy)
- **Infrastructure**: Docker / Docker Compose

## Project Structure
```text
.
├── src/                # Source code
│   ├── config.py       # All hyperparameters (dataclass)
│   ├── model.py        # Neural Network definition
│   ├── dataset.py      # Data loading logic
│   └── train.py        # Training loop
├── tests/              # Tests (Mirror src structure)
│   ├── test_model.py
│   └── test_train.py
├── docker/             # Docker config (if needed)
├── Dockerfile
├── docker-compose.yml
├── pyproject.toml      # Pysen & Tool config
├── requirements.txt
└── AGENTS.md / CLAUDE.md
````

## Essential Commands (Docker Wrapper)

All commands must be run via Docker to ensure reproducibility.

### 1\. Setup & Build

```bash
docker compose build
```

### 2\. Testing (Priority)

Run this frequently.

```bash
docker compose run --rm app pytest
```

### 3\. Linting & Formatting

Run this before completing any task.

```bash
# Format code
docker compose run --rm app pysen run format
# Check types and lint
docker compose run --rm app pysen run lint
```

### 4\. Training

```bash
docker compose run --rm app python src/train.py
```

## Implementation Details

  - **MNIST**: Use `torchvision.datasets.MNIST`.
  - **Model**: A simple CNN (Convolution -\> ReLU -\> Pool -\> Linear).
  - **Reproducibility**: Set `torch.manual_seed` and `random.seed`.
