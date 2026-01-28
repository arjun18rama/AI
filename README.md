# Self-Play Humanoid Arena

This project trains two humanoid agents in a shared MuJoCo arena using PPO and a self-play opponent pool.

## Environment setup

### Prerequisites

- **Python 3.10+**
- **MuJoCo** runtime dependencies (see the official MuJoCo install docs for your OS)

### Create a virtual environment and install dependencies

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

For local development tools (formatting, linting, tests), also install:

```bash
pip install -r requirements-dev.txt
```

## Common scripts

These commands are available via the `Makefile`:

| Command | Description |
| --- | --- |
| `make install` | Install runtime dependencies. |
| `make install-dev` | Install runtime + dev dependencies. |
| `make train` | Run PPO self-play training with the default config. |
| `make lint` | Run Ruff lint checks. |
| `make format` | Format code with Black. |
| `make test` | Run tests (pytest). |

You can also run the training script directly:

```bash
python train.py --config configs/default.yaml
```

## Expected local tooling

- **Ruff** for linting (`ruff check .`)
- **Black** for formatting (`black .`)
- **Pytest** for tests (`pytest`)

Install these via `requirements-dev.txt` if you plan to run the developer tooling locally.
