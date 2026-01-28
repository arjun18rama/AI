# macOS setup & one-command run

## First-time setup (downloads repo + installs dependencies)

1. Pick a folder where you want the repo cloned (defaults to `~/Downloads`).
2. Run the bootstrap script from anywhere, passing the Git repo URL. If you have a raw file URL handy, you can also `curl` it directly:

```bash
./scripts/bootstrap_macos.sh https://github.com/your-org/ai-selfplay.git
```

```bash
curl -fsSL https://raw.githubusercontent.com/your-org/ai-selfplay/main/scripts/bootstrap_macos.sh | \\
  bash -s -- https://github.com/your-org/ai-selfplay.git
```

Optional environment variables:
- `INSTALL_ROOT`: parent directory for the clone (default: `~/Downloads`).
- `REPO_DIR_NAME`: folder name for the clone (default: `ai-selfplay`).

Example with overrides:
```bash
INSTALL_ROOT="$HOME" REPO_DIR_NAME="ai-selfplay" \
  ./scripts/bootstrap_macos.sh https://github.com/your-org/ai-selfplay.git
```

## Daily run (single command)

From the repo directory:

```bash
./scripts/run_macos.sh
```

This activates the virtual environment, ensures requirements are installed, and runs the default training config.

## Troubleshooting

- If you see "Virtual environment not found", rerun the bootstrap script.
- Ensure Xcode Command Line Tools are installed (`xcode-select --install`).
- MuJoCo installation is required for the environment. Follow MuJoCo's macOS install steps if you get import errors.
