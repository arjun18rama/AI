#!/usr/bin/env bash
set -euo pipefail

REPO_URL="${1:-}"
INSTALL_ROOT="${INSTALL_ROOT:-$HOME/Downloads}"
REPO_DIR_NAME="${REPO_DIR_NAME:-ai-selfplay}"
PROJECT_DIR="$INSTALL_ROOT/$REPO_DIR_NAME"

if [[ -z "$REPO_URL" ]]; then
  echo "Usage: $0 <repo_url>" >&2
  echo "Example: $0 https://github.com/your-org/ai-selfplay.git" >&2
  exit 1
fi

mkdir -p "$INSTALL_ROOT"

if [[ -d "$PROJECT_DIR/.git" ]]; then
  echo "Repo already exists at $PROJECT_DIR. Pulling latest changes."
  git -C "$PROJECT_DIR" pull --ff-only
else
  echo "Cloning repo into $PROJECT_DIR"
  git clone "$REPO_URL" "$PROJECT_DIR"
fi

cd "$PROJECT_DIR"

python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

echo "Setup complete."
echo "Next time, run: $PROJECT_DIR/scripts/run_macos.sh"
