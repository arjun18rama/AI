#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

cd "$PROJECT_DIR"

if [[ ! -d .venv ]]; then
  echo "Virtual environment not found. Run scripts/bootstrap_macos.sh first." >&2
  exit 1
fi

source .venv/bin/activate
pip install -r requirements.txt

python train.py --config configs/default.yaml
