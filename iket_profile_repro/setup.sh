#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
UV_DIR="$ROOT/.uv"
export UV_CACHE_DIR="$ROOT/.uv-cache"
export TMPDIR="$ROOT/tmp"
mkdir -p "$UV_CACHE_DIR" "$TMPDIR"

if [ ! -x "$UV_DIR/uv" ]; then
  curl -LsSf https://astral.sh/uv/install.sh | \
    env UV_INSTALL_DIR="$UV_DIR" UV_NO_MODIFY_PATH=1 sh
fi

"$UV_DIR/uv" venv --python 3.12 "$ROOT/.venv"
"$UV_DIR/uv" pip install --python "$ROOT/.venv/bin/python" -r "$ROOT/requirements.lock"

echo "Environment ready: $ROOT/.venv"
