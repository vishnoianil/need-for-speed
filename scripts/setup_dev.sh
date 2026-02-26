#!/usr/bin/env bash
# Setup script for Triton GEMM environment
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_DIR"

# Check if uv is available
if ! command -v uv &> /dev/null; then
    echo "uv not found. Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | bash
fi

# Create virtual environment if it doesn't exist
if [ ! -d "$PROJECT_DIR/.venv" ]; then
    echo "Creating virtual environment..."
    uv -m venv "$PROJECT_DIR/.venv" --python 3.12 --seed
fi

# Activate and install dependencies
source "$PROJECT_DIR/.venv/bin/activate"

echo "=== Setting up dev environment for Triton and Mojo ==="
uv pip install -e .

echo "=== Dev environment setup complete ==="
