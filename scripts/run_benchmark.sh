#!/usr/bin/env bash
# Run all tests and benchmarks
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_DIR"

# Activate virtual environment
source "$PROJECT_DIR/.venv/bin/activate"


run_benchmarks() {
    echo ""
    echo "========================================"
    echo "  Running Unified Benchmark"
    echo "========================================"
    python benchmarks/benchmark.py
}

# Default to --all if no argument is provided
run_benchmarks

echo ""
echo "========================================"
echo "  All done!"
echo "========================================"
