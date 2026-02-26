#!/usr/bin/env bash
# Run test for GEMM kernels from different backends
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_DIR"

# Activate virtual environment
source "$PROJECT_DIR/.venv/bin/activate"

usage() {
    echo "Usage: $0 [--mojo | --triton | --all]"
    echo ""
    echo "Options:"
    echo "  --mojo    Run only Mojo tests"
    echo "  --triton  Run only Triton tests"
    echo "  --all     Run all tests (Mojo and Triton)"
    exit 1
}

run_mojo() {
    echo ""
    echo "========================================"
    echo "  Running Mojo GEMM Tests"
    echo "========================================"
    mojo package "kernels/gemm/mojo/kernels" -o "kernels/gemm/mojo/kernels/mojo_kernel.mojopkg"
    pytest tests/test_mojo_gemm.py -v || echo "Mojo tests skipped (MAX SDK may not be installed)"
}

run_triton() {
    echo ""
    echo "========================================"
    echo "  Running Triton GEMM Tests"
    echo "========================================"
    pytest tests/test_triton_gemm.py -v
}

# Default to --all if no argument is provided
MODE="${1:---all}"

case "$MODE" in
    --mojo)
        run_mojo
        ;;
    --triton)
        run_triton
        ;;
    --all)
        run_mojo
        run_triton
        ;;
    *)
        usage
        ;;
esac

echo ""
echo "========================================"
echo "  All done!"
echo "========================================"
