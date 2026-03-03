"""Unified benchmark: Triton vs Mojo (tiled) vs cuBLAS GEMM on A100.

Measures median/best kernel time and computes TFLOPS for square fp32/fp16/bfp16 matrices.
"""

import json
from datetime import datetime
from pathlib import Path

import torch
import triton

from kernels.gemm.triton.wrapper import matmul_tiled as triton_matmul_tiled


# Try importing Mojo wrapper — may not be available if MAX SDK not installed
try:
    from kernels.gemm.mojo.wrapper import matmul_tiled as mojo_matmul_tiled

    HAS_MOJO = True
except ImportError:
    HAS_MOJO = False
    print("Warning: Mojo GEMM not available. Skipping Mojo benchmarks.")


MATRIX_SIZES = [1024, 2048, 4096]
WARMUP = 50
REP = 200
DTYPE = torch.float32


def cublas_matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """cuBLAS GEMM via torch.matmul (reference baseline)."""
    return torch.matmul(a, b)


def compute_tflops(M: int, N: int, K: int, time_ms: float) -> float:
    """Compute TFLOPS for a GEMM of shape (M, K) x (K, N)."""
    flops = 2.0 * M * N * K  # multiply-accumulate = 2 FLOPs per element
    return flops / (time_ms * 1e-3) / 1e12


def benchmark_fn(fn, a, b):
    """Benchmark a matmul function using triton.testing.do_bench.

    Returns:
        (median_ms, best_ms)
    """
    ms_values = triton.testing.do_bench(
        lambda: fn(a, b), warmup=WARMUP, rep=REP, return_mode="all"
    )
    median_ms = ms_values[0]  # median
    best_ms = ms_values[1]  # min
    return median_ms, best_ms


def run_benchmarks():
    """Run all benchmarks and return results."""
    results = []

    for size in MATRIX_SIZES:
        M = K = N = size
        torch.manual_seed(42)
        a = torch.randn(M, K, device="cuda", dtype=DTYPE)
        b = torch.randn(K, N, device="cuda", dtype=DTYPE)

        # cuBLAS (baseline)
        cublas_median, cublas_best = benchmark_fn(cublas_matmul, a, b)
        cublas_tflops = compute_tflops(M, N, K, cublas_median)

        row = {
            "size": f"{M}x{K}x{N}",
            "cublas_median_ms": round(cublas_median, 4),
            "cublas_best_ms": round(cublas_best, 4),
            "cublas_tflops": round(cublas_tflops, 2),
        }

        # Triton
        triton_median, triton_best = benchmark_fn(triton_matmul_tiled, a, b)
        triton_tflops = compute_tflops(M, N, K, triton_median)
        row["triton_median_ms"] = round(triton_median, 4)
        row["triton_best_ms"] = round(triton_best, 4)
        row["triton_tflops"] = round(triton_tflops, 2)
        row["triton_pct_cublas"] = round(triton_tflops / cublas_tflops * 100, 1)

        # Mojo (tiled)
        if HAS_MOJO:
            mojo_median, mojo_best = benchmark_fn(mojo_matmul_tiled, a, b)
            mojo_tflops = compute_tflops(M, N, K, mojo_median)
            row["mojo_median_ms"] = round(mojo_median, 4)
            row["mojo_best_ms"] = round(mojo_best, 4)
            row["mojo_tflops"] = round(mojo_tflops, 2)
            row["mojo_pct_cublas"] = round(mojo_tflops / cublas_tflops * 100, 1)

        results.append(row)

    return results


def print_table(results):
    """Print a formatted comparison table."""
    from tabulate import tabulate

    headers = [
        "Size",
        "cuBLAS (ms)",
        "cuBLAS TFLOPS",
        "Triton (ms)",
        "Triton TFLOPS",
        "Triton %cuBLAS",
    ]

    if HAS_MOJO:
        headers += ["Mojo (ms)", "Mojo TFLOPS", "Mojo %cuBLAS"]

    rows = []
    for r in results:
        row = [
            r["size"],
            r["cublas_median_ms"],
            r["cublas_tflops"],
            r["triton_median_ms"],
            r["triton_tflops"],
            f"{r['triton_pct_cublas']}%",
        ]
        if HAS_MOJO:
            row += [
                r["mojo_median_ms"],
                r["mojo_tflops"],
                f"{r['mojo_pct_cublas']}%",
            ]
        rows.append(row)

    print("\n" + "=" * 80)
    print(f"GEMM Benchmark — {DTYPE} on A100")
    print(f"Warmup: {WARMUP} | Repetitions: {REP}")
    print("=" * 80)
    print(tabulate(rows, headers=headers, tablefmt="grid"))
    print()


def save_results(results):
    """Save benchmark results as JSON."""
    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    gpu_name = torch.cuda.get_device_name(0).replace(" ", "_")
    filename = f"benchmark_{gpu_name}_{timestamp}.json"

    output = {
        "gpu": torch.cuda.get_device_name(0),
        "dtype": str(DTYPE),
        "warmup": WARMUP,
        "rep": REP,
        "results": results,
    }

    filepath = results_dir / filename
    with open(filepath, "w") as f:
        json.dump(output, f, indent=2)

    print(f"Results saved to {filepath}")


if __name__ == "__main__":
    assert torch.cuda.is_available(), "CUDA is required to run benchmarks"
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"PyTorch: {torch.__version__}")
    print(f"Triton: {triton.__version__}")

    results = run_benchmarks()
    print_table(results)
    save_results(results)
