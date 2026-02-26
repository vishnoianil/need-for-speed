"""Python wrapper for Mojo GEMM custom ops loaded via
max.torch.CustomOpLibrary."""

from pathlib import Path

import torch
from max.torch import CustomOpLibrary

# Load Mojo custom ops from the kernels directory
_kernels_dir = Path(__file__).parent / "kernels"
_ops = CustomOpLibrary(_kernels_dir)


@torch.compile
def matmul_tiled(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Tiled GEMM using Mojo custom op (shared memory, FP32 accumulation).

    Args:
        a: (M, K) tensor, float16, on CUDA
        b: (K, N) tensor, float16, on CUDA

    Returns:
        c: (M, N) tensor, float16, on CUDA
    """
    M, K = a.shape
    N = b.shape[1]
    """ Mojo doesn't support any data type that is less than 4 bytes,
    so tiled matmul will fail for bfloat16/float16. """
    output = torch.empty((M, N), device=a.device, dtype=a.dtype)
    _ops.matmul_tiled(output, a, b)
    return output


@torch.compile
def matmul_naive(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Naive GEMM using Mojo custom op (1 thread per element, global memory).

    Args:
        a: (M, K) tensor, float16, on CUDA
        b: (K, N) tensor, float16, on CUDA

    Returns:
        c: (M, N) tensor, float16, on CUDA
    """
    M, K = a.shape
    N = b.shape[1]
    output = torch.empty((M, N), device=a.device, dtype=a.dtype)
    _ops.matmul_naive(output, a, b)
    return output


def main():
    M = K = N = 64
    DTYPE = torch.float32
    torch.manual_seed(42)
    a = torch.randn(M, K, device="cuda", dtype=DTYPE)
    b = torch.randn(K, N, device="cuda", dtype=DTYPE)
    ref = torch.zeros(M, N, device="cuda", dtype=DTYPE)
    torch.matmul(a, b, out=ref)

    print(a.element_size(), a.dtype)
    print(b.element_size(), b.dtype)
    print(ref.element_size(), ref.dtype)

    # Matmul using Mojo Naive kernel
    mn = matmul_naive(a, b)
    try:
        torch.testing.assert_close(ref, mn, rtol=1e-2, atol=1e-2)
    except Exception as e:
        print(f"Mojo naive kernel output is not same as torch output {e}")
        return

    # Matmul using Mojo tiled kernel
    mt = matmul_tiled(a, b)
    try:
        torch.testing.assert_close(ref, mt, rtol=1e-2, atol=1e-2)
    except Exception as e:
        print(f"Mojo tile kernel output is not same as torch output. {e}")
        return

    print("Both mojo tile and naive kernel ran successfully and results are correct")


if __name__ == "__main__":
    main()
