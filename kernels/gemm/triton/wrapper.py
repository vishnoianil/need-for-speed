import torch
import triton
from torch.utils._triton import has_triton

from kernels.gemm.triton.kernels.matmul import matmul_kernel


@torch.compile(fullgraph=True)
def matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Compute matrix multiplication C = A @ B using the Triton GEMM kernel.

    Args:
        a: (M, K) tensor, float16/32/bfloat16, on CUDA
        b: (K, N) tensor, float16/32/bfloat16, on CUDA

    Returns:
        c: (M, N) tensor, float16/32/bfloat16, on CUDA
    """
    assert a.device.type == "cuda", "Input tensor 'a' must be on CUDA"
    assert b.device.type == "cuda", "Input tensor 'b' must be on CUDA"
    assert a.shape[1] == b.shape[0], (
        f"Incompatible dimensions: a is {a.shape}, b is {b.shape}"
    )

    # Ensure contiguous layout
    a = a.contiguous()
    b = b.contiguous()

    M, K = a.shape
    K, N = b.shape

    c = torch.empty((M, N), device=a.device, dtype=a.dtype)

    grid = lambda META: (  # noqa: E731
        triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]),
    )

    matmul_kernel[grid](
        a,
        b,
        c,
        M,
        N,
        K,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        c.stride(0),
        c.stride(1),
    )

    return c


def main():
    if not has_triton():
        print("Triton is not available. Please install Triton to run this code.")
        return

    # Example usage
    M, K, N = 128, 128, 128
    a = torch.randn((M, K), dtype=torch.bfloat16, device="cuda")
    b = torch.randn((K, N), dtype=torch.bfloat16, device="cuda")
    c = matmul(a, b)
    print("Output shape:", c.shape)
    print("Output:", c)


if __name__ == "__main__":
    main()
