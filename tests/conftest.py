"""Shared pytest fixtures for GEMM kernel correctness tests."""

import pytest
import torch


MATRIX_SIZES = [64, 128, 256, 1024]

DTYPE = torch.float32


def check_fp16_support():
    if not torch.cuda.is_available():
        return "No CUDA GPU detected."

    # Get the name and compute capability
    major, minor = torch.cuda.get_device_capability()
    gpu_name = torch.cuda.get_device_name()

    # FP16 arithmetic (Tensor Cores) requires Compute Capability 7.0 or higher
    # (Volta, Turing, Ampere, Ada, Hopper architectures)
    supports_fp16 = major >= 7

    print(f"GPU: {gpu_name}")
    print(f"Compute Capability: {major}.{minor}")
    print(
        f"Hardware FP16 Acceleration: \
            {'Supported' if supports_fp16 else 'Not Supported'}"
    )


@pytest.fixture(params=MATRIX_SIZES, ids=lambda s: f"{s}x{s}")
def matrix_size(request):
    """Parametrized matrix size."""
    return request.param


@pytest.fixture
def fp_matrices(matrix_size):
    """Generate random FP16/BF16/FP32 matrices on CUDA with reference result.

    Returns:
        (a, b, ref): where a is (M, K), b is (K, N), ref = torch.matmul(a, b)
    """
    M = K = N = matrix_size
    torch.manual_seed(42)
    a = torch.randn(M, K, device="cuda", dtype=DTYPE)
    b = torch.randn(K, N, device="cuda", dtype=DTYPE)
    ref = torch.zeros(M, N, device="cuda", dtype=DTYPE)
    ref = torch.matmul(a, b, out=ref)
    return a, b, ref


@pytest.fixture
def identity_matrices(matrix_size):
    """Generate an identity matrix and a random matrix for
    identity multiply test.

    Returns:
        (eye, mat): where eye is (N, N) identity, mat is (N, N) random FP16
    """
    N = matrix_size
    torch.manual_seed(42)
    eye = torch.eye(N, device="cuda", dtype=DTYPE)
    mat = torch.randn(N, N, device="cuda", dtype=DTYPE)
    return eye, mat


@pytest.fixture
def zero_matrix(matrix_size):
    """Generate a zero matrix and a random matrix.

    Returns:
        (zeros, mat): where zeros is (N, N) all-zero, mat is (N, N) random FP16
    """
    N = matrix_size
    torch.manual_seed(42)
    zeros = torch.zeros(N, N, device="cuda", dtype=DTYPE)
    mat = torch.randn(N, N, device="cuda", dtype=DTYPE)
    return zeros, mat
