"""Correctness tests for the Triton GEMM kernel."""

import torch

from kernels.gemm.triton.wrapper import matmul_tiled


RTOL = 1e-2
ATOL = 1e-2


class TestTritonGEMMCorrectness:
    """Verify Triton kernel output matches torch.matmul within FP16 tolerance."""

    def test_correctness(self, fp_matrices):
        a, b, ref = fp_matrices
        out = matmul_tiled(a, b)
        torch.testing.assert_close(out, ref, rtol=RTOL, atol=ATOL)

    def test_output_dtype(self, fp_matrices):
        a, b, _ = fp_matrices
        out = matmul_tiled(a, b)
        assert out.dtype == a.dtype

    def test_output_shape(self, fp_matrices):
        a, b, _ = fp_matrices
        out = matmul_tiled(a, b)
        M = a.shape[0]
        N = b.shape[1]
        assert out.shape == (M, N)

    def test_identity_multiply(self, identity_matrices):
        eye, mat = identity_matrices
        out = matmul_tiled(eye, mat)
        torch.testing.assert_close(out, mat, rtol=RTOL, atol=ATOL)

    def test_zero_matrix(self, zero_matrix):
        zeros, mat = zero_matrix
        out = matmul_tiled(zeros, mat)
        expected = torch.zeros_like(out)
        torch.testing.assert_close(out, expected, rtol=RTOL, atol=ATOL)

    def test_output_contiguous(self, fp_matrices):
        a, b, _ = fp_matrices
        out = matmul_tiled(a, b)
        assert out.is_contiguous()

    def test_non_contiguous_input(self, fp_matrices):
        """Kernel should handle non-contiguous inputs
        (wrapper makes them contiguous)."""
        a, b, _ = fp_matrices
        a_nc = torch.as_strided(a, a.shape, (1, a.shape[0]))  # col-major stride
        out = matmul_tiled(a_nc, b)
        ref = torch.matmul(a_nc, b)
        torch.testing.assert_close(out, ref, rtol=RTOL, atol=ATOL)
