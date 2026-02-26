"""Correctness tests for the Mojo GEMM custom ops."""

import torch

from kernels.gemm.mojo.wrapper import matmul, matmul_naive


RTOL = 1e-2
ATOL = 1e-2


class TestMojoTiledGEMMCorrectness:
    """Verify Mojo tiled kernel output matches torch.matmul
    within FP16 tolerance."""

    def test_correctness(self, fp_matrices):
        a, b, ref = fp_matrices
        out = matmul(a, b)
        torch.testing.assert_close(out, ref, rtol=RTOL, atol=ATOL)

    def test_output_dtype(self, fp_matrices):
        a, b, _ = fp_matrices
        out = matmul(a, b)
        assert out.dtype == a.dtype

    def test_output_shape(self, fp_matrices):
        a, b, _ = fp_matrices
        out = matmul(a, b)
        M = a.shape[0]
        N = b.shape[1]
        assert out.shape == (M, N)

    def test_identity_multiply(self, identity_matrices):
        eye, mat = identity_matrices
        out = matmul(eye, mat)
        torch.testing.assert_close(out, mat, rtol=RTOL, atol=ATOL)

    def test_zero_matrix(self, zero_matrix):
        zeros, mat = zero_matrix
        out = matmul(zeros, mat)
        expected = torch.zeros_like(out)
        torch.testing.assert_close(out, expected, rtol=RTOL, atol=ATOL)

    def test_output_contiguous(self, fp_matrices):
        a, b, _ = fp_matrices
        out = matmul(a, b)
        assert out.is_contiguous()


class TestMojoNaiveGEMMCorrectness:
    """Verify Mojo naive kernel output matches torch.matmul
    within FP16 tolerance."""

    def test_correctness(self, fp_matrices):
        a, b, ref = fp_matrices
        out = matmul_naive(a, b)
        torch.testing.assert_close(out, ref, rtol=RTOL, atol=ATOL)

    def test_output_dtype(self, fp_matrices):
        a, b, _ = fp_matrices
        out = matmul_naive(a, b)
        assert out.dtype == a.dtype

    def test_output_shape(self, fp_matrices):
        a, b, _ = fp_matrices
        out = matmul_naive(a, b)
        M = a.shape[0]
        N = b.shape[1]
        assert out.shape == (M, N)

    def test_identity_multiply(self, identity_matrices):
        eye, mat = identity_matrices
        out = matmul_naive(eye, mat)
        torch.testing.assert_close(out, mat, rtol=RTOL, atol=ATOL)

    def test_zero_matrix(self, zero_matrix):
        zeros, mat = zero_matrix
        out = matmul_naive(zeros, mat)
        expected = torch.zeros_like(out)
        torch.testing.assert_close(out, expected, rtol=RTOL, atol=ATOL)
