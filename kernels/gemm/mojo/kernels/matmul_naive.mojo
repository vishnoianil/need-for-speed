"""Naive GEMM custom op — 1 thread per output element, global memory reads."""

import compiler
from gpu import block_dim, block_idx, thread_idx
from runtime.asyncrt import DeviceContextPtr
from tensor import InputTensor, ManagedTensorSlice, OutputTensor
from utils.index import IndexList
from math import ceildiv


@compiler.register("matmul_naive")
struct MatmulNaive:
    @staticmethod
    fn execute[
        target: StaticString,
    ](
        output: OutputTensor[rank=2],
        a: InputTensor[dtype=output.dtype, rank=output.rank],
        b: InputTensor[dtype=output.dtype, rank=output.rank],
        ctx: DeviceContextPtr,
    ) raises:
        var M = a.dim_size(0)
        var K = a.dim_size(1)
        var N = b.dim_size(1)

        @parameter
        if target == "gpu":
            var gpu_ctx = ctx.get_device_context()

            @parameter
            fn naive_gemm_kernel(M_val: Int, N_val: Int, K_val: Int):
                row = block_idx.y * block_dim.y + thread_idx.y
                col = block_idx.x * block_dim.x + thread_idx.x

                if row < UInt(M_val) and col < UInt(N_val):
                    var acc: Float32 = 0
                    for k in range(K_val):
                        acc += a[Int(row), k].cast[DType.float32]() * b[k, Int(col)].cast[DType.float32]()
                    output[Int(row), Int(col)] = acc.cast[output.dtype]()

            comptime BLOCK_SIZE = 16
            var grid_x = ceildiv(N, BLOCK_SIZE)
            var grid_y = ceildiv(M, BLOCK_SIZE)

            gpu_ctx.enqueue_function_experimental[naive_gemm_kernel](
                M, N, K,
                grid_dim=(grid_x, grid_y),
                block_dim=(BLOCK_SIZE, BLOCK_SIZE),
            )
        else:
            print("Unsupported target: %s\n", target)
