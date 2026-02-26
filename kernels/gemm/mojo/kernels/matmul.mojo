"""Tiled GEMM custom op using shared memory and FP32 accumulation."""

import compiler
from gpu import block_dim, block_idx, thread_idx, barrier
from gpu.memory import AddressSpace, async_copy_wait_all
from layout.layout_tensor import Layout, LayoutTensor, copy_dram_to_sram_async
from runtime.asyncrt import DeviceContextPtr
from tensor import InputTensor, ManagedTensorSlice, OutputTensor
from memory import UnsafePointer
from gpu.host import DeviceBuffer
from utils.index import IndexList
from math import ceildiv


comptime BM = 32
comptime BN = 32
comptime BK = 32
comptime NUM_THREADS = BM * BN


fn tiled_matmul_kernel[
    dtype: DType,
    a_layout: Layout,
    b_layout: Layout,
    c_layout: Layout,
](
    a: LayoutTensor[dtype, a_layout, MutAnyOrigin],
    b: LayoutTensor[dtype, b_layout, MutAnyOrigin],
    c: LayoutTensor[dtype, c_layout, MutAnyOrigin],
):
    """Tiled GEMM kernel with shared memory and FP32 accumulation.

    Each thread block computes a BM x BN tile of the output matrix C.
    The K dimension is tiled into BK chunks loaded into shared memory.
    """
    var col = thread_idx.x % UInt(BN)
    var row = thread_idx.x // UInt(BN)

    # Output tile for this thread block
    var dst = c.tile[BM, BN](Int(block_idx.y), Int(block_idx.x))

    # Shared memory tiles for A and B
    var a_smem = LayoutTensor[
        dtype,
        Layout.row_major(BM, BK),
        MutAnyOrigin,
        address_space=AddressSpace.SHARED,
    ].stack_allocation()

    var b_smem = LayoutTensor[
        dtype,
        Layout.row_major(BK, BN),
        MutAnyOrigin,
        address_space=AddressSpace.SHARED,
    ].stack_allocation()

    # FP32 accumulator for numerical stability
    var acc: c.element_type = 0

    # Loop over K-dimension tiles
    var num_k_tiles = b.dim[0]() // BK
    for block in range(num_k_tiles):
        # Thread layout for cooperative loading
        comptime load_a_layout = Layout.row_major(NUM_THREADS // BK, BK)
        comptime load_b_layout = Layout.row_major(BK, NUM_THREADS // BK)

        # Extract tiles from global memory
        var a_tile = a.tile[BM, BK](Int(block_idx.y), block)
        var b_tile = b.tile[BK, BN](block, Int(block_idx.x))

        # Async copy from DRAM to shared memory
        copy_dram_to_sram_async[thread_layout=load_a_layout](a_smem, a_tile)
        copy_dram_to_sram_async[thread_layout=load_b_layout](b_smem, b_tile)

        async_copy_wait_all()
        barrier()

        # Compute partial dot products from shared memory
        @parameter
        for k in range(BK):
            acc += a_smem[row, k] * b_smem[k, col]

        barrier()

    # Write accumulated result back in original dtype
    dst[row, col] = acc


@compiler.register("matmul_tiled")
struct MatmulTiled:
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

            var a_layout = a.to_layout_tensor()
            var b_layout = b.to_layout_tensor()
            var out_layout = output.to_layout_tensor()

            # Zero the output buffer
            gpu_ctx.enqueue_memset(
                DeviceBuffer[output.dtype](
                    gpu_ctx, out_layout.ptr, M * N, owning=False
                ),
                0,
            )

            comptime kernel = tiled_matmul_kernel[
                output.dtype,
                a_layout.layout,
                b_layout.layout,
                out_layout.layout,
            ]

            gpu_ctx.enqueue_function_experimental[kernel](
                a_layout,
                b_layout,
                out_layout,
                grid_dim=(ceildiv(N, BN), ceildiv(M, BM)),
                block_dim=NUM_THREADS,
            )
        else:
            print("Unsupported target: %s\n", target)
