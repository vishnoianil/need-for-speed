# Need for Speed

**Compare Kernels written in Different GPU Programming Languages**

Need for Speed is a benchmarking and comparison framework for GPU kernel implementations. Aim of this project is write kernels in different GPU programming languages, and compare their correctness and performance against PyTorch ops as the baseline. The project currently includes Tiled GEMM (General Matrix Multiplication) kernels implemented in Triton and Mojo, with more languages and kernels planned for the future.


## Available Kernels

The table lists all the kernels and the programming languages they are implemented in. Each kernel is designed to perform the same mathematical operation (GEMM) but may use different optimization techniques and memory access patterns based on the capabilities of the language:

| Language | Kernel | Description |
|----------|--------|-------------|
| **Triton** | Tiled GEMM | Auto-tuned tiled kernel with FP32 accumulation, Tensor Core utilization, and grouped program IDs for L2 cache locality |
| **Mojo** | Tiled GEMM | Shared-memory tiled kernel with async DRAM-to-SRAM copies, barrier synchronization, and FP32 accumulation |
| **Mojo** | Naive GEMM | One-thread-per-element baseline using global memory only |
| **cuBLAS** | `torch.matmul` | Reference baseline via PyTorch's cuBLAS integration |

Kernel source locations:

```
kernels/
└── gemm/
    ├── triton/
    │   ├── wrapper.py                  # Python wrapper
    │   └── kernels/matmul.py           # Triton kernel implementation
    └── mojo/
        ├── wrapper.py                  # Python wrapper
        └── kernels/
            ├── matmul.mojo             # Tiled GEMM kernel
            └── matmul_naive.mojo       # Naive GEMM kernel
```

## How Kernels Are Launched

Each kernel implementation has a Python wrapper (`wrapper.py`) that provides a consistent interface. The wrappers handle:

- Input validation (CUDA device checks, shape compatibility)
- Memory layout enforcement (contiguous tensors)
- Grid/block computation
- Kernel dispatch

Both Triton and Mojo wrappers are decorated with `@torch.compile` to enable PyTorch's JIT compilation and optimization pipeline.

## Running Individual Kernels

Each wrapper includes a `main()` function that runs a small correctness check:

```bash
# Run the Triton kernel
python -m kernels.gemm.triton.wrapper

# Run the Mojo kernels
python -m kernels.gemm.mojo.wrapper
```

## Running Tests

The test suite validates correctness across multiple matrix sizes (64, 128, 256, 1024) using identity matrices, zero matrices, and random inputs, all checked against `torch.matmul`.

```bash
# Run all tests
bash scripts/run_test.sh --all

# Run only Triton tests
bash scripts/run_test.sh --triton

# Run only Mojo tests
bash scripts/run_test.sh --mojo
```

> **Note:** Mojo tests require the Mojo package to be compiled first. The `run_test.sh` script handles this automatically. To compile manually: `mojo package kernels/gemm/mojo/kernels -o kernels/gemm/mojo/kernels/mojo_kernel.mojopkg`


## Running Benchmarks

Benchmarks measure kernel execution time and compute TFLOPS for square matrices at sizes 1024, 2048, and 4096, comparing each kernel against cuBLAS.

```bash
bash scripts/run_benchmark.sh
```

Results are printed as a formatted table and saved as JSON to `benchmarks/results/`.

## Setup

**Prerequisites:** Python >= 3.10, NVIDIA GPU with CUDA support, [uv](https://astral.sh/uv/) package manager.

```bash
# Set up the development environment
bash scripts/setup_dev.sh
```

This creates a virtual environment, installs PyTorch, Triton, the Modular SDK (for Mojo), and all other dependencies.

## Contributing

Contributions are welcome! This project is designed to be extended with various compute kernels written in additional GPU programming languages. If you have experience with any other GPU programming language, we encourage you to contribute a kernel implementation.

To add a new kernel:

1. Create a new directory under `kernels/gemm/<language>/` with a `wrapper.py` that exposes a `matmul` function matching the existing interface.
2. Add corresponding tests in `tests/`.
3. Register the new kernel in `benchmarks/benchmark.py` so it is included in benchmark runs.

Please see the existing Triton and Mojo implementations as reference for the expected structure and conventions.

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
