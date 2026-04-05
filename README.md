# x_gemms - GEMM Benchmarking Framework

A comprehensive benchmarking framework for General Matrix Multiplication (GEMM) operations across multiple hardware platforms and data types.

## Overview

This project benchmarks various GEMM implementations ranging from naive algorithms to research-level optimizations across:
- **CPU**: Python (NumPy, Numba) and C++ with SIMD (SSE, AVX2, AVX512) and parallelization (OpenMP, threads)
- **GPU**: CUDA with cuBLAS and Tensor Cores

## Quick Start

```bash
# Install Python dependencies
pip install -r requirements.txt

# Build all (C++ and CUDA)
make

# Run benchmarks using CLI
py_env/bin/python -m bench run --quick
py_env/bin/python -m bench run --medium
py_env/bin/python -m bench run --large

# Or run directly via test script
py_env/bin/python tests/test_all.py --quick
```

## CLI Usage

```bash
# Run benchmarks
py_env/bin/python -m bench run --quick       # Sizes 64-512
py_env/bin/python -m bench run --medium      # Sizes up to 2048
py_env/bin/python -m bench run --large       # Sizes up to 20000

# Control CPU threads
py_env/bin/python -m bench run --quick --threads 4

# Force run all sizes (dangerous - may hang!)
py_env/bin/python -m bench run --medium --force

# Run without certain modules
py_env/bin/python -m bench run --quick --no-cpp
py_env/bin/python -m bench run --quick --no-gpu
py_env/bin/python -m bench run --quick --no-cutlass
py_env/bin/python -m bench run --quick --no-sparse

# Generate plots from results
py_env/bin/python -m bench plot

# Using Makefile
make test          # Build and run quick benchmarks
make test-medium   # Build and run medium benchmarks
make test-large    # Build and run large benchmarks
```

## Project Structure

```
x_gemms/
├── config.py              # Central configuration
├── requirements.txt       # Python dependencies
├── Makefile              # Top-level build (calls src/cpp and src/cuda)
├── README.md / CONTEXT.md
│
├── src/                  # Algorithm implementations
│   ├── python/           # Python implementations
│   │   ├── numpy.py     # NumPy baseline
│   │   ├── numba.py     # Numba JIT
│   │   └── sparse.py    # Sparse matrix (CPU/GPU)
│   ├── cpp/             # C++ CPU implementations
│   │   ├── Makefile
│   │   ├── main.cpp
│   │   ├── basic/       # naive, blocked, cache_aware
│   │   ├── simd/        # sse, avx2, avx512
│   │   └── parallel/    # openmp, std_thread
│   ├── cuda/            # CUDA GPU implementations
│   │   ├── Makefile
│   │   ├── main.cu
│   │   ├── basic/       # naive kernel
│   │   ├── cublas/      # cuBLAS reference
│   │   └── cutlass/     # CUTLASS reference
│   └── dormant/         # Dormant implementations (Google TPU)
│
├── bench/                # Benchmarking utilities
│   ├── runner.py        # Main orchestrator
│   ├── plotter.py       # Visualization
│   ├── utils.py        # Matrix generation, timing
│   ├── profiler.py     # GPU profiling (NSight)
│   ├── memory_tracker.py
│   └── __main__.py     # CLI entry point
│
├── tests/               # Test scripts
│   ├── test_all.py     # Main orchestrator
│   ├── test_basic.py   # C++ tests
│   ├── test_gpu.py     # GPU tests
│   └── test_tpu.py     # Tensor Core tests
│
└── results/             # Output directory
    ├── benchmarks.csv
    ├── memory_tracking.csv
    └── *.png           # Generated plots
```

## Benchmark Test Matrix

- **Sizes**: 64, 128, 256, 512, 1024, 2048, 4096, 8192 (+ sparse variants)
- **Data types**: FP32, FP16, BF16, INT8
- **Metrics**: GFLOPS, latency (ms), memory bandwidth (GB/s)

## Benchmark Naming Convention

Results use the format: `(Device)-(Implementation)-(Library)-(Language)`

| Tag | Description |
|-----|-------------|
| `CPU-Naive-Cpp` | C++ triple-loop naive |
| `CPU-Naive-NumPy-Python` | NumPy @ operator |
| `CPU-Naive-Numba-Python` | Numba JIT |
| `CPU-SIMD-SSE-Cpp` | C++ SSE 128-bit |
| `CPU-SIMD-AVX2-Cpp` | C++ AVX2 256-bit |
| `CPU-SIMD-AVX512-Cpp` | C++ AVX512 512-bit |
| `CPU-Blocked-Cpp` | C++ cache-blocked |
| `CPU-Parallel-OpenMP-Cpp` | OpenMP parallel (naive) |
| `CPU-Parallel-OpenMP-Cpp-Blocked` | OpenMP parallel (blocked) |
| `CPU-Parallel-StdThread-Cpp` | std::thread parallel |
| `GPU-CUDA-Cpp` | CUDA naive kernel |
| `GPU-CUDA-cuBLAS-Cpp` | cuBLAS |
| `GPU-CUDA-CUTLASS-Cpp` | CUTLASS |
| `GPU-TensorCore-PyTorch-fp16-Python` | PyTorch Tensor Core fp16 |
| `GPU-TensorCore-PyTorch-bf16-Python` | PyTorch Tensor Core bf16 |
| `CPU-Sparse-SciPy-Python` | SciPy sparse CSR |
| `GPU-Sparse-PyTorch-Python` | PyTorch sparse |

### Size Limits

| Category | Sizes | Single-thread | Parallel/GPU |
|----------|-------|---------------|--------------|
| Quick | 64, 128, 256, 512 | ✓ | ✓ |
| Medium | 64-2048 | ✓ | ✓ |
| Large | up to 20000 | ✗ (skipped) | ✓ up to 8192 |

Use `--force` flag to override size limits (dangerous - may hang!).

## Output

Results are saved to `results/` as CSV files. Use `bench/plotter.py` or `python -m bench plot` to generate performance graphs.

## Sparse Matrix Behavior (TODO: Study)

**Observations from GPU sparse benchmarks:**
- At 90% sparsity (10% non-zero), GPU sparse CSR format is ~12x slower than dense
- This is because the overhead of sparse tensor format conversion exceeds the benefit
- Dense operations on GPU can leverage massive parallelism more efficiently

**Technical Notes:**
- PyTorch sparse CSR tensor support is in "beta state"
- `torch.sparse.check_sparse_tensor_invariants.enable()` enables runtime checks at slight performance cost

**When to study:**
- Determine the crossover point where sparse becomes faster than dense
- Investigate higher sparsity levels (95%, 99%)
- Compare CPU vs GPU sparse performance
