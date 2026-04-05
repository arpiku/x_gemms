# x_gemms - GEMM Benchmarking Framework

## 1. Project Overview

A comprehensive benchmarking framework for General Matrix Multiplication (GEMM) operations across multiple hardware platforms, data types, and algorithmic implementations. The project is designed to study and compare various matrix multiplication algorithms, with a focus on understanding the trade-offs between different approaches.

### Primary Research Goal
**Strassen's Matrix Multiplication Algorithm** - The project is designed to benchmark and understand the benefits/drawbacks of Strassen's algorithm compared to naive and optimized GEMM implementations.

---

## 2. Directory Structure

```
x_gemms/
├── config.py              # Central configuration
├── requirements.txt       # Python dependencies
├── Makefile              # Top-level build (calls src/cpp and src/cuda)
├── CMakeLists.txt        # CMake configuration
├── .gitignore           # Git ignore rules
├── README.md            # Project overview
├── CONTEXT.md           # This file
│
├── src/                  # Algorithm implementations
│   ├── python/           # Python implementations (CPU)
│   │   ├── __init__.py
│   │   ├── numpy.py     # NumPy baseline
│   │   ├── numba.py     # Numba JIT
│   │   └── sparse.py    # Sparse matrix (CSR/COO)
│   ├── cpp/              # C++ CPU implementations
│   │   ├── Makefile
│   │   ├── main.cpp
│   │   ├── basic/        # naive, blocked, cache_aware
│   │   ├── simd/         # sse, avx2, avx512
│   │   └── parallel/    # openmp, std_thread
│   ├── cuda/             # CUDA GPU implementations
│   │   ├── Makefile
│   │   ├── main.cu
│   │   ├── basic/        # naive kernel
│   │   ├── cublas/       # cuBLAS reference
│   │   └── cutlass/      # CUTLASS reference
│   └── dormant/           # Dormant implementations
│       ├── __init__.py
│       ├── jax_tpu.py   # [TODO] Google TPU (requires cloud)
│       └── pytorch_tpu.py # [TODO] PyTorch XLA TPU (requires cloud)
│
├── bench/                 # Benchmarking utilities
│   ├── __init__.py
│   ├── __main__.py       # CLI entry point
│   ├── runner.py         # Main benchmark orchestrator
│   ├── plotter.py        # Visualization (plots)
│   ├── utils.py          # Matrix generation, timing helpers
│   ├── profiler.py       # GPU profiling (NSight)
│   └── memory_tracker.py # CPU + GPU memory tracking
│
├── tests/                # Test scripts
│   ├── test_all.py      # Main orchestrator
│   ├── test_basic.py    # C++ tests
│   ├── test_gpu.py      # GPU tests
│   └── test_tpu.py      # Tensor Core tests
│
└── results/             # Output directory
    ├── benchmarks.csv
    ├── memory_tracking.csv
    └── *.png            # Generated plots
```

---

## 3. Module Documentation

### 3.1 src/python/ - Python Implementations

| File | Description | Algorithm |
|------|-------------|-----------|
| `numpy.py` | NumPy baseline | `A @ B` |
| `numba.py` | Numba JIT-accelerated | Triple-loop |
| `sparse.py` | Sparse matrix (CSR/COO) | scipy.sparse + PyTorch |

### 3.2 src/cpp/ - C++ CPU Implementations

#### basic/
| File | Algorithm | Complexity |
|------|-----------|------------|
| `naive.cpp` | Triple-loop | O(n³) |
| `blocked.cpp` | Blocked | O(n³) |
| `cache_aware.cpp` | Cache-aware | O(n³) |

#### simd/
| File | Vector Width |
|------|--------------|
| `sse.cpp` | 128-bit (4x fp32) |
| `avx2.cpp` | 256-bit (8x fp32) |
| `avx512.cpp` | 512-bit (16x fp32) |

#### parallel/
| File | Parallelization |
|------|-----------------|
| `openmp.cpp` | OpenMP pragmas |
| `std_thread.cpp` | std::thread |

### 3.3 src/cuda/ - CUDA GPU Implementations

| File | Description |
|------|-------------|
| `main.cu` | CUDA entry point |
| `basic/naive.cu` | Naive CUDA kernel |
| `cublas/cublas_bench.cpp` | cuBLAS reference |
| `cutlass/cutlass_bench.cpp` | CUTLASS reference |

### 3.4 src/cuda/tensor_core.py - Tensor Core

| Function | Description |
|----------|-------------|
| `benchmark_tensor_core()` | fp16/bf16 Tensor Core via PyTorch |
| `run_tensor_core_benchmarks()` | Run multiple sizes/types |

### 3.5 bench/ - Benchmarking Utilities

| File | Description |
|------|-------------|
| `runner.py` | Main orchestrator, GFLOPS calculation |
| `plotter.py` | Visualization (GFLOPS, timing, bandwidth plots) |
| `utils.py` | Matrix generation, timing helpers |
| `profiler.py` | NSight Compute integration |
| `memory_tracker.py` | Memory footprint tracking |
| `__main__.py` | CLI entry point |

---

## 4. Benchmark Configuration

### Matrix Sizes (config.py)

| Category | Sizes |
|----------|-------|
| Quick | 64, 128, 256, 512, 1024 |
| Medium | 64, 128, 256, 512, 1024, 2048, 4096, 8192 |
| All | Up to 20000 |

### Data Types

| Type | Python | C++ | CUDA |
|------|--------|-----|------|
| FP32 | ✓ | ✓ | ✓ |
| FP16 | ✓ | - | ✓ |
| BF16 | ✓ | - | ✓ |
| INT8 | ✓ | ✓ | ✓ |

---

## 5. Running Benchmarks

### CLI (Recommended)

```bash
# Build all
make

# Run benchmarks
py_env/bin/python -m bench run --quick
py_env/bin/python -m bench run --medium
py_env/bin/python -m bench run --large

# Generate plots
py_env/bin/python -m bench plot

# Using Makefile
make test          # Build and run quick
make test-medium   # Build and run medium
make test-large    # Build and run large
```

### Direct test scripts

```bash
py_env/bin/python tests/test_all.py --quick
py_env/bin/python tests/test_basic.py
py_env/bin/python tests/test_tpu.py
```

---

## 6. Hardware Configuration

| Component | Specification |
|-----------|---------------|
| **CPU** | Intel Core Ultra 7 265K |
| **GPU** | NVIDIA RTX 5070 (Blackwell, sm_120) |
| **GPU Memory** | 12 GB GDDR7 |
| **System RAM** | 64 GB DDR5 |
| **CUDA Version** | 13.1 |

---

## 7. Future Work: Strassen's Algorithm (TODO)

### 7.1 Overview

Strassen's Matrix Multiplication (1969) reduces multiplications from 8 to 7 for 2x2 blocks, achieving O(n^2.807) instead of O(n³).

### 7.2 Implementation Location

Create: `src/python/strassen.py` or `src/cuda/strassen.cu`

### 7.3 Key Study Questions

1. **Crossover point**: At what size does Strassen become faster?
2. **Memory trade-off**: Is speedup worth 30% memory increase?
3. **Accuracy**: Numerical precision for fp16/bf16?
4. **Cache behavior**: How does recursive splitting affect cache?

### 7.4 Benchmarking Requirements

| Metric | Current Support |
|--------|-----------------|
| GFLOPS | ✓ Full |
| Timing | ✓ Linear scale |
| Memory | ✓ memory_tracker.py |
| Bandwidth | ✓ plotter.py |
| Accuracy | [TODO] Need implementation |
| Cache | [TODO] Need NSight |

---

*Last Updated: April 2026*
