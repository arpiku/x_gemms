# x_gemms - Project Context & Module Documentation

## Directory Structure

```
x_gemms/
├── README.md              # Project overview
├── CONTEXT.md             # This file
├── requirements.txt      # Python dependencies
├── Makefile              # Build system
├── CMakeLists.txt        # CMake configuration
│
├── pybench/              # Python benchmark orchestration
│   ├── __init__.py
│   ├── runner.py        # Main benchmark runner
│   ├── plotter.py       # Visualization
│   ├── utils.py         # Shared utilities
│   └── results/         # Output directory
│
├── cpp/                  # C++ CPU benchmarks
│   ├── CMakeLists.txt
│   ├── main.cpp
│   ├── basic/           # Basic algorithms
│   │   ├── naive.cpp
│   │   ├── blocked.cpp
│   │   └── cache_aware.cpp
│   ├── simd/            # SIMD optimized
│   │   ├── sse.cpp
│   │   ├── avx2.cpp
│   │   └── avx512.cpp
│   └── parallel/        # Threading
│       ├── openmp.cpp
│       └── std_thread.cpp
│
├── gpu/                  # GPU (CUDA) benchmarks
│   ├── CMakeLists.txt
│   ├── main.cu
│   ├── basic/           # Naive CUDA
│   ├── cublas/          # cuBLAS reference
│   ├── cutlass/         # CUTLASS implementations
│   └── custom/          # Research algorithms (placeholder)
│
├── tpu/                  # TPU benchmarks
│   ├── __init__.py
│   ├── jax_tpu.py       # JAX/XLA TPU
│   ├── pytorch_tpu.py   # PyTorch XLA
│   └── reference.py     # Baseline implementations
│
└── tests/
    ├── test_all.py      # Main orchestrator
    ├── test_basic.py
    ├── test_gpu.py
    └── test_tpu.py
```

## Module Details

### pybench/ - Python Benchmark Orchestration

| File | Description |
|------|-------------|
| `runner.py` | Main benchmark runner - executes all module benchmarks, collects timing, computes GFLOPS |
| `plotter.py` | Generates performance comparison graphs from CSV results |
| `utils.py` | Matrix generation, timing helpers, result formatting |

### cpp/ - C++ CPU Implementations

| Module | Description |
|--------|-------------|
| `basic/naive.cpp` | Basic triple-loop GEMM, O(n³) |
| `basic/blocked.cpp` | Blocked matrix multiplication for cache efficiency |
| `basic/cache_aware.cpp` | Cache-aware tiling optimization |
| `simd/sse.cpp` | SSE 128-bit vectorization |
| `simd/avx2.cpp` | AVX2 256-bit vectorization |
| `simd/avx512.cpp` | AVX512 512-bit vectorization |
| `parallel/openmp.cpp` | OpenMP parallelized GEMM |
| `parallel/std_thread.cpp` | std::thread based parallelization |

**Supported data types**: FP32, FP16, INT8
**Build**: Compiled to shared library `libgemm_bench.so`

### gpu/ - CUDA GPU Implementations

| Module | Description |
|--------|-------------|
| `basic/naive.cu` | Naive CUDA kernel |
| `cublas/cublas_bench.cpp` | cuBLAS reference (sgemm, dgemm, etc.) |
| `cutlass/cutlass_bench.cpp` | CUTLASS optimized kernels |
| `custom/` | Research algorithms (placeholder for future) |

**Supported data types**: FP32, FP16, BF16, INT8, TF32
**Build**: Compiled to `gpu_bench` executable

### tpu/ - TPU Benchmarks

| Module | Description |
|--------|-------------|
| `jax_tpu.py` | JAX XLA TPU backend via `jax.lax.dot` |
| `pytorch_tpu.py` | PyTorch XLA via `torch.matmul` on TPU |
| `reference.py` | NumPy/Numba baseline for comparison |

**Supported data types**: FP32, FP16, BF16, INT8, bfloat16
**Note**: Requires TPU hardware or TPU VM emulation

### tests/ - Test Orchestration

| File | Description |
|------|-------------|
| `test_all.py` | Main orchestrator - runs ALL modules with varying sizes/types |
| `test_basic.py` | Tests C++ basic implementations |
| `test_gpu.py` | Tests GPU implementations |
| `test_tpu.py` | Tests TPU implementations |

## Benchmark Sizes

| Category | Sizes |
|----------|-------|
| Small | 64, 128, 256 |
| Medium | 512, 1024 |
| Large | 2048, 4096, 8192 |
| Sparse | 1024, 2048 (random 90% sparsity) |

## Result Format

CSV output with columns:
- `module`, `algorithm`, `size`, `dtype`, `time_ms`, `gfops`, `bandwidth_gbs`

## Adding New Modules

1. Create implementation in appropriate directory (cpp/, gpu/, tpu/)
2. Add build entry (CMakeLists.txt or setup.py)
3. Add benchmark function to appropriate test file
4. Update `test_all.py` to include new module

## Data Types

| Type | Code | Description |
|------|------|-------------|
| FP32 | `fp32` | 32-bit float (IEEE 754) |
| FP16 | `fp16` | 16-bit float |
| BF16 | `bf16` | Brain float (TPU native) |
| INT8 | `int8` | 8-bit integer |

## Hardware Target

- **CPU**: Intel Core Ultra 265K (Efficiency + Performance cores)
- **GPU**: NVIDIA GPU (via CUDA)
- **TPU**: Google TPU (via JAX/PyTorch XLA)