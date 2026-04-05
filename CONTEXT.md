# x_gemms - GEMM Benchmarking Framework

## 1. Project Overview

A comprehensive benchmarking framework for General Matrix Multiplication (GEMM) operations across multiple hardware platforms, data types, and algorithmic implementations. The project is designed to study and compare various matrix multiplication algorithms, with a focus on understanding the trade-offs between different approaches.

### Primary Research Goal
**Strassen's Matrix Multiplication Algorithm** - The project is designed to benchmark and understand the benefits/drawbacks of Strassen's algorithm compared to naive and optimized GEMM implementations.

---

## 2. Directory Structure

```
x_gemms/
├── README.md                    # Project overview
├── CONTEXT.md                   # This file
├── config.py                    # Central configuration
├── requirements.txt             # Python dependencies
├── Makefile                     # C++ build system
├── CMakeLists.txt               # CMake configuration
├── .gitignore                   # Git ignore rules
│
├── pybench/                     # Python benchmark orchestration
│   ├── runner.py               # Main benchmark runner
│   ├── plotter.py              # Visualization (plots)
│   ├── utils.py                # Matrix generation, timing helpers
│   └── results/                # Output directory (CSV + PNG)
│
├── cpp/                         # C++ CPU benchmarks
│   ├── main.cpp                # Entry point
│   ├── Makefile                # Build configuration
│   ├── basic/                  # Basic algorithms
│   │   ├── naive.cpp           # Triple-loop O(n³)
│   │   ├── blocked.cpp         # Blocked matrix multiplication
│   │   └── cache_aware.cpp     # Cache-aware tiling
│   ├── simd/                    # SIMD optimized
│   │   ├── sse.cpp             # SSE 128-bit
│   │   ├── avx2.cpp            # AVX2 256-bit
│   │   └── avx512.cpp          # AVX512 512-bit
│   └── parallel/                # Threading
│       ├── openmp.cpp          # OpenMP parallelization
│       └── std_thread.cpp      # std::thread parallelization
│
├── gpu/                         # CUDA GPU benchmarks
│   ├── main.cu                 # CUDA entry point
│   ├── Makefile               # GPU build configuration
│   ├── basic/                 # Naive CUDA kernel
│   ├── cublas/                # cuBLAS reference
│   ├── cutlass/               # CUTLASS implementations
│   └── custom/                 # Research algorithms (placeholder)
│
├── tpu/                         # Tensor Core / Accelerators
│   ├── __init__.py
│   ├── reference.py            # NumPy/Numba baseline
│   ├── jax_tpu.py              # [TODO] JAX Google TPU (requires cloud)
│   ├── pytorch_tpu.py          # [TODO] PyTorch Google TPU (requires cloud)
│   ├── cutlass_bench.py        # CUTLASS-backed Tensor Core benchmarks
│   ├── sparse.py              # Sparse matrix benchmarks (CSR/COO)
│   ├── memory_tracker.py       # CPU + GPU memory tracking
│   └── gpu_profiler.py         # NSight Compute integration
│
└── tests/
    ├── test_all.py             # Main orchestrator
    ├── test_basic.py
    ├── test_gpu.py
    └── test_tpu.py
```

---

## 3. Module Documentation

### 3.1 pybench/ - Benchmark Orchestration

| File | Description |
|------|-------------|
| `runner.py` | Main benchmark runner - executes all modules, collects timing, computes GFLOPS |
| `plotter.py` | Generates performance comparison graphs (5+ plot types) |
| `utils.py` | Matrix generation, timing helpers, result formatting |

**Study Notes**: This module orchestrates all benchmarks and provides visualization for performance analysis. Key plots include GFLOPS comparison, timing graphs (linear scale), bandwidth analysis, and category comparisons.

---

### 3.2 cpp/ - CPU Implementations

#### 3.2.1 basic/ - Basic Algorithms

| Module | Algorithm | Complexity | Benefits | Study Notes |
|--------|-----------|------------|----------|-------------|
| `naive.cpp` | Triple-loop | O(n³) | Simple, correct baseline | Good for understanding basic multiplication; shows exponential time growth |
| `blocked.cpp` | Blocked | O(n³) | Cache-efficient | Reduces cache misses by processing blocks; ~20-30x faster than naive at N=1024 |
| `cache_aware.cpp` | Cache-aware | O(n³) | Multi-level tiling | Optimized for L1/L2/L3 cache hierarchy; best CPU performance |

#### 3.2.2 simd/ - SIMD Optimized

| Module | Vector Width | Benefits | Study Notes |
|--------|--------------|----------|--------------|
| `sse.cpp` | 128-bit (4x fp32) | 4x parallelism | Limited to small matrices; demonstrates SIMD basics |
| `avx2.cpp` | 256-bit (8x fp32) | 8x parallelism | Good balance of performance and compatibility |
| `avx512.cpp` | 512-bit (16x fp32) | 16x parallelism | Requires AVX512 support; fastest CPU but limited hardware |

#### 3.2.3 parallel/ - Multi-threaded

| Module | Parallelization | Benefits | Study Notes |
|--------|-----------------|----------|--------------|
| `openmp.cpp` | OpenMP pragmas | Easy to implement, portable | Uses all available cores; ~50-150x speedup |
| `std_thread.cpp` | std::thread | Manual control | Similar performance to OpenMP; more code complexity |

**Supported data types**: FP32 (full), FP16 (partial), INT8 (template-based)
**Build**: Compiled to `gemm_bench` executable

---

### 3.3 gpu/ - CUDA GPU Implementations

| Module | Description | Benefits | Study Notes |
|--------|-------------|----------|--------------|
| `basic/naive.cu` | Naive CUDA kernel | Simple baseline | Shows GPU vs CPU speedup; ~1000x faster than CPU naive |
| `cublas/cublas_bench.cpp` | cuBLAS reference | Optimized library | Near-peak performance; comparison benchmark |
| `cutlass/` | CUTLASS kernels | Research-level | [See Section 5 for TODO] |
| `custom/` | Custom algorithms | Placeholder | [See Section 5 for TODO] |

**Supported data types**: FP32, FP16, BF16, INT8, TF32
**Build**: Compiled to `gpu_bench` executable

---

### 3.4 tpu/ - Tensor Core / Accelerators

#### Active Modules (Working on RTX 5070)

| Module | Description | Benefits | Study Notes |
|--------|-------------|----------|--------------|
| `cutlass_bench.py` | Tensor Core via PyTorch | CUTLASS-backed performance | 40,000-50,000 GFLOPS at N=1024; uses fp16/bf16 |
| `sparse.py` | Sparse matrix (CSR/COO) | Memory-efficient | 90% sparsity testing; compares formats |
| `memory_tracker.py` | Memory footprint | Full memory analysis | Tracks GPU + CPU memory; outputs to memory_tracking.csv |
| `gpu_profiler.py` | NSight integration | Detailed profiling | DRAM bandwidth, L2 cache (when NSight available) |
| `reference.py` | NumPy/Numba baseline | CPU reference | 1-7 GFLOPS; slower than optimized |

#### Dormant Modules (Require Google Cloud TPU)

| Module | Status | Requirements |
|--------|--------|---------------|
| `jax_tpu.py` | TODO | Google Cloud TPU or TPU VM |
| `pytorch_tpu.py` | TODO | Google Cloud TPU or TPU VM |

**Note**: These modules require Google Cloud TPU hardware which is not available on the current system. They are kept as reference for future expansion.

---

### 3.5 tests/ - Test Orchestration

| File | Description |
|------|-------------|
| `test_all.py` | Main orchestrator - runs ALL modules |
| `test_basic.py` | Tests C++ implementations |
| `test_gpu.py` | Tests GPU implementations |
| `test_tpu.py` | Tests Tensor Core implementations |

---

## 4. Benchmark Configuration

### Matrix Sizes (config.py)

| Category | Sizes |
|----------|-------|
| Small | 64, 128, 256 |
| Medium | 512, 1024 |
| Large | 2048, 4096, 8192 |
| Very Large | 10000, 15000, 20000 |
| Sparse | 1024, 2048, 4096 |

### Data Types

| Type | CPU (C++) | GPU (CUDA) | Tensor Core | Reference |
|------|-----------|------------|-------------|-----------|
| FP32 | ✓ | ✓ | N/A | ✓ |
| FP16 | N/A | ✓ | ✓ | ✓ |
| BF16 | N/A | ✓ | ✓ | ✓ (fallback to FP32) |
| INT8 | ✓ | ✓ | ✓ | N/A |

---

## 5. Future Work: Strassen's Algorithm (TODO)

### 5.1 Overview

Strassen's Matrix Multiplication (1969) is a divide-and-conquer algorithm that reduces the number of multiplications from 8 to 7 for 2x2 matrix blocks, achieving O(n^2.807) complexity instead of O(n³).

### 5.2 Implementation Requirements

- [ ] Recursive matrix splitting with base case threshold
- [ ] 7 temporary matrix computation (Strassen step)
- [ ] Addition/subtraction optimization
- [ ] Memory footprint tracking (expect ~30% more memory)

### 5.3 Benchmarking Requirements

| Metric | Description | Current Support |
|--------|-------------|-----------------|
| GFLOPS | Compute performance | ✓ Full |
| Timing | Execution time | ✓ Linear scale added |
| Memory | GPU/CPU footprint | ✓ memory_tracker.py |
| Bandwidth | Memory bandwidth | ✓ plotter.py + refs |
| Accuracy | Relative error vs naive | [TODO] Need implementation |
| Cache | L2 cache behavior | [TODO] Need NSight |

### 5.4 Key Study Questions

1. **Crossover point**: At what matrix size does Strassen become faster than naive/blocked?
2. **Memory trade-off**: Is the speedup worth the 30% memory increase?
3. **Accuracy**: How does numerical precision compare for fp16/bf16?
4. **Cache behavior**: How does recursive splitting affect cache utilization?

### 5.5 Implementation Location

Create new module: `tpu/strassen.py`

### 5.6 Related Reading

- Strassen, V. (1969). "Gaussian Elimination is Not Optimal"
- Higham, N. "Accuracy and Stability of Numerical Algorithms"
- Various optimized Strassen implementations (Plasmen, KAP)

---

## 6. Hardware Configuration

### Current System

| Component | Specification |
|-----------|---------------|
| **CPU** | Intel Core Ultra 7 265K |
| **GPU** | NVIDIA RTX 5070 (Blackwell, sm_120) |
| **GPU Memory** | 12 GB GDDR7 |
| **System RAM** | 64 GB DDR5 |
| **CUDA Version** | 13.1 |

### Memory Bandwidth Reference

| Component | Theoretical Bandwidth |
|-----------|----------------------|
| RTX 5070 (GDDR7) | ~500 GB/s |
| CPU DDR5 per channel | ~50 GB/s |
| CPU L1 Cache | ~2 TB/s |
| CPU L2 Cache | ~1 TB/s |
| CPU L3 Cache | ~500 GB/s |

---

## 7. Running Benchmarks

### Quick Test (sizes 64-1024)
```bash
python tests/test_all.py --quick
```

### Full Test with Large Sizes (up to 20000)
```bash
python tests/test_all.py --large
```

### Individual Tests
```bash
# C++ benchmarks
./cpp/gemm_bench 8

# GPU benchmarks  
./gpu/gpu_bench

# Memory tracking
python -c "from tpu.memory_tracker import run_memory_benchmarks; run_memory_benchmarks()"

# GPU Profiling
python -c "from tpu.gpu_profiler import run_gpu_profiling; run_gpu_profiling()"
```

### Output Files (pybench/results/)

| File | Description |
|------|-------------|
| `benchmarks.csv` | Main benchmark results |
| `memory_tracking.csv` | Memory usage metrics |
| `gpu_profiling.csv` | Detailed GPU profiling |
| `gflops_by_size.png` | GFLOPS vs size (log) |
| `timing_linear.png` | Execution time (linear) |
| `bandwidth_by_size.png` | Bandwidth with reference lines |
| `category_comparison.png` | Category comparison |
| `sparse_comparison.png` | Sparse format comparison |
| `tensor_core_comparison.png` | FP16 vs BF16 |

---

## 8. Continuing This Project

### Prerequisites
```bash
# Python environment
source py_env/bin/activate

# Dependencies (already installed)
pip install -r requirements.txt

# C++ build (if modified)
cd cpp && make clean && make

# GPU build (if modified)
cd gpu && make clean && make
```

### Quick Status Check
```bash
# Run quick benchmarks
python tests/test_all.py --quick

# Check outputs
ls pybench/results/
```

### Notes for Future Sessions
- All modules are self-contained
- Configuration centralized in `config.py`
- CONTEXT.md contains complete context
- Git repository tracks all changes

---

*Last Updated: April 2026*
*Project: GEMM Benchmarking for Algorithm Study (Strassen focus)*
