# x_gemms - GEMM Benchmarking Framework

A comprehensive benchmarking framework for General Matrix Multiplication (GEMM) operations across multiple hardware platforms and data types.

## Overview

This project benchmarks various GEMM implementations ranging from naive algorithms to research-level optimizations across:
- **CPU**: C++ implementations with SIMD (SSE, AVX2, AVX512) and parallelization (OpenMP, threads)
- **GPU**: CUDA with cuBLAS and CUTLASS
- **TPU**: JAX/PyTorch XLA backends

## Quick Start

```bash
# Install Python dependencies
pip install -r requirements.txt

# Build C++ modules
make

# Run all benchmarks
python tests/test_all.py

# Run specific module benchmarks
python tests/test_basic.py
python tests/test_gpu.py
python tests/test_tpu.py
```

## Benchmark Test Matrix

- **Sizes**: 64, 128, 256, 512, 1024, 2048, 4096, 8192 (+ sparse variants)
- **Data types**: FP32, FP16, BF16, INT8
- **Metrics**: GFLOPS, latency (ms), memory bandwidth (GB/s)

## Output

Results are saved to `pybench/results/` as CSV files. Use `pybench/plotter.py` to generate performance graphs.

## Project Structure

See [CONTEXT.md](CONTEXT.md) for detailed module documentation.