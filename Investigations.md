# Investigations - x_gemms

This file documents curious observations and performance anomalies that warrant further investigation.

---

## 1. C++ OpenMP Performance Anomaly

**Date**: April 2025

### Observation

Plain OpenMP (`openmp` algorithm) is surprisingly **slower** than single-threaded SIMD implementations (`avx2`, `sse`):

| Algorithm | N=1024 GFLOPS |
|-----------|---------------|
| avx2 | 21.2 |
| openmp | 13.2 |
| **openmp_blocked** | **155.5** |

### Analysis

- `openmp` uses naive triple-loop with OpenMP parallelization
- `openmp_blocked` uses blocked matrix multiplication with OpenMP
- The blocked version achieves ~10x speedup over plain OpenMP

### Why?

Plain OpenMP on naive algorithm likely suffers from:
1. **Cache thrashing**: Each thread processes random memory locations
2. **False sharing**: No cache-locality optimization
3. **Memory bandwidth bottleneck**: Threads compete for memory access

### TODO

- [ ] Profile OpenMP with `OMP_PROFILER` or `perf`
- [ ] Test with different thread counts
- [ ] Compare with cache-blocking only (no OpenMP)
- [ ] Determine if this is a threading or algorithm issue

---

## 2. Sparse Matrix GPU Performance

**Date**: April 2025

### Observation

At 90% sparsity (10% non-zero), GPU sparse CSR format is **~12x slower** than dense:

| Format | N=1024 GFLOPS |
|--------|---------------|
| dense (GPU) | 18,742 |
| csr (GPU) | 1,524 |

### Why?

PyTorch sparse tensor operations have significant overhead:
1. Sparse tensor creation/formatting overhead
2. Beta-stage implementation not fully optimized
3. Memory access pattern less efficient than dense for this sparsity

### TODO

- [ ] Test higher sparsity levels (95%, 99%)
- [ ] Find crossover point where sparse becomes faster
- [ ] Compare CPU sparse vs GPU sparse
- [ ] Profile with NSight to identify bottleneck

---

## 3. Numba Performance Degradation

**Date**: April 2025

### Observation

Numba JIT shows significant performance degradation with larger matrix sizes:

| N | Numba GFLOPS |
|---|--------------|
| 64 | 7.5 |
| 512 | 3.3 |
| 1024 | 1.0 |

This is **worse** than NumPy at larger sizes, which defeats the purpose of JIT compilation.

### Why?

- Numba may not be utilizing cache effectively
- Memory allocation overhead increases with size
- Single-threaded naive algorithm doesn't scale

### TODO

- [ ] Test with different Numba settings (parallel=True)
- [ ] Compare with blocked/optimized Numba implementation
- [ ] Profile with memory/timing tools

---

## 4. Tensor Core Variance

**Date**: April 2025

### Observation

Tensor Core results show some variance between runs:

- fp16 N=1024: ~47,500 GFLOPS ± 500
- bf16 N=1024: ~47,300 GFLOPS ± 500

### TODO

- [ ] Run more iterations to establish stable baseline
- [ ] Check if variance is due to thermal throttling
- [ ] Compare with cuBLAS directly

---

## 5. C++ vs GPU Crossover Point

**Date**: April 2025

### Observation

At what size does GPU become faster than multi-threaded C++?

| N | C++ (openmp_blocked) | GPU | Tensor Core |
|---|---------------------|-----|-------------|
| 64 | 16.2 GFLOPS | 9.1 GFLOPS | 55.8 GFLOPS |
| 512 | 143.2 GFLOPS | 615.5 GFLOPS | 16,818 GFLOPS |
| 1024 | 155.5 GFLOPS | 2,000 GFLOPS | 47,500 GFLOPS |

GPU is faster even at smallest size tested.

### TODO

- [ ] Test very small sizes (N=8, 16, 32)
- [ ] Test N=2048+ for CPU scaling
- [ ] Profile GPU kernel launch overhead

---

## 6. Thread Scaling

**Date**: April 2025

### Observation

OpenMP blocked scales with thread count but not linearly:

| Threads | N=1024 GFLOPS |
|---------|---------------|
| 1 | ~25 |
| 4 | ~80 |
| 8 | ~155 |
| 16 | ? |

### TODO

- [ ] Test with MAX_CPU_THREADS (all cores)
- [ ] Compare OpenMP vs std::thread scaling
- [ ] Investigate NUMA effects

---

*Last Updated: April 2025*
