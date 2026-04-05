"""Reference GEMM implementations for baseline comparison.

Uses NumPy/Numba for CPU baseline when TPU/GPU not available.
"""

import numpy as np
from typing import Optional

try:
    import numba
    from numba import jit
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    jit = lambda **kw: lambda f: f


if NUMBA_AVAILABLE:
    @jit(nopython=True, cache=True)
    def gemm_numba(A: np.ndarray, B: np.ndarray, C: np.ndarray) -> np.ndarray:
        """Numba-accelerated GEMM."""
        N = A.shape[0]
        for i in range(N):
            for j in range(N):
                s = 0.0
                for k in range(N):
                    s += A[i, k] * B[k, j]
                C[i, j] = s
        return C
else:
    def gemm_numba(A: np.ndarray, B: np.ndarray, C: np.ndarray) -> np.ndarray:
        """Fallback to NumPy."""
        return A @ B


def gemm_numpy(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Standard NumPy GEMM."""
    return A @ B


def benchmark_reference(size: int, dtype: str = "fp32", use_numba: bool = True, 
                        warmup: int = 3, iterations: int = 10) -> dict:
    """Benchmark reference CPU implementation."""
    dtype_map = {"fp32": np.float32, "fp16": np.float16, "bf16": np.bfloat16, "int8": np.int8}
    np_dtype = dtype_map.get(dtype, np.float32)

    np.random.seed(42)
    A = np.random.randn(size, size).astype(np_dtype)
    B = np.random.randn(size, size).astype(np_dtype)
    C = np.zeros((size, size), dtype=np_dtype)

    if use_numba and NUMBA_AVAILABLE:
        gemm_func = gemm_numba
    else:
        def wrapper(A, B, C):
            C[:] = A @ B
            return C
        gemm_func = wrapper

    for _ in range(warmup):
        C = gemm_func(A, B, C)

    import time
    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        C = gemm_func(A, B, C)
        end = time.perf_counter()
        times.append((end - start) * 1000)

    time_ms = np.median(times)
    gflops = 2 * size**3 / time_ms / 1e6

    return {
        "module": "reference",
        "algorithm": "numba" if use_numba and NUMBA_AVAILABLE else "numpy",
        "size": size,
        "dtype": dtype,
        "time_ms": time_ms,
        "gfops": gflops,
        "bandwidth_gbs": 3 * size * size * 4 / time_ms / 1e6,
    }


def run_reference_benchmarks(sizes: list[int] = None, dtypes: list[str] = None) -> list[dict]:
    """Run reference benchmarks."""
    if sizes is None:
        sizes = [64, 128, 256, 512, 1024]
    if dtypes is None:
        dtypes = ["fp32"]

    results = []
    for size in sizes:
        for dtype in dtypes:
            result = benchmark_reference(size, dtype)
            results.append(result)
    return results
