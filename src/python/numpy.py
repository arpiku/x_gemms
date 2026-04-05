"""NumPy GEMM implementation."""

import numpy as np
import time
from typing import Optional


def gemm_numpy(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Standard NumPy GEMM."""
    return A @ B


def benchmark_numpy(size: int, dtype: str = "fp32", warmup: int = 3, iterations: int = 10) -> dict:
    """Benchmark NumPy implementation."""
    dtype_map = {"fp32": np.float32, "fp16": np.float16, "bf16": np.float32, "int8": np.int8}
    np_dtype = dtype_map.get(dtype, np.float32)

    np.random.seed(42)
    A = np.random.randn(size, size).astype(np_dtype)
    B = np.random.randn(size, size).astype(np_dtype)

    for _ in range(warmup):
        C = A @ B

    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        C = A @ B
        end = time.perf_counter()
        times.append((end - start) * 1000)

    time_ms = np.median(times)
    gflops = 2 * size**3 / time_ms / 1e6

    bytes_per_elem = {"fp32": 4, "fp16": 2, "bf16": 2, "int8": 1}.get(dtype, 4)
    bandwidth = 3 * size * size * bytes_per_elem / time_ms / 1e6

    return {
        "module": "python",
        "algorithm": "numpy",
        "size": size,
        "dtype": dtype,
        "time_ms": time_ms,
        "gfops": gflops,
        "bandwidth_gbs": bandwidth,
    }


def run_numpy_benchmarks(sizes: list[int] = None, dtypes: list[str] = None) -> list[dict]:
    """Run NumPy benchmarks."""
    if sizes is None:
        sizes = [64, 128, 256, 512, 1024]
    if dtypes is None:
        dtypes = ["fp32"]

    results = []
    for size in sizes:
        for dtype in dtypes:
            result = benchmark_numpy(size, dtype)
            results.append(result)
    return results
