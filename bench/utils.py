"""Shared utilities - matrix generation, timing helpers, result formatting."""

import numpy as np
from typing import Optional
import time


DTYPE_MAP = {
    "fp32": np.float32,
    "fp16": np.float16,
    "bf16": np.float32,  # NumPy doesn't have bfloat16, use float32
    "int8": np.int8,
}


def generate_random_matrix(size: int, dtype: str = "fp32", seed: Optional[int] = 42) -> np.ndarray:
    """Generate a random matrix of given size and data type."""
    np.random.seed(seed)
    dtype_np = DTYPE_MAP.get(dtype, np.float32)
    if dtype == "int8":
        return np.random.randint(-128, 127, (size, size), dtype=dtype_np)
    return np.random.randn(size, size).astype(dtype_np)


def generate_sparse_matrix(size: int, dtype: str = "fp32", sparsity: float = 0.9, seed: int = 42) -> np.ndarray:
    """Generate a sparse matrix with given sparsity (0.9 = 90% zeros)."""
    np.random.seed(seed)
    dtype_np = DTYPE_MAP.get(dtype, np.float32)
    if dtype == "int8":
        dense = np.random.randint(-128, 127, (size, size), dtype=dtype_np)
    else:
        dense = np.random.randn(size, size).astype(dtype_np)
    
    mask = np.random.random((size, size)) > sparsity
    return dense * mask


def generate_matrices(size: int, dtype: str = "fp32", seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """Generate pair of matrices for GEMM: C = A @ B."""
    A = generate_random_matrix(size, dtype, seed)
    B = generate_random_matrix(size, dtype, seed + 1)
    return A, B


class Timer:
    """Simple context manager for timing code blocks."""

    def __init__(self):
        self.start = None
        self.elapsed_ms = None

    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.elapsed_ms = (time.perf_counter() - self.start) * 1000


def verify_result(A: np.ndarray, B: np.ndarray, C: np.ndarray, rtol: float = 1e-3, atol: float = 1e-3) -> bool:
    """Verify GEMM result against NumPy reference."""
    expected = A @ B
    return np.allclose(C, expected, rtol=rtol, atol=atol)


def format_result(result: dict) -> str:
    """Format a benchmark result for display."""
    return (
        f"{result['module']}.{result['algorithm']} "
        f"[N={result['size']}, {result['dtype']}] "
        f"{result['time_ms']:.2f}ms, "
        f"{result['gflops']:.2f} GFLOPS, "
        f"{result['bandwidth_gbs']:.2f} GB/s"
    )
