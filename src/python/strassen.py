"""Strassen's Matrix Multiplication Algorithm implementations."""

import numpy as np
import time
from typing import Optional
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
import config


def strassen_naive(A: np.ndarray, B: np.ndarray, threshold: int = None) -> np.ndarray:
    """
    Pure Python/NumPy Strassen's algorithm for matrix multiplication.
    Uses explicit loops for base case (naive), NumPy slicing for recursion.
    
    Args:
        A: First matrix (N x N)
        B: Second matrix (N x N)
        threshold: Crossover size to switch to standard multiplication
    
    Returns:
        C: Result matrix (N x N)
    """
    if threshold is None:
        threshold = config.STRASSEN_CROSSOVER_THRESHOLD
    
    n = A.shape[0]
    
    if n <= threshold or n % 2 != 0:
        return _gemm_naive_python(A, B)
    
    mid = n // 2
    
    A11 = A[:mid, :mid]
    A12 = A[:mid, mid:]
    A21 = A[mid:, :mid]
    A22 = A[mid:, mid:]
    
    B11 = B[:mid, :mid]
    B12 = B[:mid, mid:]
    B21 = B[mid:, :mid]
    B22 = B[mid:, mid:]
    
    M1 = strassen_naive(A11 + A22, B11 + B22, threshold)
    M2 = strassen_naive(A21 + A22, B11, threshold)
    M3 = strassen_naive(A11, B12 - B22, threshold)
    M4 = strassen_naive(A22, B21 - B11, threshold)
    M5 = strassen_naive(A11 + A12, B22, threshold)
    M6 = strassen_naive(A21 - A11, B11 + B12, threshold)
    M7 = strassen_naive(A12 - A22, B21 + B22, threshold)
    
    C = np.zeros((n, n), dtype=A.dtype)
    
    C[:mid, :mid] = M1 + M4 - M5 + M7
    C[:mid, mid:] = M3 + M5
    C[mid:, :mid] = M2 + M4
    C[mid:, mid:] = M1 - M2 + M3 + M6
    
    return C


def strassen_numpy(A: np.ndarray, B: np.ndarray, threshold: int = None) -> np.ndarray:
    """
    NumPy-optimized Strassen's algorithm.
    Uses NumPy @ operator for base case, which leverages BLAS.
    
    Args:
        A: First matrix (N x N)
        B: Second matrix (N x N)
        threshold: Crossover size to switch to standard multiplication
    
    Returns:
        C: Result matrix (N x N)
    """
    if threshold is None:
        threshold = config.STRASSEN_CROSSOVER_THRESHOLD
    
    n = A.shape[0]
    
    if n <= threshold or n % 2 != 0:
        return A @ B
    
    mid = n // 2
    
    A11 = A[:mid, :mid]
    A12 = A[:mid, mid:]
    A21 = A[mid:, :mid]
    A22 = A[mid:, mid:]
    
    B11 = B[:mid, :mid]
    B12 = B[:mid, mid:]
    B21 = B[mid:, :mid]
    B22 = B[mid:, mid:]
    
    M1 = strassen_numpy(A11 + A22, B11 + B22, threshold)
    M2 = strassen_numpy(A21 + A22, B11, threshold)
    M3 = strassen_numpy(A11, B12 - B22, threshold)
    M4 = strassen_numpy(A22, B21 - B11, threshold)
    M5 = strassen_numpy(A11 + A12, B22, threshold)
    M6 = strassen_numpy(A21 - A11, B11 + B12, threshold)
    M7 = strassen_numpy(A12 - A22, B21 + B22, threshold)
    
    C = np.zeros((n, n), dtype=A.dtype)
    
    C[:mid, :mid] = M1 + M4 - M5 + M7
    C[:mid, mid:] = M3 + M5
    C[mid:, :mid] = M2 + M4
    C[mid:, mid:] = M1 - M2 + M3 + M6
    
    return C


def _gemm_naive_python(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Pure Python triple-loop GEMM for base case."""
    n = A.shape[0]
    C = np.zeros((n, n), dtype=A.dtype)
    for i in range(n):
        for j in range(n):
            s = A.dtype.type(0)
            for k in range(n):
                s += A[i, k] * B[k, j]
            C[i, j] = s
    return C


def _pad_to_even(A: np.ndarray) -> tuple[np.ndarray, int]:
    """Pad matrix to even dimensions if necessary."""
    n = A.shape[0]
    if n % 2 == 0:
        return A, n
    new_n = n + 1
    A_padded = np.zeros((new_n, new_n), dtype=A.dtype)
    A_padded[:n, :n] = A
    return A_padded, n


def benchmark_strassen_naive(size: int, dtype: str = "fp32", warmup: int = 3, iterations: int = 10, threshold: int = None) -> dict:
    """Benchmark Strassen naive implementation."""
    if threshold is None:
        threshold = config.STRASSEN_CROSSOVER_THRESHOLD
    
    dtype_map = {"fp32": np.float32, "fp16": np.float16, "bf16": np.float32, "int8": np.int8}
    np_dtype = dtype_map.get(dtype, np.float32)
    
    np.random.seed(42)
    A = np.random.randn(size, size).astype(np_dtype)
    B = np.random.randn(size, size).astype(np_dtype)
    
    A_padded, original_n = _pad_to_even(A)
    B_padded, _ = _pad_to_even(B)
    
    for _ in range(warmup):
        C = strassen_naive(A_padded, B_padded, threshold)
    
    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        C = strassen_naive(A_padded, B_padded, threshold)
        end = time.perf_counter()
        times.append((end - start) * 1000)
    
    time_ms = np.median(times)
    gflops = 2 * size**3 / time_ms / 1e6
    
    bytes_per_elem = {"fp32": 4, "fp16": 2, "bf16": 2, "int8": 1}.get(dtype, 4)
    bandwidth = 3 * size * size * bytes_per_elem / time_ms / 1e6
    
    return {
        "module": "python",
        "algorithm": "strassen_naive",
        "size": size,
        "dtype": dtype,
        "time_ms": time_ms,
        "gfops": gflops,
        "bandwidth_gbs": bandwidth,
    }


def benchmark_strassen_numpy(size: int, dtype: str = "fp32", warmup: int = 3, iterations: int = 10, threshold: int = None) -> dict:
    """Benchmark Strassen NumPy implementation."""
    if threshold is None:
        threshold = config.STRASSEN_CROSSOVER_THRESHOLD
    
    dtype_map = {"fp32": np.float32, "fp16": np.float16, "bf16": np.float32, "int8": np.int8}
    np_dtype = dtype_map.get(dtype, np.float32)
    
    np.random.seed(42)
    A = np.random.randn(size, size).astype(np_dtype)
    B = np.random.randn(size, size).astype(np_dtype)
    
    A_padded, original_n = _pad_to_even(A)
    B_padded, _ = _pad_to_even(B)
    
    for _ in range(warmup):
        C = strassen_numpy(A_padded, B_padded, threshold)
    
    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        C = strassen_numpy(A_padded, B_padded, threshold)
        end = time.perf_counter()
        times.append((end - start) * 1000)
    
    time_ms = np.median(times)
    gflops = 2 * size**3 / time_ms / 1e6
    
    bytes_per_elem = {"fp32": 4, "fp16": 2, "bf16": 2, "int8": 1}.get(dtype, 4)
    bandwidth = 3 * size * size * bytes_per_elem / time_ms / 1e6
    
    return {
        "module": "python",
        "algorithm": "strassen_numpy",
        "size": size,
        "dtype": dtype,
        "time_ms": time_ms,
        "gfops": gflops,
        "bandwidth_gbs": bandwidth,
    }


def run_strassen_benchmarks(sizes: list[int] = None, dtypes: list[str] = None, threshold: int = None) -> list[dict]:
    """Run Strassen benchmarks for both naive and NumPy implementations."""
    if sizes is None:
        sizes = config.QUICK_SIZES
    if dtypes is None:
        dtypes = ["fp32"]
    if threshold is None:
        threshold = config.STRASSEN_CROSSOVER_THRESHOLD
    
    results = []
    for size in sizes:
        for dtype in dtypes:
            result = benchmark_strassen_naive(size, dtype, threshold=threshold)
            results.append(result)
            
            result = benchmark_strassen_numpy(size, dtype, threshold=threshold)
            results.append(result)
    
    return results


def verify_correctness(size: int = 8, threshold: int = 2) -> bool:
    """Verify Strassen implementation correctness against NumPy."""
    np.random.seed(42)
    A = np.random.randn(size, size).astype(np.float32)
    B = np.random.randn(size, size).astype(np.float32)
    
    C_numpy = A @ B
    C_strassen_naive = strassen_naive(A, B, threshold)
    C_strassen_numpy = strassen_numpy(A, B, threshold)
    
    naive_correct = np.allclose(C_numpy, C_strassen_naive, rtol=1e-4, atol=1e-4)
    numpy_correct = np.allclose(C_numpy, C_strassen_numpy, rtol=1e-4, atol=1e-4)
    
    print(f"Strassen naive correct: {naive_correct}")
    print(f"Strassen numpy correct: {numpy_correct}")
    
    return naive_correct and numpy_correct


if __name__ == "__main__":
    print("Verifying Strassen correctness...")
    verify_correctness(size=16, threshold=4)
    
    print("\nRunning small benchmark...")
    results = run_strassen_benchmarks(sizes=[64, 128, 256])
    for r in results:
        print(f"  {r['algorithm']} N={r['size']}: {r['gfops']:.2f} GFLOPS")
