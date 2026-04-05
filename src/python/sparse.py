"""
Sparse Matrix GEMM implementations.

Tests different sparse matrix formats:
- dense: Standard dense matrix multiplication (baseline)
- csr: Compressed Sparse Row format (optimized for row access)
- coo: Coordinate format (simpler, good for construction)

Uses scipy.sparse for CPU and PyTorch for GPU sparse operations.
"""

import numpy as np
import time
from typing import Optional, Dict, List, Tuple
import warnings

warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', message='.*Sparse.*')
import torch
torch.sparse.check_sparse_tensor_invariants.enable()

try:
    from scipy import sparse
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    sparse = None

TORCH_AVAILABLE = torch.cuda.is_available()

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

try:
    import config
    SPARSITY = config.SPARSE_SPARSITY
except:
    SPARSITY = 0.9  # 90% zeros


def compute_gflops(size: int, time_ms: float, dtype: str, density: float = 1.0) -> float:
    """Compute effective GFLOPS for sparse matrix multiplication."""
    bytes_per_elem = {"fp32": 4, "fp16": 2, "bf16": 2, "int8": 1}.get(dtype, 4)
    effective_nnz = int(size * size * density)
    ops = 2 * effective_nnz * size
    if time_ms > 0:
        return (ops / time_ms) / 1e6
    return 0.0


def compute_bandwidth(size: int, time_ms: float, dtype: str, density: float = 1.0) -> float:
    """Compute effective memory bandwidth."""
    bytes_per_elem = {"fp32": 4, "fp16": 2, "bf16": 2, "int8": 1}.get(dtype, 4)
    bytes_read_sparse = size * size * density * bytes_per_elem
    bytes_read_dense = 2 * size * size * bytes_per_elem
    bytes_written = size * size * bytes_per_elem
    bytes_transferred = bytes_read_sparse + bytes_read_dense + bytes_written
    if time_ms > 0:
        return (bytes_transferred / time_ms) / 1e6
    return 0.0


def generate_sparse_matrix(
    size: int,
    density: float,
    format: str = "csr",
    dtype: np.dtype = np.float32,
    seed: int = 42
):
    """Generate a sparse matrix with given density."""
    np.random.seed(seed)
    
    if dtype == np.int8:
        data = np.random.randint(-128, 127, int(size * size * density))
    else:
        data = np.random.randn(int(size * size * density)).astype(dtype)
    
    row = np.random.randint(0, size, int(size * size * density))
    col = np.random.randint(0, size, int(size * size * density))
    
    if format == "csr":
        return sparse.csr_matrix((data, (row, col)), shape=(size, size), dtype=dtype)
    elif format == "coo":
        return sparse.coo_matrix((data, (row, col)), shape=(size, size), dtype=dtype)
    else:
        raise ValueError(f"Unknown format: {format}")


def benchmark_sparse_dense_cpu(
    size: int,
    sparsity: float = SPARSITY,
    format: str = "dense",
    dtype: str = "fp32",
    warmup: int = 3,
    iterations: int = 10
) -> Dict:
    """Benchmark sparse matrix multiplication on CPU."""
    if not SCIPY_AVAILABLE:
        return {"error": "scipy not available", "module": "sparse", "algorithm": format}
    
    dtype_map = {"fp32": np.float32, "fp16": np.float16, "int8": np.int8}
    np_dtype = dtype_map.get(dtype, np.float32)
    density = 1.0 - sparsity
    
    if format == "dense":
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
        gflops = compute_gflops(size, time_ms, dtype, density=1.0)
        bandwidth = compute_bandwidth(size, time_ms, dtype, density=1.0)
    else:
        A_sparse = generate_sparse_matrix(size, density, format, np_dtype, seed=42)
        B = np.random.randn(size, size).astype(np_dtype)
        
        for _ in range(warmup):
            C = A_sparse @ B
        
        times = []
        for _ in range(iterations):
            start = time.perf_counter()
            C = A_sparse @ B
            end = time.perf_counter()
            times.append((end - start) * 1000)
        
        time_ms = np.median(times)
        gflops = compute_gflops(size, time_ms, dtype, density=density)
        bandwidth = compute_bandwidth(size, time_ms, dtype, density=density)
    
    return {
        "module": "sparse",
        "algorithm": format,
        "size": size,
        "dtype": dtype,
        "sparsity": sparsity,
        "time_ms": time_ms,
        "gfops": gflops,
        "bandwidth_gbs": bandwidth,
    }


def benchmark_sparse_gpu(
    size: int,
    sparsity: float = SPARSITY,
    format: str = "dense",
    dtype: str = "fp32",
    warmup: int = 3,
    iterations: int = 10
) -> Dict:
    """Benchmark sparse matrix multiplication on GPU using PyTorch."""
    if not TORCH_AVAILABLE:
        return {"error": "PyTorch not available", "module": "sparse_gpu", "algorithm": format}
    
    if not torch.cuda.is_available():
        return {"error": "CUDA not available", "module": "sparse_gpu", "algorithm": format}
    
    device = torch.device("cuda")
    dtype_map = {"fp32": torch.float32, "fp16": torch.float16}
    torch_dtype = dtype_map.get(dtype, torch.float32)
    density = 1.0 - sparsity
    
    if format == "dense":
        A = torch.randn(size, size, device=device, dtype=torch_dtype)
        B = torch.randn(size, size, device=device, dtype=torch_dtype)
        
        for _ in range(warmup):
            C = torch.matmul(A, B)
        torch.cuda.synchronize()
        
        times = []
        for _ in range(iterations):
            start = time.perf_counter()
            C = torch.matmul(A, B)
            torch.cuda.synchronize()
            end = time.perf_counter()
            times.append((end - start) * 1000)
        
        time_ms = np.median(times)
        gflops = compute_gflops(size, time_ms, dtype, density=1.0)
        bandwidth = compute_bandwidth(size, time_ms, dtype, density=1.0)
    else:
        A_scipy = generate_sparse_matrix(size, density, "csr", 
                                         np.float32 if dtype == "fp32" else np.float16)
        
        A_sparse = torch.sparse_csr_tensor(
            torch.from_numpy(A_scipy.indptr.astype(np.int64)),
            torch.from_numpy(A_scipy.indices.astype(np.int64)),
            torch.from_numpy(A_scipy.data.astype(np.float32 if dtype == "fp32" else np.float16)),
            size=(size, size),
            device=device,
            dtype=torch_dtype
        )
        
        B = torch.randn(size, size, device=device, dtype=torch_dtype)
        
        for _ in range(warmup):
            C = torch.sparse.mm(A_sparse, B)
        torch.cuda.synchronize()
        
        times = []
        for _ in range(iterations):
            start = time.perf_counter()
            C = torch.sparse.mm(A_sparse, B)
            torch.cuda.synchronize()
            end = time.perf_counter()
            times.append((end - start) * 1000)
        
        time_ms = np.median(times)
        gflops = compute_gflops(size, time_ms, dtype, density=density)
        bandwidth = compute_bandwidth(size, time_ms, dtype, density=density)
    
    return {
        "module": "sparse_gpu",
        "algorithm": format,
        "size": size,
        "dtype": dtype,
        "sparsity": sparsity,
        "time_ms": time_ms,
        "gfops": gflops,
        "bandwidth_gbs": bandwidth,
    }


def run_sparse_benchmarks(
    sizes: List[int] = None,
    formats: List[str] = None,
    sparsity: float = SPARSITY,
    dtypes: List[str] = None,
    backend: str = "cpu"
) -> List[Dict]:
    """Run sparse matrix benchmarks."""
    if sizes is None:
        sizes = [1024, 2048, 4096]
    if formats is None:
        formats = ["dense", "csr", "coo"]
    if dtypes is None:
        dtypes = ["fp32"]
    
    results = []
    
    for size in sizes:
        for fmt in formats:
            for dtype in dtypes:
                if backend == "cpu":
                    result = benchmark_sparse_dense_cpu(size, sparsity, fmt, dtype)
                else:
                    result = benchmark_sparse_gpu(size, sparsity, fmt, dtype)
                
                if "error" not in result:
                    results.append(result)
    
    return results
