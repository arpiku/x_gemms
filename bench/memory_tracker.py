"""
Memory Tracking Module - Tracks CPU and GPU memory usage during GEMM operations.

This module records memory footprint for each benchmark, enabling analysis of:
- Peak GPU memory allocation
- CPU memory usage  
- Intermediate buffer sizes
- Memory bandwidth efficiency

Outputs to: results/memory_tracking.csv
"""

import csv
import time
import gc
import os
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass

import numpy as np

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


RESULTS_DIR = Path(__file__).parent.parent / "results"
MEMORY_CSV = RESULTS_DIR / "memory_tracking.csv"


@dataclass
class MemoryMetrics:
    """Container for memory metrics."""
    algorithm: str
    size: int
    dtype: str
    peak_gpu_memory_mb: float
    host_memory_mb: float
    temp_memory_mb: float
    effective_bandwidth_gbs: float
    time_ms: float


def get_gpu_memory_mb() -> float:
    """Get current GPU memory allocated in MB."""
    if TORCH_AVAILABLE and torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / (1024 * 1024)
    return 0.0


def get_cpu_memory_mb() -> float:
    """Get current CPU memory usage in MB (Linux)."""
    try:
        import resource
        # Get current memory usage
        rusage = resource.getrusage(resource.RUSAGE_SELF)
        return rusage.ru_maxrss / 1024  # Convert KB to MB
    except:
        return 0.0


def reset_memory_stats():
    """Reset memory tracking counters."""
    if TORCH_AVAILABLE and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
    gc.collect()


def compute_memory_bandwidth(size: int, time_ms: float, dtype: str) -> float:
    """Compute effective memory bandwidth."""
    bytes_per_elem = {"fp32": 4, "fp16": 2, "bf16": 2, "int8": 1}.get(dtype, 4)
    bytes_transferred = 3 * size * size * bytes_per_elem
    if time_ms > 0:
        return (bytes_transferred / time_ms) / 1e6
    return 0.0


def benchmark_with_memory_tracking(
    algorithm: str,
    size: int,
    dtype: str,
    gemm_func,
    iterations: int = 10,
    warmup: int = 3
) -> MemoryMetrics:
    """
    Run GEMM benchmark with full memory tracking.
    
    Args:
        algorithm: Algorithm name
        size: Matrix size N
        dtype: Data type
        gemm_func: Function to benchmark
        iterations: Number of timing iterations
        warmup: Number of warmup iterations
    
    Returns:
        MemoryMetrics with all memory data
    """
    reset_memory_stats()
    
    cpu_before = get_cpu_memory_mb()
    
    # Warmup
    for _ in range(warmup):
        _ = gemm_func()
    
    if TORCH_AVAILABLE and torch.cuda.is_available():
        torch.cuda.synchronize()
    
    # Benchmark
    times = []
    for _ in range(iterations):
        reset_memory_stats()
        
        start = time.perf_counter()
        _ = gemm_func()
        
        if TORCH_AVAILABLE and torch.cuda.is_available():
            torch.cuda.synchronize()
        
        end = time.perf_counter()
        times.append((end - start) * 1000)
    
    time_ms = np.median(times)
    
    # Get memory metrics
    peak_gpu = get_gpu_memory_mb()
    cpu_after = get_cpu_memory_mb()
    host_memory = cpu_after - cpu_before
    
    # Estimate temp memory (rough approximation based on algorithm)
    temp_memory = estimate_temp_memory(size, dtype, algorithm)
    
    bandwidth = compute_memory_bandwidth(size, time_ms, dtype)
    
    return MemoryMetrics(
        algorithm=algorithm,
        size=size,
        dtype=dtype,
        peak_gpu_memory_mb=peak_gpu,
        host_memory_mb=host_memory,
        temp_memory_mb=temp_memory,
        effective_bandwidth_gbs=bandwidth,
        time_ms=time_ms
    )


def estimate_temp_memory(size: int, dtype: str, algorithm: str) -> float:
    """
    Estimate temporary memory needed for algorithm.
    
    This is a rough approximation based on algorithm complexity.
    """
    bytes_per_elem = {"fp32": 4, "fp16": 2, "bf16": 2, "int8": 1}.get(dtype, 4)
    base_memory = size * size * bytes_per_elem / (1024 * 1024)  # MB
    
    # Additional memory based on algorithm
    if "blocked" in algorithm or "tile" in algorithm:
        # Blocked algorithms may need extra for blocking
        return base_memory * 0.5
    elif "strassen" in algorithm:
        # Strassen needs 7 temporary matrices
        return base_memory * 7
    elif "naive" in algorithm:
        # Naive just needs output buffer
        return base_memory
    else:
        return base_memory * 0.3


def save_memory_metrics(metrics_list: List[MemoryMetrics], output_file: str = "memory_tracking.csv"):
    """Save memory metrics to CSV file."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = RESULTS_DIR / output_file
    
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "algorithm", "size", "dtype", 
            "peak_gpu_memory_mb", "host_memory_mb", "temp_memory_mb",
            "effective_bandwidth_gbs", "time_ms"
        ])
        
        for m in metrics_list:
            writer.writerow([
                m.algorithm, m.size, m.dtype,
                f"{m.peak_gpu_memory_mb:.2f}", f"{m.host_memory_mb:.2f}", f"{m.temp_memory_mb:.2f}",
                f"{m.effective_bandwidth_gbs:.2f}", f"{m.time_ms:.2f}"
            ])
    
    print(f"Memory tracking saved to {output_path}")


def load_memory_metrics(csv_file: str = "memory_tracking.csv") -> List[MemoryMetrics]:
    """Load memory metrics from CSV file."""
    filepath = RESULTS_DIR / csv_file
    if not filepath.exists():
        return []
    
    metrics_list = []
    with open(filepath, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            metrics_list.append(MemoryMetrics(
                algorithm=row["algorithm"],
                size=int(row["size"]),
                dtype=row["dtype"],
                peak_gpu_memory_mb=float(row["peak_gpu_memory_mb"]),
                host_memory_mb=float(row["host_memory_mb"]),
                temp_memory_mb=float(row["temp_memory_mb"]),
                effective_bandwidth_gbs=float(row["effective_bandwidth_gbs"]),
                time_ms=float(row["time_ms"])
            ))
    return metrics_list


def run_memory_benchmarks(
    sizes: List[int] = None,
    dtypes: List[str] = None
) -> List[MemoryMetrics]:
    """Run memory tracking benchmarks for multiple sizes and types."""
    if sizes is None:
        sizes = [256, 512, 1024]
    if dtypes is None:
        dtypes = ["fp32", "fp16"]
    
    results = []
    
    if not TORCH_AVAILABLE:
        print("PyTorch not available, skipping memory tracking")
        return results
    
    if not torch.cuda.is_available():
        print("CUDA not available, skipping GPU memory tracking")
        return results
    
    device = torch.device("cuda")
    
    for size in sizes:
        for dtype in dtypes:
            torch_dtype = {"fp32": torch.float32, "fp16": torch.float16}.get(dtype, torch.float32)
            
            # Allocate matrices
            A = torch.randn(size, size, device=device, dtype=torch_dtype)
            B = torch.randn(size, size, device=device, dtype=torch_dtype)
            
            def gemm_op():
                C = torch.matmul(A, B)
                torch.cuda.synchronize()
                return C
            
            # Run with memory tracking
            metrics = benchmark_with_memory_tracking(
                algorithm=f"tensor_core_{dtype}",
                size=size,
                dtype=dtype,
                gemm_func=gemm_op
            )
            results.append(metrics)
            print(f"  {metrics.algorithm} N={size}: GPU={metrics.peak_gpu_memory_mb:.1f}MB, "
                  f"Host={metrics.host_memory_mb:.1f}MB, BW={metrics.effective_bandwidth_gbs:.1f}GB/s")
    
    save_memory_metrics(results)
    return results


if __name__ == "__main__":
    print("Memory Tracking Benchmarks")
    print("=" * 50)
    
    results = run_memory_benchmarks([256, 512, 1024], ["fp32", "fp16"])
    
    print("\nMemory Metrics Summary:")
    for m in results:
        print(f"  {m.algorithm} N={m.size}: "
              f"GPU={m.peak_gpu_memory_mb:.1f}MB, "
              f"Host={m.host_memory_mb:.1f}MB")
