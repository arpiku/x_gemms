"""
GPU Profiler Module - Detailed GPU performance profiling using NSight Compute.

This module provides detailed GPU profiling including:
- DRAM bandwidth utilization
- L2 cache statistics
- SM (Streaming Multiprocessor) efficiency
- Memory latency analysis

Requires: NVIDIA NSight Compute installed at /usr/local/NVIDIA-Nsight-Compute-2026.1

Fallback: Uses PyTorch profiler if NSight is not available.
"""

import subprocess
import csv
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


NSIGHT_PATH = "/usr/local/NVIDIA-Nsight-Compute-2026.1"
NSIGHT_CLI = os.path.join(NSIGHT_PATH, "ncu")  # Use ncu directly

RESULTS_DIR = Path(__file__).parent.parent / "pybench" / "results"


@dataclass
class GPUProfileMetrics:
    """Container for detailed GPU profiling metrics."""
    algorithm: str
    size: int
    cuda_time_ms: float
    dram_bandwidth_gbs: float
    l2_hit_rate_pct: float
    sm_efficiency_pct: float
    memory_throughput_gbs: float
    achieved_occupancy_pct: float


def is_nsight_available() -> bool:
    """Check if NSight Compute CLI is available."""
    return os.path.exists(NSIGHT_CLI)


def is_cuda_available() -> bool:
    """Check if CUDA is available."""
    return TORCH_AVAILABLE and torch.cuda.is_available()


def parse_nsight_metrics(output: str) -> Dict[str, float]:
    """Parse NSight Compute output to extract metrics."""
    metrics = {
        "dram_bandwidth_gbs": 0.0,
        "l2_hit_rate_pct": 0.0,
        "sm_efficiency_pct": 0.0,
        "achieved_occupancy_pct": 0.0,
    }
    
    # Try to parse JSON output if available
    try:
        data = json.loads(output)
        # Extract metrics from JSON
        if "metrics" in data:
            metrics_data = data["metrics"]
            metrics["dram_bandwidth_gbs"] = metrics_data.get("dram__bytes.sum", 0) / 1e9
    except:
        # Fallback: parse text output
        lines = output.split('\n')
        for line in lines:
            if "dram__bytes.sum" in line:
                parts = line.split()
                if len(parts) >= 2:
                    try:
                        metrics["dram_bandwidth_gbs"] = float(parts[1]) / 1e9
                    except:
                        pass
            elif "l2tex__t_sectors_hit_pct" in line:
                parts = line.split()
                if len(parts) >= 2:
                    try:
                        metrics["l2_hit_rate_pct"] = float(parts[1])
                    except:
                        pass
    
    return metrics


def profile_with_nsight_cli(
    binary_path: str,
    metrics: List[str] = None,
    output_file: str = "nsight_output.txt"
) -> Dict[str, float]:
    """
    Run NSight Compute CLI on a binary.
    
    Args:
        binary_path: Path to GPU binary to profile
        metrics: List of metrics to collect
        output_file: Output file for results
    
    Returns:
        Dictionary of collected metrics
    """
    if not is_nsight_available():
        return {"error": "NSight not available"}
    
    if metrics is None:
        metrics = [
            "dram__bytes.sum",
            "dram__bytes_per_second.sum",
            "l2tex__t_sectors_hit_pct",
            "smsp__average_occupancy.pct",
            "smsp__thread_inst_executed.per_half_sm.sum"
        ]
    
    # Build command
    metric_str = ",".join(metrics)
    cmd = [
        NSIGHT_CLI,
        "-m", metric_str,
        "--capture", binary_path,
        "-o", str(RESULTS_DIR / output_file)
    ]
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=120
        )
        
        if result.returncode == 0:
            output_path = RESULTS_DIR / output_file
            if output_path.exists():
                with open(output_path, "r") as f:
                    return parse_nsight_metrics(f.read())
        else:
            return {"error": f"NSight failed: {result.stderr}"}
    except subprocess.TimeoutExpired:
        return {"error": "NSight timeout"}
    except Exception as e:
        return {"error": str(e)}
    
    return {}


def profile_with_pytorch_profiler(
    size: int,
    dtype: str = "fp16",
    iterations: int = 10
) -> GPUProfileMetrics:
    """
    Fallback: Use PyTorch profiler when NSight is not available.
    
    Provides basic memory and timing metrics.
    """
    if not is_cuda_available():
        return GPUProfileMetrics(
            algorithm="pytorch_profiler",
            size=size,
            cuda_time_ms=0.0,
            dram_bandwidth_gbs=0.0,
            l2_hit_rate_pct=0.0,
            sm_efficiency_pct=0.0,
            memory_throughput_gbs=0.0,
            achieved_occupancy_pct=0.0
        )
    
    device = torch.device("cuda")
    torch_dtype = {"fp32": torch.float32, "fp16": torch.float16}.get(dtype, torch.float32)
    
    A = torch.randn(size, size, device=device, dtype=torch_dtype)
    B = torch.randn(size, size, device=device, dtype=torch_dtype)
    
    # Warmup
    for _ in range(3):
        C = torch.matmul(A, B)
    torch.cuda.synchronize()
    
    # Profile with PyTorch
    import time
    times = []
    
    for _ in range(iterations):
        torch.cuda.synchronize()
        start = time.perf_counter()
        C = torch.matmul(A, B)
        torch.cuda.synchronize()
        end = time.perf_counter()
        times.append((end - start) * 1000)
    
    cuda_time = sorted(times)[len(times)//2]
    
    # Estimate bandwidth
    bytes_per_elem = 2 if dtype == "fp16" else 4
    bytes_transferred = 3 * size * size * bytes_per_elem
    bandwidth = bytes_transferred / cuda_time / 1e6
    
    return GPUProfileMetrics(
        algorithm=f"tensor_core_{dtype}",
        size=size,
        cuda_time_ms=cuda_time,
        dram_bandwidth_gbs=bandwidth,
        l2_hit_rate_pct=0.0,  # Not available via PyTorch
        sm_efficiency_pct=0.0,
        memory_throughput_gbs=bandwidth,
        achieved_occupancy_pct=0.0
    )


def run_gpu_profiling(
    sizes: List[int] = None,
    dtypes: List[str] = None,
    use_nsight: bool = True,
    binary_path: str = None
) -> List[GPUProfileMetrics]:
    """
    Run GPU profiling benchmarks.
    
    Args:
        sizes: Matrix sizes to profile
        dtypes: Data types to test
        use_nsight: Whether to use NSight (if available)
        binary_path: Path to GPU binary for NSight profiling
    
    Returns:
        List of GPUProfileMetrics
    """
    if sizes is None:
        sizes = [512, 1024]
    if dtypes is None:
        dtypes = ["fp16"]
    
    results = []
    
    nsight_available = is_nsight_available()
    cuda_available = is_cuda_available()
    
    print("GPU Profiling")
    print("=" * 50)
    print(f"NSight available: {nsight_available}")
    print(f"CUDA available: {cuda_available}")
    
    if not cuda_available:
        print("CUDA not available, skipping profiling")
        return results
    
    # If NSight requested and available, try to use it
    if use_nsight and nsight_available and binary_path:
        print(f"Using NSight CLI on {binary_path}")
        
        # Run NSight on binary
        nsight_results = profile_with_nsight_cli(binary_path)
        
        if "error" not in nsight_results:
            for size in sizes:
                results.append(GPUProfileMetrics(
                    algorithm="nsight_profiled",
                    size=size,
                    cuda_time_ms=0.0,
                    dram_bandwidth_gbs=nsight_results.get("dram_bandwidth_gbs", 0.0),
                    l2_hit_rate_pct=nsight_results.get("l2_hit_rate_pct", 0.0),
                    sm_efficiency_pct=nsight_results.get("sm_efficiency_pct", 0.0),
                    memory_throughput_gbs=nsight_results.get("dram_bandwidth_gbs", 0.0),
                    achieved_occupancy_pct=nsight_results.get("achieved_occupancy_pct", 0.0)
                ))
    else:
        print("Falling back to PyTorch profiler")
        for size in sizes:
            for dtype in dtypes:
                metrics = profile_with_pytorch_profiler(size, dtype)
                results.append(metrics)
                print(f"  {metrics.algorithm} N={size}: "
                      f"time={metrics.cuda_time_ms:.2f}ms, "
                      f"BW={metrics.dram_bandwidth_gbs:.1f}GB/s")
    
    return results


def save_profiling_results(
    metrics_list: List[GPUProfileMetrics],
    output_file: str = "gpu_profiling.csv"
):
    """Save GPU profiling results to CSV."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = RESULTS_DIR / output_file
    
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "algorithm", "size", "cuda_time_ms",
            "dram_bandwidth_gbs", "l2_hit_rate_pct",
            "sm_efficiency_pct", "memory_throughput_gbs", "achieved_occupancy_pct"
        ])
        
        for m in metrics_list:
            writer.writerow([
                m.algorithm, m.size, f"{m.cuda_time_ms:.2f}",
                f"{m.dram_bandwidth_gbs:.2f}", f"{m.l2_hit_rate_pct:.2f}",
                f"{m.sm_efficiency_pct:.2f}", f"{m.memory_throughput_gbs:.2f}",
                f"{m.achieved_occupancy_pct:.2f}"
            ])
    
    print(f"Profiling results saved to {output_path}")


if __name__ == "__main__":
    # Test the profiler
    results = run_gpu_profiling(
        sizes=[512, 1024],
        dtypes=["fp16"],
        use_nsight=False  # Use PyTorch profiler by default
    )
    
    if results:
        save_profiling_results(results)
