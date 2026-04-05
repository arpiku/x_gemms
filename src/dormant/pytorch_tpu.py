"""PyTorch XLA TPU backend for GEMM benchmarks.

Requires PyTorch XLA: pip install torch xla
"""

import numpy as np
from typing import Optional

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None


try:
    import torch_xla
    import torch_xla.core.xla_model as xm
    XLA_AVAILABLE = True
except ImportError:
    XLA_AVAILABLE = False
    xm = None


def is_tpu_available() -> bool:
    """Check if TPU is available via PyTorch XLA."""
    if not XLA_AVAILABLE:
        return False
    try:
        devices = xm.get_xla_supported_devices()
        return any("TPU" in str(d) for d in devices)
    except:
        return False


def get_tpu_devices():
    """Get available TPU devices."""
    if not XLA_AVAILABLE:
        return []
    return xm.get_xla_supported_devices()


def gemm_pytorch(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """PyTorch GEMM using torch.matmul."""
    return torch.matmul(A, B)


def benchmark_pytorch_tpu(size: int, dtype: str = "fp32", warmup: int = 3, iterations: int = 10) -> dict:
    """Benchmark PyTorch GEMM on TPU."""
    if not TORCH_AVAILABLE or not XLA_AVAILABLE:
        return {"error": "PyTorch XLA not available"}

    dtype_map = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}
    torch_dtype = dtype_map.get(dtype, torch.float32)

    device = xm.xla_device()
    
    A = torch.ones((size, size), dtype=torch_dtype, device=device)
    B = torch.ones((size, size), dtype=torch_dtype, device=device)

    for _ in range(warmup):
        _ = gemm_pytorch(A, B)
    xm.mark_step()

    import time
    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        C = gemm_pytorch(A, B)
        xm.mark_step()
        end = time.perf_counter()
        times.append((end - start) * 1000)

    time_ms = np.median(times)
    gflops = 2 * size**3 / time_ms / 1e6

    return {
        "module": "pytorch_tpu",
        "algorithm": "matmul",
        "size": size,
        "dtype": dtype,
        "time_ms": time_ms,
        "gflops": gflops,
        "bandwidth_gbs": 3 * size * size * 4 / time_ms / 1e6,
    }


def run_pytorch_benchmarks(sizes: list[int] = None, dtypes: list[str] = None) -> list[dict]:
    """Run PyTorch TPU benchmarks for multiple sizes and types."""
    if sizes is None:
        sizes = [64, 128, 256, 512, 1024]
    if dtypes is None:
        dtypes = ["fp32", "fp16", "bf16"]

    results = []
    for size in sizes:
        for dtype in dtypes:
            result = benchmark_pytorch_tpu(size, dtype)
            results.append(result)
    return results
