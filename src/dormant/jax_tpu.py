"""JAX/XLA TPU backend for GEMM benchmarks.

Requires JAX with TPU support: pip install jax[tpu]
"""

import os
from typing import Optional
import numpy as np

try:
    import jax
    import jax.numpy as jnp
    from jax import lax
    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False
    jnp = None


def is_tpu_available() -> bool:
    """Check if TPU is available."""
    if not JAX_AVAILABLE:
        return False
    try:
        return "tpu" in jax.default_backend().lower()
    except:
        return False


def get_tpu_devices():
    """Get available TPU devices."""
    if not JAX_AVAILABLE:
        return []
    return jax.devices()


def gemm_jax(A: jnp.ndarray, B: jnp.ndarray, precision: str = "default") -> jnp.ndarray:
    """JAX GEMM using lax.dot with optional precision control."""
    return lax.dot(A, B, precision=precision)


def benchmark_jax_tpu(size: int, dtype: str = "fp32", warmup: int = 3, iterations: int = 10) -> dict:
    """Benchmark JAX GEMM on TPU."""
    if not JAX_AVAILABLE:
        return {"error": "JAX not available"}

    dtype_map = {"fp32": jnp.float32, "fp16": jnp.float16, "bf16": jnp.bfloat16}
    jax_dtype = dtype_map.get(dtype, jnp.float32)

    A = jnp.ones((size, size), dtype=jax_dtype)
    B = jnp.ones((size, size), dtype=jax_dtype)

    for _ in range(warmup):
        _ = gemm_jax(A, B)

    import time
    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        C = gemm_jax(A, B)
        C.block_until_ready()
        end = time.perf_counter()
        times.append((end - start) * 1000)

    time_ms = np.median(times)
    gflops = 2 * size**3 / time_ms / 1e6

    return {
        "module": "jax_tpu",
        "algorithm": "lax_dot",
        "size": size,
        "dtype": dtype,
        "time_ms": time_ms,
        "gfops": gflops,
        "bandwidth_gbs": 3 * size * size * 4 / time_ms / 1e6,
    }


def run_jax_benchmarks(sizes: list[int] = None, dtypes: list[str] = None) -> list[dict]:
    """Run JAX TPU benchmarks for multiple sizes and types."""
    if sizes is None:
        sizes = [64, 128, 256, 512, 1024]
    if dtypes is None:
        dtypes = ["fp32", "fp16", "bf16"]

    results = []
    for size in sizes:
        for dtype in dtypes:
            result = benchmark_jax_tpu(size, dtype)
            results.append(result)
    return results
