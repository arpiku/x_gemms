"""Benchmark runner module - executes all module benchmarks, collects timing, computes GFLOPS."""

import time
import csv
import os
from pathlib import Path
from typing import Callable, Any
import numpy as np


BENCHMARK_SIZES = [64, 128, 256, 512, 1024, 2048, 4096, 8192]
SPARSE_SIZES = [1024, 2048]
DATA_TYPES = ["fp32", "fp16", "bf16", "int8"]
RESULTS_DIR = Path(__file__).parent.parent / "results"


def compute_gflops(size: int, time_ms: float, dtype: str) -> float:
    """Compute GFLOPS for square matrix multiplication."""
    ops = 2 * (size ** 3)
    if time_ms > 0:
        return (ops / time_ms) / 1e6
    return 0.0


def compute_bandwidth(size: int, time_ms: float, dtype: str) -> float:
    """Compute effective memory bandwidth in GB/s."""
    bytes_per_elem = {"fp32": 4, "fp16": 2, "bf16": 2, "int8": 1}.get(dtype, 4)
    bytes_transferred = 3 * size * size * bytes_per_elem
    if time_ms > 0:
        return (bytes_transferred / time_ms) / 1e6
    return 0.0


class BenchmarkRunner:
    """Main benchmark runner class."""

    def __init__(self, output_file: str = "benchmarks.csv"):
        self.results = []
        self.output_file = RESULTS_DIR / output_file
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    def run_benchmark(
        self,
        name: str,
        func: Callable,
        size: int,
        dtype: str,
        warmup: int = 3,
        iterations: int = 10,
        **kwargs
    ) -> dict:
        """Run a single benchmark with warmup and iterations."""
        for _ in range(warmup):
            func(size=size, dtype=dtype, **kwargs)

        times = []
        for _ in range(iterations):
            start = time.perf_counter()
            result = func(size=size, dtype=dtype, **kwargs)
            end = time.perf_counter()
            times.append((end - start) * 1000)

        time_ms = np.median(times)
        gflops = compute_gflops(size, time_ms, dtype)
        bandwidth = compute_bandwidth(size, time_ms, dtype)

        result = {
            "module": name,
            "algorithm": kwargs.get("algorithm", "default"),
            "size": size,
            "dtype": dtype,
            "time_ms": time_ms,
            "gfops": gflops,
            "bandwidth_gbs": bandwidth,
        }
        self.results.append(result)
        return result

    def save_results(self):
        """Save results to CSV file."""
        if not self.results:
            return
        with open(self.output_file, "w", newline="") as f:
            writer = csv.DictWriter(
                f, fieldnames=["module", "algorithm", "size", "dtype", "time_ms", "gfops", "bandwidth_gbs"]
            )
            writer.writeheader()
            writer.writerows(self.results)
        print(f"Results saved to {self.output_file}")

    def get_results(self) -> list[dict]:
        """Get all benchmark results."""
        return self.results


def benchmark_wrapper(func: Callable) -> Callable:
    """Decorator to wrap benchmark functions with timing."""
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    return wrapper
