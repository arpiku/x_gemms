# x_gemms Configuration
# Central benchmark configuration for GEMM operations

from typing import Dict, List

# =============================================================================
# Matrix Sizes
# =============================================================================

BENCHMARK_SIZES: Dict[str, List[int]] = {
    "small": [64, 128, 256],
    "medium": [512, 1024],
    "large": [2048, 4096, 8192],
    "very_large": [10000, 15000, 20000],
    "sparse": [1024, 2048, 4096],
}

# All sizes for full benchmark run (combines all categories)
ALL_SIZES: List[int] = [
    64, 128, 256,
    512, 1024,
    2048, 4096, 8192,
    10000, 15000, 20000,
]

# Sizes for quick tests (subset of ALL_SIZES)
QUICK_SIZES: List[int] = [64, 128, 256, 512, 1024]

# Sizes for medium tests (excludes very large, good for development)
MEDIUM_SIZES: List[int] = [64, 128, 256, 512, 1024, 2048, 4096, 8192]

# Timeout per benchmark in seconds (prevents hanging on large sizes)
BENCHMARK_TIMEOUT_SECONDS: int = 120  # 2 minutes max per test

# =============================================================================
# Data Types
# =============================================================================

DATA_TYPES: List[str] = ["fp32", "fp16", "bf16", "int8"]

# Supported data types per backend
SUPPORTED_DTYPES: Dict[str, Dict[str, bool]] = {
    "cpp": {
        "fp32": True,
        "fp16": False,  # Not implemented in C++ scaffold
        "bf16": False,  # Not implemented in C++ scaffold
        "int8": True,   # Template supports it
    },
    "gpu_cuda": {
        "fp32": True,
        "fp16": True,
        "bf16": True,
        "int8": True,
    },
    "tensor_core": {
        "fp32": False,
        "fp16": True,
        "bf16": True,
        "int8": True,
    },
    "reference": {
        "fp32": True,
        "fp16": True,
        "bf16": "fallback",  # Uses fp32 as fallback
        "int8": False,       # Not supported in NumPy @ operations
    },
}

# =============================================================================
# CPU Configuration
# =============================================================================

DEFAULT_CPU_THREADS: int = 8
CPU_AVX512_SUPPORTED: bool = False  # Set based on runtime CPU detection

# =============================================================================
# GPU Configuration
# =============================================================================

GPU_ARCH: str = "sm_120"  # Blackwell/RTX 5070
CUDA_VERSION: str = "13.1"

# =============================================================================
# Sparse Matrix Configuration
# =============================================================================

SPARSE_SPARSITY: float = 0.9  # 90% zeros
SPARSE_FORMATS: List[str] = ["dense", "csr", "coo"]

# =============================================================================
# Benchmark Settings
# =============================================================================

WARMUP_ITERATIONS: int = 3
BENCHMARK_ITERATIONS: int = 10

# =============================================================================
# Memory Bandwidth Reference (theoretical max)
# =============================================================================

MEMORY_BANDWIDTH: Dict[str, str] = {
    "gpu_ddr5": "~500+ GB/s (RTX 5070 Blackwell)",
    "cpu_ddr5_per_channel": "~50 GB/s (DDR5-4800)",
    "cpu_l1_cache": "~2 TB/s",
    "cpu_l2_cache": "~1 TB/s",
    "cpu_l3_cache": "~500 GB/s",
}

# =============================================================================
# Output Configuration
# =============================================================================

RESULTS_DIR: str = "pybench/results"
DEFAULT_CSV_OUTPUT: str = "benchmarks.csv"
SPARSE_CSV_OUTPUT: str = "sparse_benchmarks.csv"

# =============================================================================
# Helper Functions
# =============================================================================

def is_dtype_supported(backend: str, dtype: str) -> bool:
    """Check if a data type is supported by a specific backend."""
    if backend not in SUPPORTED_DTYPES:
        return False
    if dtype not in SUPPORTED_DTYPES[backend]:
        return False
    value = SUPPORTED_DTYPES[backend][dtype]
    return value is True


def get_dtype_fallback(backend: str, dtype: str) -> str:
    """Get fallback dtype if original is not supported."""
    if backend == "reference" and dtype == "bf16":
        return "fp32"
    if backend == "reference" and dtype == "int8":
        return "fp32"
    return dtype


def get_effective_dtype(backend: str, dtype: str) -> str:
    """Get effective dtype to use (considering fallbacks)."""
    if is_dtype_supported(backend, dtype):
        return dtype
    return get_dtype_fallback(backend, dtype)
