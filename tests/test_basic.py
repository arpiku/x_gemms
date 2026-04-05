"""Test C++ basic implementations."""

import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from pybench.runner import BenchmarkRunner, BENCHMARK_SIZES


def test_basic():
    """Test basic C++ implementations."""
    print("Testing basic C++ implementations...")
    
    for size in BENCHMARK_SIZES[:4]:
        result = subprocess.run(
            ["./cpp/gemm_bench", str(size)],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent
        )
        if result.returncode == 0:
            print(f"  Size {size}: OK")
        else:
            print(f"  Size {size}: FAILED")


if __name__ == "__main__":
    test_basic()
