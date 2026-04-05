"""Test Tensor Core / TPU implementations."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def test_tpu():
    """Test Tensor Core implementations."""
    print("Testing Tensor Core implementations...")
    
    try:
        from src.cuda.tensor_core import is_tensor_core_available, run_tensor_core_benchmarks
        if is_tensor_core_available():
            results = run_tensor_core_benchmarks(sizes=[256, 512])
            for r in results:
                print(f"  {r['algorithm']} [N={r['size']}] {r['gfops']:.2f} GFLOPS")
        else:
            print("  No Tensor Core available - skipping")
    except ImportError as e:
        print(f"  PyTorch not available: {e}")


if __name__ == "__main__":
    test_tpu()
