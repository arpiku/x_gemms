"""Test TPU implementations."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def test_tpu():
    """Test TPU implementations."""
    print("Testing TPU implementations...")
    
    try:
        from tpu.jax_tpu import is_tpu_available, run_jax_benchmarks
        if is_tpu_available():
            results = run_jax_benchmarks()
            for r in results:
                print(f"  {r['module']} [N={r['size']}] {r['gfops']:.2f} GFLOPS")
        else:
            print("  No TPU available - skipping")
    except ImportError as e:
        print(f"  JAX not available: {e}")


if __name__ == "__main__":
    test_tpu()
