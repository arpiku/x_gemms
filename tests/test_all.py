"""Main test orchestrator - runs all modules with varying sizes/types."""

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from pybench.runner import BenchmarkRunner, BENCHMARK_SIZES, DATA_TYPES
from pybench.utils import generate_matrices, DTYPE_MAP
from tpu.reference import run_reference_benchmarks


def test_reference(sizes=None, dtypes=None):
    """Test reference implementations (NumPy/Numba)."""
    print("Running reference benchmarks...")
    results = run_reference_benchmarks(sizes, dtypes)
    for r in results:
        print(f"  {r['module']}.{r['algorithm']} [N={r['size']}] {r['gfops']:.2f} GFLOPS")
    return results


def test_cpp():
    """Test C++ implementations via subprocess."""
    print("Running C++ benchmarks...")
    import subprocess
    result = subprocess.run(
        ["./cpp/gemm_bench"],
        capture_output=True,
        text=True,
        cwd=Path(__file__).parent.parent
    )
    if result.returncode == 0:
        print(result.stdout)
    else:
        print(f"C++ benchmark failed: {result.stderr}")
    return []


def test_gpu():
    """Test GPU implementations."""
    print("Running GPU benchmarks...")
    print("  GPU benchmarks require CUDA - skipping in scaffold")
    return []


def test_tpu():
    """Test TPU implementations."""
    print("Running TPU benchmarks...")
    print("  TPU benchmarks require hardware - skipping in scaffold")
    return []


def generate_plots():
    """Generate benchmark plots."""
    from pybench.plotter import generate_all_plots
    generate_all_plots()


def test_all(sizes=None, dtypes=None, run_cpp=True, run_gpu=False, run_tpu=False):
    """Run all benchmark tests."""
    print("=" * 60)
    print("GEMM BENCHMARK SUITE")
    print("=" * 60)
    
    all_results = []
    
    all_results.extend(test_reference(sizes, dtypes))
    
    if run_cpp:
        all_results.extend(test_cpp())
    
    if run_gpu:
        all_results.extend(test_gpu())
    
    if run_tpu:
        all_results.extend(test_tpu())
    
    print("\n" + "=" * 60)
    print("All benchmarks complete!")
    print("=" * 60)
    
    generate_plots()
    
    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run GEMM benchmarks")
    parser.add_argument("--sizes", nargs="+", type=int, default=None,
                        help="Matrix sizes to benchmark")
    parser.add_argument("--dtypes", nargs="+", default=None,
                        help="Data types to benchmark")
    parser.add_argument("--no-cpp", action="store_true", help="Skip C++ benchmarks")
    parser.add_argument("--gpu", action="store_true", help="Run GPU benchmarks")
    parser.add_argument("--tpu", action="store_true", help="Run TPU benchmarks")
    
    args = parser.parse_args()
    
    test_all(
        sizes=args.sizes,
        dtypes=args.dtypes,
        run_cpp=not args.no_cpp,
        run_gpu=args.gpu,
        run_tpu=args.tpu
    )
