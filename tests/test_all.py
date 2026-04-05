"""Main test orchestrator - runs all modules with varying sizes/types."""

import sys
import csv
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import config
from tpu.reference import run_reference_benchmarks
from tpu.cutlass_bench import run_cutlass_benchmarks, is_tensor_core_available
from tpu.sparse import run_sparse_benchmarks


def parse_csv_results(csv_output: str) -> list[dict]:
    """Parse CSV output from benchmark binaries."""
    results = []
    lines = csv_output.strip().split('\n')
    if len(lines) < 2:
        return results
    
    headers = lines[0].split(',')
    for line in lines[1:]:
        values = line.split(',')
        if len(values) >= 5:
            result = {
                "module": values[0].split('.')[0] if '.' in values[0] else values[0],
                "algorithm": values[0].split('.')[-1] if '.' in values[0] else values[0],
                "size": int(values[1]),
                "dtype": "fp32",
                "time_ms": float(values[2]),
                "gfops": float(values[3]),
                "bandwidth_gbs": float(values[4]),
            }
            results.append(result)
    return results


def test_reference(sizes=None, dtypes=None):
    """Test reference implementations (NumPy/Numba)."""
    print("Running reference benchmarks...")
    
    if sizes is None:
        sizes = config.QUICK_SIZES
    if dtypes is None:
        dtypes = ["fp32"]
    
    results = run_reference_benchmarks(sizes, dtypes)
    for r in results:
        print(f"  {r['module']}.{r['algorithm']} [N={r['size']}] {r['gfops']:.2f} GFLOPS")
    return results


def test_cpp(sizes=None):
    """Test C++ implementations via subprocess."""
    print("Running C++ benchmarks...")
    import subprocess
    
    cpp_binary = Path(__file__).parent.parent / "cpp" / "gemm_bench"
    if not cpp_binary.exists():
        print(f"  C++ binary not found: {cpp_binary}")
        return []
    
    result = subprocess.run(
        [str(cpp_binary), str(config.DEFAULT_CPU_THREADS)],
        capture_output=True,
        text=True,
        cwd=Path(__file__).parent.parent
    )
    if result.returncode == 0:
        return parse_csv_results(result.stdout)
    else:
        print(f"  C++ benchmark failed: {result.stderr}")
        return []


def test_gpu():
    """Test GPU implementations via CUDA binary."""
    print("Running GPU benchmarks...")
    import subprocess
    
    gpu_binary = Path(__file__).parent.parent / "gpu" / "gpu_bench"
    if not gpu_binary.exists():
        print(f"  GPU binary not found: {gpu_binary}")
        return []
    
    result = subprocess.run(
        [str(gpu_binary)],
        capture_output=True,
        text=True,
        cwd=Path(__file__).parent.parent
    )
    if result.returncode == 0:
        return parse_csv_results(result.stdout)
    else:
        print(f"  GPU benchmark failed: {result.stderr}")
        return []


def test_cutlass(sizes=None, dtypes=None):
    """Test CUTLASS-backed Tensor Core benchmarks via PyTorch."""
    print("Running CUTLASS (Tensor Core) benchmarks...")
    
    if not is_tensor_core_available():
        print("  Tensor Core not available, skipping")
        return []
    
    if sizes is None:
        sizes = [256, 512, 1024, 2048]
    if dtypes is None:
        dtypes = ["fp16", "bf16"]
    
    results = run_cutlass_benchmarks(sizes, dtypes)
    for r in results:
        print(f"  {r['module']}.{r['algorithm']} [N={r['size']}, {r['dtype']}] {r['gfops']:.2f} GFLOPS")
    return results


def test_sparse(sizes=None, backend="gpu"):
    """Test sparse matrix benchmarks."""
    print(f"Running sparse benchmarks ({backend})...")
    
    if sizes is None:
        sizes = config.BENCHMARK_SIZES["sparse"]
    
    results = run_sparse_benchmarks(
        sizes=sizes,
        formats=config.SPARSE_FORMATS,
        sparsity=config.SPARSE_SPARSITY,
        backend=backend
    )
    for r in results:
        print(f"  {r['module']}.{r['algorithm']} [N={r['size']}, sparsity={r.get('sparsity', 0)}] {r['gfops']:.2f} GFLOPS")
    return results


def test_tpu():
    """Test TPU (Tensor Core) - now handled by CUTLASS module."""
    # Keeping for backward compatibility, but CUTLASS module handles Tensor Core
    return test_cutlass()


def generate_plots():
    """Generate benchmark plots."""
    from pybench.plotter import generate_all_plots
    generate_all_plots()


def save_results(results: list[dict], output_file: str = "benchmarks.csv"):
    """Save all results to CSV."""
    if not results:
        return
    
    output_path = Path(__file__).parent.parent / "pybench" / "results" / output_file
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Standard fieldnames
    fieldnames = ["module", "algorithm", "size", "dtype", "time_ms", "gfops", "bandwidth_gbs", "sparsity"]
    
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(results)
    
    print(f"Results saved to {output_path}")


def test_all(
    sizes=None,
    dtypes=None,
    run_cpp=True,
    run_gpu=True,
    run_cutlass=True,
    run_sparse=True,
    sparse_backend="gpu",
    large_sizes=False
):
    """Run all benchmark tests."""
    print("=" * 60)
    print("GEMM BENCHMARK SUITE")
    print("=" * 60)
    
    if large_sizes:
        sizes = config.ALL_SIZES
    else:
        sizes = sizes or config.QUICK_SIZES
    
    print(f"Matrix sizes: {sizes}")
    print()
    
    all_results = []
    
    # Reference (CPU NumPy/Numba)
    all_results.extend(test_reference(sizes, dtypes))
    
    # C++ CPU benchmarks
    if run_cpp:
        all_results.extend(test_cpp(sizes))
    
    # GPU CUDA benchmarks
    if run_gpu:
        all_results.extend(test_gpu())
    
    # CUTLASS Tensor Core benchmarks
    if run_cutlass:
        all_results.extend(test_cutlass(sizes, dtypes))
    
    # Sparse matrix benchmarks
    if run_sparse:
        all_results.extend(test_sparse(backend=sparse_backend))
    
    print("\n" + "=" * 60)
    print("All benchmarks complete!")
    print("=" * 60)
    
    if all_results:
        save_results(all_results)
        generate_plots()
    
    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run GEMM benchmarks")
    parser.add_argument("--sizes", nargs="+", type=int, default=None,
                        help="Matrix sizes to benchmark")
    parser.add_argument("--dtypes", nargs="+", default=None,
                        help="Data types to benchmark")
    parser.add_argument("--no-cpp", action="store_true", help="Skip C++ benchmarks")
    parser.add_argument("--no-gpu", action="store_true", help="Skip GPU benchmarks")
    parser.add_argument("--no-cutlass", action="store_true", help="Skip CUTLASS/Tensor Core benchmarks")
    parser.add_argument("--no-sparse", action="store_true", help="Skip sparse benchmarks")
    parser.add_argument("--sparse-backend", choices=["cpu", "gpu"], default="gpu",
                        help="Sparse benchmark backend")
    parser.add_argument("--large", action="store_true", help="Use large matrix sizes (up to 20000)")
    parser.add_argument("--quick", action="store_true", help="Use quick test sizes only")
    
    args = parser.parse_args()
    
    if args.quick:
        sizes = config.QUICK_SIZES
    elif args.large:
        sizes = config.ALL_SIZES
    else:
        sizes = args.sizes
    
    test_all(
        sizes=sizes,
        dtypes=args.dtypes,
        run_cpp=not args.no_cpp,
        run_gpu=not args.no_gpu,
        run_cutlass=not args.no_cutlass,
        run_sparse=not args.no_sparse,
        sparse_backend=args.sparse_backend,
        large_sizes=args.large
    )
