"""Main test orchestrator - runs all modules with varying sizes/types."""

import sys
import csv
import argparse
import time
import subprocess
from pathlib import Path
from concurrent.futures import TimeoutError as FuturesTimeoutError

sys.path.insert(0, str(Path(__file__).parent.parent))

import config
from src.python.numba import run_numba_benchmarks
from src.python.numpy import run_numpy_benchmarks
from src.cuda.tensor_core import run_tensor_core_benchmarks, is_tensor_core_available
from src.python.sparse import run_sparse_benchmarks

# Timeout for each benchmark in seconds
BENCHMARK_TIMEOUT = getattr(config, 'BENCHMARK_TIMEOUT_SECONDS', 120)


def parse_csv_results(csv_output: str) -> list[dict]:
    """Parse CSV output from benchmark binaries."""
    results = []
    lines = csv_output.strip().split('\n')
    if len(lines) < 2:
        return results
    
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


def get_backend_label(algorithm: str, module: str = "") -> str:
    """Get backend label for an algorithm."""
    alg_lower = algorithm.lower()
    mod_lower = module.lower()
    
    if 'tensor_core' in alg_lower or 'matmul' in alg_lower:
        return "GPU TensorCore"
    elif 'cublas' in alg_lower or 'cuda' in alg_lower:
        return "CUDA"
    elif 'openmp' in alg_lower and 'blocked' in alg_lower:
        threads = getattr(config, 'DEFAULT_CPU_THREADS', 8)
        return f"CPU multi-core ({threads} threads)"
    elif 'openmp' in alg_lower or 'thread' in alg_lower or 'std' in alg_lower:
        threads = getattr(config, 'DEFAULT_CPU_THREADS', 8)
        return f"CPU multi-core ({threads} threads)"
    elif any(x in alg_lower for x in ['sse', 'avx', 'naive', 'blocked', 'cache']):
        return "CPU single-core"
    elif 'sparse' in mod_lower:
        return "GPU Sparse"
    else:
        return "CPU single-core"


def test_reference(sizes=None, dtypes=None):
    """Test reference implementations (NumPy/Numba)."""
    print("\n[Module: Reference/Numba]")
    
    if sizes is None:
        sizes = config.QUICK_SIZES
    if dtypes is None:
        dtypes = ["fp32"]
    
    # Skip very large sizes for reference (too slow)
    sizes = [s for s in sizes if s <= 4096]
    
    results = []
    for size in sizes:
        for dtype in dtypes:
            print(f"  [N={size}, dtype={dtype}] CPU single-core", end=" ", flush=True)
            try:
                # Try Numba first, fall back to NumPy
                r = run_numba_benchmarks([size], [dtype])
                if not r:
                    r = run_numpy_benchmarks([size], [dtype])
                if r:
                    result = r[0]
                    print(f"-> {result['gfops']:.2f} GFLOPS")
                    results.append(result)
                else:
                    print("-> FAILED")
            except Exception as e:
                print(f"-> ERROR: {str(e)[:50]}")
    
    return results


def test_cpp(sizes=None):
    """Test C++ implementations via subprocess."""
    print("\n[Module: C++ CPU]")
    import subprocess
    
    cpp_binary = Path(__file__).parent.parent / "src" / "cpp" / "gemm_bench"
    if not cpp_binary.exists():
        print(f"  C++ binary not found: {cpp_binary}")
        return []
    
    print(f"  Running with {config.DEFAULT_CPU_THREADS} threads...")
    
    try:
        result = subprocess.run(
            [str(cpp_binary), str(config.DEFAULT_CPU_THREADS)],
            capture_output=True,
            text=True,
            timeout=300,  # 5 min timeout for C++ tests
            cwd=Path(__file__).parent.parent
        )
        if result.returncode == 0:
            return parse_csv_results(result.stdout)
        else:
            print(f"  C++ benchmark failed: {result.stderr[:100]}")
            return []
    except subprocess.TimeoutExpired:
        print("  C++ benchmark timed out")
        return []
    except Exception as e:
        print(f"  C++ error: {str(e)[:50]}")
        return []


def test_gpu():
    """Test GPU implementations via CUDA binary."""
    print("\n[Module: GPU/CUDA]")
    
    gpu_binary = Path(__file__).parent.parent / "src" / "cuda" / "gpu_bench"
    if not gpu_binary.exists():
        print(f"  GPU binary not found: {gpu_binary}")
        return []
    
    try:
        result = subprocess.run(
            [str(gpu_binary)],
            capture_output=True,
            text=True,
            timeout=300,
            cwd=Path(__file__).parent.parent
        )
        if result.returncode == 0:
            results = parse_csv_results(result.stdout)
            for r in results:
                backend = get_backend_label(r['algorithm'], r['module'])
                print(f"  [N={r['size']}] {backend} -> {r['gfops']:.2f} GFLOPS")
            return results
        else:
            print(f"  GPU benchmark failed: {result.stderr[:100]}")
            return []
    except subprocess.TimeoutExpired:
        print("  GPU benchmark timed out")
        return []
    except Exception as e:
        print(f"  GPU error: {str(e)[:50]}")
        return []


def test_cutlass(sizes=None, dtypes=None):
    """Test Tensor Core benchmarks via PyTorch."""
    print("\n[Module: TensorCore]")
    
    if not is_tensor_core_available():
        print("  Tensor Core not available, skipping")
        return []
    
    if sizes is None:
        sizes = [256, 512, 1024, 2048]
    if dtypes is None:
        dtypes = ["fp16", "bf16"]
    
    # Skip very large for TensorCore (optional, can enable)
    sizes = [s for s in sizes if s <= 4096]
    
    results = []
    for size in sizes:
        for dtype in dtypes:
            print(f"  [N={size}, dtype={dtype}] GPU TensorCore", end=" ", flush=True)
            try:
                r = run_tensor_core_benchmarks([size], [dtype])
                if r:
                    result = r[0]
                    print(f"-> {result['gfops']:.2f} GFLOPS")
                    results.append(result)
                else:
                    print("-> FAILED")
            except Exception as e:
                print(f"-> ERROR: {str(e)[:50]}")
    
    return results


def test_sparse(sizes=None, backend="gpu"):
    """Test sparse matrix benchmarks."""
    print(f"\n[Module: Sparse ({backend})]")
    
    if sizes is None:
        sizes = config.BENCHMARK_SIZES["sparse"]
    
    results = []
    for size in sizes:
        print(f"  [N={size}, 90% sparsity] ", end="", flush=True)
        try:
            r = run_sparse_benchmarks(
                sizes=[size],
                formats=config.SPARSE_FORMATS[:2],  # Just dense and CSR for speed
                sparsity=config.SPARSE_SPARSITY,
                backend=backend
            )
            if r:
                for res in r:
                    print(f"{res['algorithm']}->{res['gfops']:.2f} ", end="")
                    results.append(res)
                print()
            else:
                print("FAILED")
        except Exception as e:
            print(f"ERROR: {str(e)[:50]}")
    
    return results


def test_tpu():
    """Test TPU (Tensor Core) - now handled by CUTLASS module."""
    return test_cutlass()


def generate_plots():
    """Generate benchmark plots."""
    from bench.plotter import generate_all_plots
    generate_all_plots()


def save_results(results: list[dict], output_file: str = "benchmarks.csv"):
    """Save all results to CSV."""
    if not results:
        return
    
    output_path = Path(__file__).parent.parent / "results" / output_file
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
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
    start_time = time.perf_counter()
    
    print("=" * 60)
    print("GEMM BENCHMARK SUITE")
    print("=" * 60)
    
    if large_sizes:
        sizes = config.ALL_SIZES
    else:
        sizes = sizes or config.QUICK_SIZES
    
    print(f"Matrix sizes: {sizes}")
    print(f"Timeout per test: {BENCHMARK_TIMEOUT}s")
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
    
    end_time = time.perf_counter()
    total_time = end_time - start_time
    
    print("\n" + "=" * 60)
    print("All benchmarks complete!")
    print(f"Total test time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
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
    parser.add_argument("--medium", action="store_true", help="Use medium test sizes (up to 8192)")
    
    args = parser.parse_args()
    
    if args.quick:
        sizes = config.QUICK_SIZES
    elif args.medium:
        sizes = config.MEDIUM_SIZES
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