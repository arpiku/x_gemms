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
from src.python.strassen import run_strassen_benchmarks
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
    """Get backend label for an algorithm using new naming convention."""
    alg_lower = algorithm.lower()
    mod_lower = module.lower()
    
    # New naming: (Device)-(Implementation)-(Library)-(Language)
    if 'tensor_core' in alg_lower:
        if 'fp16' in alg_lower:
            return "GPU-TensorCore-PyTorch-fp16-Python"
        elif 'bf16' in alg_lower:
            return "GPU-TensorCore-PyTorch-bf16-Python"
        return "GPU-TensorCore-PyTorch-Python"
    elif 'cublas' in alg_lower:
        return "GPU-CUDA-cuBLAS-Cpp"
    elif 'cutlass' in alg_lower:
        return "GPU-CUDA-CUTLASS-Cpp"
    elif 'cuda' in alg_lower:
        return "GPU-CUDA-Cpp"
    elif 'openmp' in alg_lower and 'blocked' in alg_lower:
        return "CPU-Parallel-OpenMP-Cpp-Blocked"
    elif 'openmp' in alg_lower:
        return "CPU-Parallel-OpenMP-Cpp"
    elif 'std_thread' in alg_lower or 'std' in alg_lower:
        return "CPU-Parallel-StdThread-Cpp"
    elif 'avx512' in alg_lower:
        return "CPU-SIMD-AVX512-Cpp"
    elif 'avx2' in alg_lower:
        return "CPU-SIMD-AVX2-Cpp"
    elif 'sse' in alg_lower:
        return "CPU-SIMD-SSE-Cpp"
    elif 'blocked' in alg_lower or 'cache_aware' in alg_lower:
        return "CPU-Blocked-Cpp"
    elif 'naive' in alg_lower:
        if mod_lower == 'reference' or mod_lower == 'numba' or mod_lower == 'python':
            return "CPU-Naive-Numba-Python"
        return "CPU-Naive-Cpp"
    elif 'numpy' in alg_lower:
        return "CPU-Naive-NumPy-Python"
    elif 'sparse' in mod_lower or 'sparse' in alg_lower:
        if 'gpu' in mod_lower or 'cuda' in mod_lower:
            return "GPU-Sparse-PyTorch-Python"
        return "CPU-Sparse-SciPy-Python"
    elif 'strassen' in alg_lower:
        if 'cublas' in alg_lower:
            return "GPU-Strassen-cuBLAS-Cpp"
        elif 'cuda' in alg_lower or 'custom' in alg_lower:
            return "GPU-Strassen-CUDA-Cpp"
        elif 'numpy' in alg_lower:
            return "CPU-Strassen-NumPy-Python"
        elif 'blocked' in alg_lower:
            return "CPU-Strassen-Blocked-Cpp"
        elif 'cache_aware' in alg_lower:
            return "CPU-Strassen-CacheAware-Cpp"
        elif 'naive' in alg_lower:
            if mod_lower == 'python':
                return "CPU-Strassen-Naive-Python"
            return "CPU-Strassen-Naive-Cpp"
        return "CPU-Strassen-Cpp"
    else:
        return "CPU-Naive-Cpp"


def should_skip_size(algorithm: str, size: int) -> tuple[bool, str]:
    """Check if algorithm should be skipped for given size."""
    if config.FORCE_LARGE_SIZES:
        return False, ""
    
    alg_lower = algorithm.lower()
    
    # Single-thread algorithms: limit to SINGLE_THREAD_MAX_SIZE
    single_thread_algos = ['naive', 'sse', 'avx2', 'avx512', 'blocked', 'cache_aware', 
                           'numpy', 'numba', 'reference']
    is_parallel = any(x in alg_lower for x in ['openmp', 'std_thread', 'thread'])
    is_gpu = any(x in alg_lower for x in ['cuda', 'tensor_core', 'cublas', 'cutlass', 'sparse_gpu'])
    is_tensor_core = 'tensor_core' in alg_lower
    
    if is_gpu or is_tensor_core:
        # GPU/TensorCore: up to PARALLEL_MAX_SIZE
        if size > config.PARALLEL_MAX_SIZE:
            return True, f"size > {config.PARALLEL_MAX_SIZE} for GPU"
        return False, ""
    elif is_parallel:
        # Parallel CPU: up to PARALLEL_MAX_SIZE
        if size > config.PARALLEL_MAX_SIZE:
            return True, f"size > {config.PARALLEL_MAX_SIZE} for parallel CPU"
        return False, ""
    else:
        # Single-thread: up to SINGLE_THREAD_MAX_SIZE
        if size > config.SINGLE_THREAD_MAX_SIZE:
            return True, f"size > {config.SINGLE_THREAD_MAX_SIZE} for single-thread"
        return False, ""


def test_reference(sizes=None, dtypes=None):
    """Test reference implementations (NumPy/Numba)."""
    print("\n[Module: Reference/Numba]")
    
    if sizes is None:
        sizes = config.QUICK_SIZES
    if dtypes is None:
        dtypes = ["fp32"]
    
    # Filter sizes based on algorithm type
    filtered_sizes = []
    for size in sizes:
        skip, reason = should_skip_size("numba", size)
        if skip:
            print(f"  [N={size}] CPU-Naive-Numba-Python -> Skipping ({reason})")
        else:
            filtered_sizes.append(size)
    
    if not filtered_sizes:
        return []
    
    results = []
    for size in filtered_sizes:
        for dtype in dtypes:
            label = "CPU-Naive-Numba-Python"
            print(f"  [N={size}, dtype={dtype}] {label}", end=" ", flush=True)
            try:
                # Try Numba first, fall back to NumPy
                r = run_numba_benchmarks([size], [dtype])
                if not r:
                    r = run_numpy_benchmarks([size], [dtype])
                    label = "CPU-Naive-NumPy-Python"
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
    
    # Use provided sizes or default to MEDIUM_SIZES
    if sizes is None:
        sizes = config.get_default_sizes()
    
    # Filter sizes based on parallel vs single-thread limits
    filtered_sizes = []
    for size in sizes:
        skip, reason = should_skip_size("openmp", size)  # Check with parallel algo
        if skip:
            print(f"  [N={size}] Skipping for single-thread ({reason})")
        else:
            filtered_sizes.append(size)
    
    if not filtered_sizes:
        print("  No sizes to run after filtering")
        return []
    
    # Build command: gemm_bench <threads> <sizes...>
    cmd = [str(cpp_binary), str(config.DEFAULT_CPU_THREADS)] + [str(s) for s in filtered_sizes]
    print(f"  Running with {config.DEFAULT_CPU_THREADS} threads, sizes: {filtered_sizes}")
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600,  # 10 min timeout for C++ tests
            cwd=Path(__file__).parent.parent
        )
        if result.returncode == 0:
            # Check for INFO message about default sizes
            if "INFO: No sizes provided" in result.stderr:
                print(f"  Note: C++ defaulted to MEDIUM_SIZES (no sizes passed)")
            
            cpp_results = parse_csv_results(result.stdout)
            for r in cpp_results:
                backend = get_backend_label(r['algorithm'], r['module'])
                print(f"  [N={r['size']}] {backend} -> {r['gfops']:.2f} GFLOPS")
            return cpp_results
        else:
            print(f"  C++ benchmark failed: {result.stderr[:100]}")
            return []
    except subprocess.TimeoutExpired:
        print("  C++ benchmark timed out")
        return []
    except Exception as e:
        print(f"  C++ error: {str(e)[:50]}")
        return []


def test_gpu(sizes=None):
    """Test GPU implementations via CUDA binary."""
    print("\n[Module: GPU/CUDA]")
    
    gpu_binary = Path(__file__).parent.parent / "src" / "cuda" / "gpu_bench"
    if not gpu_binary.exists():
        print(f"  GPU binary not found: {gpu_binary}")
        return []
    
    # Use provided sizes or default to MEDIUM_SIZES
    if sizes is None:
        sizes = config.get_default_sizes()
    
    # Build command: gpu_bench <sizes...>
    cmd = [str(gpu_binary)] + [str(s) for s in sizes]
    print(f"  Running with sizes: {sizes}")
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600,
            cwd=Path(__file__).parent.parent
        )
        if result.returncode == 0:
            # Check for INFO message about default sizes
            if "INFO: No sizes provided" in result.stderr:
                print(f"  Note: CUDA defaulted to MEDIUM_SIZES (no sizes passed)")
            
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
                formats=config.SPARSE_FORMATS[:2],
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


def test_strassen_python(sizes=None, dtypes=None):
    """Test Python Strassen implementations."""
    print("\n[Module: Strassen Python]")
    
    if sizes is None:
        sizes = config.QUICK_SIZES
    if dtypes is None:
        dtypes = ["fp32"]
    
    filtered_sizes = []
    for size in sizes:
        skip, reason = should_skip_size("strassen", size)
        if skip:
            print(f"  [N={size}] CPU-Strassen-Python -> Skipping ({reason})")
        else:
            filtered_sizes.append(size)
    
    if not filtered_sizes:
        return []
    
    results = []
    try:
        r = run_strassen_benchmarks(sizes=filtered_sizes, dtypes=dtypes)
        if r:
            for res in r:
                label = get_backend_label(res['algorithm'], res['module'])
                print(f"  [N={res['size']}] {label} -> {res['gfops']:.2f} GFLOPS")
                results.append(res)
    except Exception as e:
        print(f"  Strassen error: {str(e)[:50]}")
    
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
    run_strassen=True,
    sparse_backend="gpu",
    large_sizes=False
):
    """Run all benchmark tests."""
    start_time = time.perf_counter()
    
    print("=" * 60)
    print("GEMM BENCHMARK SUITE")
    print("=" * 60)
    
    if large_sizes:
        cpp_sizes = config.ALL_SIZES
        python_sizes = config.QUICK_SIZES
    else:
        cpp_sizes = sizes or config.MEDIUM_SIZES_CPP
        python_sizes = config.QUICK_SIZES
    
    print(f"Matrix sizes (C++/GPU): {cpp_sizes}")
    print(f"Matrix sizes (Python): {python_sizes}")
    print(f"Timeout per test: {BENCHMARK_TIMEOUT}s")
    print()
    
    all_results = []
    
    all_results.extend(test_reference(python_sizes, dtypes))
    
    if run_strassen:
        all_results.extend(test_strassen_python(python_sizes, dtypes))
    
    if run_cpp:
        all_results.extend(test_cpp(cpp_sizes))
    
    if run_gpu:
        all_results.extend(test_gpu(cpp_sizes))
    
    if run_cutlass:
        all_results.extend(test_cutlass(cpp_sizes, dtypes))
    
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
    parser.add_argument("--no-strassen", action="store_true", help="Skip Strassen benchmarks")
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
        run_strassen=not args.no_strassen,
        sparse_backend=args.sparse_backend,
        large_sizes=args.large
    )