"""CLI entry point for x_gemms benchmarks."""

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def main():
    parser = argparse.ArgumentParser(
        description="x_gemms - GEMM Benchmarking Framework",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m bench run --quick              Run quick benchmarks (sizes 64-1024)
  python -m bench run --medium             Run medium benchmarks (up to N=2048)
  python -m bench run --large              Run full benchmarks (up to N=20000)
  python -m bench run --quick --save       Run and save with timestamp
  python -m bench run --medium --save --tag baseline  Save with tag
  python -m bench run --quick --threads 4  Run with 4 CPU threads
  python -m bench run --medium --force     Force run all sizes (may hang!)
  python -m bench analyze                  Launch interactive Jupyter notebook
  python -m bench analyze results/*.csv    Analyze specific CSV files
  python -m bench                          Show this help
        """
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Commands")
    
    # run subcommand
    run_parser = subparsers.add_parser("run", help="Run benchmarks")
    run_parser.add_argument("--quick", action="store_true", help="Quick benchmarks (sizes 64-1024)")
    run_parser.add_argument("--medium", action="store_true", help="Medium benchmarks (up to N=2048)")
    run_parser.add_argument("--large", action="store_true", help="Full benchmarks (up to N=20000)")
    run_parser.add_argument("--threads", type=int, default=None, help="Number of CPU threads for C++ benchmarks")
    run_parser.add_argument("--force", action="store_true", help="Force run all sizes (dangerous - may hang!)")
    run_parser.add_argument("--save", action="store_true", help="Save results with timestamp")
    run_parser.add_argument("--tag", type=str, default=None, help="Optional tag for filename (e.g., 'baseline')")
    run_parser.add_argument("--no-cpp", action="store_true", help="Skip C++ benchmarks")
    run_parser.add_argument("--no-gpu", action="store_true", help="Skip GPU benchmarks")
    run_parser.add_argument("--no-cutlass", action="store_true", help="Skip Tensor Core benchmarks")
    run_parser.add_argument("--no-sparse", action="store_true", help="Skip sparse benchmarks")
    run_parser.add_argument("--no-strassen", action="store_true", help="Skip Strassen benchmarks")
    run_parser.add_argument("--no-winograd", action="store_true", help="Skip Winograd benchmarks")
    run_parser.add_argument("--sparse-backend", choices=["cpu", "gpu"], default="gpu", help="Sparse backend")
    
    # analyze subcommand
    analyze_parser = subparsers.add_parser("analyze", help="Launch interactive analysis notebook")
    analyze_parser.add_argument("files", nargs="*", help="CSV files to analyze (default: load latest)")
    analyze_parser.add_argument("--no-browser", action="store_true", help="Don't auto-open browser")
    
    args = parser.parse_args()
    
    if args.command == "run":
        from tests.test_all import test_all
        import config
        
        sizes = None
        if args.quick:
            sizes = config.QUICK_SIZES
        elif args.medium:
            sizes = config.MEDIUM_SIZES
        elif args.large:
            sizes = config.ALL_SIZES
        
        if args.threads:
            import config as cfg
            cfg.DEFAULT_CPU_THREADS = args.threads
            cfg.CPU_THREADS_CONFIG = args.threads
            print(f"Using {args.threads} CPU threads (override)")
        
        if args.force:
            import config as cfg
            cfg.FORCE_LARGE_SIZES = True
            print(f"Force running all sizes (WARNING: may hang!)")
        
        test_all(
            sizes=sizes,
            run_cpp=not args.no_cpp,
            run_gpu=not args.no_gpu,
            run_cutlass=not args.no_cutlass,
            run_sparse=not args.no_sparse,
            run_strassen=not args.no_strassen,
            run_winograd=not args.no_winograd,
            sparse_backend=args.sparse_backend,
            save_results_flag=args.save,
            tag=args.tag
        )
        
    elif args.command == "analyze":
        from bench.analyzer import launch_notebook, BenchmarkAnalyzer, RESULTS_DIR
        
        if args.files:
            csv_files = [str(f) for f in args.files]
        else:
            csv_files = None
        
        print("Launching interactive analysis notebook...")
        print(f"Results directory: {RESULTS_DIR}")
        if csv_files:
            print(f"Loading files: {csv_files}")
        
        launch_notebook(csv_files)
        
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
