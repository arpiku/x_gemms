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
  python -m bench run --quick --threads 4  Run with 4 CPU threads
  python -m bench run --medium --force     Force run all sizes (may hang!)
  python -m bench plot                     Generate plots from results
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
    run_parser.add_argument("--no-cpp", action="store_true", help="Skip C++ benchmarks")
    run_parser.add_argument("--no-gpu", action="store_true", help="Skip GPU benchmarks")
    run_parser.add_argument("--no-cutlass", action="store_true", help="Skip Tensor Core benchmarks")
    run_parser.add_argument("--no-sparse", action="store_true", help="Skip sparse benchmarks")
    run_parser.add_argument("--sparse-backend", choices=["cpu", "gpu"], default="gpu", help="Sparse backend")
    
    # plot subcommand
    plot_parser = subparsers.add_parser("plot", help="Generate plots from results")
    plot_parser.add_argument("--csv", default="benchmarks.csv", help="CSV file to plot")
    
    args = parser.parse_args()
    
    if args.command == "run":
        # Import test_all for running benchmarks
        from tests.test_all import test_all
        import config
        
        sizes = None
        if args.quick:
            sizes = config.QUICK_SIZES
        elif args.medium:
            sizes = config.MEDIUM_SIZES
        elif args.large:
            sizes = config.ALL_SIZES
        
        # Override CPU threads if specified
        if args.threads:
            import config as cfg
            cfg.DEFAULT_CPU_THREADS = args.threads
            cfg.CPU_THREADS_CONFIG = args.threads
            print(f"Using {args.threads} CPU threads (override)")
        
        # Override size limits if --force specified
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
            sparse_backend=args.sparse_backend
        )
        
    elif args.command == "plot":
        from bench.plotter import generate_all_plots
        generate_all_plots(args.csv)
        
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
