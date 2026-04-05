"""Visualization module - generates performance comparison graphs."""

import csv
from pathlib import Path
from typing import Optional, List, Dict
import numpy as np


try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


RESULTS_DIR = Path(__file__).parent.parent / "results"

# Create results directory if it doesn't exist
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def load_results(csv_file: str = "benchmarks.csv") -> list[dict]:
    """Load benchmark results from CSV file."""
    filepath = RESULTS_DIR / csv_file
    if not filepath.exists():
        return []
    
    with open(filepath, "r") as f:
        reader = csv.DictReader(f)
        return list(reader)


def get_module_category(algorithm: str) -> str:
    """Categorize algorithms for grouping."""
    if 'tensor_core' in algorithm or 'matmul' in algorithm:
        return 'Tensor Core'
    elif 'cublas' in algorithm or 'cuda' in algorithm:
        return 'GPU'
    elif 'openmp' in algorithm or 'thread' in algorithm or any(x in algorithm for x in ['sse', 'avx', 'naive', 'blocked', 'cache']):
        return 'CPU'
    elif 'sparse' in algorithm:
        return 'Sparse'
    else:
        return 'Reference'


COLORS = {
    'Tensor Core': '#E63946',  # Red
    'GPU': '#457B9D',         # Blue
    'CPU': '#2A9D8F',         # Teal
    'Sparse': '#E9C46A',      # Yellow
    'Reference': '#8D99AE',   # Gray
}


# Enhanced color palette - unique colors per algorithm
ALGORITHM_COLORS = {
    # CPU Single-thread
    'CPU-Naive-Cpp': '#E63946',      # Red
    'CPU-Naive-Numba-Python': '#FF6B6B',  # Light Red
    'CPU-Naive-NumPy-Python': '#FF8FA3',  # Pink
    'CPU-SIMD-SSE-Cpp': '#2A9D8F',   # Teal
    'CPU-SIMD-AVX2-Cpp': '#264653',  # Dark Teal
    'CPU-SIMD-AVX512-Cpp': '#A8DADC', # Light Teal
    'CPU-Blocked-Cpp': '#457B9D',    # Blue
    'CPU-Parallel-OpenMP-Cpp': '#F4A261',  # Orange
    'CPU-Parallel-OpenMP-Cpp-Blocked': '#E76F51',  # Burnt Orange
    'CPU-Parallel-StdThread-Cpp': '#E9C46A',  # Yellow
    # GPU
    'GPU-CUDA-Cpp': '#6D597A',      # Purple
    'GPU-CUDA-cuBLAS-Cpp': '#B56576', # Magenta
    'GPU-CUDA-CUTLASS-Cpp': '#C9B1FF', # Light Purple
    # Tensor Core
    'GPU-TensorCore-PyTorch-fp16-Python': '#00F5D4',  # Cyan
    'GPU-TensorCore-PyTorch-bf16-Python': '#00BBF9',  # Light Blue
    # Sparse
    'CPU-Sparse-SciPy-Python': '#9B5DE5',  # Violet
    'GPU-Sparse-PyTorch-Python': '#F15BB5',  # Pink
}


def plot_gflops_by_size(
    results: list[dict],
    output_file: str = "gflops_by_size.png",
    title: str = "GFLOPS by Matrix Size"
) -> Optional[str]:
    """Plot GFLOPS vs matrix size for different implementations."""
    if not MATPLOTLIB_AVAILABLE:
        print("matplotlib not available, skipping plot")
        return None

    modules = {}
    for r in results:
        key = f"{r['module']}.{r['algorithm']}"
        if key not in modules:
            modules[key] = {"sizes": [], "gfops": [], "category": get_module_category(r['algorithm'])}
        modules[key]["sizes"].append(int(r["size"]))
        modules[key]["gfops"].append(float(r["gfops"]))

    fig, ax = plt.subplots(figsize=(14, 10))
    
    for name, data in modules.items():
        sorted_idx = np.argsort(data["sizes"])
        # Use unique color per algorithm, fallback to category color
        color = ALGORITHM_COLORS.get(name, COLORS.get(data["category"], '#333333'))
        ax.plot(
            np.array(data["sizes"])[sorted_idx],
            np.array(data["gfops"])[sorted_idx],
            marker="o",
            label=name,
            color=color,
            linewidth=2.5,
            markersize=8
        )

    ax.set_xlabel("Matrix Size (N)", fontsize=16)
    ax.set_ylabel("GFLOPS", fontsize=16)
    ax.set_title(title, fontsize=18, fontweight='bold', pad=20)
    ax.legend(loc='upper left', fontsize=9, framealpha=0.9, ncol=2)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3, which="both")
    ax.tick_params(axis='both', labelsize=12)

    plt.tight_layout()
    output_path = RESULTS_DIR / output_file
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    return str(output_path)


def plot_bandwidth_by_size(
    results: list[dict],
    output_file: str = "bandwidth_by_size.png",
    title: str = "Memory Bandwidth by Matrix Size"
) -> Optional[str]:
    """Plot memory bandwidth vs matrix size."""
    if not MATPLOTLIB_AVAILABLE:
        return None

    modules = {}
    for r in results:
        key = f"{r['module']}.{r['algorithm']}"
        if key not in modules:
            modules[key] = {"sizes": [], "bandwidth": [], "category": get_module_category(r['algorithm'])}
        modules[key]["sizes"].append(int(r["size"]))
        modules[key]["bandwidth"].append(float(r["bandwidth_gbs"]))

    fig, ax = plt.subplots(figsize=(14, 10))
    for name, data in modules.items():
        sorted_idx = np.argsort(data["sizes"])
        color = ALGORITHM_COLORS.get(name, COLORS.get(data["category"], '#333333'))
        ax.plot(
            np.array(data["sizes"])[sorted_idx],
            np.array(data["bandwidth"])[sorted_idx],
            marker="s",
            label=name,
            color=color,
            linewidth=2.5,
            markersize=8
        )

    # Add theoretical bandwidth reference lines
    ax.axhline(y=500, color='#DC143C', linestyle='--', linewidth=2, alpha=0.7,
               label='RTX 5070 Peak (~500 GB/s)')
    ax.axhline(y=50, color='#FF8C00', linestyle='--', linewidth=2, alpha=0.7,
               label='CPU DDR5 per channel (~50 GB/s)')

    ax.set_xlabel("Matrix Size (N)", fontsize=16)
    ax.set_ylabel("Bandwidth (GB/s)", fontsize=16)
    ax.set_title(title, fontsize=18, fontweight='bold', pad=20)
    ax.legend(loc='upper left', fontsize=9, framealpha=0.9, ncol=2)
    ax.set_xscale("log")
    ax.grid(True, alpha=0.3, which="both")
    ax.tick_params(axis='both', labelsize=12)

    plt.tight_layout()
    output_path = RESULTS_DIR / output_file
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    return str(output_path)


def plot_timing_linear(
    results: list[dict],
    output_file: str = "timing_linear.png",
    title: str = "Execution Time by Matrix Size"
) -> Optional[str]:
    """Plot execution time vs matrix size on linear scale (ALL algorithms)."""
    if not MATPLOTLIB_AVAILABLE:
        return None

    modules = {}
    for r in results:
        key = f"{r['module']}.{r['algorithm']}"
        if key not in modules:
            modules[key] = {"sizes": [], "time_ms": [], "category": get_module_category(r['algorithm'])}
        modules[key]["sizes"].append(int(r["size"]))
        modules[key]["time_ms"].append(float(r["time_ms"]))

    fig, ax = plt.subplots(figsize=(14, 10))
    
    for name, data in modules.items():
        sorted_idx = np.argsort(data["sizes"])
        color = ALGORITHM_COLORS.get(name, COLORS.get(data["category"], '#333333'))
        ax.plot(
            np.array(data["sizes"])[sorted_idx],
            np.array(data["time_ms"])[sorted_idx],
            marker="o",
            label=name,
            color=color,
            linewidth=2.5,
            markersize=8
        )

    ax.set_xlabel("Matrix Size (N)", fontsize=16)
    ax.set_ylabel("Time (ms)", fontsize=16)
    ax.set_title(title, fontsize=18, fontweight='bold', pad=20)
    ax.legend(loc='upper left', fontsize=9, framealpha=0.9, ncol=2)
    ax.set_xscale("linear")
    ax.set_yscale("linear")
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis='both', labelsize=12)
    
    plt.tight_layout()
    output_path = RESULTS_DIR / output_file
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    return str(output_path)


def plot_category_comparison(
    results: list[dict],
    output_file: str = "category_comparison.png",
    title: str = "Performance by Category"
) -> Optional[str]:
    """Plot comparison by category (CPU, GPU, Tensor Core, Sparse)."""
    if not MATPLOTLIB_AVAILABLE:
        return None

    # Group by category and size
    category_data = {}
    for r in results:
        category = get_module_category(r['algorithm'])
        size = int(r["size"])
        gflops = float(r["gfops"])
        
        if category not in category_data:
            category_data[category] = {"sizes": set(), "gfops": []}
        category_data[category]["sizes"].add(size)
        category_data[category]["gfops"].append((size, gflops))

    # Find common sizes
    all_sizes = set()
    for cat_data in category_data.values():
        all_sizes |= cat_data["sizes"]
    common_sizes = sorted(all_sizes)

    # Average GFLOPS per category at each size
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # GFLOPS by category
    ax1 = axes[0]
    for category, data in category_data.items():
        size_gflops = {}
        for size, gflops in data["gfops"]:
            if size not in size_gflops:
                size_gflops[size] = []
            size_gflops[size].append(gflops)
        
        sizes = sorted(size_gflops.keys())
        avg_gflops = [np.mean(size_gflops[s]) for s in sizes]
        ax1.plot(sizes, avg_gflops, marker='o', label=category, 
                 color=COLORS.get(category, '#333333'), linewidth=2)
    
    ax1.set_xlabel("Matrix Size (N)", fontsize=12)
    ax1.set_ylabel("GFLOPS", fontsize=12)
    ax1.set_title("Performance by Category", fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.grid(True, alpha=0.3)

    # Bar chart for largest size
    ax2 = axes[1]
    largest_size = max(all_sizes)
    category_max = {}
    for category, data in category_data.items():
        max_gflops = max([g for s, g in data["gfops"] if s == largest_size], default=0)
        category_max[category] = max_gflops
    
    cats = list(category_max.keys())
    values = list(category_max.values())
    colors = [COLORS.get(c, '#333333') for c in cats]
    ax2.bar(cats, values, color=colors)
    ax2.set_xlabel("Category", fontsize=12)
    ax2.set_ylabel("GFLOPS", fontsize=12)
    ax2.set_title(f"Peak Performance (N={largest_size})", fontsize=14, fontweight='bold')
    ax2.set_yscale("log")
    
    plt.tight_layout()
    output_path = RESULTS_DIR / output_file
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    return str(output_path)


def plot_sparse_comparison(
    results: list[dict],
    output_file: str = "sparse_comparison.png",
    title: str = "Sparse Matrix Performance"
) -> Optional[str]:
    """Plot sparse matrix benchmark results."""
    if not MATPLOTLIB_AVAILABLE:
        return None

    sparse_results = [r for r in results if 'sparse' in r.get('module', '').lower() or 'sparse' in r.get('algorithm', '').lower()]
    if not sparse_results:
        return None

    formats = {}
    for r in sparse_results:
        fmt = r['algorithm']
        if fmt not in formats:
            formats[fmt] = {"sizes": [], "gfops": []}
        formats[fmt]["sizes"].append(int(r["size"]))
        formats[fmt]["gfops"].append(float(r["gfops"]))

    fig, ax = plt.subplots(figsize=(12, 8))
    for fmt, data in formats.items():
        sorted_idx = np.argsort(data["sizes"])
        ax.plot(
            np.array(data["sizes"])[sorted_idx],
            np.array(data["gfops"])[sorted_idx],
            marker='o',
            label=fmt,
            linewidth=2.5,
            markersize=8
        )

    ax.set_xlabel("Matrix Size (N)", fontsize=14)
    ax.set_ylabel("GFLOPS", fontsize=14)
    ax.set_title(title, fontsize=16, fontweight='bold', pad=15)
    ax.legend(loc='upper left', fontsize=10, framealpha=0.9)
    ax.set_xscale("log")
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis='both', labelsize=11)

    plt.tight_layout()
    output_path = RESULTS_DIR / output_file
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    return str(output_path)


def plot_tensor_core_comparison(
    results: list[dict],
    output_file: str = "tensor_core_comparison.png",
    title: str = "Tensor Core: FP16 vs BF16"
) -> Optional[str]:
    """Compare FP16 vs BF16 Tensor Core performance."""
    if not MATPLOTLIB_AVAILABLE:
        return None

    tensor_core_results = [r for r in results if 'tensor_core' in r.get('algorithm', '').lower()]
    if not tensor_core_results:
        return None

    fp16_data = {"sizes": [], "gfops": []}
    bf16_data = {"sizes": [], "gfops": []}

    for r in tensor_core_results:
        dtype = r.get('dtype', '')
        if 'fp16' in dtype.lower():
            fp16_data["sizes"].append(int(r["size"]))
            fp16_data["gfops"].append(float(r["gfops"]))
        elif 'bf16' in dtype.lower():
            bf16_data["sizes"].append(int(r["size"]))
            bf16_data["gfops"].append(float(r["gfops"]))

    fig, ax = plt.subplots(figsize=(12, 8))
    
    if fp16_data["sizes"]:
        sorted_idx = np.argsort(fp16_data["sizes"])
        ax.plot(
            np.array(fp16_data["sizes"])[sorted_idx],
            np.array(fp16_data["gfops"])[sorted_idx],
            marker='o',
            label='FP16',
            linewidth=2.5,
            color='#00F5D4'
        )
    
    if bf16_data["sizes"]:
        sorted_idx = np.argsort(bf16_data["sizes"])
        ax.plot(
            np.array(bf16_data["sizes"])[sorted_idx],
            np.array(bf16_data["gfops"])[sorted_idx],
            marker='s',
            label='BF16',
            linewidth=2.5,
            color='#00BBF9'
        )

    ax.set_xlabel("Matrix Size (N)", fontsize=14)
    ax.set_ylabel("GFLOPS", fontsize=14)
    ax.set_title(title, fontsize=16, fontweight='bold', pad=15)
    ax.legend(loc='upper left', fontsize=11, framealpha=0.9)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis='both', labelsize=11)

    plt.tight_layout()
    output_path = RESULTS_DIR / output_file
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    return str(output_path)


def generate_all_plots(csv_file: str = "benchmarks.csv"):
    """Generate all standard plots from benchmark results."""
    results = load_results(csv_file)
    if not results:
        print(f"No results found in {csv_file}")
        return

    print("Generating plots...")
    
    plot_gflops_by_size(results)
    print("  - gflops_by_size.png")
    
    plot_timing_linear(results)
    print("  - timing_linear.png")
    
    plot_bandwidth_by_size(results)
    print("  - bandwidth_by_size.png")
    
    plot_category_comparison(results)
    print("  - category_comparison.png")
    
    plot_sparse_comparison(results)
    print("  - sparse_comparison.png")
    
    plot_tensor_core_comparison(results)
    print("  - tensor_core_comparison.png")
    
    print("All plots generated successfully")


def plot_summary_table(results: list[dict]) -> str:
    """Generate a summary table of results."""
    if not results:
        return "No results to summarize"
    
    # Get unique categories
    categories = set()
    for r in results:
        categories.add(get_module_category(r['algorithm']))
    
    # Find max GFLOPS per category
    summary = []
    for cat in sorted(categories):
        cat_results = [r for r in results if get_module_category(r['algorithm']) == cat]
        if cat_results:
            max_result = max(cat_results, key=lambda x: float(x['gfops']))
            summary.append({
                'Category': cat,
                'Best Algorithm': max_result['algorithm'],
                'Size': max_result['size'],
                'GFLOPS': f"{float(max_result['gfops']):.2f}",
                'Bandwidth (GB/s)': f"{float(max_result['bandwidth_gbs']):.2f}"
            })
    
    return summary


if __name__ == "__main__":
    generate_all_plots()
