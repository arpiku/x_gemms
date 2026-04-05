"""Visualization module - generates performance comparison graphs."""

import csv
from pathlib import Path
from typing import Optional
import numpy as np


try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


RESULTS_DIR = Path(__file__).parent / "results"


def load_results(csv_file: str = "benchmarks.csv") -> list[dict]:
    """Load benchmark results from CSV file."""
    filepath = RESULTS_DIR / csv_file
    if not filepath.exists():
        return []
    
    with open(filepath, "r") as f:
        reader = csv.DictReader(f)
        return list(reader)


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
            modules[key] = {"sizes": [], "gfops": []}
        modules[key]["sizes"].append(int(r["size"]))
        modules[key]["gfops"].append(float(r["gfops"]))

    fig, ax = plt.subplots(figsize=(10, 6))
    for name, data in modules.items():
        sorted_idx = np.argsort(data["sizes"])
        ax.plot(
            np.array(data["sizes"])[sorted_idx],
            np.array(data["gfops"])[sorted_idx],
            marker="o",
            label=name
        )

    ax.set_xlabel("Matrix Size (N)")
    ax.set_ylabel("GFLOPS")
    ax.set_title(title)
    ax.legend()
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)

    output_path = RESULTS_DIR / output_file
    plt.savefig(output_path)
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
            modules[key] = {"sizes": [], "bandwidth": []}
        modules[key]["sizes"].append(int(r["size"]))
        modules[key]["bandwidth"].append(float(r["bandwidth_gbs"]))

    fig, ax = plt.subplots(figsize=(10, 6))
    for name, data in modules.items():
        sorted_idx = np.argsort(data["sizes"])
        ax.plot(
            np.array(data["sizes"])[sorted_idx],
            np.array(data["bandwidth"])[sorted_idx],
            marker="s",
            label=name
        )

    ax.set_xlabel("Matrix Size (N)")
    ax.set_ylabel("Bandwidth (GB/s)")
    ax.set_title(title)
    ax.legend()
    ax.set_xscale("log")
    ax.grid(True, alpha=0.3)

    output_path = RESULTS_DIR / output_file
    plt.savefig(output_path)
    plt.close()
    return str(output_path)


def generate_all_plots(csv_file: str = "benchmarks.csv"):
    """Generate all standard plots from benchmark results."""
    results = load_results(csv_file)
    if not results:
        print(f"No results found in {csv_file}")
        return

    plot_gflops_by_size(results)
    plot_bandwidth_by_size(results)
    print("Plots generated successfully")
