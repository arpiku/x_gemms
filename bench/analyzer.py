"""Benchmark results analyzer module with Plotly visualization."""

import csv
import json
import os
import socket
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Union

import pandas as pd

try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

RESULTS_DIR = Path(__file__).parent.parent / "results"


class BenchmarkAnalyzer:
    """Analyze and visualize benchmark results."""

    def __init__(self, results: Optional[pd.DataFrame] = None):
        self.results = results
        self.metadata = {}

    @classmethod
    def from_csv(cls, path: Union[str, Path]) -> 'BenchmarkAnalyzer':
        """Load benchmark results from a single CSV file."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"CSV file not found: {path}")
        
        instance = cls()
        instance._load_csv(path)
        return instance

    @classmethod
    def from_multiple(cls, paths: List[Union[str, Path]]) -> 'BenchmarkAnalyzer':
        """Load and merge multiple CSV files."""
        instance = cls()
        dfs = []
        for p in paths:
            p = Path(p)
            if p.exists():
                df = pd.read_csv(p, comment='#')
                df['source_file'] = p.name
                dfs.append(df)
        
        if dfs:
            instance.results = pd.concat(dfs, ignore_index=True)
            instance._extract_metadata()
        return instance

    @classmethod
    def from_latest(cls, results_dir: Path = RESULTS_DIR) -> 'BenchmarkAnalyzer':
        """Load the most recent CSV file from results directory."""
        csv_files = list(results_dir.glob("benchmark_*.csv"))
        if not csv_files:
            raise FileNotFoundError(f"No benchmark CSV files found in {results_dir}")
        
        latest = max(csv_files, key=lambda f: f.stat().st_mtime)
        return cls.from_csv(latest)

    def _load_csv(self, path: Path) -> None:
        """Load CSV file with metadata extraction."""
        self.metadata = {'source_file': path.name}
        
        with open(path, 'r') as f:
            for line in f:
                if line.startswith('# timestamp:'):
                    self.metadata['timestamp'] = line.split(':', 1)[1].strip()
                elif line.startswith('# hostname:'):
                    self.metadata['hostname'] = line.split(':', 1)[1].strip()
                elif line.startswith('# cuda:'):
                    self.metadata['cuda'] = line.split(':', 1)[1].strip()
                elif line.startswith('# gpu:'):
                    self.metadata['gpu'] = line.split(':', 1)[1].strip()
                elif not line.startswith('#'):
                    break
        
        self.results = pd.read_csv(path, comment='#')

    def _extract_metadata(self) -> None:
        """Extract metadata from merged results."""
        self.metadata = {}
        if self.results is not None and 'source_file' in self.results.columns:
            files = self.results['source_file'].unique()
            if len(files) == 1:
                self.metadata['source_file'] = files[0]

    def filter(
        self,
        algorithms: Optional[List[str]] = None,
        sizes: Optional[tuple[int, int]] = None,
        dtypes: Optional[List[str]] = None,
        source_files: Optional[List[str]] = None
    ) -> 'BenchmarkAnalyzer':
        """Filter results (chainable)."""
        if self.results is None:
            return BenchmarkAnalyzer()
        
        df = self.results.copy()
        
        if algorithms:
            df = df[df['algorithm'].isin(algorithms)]
        
        if sizes:
            min_size, max_size = sizes
            df = df[(df['size'] >= min_size) & (df['size'] <= max_size)]
        
        if dtypes:
            df = df[df['dtype'].isin(dtypes)]
        
        if source_files and 'source_file' in df.columns:
            df = df[df['source_file'].isin(source_files)]
        
        result = BenchmarkAnalyzer(df)
        result.metadata = self.metadata.copy()
        return result

    def get_algorithms(self) -> List[str]:
        """Get list of unique algorithms in results."""
        if self.results is None:
            return []
        return sorted(self.results['algorithm'].unique().tolist())

    def get_sizes(self) -> List[int]:
        """Get list of unique sizes in results."""
        if self.results is None:
            return []
        return sorted(self.results['size'].unique().tolist())

    def get_summary(self) -> pd.DataFrame:
        """Get summary statistics per algorithm."""
        if self.results is None:
            return pd.DataFrame()
        
        return self.results.groupby('algorithm').agg({
            'gflops': ['mean', 'max', 'min', 'std'],
            'time_ms': ['mean', 'min'],
            'size': ['min', 'max', 'count']
        }).round(2)

    def get_best_performers(self, metric: str = 'gflops') -> pd.DataFrame:
        """Get best performer at each size."""
        if self.results is None:
            return pd.DataFrame()
        
        idx = self.results.groupby('size')[metric].idxmax()
        return self.results.loc[idx][['size', 'algorithm', metric, 'time_ms']]

    def get_speedup(self, baseline: str, metric: str = 'gflops') -> pd.DataFrame:
        """Calculate speedup relative to baseline algorithm."""
        if self.results is None:
            return pd.DataFrame()
        
        baseline_df = self.results[self.results['algorithm'] == baseline].copy()
        if baseline_df.empty:
            raise ValueError(f"Baseline algorithm '{baseline}' not found in results")
        
        baseline_df = baseline_df[['size', metric]].rename(columns={metric: f'{metric}_baseline'})
        
        merged = self.results.merge(baseline_df, on='size')
        merged['speedup'] = merged[metric] / merged[f'{metric}_baseline']
        
        return merged[['algorithm', 'size', metric, 'speedup']].sort_values(['algorithm', 'size'])

    def plot_gflops(self, log_scale: bool = True, title: Optional[str] = None) -> go.Figure:
        """Plot GFLOPS vs matrix size."""
        if not PLOTLY_AVAILABLE:
            raise ImportError("Plotly is required for plotting. Install with: pip install plotly")
        
        if self.results is None or self.results.empty:
            return go.Figure()
        
        fig = go.Figure()
        
        for algo in self.get_algorithms():
            df = self.results[self.results['algorithm'] == algo].sort_values('size')
            fig.add_trace(go.Scatter(
                x=df['size'],
                y=df['gflops'],
                mode='lines+markers',
                name=algo,
                hovertemplate=f'<b>{algo}</b><br>Size: %{{x}}<br>GFLOPS: %{{y:.2f}}<extra></extra>'
            ))
        
        title = title or "GFLOPS by Matrix Size"
        fig.update_layout(
            title=title,
            xaxis_title="Matrix Size (N)",
            yaxis_title="GFLOPS",
            xaxis_type="log" if log_scale else "linear",
            yaxis_type="log" if log_scale else "linear",
            hovermode='closest',
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
            template="plotly_white"
        )
        return fig

    def plot_bandwidth(self, title: Optional[str] = None) -> go.Figure:
        """Plot memory bandwidth vs matrix size."""
        if not PLOTLY_AVAILABLE:
            raise ImportError("Plotly is required for plotting")
        
        if self.results is None or self.results.empty:
            return go.Figure()
        
        fig = go.Figure()
        
        for algo in self.get_algorithms():
            df = self.results[self.results['algorithm'] == algo].sort_values('size')
            fig.add_trace(go.Scatter(
                x=df['size'],
                y=df['bandwidth_gbs'],
                mode='lines+markers',
                name=algo,
                hovertemplate=f'<b>{algo}</b><br>Size: %{{x}}<br>Bandwidth: %{{y:.2f}} GB/s<extra></extra>'
            ))
        
        title = title or "Memory Bandwidth by Matrix Size"
        fig.update_layout(
            title=title,
            xaxis_title="Matrix Size (N)",
            yaxis_title="Bandwidth (GB/s)",
            xaxis_type="log",
            hovermode='closest',
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
            template="plotly_white"
        )
        return fig

    def plot_comparison(
        self,
        algorithms: List[str],
        metric: str = 'gflops',
        title: Optional[str] = None
    ) -> go.Figure:
        """Compare specific algorithms."""
        if not PLOTLY_AVAILABLE:
            raise ImportError("Plotly is required for plotting")
        
        filtered = self.filter(algorithms=algorithms)
        if filtered.results is None or filtered.results.empty:
            return go.Figure()
        
        fig = go.Figure()
        
        for algo in algorithms:
            df = filtered.results[filtered.results['algorithm'] == algo].sort_values('size')
            fig.add_trace(go.Scatter(
                x=df['size'],
                y=df[metric],
                mode='lines+markers',
                name=algo,
                line=dict(width=3),
                marker=dict(size=10)
            ))
        
        title = title or f"Comparison: {metric}"
        fig.update_layout(
            title=title,
            xaxis_title="Matrix Size (N)",
            yaxis_title=metric.upper() if metric != 'time_ms' else "Time (ms)",
            xaxis_type="log",
            yaxis_type="log",
            hovermode='closest',
            template="plotly_white"
        )
        return fig

    def plot_speedup(self, baseline: str, title: Optional[str] = None) -> go.Figure:
        """Plot speedup relative to baseline."""
        if not PLOTLY_AVAILABLE:
            raise ImportError("Plotly is required for plotting")
        
        speedup_df = self.get_speedup(baseline)
        if speedup_df.empty:
            return go.Figure()
        
        fig = go.Figure()
        
        for algo in speedup_df['algorithm'].unique():
            if algo == baseline:
                continue
            df = speedup_df[speedup_df['algorithm'] == algo].sort_values('size')
            fig.add_trace(go.Scatter(
                x=df['size'],
                y=df['speedup'],
                mode='lines+markers',
                name=f"{algo} vs {baseline}",
                hovertemplate=f'<b>{algo}</b><br>Size: %{{x}}<br>Speedup: %{{y:.2f}}x<extra></extra>'
            ))
        
        fig.add_hline(y=1, line_dash="dash", line_color="gray", 
                      annotation_text=f"Baseline: {baseline}")
        
        title = title or f"Speedup vs {baseline}"
        fig.update_layout(
            title=title,
            xaxis_title="Matrix Size (N)",
            yaxis_title="Speedup (x)",
            xaxis_type="log",
            hovermode='closest',
            template="plotly_white"
        )
        return fig

    def plot_heatmap(self, metric: str = 'gflops', title: Optional[str] = None) -> go.Figure:
        """Heatmap of algorithm performance across sizes."""
        if not PLOTLY_AVAILABLE:
            raise ImportError("Plotly is required for plotting")
        
        if self.results is None or self.results.empty:
            return go.Figure()
        
        pivot = self.results.pivot_table(
            values=metric,
            index='algorithm',
            columns='size',
            aggfunc='max'
        )
        
        title = title or f"{metric} Heatmap"
        fig = go.Figure(data=go.Heatmap(
            z=pivot.values,
            x=pivot.columns,
            y=pivot.index,
            colorscale='Viridis',
            hovertemplate='Algorithm: %{y}<br>Size: %{x}<br>' + f'{metric}: %{{z:.2f}}<extra></extra>'
        ))
        
        fig.update_layout(
            title=title,
            xaxis_title="Matrix Size (N)",
            yaxis_title="Algorithm",
            template="plotly_white"
        )
        return fig

    def plot_timing_comparison(
        self,
        algorithms: Optional[List[str]] = None,
        sizes: Optional[List[int]] = None,
        title: Optional[str] = None
    ) -> go.Figure:
        """Plot execution time comparison as grouped bar chart."""
        if not PLOTLY_AVAILABLE:
            raise ImportError("Plotly is required for plotting")
        
        if self.results is None or self.results.empty:
            return go.Figure()
        
        df = self.results.copy()
        
        if algorithms:
            df = df[df['algorithm'].isin(algorithms)]
        if sizes:
            df = df[df['size'].isin(sizes)]
        
        df = df.sort_values(['size', 'algorithm'])
        
        title = title or "Execution Time Comparison"
        fig = go.Figure()
        
        unique_algos = df['algorithm'].unique()
        colors = px.colors.qualitative.Plotly
        
        for i, algo in enumerate(unique_algos):
            algo_df = df[df['algorithm'] == algo]
            fig.add_trace(go.Bar(
                x=algo_df['size'].astype(str),
                y=algo_df['time_ms'],
                name=algo,
                marker_color=colors[i % len(colors)],
                hovertemplate=f'<b>{algo}</b><br>Size: %{{x}}<br>Time: %{{y:.2f}} ms<extra></extra>'
            ))
        
        fig.update_layout(
            title=title,
            xaxis_title="Matrix Size (N)",
            yaxis_title="Time (ms)",
            barmode='group',
            hovermode='closest',
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
            template="plotly_white"
        )
        return fig

    def plot_timing_log(
        self,
        algorithms: Optional[List[str]] = None,
        title: Optional[str] = None
    ) -> go.Figure:
        """Plot execution time on log scale (better for wide range)."""
        if not PLOTLY_AVAILABLE:
            raise ImportError("Plotly is required for plotting")
        
        if self.results is None or self.results.empty:
            return go.Figure()
        
        df = self.results.copy()
        
        if algorithms:
            df = df[df['algorithm'].isin(algorithms)]
        
        df = df.sort_values(['size', 'algorithm'])
        
        title = title or "Execution Time (Log Scale)"
        fig = go.Figure()
        
        for algo in df['algorithm'].unique():
            algo_df = df[df['algorithm'] == algo]
            fig.add_trace(go.Scatter(
                x=algo_df['size'],
                y=algo_df['time_ms'],
                mode='lines+markers',
                name=algo,
                hovertemplate=f'<b>{algo}</b><br>Size: %{{x}}<br>Time: %{{y:.2f}} ms<extra></extra>'
            ))
        
        fig.update_layout(
            title=title,
            xaxis_title="Matrix Size (N)",
            yaxis_title="Time (ms)",
            xaxis_type="log",
            yaxis_type="log",
            hovermode='closest',
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
            template="plotly_white"
        )
        return fig

    def export_plot(self, fig: go.Figure, filename: str, format: str = 'png') -> Path:
        """Export plot to file."""
        output_path = RESULTS_DIR / filename
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if format == 'html':
            fig.write_html(str(output_path))
        else:
            fig.write_image(str(output_path), scale=2)
        
        return output_path

    def export_filtered_csv(self, filename: str) -> Path:
        """Export filtered results to CSV."""
        if self.results is None:
            raise ValueError("No results to export")
        
        output_path = RESULTS_DIR / filename
        output_path.parent.mkdir(parents=True, exist_ok=True)
        self.results.to_csv(output_path, index=False)
        return output_path


def save_results_with_timestamp(
    results: List[dict],
    tag: Optional[str] = None,
    results_dir: Path = RESULTS_DIR
) -> Path:
    """Save benchmark results with timestamp in filename."""
    results_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    
    if tag:
        filename = f"benchmark_{timestamp}_{tag}.csv"
    else:
        filename = f"benchmark_{timestamp}.csv"
    
    output_path = results_dir / filename
    
    hostname = socket.gethostname()
    
    cuda_version = "N/A"
    gpu_name = "N/A"
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=driver_version,name', '--format=csv,noheader'],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            parts = result.stdout.strip().split('\n')
            if parts:
                gpu_info = parts[0].split(',')
                if len(gpu_info) >= 2:
                    gpu_name = gpu_info[1].strip()
    except Exception:
        pass
    
    try:
        result = subprocess.run(
            ['nvcc', '--version'],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            for line in result.stdout.split('\n'):
                if 'release' in line:
                    cuda_version = line.split('release')[-1].strip().split(',')[0].strip()
                    break
    except Exception:
        pass
    
    fieldnames = ['module', 'algorithm', 'size', 'dtype', 'time_ms', 'gflops', 'bandwidth_gbs', 'sparsity']
    
    with open(output_path, 'w', newline='') as f:
        f.write(f"# x_gemms benchmark results\n")
        f.write(f"# timestamp: {datetime.now().isoformat()}\n")
        f.write(f"# hostname: {hostname}\n")
        f.write(f"# cuda: {cuda_version}\n")
        f.write(f"# gpu: {gpu_name}\n")
        
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(results)
    
    return output_path


def launch_notebook(csv_files: Optional[List[str]] = None) -> None:
    """Launch Jupyter notebook for interactive analysis."""
    notebooks_dir = Path(__file__).parent / "notebooks"
    notebooks_dir.mkdir(parents=True, exist_ok=True)
    
    template_path = notebooks_dir / "analyze.ipynb"
    if not template_path.exists():
        _create_default_notebook(template_path)
    
    env = os.environ.copy()
    if csv_files:
        abs_paths = [str(Path(f).resolve()) for f in csv_files]
        env['X_GEMMS_CSV_FILES'] = ','.join(abs_paths)
    
    try:
        subprocess.run(['jupyter', 'notebook', str(template_path)], env=env)
    except FileNotFoundError:
        print("Jupyter not found. Install with: pip install jupyter")
        print(f"Notebook template located at: {template_path}")


def _create_default_notebook(path: Path) -> None:
    """Create default analysis notebook template."""
    notebook_content = {
        "cells": [
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "import os\n",
                    "import sys\n",
                    "from pathlib import Path\n",
                    "\n",
                    "# Add project root to path (notebook is in bench/notebooks/)\n",
                    "project_root = Path.cwd().parent.parent if 'notebooks' in str(Path.cwd()) else Path.cwd().parent\n",
                    "sys.path.insert(0, str(project_root))\n",
                    "\n",
                    "from bench.analyzer import BenchmarkAnalyzer, RESULTS_DIR\n",
                    "import plotly.io as pio\n",
                    "pio.renderers.default = 'notebook'"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Load data - modify path as needed\n",
                    "csv_files = os.environ.get('X_GEMMS_CSV_FILES', '')\n",
                    "if csv_files:\n",
                    "    files = csv_files.split(',')\n",
                    "    analyzer = BenchmarkAnalyzer.from_multiple(files)\n",
                    "    print(f\"Loaded files: {files}\")\n",
                    "else:\n",
                    "    # Load latest by default\n",
                    "    analyzer = BenchmarkAnalyzer.from_latest()\n",
                    "    print(f\"Loaded latest results\")\n",
                    "\n",
                    "print(f\"\\nLoaded {len(analyzer.results)} results\")\n",
                    "print(f\"Algorithms: {analyzer.get_algorithms()}\")\n",
                    "print(f\"Sizes: {analyzer.get_sizes()}\")"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Summary statistics\n",
                    "analyzer.get_summary()"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Best performers at each size\n",
                    "analyzer.get_best_performers()"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# GFLOPS plot\n",
                    "fig = analyzer.plot_gflops()\n",
                    "fig.show()"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Compare specific algorithms\n",
                    "algorithms = ['cublas', 'strassen_cublas', 'winograd_cublas']\n",
                    "fig = analyzer.plot_comparison(algorithms)\n",
                    "fig.show()"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Speedup relative to baseline\n",
                    "baseline = 'cublas'\n",
                    "fig = analyzer.plot_speedup(baseline)\n",
                    "fig.show()"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Execution time comparison (bar chart)\n",
                    "fig = analyzer.plot_timing_comparison()\n",
                    "fig.show()"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Execution time comparison (log scale)\n",
                    "fig = analyzer.plot_timing_log()\n",
                    "fig.show()"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Heatmap view\n",
                    "fig = analyzer.plot_heatmap(metric='gflops')\n",
                    "fig.show()"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Filter to specific range\n",
                    "filtered = analyzer.filter(sizes=(256, 4096), dtypes=['fp32'])\n",
                    "fig = filtered.plot_gflops(log_scale=True)\n",
                    "fig.show()"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Export plot\n",
                    "# analyzer.export_plot(fig, 'my_plot.png')\n",
                    "# analyzer.export_plot(fig, 'my_plot.html', format='html')"
                ]
            }
        ],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python",
                "version": "3.11.0"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }
    
    with open(path, 'w') as f:
        json.dump(notebook_content, f, indent=2)
