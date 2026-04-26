# graph_results.py — generates benchmark graphs from benchmark_results.csv.
# Graphs: execution time vs size, speedup vs threads, parallel efficiency,
#         memory usage, comparison count, and sequential head-to-head comparison.

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import numpy as np
from scipy.optimize import curve_fit
import argparse
import os
import json
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

PALETTE_DATASETS = "tab10"
PALETTE_SIZES    = "Set1"
ALGO_COLORS = {
    "quicksort_seq": "#2980b9",
    "quicksort_par": "#3498db",
    "mergesort_seq": "#c0392b",
    "mergesort_par": "#e74c3c",
}


def load_hw_info(csv_file: str) -> str:
    """Loads hardware metadata from the sidecar JSON, returns a display string."""
    hw_path = csv_file.replace(".csv", "_hardware.json")
    if not os.path.exists(hw_path):
        return ""
    with open(hw_path) as f:
        hw = json.load(f)
    return (
        f"{hw.get('cpu_model','?')}  |  "
        f"{hw.get('physical_cores','?')} physical / {hw.get('logical_cores','?')} logical cores  |  "
        f"{hw.get('cache_info','?')}"
    )


def load_and_clean(csv_file: str) -> pd.DataFrame:
    """Loads the CSV and drops rows with zero execution time (sub-ms timer artefact)."""
    df = pd.read_csv(csv_file)
    for col in ['execution_time_ms', 'memory_usage_kb', 'comparison_count', 'input_size', 'threads']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    n_before = len(df)
    df = df[df['execution_time_ms'] > 0].copy()
    n_dropped = n_before - len(df)
    if n_dropped:
        print(f"  Dropped {n_dropped} zero-time rows (sub-ms timer artefact).")
    return df


def aggregate(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregates repeated runs into median ± IQR per configuration. Flags outlier groups."""
    grp = ['algorithm', 'dataset_type', 'input_size', 'threads']
    agg = (df.groupby(grp)
             .agg(
                 time_median=('execution_time_ms', 'median'),
                 time_q25=('execution_time_ms', lambda x: x.quantile(0.25)),
                 time_q75=('execution_time_ms', lambda x: x.quantile(0.75)),
                 memory_median=('memory_usage_kb', 'median'),
                 comp_median=('comparison_count', 'median'),
                 n_runs=('execution_time_ms', 'count')
             )
             .reset_index())
    agg['time_err_lo'] = agg['time_median'] - agg['time_q25']
    agg['time_err_hi'] = agg['time_q75']  - agg['time_median']
    group_max = df.groupby(grp)['execution_time_ms'].max().reset_index()
    group_max.columns = grp + ['time_max']
    agg = agg.merge(group_max, on=grp)
    agg['has_outlier'] = agg['time_max'] > 3 * agg['time_median']  # flag runs >3x median
    return agg


def plot_time_vs_size(agg: pd.DataFrame, output_dir: str, hw_label: str = "") -> None:
    """Plots execution time vs input size, faceted by dataset type, for each algorithm."""
    print("Generating Execution Time vs Input Size (faceted by dataset type)...")
    dataset_types = sorted(agg['dataset_type'].unique())
    n_types = len(dataset_types)

    for algo in sorted(agg['algorithm'].unique()):
        is_seq = algo.endswith('_seq')
        subset = agg[agg['algorithm'] == algo]
        thread_vals = sorted(subset['threads'].unique())

        ncols = min(3, n_types)
        nrows = (n_types + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4.5 * nrows),
                                 sharex=False, sharey=False)
        axes = np.array(axes).flatten()

        palette = sns.color_palette(PALETTE_DATASETS, len(thread_vals))
        color_map = dict(zip(thread_vals, palette))

        for ax, dtype in zip(axes, dataset_types):
            sub = subset[subset['dataset_type'] == dtype].sort_values('input_size')
            for t in thread_vals:
                tsub = sub[sub['threads'] == t].copy()
                if tsub.empty:
                    continue
                label = "sequential" if is_seq else f"{t} thread{'s' if t>1 else ''}"
                ax.errorbar(
                    tsub['input_size'], tsub['time_median'],
                    yerr=[tsub['time_err_lo'], tsub['time_err_hi']],
                    marker='o', markersize=3, linewidth=1.2,
                    color=color_map[t], label=label, capsize=2
                )
                outliers = tsub[tsub['has_outlier']]
                if not outliers.empty:
                    ax.scatter(outliers['input_size'], outliers['time_median'],
                               marker='*', s=60, color='red', zorder=5,
                               label='outlier run (>3x median)')
                if is_seq:
                    break

            ax.set_xscale('log')
            ax.set_yscale('log')
            ax.set_title(dtype.replace('_', ' '), fontsize=9)
            ax.set_xlabel('Input Size (N)')
            ax.set_ylabel('Time (ms)')
            ax.grid(True, which='both', linestyle='--', alpha=0.35)

        for ax in axes[n_types:]:
            ax.set_visible(False)

        handles, labels = axes[0].get_legend_handles_labels()
        seen = {}
        for h, l in zip(handles, labels):
            if l not in seen:
                seen[l] = h
        fig.legend(seen.values(), seen.keys(), title='Configuration', loc='lower right', fontsize=8)

        subtitle = f"\n{hw_label}" if hw_label else ""
        fig.suptitle(f'{algo}: Execution Time vs Input Size (median ± IQR){subtitle}', fontsize=12)
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        path = os.path.join(output_dir, f'{algo}_time_vs_size.png')
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"  Saved {path}")


def amdahl(p, f_seq):
    """Amdahl's law: S(p) = 1 / (f_seq + (1-f_seq)/p)."""
    return 1.0 / (f_seq + (1.0 - f_seq) / np.array(p, dtype=float))


def fit_amdahl(thread_vals, speedup_vals) -> float:
    """Fits Amdahl's law to observed speedup data; returns best-fit f_seq."""
    if len(thread_vals) < 2:
        return 0.5
    try:
        popt, _ = curve_fit(amdahl, thread_vals, speedup_vals, p0=[0.5], bounds=(1e-6, 1.0 - 1e-6))
        return float(popt[0])
    except Exception:
        return 0.5


def plot_speedup(agg: pd.DataFrame, output_dir: str, hw_label: str = "") -> None:
    """Plots speedup vs thread count with a fitted Amdahl curve for each algorithm."""
    print("Generating Speedup vs Threads (multi-size + fitted Amdahl's law)...")
    base_algos = sorted(set(a.rsplit('_', 1)[0] for a in agg['algorithm'].unique()))

    for base_algo in base_algos:
        par_algo = f"{base_algo}_par"
        seq_algo = f"{base_algo}_seq"
        if par_algo not in agg['algorithm'].values or seq_algo not in agg['algorithm'].values:
            continue

        par_df = agg[agg['algorithm'] == par_algo]
        seq_df = agg[agg['algorithm'] == seq_algo]
        if par_df.empty or seq_df.empty:
            continue

        # Prefer uniform distribution to avoid bias from degenerate types (e.g., same_value).
        chosen_type = None
        for pt in ['uniform', 'gaussian']:
            if pt in par_df['dataset_type'].values and pt in seq_df['dataset_type'].values:
                chosen_type = pt
                break
        if chosen_type is None:
            chosen_type = par_df.groupby('dataset_type')['input_size'].nunique().idxmax()

        par_type = par_df[par_df['dataset_type'] == chosen_type]
        seq_type = seq_df[seq_df['dataset_type'] == chosen_type]

        common_sizes = sorted(set(par_type['input_size'].unique()) & set(seq_type['input_size'].unique()))
        if not common_sizes:
            print(f"  No common sizes for {base_algo} [{chosen_type}]; skipping.")
            continue

        pcts = [0.25, 0.5, 0.75, 1.0]
        target_sizes = sorted(set(common_sizes[max(0, int(p * (len(common_sizes) - 1)))] for p in pcts))
        thread_vals = sorted(par_df['threads'].unique())
        palette = sns.color_palette(PALETTE_SIZES, len(target_sizes))

        fig, ax = plt.subplots(figsize=(10, 6))
        all_measured_speedups = []

        for color, size in zip(palette, target_sizes):
            seq_rows = seq_type[seq_type['input_size'] == size]
            t_seq = seq_rows['time_median'].mean()
            if t_seq == 0 or pd.isna(t_seq):
                continue
            par_rows = par_type[par_type['input_size'] == size]
            grouped = (par_rows.groupby('threads')
                               .agg(time_med=('time_median', 'mean'),
                                    time_q25=('time_q25', 'mean'),
                                    time_q75=('time_q75', 'mean'))
                               .reset_index())
            grouped['speedup'] = t_seq / grouped['time_med']
            grouped['err_lo']  = (grouped['speedup'] - t_seq / grouped['time_q75']).clip(lower=0)
            grouped['err_hi']  = (t_seq / grouped['time_q25'] - grouped['speedup']).clip(lower=0)

            ax.errorbar(grouped['threads'], grouped['speedup'],
                        yerr=[grouped['err_lo'], grouped['err_hi']],
                        marker='o', color=color, label=f"N={int(size):,}", linewidth=2, capsize=3)
            all_measured_speedups.extend(zip(grouped['threads'].tolist(), grouped['speedup'].tolist()))

        ax.plot(thread_vals, thread_vals, 'k--', linewidth=1.5, label='Ideal (linear)')

        if all_measured_speedups:
            t_arr = np.array([x[0] for x in all_measured_speedups], dtype=float)
            s_arr = np.array([x[1] for x in all_measured_speedups], dtype=float)
            f_fitted = fit_amdahl(t_arr, s_arr)
            p_range = np.linspace(1, max(thread_vals) * 1.1, 300)
            ax.plot(p_range, amdahl(p_range, f_fitted),
                    linestyle='-.', color='#8e44ad', linewidth=2.0,
                    label=f'Amdahl fit  f_seq={f_fitted:.3f}  ->  max speedup ~{1/f_fitted:.1f}x')
            print(f"  {base_algo} [{chosen_type}]: fitted f_seq = {f_fitted:.4f}  "
                  f"(predicted max speedup = {1/f_fitted:.2f}x)")

        ax.set_xlabel('Number of Threads', fontsize=12)
        ax.set_ylabel('Speedup  (T_seq / T_par)', fontsize=12)
        ax.set_xticks(thread_vals)
        subtitle = f"\n{hw_label}" if hw_label else ""
        ax.set_title(f'{base_algo.capitalize()}: Speedup vs Threads  [{chosen_type} distribution]{subtitle}', fontsize=11)
        ax.legend(fontsize=9, ncol=2)
        ax.grid(True, linestyle='--', alpha=0.4)
        fig.tight_layout()
        path = os.path.join(output_dir, f'{base_algo}_speedup_vs_threads.png')
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"  Saved {path}")


def plot_efficiency(agg: pd.DataFrame, output_dir: str, hw_label: str = "") -> None:
    """Plots parallel efficiency E(p) = Speedup / p for each algorithm."""
    print("Generating Parallel Efficiency vs Threads...")
    base_algos = sorted(set(a.rsplit('_', 1)[0] for a in agg['algorithm'].unique()))

    fig, axes = plt.subplots(1, len(base_algos), figsize=(7 * len(base_algos), 5), sharey=True)
    if len(base_algos) == 1:
        axes = [axes]

    for ax, base_algo in zip(axes, base_algos):
        par_algo = f"{base_algo}_par"
        seq_algo = f"{base_algo}_seq"
        if par_algo not in agg['algorithm'].values:
            continue

        par_df = agg[agg['algorithm'] == par_algo]
        seq_df = agg[agg['algorithm'] == seq_algo]

        chosen_type = None
        for pt in ['uniform', 'gaussian']:
            if pt in par_df['dataset_type'].values and pt in seq_df['dataset_type'].values:
                chosen_type = pt
                break
        if chosen_type is None:
            chosen_type = par_df.groupby('dataset_type')['input_size'].nunique().idxmax()

        par_type = par_df[par_df['dataset_type'] == chosen_type]
        seq_type = seq_df[seq_df['dataset_type'] == chosen_type]

        common_sizes = sorted(set(par_type['input_size'].unique()) & set(seq_type['input_size'].unique()))
        if not common_sizes:
            continue

        pcts = [0.25, 0.5, 0.75, 1.0]
        target_sizes = sorted(set(common_sizes[max(0, int(p * (len(common_sizes) - 1)))] for p in pcts))
        thread_vals = sorted(par_type['threads'].unique())
        palette = sns.color_palette(PALETTE_SIZES, len(target_sizes))

        for color, size in zip(palette, target_sizes):
            t_seq = seq_type[seq_type['input_size'] == size]['time_median'].mean()
            if not t_seq or pd.isna(t_seq):
                continue
            par_rows = par_type[par_type['input_size'] == size]
            grouped = (par_rows.groupby('threads').agg(time_med=('time_median', 'mean')).reset_index())
            grouped['speedup']    = t_seq / grouped['time_med']
            grouped['efficiency'] = grouped['speedup'] / grouped['threads']
            ax.plot(grouped['threads'], grouped['efficiency'],
                    marker='o', color=color, linewidth=2, label=f"N={int(size):,}")

        ax.axhline(1.0, color='black',  linestyle='--', linewidth=1.2, label='Perfect efficiency')
        ax.axhline(0.5, color='orange', linestyle=':',  linewidth=1.5, label='50% efficiency')
        ax.set_xlabel('Number of Threads', fontsize=12)
        ax.set_ylabel('Efficiency  (Speedup / Threads)', fontsize=11)
        ax.set_xticks(thread_vals)
        ax.set_ylim(0, 1.15)
        ax.set_title(f'{base_algo.capitalize()}  [{chosen_type}]', fontsize=12)
        ax.legend(fontsize=8)
        ax.grid(True, linestyle='--', alpha=0.4)

    subtitle = f"\n{hw_label}" if hw_label else ""
    fig.suptitle(f'Parallel Efficiency vs Threads  E(p) = Speedup / p{subtitle}', fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    path = os.path.join(output_dir, 'parallel_efficiency.png')
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved {path}")


def plot_memory(agg: pd.DataFrame, output_dir: str, hw_label: str = "") -> None:
    """Plots peak memory usage vs input size with baseline subtracted and O(N) reference lines."""
    print("Generating Memory Usage vs Input Size...")

    fig, ax = plt.subplots(figsize=(10, 6))
    seq_algos = [a for a in sorted(agg['algorithm'].unique()) if a.endswith('_seq')]
    colors = {'quicksort_seq': '#2980b9', 'mergesort_seq': '#c0392b'}

    ref_plotted = False
    for algo in seq_algos:
        subset = agg[(agg['algorithm'] == algo) & (agg['threads'] == 1)].copy()
        if subset.empty:
            continue

        sub = (subset.groupby('input_size').agg(mem=('memory_median', 'mean'))
                     .reset_index().sort_values('input_size'))
        baseline_kb = sub['mem'].iloc[0]
        sub['mem_delta'] = (sub['mem'] - baseline_kb).clip(lower=0)

        ax.plot(sub['input_size'], sub['mem_delta'],
                marker='o', markersize=3, linewidth=1.5,
                color=colors.get(algo, 'grey'),
                label=f'{algo} (measured, baseline-subtracted)')

        if not ref_plotted:
            sizes = sub['input_size'].values
            bytes_per_float = 4
            ax.plot(sizes, sizes * bytes_per_float * 2 / 1024, 'b--', linewidth=1.2, alpha=0.6,
                    label='Quicksort theory: O(N)x2')
            ax.plot(sizes, sizes * bytes_per_float * 3 / 1024, 'r--', linewidth=1.2, alpha=0.6,
                    label='Mergesort theory: O(N)x3')
            ref_plotted = True

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Input Size (N)')
    ax.set_ylabel('Peak RSS - Baseline (KB)')
    subtitle = f"\n{hw_label}" if hw_label else ""
    ax.set_title(f'Memory Usage vs Input Size (baseline-subtracted){subtitle}', fontsize=11)
    ax.legend(fontsize=8)
    ax.grid(True, which='both', linestyle='--', alpha=0.35)
    fig.tight_layout()
    path = os.path.join(output_dir, 'memory_comparison.png')
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved {path}")


def plot_comparisons(agg: pd.DataFrame, output_dir: str) -> None:
    """Plots comparison count vs input size, one line per dataset type, with O(N log N) reference."""
    print("Generating Comparison Count vs Input Size (per dataset type)...")
    base_algos = sorted(set(a.rsplit('_', 1)[0] for a in agg['algorithm'].unique()))

    for base_algo in base_algos:
        seq_algo = f"{base_algo}_seq"
        subset = agg[(agg['algorithm'] == seq_algo) & (agg['threads'] == 1)]
        if subset.empty:
            continue

        dataset_types = sorted(subset['dataset_type'].unique())
        palette = sns.color_palette(PALETTE_DATASETS, len(dataset_types))
        fig, ax = plt.subplots(figsize=(10, 6))

        for color, dtype in zip(palette, dataset_types):
            sub = subset[subset['dataset_type'] == dtype].sort_values('input_size')
            if sub.empty:
                continue
            ax.plot(sub['input_size'], sub['comp_median'],
                    marker='o', markersize=3, linewidth=1.2,
                    color=color, label=dtype.replace('_', ' '))

        all_sizes = np.sort(subset['input_size'].unique())
        if len(all_sizes) > 0:
            ref_n = all_sizes[len(all_sizes) // 2]
            ref_c = subset[subset['input_size'] == ref_n]['comp_median'].mean()
            if ref_c > 0 and ref_n > 1:
                scale = ref_c / (ref_n * np.log2(ref_n))
                ref_y = scale * all_sizes * np.log2(np.maximum(all_sizes, 2))
                ax.plot(all_sizes, ref_y, 'k--', linewidth=1.5, label='O(N log2 N) reference')

        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel('Input Size (N)')
        ax.set_ylabel('Number of Comparisons')
        ax.set_title(f'{base_algo.capitalize()}: Comparison Count vs Input Size')
        ax.legend(fontsize=8, ncol=2)
        ax.grid(True, which='both', linestyle='--', alpha=0.35)
        fig.tight_layout()
        path = os.path.join(output_dir, f'{base_algo}_comparisons_vs_size.png')
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"  Saved {path}")


def plot_algo_comparison(agg: pd.DataFrame, output_dir: str, hw_label: str = "") -> None:
    """Side-by-side sequential quicksort vs mergesort across all dataset types."""
    print("Generating algorithm comparison (QS vs MS, sequential)...")
    dataset_types = sorted(agg['dataset_type'].unique())

    ncols = min(3, len(dataset_types))
    nrows = (len(dataset_types) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4.5 * nrows),
                             sharex=False, sharey=False)
    axes = np.array(axes).flatten()

    algo_colors = {'quicksort_seq': '#2980b9', 'mergesort_seq': '#c0392b'}

    for ax, dtype in zip(axes, dataset_types):
        for algo, color in algo_colors.items():
            sub = (agg[(agg['algorithm'] == algo) &
                       (agg['dataset_type'] == dtype) &
                       (agg['threads'] == 1)]
                   .sort_values('input_size'))
            if sub.empty:
                continue
            ax.errorbar(
                sub['input_size'], sub['time_median'],
                yerr=[sub['time_err_lo'], sub['time_err_hi']],
                marker='o', markersize=3, linewidth=1.2,
                color=color, label=algo.replace('_', ' '), capsize=2
            )
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_title(dtype.replace('_', ' '), fontsize=9)
        ax.set_xlabel('Input Size (N)')
        ax.set_ylabel('Time (ms)')
        ax.grid(True, which='both', linestyle='--', alpha=0.35)
        ax.legend(fontsize=7)

    for ax in axes[len(dataset_types):]:
        ax.set_visible(False)

    subtitle = f"\n{hw_label}" if hw_label else ""
    fig.suptitle(f'Sequential Comparison: Quicksort vs Mergesort (median ± IQR){subtitle}', fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    path = os.path.join(output_dir, 'sequential_comparison.png')
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved {path}")


def create_graphs(csv_file: str, output_dir: str) -> None:
    if not os.path.exists(csv_file):
        print(f"Error: {csv_file} not found.")
        return

    os.makedirs(output_dir, exist_ok=True)
    hw_label = load_hw_info(csv_file)

    print(f"Reading {csv_file}...")
    df = load_and_clean(csv_file)
    agg = aggregate(df)

    print(f"\nData summary:")
    print(f"  Algorithms    : {sorted(df['algorithm'].unique())}")
    print(f"  Dataset types : {sorted(df['dataset_type'].unique())}")
    sizes = df['input_size']
    print(f"  Input sizes   : {int(sizes.min()):,} - {int(sizes.max()):,}  ({sizes.nunique()} unique)")
    print(f"  Thread counts : {sorted(df['threads'].unique())}")
    counts = df.groupby(['algorithm', 'input_size', 'threads'])['execution_time_ms'].count()
    print(f"  Runs per cfg  : min={counts.min()}  median={counts.median():.0f}  max={counts.max()}")
    outlier_pct = 100 * agg['has_outlier'].mean()
    print(f"  Configs with outlier runs (>3x median): {outlier_pct:.1f}%")
    if hw_label:
        print(f"\n  Hardware: {hw_label}")
    print()

    sns.set_theme(style="whitegrid")

    plot_time_vs_size(agg, output_dir, hw_label)
    plot_speedup(agg, output_dir, hw_label)
    plot_efficiency(agg, output_dir, hw_label)
    plot_memory(agg, output_dir, hw_label)
    plot_comparisons(agg, output_dir)
    plot_algo_comparison(agg, output_dir, hw_label)

    print(f"\nAll graphs saved to '{output_dir}/'")


def main():
    parser = argparse.ArgumentParser(description="Graph sorting benchmark results.")
    parser.add_argument('--input', type=str, default='benchmark_results.csv')
    parser.add_argument('--output_dir', type=str, default='graphs')
    args = parser.parse_args()
    create_graphs(args.input, args.output_dir)


if __name__ == "__main__":
    main()
