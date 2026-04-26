"""
benchmark.py
============
Runs sorting benchmarks across all dataset files and writes results to a CSV.

Key design decisions:
  - Warmup runs: the first `--warmup` runs of each configuration are executed
    but their results are discarded. This ensures the process I/O, OS page
    faults, and CPU branch predictor are in a warm state before measurement.
  - Algorithm order randomised per file to eliminate cache-warming bias.
  - Results are flushed to CSV after every row so a crash/interrupt loses
    at most one measurement.
  - Hardware info is written to a sidecar JSON file alongside the CSV so
    that results can always be traced back to the machine they were collected on.
"""

import os
import json
import platform
import subprocess
import csv
import re
import random
from datetime import datetime
from tqdm import tqdm
import argparse


# ---------------------------------------------------------------------------
# Hardware profiling
# ---------------------------------------------------------------------------

def collect_hardware_info() -> dict:
    """
    Collect host CPU/memory info and return as a dict.
    Written to a sidecar .json file so benchmark results are always traceable.
    """
    info = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "python_platform": platform.platform(),
        "cpu_model": "unknown",
        "physical_cores": "unknown",
        "logical_cores": "unknown",
        "cache_info": "unknown",
    }

    # macOS / BSD
    try:
        def sysctl(key):
            r = subprocess.run(["sysctl", "-n", key], capture_output=True, text=True)
            return r.stdout.strip() if r.returncode == 0 else None

        info["cpu_model"]      = sysctl("machdep.cpu.brand_string") or "unknown"
        info["physical_cores"] = sysctl("hw.physicalcpu") or "unknown"
        info["logical_cores"]  = sysctl("hw.logicalcpu") or "unknown"
        l1 = sysctl("hw.l1dcachesize")
        l2 = sysctl("hw.l2cachesize")
        l3 = sysctl("hw.l3cachesize")
        info["cache_info"] = f"L1d={l1}B  L2={l2}B  L3={l3 or 'N/A'}B"
    except FileNotFoundError:
        pass  # Not macOS — try Linux next

    # Linux fallback
    if info["physical_cores"] == "unknown":
        try:
            r = subprocess.run(["nproc", "--all"], capture_output=True, text=True)
            info["logical_cores"] = r.stdout.strip()
            r2 = subprocess.run(
                ["grep", "-m1", "model name", "/proc/cpuinfo"],
                capture_output=True, text=True
            )
            info["cpu_model"] = r2.stdout.split(":")[-1].strip()
        except Exception:
            pass

    return info


# ---------------------------------------------------------------------------
# Filename parsing
# ---------------------------------------------------------------------------

def extract_size_from_filename(filename: str):
    """
    Extract element count from filename. Handles two naming conventions:
      - Standard:     uniform-input-410000-1-float.txt  -> 410000
      - Combinations: ...-input50000.txt                -> 50000
    """
    # Standard pattern: -input-<N>-
    match = re.search(r'-input-(\d+)-', filename)
    if match:
        return int(match.group(1))
    # Combinations pattern: -input<N>.txt (no surrounding dashes)
    match = re.search(r'-input(\d+)\.txt$', filename)
    if match:
        return int(match.group(1))
    return None


# ---------------------------------------------------------------------------
# Single-run executor
# ---------------------------------------------------------------------------

def run_once(cmd: list) -> list | None:
    """
    Run a command once and return (time_ms, memory_kb, comparisons, size, threads, mode)
    as a list of strings, or None on failure.
    """
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=600)
        output = result.stdout.strip()
        parts = output.split(',')
        if len(parts) == 6:
            return parts  # mode, time_ms, memory_kb, comparisons, size, threads
        return None
    except subprocess.CalledProcessError as e:
        print(f"\nError running {' '.join(cmd)}: {e.stderr.strip()}")
        return None
    except subprocess.TimeoutExpired:
        print(f"\nTimeout running {' '.join(cmd)}")
        return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Run sorting benchmarks.")
    parser.add_argument('--dataset_dir', type=str,
                        default='benchmark-dataset-for-sorting-algorithms',
                        help='Path to dataset directory')
    parser.add_argument('--max_threads', type=int, default=16,
                        help='Max number of threads to test (e.g., 8 or 16)')
    parser.add_argument('--output', type=str, default='benchmark_results.csv',
                        help='Output CSV file')
    parser.add_argument('--limit_files', type=int, default=None,
                        help='Limit number of files to test (for quick testing)')
    parser.add_argument('--runs', type=int, default=5,
                        help='Number of *measured* runs per configuration')
    parser.add_argument('--warmup', type=int, default=1,
                        help=(
                            'Number of warmup runs before measured runs. '
                            'Warmup runs are executed but their output is discarded. '
                            'This ensures the OS page cache, branch predictor, and '
                            'CPU frequency scaling are in steady state before timing.'
                        ))
    parser.add_argument('--append', action='store_true',
                        help='Append to existing CSV instead of overwriting')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed for reproducible algorithm ordering. '
                             'Set this so results can be reproduced exactly.')
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        print(f"Random seed set to {args.seed} for reproducibility.")

    # Thread counts: always include the physical core count (10 for Apple M4).
    # Testing at exactly the physical core count shows peak efficiency without
    # oversubscription. We also keep 16 to demonstrate the oversubscription regime.
    thread_counts = [1, 2, 4, 8]
    if args.max_threads >= 10:
        thread_counts.append(10)
    if args.max_threads >= 16:
        thread_counts.append(16)

    algorithms = ['mergesort', 'quicksort']

    # Check if binaries exist
    for algo in algorithms:
        if not os.path.exists(f"./{algo}"):
            print(f"Error: ./{algo} not found. Please run 'make' first.")
            return

    # Collect and save hardware info
    hw_info = collect_hardware_info()
    hw_path = args.output.replace('.csv', '_hardware.json')
    with open(hw_path, 'w') as f:
        json.dump(hw_info, f, indent=2)
    print(f"Hardware info saved to {hw_path}")
    print(f"  CPU : {hw_info['cpu_model']}")
    print(f"  Cores: {hw_info['physical_cores']} physical / {hw_info['logical_cores']} logical")
    print(f"  Cache: {hw_info['cache_info']}")
    print()

    # Collect all dataset files
    dataset_files = []
    for root, _, files in os.walk(args.dataset_dir):
        for file in files:
            if file.endswith('.txt'):
                size = extract_size_from_filename(file)
                if size is not None:
                    rel_path = os.path.relpath(root, args.dataset_dir)
                    parts = rel_path.split(os.sep)
                    dataset_type = parts[0] if parts[0] != '.' else 'unknown'

                    dataset_files.append({
                        'filepath': os.path.join(root, file),
                        'filename': file,
                        'size': size,
                        'type': dataset_type
                    })

    if not dataset_files:
        print(f"No valid dataset files found in {args.dataset_dir}.")
        return

    # Sort files by size for better progress visibility and graph scaling
    dataset_files.sort(key=lambda x: x['size'])

    if args.limit_files and len(dataset_files) > args.limit_files:
        # Take an evenly distributed sample across all sizes
        step = max(1, len(dataset_files) // args.limit_files)
        dataset_files = dataset_files[::step][:args.limit_files]

    # seq run + par runs, times number of measured runs (excluding warmup)
    runs_per_file = len(algorithms) * (1 + len(thread_counts)) * (args.warmup + args.runs)
    total_runs = len(dataset_files) * runs_per_file
    print(f"Found {len(dataset_files)} files across dataset types: "
          f"{sorted(set(f['type'] for f in dataset_files))}")
    print(f"Per config: {args.warmup} warmup + {args.runs} measured runs  "
          f"({len(algorithms)} algos × {1 + len(thread_counts)} modes)")
    print(f"Total benchmark invocations: {total_runs}")
    print()

    fieldnames = ['algorithm', 'dataset_type', 'filename', 'input_size', 'threads',
                  'run_number', 'execution_time_ms', 'memory_usage_kb', 'comparison_count']

    write_mode = 'a' if args.append else 'w'
    with open(args.output, write_mode, newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if not args.append:
            writer.writeheader()

        with tqdm(total=total_runs, desc="Benchmarking") as pbar:
            for file_info in dataset_files:
                # Randomize algorithm order per file to eliminate cache-warming bias
                shuffled_algos = algorithms[:]
                random.shuffle(shuffled_algos)

                for algo in shuffled_algos:
                    mode_thread_pairs = [('seq', 1)] + [('par', t) for t in thread_counts]

                    for mode, threads in mode_thread_pairs:
                        # --- Warmup runs (executed but results discarded) ---
                        for _ in range(args.warmup):
                            cmd = [f"./{algo}", file_info['filepath'], mode, str(threads)]
                            run_once(cmd)  # result intentionally discarded
                            pbar.update(1)

                        # --- Measured runs ---
                        for run_num in range(1, args.runs + 1):
                            cmd = [f"./{algo}", file_info['filepath'], mode, str(threads)]
                            parts = run_once(cmd)

                            if parts is not None:
                                out_mode, time_ms, memory_kb, comp, size, thr = parts
                                algo_name = f"{algo}_{out_mode}"

                                row = {
                                    'algorithm': algo_name,
                                    'dataset_type': file_info['type'],
                                    'filename': file_info['filename'],
                                    'input_size': size,
                                    'threads': thr,
                                    'run_number': run_num,
                                    'execution_time_ms': time_ms,
                                    'memory_usage_kb': memory_kb,
                                    'comparison_count': comp
                                }
                                writer.writerow(row)
                                csvfile.flush()  # Persist data even if interrupted

                            pbar.update(1)

    print(f"\nBenchmark complete. Results saved to {args.output}")
    print(f"Hardware context saved to {hw_path}")


if __name__ == "__main__":
    main()
