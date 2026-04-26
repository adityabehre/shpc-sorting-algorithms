# benchmark.py — runs both sorting algorithms across all dataset files and writes results to CSV.
# Warmup runs are discarded; algorithm order is shuffled per file to reduce cache bias.

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


def collect_hardware_info() -> dict:
    """Returns CPU/cache info for the current machine, saved alongside results for traceability."""
    info = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "python_platform": platform.platform(),
        "cpu_model": "unknown",
        "physical_cores": "unknown",
        "logical_cores": "unknown",
        "cache_info": "unknown",
    }

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
        pass  # not macOS, try Linux

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


def extract_size_from_filename(filename: str):
    """Parses element count from the dataset filename. Returns None if not found."""
    match = re.search(r'-input-(\d+)-', filename)
    if match:
        return int(match.group(1))
    match = re.search(r'-input(\d+)\.txt$', filename)
    if match:
        return int(match.group(1))
    return None


def run_once(cmd: list) -> list | None:
    """Runs one sort command; returns [mode, time_ms, mem_kb, comparisons, size, threads] or None."""
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=600)
        parts = result.stdout.strip().split(',')
        return parts if len(parts) == 6 else None
    except subprocess.CalledProcessError as e:
        print(f"\nError running {' '.join(cmd)}: {e.stderr.strip()}")
        return None
    except subprocess.TimeoutExpired:
        print(f"\nTimeout running {' '.join(cmd)}")
        return None


def main():
    parser = argparse.ArgumentParser(description="Run sorting benchmarks.")
    parser.add_argument('--dataset_dir', type=str, default='benchmark-dataset-for-sorting-algorithms')
    parser.add_argument('--max_threads', type=int, default=16)
    parser.add_argument('--output', type=str, default='benchmark_results.csv')
    parser.add_argument('--limit_files', type=int, default=None)
    parser.add_argument('--runs', type=int, default=5, help='Measured runs per configuration')
    parser.add_argument('--warmup', type=int, default=1, help='Warmup runs (results discarded)')
    parser.add_argument('--append', action='store_true')
    parser.add_argument('--seed', type=int, default=None)
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        print(f"Random seed set to {args.seed} for reproducibility.")

    thread_counts = [1, 2, 4, 8]
    if args.max_threads >= 10: thread_counts.append(10)
    if args.max_threads >= 16: thread_counts.append(16)

    algorithms = ['mergesort', 'quicksort']

    for algo in algorithms:
        if not os.path.exists(f"./{algo}"):
            print(f"Error: ./{algo} not found. Please run 'make' first.")
            return

    hw_info = collect_hardware_info()
    hw_path = args.output.replace('.csv', '_hardware.json')
    with open(hw_path, 'w') as f:
        json.dump(hw_info, f, indent=2)
    print(f"Hardware info saved to {hw_path}")
    print(f"  CPU : {hw_info['cpu_model']}")
    print(f"  Cores: {hw_info['physical_cores']} physical / {hw_info['logical_cores']} logical")
    print(f"  Cache: {hw_info['cache_info']}")
    print()

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

    dataset_files.sort(key=lambda x: x['size'])

    if args.limit_files and len(dataset_files) > args.limit_files:
        step = max(1, len(dataset_files) // args.limit_files)
        dataset_files = dataset_files[::step][:args.limit_files]

    runs_per_file = len(algorithms) * (1 + len(thread_counts)) * (args.warmup + args.runs)
    total_runs = len(dataset_files) * runs_per_file
    print(f"Found {len(dataset_files)} files across dataset types: "
          f"{sorted(set(f['type'] for f in dataset_files))}")
    print(f"Per config: {args.warmup} warmup + {args.runs} measured runs  "
          f"({len(algorithms)} algos x {1 + len(thread_counts)} modes)")
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
                shuffled_algos = algorithms[:]
                random.shuffle(shuffled_algos)  # randomize order to reduce cache-warming bias

                for algo in shuffled_algos:
                    mode_thread_pairs = [('seq', 1)] + [('par', t) for t in thread_counts]

                    for mode, threads in mode_thread_pairs:
                        for _ in range(args.warmup):
                            run_once([f"./{algo}", file_info['filepath'], mode, str(threads)])
                            pbar.update(1)

                        for run_num in range(1, args.runs + 1):
                            cmd = [f"./{algo}", file_info['filepath'], mode, str(threads)]
                            parts = run_once(cmd)

                            if parts is not None:
                                out_mode, time_ms, memory_kb, comp, size, thr = parts
                                writer.writerow({
                                    'algorithm': f"{algo}_{out_mode}",
                                    'dataset_type': file_info['type'],
                                    'filename': file_info['filename'],
                                    'input_size': size,
                                    'threads': thr,
                                    'run_number': run_num,
                                    'execution_time_ms': time_ms,
                                    'memory_usage_kb': memory_kb,
                                    'comparison_count': comp
                                })
                                csvfile.flush()  # persist after every row

                            pbar.update(1)

    print(f"\nBenchmark complete. Results saved to {args.output}")
    print(f"Hardware context saved to {hw_path}")


if __name__ == "__main__":
    main()
