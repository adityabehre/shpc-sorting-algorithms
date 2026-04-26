# Parallel Sorting Algorithms Benchmark

Parallel vs. sequential implementations of **Quicksort** and **Merge Sort** using raw `std::thread`.  
Benchmarks measure execution time, memory usage, and comparison counts across varying input sizes and thread counts.

---

## Hardware (results collected on)
| Property | Value |
|---|---|
| CPU | Apple M4 |
| Physical cores | 10 |
| Logical cores | 10 |
| L1d cache | 64 KB |
| L2 cache | 4 MB |

---

## Project Structure

```
.
├── quicksort.cpp               # Sequential + parallel quicksort
│                               #   Three-way partition (Dutch National Flag)
│                               #   Insertion-sort base case (N < 32)
│                               #   Median-of-three pivot
├── mergesort.cpp               # Sequential + parallel merge sort
│                               #   Pre-allocated scratch buffer (eliminates per-call heap allocation)
│                               #   Insertion-sort base case (N < 32)
├── Makefile                    # -O3 -march=native -std=c++17 -pthread
├── benchmark.py                # Runs all configurations, writes benchmark_results.csv
├── graph_results.py            # Produces all graphs from the CSV
├── generate_large_dataset.py   # Generates benchmark input files (6 distributions)
├── run_large_bench.sh          # Convenience script: benchmark only 5M/10M files
└── benchmark-dataset-for-sorting-algorithms/
    ├── uniform/                # Floats drawn uniformly in [0, 1000)
    ├── ordered/                # Ascending integers (best case)
    ├── reverse_ordered/        # Descending integers (worst case for naive QS)
    ├── nearly_sorted/          # Sorted with ~1% random swaps
    ├── same_value/             # All elements identical (degenerate)
    └── repeated_values/        # 10-element vocabulary repeated N times
```

---

## Quick Start

### 1 — Build
```bash
make
```
Requires `g++` with C++17 support. Uses `-O3 -march=native` for host-specific SIMD acceleration.  
**Note:** Binaries are NOT portable; run all benchmarks on the same machine.

### 2 — Generate datasets
```bash
# All 6 distributions at 2M, 5M, and 10M elements (seed=42 for reproducibility)
python3 generate_large_dataset.py --all --sizes 2000000 5000000 10000000 --seed 42
```
The Kaggle dataset (`benchmark-dataset-for-sorting-algorithms/`) already includes sizes up to 1.36M.

### 3 — Run benchmarks
```bash
# Full benchmark: 5 measured runs, 1 warmup, all files
python3 benchmark.py --runs 5 --warmup 1 --seed 42

# Quick test on just the large files (~20 min on M4)
bash run_large_bench.sh
```
Results are written to `benchmark_results.csv` with a hardware sidecar `benchmark_results_hardware.json`.

### 4 — Generate graphs
```bash
python3 graph_results.py
# Graphs saved to graphs/
```

---

## Graphs produced

| File | Description |
|---|---|
| `quicksort_seq_time_vs_size.png` | QS sequential: time vs N, faceted by dataset type |
| `quicksort_par_time_vs_size.png` | QS parallel: time vs N, one line per thread count |
| `mergesort_seq_time_vs_size.png` | Same for merge sort (sequential) |
| `mergesort_par_time_vs_size.png` | Same for merge sort (parallel) |
| `quicksort_speedup_vs_threads.png` | Speedup curves + fitted Amdahl's law |
| `mergesort_speedup_vs_threads.png` | Same for merge sort |
| `parallel_efficiency.png` | Efficiency E(p) = Speedup/p for both algorithms |
| `memory_comparison.png` | Peak RSS vs N (baseline-subtracted + O(N) theory lines) |
| `quicksort_comparisons_vs_size.png` | Comparison count by dataset type + O(N log N) reference |
| `mergesort_comparisons_vs_size.png` | Same for merge sort |
| `sequential_comparison.png` | Head-to-head QS vs mergesort across all distributions |

---

## Algorithm Design Decisions

### Parallelism strategy: `std::thread` with static thread-budget halving
Each recursive call splits its thread budget between left and right partitions.  
When budget is exhausted or subarray size falls below `THRESHOLD = 10000`, execution falls back to sequential.

**Why `std::thread` over OpenMP?**  
Raw `std::thread` makes the thread-creation and join points explicit, which is important for understanding the overhead. OpenMP task scheduling would adapt dynamically (via work-stealing) but would hide the overhead structure we measure.

### Pivot selection: Median-of-three + Three-way partition (quicksort)
- **Median-of-three** eliminates the O(N²) worst-case on sorted/reverse-sorted inputs.
- **Three-way partition** (Dutch National Flag): places all elements equal to the pivot in a "done" middle zone, never re-partitioned. Gives O(N) total comparisons on `same_value` and near-linear behavior on `repeated_values`.

### Scratch buffer (merge sort)
A single `vector<float> scratch(N)` is allocated once in `main()` and passed by reference through all recursive calls.  
This replaces ~20M heap allocations for N=10M elements. Thread-safety is guaranteed because left and right parallel sub-sorts use non-overlapping index ranges of the scratch array.

### Insertion-sort base case (both algorithms)
Subarrays smaller than 32 elements switch to insertion sort. At this size, insertion sort's sequential memory access pattern beats the partition/split overhead of both algorithms. This is the same strategy used by `std::sort` (introsort) and Java's `Arrays.sort`.

---

## Amdahl's Law Fit (from benchmark data)

| Algorithm | Fitted f_seq | Predicted max speedup |
|---|---|---|
| Quicksort | 0.481 | ≈ 2.1× |
| Merge sort | 0.518 | ≈ 1.9× |

The sequential fraction is dominated by: (1) the top-level sequential merge step, and (2) thread-creation overhead at the root of the recursion tree, which is proportionally large at these input sizes relative to the parallelizable sorting work.

---

## Statistical Notes
- **Metric**: Median execution time (not mean). Timing distributions are right-skewed due to OS scheduling jitter; median is robust to outlier runs.
- **Uncertainty**: IQR (interquartile range) shown as error bars on all timing graphs.
- **Warmup**: 1 warmup run per configuration is discarded before measurement to ensure steady-state CPU frequency, page cache, and branch predictor.
- **Outlier annotation**: Configurations where any run exceeded 3× the median are flagged with a red ★ on the time-vs-size graphs.
