# Parallel Sorting Algorithms Benchmark

Parallel vs. sequential implementations of **Quicksort** and **Merge Sort** using raw `std::thread`.  
Benchmarks measure execution time, memory usage, and comparison counts across varying input sizes and thread counts.

## Getting Started: Compiling and Running the Project

The dataset is already included in `small-benchmark-dataset/`, so the only steps needed are installing prerequisites, compiling the C++ code, and running the benchmark and graphing scripts.

### Prerequisites

Ensure your system has the following installed:
- **C++ Compiler**: `g++` with C++17 support. On macOS, this can be installed via Xcode command line tools (`xcode-select --install`).
- **Python 3**: For benchmarking and graphing scripts.

Install the required Python libraries using pip:

```bash
pip install pandas matplotlib seaborn numpy scipy tqdm
```

### 1. Build the Executables

Compile the C++ programs (`quicksort.cpp` and `mergesort.cpp`) using the provided Makefile. Run the following command in the project directory:

```bash
make
```

This compiles the executables with `-O3` and `-march=native` optimizations. The code should be compiled on the same machine used for benchmarking.

### 2. Run the Benchmarks

> **Note:** The committed `benchmark_results.csv` and graphs in `graphs/` come from a larger 10M-element run. Running the command below will overwrite the CSV with smaller-scale results.

Execute the main benchmarking script on the small dataset to test the sorting algorithms and record execution time, memory usage, and comparison counts.

```bash
python3 benchmark.py --dataset_dir small-benchmark-dataset --runs 5 --warmup 1 --seed 42
```
This performs 5 measured runs per configuration and records the median to mitigate OS-level variations. Results are saved to `benchmark_results.csv`.

### 3. Graph the Results

Generate visualizations of the performance data and Amdahl's Law speedup curves by running:

```bash
python3 graph_results.py
```
This script reads the CSV file and saves the resulting graphs in the `graphs/` directory.
