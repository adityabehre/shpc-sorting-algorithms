"""
generate_large_dataset.py
=========================
Generates benchmark input files for the parallel sorting project.

Distributions produced:
  uniform          — floats drawn uniformly in [0, 1000)
  ordered          — ascending integers (best case for many algorithms)
  reverse_ordered  — descending integers (worst case for naive quicksort)
  nearly_sorted    — sorted array with ~1% of elements randomly swapped;
                     the most practically important pathological case for
                     quicksort because it occurs frequently in real data.
  same_value       — every element identical (degenerate case)
  repeated_values  — small vocabulary (10 distinct values) repeated N times;
                     this is different from same_value: it tests how algorithms
                     handle many equal elements without all being identical.

Usage:
    # Generate a single large uniform file (10M elements, default)
    python3 generate_large_dataset.py

    # Specify size and directory
    python3 generate_large_dataset.py --size 5000000 --dir /path/to/dir

    # Generate ALL distributions at multiple sizes
    python3 generate_large_dataset.py --all --sizes 1000000 5000000 10000000
"""

import os
import random
import argparse


VOCAB_REPEATED = [1.0, 7.3, 14.5, 22.8, 31.1, 45.6, 63.2, 78.9, 91.4, 99.7]


def generate_dataset(size: int, filepath: str, dist_type: str = "uniform") -> None:
    """Write `size` floats of the given distribution to `filepath`."""
    print(f"  Generating {size:,} elements [{dist_type}] -> {filepath}")
    os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else ".", exist_ok=True)

    with open(filepath, "w") as f:
        if dist_type == "uniform":
            for _ in range(size):
                f.write(f"{random.uniform(0.0, 1000.0):.6f}\n")

        elif dist_type == "ordered":
            for i in range(size):
                f.write(f"{float(i):.6f}\n")

        elif dist_type == "reverse_ordered":
            for i in range(size, 0, -1):
                f.write(f"{float(i):.6f}\n")

        elif dist_type == "nearly_sorted":
            # Start with a sorted sequence and randomly swap ~1% of elements.
            # This is the most practically important stress test for quicksort:
            # real-world data is often "almost sorted" (log files, timestamps,
            # previously sorted data with a few late insertions), and this is
            # exactly the case that degrades naive pivot-selection strategies.
            data = list(range(size))
            n_swaps = max(1, size // 100)
            for _ in range(n_swaps):
                a, b = random.randrange(size), random.randrange(size)
                data[a], data[b] = data[b], data[a]
            for val in data:
                f.write(f"{float(val):.6f}\n")

        elif dist_type == "same_value":
            # Single value repeated: tests degenerate pivot behaviour.
            for _ in range(size):
                f.write(f"{42.0:.6f}\n")

        elif dist_type == "repeated_values":
            # Small vocabulary (10 distinct values) repeated N times.
            # Different from same_value: the algorithm must do real comparison
            # work, but duplicate keys are very common.  Tests stability-related
            # performance and partition balance on low-entropy data.
            for _ in range(size):
                f.write(f"{random.choice(VOCAB_REPEATED):.6f}\n")

        else:
            raise ValueError(f"Unknown distribution type: {dist_type!r}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate large datasets for parallel sorting benchmarks."
    )
    parser.add_argument(
        "--size", type=int, default=10_000_000,
        help="Number of elements (default: 10M)"
    )
    parser.add_argument(
        "--dir", type=str, default="benchmark-dataset-for-sorting-algorithms/large_uniform",
        help="Output directory for a single uniform file"
    )
    parser.add_argument(
        "--all", action="store_true",
        help="Generate ALL distributions (uniform, ordered, reverse_ordered, "
             "nearly_sorted, same_value, repeated_values)"
    )
    parser.add_argument(
        "--sizes", type=int, nargs="+", default=[10_000_000],
        help="List of sizes to generate when --all is set (default: 10M)"
    )
    parser.add_argument(
        "--base_dir", type=str, default="benchmark-dataset-for-sorting-algorithms",
        help="Base directory for --all mode (each distribution gets a sub-folder)"
    )
    parser.add_argument(
        '--seed', type=int, default=42,
        help='Random seed for reproducibility (default: 42). '
             'Fix this value so benchmark datasets can be regenerated identically.'
    )
    args = parser.parse_args()

    random.seed(args.seed)
    print(f"Random seed: {args.seed}")

    if args.all:
        distributions = [
            "uniform",
            "ordered",
            "reverse_ordered",
            "nearly_sorted",
            "same_value",
            "repeated_values",
        ]
        print(f"Generating ALL distributions at sizes: {[f'{s:,}' for s in args.sizes]}")
        for dist in distributions:
            for size in args.sizes:
                out_dir = os.path.join(args.base_dir, dist)
                # Use the same naming convention as the Kaggle dataset so
                # benchmark.py's extract_size_from_filename() parses it correctly.
                fname = f"{dist}-input-{size}-1-float.txt"
                filepath = os.path.join(out_dir, fname)
                if os.path.exists(filepath):
                    print(f"  [skip] {filepath} already exists.")
                    continue
                generate_dataset(size, filepath, dist)
        print("Done.")
    else:
        # Single uniform file (original behaviour)
        filepath = os.path.join(args.dir, f"large_uniform-input-{args.size}-1-float.txt")
        generate_dataset(args.size, filepath, "uniform")
        print(
            f"Dataset generation complete. "
            f"Run benchmark.py with this new file to test >= {args.size:,} elements."
        )


if __name__ == "__main__":
    main()
