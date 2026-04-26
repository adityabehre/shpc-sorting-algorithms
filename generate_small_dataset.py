# generate_small_dataset.py — creates a small benchmark dataset mirroring the large Kaggle one.
# Useful for quick testing without needing the full 10M-element files.

import os
import random
import argparse

VOCAB_REPEATED = [1.0, 7.3, 14.5, 22.8, 31.1, 45.6, 63.2, 78.9, 91.4, 99.7]

FLAT_DISTRIBUTIONS = ["uniform", "ordered", "reverse_ordered", "nearly_sorted", "same_value", "repeated_values"]
FLAT_SIZES   = [100_000, 500_000, 1_000_000]
GAUSS_SIZES  = [100_000, 500_000, 1_000_000]
GAUSS_TRIALS = 3
COMBO_SIZES  = [1_000, 10_000, 100_000]


def _write_flat(f, size: int, dist: str) -> None:
    """Writes `size` values for the given flat distribution to open file `f`."""
    if dist == "uniform":
        for _ in range(size):
            f.write(f"{random.uniform(0.0, 1000.0):.6f}\n")
    elif dist == "ordered":
        for i in range(size):
            f.write(f"{float(i):.6f}\n")
    elif dist == "reverse_ordered":
        for i in range(size, 0, -1):
            f.write(f"{float(i):.6f}\n")
    elif dist == "nearly_sorted":
        data = list(range(size))
        n_swaps = max(1, size // 100)
        for _ in range(n_swaps):
            a, b = random.randrange(size), random.randrange(size)
            data[a], data[b] = data[b], data[a]
        for val in data:
            f.write(f"{float(val):.6f}\n")
    elif dist == "same_value":
        for _ in range(size):
            f.write("42.000000\n")
    elif dist == "repeated_values":
        for _ in range(size):
            f.write(f"{random.choice(VOCAB_REPEATED):.6f}\n")
    else:
        raise ValueError(f"Unknown distribution: {dist!r}")


def _write_gaussian(f, size: int) -> None:
    """Writes `size` Gaussian floats (mean=0, std=1) to `f`."""
    for _ in range(size):
        f.write(f"{random.gauss(0.0, 1.0):.6f}\n")


def _write_combinations(f, size: int) -> None:
    """Writes a descending first-half + ascending second-half sequence (matches Kaggle format)."""
    half = size // 2
    for i in range(size - 1, half - 1, -1):
        f.write(f"{i}\n")
    for i in range(1, half + 1):
        f.write(f"{i}\n")


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def generate_flat(base_dir: str, sizes: list[int]) -> None:
    """Generates one file per distribution per size under `base_dir/<dist>/`."""
    for dist in FLAT_DISTRIBUTIONS:
        out_dir = os.path.join(base_dir, dist)
        _ensure_dir(out_dir)
        for size in sizes:
            fname = f"{dist}-input-{size}-1-float.txt"
            fpath = os.path.join(out_dir, fname)
            if os.path.exists(fpath):
                print(f"  [skip] {fpath}")
                continue
            print(f"  Generating {size:>9,} elements [{dist}] -> {fpath}")
            with open(fpath, "w") as f:
                _write_flat(f, size, dist)


def generate_gaussian(base_dir: str, sizes: list[int], trials: int) -> None:
    """Generates Gaussian files with multiple trials per size."""
    out_dir = os.path.join(base_dir, "gaussian", "gaussian")
    _ensure_dir(out_dir)
    for size in sizes:
        for trial in range(1, trials + 1):
            fname = f"gaussian-input-{size}-{trial}-float.txt"
            fpath = os.path.join(out_dir, fname)
            if os.path.exists(fpath):
                print(f"  [skip] {fpath}")
                continue
            print(f"  Generating {size:>9,} elements [gaussian trial {trial}] -> {fpath}")
            with open(fpath, "w") as f:
                _write_gaussian(f, size)


def generate_combinations(base_dir: str, sizes: list[int]) -> None:
    """Generates ascending/descending combination files."""
    cat = "Combinations_of_ascending_and_descending_two_sub_arrays"
    out_dir = os.path.join(base_dir, cat, cat)
    _ensure_dir(out_dir)
    for size in sizes:
        fname = f"{cat}-input{size}.txt"
        fpath = os.path.join(out_dir, fname)
        if os.path.exists(fpath):
            print(f"  [skip] {fpath}")
            continue
        print(f"  Generating {size:>9,} elements [combinations] -> {fpath}")
        with open(fpath, "w") as f:
            _write_combinations(f, size)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a small benchmark dataset.")
    parser.add_argument("--out_dir",      type=str,       default="small-benchmark-dataset")
    parser.add_argument("--flat_sizes",   type=int, nargs="+", default=FLAT_SIZES)
    parser.add_argument("--gauss_sizes",  type=int, nargs="+", default=GAUSS_SIZES)
    parser.add_argument("--combo_sizes",  type=int, nargs="+", default=COMBO_SIZES)
    parser.add_argument("--gauss_trials", type=int,       default=GAUSS_TRIALS)
    parser.add_argument("--seed",         type=int,       default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    print(f"Random seed : {args.seed}")
    print(f"Output dir  : {args.out_dir}")
    print(f"Flat sizes  : {args.flat_sizes}")
    print(f"Gauss sizes : {args.gauss_sizes}  (x{args.gauss_trials} trials)")
    print(f"Combo sizes : {args.combo_sizes}")
    print()

    print("=== Flat distributions ===")
    generate_flat(args.out_dir, args.flat_sizes)

    print("\n=== Gaussian distribution ===")
    generate_gaussian(args.out_dir, args.gauss_sizes, args.gauss_trials)

    print("\n=== Combinations distribution ===")
    generate_combinations(args.out_dir, args.combo_sizes)

    print("\nDone. Small dataset written to:", args.out_dir)


if __name__ == "__main__":
    main()
