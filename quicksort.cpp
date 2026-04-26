#include <iostream>
#include <vector>
#include <thread>
#include <algorithm>
#include <fstream>
#include <chrono>
#include <atomic>
#include <iomanip>
#include <cmath>
#include <cstdlib>
#include <utility>
#include <sys/resource.h>

using namespace std;

// Sequential fallback threshold: subarrays below this switch to sequential.
const int THRESHOLD = 10000;

// Insertion-sort threshold: subarrays this small use insertion sort directly.
// At ~32 elements the branch-predictor and cache locality of insertion sort
// outweigh the overhead of recursive partitioning. This is the same strategy
// used by std::sort (introsort) and Java's Arrays.sort (timsort).
const int INSERTION_THRESHOLD = 32;

// Global comparison counter.
atomic<unsigned long long> comparisons{0};

// ---------------------------------------------------------------------------
// Insertion sort — used as the base case for small subarrays.
// O(N^2) worst-case but cache-optimal; faster than quicksort for N < ~32
// because it has zero recursion overhead and accesses memory sequentially.
// ---------------------------------------------------------------------------
void insertionSort(vector<float>& arr, int low, int high) {
    unsigned long long local_cmp = 0;
    for (int i = low + 1; i <= high; i++) {
        float key = arr[i];
        int j = i - 1;
        while (j >= low) {
            local_cmp++;
            if (arr[j] > key) { arr[j + 1] = arr[j]; j--; }
            else break;
        }
        arr[j + 1] = key;
    }
    comparisons += local_cmp;
}

// ---------------------------------------------------------------------------
// Median-of-three on RANDOM samples.
//
// We pick three positions at random rather than always using low/mid/high.
// This breaks the adversarial structure of sorted and reverse-sorted inputs:
//   - Pure median-of-three (low/mid/high) on a sorted array always picks
//     arr[mid] as pivot, giving an O(N^1.5) comparison count because every
//     level has a 2:1 split instead of 1:1 (the sub-problems are still
//     sorted, so the next level also picks a 2:1 pivot, and so on).
//   - With random samples, the probability of consistently poor pivots on
//     any structured input is O(1/N), giving O(N log N) expected comparisons.
//
// The srand()/rand() call is intentionally NOT seeded per-call; we seed once
// in main() with a fixed seed for reproducibility across benchmark runs.
// ---------------------------------------------------------------------------
float medianOfThree(vector<float>& arr, int low, int high) {
    if (high - low < 2) return arr[low];
    int range = high - low + 1;
    int a = low  + rand() % range;
    int b = low  + rand() % range;
    int c = low  + rand() % range;
    // Sort the three sampled positions in place so arr[b] is the median.
    if (arr[a] > arr[b]) swap(arr[a], arr[b]);
    if (arr[a] > arr[c]) swap(arr[a], arr[c]);
    if (arr[b] > arr[c]) swap(arr[b], arr[c]);
    return arr[b];  // median value (elements at a, b, c now sorted by value)
}

// ---------------------------------------------------------------------------
// Three-way partition (Dutch National Flag algorithm).
//
// Returns {lt, gt} such that after the call:
//   arr[low..lt-1]  < pivot   (recurse left)
//   arr[lt..gt]    == pivot   (DONE — never recurse on equal region)
//   arr[gt+1..high] > pivot   (recurse right)
//
// WHY THREE-WAY over standard Hoare partition:
//   On data with many duplicates (same_value, repeated_values) the equal
//   region can be O(N) elements, so those N elements are sorted in O(1)
//   additional work instead of being re-partitioned O(log N) more times.
//   This gives O(N) total comparisons on all-equal input and a significant
//   constant-factor improvement on any low-entropy distribution.
//   On uniform random data the performance is identical to Hoare partition.
//
// This is the partition scheme used by Java's Arrays.sort (dual-pivot quicksort)
// and by pdqsort (the algorithm behind Rust's sort_unstable and C++20 ranges::sort).
// ---------------------------------------------------------------------------
pair<int, int> partition3way(vector<float>& arr, int low, int high) {
    float pivot = medianOfThree(arr, low, high);
    int lt = low;    // arr[low..lt-1]  < pivot
    int gt = high;   // arr[gt+1..high] > pivot
    int i  = low;    // current element

    unsigned long long local_cmp = 0;

    while (i <= gt) {
        local_cmp++;
        if (arr[i] < pivot) {
            swap(arr[lt++], arr[i++]);
        } else {
            local_cmp++;
            if (arr[i] > pivot) {
                swap(arr[i], arr[gt--]);
                // Do NOT advance i: the element swapped from arr[gt] is unexamined.
            } else {
                i++;   // arr[i] == pivot
            }
        }
    }

    comparisons += local_cmp;
    return {lt, gt};
}

// Sequential quicksort (three-way partition + insertion-sort base case).
void quicksortSequential(vector<float>& arr, int low, int high) {
    if (high - low < INSERTION_THRESHOLD) {
        if (low < high) insertionSort(arr, low, high);
        return;
    }
    auto [lt, gt] = partition3way(arr, low, high);
    quicksortSequential(arr, low, lt - 1);
    quicksortSequential(arr, gt + 1, high);
}

// ---------------------------------------------------------------------------
// Parallel quicksort with thread budget.
//
// PARALLELISM STRATEGY — raw std::thread with static budget halving:
//   We chose std::thread over OpenMP because it makes the parallelism model
//   explicit and transparent for analysis. OpenMP task scheduling would adapt
//   to skewed partitions automatically (work stealing), but would hide the
//   overhead structure we want to measure.
//
// LIMITATION: Budget halving assigns equal threads to both partitions
//   regardless of their size. With three-way partition the left and right
//   regions are typically unequal, so one thread may be idle while the other
//   has work. This is the fundamental motivation for work-stealing schedulers.
// ---------------------------------------------------------------------------
void quicksortParallel(vector<float>& arr, int low, int high, int threads) {
    if (high - low < INSERTION_THRESHOLD) {
        if (low < high) insertionSort(arr, low, high);
        return;
    }
    if (threads <= 1 || (high - low) < THRESHOLD) {
        quicksortSequential(arr, low, high);
        return;
    }

    auto [lt, gt] = partition3way(arr, low, high);

    int left_threads  = threads / 2;
    int right_threads = threads - left_threads;

    thread t1(quicksortParallel, ref(arr), low,    lt - 1, left_threads);
    quicksortParallel(arr, gt + 1, high, right_threads);
    t1.join();
}

long getMemoryUsageKB() {
    struct rusage usage;
    getrusage(RUSAGE_SELF, &usage);
#ifdef __APPLE__
    return usage.ru_maxrss / 1024;
#else
    return usage.ru_maxrss;
#endif
}

bool isCorrect(const vector<float>& arr, const vector<float>& reference) {
    return arr == reference;
}

int main(int argc, char* argv[]) {
    if (argc != 4) {
        cerr << "Usage: " << argv[0] << " <filename> <mode> <num_threads>\n";
        return 1;
    }

    string filename = argv[1];
    string mode     = argv[2];
    int num_threads = stoi(argv[3]);

    // Fixed seed for reproducibility — random pivot selection in medianOfThree
    // uses rand() seeded here so results are identical across runs.
    srand(42);

    vector<float> arr;
    ifstream file(filename);
    if (!file.is_open()) { cerr << "Error opening file: " << filename << "\n"; return 1; }
    float val;
    while (file >> val) arr.push_back(val);
    file.close();

    int n = arr.size();
    if (n == 0) {
        cout << mode << ",0.000,0,0,0," << num_threads << "\n";
        return 0;
    }

    // NaN / Inf guard — both break strict-weak-ordering in std::sort.
    for (int idx = 0; idx < n; idx++) {
        if (isnan(arr[idx]) || isinf(arr[idx])) {
            cerr << "Error: NaN or Inf at index " << idx << "\n";
            return 1;
        }
    }

    vector<float> reference = arr;
    sort(reference.begin(), reference.end());

    comparisons = 0;
    auto start = chrono::high_resolution_clock::now();

    if (mode == "seq")       quicksortSequential(arr, 0, n - 1);
    else if (mode == "par")  quicksortParallel(arr, 0, n - 1, num_threads);
    else { cerr << "Invalid mode: " << mode << "\n"; return 1; }

    auto end = chrono::high_resolution_clock::now();
    double ms = chrono::duration_cast<chrono::microseconds>(end - start).count() / 1000.0;
    long mem  = getMemoryUsageKB();

    if (!isCorrect(arr, reference)) {
        cerr << "Correctness check FAILED!\n";
        return 1;
    }

    cout << mode << "," << fixed << setprecision(3) << ms << ","
         << mem << "," << comparisons.load() << "," << n << "," << num_threads << "\n";
    return 0;
}