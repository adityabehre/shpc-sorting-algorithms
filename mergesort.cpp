#include <iostream>
#include <vector>
#include <thread>
#include <algorithm>
#include <fstream>
#include <chrono>
#include <atomic>
#include <iomanip>
#include <cmath>
#include <sys/resource.h>

using namespace std;

const int THRESHOLD          = 10000;
const int INSERTION_THRESHOLD = 32;

atomic<unsigned long long> comparisons{0};

// ---------------------------------------------------------------------------
// Insertion sort — base case for small subarrays.
// Same reasoning as in quicksort.cpp: at N < 32 the cache and branch-
// predictor benefits of insertion sort outweigh recursive overhead.
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
// Merge using a pre-allocated scratch buffer.
//
// WHY PRE-ALLOCATED SCRATCH:
//   The textbook implementation allocates a new vector<float> temp on every
//   merge() call. For N=10M elements, this is ~20M heap allocations during
//   a full sort, each of which must zero-initialize the memory and later free
//   it. This creates significant allocator contention — especially in the
//   parallel version where multiple threads call merge() simultaneously.
//
//   By allocating a single scratch buffer of size N once in main() and passing
//   it by reference, we eliminate all per-call heap traffic. The buffer is
//   safe to share between parallel sub-sorts because left and right threads
//   operate on non-overlapping index ranges:
//     Left  thread: arr[left..mid],   scratch[left..mid]
//     Right thread: arr[mid+1..right], scratch[mid+1..right]
//   After both threads finish, the sequential merge step uses scratch[left..right]
//   as a whole — no overlap with any live thread at that point.
// ---------------------------------------------------------------------------
void merge(vector<float>& arr, vector<float>& scratch, int left, int mid, int right) {
    int i = left, j = mid + 1, k = left;
    unsigned long long local_cmp = 0;

    while (i <= mid && j <= right) {
        local_cmp++;
        scratch[k++] = (arr[i] <= arr[j]) ? arr[i++] : arr[j++];
    }
    while (i <= mid)  scratch[k++] = arr[i++];
    while (j <= right) scratch[k++] = arr[j++];
    for (int t = left; t <= right; t++) arr[t] = scratch[t];

    comparisons += local_cmp;
}

// Sequential merge sort with insertion-sort base case and pre-allocated scratch.
void mergesortSequential(vector<float>& arr, vector<float>& scratch, int left, int right) {
    if (right - left < INSERTION_THRESHOLD) {
        if (left < right) insertionSort(arr, left, right);
        return;
    }
    int mid = left + (right - left) / 2;
    mergesortSequential(arr, scratch, left, mid);
    mergesortSequential(arr, scratch, mid + 1, right);
    merge(arr, scratch, left, mid, right);
}

// ---------------------------------------------------------------------------
// Parallel merge sort with thread budget and pre-allocated scratch buffer.
//
// PARALLELISM STRATEGY — same std::thread / budget-halving approach as
//   quicksort.  The sequential merge step after joining is the Amdahl
//   bottleneck: it processes O(N) elements on a single thread regardless of
//   how many threads sorted the two halves.  The fitted f_seq ≈ 0.52
//   (from our benchmark data) implies a theoretical max speedup of ~1.9×,
//   which aligns with our observed 1.7-2.1× at 16 threads.
//
// SCRATCH BUFFER SAFETY (see merge() comment above):
//   Left and right threads use non-overlapping index ranges of scratch[], so
//   there is no data race on the scratch buffer during parallel sorting.
// ---------------------------------------------------------------------------
void mergesortParallel(vector<float>& arr, vector<float>& scratch, int left, int right, int threads) {
    if (right - left < INSERTION_THRESHOLD) {
        if (left < right) insertionSort(arr, left, right);
        return;
    }
    if (threads <= 1 || (right - left) < THRESHOLD) {
        mergesortSequential(arr, scratch, left, right);
        return;
    }

    int mid          = left + (right - left) / 2;
    int left_threads  = threads / 2;
    int right_threads = threads - left_threads;

    // Left thread: arr[left..mid], scratch[left..mid]  — no overlap with right.
    thread t1(mergesortParallel, ref(arr), ref(scratch), left, mid, left_threads);
    mergesortParallel(arr, scratch, mid + 1, right, right_threads);
    t1.join();

    merge(arr, scratch, left, mid, right);  // sequential Amdahl bottleneck
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

    for (int idx = 0; idx < n; idx++) {
        if (isnan(arr[idx]) || isinf(arr[idx])) {
            cerr << "Error: NaN or Inf at index " << idx << "\n";
            return 1;
        }
    }

    vector<float> reference = arr;
    sort(reference.begin(), reference.end());

    // Pre-allocate one scratch buffer of size N, reused across all merge calls.
    // This eliminates ~20M heap allocations for N=10M (one per merge() call).
    vector<float> scratch(n);

    comparisons = 0;
    auto start = chrono::high_resolution_clock::now();

    if (mode == "seq")       mergesortSequential(arr, scratch, 0, n - 1);
    else if (mode == "par")  mergesortParallel(arr, scratch, 0, n - 1, num_threads);
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