// mergesort.cpp — sequential and parallel mergesort for float arrays.
// Usage: ./mergesort <file> <seq|par> <num_threads>

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

const int THRESHOLD = 10000; // subarray size below which parallel falls back to sequential
const int INSERTION_THRESHOLD = 32;    // subarray size below which we use insertion sort

atomic<unsigned long long> comparisons{0};

// Simple O(N^2) sort; faster than mergesort for very small subarrays.
void insertionSort(vector<float>& arr, int low, int high) {
    unsigned long long local_cmp = 0;
    for (int i = low + 1; i <= high; i++) {
        float key = arr[i];
        int j = i - 1;
        while (j >= low) {
            local_cmp++;
            if (arr[j] > key) {
                arr[j + 1] = arr[j];
                j--;
            }
            else break;
        }
        arr[j + 1] = key;
    }
    comparisons += local_cmp;
}

// Merges two sorted halves into one using a pre-allocated scratch buffer.
// Using a shared scratch buffer avoids repeated heap allocation across calls.
void merge(vector<float>& arr, vector<float>& scratch, int left, int mid, int right) {
    int i = left, j = mid + 1, k = left;
    unsigned long long local_cmp = 0;
    while (i <= mid && j <= right) {
        local_cmp++;
        scratch[k++] = (arr[i] <= arr[j]) ? arr[i++] : arr[j++];
    }
    while (i <= mid) {
        scratch[k++] = arr[i++];
    }
    while (j <= right) {
        scratch[k++] = arr[j++];
    }
    for (int t = left; t <= right; t++) {
        arr[t] = scratch[t];
    }
    comparisons += local_cmp;
}

void mergesortSequential(vector<float>& arr, vector<float>& scratch, int left, int right) {
    if (right - left < INSERTION_THRESHOLD) {
        if (left < right) {
            insertionSort(arr, left, right);
        }
        return;
    }
    int mid = left + (right - left) / 2;
    mergesortSequential(arr, scratch, left, mid);
    mergesortSequential(arr, scratch, mid + 1, right);
    merge(arr, scratch, left, mid, right);
}

// Sorts both halves in parallel, then merges sequentially.
// The sequential merge is the Amdahl bottleneck — it limits max speedup.
void mergesortParallel(vector<float>& arr, vector<float>& scratch, int left, int right, int threads) {
    if (right - left < INSERTION_THRESHOLD) {
        if (left < right) {
            insertionSort(arr, left, right);
        }
        return;
    }
    if (threads <= 1 || (right - left) < THRESHOLD) {
        mergesortSequential(arr, scratch, left, right);
        return;
    }
    int mid = left + (right - left) / 2;
    thread t1(mergesortParallel, ref(arr), ref(scratch), left, mid, threads / 2);
    mergesortParallel(arr, scratch, mid + 1, right, threads - threads / 2);
    t1.join();
    merge(arr, scratch, left, mid, right);
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
    if (!file.is_open()) {
        cerr << "Error opening file: " << filename << "\n";
        return 1;
    }
    float val;
    while (file >> val) {
        arr.push_back(val);
    }
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

    vector<float> scratch(n); // single buffer reused across all merge() calls

    comparisons = 0;
    auto start = chrono::high_resolution_clock::now();

    if (mode == "seq") {
         mergesortSequential(arr, scratch, 0, n - 1);
    }
    else if (mode == "par") {
        mergesortParallel(arr, scratch, 0, n - 1, num_threads);
    }
    else {
        cerr << "Invalid mode: " << mode << "\n"; 
        return 1; 
    }

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