// quicksort.cpp — sequential and parallel quicksort for float arrays.

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

const int THRESHOLD           = 10000; // subarray size below which parallel falls back to sequential
const int INSERTION_THRESHOLD = 32;    // subarray size below which we use insertion sort

atomic<unsigned long long> comparisons{0};

// Simple O(N^2) sort; faster than quicksort for very small subarrays.
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

// Picks a pivot as the median of three randomly chosen elements.
// Random sampling avoids worst-case behavior on sorted/reverse-sorted input.
float medianOfThree(vector<float>& arr, int low, int high) {
    if (high - low < 2) return arr[low];
    int range = high - low + 1;
    int a = low + rand() % range;
    int b = low + rand() % range;
    int c = low + rand() % range;
    if (arr[a] > arr[b]) swap(arr[a], arr[b]);
    if (arr[a] > arr[c]) swap(arr[a], arr[c]);
    if (arr[b] > arr[c]) swap(arr[b], arr[c]);
    return arr[b];
}

// Dutch National Flag partition: splits into <pivot, ==pivot, >pivot regions.
// Equal elements are never revisited, which helps on low-entropy data.
pair<int, int> partition3way(vector<float>& arr, int low, int high) {
    float pivot = medianOfThree(arr, low, high);
    int lt = low, gt = high, i = low;
    unsigned long long local_cmp = 0;
    while (i <= gt) {
        local_cmp++;
        if (arr[i] < pivot)      { swap(arr[lt++], arr[i++]); }
        else if (arr[i] > pivot) { local_cmp++; swap(arr[i], arr[gt--]); }
        else                     { i++; }
    }
    comparisons += local_cmp;
    return {lt, gt};
}

void quicksortSequential(vector<float>& arr, int low, int high) {
    if (high - low < INSERTION_THRESHOLD) {
        if (low < high) insertionSort(arr, low, high);
        return;
    }
    auto [lt, gt] = partition3way(arr, low, high);
    quicksortSequential(arr, low, lt - 1);
    quicksortSequential(arr, gt + 1, high);
}

// Spawns a thread for the left partition; runs the right partition on the current thread.
// Thread budget is halved at each level; falls back to sequential when budget is exhausted.
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
    thread t1(quicksortParallel, ref(arr), low, lt - 1, threads / 2);
    quicksortParallel(arr, gt + 1, high, threads - threads / 2);
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

    srand(42); // fixed seed for reproducible pivot selection

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