#include <iostream>
#include <vector>
#include <thread>
#include <algorithm>

using namespace std;

// Might need to tune this later
const int THRESHOLD = 10000;

// Partition function
int partition(vector<int>& arr, int low, int high) {
    int pivot = arr[high];
    int i = low - 1;

    for (int j = low; j < high; j++) {
        if (arr[j] <= pivot) {
            i++;
            swap(arr[i], arr[j]);
        }
    }

    swap(arr[i + 1], arr[high]);
    return i + 1;
}

// Sequential quicksort
void quicksortSequential(vector<int>& arr, int low, int high) {
    if (low < high) {
        int pi = partition(arr, low, high);
        quicksortSequential(arr, low, pi - 1);
        quicksortSequential(arr, pi + 1, high);
    }
}

// Parallel quicksort with thread budget
void quicksortParallel(vector<int>& arr, int low, int high, int threads) {
    if (low >= high) return;

    // If small or no threads left, go sequential
    if (threads <= 1 || (high - low) < THRESHOLD) {
        quicksortSequential(arr, low, high);
        return;
    }

    int pi = partition(arr, low, high);

    // Split the thread budget
    int left_threads = threads / 2;
    int right_threads = threads - left_threads;

    thread t1(quicksortParallel, ref(arr), low, pi - 1, left_threads);

    // Do the right side in current thread
    quicksortParallel(arr, pi + 1, high, right_threads);

    t1.join();
}

// Helper wrapper
void runParallelQuicksort(vector<int>& arr, int num_threads) {
    quicksortParallel(arr, 0, arr.size() - 1, num_threads);
}

// Simple correctness check
bool isSorted(const vector<int>& arr) {
    for (int i = 1; i < arr.size(); i++) {
        if (arr[i] < arr[i - 1]) return false;
    }
    return true;
}

// Added this for testing, dont need this once we add benchmarking
int main() {
    vector<vector<int>> testCases = {
        {5, 2, 9, 1, 5, 6, 3, 7, 8, 4},   // random
        {1, 2, 3, 4, 5},                  // already sorted
        {5, 4, 3, 2, 1},                  // reverse sorted
        {7, 7, 7, 7, 7},                  // all duplicates
        {42},                             // single element
        {},                               // empty
        {3, 1},                           // two elements
        {10, -1, 2, -5, 0, 3}            // includes negatives
    };

    int threads = 4;

    for (int i = 0; i < testCases.size(); i++) {
        vector<int> arr = testCases[i];

        cout << "Test " << i + 1 << " original: ";
        for (int x : arr) cout << x << " ";
        cout << endl;

        runParallelQuicksort(arr, threads);

        cout << "Test " << i + 1 << " sorted:   ";
        for (int x : arr) cout << x << " ";
        cout << endl;

        cout << "Sorted? " << (isSorted(arr) ? "Yes" : "No") << endl;
    }

    return 0;
}