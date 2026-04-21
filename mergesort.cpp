#include <iostream>
#include <vector>
#include <thread>
#include <algorithm>

using namespace std;

// Might need to tune this later
const int THRESHOLD = 10000;

// Merge function
void merge(vector<int>& arr, int left, int mid, int right) {
    vector<int> temp(right - left + 1);

    int i = left;
    int j = mid + 1;
    int k = 0;

    while (i <= mid && j <= right) {
        if (arr[i] <= arr[j]) {
            temp[k++] = arr[i++];
        } else {
            temp[k++] = arr[j++];
        }
    }

    while (i <= mid) temp[k++] = arr[i++];
    while (j <= right) temp[k++] = arr[j++];

    for (int t = 0; t < temp.size(); t++) {
        arr[left + t] = temp[t];
    }
}

// Sequential merge sort
void mergesortSequential(vector<int>& arr, int left, int right) {
    if (left >= right) return;

    int mid = left + (right - left) / 2;

    mergesortSequential(arr, left, mid);
    mergesortSequential(arr, mid + 1, right);

    merge(arr, left, mid, right);
}

// Parallel merge sort with thread budget
void mergesortParallel(vector<int>& arr, int left, int right, int threads) {
    if (left >= right) return;

    if (threads <= 1 || (right - left) < THRESHOLD) {
        mergesortSequential(arr, left, right);
        return;
    }

    int mid = left + (right - left) / 2;

    int left_threads = threads / 2;
    int right_threads = threads - left_threads;

    thread t1(mergesortParallel, ref(arr), left, mid, left_threads);

    mergesortParallel(arr, mid + 1, right, right_threads);

    t1.join();

    merge(arr, left, mid, right);
}

// Helper wrapper
void runParallelMergesort(vector<int>& arr, int num_threads) {
    if (!arr.empty())
        mergesortParallel(arr, 0, arr.size() - 1, num_threads);
}

// Simple correctness check
bool isSorted(const vector<int>& arr) {
    for (int i = 1; i < arr.size(); i++) {
        if (arr[i] < arr[i - 1]) return false;
    }
    return true;
}

// Testing
int main() {
    vector<vector<int>> testCases = {
        {5, 2, 9, 1, 5, 6, 3, 7, 8, 4},
        {1, 2, 3, 4, 5},
        {5, 4, 3, 2, 1},
        {7, 7, 7, 7, 7},
        {42},
        {},
        {3, 1},
        {10, -1, 2, -5, 0, 3}
    };

    int threads = 4;
    
    cout << "PARALLEL MERGESORT RESULT:" << endl;
    for (int i = 0; i < testCases.size(); i++) {
        vector<int> arr = testCases[i];

        cout << "Test " << i + 1 << " original: ";
        for (int x : arr) cout << x << " ";
        cout << endl;

        runParallelMergesort(arr, threads);

        cout << "Test " << i + 1 << " sorted:   ";
        for (int x : arr) cout << x << " ";
        cout << endl;

        cout << "Sorted? " << (isSorted(arr) ? "Yes" : "No") << endl;
    }

    cout << "SEQUENTIAL MERGESORT RESULT:" << endl;
    for (int i = 0; i < testCases.size(); i++) {
        vector<int> arr = testCases[i];

        cout << "Test " << i + 1 << " original: ";
        for (int x : arr) cout << x << " ";
        cout << endl;

        mergesortSequential(arr, 0, arr.size() - 1);

        cout << "Test " << i + 1 << " sorted:   ";
        for (int x : arr) cout << x << " ";
        cout << endl;

        cout << "Sorted? " << (isSorted(arr) ? "Yes" : "No") << endl;
    }

    return 0;
}