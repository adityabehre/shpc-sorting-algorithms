all: quicksort mergesort

# -O3         : Full optimizations (loop unrolling, vectorization, inlining)
# -march=native: Enable all CPU-specific instructions (AVX2, SSE4, etc.) for
#               the host machine. Gives ~10-30% speedup on float comparisons
#               vs generic -O3. NOTE: Binaries are NOT portable across machines
#               with different ISAs — all benchmark runs must occur on the same
#               hardware to ensure fair comparison.
# -std=c++17  : Use C++17 for better std library support
# -pthread    : Link POSIX thread library for std::thread
CXXFLAGS = -O3 -march=native -std=c++17 -pthread

quicksort:
	g++ $(CXXFLAGS) -o quicksort quicksort.cpp

mergesort:
	g++ $(CXXFLAGS) -o mergesort mergesort.cpp

clean:
	rm -f mergesort quicksort