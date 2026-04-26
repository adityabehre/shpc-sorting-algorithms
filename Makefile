all: quicksort mergesort

# Compiler flags for optimization and threading support
CXXFLAGS = -O3 -march=native -std=c++17 -pthread

quicksort:
	g++ $(CXXFLAGS) -o quicksort quicksort.cpp

mergesort:
	g++ $(CXXFLAGS) -o mergesort mergesort.cpp

clean:
	rm -f mergesort quicksort