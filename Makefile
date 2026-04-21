quicksort:
	g++ -o quicksort quicksort.cpp
	./quicksort

mergesort:
	g++ -o mergesort mergesort.cpp
	./mergesort

clean:
	rm -f mergesort
	rm -f quicksort