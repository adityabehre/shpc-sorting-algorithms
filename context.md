

execution_time_ms: Average runtime in milliseconds.
memory_usage_kb: Memory consumed in kilobytes.
comparison_count: Number of comparisons mades
Input size
Number of threads
Technical Consideration: The Threshold
In your implementation, remember to include a threshold. In parallel divide-and-conquer, if you keep spawning threads for every tiny sub-problem, your program will slow down significantly. A common strategy is:
If array_size > threshold: Spawn new threads for the left and right halves.
If array_size <= threshold: Switch to the sequential version of the algorithm.
This threshold is a perfect variable to test in your experiments. You can analyze how the optimal threshold changes when switching from simple synthetic integers to complex real-world wildfire data.
Some reasoning or hypothesis for results
Team Plan

Professor Parikh's comments on our proposal:
There are many parallelization schemes for sorting algorithms. For your project, you will have to provide which parallelization strategies you will be using and why. I expect some reasoning as to why you chose the one that you did, it's short comings and when the others may be more appropriate solutions. Also specify what datasets you going to try sorting on, and spend some time thinking on how you are going to evaluate if your implementation is doing well or not. I'd like to see some reasoning on what performance can be expected. 
Title
Correctness and Performance Analysis of Parallel Sorting Algorithms
Team Member 1, Team Member 2, Team Member 3
 Team Name (Group #)
 March 8, 2026

1 Project Description
As Rosca and C˘arbureanu explain, sorting algorithms “play a fundamental role” in computer
science, and they influence “the efficiency of applications that operate on large-scale datasets”
which is important, as modern programs often handle large datasets [1].

The goal of this project is to explore how parallelism can improve the performance of sorting
algorithms while still maintaining correctness. Specifically, we plan to implement both sequential
and parallel versions of the quicksort and merge sort algorithms, and analyze how their performance changes when multiple threads are used. Quicksort and merge sort are very popular algorithms and they have average-case time complexities of O(n log n). Both algorithms also follow a divide-and-conquer structure, which makes them well suited for parallel execution. By implementing both sequential and parallel versions, we can directly compare how parallelization affects the performance of each algorithm. After implementing these algorithms, we plan on running experiments using different input sizes and different numbers of threads. By comparing the runtime of sequential and parallel implementations, we hope to better understand when parallelization provides meaningful performance improvements and when the overhead from thread creation or synchronization may reduce performance.

Background and Relevance (Improved with Sources)
Sorting is one of the most well-studied problems in computer science, and efficient sorting is critical in a wide-range of applications. As datasets grow, the performance of sorting algorithms become increasingly important, motivating the use of parallel computing to reduce runtime.
As previously mentioned, there are some Sequential sorting algorithms that achieve O(n log n)
average-case time complexity, but they process data using a single thread, leaving multi-core hardware underutilized. Both algorithms break the input into smaller subproblems, solving them independently, and combining the results; which can map naturally onto parallel execution, where independent subproblems can be handled by separate threads simultaneously. However, parallelization introduces new challenges. Correctness becomes harder to guarantee when
multiple threads access and modify shared data concurrently. This may result in race conditions or inconsistent results. Performance gains are also not automatic: spawning threads and synchronizing their work carries overhead that can outweigh the benefits, especially on small inputs or with too many threads. This project is extremely relevant to the topics we have learned this semester. We must ensure that our parallel implementations produce the same output as their sequential counterparts under all conditions. On the performance side, we will measure how runtime changes with input size and thread count, and assess the conditions under which parallelism provides efficient behavior.
Together, these goals make parallel sorting a compelling case study in correctness and efficiency

3 Expected Results
As mentioned earlier, we will be implementing four algorithms, namely sequential quicksort, sequential merge sort, parallel quicksort, and parallel merge sort.
The first step will be to make sure that each of our implementations produces the correct results.
To be more specific, each sorting algorithm should output the same elements as the input but in
sorted order for it to be correct.
After that, we will measure the runtime of each algorithm using different input sizes and different
numbers of threads. These tests will allow us to compare how the sequential and parallel implementations of the algorithms perform under different conditions. We will then analyze the results
to see how much parallelization truly improves performance. We will use visuals such as tables and graphs to show the results and discuss which algorithms perform best and why

Team Plan
or our project, we will divide responsibilities by using active communication and by the areas of
work we have already discussed. For active communication, we will make sure that we are aware of each other’s schedules and respect each other’s time during work sessions. As for the areas of work we have already discussed, we will evenly split the four algorithms evenly among the two of us. In addition, Akshat has agreed to develop unit tests to ensure code correctness. Finally, for viewing results, Aaron has agreed to make the code for visuals such as tables and graph

Submission details:

Submit your project via GitHub/GitLab and provide a link to your repository on Canvas.
Make sure to keep your repository private and give access to me. My Github/Gitlab username: dnparikh.
The README on your repository must contain instructions on how to compile and run your project. Please include all instructions for getting/building/installing any dependencies.
I will be grading based on a branch called code-freeze, so please make sure you have a working final version of the project on that branch.
Your repository should also include your final report. Create a doc or report directory that includes your tex file and final pdf for your report.
Your report should be 3–5 pages and must include:
the necessary background to understand your project
a description of what you accomplished
a description of the artifacts you’ve produced and/or graphs showing the performance of what you have done.
a list of references (does not need to be MLA but make sure the all necessary information to find the reference is there)

Kaggle datasets:

https://www.kaggle.com/datasets/bekiremirhanakay/benchmark-dataset-for-sorting-algorithms

https://www.kaggle.com/datasets/rtatman/188-million-us-wildfires

Threshold for array size to change to parallel alg. 

We’re going to use task-based parallelism for both algorithms. For quicksort, after we partition the array, the left and right halves are independent, so we can sort them at the same time on different threads. For merge sort, we’ll recursively split the array and sort each half in parallel, then merge them at the end.
We’re choosing this approach because it fits naturally with the divide-and-conquer structure of both algorithms and is relatively easy to implement. The main tradeoff is that parallel quicksort can have load imbalance if the splits are uneven, while merge sort is more balanced but uses extra memory for merging.

Here are the main alternative strategies, explained super briefly
One is data parallelism (chunk-based sorting). You split the array into equal chunks, sort each chunk in parallel, and then merge everything. This is easy to implement and balances work well, but the final merge step can be expensive and becomes a bottleneck.
Another is parallel partitioning (more advanced quicksort). Instead of just parallelizing the recursive calls, multiple threads work together to partition the array at the same time. This can improve performance, but it’s much harder to implement correctly and introduces more synchronization overhead.
There’s also pipeline parallelism, where different stages (like splitting, sorting, merging) run concurrently. This can improve throughput, but it’s not a great fit for sorting since the stages aren’t cleanly separable.
So overall, task parallelism (what you chose) is the best balance — it’s simple, maps well to the algorithms, and still gives solid speedups, even if it’s not the absolute most optimized approach.

https://www.sciencedirect.com/science/article/abs/pii/S0167739X24001225?via%3Dihub
Merge sort: https://redixhumayun.github.io/systems/2023/12/29/parallel-merge-sort.html


Test with 1, 2, 4, 8, 16 threads!!
