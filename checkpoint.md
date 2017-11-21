## Parallelizing Convolutional Neural Networks using NVIDIA's CUDA Architecture


### SUMMARY TILL DATE
We have written a single-threaded version of a convolutional neural network in python using numpy without any optimizations in memory footprint or locality. This implementation takes about 7 hours to run on the MNIST dataset. Currently, this is way too slow for any practical application. We have also installed PyCUDA and finished a tutorial to understand the basics of how to transfer data between the CPU/GPU, how to create grids and blocks and synchronize threads. We have also tested it out on smaller and easily parallelizable functions and have gotten expected results.

### NEXT STEPS(TIMELINE) :-  
Over the week of **November 20th – 26th**, we plan to convert the convolution and subsampling layers into one CUDA function to reduce our memory footprint. Also, we will write out our main function that will make all the memory transfers between the CPU and GPU. At the same time, we plan to optimize the back-propagation algorithm to “push” the error signals so that we can get better locality in this step than what we currently have. We expect to get better memory utilization implementing this and as memory is a crucial bottleneck for GPUs, we expect a decent speedup. We will split this work up so that they can be done concurrently.
 
**November 27th – 3rd December**: During this week, we will first look for suitable python code profilers to help us find any bottlenecks in our code. We will also continue optimizing our code for CUDA making changes in batch sizes to fit the memory capacity of the GPU hardware - so to train on as many images as possible before loading a new set. We will also implement the circular buffer to hold the gradients so we can reuse this data. We expect this part to take a non-trivial amount of time as we are unsure of the speedups we will get and will base our further work on the results from the profiler. 
 
**December 4th – 12th December**: At this stage, we will continue on any left over work from the previous week and start working on producing graphs of speedups, memory latencies, and code profiling for the poster session. We will also work on the final report. 


### Issues:

We are slightly behind our initially proposed schedule and as a result we will have to devote a significant portion of our time these coming weeks on working on the project to achieve our goals. As a result, we don’t think we can develop a framework for parallelizing convolutional neural nets in general but we are still on track to highly optimize our current CNN implementation for GPU’s. 
### Poster Session Delivarables:
We will be showing speedup graphs along with detailed timing measurements comparing our sequential version of the code as well as our speedups after implementing each technique. We will be timing our speedups after 5 stages:
1. Initial single threaded version on python using numpy
2. After initial trivial optimizations and implementing some preliminary optimizations  presented in Lecture 27
3. After combining convolution and subsampling algorithms into one step as well as optimizing the back propagation algorithm
4. After adding a circular buffer to reuse the gradients
5. After trying to remove any bottlenecks that we find

We also plan on showing images comapring the different stages our sequential and parallel implementations actually begin to discern the digit inputted to it. 

