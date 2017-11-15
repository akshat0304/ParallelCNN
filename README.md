## Parallelizing Convolutional Neural Networks using NVIDIA's CUDA Architecture


### SUMMARY
We are going to implement a parallel Convolutional Neural Network (CNN) on the NVIDIA CUDA GPU architecture. We are going to start with an existing sequential implementation of a CNN and parallelize both the back and forward propagation phases along with reduce memory footprint and improve memory efficiency to achieve a higher speed up at the cost of the lowest possible accuracy drop.


### BACKGROUND
Over the last decade, there has been a significant outgrowth in academic research and industry over the use of CNNs for tasks related to image recognition and classification. This is a consequence of the architecture of CNNs – which takes an image as input and extracts more and more complex features at each layer at a lower resolution. After these feature-extraction layers, there are a few activation layers which use the resulting complex features to make a “decision” about the image. 

The biggest drawback with using CNNs is training them as they require very large datasets (usually in the order of hundreds of thousands) to converge to their global optima. As a result, they are very computationally intensive and can take many hours and even days to run on traditional CPUs. 

In this project, we aim to exploit the parallelism in each layer of the CNN and modify the back-propagation algorithm to make it suited for CUDA architecture and achieve high speedups.  

### CHALLENGES

- **Low Arithematic Intensity :-** The computational results of each layer need to be shared among all threads as the computation of the next layer depends on the results of the previous layer. Similarly, the back-propagation step also requires results from previous layers. As a result, CNNs inherently have a high communication to computation ratio. 

- **Memory Limitation :-** The number of training examples that can be computed in parallel is significantly restricted by the amount of global memory available to the GPU. For example, a 512x512 image with 32-bit precision takes up ~0.83 MB of space. 

- **Dependencies :-** As discussed earlier, layer l depends on the result of layer l – 1. So there is parallelism within a layer but not between layers. 

Overcoming the aforementioned challenges would require us to optimize the CNN architecture for lower memory utilization and fewer communication requirements. This would involve exploiting spatial locality so that threads in the same block work on memory close to each other, reusing memory so that it doesn’t have to be loaded numerous times into the GPU from the CPU, as well as modifying the back-propagation algorithm to achieve some sort of parallelism

### RESOURCES

We will be starting from an existing implementation of a commonly used Convolutional Neural Network and work on parallelizing it. We will be using the NVIDIA GPU that was made available to use in Assignment 2 and running the application on PYCUDA which is a python API for execution on a NVIDIA GPU. 

### PLATFORM CHOICE
We chose PYCUDA because a lot of CNN’s are written in Python and using PYCUDA and numpy will make our job much easier. 

### APPROACH

We have limited memory resources on the GPU hardware - 

We perform convolution and subsampling in one step instead of two increasing temporal locality and reduce reloading of memory. 

##### To improve locality - 
During the back-propagation step, the loss function (error) is usually “pulled” by lower layers from higher layers. This method does not exploit locality as a weight in the lower level may be affected by different weights in the higher layer. As a result, we will try employ a technique called “pushing” where for each unit in the higher layer, we will update a fixed number of units in the lower layer so that threads will now modify contiguous memory - which is better suited for CUDA architecture. 

We also plan to use a circular buffer in order to get the most out of the extremely limited shared memory. Even if we use the whole 96 KB of shared memory only a fraction of the source feature maps can be loaded. Using a circular buffer that only holds a small region of the feature maps helps us make sure that in each iteration only some of the rows of the buffer need to be exchanged. 

### GOALS AND DELIVERABLES

#### Plan to Achieve
In this project, we plan to develop a parallelized convolutional neural network framework running on the CUDA GPU. We will start working on an existing implementation and make modifications to it such as trying to do the convolution and subsampling in one step to reduce the memory footprint of each layer as well as reusing data from the error signals of the higher layer.  

We plan to build upon the knowledge of Lecture 25 and add additional optimizations to reach a speedup that is much faster than the sequential implementation and has a lower memory footprint and lower computation cost.

#### Hope to Achieve
If we are ahead of schedule, we plan to turn this parallelized neural network into a framework that helps you define and train a neural network.

### DEMO
We will be demonstrating our parallelized application in action and comparing the running times on small datasets. We will also be presenting speedup graphs comparing the sequential and parallel implementations of our neural network as well as describing some optimizations that we used. 

### REFERENCES

1.	P. Y. Simard, D. Steinkraus, and J. C. Platt. Best Practice for Convolutional Neural Networks Applied to Visual Document Analysis. In ICDAR, 2003. 


### SCHEDULE

**Date** | **Plan to Acheive** | **Status**
---------| --------------------| ----------
**October 30th-November 5th**  |Write Proposal. Learn PYCUDA. | 
**November 6th-November 12th** | Write out serial CPU implementation with 4 convolution/subsampling layers and 2 fully connected layers and test for correctness and timing on MNIST Dataset. | 
**November 13th-November 19th** |Write naive CUDA functions for the convolution/subsampling layer and the backpropogation algorithm. Write Checkpoint Report |
**November 20th-November 26th** | Make memory optimizations as detailed in the proposal and implement a circular buffer to hold the backpropogation gradients. |
**November 27th –December 3rd** | Make memory optimizations as detailed in the proposal and implement a circular buffer to hold the backpropogation gradients.  |
**December 3rd-December 9th**  | Test our GPU implementation and take measurementsIf all previous tasks are completed, start working on a framework that allows you to define a neural network and train it. After work is completed, test and time the framework. | 
**December 10th-December 12th** | Write out final report and make presentation poster.




