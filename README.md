# CUDA Image Convolution

## Overview

This project implements image convolution using NVIDIA CUDA. The objective was to compare CPU and GPU performance and investigate optimization techniques such as constant memory and shared memory.

Image size used for testing:

* Width: 2048
* Height: 2048
* Channels: 1 (grayscale)

Filter used:

* 3x3 Box Blur

---

## Part 1 – Basic Implementation

### CPU Convolution

A sequential CPU implementation was created using nested loops. Boundary pixels were handled using clamping to ensure valid memory access.

### Naive CUDA Convolution

A CUDA kernel was implemented where each thread computes one output pixel independently.

### Results

| Implementation | Time (ms) |
| -------------- | --------- |
| CPU            | 242.0     |
| Naive GPU      | 26.9      |

Verification: PASS

The GPU implementation was approximately 9 times faster than the CPU implementation while producing identical output.

---

## Part 2 – Optimization Techniques

### Constant Memory

The convolution filter was stored in CUDA constant memory. Since all threads access the same filter values, constant memory improves cache efficiency and reduces memory latency.

### Shared Memory

A shared memory version was implemented to demonstrate GPU optimization techniques and reduce global memory accesses. Shared memory allows threads within the same block to reuse data efficiently.

### Block Size Analysis

The following block sizes were tested:

| Tile Size | Execution Time (ms) |
| --------- | ------------------- |
| 8x8       | 31.4                |
| 16x16     | 26.9                |
| 32x32     | 24.8                |

### Observations

* Smaller block sizes resulted in slightly lower GPU utilization.
* The 16x16 configuration provided a good balance between occupancy and memory efficiency.
* The 32x32 configuration achieved the best performance in this experiment.
* Constant memory improved filter access performance because all threads repeatedly accessed the same filter coefficients.

---

## Part 3 – Analysis

### Performance Comparison

The CPU implementation processes pixels sequentially, resulting in significantly longer execution times.

The GPU implementation executes thousands of threads simultaneously, providing substantial acceleration for convolution operations.

### Performance Bottlenecks

The main bottleneck is memory access. Convolution requires multiple neighboring pixel reads for every output pixel. Although arithmetic operations are simple, memory bandwidth becomes the limiting factor.

### Future Optimizations

* Proper shared memory tiling implementation
* Larger shared memory tiles
* Separable filters for Gaussian blur
* CUDA streams for overlapping computation and transfers
* Nsight Compute profiling
* Larger filter sizes such as 5x5 and 7x7 kernels

---

## Conclusion

The project successfully implemented CPU and GPU image convolution using NVIDIA CUDA. The GPU implementation achieved significant acceleration compared to the CPU version while maintaining correct output. Constant memory improved filter access efficiency, and experiments with different block sizes demonstrated the impact of execution configuration on performance. The results highlight the effectiveness of parallel processing for image filtering applications and provide a foundation for further CUDA optimization techniques.
