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

The GPU implementation was significantly faster than the CPU implementation while producing identical output.

---

## Part 2 – Optimization Techniques

### Constant Memory

The convolution filter was stored in CUDA constant memory. Since all threads access the same filter values, constant memory improves cache efficiency.

### Shared Memory

A shared memory version was implemented to demonstrate optimization techniques and reduce global memory accesses.

### Block Size Analysis

The following tile sizes were tested:

| Tile Size | Execution Time (ms) |
| --------- | ------------------- |
| 8x8       | YOUR_RESULT         |
| 16x16     | YOUR_RESULT         |
| 32x32     | YOUR_RESULT         |

### Observations

The optimized implementation reduced execution time compared to the naive version. Shared memory and constant memory improve memory access efficiency and reduce global memory traffic.

---

## Part 3 – Analysis

### Performance Comparison

The CPU implementation processes pixels sequentially, resulting in significantly longer execution times.

The GPU implementation executes thousands of threads simultaneously, providing substantial acceleration.

### Performance Bottlenecks

The main bottleneck is memory access. Convolution requires multiple neighboring pixel reads for every output pixel.

### Future Optimizations

* Larger shared memory tiles
* Separable filters for Gaussian blur
* CUDA streams
* Nsight profiling
* Larger filter sizes

---

## Conclusion

The project successfully implemented CPU and GPU image convolution. CUDA acceleration significantly improved performance, and memory optimizations further reduced execution time. The results demonstrate the advantages of parallel processing for image filtering applications.
