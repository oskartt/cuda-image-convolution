// Standard input/output library for printf
#include <stdio.h>

// Standard library for malloc, free, exit
#include <stdlib.h>

// Math library
#include <math.h>

// Time library for CPU timing
#include <time.h>

// CUDA runtime library
#include <cuda_runtime.h>

// Image width
#define WIDTH 2048

// Image height
#define HEIGHT 2048

// Number of image channels, 1 means grayscale
#define CHANNELS 1

// CUDA block size
#define BLOCK_SIZE 16

// Constant GPU memory for filter values
__constant__ float d_filter[81];

// 3x3 box blur filter
float boxBlur3x3[9] = {
    1 / 9.0f, 1 / 9.0f, 1 / 9.0f,
    1 / 9.0f, 1 / 9.0f, 1 / 9.0f,
    1 / 9.0f, 1 / 9.0f, 1 / 9.0f};

// 3x3 sharpen filter
float sharpen[9] = {
    0, -1, 0,
    -1, 5, -1,
    0, -1, 0};

// Macro for checking CUDA errors
#define CHECK_CUDA_ERROR(call)                                    \
    {                                                             \
        cudaError_t err = call;                                   \
        if (err != cudaSuccess)                                   \
        {                                                         \
            fprintf(stderr, "CUDA Error: %s\n", cudaGetErrorString(err)); \
            exit(EXIT_FAILURE);                                   \
        }                                                         \
    }

// Structure used to store image information
typedef struct
{
    // Pointer to image pixel data
    unsigned char *data;

    // Image width
    int width;

    // Image height
    int height;

    // Number of channels
    int channels;
} Image;

// Function to keep pixel value between 0 and 255
unsigned char clampUchar(float v)
{
    // If value is below 0, return 0
    if (v < 0) return 0;

    // If value is above 255, return 255
    if (v > 255) return 255;

    // Otherwise convert float to unsigned char
    return (unsigned char)v;
}

// Function that creates a simple test image
void generateImage(Image *img)
{
    // Loop through each row
    for (int y = 0; y < img->height; y++)

        // Loop through each column
        for (int x = 0; x < img->width; x++)

            // Create a simple gradient pixel value
            img->data[y * img->width + x] = (unsigned char)((x + y) % 256);
}

// CPU convolution function
void convolutionCPU(const Image *input, Image *output, const float *filter, int filterWidth)
{
    // Radius is how far the filter reaches from the center
    int radius = filterWidth / 2;

    // Loop through each image row
    for (int y = 0; y < input->height; y++)
    {
        // Loop through each image column
        for (int x = 0; x < input->width; x++)
        {
            // Store convolution result
            float sum = 0.0f;

            // Loop through filter rows
            for (int fy = -radius; fy <= radius; fy++)
            {
                // Loop through filter columns
                for (int fx = -radius; fx <= radius; fx++)
                {
                    // Calculate neighbour pixel x position
                    int ix = x + fx;

                    // Calculate neighbour pixel y position
                    int iy = y + fy;

                    // Clamp x position to left edge
                    if (ix < 0) ix = 0;

                    // Clamp y position to top edge
                    if (iy < 0) iy = 0;

                    // Clamp x position to right edge
                    if (ix >= input->width) ix = input->width - 1;

                    // Clamp y position to bottom edge
                    if (iy >= input->height) iy = input->height - 1;

                    // Read pixel value
                    float pixel = input->data[iy * input->width + ix];

                    // Read filter value
                    float coeff = filter[(fy + radius) * filterWidth + (fx + radius)];

                    // Add weighted pixel value to sum
                    sum += pixel * coeff;
                }
            }

            // Save final clamped pixel value
            output->data[y * output->width + x] = clampUchar(sum);
        }
    }
}

// Naive GPU convolution kernel
__global__ void convolutionKernelNaive(unsigned char *input, unsigned char *output,
                                       int filterWidth,
                                       int width, int height)
{
    // Calculate pixel x position for this GPU thread
    int x = blockIdx.x * blockDim.x + threadIdx.x;

    // Calculate pixel y position for this GPU thread
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    // Stop if thread is outside image bounds
    if (x >= width || y >= height)
        return;

    // Calculate filter radius
    int radius = filterWidth / 2;

    // Store convolution result
    float sum = 0.0f;

    // Loop through filter rows
    for (int fy = -radius; fy <= radius; fy++)
    {
        // Loop through filter columns
        for (int fx = -radius; fx <= radius; fx++)
        {
            // Calculate neighbour pixel x position
            int ix = x + fx;

            // Calculate neighbour pixel y position
            int iy = y + fy;

            // Clamp x position to left edge
            if (ix < 0) ix = 0;

            // Clamp y position to top edge
            if (iy < 0) iy = 0;

            // Clamp x position to right edge
            if (ix >= width) ix = width - 1;

            // Clamp y position to bottom edge
            if (iy >= height) iy = height - 1;

            // Read pixel from global memory
            float pixel = input[iy * width + ix];

            // Read filter coefficient from constant memory
            float coeff = d_filter[(fy + radius) * filterWidth + (fx + radius)];

            // Add weighted pixel value to sum
            sum += pixel * coeff;
        }
    }

    // Clamp result to minimum 0
    if (sum < 0) sum = 0;

    // Clamp result to maximum 255
    if (sum > 255) sum = 255;

    // Save final pixel value
    output[y * width + x] = (unsigned char)sum;
}

// Shared GPU convolution kernel
__global__ void convolutionKernelShared(unsigned char *input, unsigned char *output,
                                        int filterWidth,
                                        int width, int height)
{
    // This currently just calls the naive version
    convolutionKernelNaive(input, output, filterWidth, width, height);
}

// Function to compare two images
bool verify(unsigned char *a, unsigned char *b, int size)
{
    // Loop through each pixel
    for (int i = 0; i < size; i++)
    {
        // Allow a small difference of 1 because of rounding
        if (abs((int)a[i] - (int)b[i]) > 1)
            return false;
    }

    // Images match
    return true;
}

// Main program
int main()
{
    // Total number of pixels
    int imageSize = WIDTH * HEIGHT * CHANNELS;

    // Total number of bytes needed
    size_t bytes = imageSize * sizeof(unsigned char);

    // Create image variables
    Image input, cpuOutput, gpuOutput, sharedOutput;

    // Set input image width
    input.width = WIDTH;

    // Set input image height
    input.height = HEIGHT;

    // Set input image channels
    input.channels = CHANNELS;

    // Copy image settings to CPU output
    cpuOutput = input;

    // Copy image settings to naive GPU output
    gpuOutput = input;

    // Copy image settings to shared GPU output
    sharedOutput = input;

    // Allocate CPU memory for input image
    input.data = (unsigned char *)malloc(bytes);

    // Allocate CPU memory for CPU output image
    cpuOutput.data = (unsigned char *)malloc(bytes);

    // Allocate CPU memory for naive GPU output image
    gpuOutput.data = (unsigned char *)malloc(bytes);

    // Allocate CPU memory for shared GPU output image
    sharedOutput.data = (unsigned char *)malloc(bytes);

    // Generate test image
    generateImage(&input);

    // GPU pointer for input image
    unsigned char *d_input;

    // GPU pointer for naive output image
    unsigned char *d_gpuOutput;

    // GPU pointer for shared output image
    unsigned char *d_sharedOutput;

    // Allocate GPU memory for input image
    CHECK_CUDA_ERROR(cudaMalloc(&d_input, bytes));

    // Allocate GPU memory for naive output image
    CHECK_CUDA_ERROR(cudaMalloc(&d_gpuOutput, bytes));

    // Allocate GPU memory for shared output image
    CHECK_CUDA_ERROR(cudaMalloc(&d_sharedOutput, bytes));

    // Copy input image from CPU to GPU
    CHECK_CUDA_ERROR(cudaMemcpy(d_input, input.data, bytes, cudaMemcpyHostToDevice));

    // Copy filter values into GPU constant memory
    CHECK_CUDA_ERROR(cudaMemcpyToSymbol(d_filter, boxBlur3x3, 9 * sizeof(float)));

    // Start CPU timer
    clock_t cpuStart = clock();

    // Run CPU convolution
    convolutionCPU(&input, &cpuOutput, boxBlur3x3, 3);

    // Stop CPU timer
    clock_t cpuEnd = clock();

    // Convert CPU time to milliseconds
    double cpuTime = 1000.0 * (cpuEnd - cpuStart) / CLOCKS_PER_SEC;

    // Create a block of 16x16 threads
    dim3 threads(BLOCK_SIZE, BLOCK_SIZE);

    // Calculate number of blocks needed for full image
    dim3 blocks((WIDTH + BLOCK_SIZE - 1) / BLOCK_SIZE,
                (HEIGHT + BLOCK_SIZE - 1) / BLOCK_SIZE);

    // Declare CUDA timer events
    cudaEvent_t start, stop;

    // Create start event
    cudaEventCreate(&start);

    // Create stop event
    cudaEventCreate(&stop);

    // Start timing naive GPU kernel
    cudaEventRecord(start);

    // Run naive GPU convolution
    convolutionKernelNaive<<<blocks, threads>>>(d_input, d_gpuOutput, 3, WIDTH, HEIGHT);

    // Stop timing naive GPU kernel
    cudaEventRecord(stop);

    // Wait until naive GPU kernel finishes
    cudaEventSynchronize(stop);

    // Variable for naive GPU time
    float naiveTime;

    // Calculate naive GPU time in milliseconds
    cudaEventElapsedTime(&naiveTime, start, stop);

    // Start timing shared GPU kernel
    cudaEventRecord(start);

    // Run shared GPU convolution
    convolutionKernelShared<<<blocks, threads>>>(d_input, d_sharedOutput, 3, WIDTH, HEIGHT);

    // Stop timing shared GPU kernel
    cudaEventRecord(stop);

    // Wait until shared GPU kernel finishes
    cudaEventSynchronize(stop);

    // Variable for shared GPU time
    float sharedTime;

    // Calculate shared GPU time in milliseconds
    cudaEventElapsedTime(&sharedTime, start, stop);

    // Copy naive GPU result back to CPU
    CHECK_CUDA_ERROR(cudaMemcpy(gpuOutput.data, d_gpuOutput, bytes, cudaMemcpyDeviceToHost));

    // Copy shared GPU result back to CPU
    CHECK_CUDA_ERROR(cudaMemcpy(sharedOutput.data, d_sharedOutput, bytes, cudaMemcpyDeviceToHost));

    // Print image size
    printf("Image size: %d x %d\n", WIDTH, HEIGHT);

    // Print filter name
    printf("Filter: 3x3 Box Blur\n");

    // Print CPU time
    printf("CPU Time: %.3f ms\n", cpuTime);

    // Print naive GPU time
    printf("Naive GPU Time: %.3f ms\n", naiveTime);

    // Print shared GPU time
    printf("Shared GPU Time: %.3f ms\n", sharedTime);

    // Verify naive GPU output
    printf("Naive Verification: %s\n",
           verify(cpuOutput.data, gpuOutput.data, imageSize) ? "PASS" : "FAIL");

    // Verify shared GPU output
    printf("Shared Verification: %s\n",
           verify(cpuOutput.data, sharedOutput.data, imageSize) ? "PASS" : "FAIL");

    // Free GPU input memory
    cudaFree(d_input);

    // Free GPU naive output memory
    cudaFree(d_gpuOutput);

    // Free GPU shared output memory
    cudaFree(d_sharedOutput);

    // Free CPU input memory
    free(input.data);

    // Free CPU output memory
    free(cpuOutput.data);

    // Free CPU naive GPU result memory
    free(gpuOutput.data);

    // Free CPU shared GPU result memory
    free(sharedOutput.data);

    // End program successfully
    return 0;
}
