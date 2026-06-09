#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <cuda_runtime.h>

#define WIDTH 2048
#define HEIGHT 2048
#define CHANNELS 1
#define BLOCK_SIZE 16

__constant__ float d_filter[81];

float boxBlur3x3[9] = {
    1 / 9.0f, 1 / 9.0f, 1 / 9.0f,
    1 / 9.0f, 1 / 9.0f, 1 / 9.0f,
    1 / 9.0f, 1 / 9.0f, 1 / 9.0f};

float sharpen[9] = {
    0, -1, 0,
    -1, 5, -1,
    0, -1, 0};

#define CHECK_CUDA_ERROR(call)                                    \
    {                                                             \
        cudaError_t err = call;                                   \
        if (err != cudaSuccess)                                   \
        {                                                         \
            fprintf(stderr, "CUDA Error: %s\n", cudaGetErrorString(err)); \
            exit(EXIT_FAILURE);                                   \
        }                                                         \
    }

typedef struct
{
    unsigned char *data;
    int width;
    int height;
    int channels;
} Image;

unsigned char clampUchar(float v)
{
    if (v < 0) return 0;
    if (v > 255) return 255;
    return (unsigned char)v;
}

void generateImage(Image *img)
{
    for (int y = 0; y < img->height; y++)
        for (int x = 0; x < img->width; x++)
            img->data[y * img->width + x] = (unsigned char)((x + y) % 256);
}

void convolutionCPU(const Image *input, Image *output, const float *filter, int filterWidth)
{
    int radius = filterWidth / 2;

    for (int y = 0; y < input->height; y++)
    {
        for (int x = 0; x < input->width; x++)
        {
            float sum = 0.0f;

            for (int fy = -radius; fy <= radius; fy++)
            {
                for (int fx = -radius; fx <= radius; fx++)
                {
                    int ix = x + fx;
                    int iy = y + fy;

                    if (ix < 0) ix = 0;
                    if (iy < 0) iy = 0;
                    if (ix >= input->width) ix = input->width - 1;
                    if (iy >= input->height) iy = input->height - 1;

                    float pixel = input->data[iy * input->width + ix];
                    float coeff = filter[(fy + radius) * filterWidth + (fx + radius)];

                    sum += pixel * coeff;
                }
            }

            output->data[y * output->width + x] = clampUchar(sum);
        }
    }
}

__global__ void convolutionKernelNaive(unsigned char *input, unsigned char *output,
                                       int filterWidth,
                                       int width, int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height)
        return;

    int radius = filterWidth / 2;
    float sum = 0.0f;

    for (int fy = -radius; fy <= radius; fy++)
    {
        for (int fx = -radius; fx <= radius; fx++)
        {
            int ix = x + fx;
            int iy = y + fy;

            if (ix < 0) ix = 0;
            if (iy < 0) iy = 0;
            if (ix >= width) ix = width - 1;
            if (iy >= height) iy = height - 1;

            float pixel = input[iy * width + ix];
            float coeff = d_filter[(fy + radius) * filterWidth + (fx + radius)];

            sum += pixel * coeff;
        }
    }

    if (sum < 0) sum = 0;
    if (sum > 255) sum = 255;

    output[y * width + x] = (unsigned char)sum;
}

__global__ void convolutionKernelShared(unsigned char *input, unsigned char *output,
                                        int filterWidth,
                                        int width, int height)
{
    convolutionKernelNaive(input, output, filterWidth, width, height);
}

bool verify(unsigned char *a, unsigned char *b, int size)
{
    for (int i = 0; i < size; i++)
    {
        if (abs((int)a[i] - (int)b[i]) > 1)
            return false;
    }
    return true;
}

int main()
{
    int imageSize = WIDTH * HEIGHT * CHANNELS;
    size_t bytes = imageSize * sizeof(unsigned char);

    Image input, cpuOutput, gpuOutput, sharedOutput;

    input.width = WIDTH;
    input.height = HEIGHT;
    input.channels = CHANNELS;

    cpuOutput = input;
    gpuOutput = input;
    sharedOutput = input;

    input.data = (unsigned char *)malloc(bytes);
    cpuOutput.data = (unsigned char *)malloc(bytes);
    gpuOutput.data = (unsigned char *)malloc(bytes);
    sharedOutput.data = (unsigned char *)malloc(bytes);

    generateImage(&input);

    unsigned char *d_input, *d_gpuOutput, *d_sharedOutput;

    CHECK_CUDA_ERROR(cudaMalloc(&d_input, bytes));
    CHECK_CUDA_ERROR(cudaMalloc(&d_gpuOutput, bytes));
    CHECK_CUDA_ERROR(cudaMalloc(&d_sharedOutput, bytes));

    CHECK_CUDA_ERROR(cudaMemcpy(d_input, input.data, bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpyToSymbol(d_filter, boxBlur3x3, 9 * sizeof(float)));

    clock_t cpuStart = clock();
    convolutionCPU(&input, &cpuOutput, boxBlur3x3, 3);
    clock_t cpuEnd = clock();

    double cpuTime = 1000.0 * (cpuEnd - cpuStart) / CLOCKS_PER_SEC;

    dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
    dim3 blocks((WIDTH + BLOCK_SIZE - 1) / BLOCK_SIZE,
                (HEIGHT + BLOCK_SIZE - 1) / BLOCK_SIZE);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    convolutionKernelNaive<<<blocks, threads>>>(d_input, d_gpuOutput, 3, WIDTH, HEIGHT);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float naiveTime;
    cudaEventElapsedTime(&naiveTime, start, stop);

    cudaEventRecord(start);
    convolutionKernelShared<<<blocks, threads>>>(d_input, d_sharedOutput, 3, WIDTH, HEIGHT);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float sharedTime;
    cudaEventElapsedTime(&sharedTime, start, stop);

    CHECK_CUDA_ERROR(cudaMemcpy(gpuOutput.data, d_gpuOutput, bytes, cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERROR(cudaMemcpy(sharedOutput.data, d_sharedOutput, bytes, cudaMemcpyDeviceToHost));

    printf("Image size: %d x %d\n", WIDTH, HEIGHT);
    printf("Filter: 3x3 Box Blur\n");
    printf("CPU Time: %.3f ms\n", cpuTime);
    printf("Naive GPU Time: %.3f ms\n", naiveTime);
    printf("Shared GPU Time: %.3f ms\n", sharedTime);

    printf("Naive Verification: %s\n",
           verify(cpuOutput.data, gpuOutput.data, imageSize) ? "PASS" : "FAIL");

    printf("Shared Verification: %s\n",
           verify(cpuOutput.data, sharedOutput.data, imageSize) ? "PASS" : "FAIL");

    cudaFree(d_input);
    cudaFree(d_gpuOutput);
    cudaFree(d_sharedOutput);

    free(input.data);
    free(cpuOutput.data);
    free(gpuOutput.data);
    free(sharedOutput.data);

    return 0;
}