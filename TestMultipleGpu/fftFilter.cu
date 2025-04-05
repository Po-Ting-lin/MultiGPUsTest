#include "fftFilter.h"



__global__ void mulKernel(float2* dst, float2* data, float2* kernel, int dataSize, float c) {
    const int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= dataSize) return;
    float2 a = kernel[i];
    float2 b = data[i];
    dst[i].x = c * (a.x * b.x - a.y * b.y);
    dst[i].y = c * (a.y * b.x + a.x * b.y);
}

void runComplexMul(cufftComplex* dst, cufftComplex* src, cufftComplex* kernel, int fftH, int fftW, float c, cudaStream_t& stream) {
    const int dataSize = fftH * fftW;
    if (dataSize % 2 != 0) return;
    dim3 block(512);
    dim3 grid(iDivUp(dataSize, 512));
    mulKernel << < grid, block, 0, stream>>>((float2*)dst, (float2*)src, (float2*)kernel, dataSize, c);
    getLastCudaError("mulKernel execution failed\n");
}