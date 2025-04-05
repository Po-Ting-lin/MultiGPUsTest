#include "fftFilter.h"

__global__ void warmUpKernel(float* d_src1, float* d_src2, float* d_dst, int nx, int ny) {
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int y = threadIdx.y + blockDim.y * blockIdx.y;
    if (x >= nx || y >= ny) return;
    d_dst[y * nx + x] = d_src1[y * nx + x] + d_src2[y * nx + x];
}

void warmUpGPU() {
    const int x = 1024;
    const int y = 1024;
    float* h_src1 = new float[x * y];
    float* h_src2 = new float[x * y];
    float* h_dst = new float[x * y];
    float* d_src1;
    float* d_src2;
    float* d_dst;
    Check(cudaMalloc(&d_src1, x * y * sizeof(float)));
    Check(cudaMalloc(&d_src2, x * y * sizeof(float)));
    Check(cudaMalloc(&d_dst, x * y * sizeof(float)));

    dim3 block(32, 32);
    dim3 grid(iDivUp(x, 32), iDivUp(y, 32));

    Check(cudaMemcpy(d_src1, h_src1, x * y * sizeof(float), cudaMemcpyHostToDevice));
    Check(cudaMemcpy(d_src2, h_src2, x * y * sizeof(float), cudaMemcpyHostToDevice));

    warmUpKernel << <grid, block >> > (d_src1, d_src2, d_dst, x, y);

    Check(cudaMemcpy(h_dst, d_dst, x * y * sizeof(float), cudaMemcpyDeviceToHost));

    Check(cudaFree(d_src1));
    Check(cudaFree(d_src2));
    Check(cudaFree(d_dst));
    delete[] h_src1;
    delete[] h_src2;
    delete[] h_dst;
}
