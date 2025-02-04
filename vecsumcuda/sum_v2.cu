
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

/* sum all entries in x and asign to y
 * block dim must be 256 */
__global__ void asum_stg_1(const float* x, float* y, int N)
{
    __shared__ float sdata[256];
    
    int tid = threadIdx.x;
    int bid = blockIdx.x + blockDim.x * blockIdx.y;
    int idx = threadIdx.x + bid * 256;

    /* load data to shared mem */
    if (idx < N) {
        sdata[tid] = x[idx];
    }
    else {
        sdata[tid] = 0;
    }

    __syncthreads();

    /* reduction using shared mem */
    if (tid < 128) sdata[tid] += sdata[tid + 128];
    __syncthreads();

    if (tid < 64) sdata[tid] += sdata[tid + 64];
    __syncthreads();

    if (tid < 32) sdata[tid] += sdata[tid + 32];
    __syncthreads();

    if (tid < 16) sdata[tid] += sdata[tid + 16];
    __syncthreads();

    if (tid < 8) sdata[tid] += sdata[tid + 8];
    __syncthreads();

    if (tid < 4) sdata[tid] += sdata[tid + 4];
    __syncthreads();

    if (tid < 2) sdata[tid] += sdata[tid + 2];
    __syncthreads();

    if (tid == 0) {
        y[bid] = sdata[0] + sdata[1];
    }
}

__global__ void asum_stg_3(float* x, int N)
{
    __shared__ float sdata[128];
    int tid = threadIdx.x;
    int i;

    sdata[tid] = 0;

    /* load data to shared mem */
    for (i = 0; i < N; i += 128) {
        if (tid + i < N) sdata[tid] += x[i + tid];
    }

    __syncthreads();

    /* reduction using shared mem */
    if (tid < 64) sdata[tid] = sdata[tid] + sdata[tid + 64];
    __syncthreads();

    if (tid < 32) sdata[tid] = sdata[tid] + sdata[tid + 32];
    __syncthreads();

    if (tid < 16) sdata[tid] += sdata[tid + 16];
    __syncthreads();

    if (tid < 8) sdata[tid] += sdata[tid + 8];
    __syncthreads();

    if (tid < 4) sdata[tid] += sdata[tid + 4];
    __syncthreads();

    if (tid < 2) sdata[tid] += sdata[tid + 2];
    __syncthreads();

    if (tid == 0) {
        x[0] = sdata[0] + sdata[1];
    }
}

/* dy and dz serve as cache: result stores in dz[0] */
void asum(float* dx, float* dy, float* dz, int N)
{
    /* 1D block */
    int bs = 256;

    int blocks = (N + bs - 1) / bs;

    /* 2D grid */
    int s = ceil(sqrt((N + bs - 1.) / bs));
    dim3 grid = dim3(s, s);
    int gs = 0;

    /* stage 1 */
    asum_stg_1 << <grid, bs >> > (dx, dy, N);

    /* stage 2 */
    {
        /* 1D grid */
        int N2 = (N + bs - 1) / bs;

        int s2 = ceil(sqrt((N2 + bs - 1.) / bs));
        dim3 grid2 = dim3(s2, s2);

        asum_stg_1 << <grid2, bs >> > (dy, dz, N2);

        /* record gs */
        gs = (N2 + bs - 1.) / bs;
    }

    /* stage 3 */
    asum_stg_3 << <1, 128 >> > (dz, gs);
}

/* host, add */
float asum_host(float* x, int N);

float asum_host(float* x, int N)
{
    int i;
    float t = 0;

    for (i = 0; i < N; i++) t += x[i];

    return t;
}

int main(int argc, char** argv)
{
    int N = 10000070;
    int nbytes = N * sizeof(float);

    float* dx = NULL, * hx = NULL;
    float* dy = NULL, * dz = NULL;
    int i, itr = 20;
    float asd = 0, ash;


    /* allocate GPU mem */
    cudaMalloc((void**)&dx, nbytes);
    cudaMalloc((void**)&dy, sizeof(float) * ((N + 255) / 256));
    cudaMalloc((void**)&dz, sizeof(float) * ((N + 255) / 256));

    if (dx == NULL || dy == NULL || dz == NULL) {
        printf("couldn't allocate GPU memory\n");
        return -1;
    }

    /* alllocate CPU mem */
    hx = (float*)malloc(nbytes);

    if (hx == NULL) {
        printf("couldn't allocate CPU memory\n");
        return -2;
    }

    /* init */
    for (i = 0; i < N; i++) {
        hx[i] = 1;
    }

    /* copy data to GPU */
    cudaMemcpy(dx, hx, nbytes, cudaMemcpyHostToDevice);


    /* call GPU */
    for (i = 0; i < itr; i++) asum(dx, dy, dz, N);

    /* copy data from GPU */
    cudaMemcpy(&asd, dz, sizeof(float), cudaMemcpyDeviceToHost);

    printf("asum, answer: %d, calculated by GPU:%f, calculated by CPU:%f\n", N, asd, ash);

    cudaFree(dx);
    cudaFree(dy);
    cudaFree(dz);
    free(hx);

    return 0;
}