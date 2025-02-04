
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>

__global__ void VecSumKnl(const float* x, float* y)
{
    __shared__ float sdata[256];
    int tid = threadIdx.x;

    /* load data to shared mem */
    sdata[tid] = x[tid];
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
        *y = sdata[0] + sdata[1];
    }
}

int main()
{
    int N = 256;   /* must be 256 */
    int nbytes = N * sizeof(float);

    float* dx = NULL, * hx = NULL;
    float* dy = NULL;
    int i;
    float as = 0;

    /* allocate GPU mem */
    cudaMalloc((void**)&dx, nbytes);
    cudaMalloc((void**)&dy, sizeof(float));

    if (dx == NULL || dy == NULL) {
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
    VecSumKnl << <1, N >> > (dx, dy);

    /* let GPU finish */
    cudaDeviceSynchronize();

    /* copy data from GPU */
    cudaMemcpy(&as, dy, sizeof(float), cudaMemcpyDeviceToHost);

    printf("VecSumKnl, answer: 256, calculated by GPU: %f\n", as);

    cudaFree(dx);
    cudaFree(dy);
    free(hx);

    return 0;
}


