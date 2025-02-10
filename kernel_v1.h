#ifndef KERNEL_V1_H
#define KERNEL_V1_H


int sgemm(float* A, float* B, float* C, int M, int N, int K);

void sgemm_n(float* A, float* B, float* C, int M, int N, int K);

#endif