#ifndef SGEMM_H
#define SGEMM_H

void sgemm_v1(float* A, float* B, float* C, int M, int N, int K);
void sgemm_v1_mp(float* A, float* B, float* C, int M, int N, int K);

void sgemm_v2(float* A, float* B, float* C, int M, int N, int K);
void sgemm_v2_mp(float* A, float* B, float* C, int M, int N, int K);

void sgemm_v3(float *A, float *B, float *C, int M, int N, int K);
void sgemm_v3_mp(float *A, float *B, float *C, int M, int N, int K);

void sgemm(float* A, float* B, float* C, int M, int N, int K);

#endif