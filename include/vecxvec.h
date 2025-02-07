#ifndef VECXVEC_H
#define VECXVEC_H

void vecmul(float *A, float *B, float *C, int n);

void vecmul_mp(float *A, float *B, float *C, int n);

void vecadd(float *A, float *B, float *C, int n);

void vecadd_mp(float *A, float *B, float *C, int n);

void vecsub(float *A, float *B, float *C, int n);

void vecsub_mp(float *A, float *B, float *C, int n);

float vecdot(float *A, float *B, int n);

float vecdot_mp(float *A, float *B, int n);

#endif