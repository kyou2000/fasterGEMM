#ifndef VECXMAT_H
#define VECXMAT_H

void matmul(float * A, float * B, float * C, int m, int n);

void matmul_mp(float * A, float * B, float * C, int m, int n);

void matmul_tanspose(float * A, float * B, float * C, int m, int n);

void matmul_tanspose_mp(float * A, float * B, float * C, int m, int n);

#endif