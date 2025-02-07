/*
矩阵转置

A为输入矩阵，B为输出矩阵
m为A的行数，n为A的列数
*/ 

#include "transpose.h"

#define A(i, j) A[(i) * n + (j)]
#define B(i, j) B[(i) * m + (j)]

void transpose(float *A, float *B, int m, int n) 
{
    for(int i = 0; i < n; i++) {
        for(int j = 0; j < m; j++) {
            B(i, j) = A(j, i);
        }
    }
}