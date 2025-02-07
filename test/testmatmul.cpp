#include <iostream>
#include "sgemm.h"
#include "tools.h"

int main(){
    const int M = 123;
    const int N = 456;
    const int K = 1201;
  
    float* A = new float[M*K];
    float* B = new float[K*N];
    float* C_1 = new float[M*N];
    float* C_2 = new float[M*N];

    // init 
    for(int i = 0; i < M*K; i++){
        A[i] = 1;
    }

    for(int i = 0; i < K*N; i++){
        B[i] = i;
    }
    // 单线程标准计算
    sgemm_v1(A, B, C_1, M, N, K);

    sgemm(A, B, C_2, M, N, K);

    compare_mat(C_1, C_2, M, N);

    return 0;

}

