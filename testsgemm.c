#include <stdlib.h>
#include <stdio.h>
#include "kernel_v1.h"

void compare(float* A, float* B, int m, int n)
{
  int is_same = 1;
  for(int i = 0; i < m*n; i++){
    int x = i / n;
    int y = i % n;
    if(A[i] != B[i]){
      is_same = 0;
    }
  }
  if(is_same == 0){
    printf("res:fail!\n");
  }else{
    printf("res:pass!\n");
  }
}

int main(){
    const int M = 1024;
    const int N = 1024;
    const int K = 1024;
    
    float* A = (float*)malloc(sizeof(float) * M * K);
    float* B = (float*)malloc(sizeof(float) * K * N);
    float* C1 = (float*)malloc(sizeof(float) * M * N);
    float* C2 = (float*)malloc(sizeof(float) * M * N);

    for(int i = 0; i < M*K; i++){
        A[i] = 1;
    }

    for(int i = 0; i < K*N; i++){
        B[i] = i;
    }

    sgemm_n(A, B, C1, M, N, K);

    sgemm(A, B, C2, M, N, K);

    compare(C1, C2, M, N);

    free(A);
    free(B);
    free(C1);
    free(C2);

    return 0;
}