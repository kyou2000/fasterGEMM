#include <iostream>
#include "sgemm.h"
// test1

void compare(float* a, float* b, int m, int n){
  // compare
  int is_same = 1;
  for(int i = 0; i < m*n; i++){
    if(a[i] != b[i]){
      is_same = 0;
    }
  }
  
  if(is_same == 0){
    std::cout << "res:error!" << std::endl;
  }else{
    std::cout << "res:pass" << std::endl;
  }
}

int main()
{
  const int M = 1024;
  const int N = 1024;
  const int K = 1024;
  
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

// test
  sgemm_v1_mp(A, B, C_2, M, N, K);
  std::cout << "sgemm_v1_mp:" << std::endl;
  compare(C_1, C_2, M, N);

  sgemm_v2(A, B, C_2, M, N, K);
  std::cout << "sgemm_v2:" << std::endl;
  compare(C_1, C_2, M, N);

  sgemm_v2_mp(A, B, C_2, M, N, K);
  std::cout << "sgemm_v2_mp:" << std::endl;
  compare(C_1, C_2, M, N);

  sgemm_v3(A, B, C_2, M, N, K);
  std::cout << "sgemm_v3:" << std::endl;
  compare(C_1, C_2, M, N);

  sgemm_v3_mp(A, B, C_2, M, N, K);
  std::cout << "sgemm_v3_mp:" << std::endl;
  compare(C_1, C_2, M, N);
  
  delete[] A;
  delete[] B;
  delete[] C_1;
  delete[] C_2;

  return 0;

}