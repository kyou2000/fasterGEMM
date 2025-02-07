#include <thread>
#include <vector>
#include "vecxmat.h"

// 向量与矩阵的计算
void matmul(float * A, float * B, float * C, int m, int n)
{
    for (int i = 0;i < n; i++){
        float sum = 0;
        for (int j = 0;j < m; j++){
            sum += A[j] * B[j*n + i];
        }
        C[i] = sum;
    }
}


void matmul_v1_t(float * A, float * B, float * C, int m, int n, int blocksize, int i){
  
    for (int s = 0;s < blocksize; s++){
        float sum = 0;
        for (int j = 0;j < m; j++){
            sum += A[j] * B[j*n + i*blocksize + s];
        }
        C[i*blocksize + s] = sum;
    }
}


void matmul_mp(float * A, float * B, float * C, int m, int n)
{   
    int threadnum = std::thread::hardware_concurrency();
    int u_size = n / threadnum;
    int y_size = n % threadnum;

    if(u_size == 0){
        return;
    }

    std::vector<std::thread> threads(threadnum);

    for (int i = 0; i < threadnum; i++){
        if(i == threadnum -1){
            threads[i] = std::thread(matmul_v1_t, A, B, C, m, n, u_size+y_size, i);
        }else{

            threads[i] = std::thread(matmul_v1_t, A, B, C, m, n, u_size, i);
        }
        
    }

    for(int i = 0; i < threadnum; i++){
        threads[i].join();
    }

}

// 转置
void matmul_tanspose(float * A, float * B, float * C, int m, int n)
{
    for(int i = 0;i < m; i++){
        float sum = 0;
        for(int j = 0;j < n; j++){
            sum += A[j] * B[i*n + j];
        }
        C[i] = sum;
    }
}

void st_t(float * A, float * B, float * C, int m, int n){
    for(int i = 0;i < m; i++){
        float sum = 0;
        for(int j = 0;j < n; j++){
            sum += A[j] * B[i*n + j];
        }
        C[i] = sum;
    }
}

void matmul_tanspose_mp(float * A, float * B, float * C, int m, int n)
{
    int threadnum = std::thread::hardware_concurrency();
    int u_size = m / threadnum;
    int y_size = m % threadnum;

    if(u_size == 0){
        return;
    }
    std::vector<std::thread> threads(threadnum);

    for (int i = 0; i < threadnum; i++){
        if(i == threadnum -1){
            threads[i] = std::thread(st_t, A, &B[n*u_size*i], &C[i*u_size], u_size + y_size, n);
        }else{
            threads[i] = std::thread(st_t, A, &B[n*u_size*i], &C[i*u_size], u_size, n);
        }
        
    }

    for(int i = 0; i < threadnum; i++){
        threads[i].join();
    }

}