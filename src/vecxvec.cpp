#include <thread>
#include <vector>
#include "vecxvec.h"

//向量与向量的计算

// 向量乘法
void vecmul(float *A, float *B, float *C, int n) 
{
    for(int i = 0; i < n; i++) {
        C[i] = A[i] * B[i];
    }
}


void vecmul_mp(float *A, float *B, float *C, int n)
{
    int threadnum = std::thread::hardware_concurrency();
    int u_size = n / threadnum;
    int y_size = n % threadnum;

    if(u_size == 0){
        vecmul(A, B, C, n);
        return;
    }

    std::vector<std::thread> threads(threadnum);

    for (int i = 0; i < threadnum; i++){
        if(i == threadnum -1){
            threads[i] = std::thread(vecmul, &A[i*u_size], &B[i*u_size], &C[i*u_size], u_size+y_size);
        }else{

            threads[i] = std::thread(vecmul, &A[i*u_size], &B[i*u_size], &C[i*u_size], u_size);
        }
        
    }

    for(int i = 0; i < threadnum; i++){
        threads[i].join();
    }

}

// 向量加法
void vecadd(float *A, float *B, float *C, int n)
{
    for(int i = 0; i < n; i++){
        C[i] = A[i] + B[i];
    }
}

void vecadd_mp(float *A, float *B, float *C, int n)
{
    int threadnum = std::thread::hardware_concurrency();
    int u_size = n / threadnum;
    int y_size = n % threadnum;

    if(u_size == 0){
        vecadd(A, B, C, n);
        return;
    }

    std::vector<std::thread> threads(threadnum);

    for (int i = 0; i < threadnum; i++){
        if(i == threadnum -1){
            threads[i] = std::thread(vecadd, &A[i*u_size], &B[i*u_size], &C[i*u_size], u_size+y_size);
        }else{

            threads[i] = std::thread(vecadd, &A[i*u_size], &B[i*u_size], &C[i*u_size], u_size);
        }
        
    }

    for(int i = 0; i < threadnum; i++){
        threads[i].join();
    }
}

// 向量减法
void vecsub(float *A, float *B, float *C, int n)
{
    for(int i = 0; i < n; i++){
        C[i] = A[i] - B[i];
    }
}

void vecsub_mp(float *A, float *B, float *C, int n)
{
    int threadnum = std::thread::hardware_concurrency();
    int u_size = n / threadnum;
    int y_size = n % threadnum;

    if(u_size == 0){
        vecadd(A, B, C, n);
        return;
    }

    std::vector<std::thread> threads(threadnum);

    for (int i = 0; i < threadnum; i++){
        if(i == threadnum -1){
            threads[i] = std::thread(vecsub, &A[i*u_size], &B[i*u_size], &C[i*u_size], u_size+y_size);
        }else{

            threads[i] = std::thread(vecsub, &A[i*u_size], &B[i*u_size], &C[i*u_size], u_size);
        }
        
    }

    for(int i = 0; i < threadnum; i++){
        threads[i].join();
    }
}

// 向量内积
float vecdot(float *A, float *B, int n)
{
    float sum = 0;
    for(int i = 0; i < n; i++){
        sum += A[i] * B[i];
    }

    return sum;
}

void vecdot_t(float *A, float *B, int n, float *temp, int index)
{
    float sum = 0;
    for(int i = 0; i < n; i++){
        sum += A[i] * B[i];
    }
    temp[index] = sum;
}

float vecdot_mp(float *A, float *B, int n)
{
    const int threadnum = std::thread::hardware_concurrency();
    int u_size = n / threadnum;
    int y_size = n % threadnum;

    if(u_size == 0){
        return vecdot(A, B, n);
        
    }

    float temp[threadnum] = {0.f};

    std::vector<std::thread> threads(threadnum);

    for(int i = 0; i < threadnum; i++){
        if(i == threadnum -1){
            threads[i] = std::thread(vecdot_t, &A[i*u_size], &B[i*u_size], u_size+y_size, &temp[0], i);
        }else{
            threads[i] = std::thread(vecdot_t, &A[i*u_size], &B[i*u_size], u_size, &temp[0], i);
        }
    }

    for(int i = 0; i < threadnum; i++){
        threads[i].join();
    }

    float sum = 0.f;
    for(int i = 0; i < threadnum; i++){
        sum += temp[i];
    }

    return sum;
}