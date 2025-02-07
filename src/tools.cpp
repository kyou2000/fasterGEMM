// 数据硬拷贝
#include <immintrin.h>
#include <iostream>
#include "tools.h"

// float
#define src(i, j) src[(i)*src_n + (j)]
#define dsc(i, j) dsc[(i)*dsc_n + (j)]

void padding(float* dsc, float* src, int dsc_m, int dsc_n, int src_m, int src_n){
    for (int i = 0; i < dsc_m; i++){
        for (int j = 0; j < dsc_n; j++){
            if(i < src_m && j < src_n){
                dsc(i, j) = src(i, j);
            }else{
                dsc(i, j) = 0.f;
            }
        }
    }
    
}

void datacpy_f(float* dsc, float* src, int dsc_m, int dsc_n, int src_m, int src_n)
{
    __m256 vec;
    for(int i = 0; i < dsc_m; i++){
        int j = 0;
        if(i < src_m){

            while(j+8 < src_n){
                vec = _mm256_loadu_ps(&src(i, j));
                _mm256_storeu_ps(&dsc(i, j), vec);
                j+=8;
            }
            
            for(j; j < src_n; j++){
                dsc(i, j) = src(i, j);
            }
            
            for(j; j < dsc_n; j++){
                dsc(i, j) = 0.f;
            }
        }else{
            for(j; j < dsc_n; j++){
                dsc(i, j) = 0.f;
            }
        }
    }
}

// 将填充后的数据结果写回
void reload_f(float* dsc, float* src, int dsc_m, int dsc_n, int src_m, int src_n)
{
    __m256 vec;
    for(int i = 0; i < dsc_m; i++){
        int j = 0;
        while(j+8 < dsc_n){
            vec = _mm256_loadu_ps(&src(i, j));
            _mm256_storeu_ps(&dsc(i, j), vec);
            j+=8;
        }
            
        for(j; j < dsc_n; j++){
            dsc(i, j) = src(i, j);
        }
    }
}

// 打印矩阵
void print_mat(float* arr, int m, int n)
{
    for(int i = 0; i < m; i++){
        for(int j = 0; j < n; j++){
            std::cout << arr[i*n + j] << " ";
        }
        std::cout << std::endl;
    }
}

// 矩阵比较
void compare_mat(float* a, float* b, int m, int n)
{
  int is_same = 1;
  for(int i = 0; i < m*n; i++){
    int x = i / n;
    int y = i % n;
    if(a[i] != b[i]){
      is_same = 0;
      std::cout << "(" << x << "," << y << ")  " << a[i] << "  " << b[i] << std::endl;
    }
  }
  if(is_same == 0){
    std::cout << "res:error!" << std::endl;
  }else{
    std::cout << "res:pass!" << std::endl;
  }
}

// 向量比较
void compare_vec(float* a, float* b, int size)
{
  int is_same = 1;
  for(int i = 0; i < size; i++){
    if(a[i] != b[i]){
      is_same = 0;
      std::cout << "index:" << i << "  " << a[i] << "  " << b[i] << std::endl;
    }
  }
  if(is_same == 0){
    std::cout << "res:error!" << std::endl;
  }else{
    std::cout << "res:pass!" << std::endl;
  }
}