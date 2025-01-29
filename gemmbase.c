#include <stdlib.h>
#include <stdio.h>
#include <windows.h>

#define threadnum 16

void gemm_v1(float* A, float* B, float* C, int m, int k, int n)
{
	for (int t = 0; t < m; t++){
		for (int i = 0; i < k; i++){
            float sum = 0;
			for (int j = 0; j < n; j++){
				sum += A[t * k + j] * B[j * n + i];
			}
            C[t * n + i] = sum;
		}
	}
}

struct Para{
    float * a;
    float * b;
    float * c;
    int m;
    int k;
    int n;
}*Paraptr;

DWORD WINAPI gemm_t2(LPVOID lpParam)
{
    struct Para * p = (struct Para *)lpParam;
    for (size_t t = 0; t < p -> m; t++){
		for (size_t i = 0; i < p -> n; i++){
            float sum = 0;
			for (size_t j = 0; j < p -> k; j++){
				sum += p -> a[t * p -> k + j] * p -> b[j * p -> n + i];
			}
            p -> c[t * p -> n + i] = sum;
		}
	}
}

void gemm_v2(float * a, float * b, float * c, int m, int k, int n){
    int u_m = m / threadnum;
    int y_m = m % threadnum;
    
    if(u_m == 0){
        return;
    }
    
    struct Para p[threadnum];
    HANDLE hThreads[threadnum];
    for(int i = 0; i < threadnum; i++){
        p[i]. a = &a[u_m * i * k];
        p[i] .b = b;
        p[i] .c = &c[u_m * i * n];
        p[i] .k = k;
        p[i] .n = n;
        if (i == threadnum -1){
            p[i].m = u_m + y_m;
        }else{
            p[i].m = u_m;
        }
    }

    for (int i = 0; i < threadnum; i++){
        hThreads[i] = CreateThread(NULL, 0, gemm_t2, (LPVOID)&p[i], 0, NULL);
        if(hThreads[i] == NULL){
            printf("error!\n");
            return;
        }
    }

    for(int i = 0; i < threadnum; i++){
        WaitForSingleObject(hThreads[i], INFINITE);
    }

    for(int i = 0; i < threadnum; i++){
        CloseHandle(hThreads[i]);
    }
}