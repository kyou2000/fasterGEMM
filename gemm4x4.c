#include <stdlib.h>
#include <stdio.h>
#include <windows.h>

#define kerSize 4
#define threadnum 16
// kernel 4*4

void package(float * a, float * b, int x, int y, int k)
{
    int index = x*k + y;
    b[0] = a[index];
    b[1] = a[index+1];
    b[2] = a[index+2];
    b[3] = a[index+3];
    b[4] = a[index+k];
    b[5] = a[index+k+1];
    b[6] = a[index+k+2];
    b[7] = a[index+k+3];
    b[8] = a[index+2*k];
    b[9] = a[index+2*k+1];
    b[10] = a[index+2*k+2];
    b[11] = a[index+2*k+3];
    b[12] = a[index+3*k];
    b[13] = a[index+3*k+1];
    b[14] = a[index+3*k+2];
    b[15] = a[index+3*k+3];
}

void turn(float * a, float * b, int x, int y, int k)
{
    int index = x*k + y;
    b[index] = a[0];
    b[index+1] = a[1];
    b[index+2] = a[2];
    b[index+3] = a[3];
    b[index+k] = a[4];
    b[index+k+1] = a[5];
    b[index+k+2] = a[6];
    b[index+k+3] = a[7];
    b[index+2*k] = a[8];
    b[index+2*k+1] = a[9];
    b[index+2*k+2] = a[10];
    b[index+2*k+3] = a[11];
    b[index+3*k] = a[12];
    b[index+3*k+1] = a[13];
    b[index+3*k+2] = a[14];
    b[index+3*k+3] = a[15];
}

struct Para{
    float * a;
    float * b;
    float * c;
    int m;
    int k;
    int n;
}*Paraptr;

DWORD WINAPI gemm_t1(LPVOID lpParam)
{   
    struct Para * p = (struct Para *)lpParam;
    float swblock1[16];
    float swblock2[16];
    float packA[16];
    float packB[16];
    int s_m = p -> m / kerSize;
    int s_k = p -> k / kerSize;
    int s_n = p -> n / kerSize;

    for(int i = 0; i < s_m; i++){
        for(int j = 0; j < s_n; j++){
            int row = i*kerSize;
            int col = j*kerSize;

            // swblock2 置零
            swblock2[0] = 0;
            swblock2[1] = 0;
            swblock2[2] = 0;
            swblock2[3] = 0;
            swblock2[4] = 0;
            swblock2[5] = 0;
            swblock2[6] = 0;
            swblock2[7] = 0;
            swblock2[8] = 0;
            swblock2[9] = 0;
            swblock2[10] = 0;
            swblock2[11] = 0;
            swblock2[12] = 0;
            swblock2[13] = 0;
            swblock2[14] = 0;
            swblock2[15] = 0;

            for(int s = 0; s < s_k; s++){
                int s_col = s*kerSize;
                package(p -> a, packA, row, s_col, p -> k);
                package(p -> b, packB, s_col, col, p -> n);
                
                swblock1[0] = packA[0]*packB[0] + packA[1]*packB[4] + packA[2]*packB[8] + packA[3]*packB[12];
                swblock1[1] = packA[0]*packB[1] + packA[1]*packB[5] + packA[2]*packB[9] + packA[3]*packB[13];
                swblock1[2] = packA[0]*packB[2] + packA[1]*packB[6] + packA[2]*packB[10] + packA[3]*packB[14];
                swblock1[3] = packA[0]*packB[3] + packA[1]*packB[7] + packA[2]*packB[11] + packA[3]*packB[15];
                swblock1[4] = packA[4]*packB[0] + packA[5]*packB[4] + packA[6]*packB[8] + packA[7]*packB[12];
                swblock1[5] = packA[4]*packB[1] + packA[5]*packB[5] + packA[6]*packB[9] + packA[7]*packB[13];
                swblock1[6] = packA[4]*packB[2] + packA[5]*packB[6] + packA[6]*packB[10] + packA[7]*packB[14];
                swblock1[7] = packA[4]*packB[3] + packA[5]*packB[7] + packA[6]*packB[11] + packA[7]*packB[15];
                swblock1[8] = packA[8]*packB[0] + packA[9]*packB[4] + packA[10]*packB[8] + packA[11]*packB[12];
                swblock1[9] = packA[8]*packB[1] + packA[9]*packB[5] + packA[10]*packB[9] + packA[11]*packB[13];
                swblock1[10] = packA[8]*packB[2] + packA[9]*packB[6] + packA[10]*packB[10] + packA[11]*packB[14];
                swblock1[11] = packA[8]*packB[3] + packA[9]*packB[7] + packA[10]*packB[11] + packA[11]*packB[15];
                swblock1[12] = packA[12]*packB[0] + packA[13]*packB[4] + packA[14]*packB[8] + packA[15]*packB[12];
                swblock1[13] = packA[12]*packB[1] + packA[13]*packB[5] + packA[14]*packB[9] + packA[15]*packB[13];
                swblock1[14] = packA[12]*packB[2] + packA[13]*packB[6] + packA[14]*packB[10] + packA[15]*packB[14];
                swblock1[15] = packA[12]*packB[3] + packA[13]*packB[7] + packA[14]*packB[11] + packA[15]*packB[15];
                
                
                swblock2[0] += swblock1[0];
                swblock2[1] += swblock1[1];
                swblock2[2] += swblock1[2];
                swblock2[3] += swblock1[3];
                swblock2[4] += swblock1[4];
                swblock2[5] += swblock1[5];
                swblock2[6] += swblock1[6];
                swblock2[7] += swblock1[7];
                swblock2[8] += swblock1[8];
                swblock2[9] += swblock1[9];
                swblock2[10] += swblock1[10];
                swblock2[11] += swblock1[11];
                swblock2[12] += swblock1[12];
                swblock2[13] += swblock1[13];
                swblock2[14] += swblock1[14];
                swblock2[15] += swblock1[15];
            }
            turn(swblock2, p -> c, row, col, p -> n);
        }
    }
}

// 比较快也稳定
void gemm(float * A, float * B, float * C, int m, int k, int n)
{
    int mini_m = m / kerSize;
    int u_size = mini_m / threadnum;
    int y_size = mini_m % threadnum;
    
    if(u_size == 0){
        return;
    }
    
    struct Para p[threadnum];
    HANDLE hThreads[threadnum];
    for(int i = 0; i < threadnum; i++){
        p[i]. a = &A[u_size * i * kerSize * k];
        p[i] .b = B;
        p[i] .c = &C[u_size * i * kerSize * n];
        p[i] .k = k;
        p[i] .n = n;
        if (i == threadnum -1){
            p[i].m = (u_size + y_size) * kerSize;
        }else{
            p[i].m = u_size * kerSize;
        }
    }

    for (int i = 0; i < threadnum; i++){
        hThreads[i] = CreateThread(NULL, 0, gemm_t1, (LPVOID)&p[i], 0, NULL);
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

int main()
{
    int w = 2048;
    int size = w * w * sizeof(float);
    float *a = (float *)malloc(size);
    float *b = (float *)malloc(size);
    float *c = (float *)malloc(size);

    for(int i = 0; i< w*w; i++){
        a[i] = 1;
        b[i] = 1;
    }

    gemm(a, b, c, w, w, w);

    for(int i = 0; i < 4; i++){
        printf("%f\n", c[i]);
    }

    free(a);
    free(b);
    free(c);

    printf("ok\n");
    
    return 0;
}