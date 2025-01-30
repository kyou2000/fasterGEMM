#include <immintrin.h>
#include <stdio.h>
#include <stdlib.h>
#include <windows.h>

#define A(i, j) A[(i)*k + (j)]
#define B(i, j) B[(i)*n + (j)]
#define C(i, j) C[(i)*n + (j)]

#define threadnum 8
// -mavx -mfma
typedef union{
    __m128 v;
    float d[4];
}rsv;

void packA(float *dsc, float * A, int x, int y, int k)
{
    float * ptr1 = &A(x, y);
    float * ptr2 = dsc;

    *ptr2++ = *ptr1++;
    *ptr2++ = *ptr1++;
    *ptr2++ = *ptr1++;
    *ptr2++ = *ptr1++;

    ptr1 = &A(x + 1, y);
    *ptr2++ = *ptr1++;
    *ptr2++ = *ptr1++;
    *ptr2++ = *ptr1++;
    *ptr2++ = *ptr1++;

    ptr1 = &A(x + 2, y);
    *ptr2++ = *ptr1++;
    *ptr2++ = *ptr1++;
    *ptr2++ = *ptr1++;
    *ptr2++ = *ptr1++;

    ptr1 = &A(x + 3, y);
    *ptr2++ = *ptr1++;
    *ptr2++ = *ptr1++;
    *ptr2++ = *ptr1++;
    *ptr2++ = *ptr1++;

}

void packB(float *dsc, float * B, int x, int y, int n)
{
    float * ptr1 = &B(x, y);
    float * ptr2 = dsc;

    *ptr2++ = *ptr1++;
    *ptr2++ = *ptr1++;
    *ptr2++ = *ptr1++;
    *ptr2++ = *ptr1++;

    ptr1 = &B(x + 1, y);
    *ptr2++ = *ptr1++;
    *ptr2++ = *ptr1++;
    *ptr2++ = *ptr1++;
    *ptr2++ = *ptr1++;

    ptr1 = &B(x + 2, y);
    *ptr2++ = *ptr1++;
    *ptr2++ = *ptr1++;
    *ptr2++ = *ptr1++;
    *ptr2++ = *ptr1++;

    ptr1 = &B(x + 3, y);
    *ptr2++ = *ptr1++;
    *ptr2++ = *ptr1++;
    *ptr2++ = *ptr1++;
    *ptr2++ = *ptr1++;
}

void matmul4x4ker(float * A, float * B, float * C, int x, int y, int m, int k, int n)
{
    float pa[16];
    float pb[16];

    rsv res0, res1, res2, res3;
    __m128 b0, b1, b2, b3;
    __m128 a0, a1, a2, a3, a4, a5, a6, a7;
    res0.v = _mm_setzero_ps();
    res1.v = _mm_setzero_ps();
    res2.v = _mm_setzero_ps();
    res3.v = _mm_setzero_ps();

    for (int s = 0; s < k; s += 4){
        packA(pa, A, x, s, k);
        packB(pb, B, s, y, n);

        b0 = _mm_loadu_ps(pb);
        b1 = _mm_loadu_ps(pb + 4);
        b2 = _mm_loadu_ps(pb + 8);
        b3 = _mm_loadu_ps(pb + 12);

        a0 = _mm_set1_ps(pa[0]);
        a1 = _mm_set1_ps(pa[1]);
        a2 = _mm_set1_ps(pa[2]);
        a3 = _mm_set1_ps(pa[3]);
        a4 = _mm_set1_ps(pa[4]);
        a5 = _mm_set1_ps(pa[5]);
        a6 = _mm_set1_ps(pa[6]);
        a7 = _mm_set1_ps(pa[7]);

        res0.v = _mm_fmadd_ps(a0, b0, res0.v);
        res0.v = _mm_fmadd_ps(a1, b1, res0.v);
        res0.v = _mm_fmadd_ps(a2, b2, res0.v);
        res0.v = _mm_fmadd_ps(a3, b3, res0.v);

        res1.v = _mm_fmadd_ps(a4, b0, res1.v);
        res1.v = _mm_fmadd_ps(a5, b1, res1.v);
        res1.v = _mm_fmadd_ps(a6, b2, res1.v);
        res1.v = _mm_fmadd_ps(a7, b3, res1.v);

        a0 = _mm_set1_ps(pa[8]);
        a1 = _mm_set1_ps(pa[9]);
        a2 = _mm_set1_ps(pa[10]);
        a3 = _mm_set1_ps(pa[11]);
        a4 = _mm_set1_ps(pa[12]);
        a5 = _mm_set1_ps(pa[13]);
        a6 = _mm_set1_ps(pa[14]);
        a7 = _mm_set1_ps(pa[15]);

        res2.v = _mm_fmadd_ps(a0, b0, res2.v);
        res2.v = _mm_fmadd_ps(a1, b1, res2.v);
        res2.v = _mm_fmadd_ps(a2, b2, res2.v);
        res2.v = _mm_fmadd_ps(a3, b3, res2.v);

        res3.v = _mm_fmadd_ps(a4, b0, res3.v);
        res3.v = _mm_fmadd_ps(a5, b1, res3.v);
        res3.v = _mm_fmadd_ps(a6, b2, res3.v);
        res3.v = _mm_fmadd_ps(a7, b3, res3.v);
    }
    C(x, y) = res0.d[0]; C(x, y+1) = res0.d[1]; C(x, y+2) = res0.d[2]; C(x, y+3) = res0.d[3];
    C(x+1, y) = res1.d[0]; C(x+1, y+1) = res1.d[1]; C(x+1, y+2) = res1.d[2]; C(x+1, y+3) = res1.d[3];
    C(x+2, y) = res2.d[0]; C(x+2, y+1) = res2.d[1]; C(x+2, y+2) = res2.d[2]; C(x+2, y+3) = res2.d[3];
    C(x+3, y) = res3.d[0]; C(x+3, y+1) = res3.d[1]; C(x+3, y+2) = res3.d[2]; C(x+3, y+3) = res3.d[3];
}

struct Para{
    float * a;
    float * b;
    float * c;
    int m;
    int k;
    int n;
    int blocksize;

}*Paraptr;

// 线程函数
DWORD WINAPI gemm_t(LPVOID lpParam){
    struct Para * p = (struct Para *)lpParam;
    for(int i = 0; i < p->m; i += 4){
        for(int j = 0; j < p->n; j += 4){
            matmul4x4ker(p->a, p->b, p->c, i, j, p->m, p->k, p->n);
        }
    }
}


void gemm(float * A, float * B, float * C, int m, int k, int n)
{

    for(int i = 0; i < m; i += 4){
        for(int j = 0; j < n; j += 4){
            matmul4x4ker(A, B, C, i, j, m, k, n);
        }
    }
}


void gemm_v2(float * A, float * B, float * C, int m, int k, int n)
{
    int mini_m = m / 4;
    int u_size = mini_m / threadnum;
    int y_size = mini_m % threadnum;

    if(u_size == 0){
        return;
    }
    
    struct Para p[threadnum];
    HANDLE hThreads[threadnum];
    for(int i = 0; i < threadnum; i++){
        p[i]. a = &A[u_size * i * 4 * k];
        p[i] .b = B;
        p[i] .c = &C[u_size * i * 4 * n];
        p[i] .k = k;
        p[i] .n = n;
        if (i == threadnum -1){
            p[i].m = (u_size + y_size) * 4;
        }else{
            p[i].m = u_size * 4;
        }
    }

    for (int i = 0; i < threadnum; i++){
        hThreads[i] = CreateThread(NULL, 0, gemm_t, (LPVOID)&p[i], 0, NULL);
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

int main(){
    int w = 2048;
    int size = w * w * sizeof(float);
    float * a = (float *)malloc(size);
    float * b = (float *)malloc(size);
    float * c = (float *)malloc(size);
    for (int i = 0; i < w*w; i++){
        a[i] = 1;
        b[i] = 1;
    }

    gemm_v2(a, b, c, w, w, w);

    for(int i = 0; i< 4; i++){
        printf("%f\n", c[i]);
    }

    free(a);
    free(b);
    free(c);

    return 0;
}