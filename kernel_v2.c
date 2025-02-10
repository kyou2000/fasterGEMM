#include <immintrin.h>
#include <stdlib.h>
#include <pthread.h>
#define A(i,j) A[(i)*K + (j)]
#define B(i,j) B[(i)*N + (j)]
#define C(i,j) C[(i)*N + (j)]
void packA(float *dsc, float * A, int x, int y, int K)
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

void packB(float *dsc, float * B, int x, int y, int N)
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

void matmul4x4ker(float * A, float * B, float * C, int x, int y, int M, int K, int N)
{
    float pa[16] __attribute__((aligned(32)));
    float pb[16] __attribute__((aligned(32)));

    __m128 res0, res1, res2, res3;
    __m128 b0, b1, b2, b3, a0, a1, a2, a3, a4, a5, a6, a7;
    res0 = _mm_setzero_ps();
    res1 = _mm_setzero_ps();
    res2 = _mm_setzero_ps();
    res3 = _mm_setzero_ps();

    for (int s = 0; s < K; s += 4){
        packA(pa, A, x, s, K);
        packB(pb, B, s, y, N);

        b0 = _mm_load_ps(pb);
        b1 = _mm_load_ps(pb + 4);
        b2 = _mm_load_ps(pb + 8);
        b3 = _mm_load_ps(pb + 12);

        a0 = _mm_set1_ps(pa[0]);
        a1 = _mm_set1_ps(pa[1]);
        a2 = _mm_set1_ps(pa[2]);
        a3 = _mm_set1_ps(pa[3]);
        a4 = _mm_set1_ps(pa[4]);
        a5 = _mm_set1_ps(pa[5]);
        a6 = _mm_set1_ps(pa[6]);
        a7 = _mm_set1_ps(pa[7]);

        res0 = _mm_fmadd_ps(a0, b0, res0);
        res0 = _mm_fmadd_ps(a1, b1, res0);
        res0 = _mm_fmadd_ps(a2, b2, res0);
        res0 = _mm_fmadd_ps(a3, b3, res0);

        res1 = _mm_fmadd_ps(a4, b0, res1);
        res1 = _mm_fmadd_ps(a5, b1, res1);
        res1 = _mm_fmadd_ps(a6, b2, res1);
        res1 = _mm_fmadd_ps(a7, b3, res1);

        a0 = _mm_set1_ps(pa[8]);
        a1 = _mm_set1_ps(pa[9]);
        a2 = _mm_set1_ps(pa[10]);
        a3 = _mm_set1_ps(pa[11]);
        a4 = _mm_set1_ps(pa[12]);
        a5 = _mm_set1_ps(pa[13]);
        a6 = _mm_set1_ps(pa[14]);
        a7 = _mm_set1_ps(pa[15]);

        res2 = _mm_fmadd_ps(a0, b0, res2);
        res2 = _mm_fmadd_ps(a1, b1, res2);
        res2 = _mm_fmadd_ps(a2, b2, res2);
        res2 = _mm_fmadd_ps(a3, b3, res2);

        res3 = _mm_fmadd_ps(a4, b0, res3);
        res3 = _mm_fmadd_ps(a5, b1, res3);
        res3 = _mm_fmadd_ps(a6, b2, res3);
        res3 = _mm_fmadd_ps(a7, b3, res3);
    }
    
    _m128_storeu_ps(&C(x, y), res0);
    _m128_storeu_ps(&C(x+1, y), res1);
    _m128_storeu_ps(&C(x+2, y), res2);
    _m128_storeu_ps(&C(x+3, y), res3);
}

void sgemm_v2(float* A, float* B, float* C, int M, int N, int K)
{
    for(int m = 0; m < M; m += 4){
        for(int n = 0; n < N; n += 4){
            matmul4x4ker(A, B, C, m, n, M, K, N);
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

void matmul_t(void* lpParam) 

{
    struct Para * p = (struct Para *)lpParam;
    for (int i = 0; i < p->m; i += 4) {
        for (int j = 0; j < p->n; j += 4) {
            kernel(p->a, p->b, p->c, p->m, p->n, p->k, i, j);
        }
    }
}


int sgemm_v2_mp(float* A, float* B, float* C, int M, int N, int K)
{
    int group_m = M/4;
    const int NUM_THREADS = 6;
    int u_size = group_m / NUM_THREADS;
    int y_size = group_m % NUM_THREADS;

    if (M % 4 != 0){
        return -1;
    }

    if (u_size == 0){
        sgemm_v2(A, B, C, M, N, K);
        return -1;
    }
    pthread_t threads[NUM_THREADS]; 
    struct Para p[NUM_THREADS];
    
    for(int i = 0; i < NUM_THREADS; i++){
        p[i]. a = &A[u_size * i * 4 * K];
        p[i] .b = B;
        p[i] .c = &C[u_size * i * 4 * N];
        p[i] .k = K;
        p[i] .n = N;
        if (i == NUM_THREADS -1){
            p[i].m = (u_size + y_size) * 4;
        }else{
            p[i].m = u_size * 4;
        }
    }

    for (int i = 0; i < NUM_THREADS; i++) {
        if (pthread_create(&threads[i], NULL, matmul_t, &p[i]) != 0) {
            perror("Failed to create thread");
            return -1;
        }
    }

    for (int i = 0; i < NUM_THREADS; i++) {
        if (pthread_join(threads[i], NULL) != 0) {
            perror("Failed to join thread");
            return -1;
        }
    }
    return 0;
}