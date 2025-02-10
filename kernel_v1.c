#include <immintrin.h>
#include <stdlib.h>
#include <pthread.h>
#include "kernel_v1.h"

#define A(i,j) A[(i)*K + (j)]
#define B(i,j) B[(i)*N + (j)]
#define C(i,j) C[(i)*N + (j)]
// -mvax -mfma
// kernel 8x8

void sgemm_n(float* A, float* B, float* C, int M, int N, int K)
{

    for (int m = 0; m < M; m++){
		for (int n = 0; n < N; n++){
            float sum = 0;
			for (int k = 0; k < K; k++){
				sum += A(m, k) * B(k, n);
			}
            C(m, n) = sum;
		}
	}
}

void packgeB(float *B, float *packgeB, int x, int y, int N) {
    float *ptr1 = &B(x,y);
    float *ptr2 = packgeB;

    *ptr2++ = *ptr1++;
    *ptr2++ = *ptr1++;
    *ptr2++ = *ptr1++;
    *ptr2++ = *ptr1++;
    *ptr2++ = *ptr1++;
    *ptr2++ = *ptr1++;
    *ptr2++ = *ptr1++;
    *ptr2++ = *ptr1;
    ptr1 = &B(x+1,y);

    *ptr2++ = *ptr1++;
    *ptr2++ = *ptr1++;
    *ptr2++ = *ptr1++;
    *ptr2++ = *ptr1++;
    *ptr2++ = *ptr1++;
    *ptr2++ = *ptr1++;
    *ptr2++ = *ptr1++;
    *ptr2++ = *ptr1;
    ptr1 = &B(x+2,y);

    *ptr2++ = *ptr1++;
    *ptr2++ = *ptr1++;
    *ptr2++ = *ptr1++;
    *ptr2++ = *ptr1++;
    *ptr2++ = *ptr1++;
    *ptr2++ = *ptr1++;
    *ptr2++ = *ptr1++;
    *ptr2++ = *ptr1;
    ptr1 = &B(x+3,y);

    *ptr2++ = *ptr1++;
    *ptr2++ = *ptr1++;
    *ptr2++ = *ptr1++;
    *ptr2++ = *ptr1++;
    *ptr2++ = *ptr1++;
    *ptr2++ = *ptr1++;
    *ptr2++ = *ptr1++;
    *ptr2++ = *ptr1;
    ptr1 = &B(x+4,y);

    *ptr2++ = *ptr1++;
    *ptr2++ = *ptr1++;
    *ptr2++ = *ptr1++;
    *ptr2++ = *ptr1++;
    *ptr2++ = *ptr1++;
    *ptr2++ = *ptr1++;
    *ptr2++ = *ptr1++;
    *ptr2++ = *ptr1;
    ptr1 = &B(x+5,y);

    *ptr2++ = *ptr1++;
    *ptr2++ = *ptr1++;
    *ptr2++ = *ptr1++;
    *ptr2++ = *ptr1++;
    *ptr2++ = *ptr1++;
    *ptr2++ = *ptr1++;
    *ptr2++ = *ptr1++;
    *ptr2++ = *ptr1;
    ptr1 = &B(x+6,y);

    *ptr2++ = *ptr1++;
    *ptr2++ = *ptr1++;
    *ptr2++ = *ptr1++;
    *ptr2++ = *ptr1++;
    *ptr2++ = *ptr1++;
    *ptr2++ = *ptr1++;
    *ptr2++ = *ptr1++;
    *ptr2++ = *ptr1;
    ptr1 = &B(x+7,y);

    *ptr2++ = *ptr1++;
    *ptr2++ = *ptr1++;
    *ptr2++ = *ptr1++;
    *ptr2++ = *ptr1++;
    *ptr2++ = *ptr1++;
    *ptr2++ = *ptr1++;
    *ptr2++ = *ptr1++;
    *ptr2 = *ptr1;
}

void packgeA(float *A, float *packgeA, int x, int y, int K) {
    float *ptr1 = &A(x,y);
    float *ptr2 = packgeA;

    *ptr2++ = *ptr1++;
    *ptr2++ = *ptr1++;
    *ptr2++ = *ptr1++;
    *ptr2++ = *ptr1++;
    *ptr2++ = *ptr1++;
    *ptr2++ = *ptr1++;
    *ptr2++ = *ptr1++;
    *ptr2++ = *ptr1;
    ptr1 = &A(x+1,y);

    *ptr2++ = *ptr1++;
    *ptr2++ = *ptr1++;
    *ptr2++ = *ptr1++;
    *ptr2++ = *ptr1++;
    *ptr2++ = *ptr1++;
    *ptr2++ = *ptr1++;
    *ptr2++ = *ptr1++;
    *ptr2++ = *ptr1;
    ptr1 = &A(x+2,y);

    *ptr2++ = *ptr1++;
    *ptr2++ = *ptr1++;
    *ptr2++ = *ptr1++;
    *ptr2++ = *ptr1++;
    *ptr2++ = *ptr1++;
    *ptr2++ = *ptr1++;
    *ptr2++ = *ptr1++;
    *ptr2++ = *ptr1;
    ptr1 = &A(x+3,y);

    *ptr2++ = *ptr1++;
    *ptr2++ = *ptr1++;
    *ptr2++ = *ptr1++;
    *ptr2++ = *ptr1++;
    *ptr2++ = *ptr1++;
    *ptr2++ = *ptr1++;
    *ptr2++ = *ptr1++;
    *ptr2++ = *ptr1;
    ptr1 = &A(x+4,y);

    *ptr2++ = *ptr1++;
    *ptr2++ = *ptr1++;
    *ptr2++ = *ptr1++;
    *ptr2++ = *ptr1++;
    *ptr2++ = *ptr1++;
    *ptr2++ = *ptr1++;
    *ptr2++ = *ptr1++;
    *ptr2++ = *ptr1;
    ptr1 = &A(x+5,y);

    *ptr2++ = *ptr1++;
    *ptr2++ = *ptr1++;
    *ptr2++ = *ptr1++;
    *ptr2++ = *ptr1++;
    *ptr2++ = *ptr1++;
    *ptr2++ = *ptr1++;
    *ptr2++ = *ptr1++;
    *ptr2++ = *ptr1;
    ptr1 = &A(x+6,y);

    *ptr2++ = *ptr1++;
    *ptr2++ = *ptr1++;
    *ptr2++ = *ptr1++;
    *ptr2++ = *ptr1++;
    *ptr2++ = *ptr1++;
    *ptr2++ = *ptr1++;
    *ptr2++ = *ptr1++;
    *ptr2++ = *ptr1;
    ptr1 = &A(x+7,y);

    *ptr2++ = *ptr1++;
    *ptr2++ = *ptr1++;
    *ptr2++ = *ptr1++;
    *ptr2++ = *ptr1++;
    *ptr2++ = *ptr1++;
    *ptr2++ = *ptr1++;
    *ptr2++ = *ptr1++;
    *ptr2 = *ptr1;

}


void kernel(float *A, float *B, float *C, int M, int N, int K, int x, int y){
    __m256 c0_res = _mm256_setzero_ps();
    __m256 c1_res = _mm256_setzero_ps();
    __m256 c2_res = _mm256_setzero_ps();
    __m256 c3_res = _mm256_setzero_ps();
    __m256 c4_res = _mm256_setzero_ps();
    __m256 c5_res = _mm256_setzero_ps();
    __m256 c6_res = _mm256_setzero_ps();
    __m256 c7_res = _mm256_setzero_ps();


    float pa[64] __attribute__((aligned(32)));
    float pb[64] __attribute__((aligned(32)));

    __m256 a_data0, a_data1, a_data2, a_data3, b_data0, b_data1, b_data2, b_data3;

    for (int i = 0; i < K; i+=8){
        packgeA(A, pa, x, i, K);
        packgeB(B, pb, i, y, N);

        b_data0 = _mm256_load_ps(pb);
        b_data1 = _mm256_load_ps(pb+8);
        b_data2 = _mm256_load_ps(pb+16);
        b_data3 = _mm256_load_ps(pb+24);

        a_data0 = _mm256_set1_ps(pa[0]);
        a_data1 = _mm256_set1_ps(pa[1]);
        a_data2 = _mm256_set1_ps(pa[2]);
        a_data3 = _mm256_set1_ps(pa[3]);

        c0_res = _mm256_fmadd_ps(a_data0, b_data0, c0_res);
        c0_res = _mm256_fmadd_ps(a_data1, b_data1, c0_res);
        c0_res = _mm256_fmadd_ps(a_data2, b_data2, c0_res);
        c0_res = _mm256_fmadd_ps(a_data3, b_data3, c0_res);

        a_data0 = _mm256_set1_ps(pa[8]);
        a_data1 = _mm256_set1_ps(pa[9]);
        a_data2 = _mm256_set1_ps(pa[10]);
        a_data3 = _mm256_set1_ps(pa[11]);

        c1_res = _mm256_fmadd_ps(a_data0, b_data0, c1_res);
        c1_res = _mm256_fmadd_ps(a_data1, b_data1, c1_res);
        c1_res = _mm256_fmadd_ps(a_data2, b_data2, c1_res);
        c1_res = _mm256_fmadd_ps(a_data3, b_data3, c1_res);

        a_data0 = _mm256_set1_ps(pa[16]);
        a_data1 = _mm256_set1_ps(pa[17]);
        a_data2 = _mm256_set1_ps(pa[18]);
        a_data3 = _mm256_set1_ps(pa[19]);

        c2_res = _mm256_fmadd_ps(a_data0, b_data0, c2_res);
        c2_res = _mm256_fmadd_ps(a_data1, b_data1, c2_res);
        c2_res = _mm256_fmadd_ps(a_data2, b_data2, c2_res);
        c2_res = _mm256_fmadd_ps(a_data3, b_data3, c2_res);

        a_data0 = _mm256_set1_ps(pa[24]);
        a_data1 = _mm256_set1_ps(pa[25]);
        a_data2 = _mm256_set1_ps(pa[26]);
        a_data3 = _mm256_set1_ps(pa[27]);

        c3_res = _mm256_fmadd_ps(a_data0, b_data0, c3_res);
        c3_res = _mm256_fmadd_ps(a_data1, b_data1, c3_res);
        c3_res = _mm256_fmadd_ps(a_data2, b_data2, c3_res);
        c3_res = _mm256_fmadd_ps(a_data3, b_data3, c3_res);

        a_data0 = _mm256_set1_ps(pa[32]);
        a_data1 = _mm256_set1_ps(pa[33]);
        a_data2 = _mm256_set1_ps(pa[34]);
        a_data3 = _mm256_set1_ps(pa[35]);

        c4_res = _mm256_fmadd_ps(a_data0, b_data0, c4_res);
        c4_res = _mm256_fmadd_ps(a_data1, b_data1, c4_res);
        c4_res = _mm256_fmadd_ps(a_data2, b_data2, c4_res);
        c4_res = _mm256_fmadd_ps(a_data3, b_data3, c4_res);

        a_data0 = _mm256_set1_ps(pa[40]);
        a_data1 = _mm256_set1_ps(pa[41]);
        a_data2 = _mm256_set1_ps(pa[42]);
        a_data3 = _mm256_set1_ps(pa[43]);

        c5_res = _mm256_fmadd_ps(a_data0, b_data0, c5_res);
        c5_res = _mm256_fmadd_ps(a_data1, b_data1, c5_res);
        c5_res = _mm256_fmadd_ps(a_data2, b_data2, c5_res);
        c5_res = _mm256_fmadd_ps(a_data3, b_data3, c5_res);

        a_data0 = _mm256_set1_ps(pa[48]);
        a_data1 = _mm256_set1_ps(pa[49]);
        a_data2 = _mm256_set1_ps(pa[50]);
        a_data3 = _mm256_set1_ps(pa[51]);

        c6_res = _mm256_fmadd_ps(a_data0, b_data0, c6_res);
        c6_res = _mm256_fmadd_ps(a_data1, b_data1, c6_res);
        c6_res = _mm256_fmadd_ps(a_data2, b_data2, c6_res);
        c6_res = _mm256_fmadd_ps(a_data3, b_data3, c6_res);

        a_data0 = _mm256_set1_ps(pa[56]);
        a_data1 = _mm256_set1_ps(pa[57]);
        a_data2 = _mm256_set1_ps(pa[58]);
        a_data3 = _mm256_set1_ps(pa[59]);

        c7_res = _mm256_fmadd_ps(a_data0, b_data0, c7_res);
        c7_res = _mm256_fmadd_ps(a_data1, b_data1, c7_res);
        c7_res = _mm256_fmadd_ps(a_data2, b_data2, c7_res);
        c7_res = _mm256_fmadd_ps(a_data3, b_data3, c7_res);

        b_data0 = _mm256_load_ps(pb+32);
        b_data1 = _mm256_load_ps(pb+40);
        b_data2 = _mm256_load_ps(pb+48);
        b_data3 = _mm256_load_ps(pb+56);

        a_data0 = _mm256_set1_ps(pa[4]);
        a_data1 = _mm256_set1_ps(pa[5]);
        a_data2 = _mm256_set1_ps(pa[6]);
        a_data3 = _mm256_set1_ps(pa[7]);

        c0_res = _mm256_fmadd_ps(a_data0, b_data0, c0_res);
        c0_res = _mm256_fmadd_ps(a_data1, b_data1, c0_res);
        c0_res = _mm256_fmadd_ps(a_data2, b_data2, c0_res);
        c0_res = _mm256_fmadd_ps(a_data3, b_data3, c0_res);

        a_data0 = _mm256_set1_ps(pa[12]);
        a_data1 = _mm256_set1_ps(pa[13]);
        a_data2 = _mm256_set1_ps(pa[14]);
        a_data3 = _mm256_set1_ps(pa[15]);

        c1_res = _mm256_fmadd_ps(a_data0, b_data0, c1_res);
        c1_res = _mm256_fmadd_ps(a_data1, b_data1, c1_res);
        c1_res = _mm256_fmadd_ps(a_data2, b_data2, c1_res);
        c1_res = _mm256_fmadd_ps(a_data3, b_data3, c1_res);

        a_data0 = _mm256_set1_ps(pa[20]);
        a_data1 = _mm256_set1_ps(pa[21]);
        a_data2 = _mm256_set1_ps(pa[22]);
        a_data3 = _mm256_set1_ps(pa[23]);

        c2_res = _mm256_fmadd_ps(a_data0, b_data0, c2_res);
        c2_res = _mm256_fmadd_ps(a_data1, b_data1, c2_res);
        c2_res = _mm256_fmadd_ps(a_data2, b_data2, c2_res);
        c2_res = _mm256_fmadd_ps(a_data3, b_data3, c2_res);

        a_data0 = _mm256_set1_ps(pa[28]);
        a_data1 = _mm256_set1_ps(pa[29]);
        a_data2 = _mm256_set1_ps(pa[30]);
        a_data3 = _mm256_set1_ps(pa[31]);

        c3_res = _mm256_fmadd_ps(a_data0, b_data0, c3_res);
        c3_res = _mm256_fmadd_ps(a_data1, b_data1, c3_res);
        c3_res = _mm256_fmadd_ps(a_data2, b_data2, c3_res);
        c3_res = _mm256_fmadd_ps(a_data3, b_data3, c3_res);

        a_data0 = _mm256_set1_ps(pa[36]);
        a_data1 = _mm256_set1_ps(pa[37]);
        a_data2 = _mm256_set1_ps(pa[38]);
        a_data3 = _mm256_set1_ps(pa[39]);

        c4_res = _mm256_fmadd_ps(a_data0, b_data0, c4_res);
        c4_res = _mm256_fmadd_ps(a_data1, b_data1, c4_res);
        c4_res = _mm256_fmadd_ps(a_data2, b_data2, c4_res);
        c4_res = _mm256_fmadd_ps(a_data3, b_data3, c4_res);

        a_data0 = _mm256_set1_ps(pa[44]);
        a_data1 = _mm256_set1_ps(pa[45]);
        a_data2 = _mm256_set1_ps(pa[46]);
        a_data3 = _mm256_set1_ps(pa[47]);

        c5_res = _mm256_fmadd_ps(a_data0, b_data0, c5_res);
        c5_res = _mm256_fmadd_ps(a_data1, b_data1, c5_res);
        c5_res = _mm256_fmadd_ps(a_data2, b_data2, c5_res);
        c5_res = _mm256_fmadd_ps(a_data3, b_data3, c5_res);

        a_data0 = _mm256_set1_ps(pa[52]);
        a_data1 = _mm256_set1_ps(pa[53]);
        a_data2 = _mm256_set1_ps(pa[54]);
        a_data3 = _mm256_set1_ps(pa[55]);

        c6_res = _mm256_fmadd_ps(a_data0, b_data0, c6_res);
        c6_res = _mm256_fmadd_ps(a_data1, b_data1, c6_res);
        c6_res = _mm256_fmadd_ps(a_data2, b_data2, c6_res);
        c6_res = _mm256_fmadd_ps(a_data3, b_data3, c6_res);

        a_data0 = _mm256_set1_ps(pa[60]);
        a_data1 = _mm256_set1_ps(pa[61]);
        a_data2 = _mm256_set1_ps(pa[62]);
        a_data3 = _mm256_set1_ps(pa[63]);

        c7_res = _mm256_fmadd_ps(a_data0, b_data0, c7_res);
        c7_res = _mm256_fmadd_ps(a_data1, b_data1, c7_res);
        c7_res = _mm256_fmadd_ps(a_data2, b_data2, c7_res);
        c7_res = _mm256_fmadd_ps(a_data3, b_data3, c7_res);
    }

    _mm256_storeu_ps(&C(x,y), c0_res);
    _mm256_storeu_ps(&C(x+1,y), c1_res);
    _mm256_storeu_ps(&C(x+2,y), c2_res);
    _mm256_storeu_ps(&C(x+3,y), c3_res);
    _mm256_storeu_ps(&C(x+4,y), c4_res);
    _mm256_storeu_ps(&C(x+5,y), c5_res);
    _mm256_storeu_ps(&C(x+6,y), c6_res);
    _mm256_storeu_ps(&C(x+7,y), c7_res);
}


void sgemm_s(float* A, float* B, float* C, int M, int N, int K) 
{
    for (int i = 0; i < M; i += 8) {
        for (int j = 0; j < N; j += 8) {
            kernel(A, B, C, M, N, K, i, j);
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
    for (int i = 0; i < p->m; i += 8) {
        for (int j = 0; j < p->n; j += 8) {
            kernel(p->a, p->b, p->c, p->m, p->n, p->k, i, j);
        }
    }
}


int sgemm(float* A, float* B, float* C, int M, int N, int K) 
{
    int group_m = M/8;
    const int NUM_THREADS = 6;
    int u_size = group_m / NUM_THREADS;
    int y_size = group_m % NUM_THREADS;

    if(u_size == 0){
        sgemm_s(A, B, C, M, N, K);
        return 0;
    }
    
    pthread_t threads[NUM_THREADS]; 
    struct Para p[NUM_THREADS];
    
    for(int i = 0; i < NUM_THREADS; i++){
        p[i]. a = &A[u_size * i * 8 * K];
        p[i] .b = B;
        p[i] .c = &C[u_size * i * 8 * N];
        p[i] .k = K;
        p[i] .n = N;
        if (i == NUM_THREADS -1){
            p[i].m = (u_size + y_size) * 8;
        }else{
            p[i].m = u_size * 8;
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
