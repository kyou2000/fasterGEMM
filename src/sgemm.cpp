#include <thread>
#include <vector>
#include <immintrin.h>
#include "sgemm.h"
#include "tools.h"

#define A(i,j) A[(i)*K + (j)]
#define B(i,j) B[(i)*N + (j)]
#define C(i,j) C[(i)*N + (j)]

// 单线程标准计算
void sgemm_v1(float* A, float* B, float* C, int M, int N, int K)
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

void sgemm_v1_mp(float* A, float* B, float* C, int M, int N, int K)
{
    int threadnum = std::thread::hardware_concurrency();
    int u_size = M / threadnum;
    int y_size = M % threadnum;

    if (u_size == 0){
        sgemm_v1(A, B, C, M, N, K);
        return;
    }
    std::vector<std::thread> threads(threadnum);

    for (int i = 0; i < threadnum; i++){
        if (i == threadnum - 1){
            threads[i] = std::thread(sgemm_v1, &A(i*u_size, 0), B, &C(i*u_size, 0), u_size + y_size, N, K);
        }else{
            threads[i] = std::thread(sgemm_v1, &A(i*u_size, 0), B, &C(i*u_size, 0), u_size, N, K);
        }
    }

    for (int i = 0; i < threadnum; i++){
        threads[i].join();
    }
}

//sgemm 4*4 SIMD

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


void sgemm_v2_mp(float* A, float* B, float* C, int M, int N, int K)
{
    int threadnum = std::thread::hardware_concurrency();
    int s_m = M / 4;
    int err = M % 4;
    if (err != 0){
        return;
    }
    
    int u_size = s_m / threadnum;
    int y_size = s_m % threadnum;

    if (u_size == 0){
        sgemm_v2(A, B, C, M, N, K);
        return;
    }
    std::vector<std::thread> threads(threadnum);

    for (int i = 0; i < threadnum; i++){
        if (i == threadnum - 1){
            threads[i] = std::thread(sgemm_v2, &A(i*u_size*4, 0), B, &C(i*u_size*4, 0), (u_size + y_size)*4, N, K);
        }else{
            threads[i] = std::thread(sgemm_v2, &A(i*u_size*4, 0), B, &C(i*u_size*4, 0), u_size*4, N, K);
        }
    }

    for (int i = 0; i < threadnum; i++){
        threads[i].join();
    }
}

// sgemm 8x8 SIMD


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

void matmul8x8_ker(float *A, float *B, float *C, int M, int N, int K, int x, int y) {
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

void sgemm_v3(float *A, float *B, float *C, int M, int N, int K) 
{
    for (int i = 0; i < M; i += 8) {
        for (int j = 0; j < N; j += 8) {
            matmul8x8_ker(A, B, C, M, N, K, i, j);
        }
    }
}

void sgemm_v3_mp(float* A, float* B, float* C, int M, int N, int K)
{
    int threadnum = std::thread::hardware_concurrency();
    int s_m = M / 8;
    int err = M % 8;
    if (err != 0){
        return;
    }
    int u_size = s_m / threadnum;
    int y_size = s_m % threadnum;

    if (u_size == 0){
        sgemm_v3(A, B, C, M, N, K);
        return;
    }
    std::vector<std::thread> threads(threadnum);

    for (int i = 0; i < threadnum; i++){
        if (i == threadnum - 1){
            threads[i] = std::thread(sgemm_v3, &A(i*u_size*8, 0), B, &C(i*u_size*8, 0), (u_size + y_size)*8, N, K);
        }else{
            threads[i] = std::thread(sgemm_v3, &A(i*u_size*8, 0), B, &C(i*u_size*8, 0), u_size*8, N, K);
        }
    }

    for (int i = 0; i < threadnum; i++){
        threads[i].join();
    }
}


void sgemm(float* A, float* B, float* C, int M, int N, int K)
{
    int s = 0;
    int check_M1 = M % 8;
    int check_N1 = N % 8;
    int check_K1 = K % 8;
    
    int check_M2 = M % 4;
    int check_N2 = N % 4;
    int check_K2 = K % 4;

    float *A_pad_ptr = nullptr;
    float *B_pad_ptr = nullptr;
    float *C_pad_ptr = nullptr;
    
    // 填充后的各方向的大小
    int sm = 0;
    int sn = 0;
    int sk = 0;

    if (M < 0 || N < 0 || K < 0){
        return;
    }

    if(M == 1){
        return;
    }
    
    if (M <= 64 && N <= 64 && K <=  64){
        sgemm_v1_mp(A, B, C, M, N, K);
        return;
    }
    if (M > 64 || N > 64 || K > 64){
        if(check_M1 == 0 && check_N1 == 0 && check_K1 == 0){
            sgemm_v3_mp(A, B, C, M, N, K);
            return;
        }
        else if(check_M2 == 0 && check_N2 == 0 && check_K2 == 0){
            sgemm_v2_mp(A, B, C, M, N, K);
            return;
        }else{
            if(check_M2 != 0){
                sm = M + (4 - check_M2);
            }

            if(check_N2 != 0){
                sn = N + (4 - check_N2);
            }

            if(check_K2 != 0){
                sk = K + (4 - check_K2);
            }
        }   
    }

    if(sm == 0 && sn == 0 && sk != 0){
        A_pad_ptr = new float[M * sk];
        B_pad_ptr = new float[sk * N];
        datacpy_f(A_pad_ptr, A, M, sk, M, K);
        datacpy_f(B_pad_ptr, B, sk, N, K, N);
        sgemm_v2_mp(A_pad_ptr, B_pad_ptr, C, M, N, sk);
    }

    else if(sm == 0 && sn != 0 && sk == 0){
        B_pad_ptr = new float[K * sn];
        C_pad_ptr = new float[M * sn];
        datacpy_f(B_pad_ptr, B, K, sn, K, N);
        sgemm_v2_mp(A, B_pad_ptr, C_pad_ptr, M, sn, K);
        reload_f(C, C_pad_ptr, M, N, M, sn);
    }

    else if(sm != 0 && sn == 0 && sk == 0){
        A_pad_ptr = new float[sm * K];
        C_pad_ptr = new float[sm * N];
        datacpy_f(A_pad_ptr, A, sm, K, M, K);
        sgemm_v2_mp(A_pad_ptr, B, C_pad_ptr, sm, N, K);
        reload_f(C, C_pad_ptr, M, N, sm, N);
    }

    else if(sm != 0 && sn != 0 && sk == 0){
        A_pad_ptr = new float[sm * K];
        B_pad_ptr = new float[K * sn];
        C_pad_ptr = new float[sm * sn];
        datacpy_f(A_pad_ptr, A, sm, K, M, K);
        datacpy_f(B_pad_ptr, B, K, sn, K, N);
        sgemm_v2_mp(A_pad_ptr, B_pad_ptr, C_pad_ptr, sm, sn, K);
        reload_f(C, C_pad_ptr, M, N, sm, sn);
    }

    else if(sm == 0 && sn != 0 && sk != 0){
        A_pad_ptr = new float[M * sk];
        B_pad_ptr = new float[sk * sn];
        C_pad_ptr = new float[M * sn];
        datacpy_f(A_pad_ptr, A, M, sk, M, K);
        datacpy_f(B_pad_ptr, B, sk, sn, K, N);
        sgemm_v2_mp(A_pad_ptr, B_pad_ptr, C_pad_ptr, M, sn, sk);
        reload_f(C, C_pad_ptr, M, N, M, sn);
    }

    else if(sm != 0 && sn == 0 && sk != 0){
        A_pad_ptr = new float[sm * sk];
        B_pad_ptr = new float[sk * N];
        C_pad_ptr = new float[sm * N];
        datacpy_f(A_pad_ptr, A, sm, sk, M, K);
        datacpy_f(B_pad_ptr, B, sk, N, K, N);
        sgemm_v2_mp(A_pad_ptr, B_pad_ptr, C_pad_ptr, sm, N, sk);
        reload_f(C, C_pad_ptr, M, N, sm, sn);
    }

    else if(sm != 0 && sn != 0 && sk != 0){
        A_pad_ptr = new float[sm * sk];
        B_pad_ptr = new float[sk * sn];
        C_pad_ptr = new float[sm * sn];
        datacpy_f(A_pad_ptr, A, sm, sk, M, K);
        datacpy_f(B_pad_ptr, B, sk, sn, K, N);
        sgemm_v2_mp(A_pad_ptr, B_pad_ptr, C_pad_ptr, sm, sn, sk);
        reload_f(C, C_pad_ptr, M, N, sm, sn);
    }

    if(A_pad_ptr != nullptr){
        delete[] A_pad_ptr;
    }
    if(B_pad_ptr != nullptr){
        delete[] B_pad_ptr;
    }
    if(C_pad_ptr != nullptr){
        delete[] C_pad_ptr;
    }
}
