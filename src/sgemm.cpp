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

typedef union{
    __m128 v;
    float d[4];
}rsv;

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
    float pa[16];
    float pb[16];

    rsv res0, res1, res2, res3;
    __m128 b0, b1, b2, b3;
    __m128 a0, a1, a2, a3, a4, a5, a6, a7;
    res0.v = _mm_setzero_ps();
    res1.v = _mm_setzero_ps();
    res2.v = _mm_setzero_ps();
    res3.v = _mm_setzero_ps();

    for (int s = 0; s < K; s += 4){
        packA(pa, A, x, s, K);
        packB(pb, B, s, y, N);

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

// 寄存器数据
typedef union{
    __m256 v;
    float data[8];
}reg_d;

void matmul8x8_ker(float *A, float *B, float *C, int M, int N, int K, int x, int y) {
    int kr_num = K / 8;

    float pa[64];
    float pb[64];

    reg_d c0_res, c1_res, c2_res, c3_res, c4_res, c5_res, c6_res, c7_res;
    reg_d b_data0, b_data1, b_data2, b_data3;
    reg_d a_data0, a_data1, a_data2, a_data3;

    c0_res.v = _mm256_setzero_ps();
    c1_res.v = _mm256_setzero_ps();
    c2_res.v = _mm256_setzero_ps();
    c3_res.v = _mm256_setzero_ps();
    c4_res.v = _mm256_setzero_ps();
    c5_res.v = _mm256_setzero_ps();
    c6_res.v = _mm256_setzero_ps();
    c7_res.v = _mm256_setzero_ps();
    
    for (int i = 0; i < kr_num; i++) {
        packgeA(A, pa, x, i*8, K);
        packgeB(B, pb, i*8, y, N);

        b_data0.v = _mm256_loadu_ps(pb);
        b_data1.v = _mm256_loadu_ps(pb+8);
        b_data2.v = _mm256_loadu_ps(pb+16);
        b_data3.v = _mm256_loadu_ps(pb+24);

        a_data0.v = _mm256_set1_ps(pa[0]);
        a_data1.v = _mm256_set1_ps(pa[1]);
        a_data2.v = _mm256_set1_ps(pa[2]);
        a_data3.v = _mm256_set1_ps(pa[3]);

        c0_res.v = _mm256_fmadd_ps(a_data0.v, b_data0.v, c0_res.v);
        c0_res.v = _mm256_fmadd_ps(a_data1.v, b_data1.v, c0_res.v);
        c0_res.v = _mm256_fmadd_ps(a_data2.v, b_data2.v, c0_res.v);
        c0_res.v = _mm256_fmadd_ps(a_data3.v, b_data3.v, c0_res.v);

        a_data0.v = _mm256_set1_ps(pa[8]);
        a_data1.v = _mm256_set1_ps(pa[9]);
        a_data2.v = _mm256_set1_ps(pa[10]);
        a_data3.v = _mm256_set1_ps(pa[11]);

        c1_res.v = _mm256_fmadd_ps(a_data0.v, b_data0.v, c1_res.v);
        c1_res.v = _mm256_fmadd_ps(a_data1.v, b_data1.v, c1_res.v);
        c1_res.v = _mm256_fmadd_ps(a_data2.v, b_data2.v, c1_res.v);
        c1_res.v = _mm256_fmadd_ps(a_data3.v, b_data3.v, c1_res.v);

        a_data0.v = _mm256_set1_ps(pa[16]);
        a_data1.v = _mm256_set1_ps(pa[17]);
        a_data2.v = _mm256_set1_ps(pa[18]);
        a_data3.v = _mm256_set1_ps(pa[19]);

        c2_res.v = _mm256_fmadd_ps(a_data0.v, b_data0.v, c2_res.v);
        c2_res.v = _mm256_fmadd_ps(a_data1.v, b_data1.v, c2_res.v);
        c2_res.v = _mm256_fmadd_ps(a_data2.v, b_data2.v, c2_res.v);
        c2_res.v = _mm256_fmadd_ps(a_data3.v, b_data3.v, c2_res.v);

        a_data0.v = _mm256_set1_ps(pa[24]);
        a_data1.v = _mm256_set1_ps(pa[25]);
        a_data2.v = _mm256_set1_ps(pa[26]);
        a_data3.v = _mm256_set1_ps(pa[27]);

        c3_res.v = _mm256_fmadd_ps(a_data0.v, b_data0.v, c3_res.v);
        c3_res.v = _mm256_fmadd_ps(a_data1.v, b_data1.v, c3_res.v);
        c3_res.v = _mm256_fmadd_ps(a_data2.v, b_data2.v, c3_res.v);
        c3_res.v = _mm256_fmadd_ps(a_data3.v, b_data3.v, c3_res.v);

        a_data0.v = _mm256_set1_ps(pa[32]);
        a_data1.v = _mm256_set1_ps(pa[33]);
        a_data2.v = _mm256_set1_ps(pa[34]);
        a_data3.v = _mm256_set1_ps(pa[35]);

        c4_res.v = _mm256_fmadd_ps(a_data0.v, b_data0.v, c4_res.v);
        c4_res.v = _mm256_fmadd_ps(a_data1.v, b_data1.v, c4_res.v);
        c4_res.v = _mm256_fmadd_ps(a_data2.v, b_data2.v, c4_res.v);
        c4_res.v = _mm256_fmadd_ps(a_data3.v, b_data3.v, c4_res.v);

        a_data0.v = _mm256_set1_ps(pa[40]);
        a_data1.v = _mm256_set1_ps(pa[41]);
        a_data2.v = _mm256_set1_ps(pa[42]);
        a_data3.v = _mm256_set1_ps(pa[43]);

        c5_res.v = _mm256_fmadd_ps(a_data0.v, b_data0.v, c5_res.v);
        c5_res.v = _mm256_fmadd_ps(a_data1.v, b_data1.v, c5_res.v);
        c5_res.v = _mm256_fmadd_ps(a_data2.v, b_data2.v, c5_res.v);
        c5_res.v = _mm256_fmadd_ps(a_data3.v, b_data3.v, c5_res.v);

        a_data0.v = _mm256_set1_ps(pa[48]);
        a_data1.v = _mm256_set1_ps(pa[49]);
        a_data2.v = _mm256_set1_ps(pa[50]);
        a_data3.v = _mm256_set1_ps(pa[51]);

        c6_res.v = _mm256_fmadd_ps(a_data0.v, b_data0.v, c6_res.v);
        c6_res.v = _mm256_fmadd_ps(a_data1.v, b_data1.v, c6_res.v);
        c6_res.v = _mm256_fmadd_ps(a_data2.v, b_data2.v, c6_res.v);
        c6_res.v = _mm256_fmadd_ps(a_data3.v, b_data3.v, c6_res.v);

        a_data0.v = _mm256_set1_ps(pa[56]);
        a_data1.v = _mm256_set1_ps(pa[57]);
        a_data2.v = _mm256_set1_ps(pa[58]);
        a_data3.v = _mm256_set1_ps(pa[59]);

        c7_res.v = _mm256_fmadd_ps(a_data0.v, b_data0.v, c7_res.v);
        c7_res.v = _mm256_fmadd_ps(a_data1.v, b_data1.v, c7_res.v);
        c7_res.v = _mm256_fmadd_ps(a_data2.v, b_data2.v, c7_res.v);
        c7_res.v = _mm256_fmadd_ps(a_data3.v, b_data3.v, c7_res.v);

        b_data0.v = _mm256_loadu_ps(pb+32);
        b_data1.v = _mm256_loadu_ps(pb+40);
        b_data2.v = _mm256_loadu_ps(pb+48);
        b_data3.v = _mm256_loadu_ps(pb+56);

        a_data0.v = _mm256_set1_ps(pa[4]);
        a_data1.v = _mm256_set1_ps(pa[5]);
        a_data2.v = _mm256_set1_ps(pa[6]);
        a_data3.v = _mm256_set1_ps(pa[7]);

        c0_res.v = _mm256_fmadd_ps(a_data0.v, b_data0.v, c0_res.v);
        c0_res.v = _mm256_fmadd_ps(a_data1.v, b_data1.v, c0_res.v);
        c0_res.v = _mm256_fmadd_ps(a_data2.v, b_data2.v, c0_res.v);
        c0_res.v = _mm256_fmadd_ps(a_data3.v, b_data3.v, c0_res.v);

        a_data0.v = _mm256_set1_ps(pa[12]);
        a_data1.v = _mm256_set1_ps(pa[13]);
        a_data2.v = _mm256_set1_ps(pa[14]);
        a_data3.v = _mm256_set1_ps(pa[15]);

        c1_res.v = _mm256_fmadd_ps(a_data0.v, b_data0.v, c1_res.v);
        c1_res.v = _mm256_fmadd_ps(a_data1.v, b_data1.v, c1_res.v);
        c1_res.v = _mm256_fmadd_ps(a_data2.v, b_data2.v, c1_res.v);
        c1_res.v = _mm256_fmadd_ps(a_data3.v, b_data3.v, c1_res.v);

        a_data0.v = _mm256_set1_ps(pa[20]);
        a_data1.v = _mm256_set1_ps(pa[21]);
        a_data2.v = _mm256_set1_ps(pa[22]);
        a_data3.v = _mm256_set1_ps(pa[23]);

        c2_res.v = _mm256_fmadd_ps(a_data0.v, b_data0.v, c2_res.v);
        c2_res.v = _mm256_fmadd_ps(a_data1.v, b_data1.v, c2_res.v);
        c2_res.v = _mm256_fmadd_ps(a_data2.v, b_data2.v, c2_res.v);
        c2_res.v = _mm256_fmadd_ps(a_data3.v, b_data3.v, c2_res.v);

        a_data0.v = _mm256_set1_ps(pa[28]);
        a_data1.v = _mm256_set1_ps(pa[29]);
        a_data2.v = _mm256_set1_ps(pa[30]);
        a_data3.v = _mm256_set1_ps(pa[31]);

        c3_res.v = _mm256_fmadd_ps(a_data0.v, b_data0.v, c3_res.v);
        c3_res.v = _mm256_fmadd_ps(a_data1.v, b_data1.v, c3_res.v);
        c3_res.v = _mm256_fmadd_ps(a_data2.v, b_data2.v, c3_res.v);
        c3_res.v = _mm256_fmadd_ps(a_data3.v, b_data3.v, c3_res.v);

        a_data0.v = _mm256_set1_ps(pa[36]);
        a_data1.v = _mm256_set1_ps(pa[37]);
        a_data2.v = _mm256_set1_ps(pa[38]);
        a_data3.v = _mm256_set1_ps(pa[39]);

        c4_res.v = _mm256_fmadd_ps(a_data0.v, b_data0.v, c4_res.v);
        c4_res.v = _mm256_fmadd_ps(a_data1.v, b_data1.v, c4_res.v);
        c4_res.v = _mm256_fmadd_ps(a_data2.v, b_data2.v, c4_res.v);
        c4_res.v = _mm256_fmadd_ps(a_data3.v, b_data3.v, c4_res.v);

        a_data0.v = _mm256_set1_ps(pa[44]);
        a_data1.v = _mm256_set1_ps(pa[45]);
        a_data2.v = _mm256_set1_ps(pa[46]);
        a_data3.v = _mm256_set1_ps(pa[47]);

        c5_res.v = _mm256_fmadd_ps(a_data0.v, b_data0.v, c5_res.v);
        c5_res.v = _mm256_fmadd_ps(a_data1.v, b_data1.v, c5_res.v);
        c5_res.v = _mm256_fmadd_ps(a_data2.v, b_data2.v, c5_res.v);
        c5_res.v = _mm256_fmadd_ps(a_data3.v, b_data3.v, c5_res.v);

        a_data0.v = _mm256_set1_ps(pa[52]);
        a_data1.v = _mm256_set1_ps(pa[53]);
        a_data2.v = _mm256_set1_ps(pa[54]);
        a_data3.v = _mm256_set1_ps(pa[55]);

        c6_res.v = _mm256_fmadd_ps(a_data0.v, b_data0.v, c6_res.v);
        c6_res.v = _mm256_fmadd_ps(a_data1.v, b_data1.v, c6_res.v);
        c6_res.v = _mm256_fmadd_ps(a_data2.v, b_data2.v, c6_res.v);
        c6_res.v = _mm256_fmadd_ps(a_data3.v, b_data3.v, c6_res.v);

        a_data0.v = _mm256_set1_ps(pa[60]);
        a_data1.v = _mm256_set1_ps(pa[61]);
        a_data2.v = _mm256_set1_ps(pa[62]);
        a_data3.v = _mm256_set1_ps(pa[63]);

        c7_res.v = _mm256_fmadd_ps(a_data0.v, b_data0.v, c7_res.v);
        c7_res.v = _mm256_fmadd_ps(a_data1.v, b_data1.v, c7_res.v);
        c7_res.v = _mm256_fmadd_ps(a_data2.v, b_data2.v, c7_res.v);
        c7_res.v = _mm256_fmadd_ps(a_data3.v, b_data3.v, c7_res.v);

    }

    C(x,y) = c0_res.data[0]; C(x,y+1) = c0_res.data[1]; C(x,y+2) = c0_res.data[2]; C(x,y+3) = c0_res.data[3]; 
    C(x,y+4) = c0_res.data[4]; C(x,y+5) = c0_res.data[5]; C(x,y+6) = c0_res.data[6]; C(x,y+7) = c0_res.data[7];

    C(x+1,y) = c1_res.data[0]; C(x+1,y+1) = c1_res.data[1]; C(x+1,y+2) = c1_res.data[2]; C(x+1,y+3) = c1_res.data[3];
    C(x+1,y+4) = c1_res.data[4]; C(x+1,y+5) = c1_res.data[5]; C(x+1,y+6) = c1_res.data[6]; C(x+1,y+7) = c1_res.data[7];

    C(x+2,y) = c2_res.data[0]; C(x+2,y+1) = c2_res.data[1]; C(x+2,y+2) = c2_res.data[2]; C(x+2,y+3) = c2_res.data[3];
    C(x+2,y+4) = c2_res.data[4]; C(x+2,y+5) = c2_res.data[5]; C(x+2,y+6) = c2_res.data[6]; C(x+2,y+7) = c2_res.data[7];

    C(x+3,y) = c3_res.data[0]; C(x+3,y+1) = c3_res.data[1]; C(x+3,y+2) = c3_res.data[2]; C(x+3,y+3) = c3_res.data[3];
    C(x+3,y+4) = c3_res.data[4]; C(x+3,y+5) = c3_res.data[5]; C(x+3,y+6) = c3_res.data[6]; C(x+3,y+7) = c3_res.data[7];

    C(x+4,y) = c4_res.data[0]; C(x+4,y+1) = c4_res.data[1]; C(x+4,y+2) = c4_res.data[2]; C(x+4,y+3) = c4_res.data[3];
    C(x+4,y+4) = c4_res.data[4]; C(x+4,y+5) = c4_res.data[5]; C(x+4,y+6) = c4_res.data[6]; C(x+4,y+7) = c4_res.data[7];
    
    C(x+5,y) = c5_res.data[0]; C(x+5,y+1) = c5_res.data[1]; C(x+5,y+2) = c5_res.data[2]; C(x+5,y+3) = c5_res.data[3];
    C(x+5,y+4) = c5_res.data[4]; C(x+5,y+5) = c5_res.data[5]; C(x+5,y+6) = c5_res.data[6]; C(x+5,y+7) = c5_res.data[7];

    C(x+6,y) = c6_res.data[0]; C(x+6,y+1) = c6_res.data[1]; C(x+6,y+2) = c6_res.data[2]; C(x+6,y+3) = c6_res.data[3];
    C(x+6,y+4) = c6_res.data[4]; C(x+6,y+5) = c6_res.data[5]; C(x+6,y+6) = c6_res.data[6]; C(x+6,y+7) = c6_res.data[7];

    C(x+7,y) = c7_res.data[0]; C(x+7,y+1) = c7_res.data[1]; C(x+7,y+2) = c7_res.data[2]; C(x+7,y+3) = c7_res.data[3];
    C(x+7,y+4) = c7_res.data[4]; C(x+7,y+5) = c7_res.data[5]; C(x+7,y+6) = c7_res.data[6]; C(x+7,y+7) = c7_res.data[7];

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