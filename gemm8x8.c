#include <immintrin.h>
#include <windows.h>
#include <stdio.h>
#include <stdlib.h>

#define A(i,j) A[(i)*k + (j)]
#define B(i,j) B[(i)*n + (j)]
#define C(i,j) C[(i)*n + (j)]

#define kr 8
// -mvax -mfma
// kernel 8x8

void packgeB(float *B, float *packgeB, int x, int y, int n) {
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

void packgeA(float *A, float *packgeA, int x, int y, int k) {
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

void kernel(float *A, float *B, float *C, int m, int n, int k, int x, int y) {
    int kr_num = k / kr;

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
        packgeA(A, pa, x, i*kr, k);
        packgeB(B, pb, i*kr, y, n);

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

struct Para{
    float * a;
    float * b;
    float * c;
    int m;
    int k;
    int n;
}*Paraptr;

DWORD WINAPI matmul_t(LPVOID lpParam) {
    struct Para * p = (struct Para *)lpParam;
    for (int i = 0; i < p->m; i += 8) {
        for (int j = 0; j < p->n; j += 8) {
            kernel(p->a, p->b, p->c, p->m, p->n, p->k, i, j);
        }
    }
}

void gemm(float * A, float * B, float * C, int m, int k, int n){

    int group_m = m/8;
    int threadnum = 16;
    int u_size = group_m / threadnum;
    int y_size = group_m % threadnum;

    if(u_size == 0){
        return;
    }
    struct Para p[threadnum];
    HANDLE hThreads[threadnum];

    for(int i = 0; i < threadnum; i++){
        p[i]. a = &A[u_size * i * kr * k];
        p[i] .b = B;
        p[i] .c = &C[u_size * i * kr * n];
        p[i] .k = k;
        p[i] .n = n;
        if (i == threadnum -1){
            p[i].m = (u_size + y_size) * kr;
        }else{
            p[i].m = u_size * kr;
        }
    }

    for (int i = 0; i < threadnum; i++){
        hThreads[i] = CreateThread(NULL, 0, matmul_t, (LPVOID)&p[i], 0, NULL);
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