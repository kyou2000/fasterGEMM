通用矩阵乘法的一般优化方法

文件中包含了3种常见的方式，基础循环方式，4x4分块以及8x8分块计算。

该分块目前还不是最快的方案，根据不同硬件的情况需要不同大小的分块，以适应不同大小的cache大小，另外，线程的数量根据cpu核心数量定，
一般情况下推荐与核心数量相同。

支持架构： X86_64平台
操作系统： Windows & Linux
示例部分由C语言编写：
其中8x8分块使用了avx和fma指令集进行计算，其他均使用了多线程计算，

在使用fma以及avx指令集时务必加上链接选项，机器本身需要支持avx或avx2。

多线程采用windowsAPI，支持在windows系统上使用，如果是在Linux或Uinx上请使用Pthread。

矩阵存储方式：行主元

新加入的4x4分块使用SIMD指令计算，后续会加入其他算子优化以及CUDA程序。


应用部分由c++编写：
安装方法：
1. 下载源码
    git clone https://github.com/kyou2000/fasterGEMM.git

2.切换到源码目录下
    cd fasterGEMM
    mkdir build
    cd build
    cmake ..
    make

编译后的动态库和静态库位于lib目录下。

该计算库学习自OpenBLAS，同样分为三级，即：
向量与向量的计算；
向量与矩阵的计算；
矩阵与矩阵的计算。

主要方法及头文件：
sgemm.h:
单线程标准矩阵乘法
void sgemm_v1(float* A, float* B, float* C, int M, int N, int K);
多线程矩阵乘法
void sgemm_v1_mp(float* A, float* B, float* C, int M, int N, int K);
4x4分块单线程乘法
void sgemm_v2(float* A, float* B, float* C, int M, int N, int K);
4x4分块多线程乘法
void sgemm_v2_mp(float* A, float* B, float* C, int M, int N, int K);
8x8分块单线程矩阵乘法
void sgemm_v3(float *A, float *B, float *C, int M, int N, int K);
8x8分块多线程矩阵乘法
void sgemm_v3_mp(float *A, float *B, float *C, int M, int N, int K);
自动适应矩阵大小乘法*（可以自动适配不同形状的矩阵）
void sgemm(float* A, float* B, float* C, int M, int N, int K);

（源矩阵分块乘法中不支持不能完整分为一个块大小的矩阵，建议使用最后一个方法。）

transpose.h:
矩阵转置方法：
A为输入矩阵，B为输出矩阵。
m为A的行数，n为A的列数。
void transpose(float *A, float *B, int m, int n)；

向量与矩阵的乘法：
vecxmat.h：
单线程：
void matmul(float * A, float * B, float * C, int m, int n);
多线程：
void matmul_mp(float * A, float * B, float * C, int m, int n);
向量与转置后的矩阵乘法：
单线程：
void matmul_tanspose(float * A, float * B, float * C, int m, int n);
多线程：
void matmul_tanspose_mp(float * A, float * B, float * C, int m, int n);

向量与向量的计算：
vecxvec.h：
向量乘法
void vecmul_mp(float *A, float *B, float *C, int n);
向量加法
void vecadd_mp(float *A, float *B, float *C, int n);
向量减法
void vecsub_mp(float *A, float *B, float *C, int n);
向量内积
float vecdot_mp(float *A, float *B, int n);


后续不断更新，更多算子将加入到项目中。
目前CUDA并未融入项目中，后续会加入。


其他方法在学习中，请支持我。
