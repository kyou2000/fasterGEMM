
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cuda.h"
#include <stdio.h>
#include <stdlib.h>

__global__ void mutmulker1(const float* a, const float* b, float* c, int m, int k, int n) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	float sum = 0;
	if (x < m && y < n) {
		for (int i = 0; i < k; i++) {
			sum += a[x * k + i] * b[i * k + y];
		}
		c[x * k + y] = sum;
	}

}


// 通过使用shreadmem 减少访存
// 这种方法下k不能过大
// 一般不推荐这么使用
template <int BLOCK_SIZE, int K_>
__global__ void matmulker2(float* A, float* B, float* C, int m, int k, int n)
{
	int x = blockDim.x * blockIdx.x + threadIdx.x;
	int y = blockDim.y * blockIdx.y + threadIdx.y;
	float* asptr = A + blockDim.y * blockIdx.y * k;
	float* bsptr = B + blockDim.x * blockIdx.x;

	__shared__ float bla[BLOCK_SIZE][K_];
	__shared__ float blb[K_][BLOCK_SIZE];

	for (int s = 0; s < k; s += blockDim.x) {
		bla[threadIdx.y][threadIdx.x + s] = asptr[threadIdx.x + s + threadIdx.y * k];
		blb[threadIdx.y + s][threadIdx.x] = bsptr[(threadIdx.y + s) * n + threadIdx.x];
	}
	__syncthreads();
	float sum = 0;
	for (int i = 0; i < k; i++) {
		sum += bla[threadIdx.y][i] * blb[i][threadIdx.x];
	}

	C[y * n + x] = sum;
}

// 对矩阵分块计算
template <int BLOCK_SIZE>
__global__ void matmulker3(float* A, float* B, float* C, int m, int k, int n)
{
	int x = blockDim.x * blockIdx.x + threadIdx.x;
	int y = blockDim.y * blockIdx.y + threadIdx.y;
	float* asptr = A + blockDim.y * blockIdx.y * k;
	float* bsptr = B + blockDim.x * blockIdx.x;

	__shared__ float bla[BLOCK_SIZE][BLOCK_SIZE];
	__shared__ float blb[BLOCK_SIZE][BLOCK_SIZE];

	float sum = 0;
	for (int s = 0; s < k; s += blockDim.x) {
		bla[threadIdx.y][threadIdx.x] = asptr[threadIdx.x + s + threadIdx.y * k];
		blb[threadIdx.y][threadIdx.x] = bsptr[(threadIdx.y + s) * n + threadIdx.x];
		__syncthreads();
		for (int t = 0; t < BLOCK_SIZE; t++) {
			// 计算每一个线程的计算结果
			sum += bla[threadIdx.y][t] * blb[t][threadIdx.x];
		}
		__syncthreads();
	}

	C[y * n + x] = sum;
}

// 通过一个线程计算更多的数据
template <int BLOCK_SIZE, int SR>
__global__ void matmulker4(float* A, float* B, float* C, int m, int k, int n)
{
	int tx = threadIdx.x;
	int ty = threadIdx.y;
	int bx = blockIdx.x;
	int by = blockIdx.y;
	const int bps = BLOCK_SIZE * SR;
	
	float* asptr = A + blockDim.y * by * k;
	float* bsptr = B + blockDim.x * bx;
	float* csptr = C + n* by * bps + bx * bps;

	__shared__ float bla[bps][bps];
	__shared__ float blb[bps][bps];

	float sum[SR][SR] = { 0.f };
	for (int s = 0; s < k; s += bps) {
		for (int i = 0; i < SR; i++) {
			for (int j = 0; j < SR; j++) {
				bla[ty + i*BLOCK_SIZE][tx + j*BLOCK_SIZE] = asptr[(ty + i*BLOCK_SIZE) * k + tx + j*BLOCK_SIZE + s];
				blb[ty + i*BLOCK_SIZE][tx + j*BLOCK_SIZE] = bsptr[(s + ty + i*BLOCK_SIZE)*n + +tx + j * BLOCK_SIZE];
			}
		}
		__syncthreads();
		for (int i = 0; i < SR; i++) {
			for (int j = 0; j < SR; j++) {
				for (int js = 0; js < bps; js++) {
					sum[i][j] += bla[ty + i * BLOCK_SIZE][js] * blb[js][tx + j * BLOCK_SIZE];
				}
			}
		}
		__syncthreads();
	}
	for (int i = 0; i < SR; i++) {
		for (int j = 0; j < SR; j++) {
			
			csptr[(i * BLOCK_SIZE + ty)*n + j * BLOCK_SIZE + tx] = sum[i][j];
		}
	}
}

// 使用float4一次读取连续的4个数据
// block切分为8x32

#define FETCH_FLOAT4(ptr) (reinterpret_cast<float4 *>(&(ptr))[0])

template<int m_pre, int n_pre, int k_pre, int pre_num>
__global__ void matmulker5(float* A, float* B, float* C, int m, int k, int n)
{
	int tx = threadIdx.x;
	int ty = threadIdx.y;
	int bx = blockIdx.x;
	int by = blockIdx.y;

	float* asptr = A + m_pre * by * k;
	float* bsptr = B + n_pre * bx;

	__shared__ float bla[m_pre][k_pre];
	__shared__ float blb[k_pre][n_pre];

	float sum[pre_num] = {0.f};
	
	for (int s = 0; s < k; s+= k_pre) {
		FETCH_FLOAT4(bla[ty][tx * pre_num]) = FETCH_FLOAT4(asptr[ty*k + s + tx*pre_num]);
		FETCH_FLOAT4(blb[ty][tx * pre_num]) = FETCH_FLOAT4(bsptr[(ty+s)*n + tx*pre_num]);

		__syncthreads();
		for (int i = 0; i < pre_num; i++) {
			for (int j = 0; j < k_pre; j++) {
				sum[i] += bla[ty][j] * blb[j][tx*pre_num + i];
			}
		}
		__syncthreads();
	}

	float* csptr = C + by * m_pre * n + bx * n_pre;
	for (int i = 0; i < pre_num; i++) {
		C[ty*n + pre_num*tx + i] = sum[i];
	}
	
}


// 利用寄存器存放数据，在读取时访问寄存器

template<int m_pre, int n_pre, int k_pre, int pre_num>
__global__ void matmulker6(float* A, float* B, float* C, int m, int k, int n)
{
	int tx = threadIdx.x;
	int ty = threadIdx.y;
	int bx = blockIdx.x;
	int by = blockIdx.y;
	int tid = ty * blockDim.x + tx;

	float* asptr = A + m_pre * by * k;
	float* bsptr = B + n_pre * bx;

	int ctx = tid % 16;
	int cty = tid / 16;

	__shared__ float bla[m_pre][k_pre];
	__shared__ float blb[k_pre][n_pre];

	const int reg_per = pre_num / 2;
	
	float reg_a[reg_per] = {0.f};
	float reg_b[reg_per] = {0.f};
	float reg_res[reg_per][reg_per] = { 0.f };
	

	for (int s = 0; s < k; s += k_pre) {
		FETCH_FLOAT4(bla[ty][tx * pre_num]) = FETCH_FLOAT4(asptr[ty * k + s + tx * pre_num]);
		FETCH_FLOAT4(blb[ty][tx * pre_num]) = FETCH_FLOAT4(bsptr[(ty + s) * n + tx * pre_num]);

		__syncthreads();
		for (int i = 0; i < k_pre; i++) {
			reg_a[0] = bla[cty*2][i];
			reg_a[1] = bla[cty * 2 +1][i];
			reg_b[0] = blb[i][ctx*2];
			reg_b[1] = blb[i][ctx * 2 +1];

			// 使用循环展开的方式计算
			reg_res[0][0] += reg_a[0] * reg_b[0];
			reg_res[0][1] += reg_a[0] * reg_b[1];
			reg_res[1][0] += reg_a[1] * reg_b[0];
			reg_res[1][1] += reg_a[1] * reg_b[1];
		}
		__syncthreads();
	}

	float* csptr = C + by * m_pre * n + bx * n_pre;
	
	csptr[cty*2 + ctx*2] = reg_res[0][0];
	csptr[cty*2 + ctx*2 +1] = reg_res[0][1];
	csptr[(cty+1)*2 + ctx*2] = reg_res[1][0];
	csptr[(cty+1)*2 + ctx*2 + 1] = reg_res[1][1];

}

// 使用更大的分块放到寄存器中
// 4x4
//一个block中有64*64个数据
template <int m_pre, int n_pre, int k_pre, int m_pre_num, int n_pre_num, int k_pre_num>
__global__ void matmulker7(float* A, float* B, float* C, int m, int k, int n)
{
	int tx = threadIdx.x;
	int ty = threadIdx.y;
	int bx = blockIdx.x;
	int by = blockIdx.y;
	

	float* asptr = A + m_pre * by * k;
	float* bsptr = B + n_pre * bx;
	float* csptr = C + by * m_pre * n + bx * n_pre;

	__shared__ float bla[m_pre][k_pre];
	__shared__ float blb[k_pre][n_pre];

	float reg_a[m_pre_num] = {0.f};
	float reg_b[n_pre_num] = {0.f};
	float res[m_pre_num][n_pre_num] = {0.f};

	for (int s = 0; s < k; s += k_pre) {

		for (int i = 0; i < k_pre_num; i++) {
			FETCH_FLOAT4(bla[ty*m_pre_num + i][tx*k_pre_num]) = FETCH_FLOAT4(asptr[(ty*m_pre_num + i)* k + s + tx*k_pre_num]);
		}

		for (int i = 0; i < k_pre_num; i++) {
			FETCH_FLOAT4(blb[ty*k_pre_num + i][tx*n_pre_num]) = FETCH_FLOAT4(bsptr[(ty*k_pre_num + i + s)*n + tx*n_pre_num]);
		}
		__syncthreads();

		for (int i = 0; i < k_pre; i++) {
			reg_a[0] = bla[ty*m_pre_num][i];
			reg_a[1] = bla[ty*m_pre_num + 1][i];
			reg_a[2] = bla[ty*m_pre_num + 2][i];
			reg_a[3] = bla[ty*m_pre_num + 3][i];

			FETCH_FLOAT4(reg_b[0]) = FETCH_FLOAT4(blb[i][tx*n_pre_num]);

			for (int s = 0; s < m_pre_num; s++) {
				for (int q = 0; q < n_pre_num; q++) {
					res[s][q] += reg_a[s] * reg_b[q];
				}
			}
		}
		__syncthreads();
	}

	// 使用float4方式写回
	/*
	for (int i = 0; i < m_pre_num; i++) {
		FETCH_FLOAT4(csptr[(ty * m_pre_num + i) * n + tx * n_pre_num]) = FETCH_FLOAT4(res[i][0]);
	}
	*/

	for (int i = 0; i < m_pre_num; i++) {
		for (int j = 0; j < n_pre_num; j++) {
			csptr[(ty*m_pre_num + i)*n + tx*n_pre_num + j] = res[i][j];
		}
	}
}

// 使用转置方式对A中的块操作，再用float4读取数据
template <int m_pre, int n_pre, int k_pre, int m_pre_num, int n_pre_num, int k_pre_num>
__global__ void matmulker8(float* A, float* B, float* C, int m, int k, int n)
{
	int tx = threadIdx.x;
	int ty = threadIdx.y;
	int bx = blockIdx.x;
	int by = blockIdx.y;


	float* asptr = A + m_pre * by * k;
	float* bsptr = B + n_pre * bx;
	float* csptr = C + by * m_pre * n + bx * n_pre;

	__shared__ float bla[k_pre][m_pre];
	__shared__ float blb[k_pre][n_pre];

	float a_[k_pre_num] = {0.f};
	float reg_a[m_pre_num] = { 0.f };
	float reg_b[n_pre_num] = { 0.f };
	float res[m_pre_num][n_pre_num] = { 0.f };

	for (int s = 0; s < k; s += k_pre) {

		for (int i = 0; i < k_pre_num; i++) {
			// FETCH_FLOAT4(bla[ty * m_pre_num + i][tx * k_pre_num]) = FETCH_FLOAT4(asptr[(ty * m_pre_num + i) * k + s + tx * k_pre_num]);
			FETCH_FLOAT4(a_[0]) = FETCH_FLOAT4(asptr[(ty * m_pre_num + i) * k + s + tx * k_pre_num]);
			bla[tx*k_pre_num][ty*m_pre_num + i] = a_[0];
			bla[tx*k_pre_num + 1][ty*m_pre_num + i] = a_[1];
			bla[tx*k_pre_num + 2][ty*m_pre_num + i] = a_[2];
			bla[tx*k_pre_num + 3][ty*m_pre_num + i] = a_[3];
		}

		for (int i = 0; i < k_pre_num; i++) {
			FETCH_FLOAT4(blb[ty * k_pre_num + i][tx * n_pre_num]) = FETCH_FLOAT4(bsptr[(ty * k_pre_num + i + s) * n + tx * n_pre_num]);
		}
		__syncthreads();

		for (int i = 0; i < k_pre; i++) {
			FETCH_FLOAT4(reg_a[0]) = FETCH_FLOAT4(bla[i][ty * n_pre_num]);

			FETCH_FLOAT4(reg_b[0]) = FETCH_FLOAT4(blb[i][tx * n_pre_num]);

			for (int s = 0; s < m_pre_num; s++) {
				for (int q = 0; q < n_pre_num; q++) {
					res[s][q] += reg_a[s] * reg_b[q];
				}
			}
		}
		__syncthreads();
	}

	// 使用float4方式写回
	/*
	for (int i = 0; i < m_pre_num; i++) {
		FETCH_FLOAT4(csptr[(ty * m_pre_num + i) * n + tx * n_pre_num]) = FETCH_FLOAT4(res[i][0]);
	}
	*/

	for (int i = 0; i < m_pre_num; i++) {
		for (int j = 0; j < n_pre_num; j++) {
			csptr[(ty * m_pre_num + i) * n + tx * n_pre_num + j] = res[i][j];
		}
	}
}

// 使用双缓冲方式读取数据
// 设定块大小为128
template <int m_pre, int n_pre, int k_pre, int m_pre_num, int n_pre_num>
__global__ void matmulker9(float* A, float* B, float* C, int m, int k, int n)
{
	int tx = threadIdx.x;
	int ty = threadIdx.y;
	int bx = blockIdx.x;
	int by = blockIdx.y;

	int tid = ty * blockDim.x + tx;


	float* asptr = A + m_pre * by * k;
	float* bsptr = B + n_pre * bx;
	float* csptr = C + by * m_pre * n + bx * n_pre;

	__shared__ float bla[2][k_pre][m_pre];
	__shared__ float blb[2][k_pre][n_pre];

	float a_[4] = {0.f};
	float reg_a[m_pre_num] = {0.f};
	float reg_b[n_pre_num] = {0.f};
	float res[m_pre_num][n_pre_num] = {0.f};

	// 线程重排
	int a_tile = k_pre / 4;
	int b_tile = n_pre / 4;

	int a_tile_tid_x = tid % a_tile;
	int a_tile_tid_y = tid / a_tile;
	int b_tile_tid_x = tid % b_tile;
	int b_tile_tid_y = tid / b_tile;

	FETCH_FLOAT4(a_[0]) = FETCH_FLOAT4(asptr[a_tile_tid_y*k + a_tile_tid_x*4]);
	bla[0][a_tile_tid_x * 4][a_tile_tid_y] = a_[0];
	bla[0][a_tile_tid_x * 4 + 1][a_tile_tid_y] = a_[0];
	bla[0][a_tile_tid_x * 4 + 2][a_tile_tid_y] = a_[0];
	bla[0][a_tile_tid_x * 4 + 3][a_tile_tid_y] = a_[0];

	FETCH_FLOAT4(blb[0][b_tile_tid_y][b_tile_tid_x*4]) = FETCH_FLOAT4(bsptr[b_tile_tid_y*n + b_tile_tid_x * 4]);
	__syncthreads();

	int sign = 1;
	for (int s = k_pre; s < k; s += k_pre) {
		FETCH_FLOAT4(a_[0]) = FETCH_FLOAT4(asptr[a_tile_tid_y * k + a_tile_tid_x * 4 + s]);
		bla[sign][a_tile_tid_x * 4][a_tile_tid_y] = a_[0];
		bla[sign][a_tile_tid_x * 4 + 1][a_tile_tid_y] = a_[0];
		bla[sign][a_tile_tid_x * 4 + 2][a_tile_tid_y] = a_[0];
		bla[sign][a_tile_tid_x * 4 + 3][a_tile_tid_y] = a_[0];

		FETCH_FLOAT4(blb[sign][b_tile_tid_y][b_tile_tid_x * 4]) = FETCH_FLOAT4(bsptr[(b_tile_tid_y+s) * n + b_tile_tid_x * 4]);

		sign = sign ^ 1;
		for (int i = 0; i < k_pre; i++) {
			FETCH_FLOAT4(reg_a[0]) = FETCH_FLOAT4(bla[sign][i][ty * n_pre_num]);
			FETCH_FLOAT4(reg_a[4]) = FETCH_FLOAT4(bla[sign][i][ty * n_pre_num + 4]);
			FETCH_FLOAT4(reg_b[0]) = FETCH_FLOAT4(blb[sign][i][tx * n_pre_num]);
			FETCH_FLOAT4(reg_b[4]) = FETCH_FLOAT4(blb[sign][i][tx * n_pre_num + 4]);

			for (int s = 0; s < m_pre_num; s++) {
				for (int q = 0; q < n_pre_num; q++) {
					res[s][q] += reg_a[s] * reg_b[q];
				}
			}
		}
		__syncthreads();
	}
	sign = sign ^ 1;
	for (int i = 0; i < k_pre; i++) {
		FETCH_FLOAT4(reg_a[0]) = FETCH_FLOAT4(bla[sign][i][ty * n_pre_num]);
		FETCH_FLOAT4(reg_a[4]) = FETCH_FLOAT4(bla[sign][i][ty * n_pre_num + 4]);
		FETCH_FLOAT4(reg_b[0]) = FETCH_FLOAT4(blb[sign][i][tx * n_pre_num]);
		FETCH_FLOAT4(reg_b[4]) = FETCH_FLOAT4(blb[sign][i][tx * n_pre_num + 4]);

		for (int s = 0; s < m_pre_num; s++) {
			for (int q = 0; q < n_pre_num; q++) {
				res[s][q] += reg_a[s] * reg_b[q];
			}
		}
	}
	__syncthreads();

	// 使用float4方式写回
	
	for (int i = 0; i < m_pre_num; i++) {
		FETCH_FLOAT4(csptr[(ty * m_pre_num + i) * n + tx * n_pre_num]) = FETCH_FLOAT4(res[i][0]);
		FETCH_FLOAT4(csptr[(ty * m_pre_num + i) * n + tx * n_pre_num + 4]) = FETCH_FLOAT4(res[i][4]);
	}
	
}


int main()
{
	const int BLOCK_SIZE = 16;
	// 缩放倍率
	const int sr = 2;
	const int w = 1024;
	int size = w * w * sizeof(float);

	float* h_a = NULL;
	float* h_b = NULL;
	float* h_c = NULL;

	float* da = NULL;
	float* db = NULL;
	float* dc = NULL;


	h_a = (float*)malloc(size);
	h_b = (float*)malloc(size);
	h_c = (float*)malloc(size);

	cudaMalloc((void**)&da, size);
	cudaMalloc((void**)&db, size);
	cudaMalloc((void**)&dc, size);
	// init


	for (int i = 0; i < w * w; i++) {
		h_a[i] = 1;
		h_b[i] = 1;
	}


	cudaMemcpy(da, h_a, size, cudaMemcpyHostToDevice);
	cudaMemcpy(db, h_b, size, cudaMemcpyHostToDevice);

	//dim3 block(BLOCK_SIZE, BLOCK_SIZE);
	//dim3 grid((w + BLOCK_SIZE - 1) / BLOCK_SIZE/ sr, (w + BLOCK_SIZE - 1) / BLOCK_SIZE/ sr);

	//matmulker<BLOCK_SIZE, sr> << <grid, block >> > (da, db, dc, w, w, w);

	// 配合8x32使用
	/*
	const int m_pre = 32;
	const int n_pre = 32;
	const int k_pre = 32;
	const int pre_num = 4;

	dim3 block(8, 32);
	dim3 grid(w / m_pre, w / n_pre);

	matmulker5<m_pre, n_pre, k_pre, pre_num> << <grid, block >> > (da, db, dc, w, w, w);
	*/
	
	// 使用6和7时每块为了能使用float4读取，并且一个线程执行4个数据计算，数据块大小变为64，线程块大小为16x16
	/*
	const int m_pre = 64;
	const int n_pre = 64;
	const int k_pre = 64;
	const int m_pre_num = 4;
	const int n_pre_num = 4;
	const int k_pre_num = 4;

	dim3 block(BLOCK_SIZE, BLOCK_SIZE);
	dim3 grid(w / m_pre, w / n_pre);

	matmulker7<m_pre, n_pre, k_pre, m_pre_num, n_pre_num, k_pre_num> << <grid, block>> > (da, db, dc, w, w, w);
	*/

	const int m_pre = 128;
	const int n_pre = 128;
	const int k_pre = 8;
	const int m_pre_num = 8;
	const int n_pre_num = 8;

	dim3 block(n_pre / n_pre_num, m_pre / m_pre_num);
	dim3 grid(w / n_pre, w / m_pre);

	matmulker9<m_pre, n_pre, k_pre, m_pre_num, n_pre_num> << <grid, block>> > (da, db, dc, w, w, w);
	cudaMemcpy(h_c, dc, size, cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();

	printf("ok!\n");

	cudaFree(da);
	cudaFree(db);
	cudaFree(dc);

	// test
	for (int i = 0; i < 10; i++) {
		printf("%f\n", h_c[i]);
	}

	free(h_a);
	free(h_b);
	free(h_c);
}
