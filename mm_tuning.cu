#include <cstdio>
#include <cstdlib>
#include <string>
#include <ctime>
#include <sys/time.h>
#include <cmath>
#include "cuda_fp16.h"
#include <mma.h>
#include <cuda.h>
#include <ptx.h>
#include <cuda_pipeline.h>

#define cdiv(x, y) (((x) + (y) - 1) / (y))

namespace m128n64k64
{
#define BLK_M (64)
#define BLK_N (64)
#define BLK_K (64)
#define WARP_M (32)
#define WARP_N (64)
#define WARP_K (BLK_K)
#define MMA_M (16)
#define MMA_N (8)
#define MMA_K (16)
#define NUM_WARP_M (BLK_M / WARP_M)
#define NUM_WARP_N (BLK_N / WARP_N)
#define WARP_SIZE (32)
#define NUM_WARP (NUM_WARP_M * NUM_WARP_N)
#define NUM_THREAD (NUM_WARP * WARP_SIZE)
#define NUM_MMA_M (WARP_M / MMA_M)
#define NUM_MMA_K (WARP_K / MMA_K)
#define NUM_MMA_N (WARP_N / MMA_N)


__device__ void load_shm_A(half* shm_A, half* A, int M, int K, int ko) {
    int tid = threadIdx.z * NUM_WARP_N * WARP_SIZE + threadIdx.y * WARP_SIZE + threadIdx.x;
    const int load_elements = 128 / 16;
    const int thread_per_row = BLK_K / load_elements;
    const int rows_per_load = NUM_THREAD / thread_per_row;
    const int load_per_thread = BLK_M / rows_per_load;
    
    for (int i = 0; i < load_per_thread; i++) {
        int row = i * rows_per_load + tid / thread_per_row;
        int col = tid % thread_per_row * load_elements;
        int shm_col = col ^ ((row & 7) << 3);
        __pipeline_memcpy_async(
            &shm_A[row * BLK_K + shm_col],
            &A[(blockIdx.x * BLK_M + row) * K + ko * BLK_K + col],
            16
        );
    }
    __syncthreads();
}

__device__ void load_shm_B(half* shm_B, half* B, int K, int N, int ko) {
    int tid = threadIdx.z * NUM_WARP_N * WARP_SIZE + threadIdx.y * WARP_SIZE + threadIdx.x;
    const int load_elements = 128 / 16;
    const int thread_per_row = BLK_N / load_elements;
    const int rows_per_load = NUM_THREAD / thread_per_row;
    const int load_per_thread = BLK_K / rows_per_load;

    for (int i = 0; i < load_per_thread; i++) {
        int row = i * rows_per_load + tid / thread_per_row;
        int col = tid % thread_per_row * load_elements;
        int shm_col = col ^ ((row & 7) << 3);
        __pipeline_memcpy_async(
            &shm_B[row * BLK_N + shm_col],
            &B[(ko * BLK_K + row) * N + blockIdx.y * BLK_N + col],
            16
        );
    }
    __syncthreads();
}

__device__ void load_reg_A(uint32_t* reg_A, half* shm_A, int ki) {
    for (int m = 0; m < WARP_M / MMA_M; m++) {
        int lane_id = threadIdx.x;
        int row = threadIdx.z * WARP_M + m * MMA_M + lane_id % 16;
        int col = ki * MMA_K + lane_id / 16 * 8;
        int shm_col = col ^ ((row & 7) << 3);
        uint32_t shm_A_lane_addr = __cvta_generic_to_shared(shm_A + row * BLK_K + shm_col);
        LDMATRIX_X4(reg_A[ki * NUM_MMA_M * 4 + m * 4], reg_A[ki * NUM_MMA_M * 4 + m * 4 + 1], 
                    reg_A[ki * NUM_MMA_M * 4 + m * 4 + 2], reg_A[ki * NUM_MMA_M * 4 + m * 4 + 3], 
                    shm_A_lane_addr);
    }
    __syncthreads();
}

__device__ void load_reg_B(uint32_t* reg_B, half* shm_B, int ki) {
    int lane_id = threadIdx.x;
    for (int ni = 0; ni < WARP_N / (MMA_N * 2); ni++) {
        int row = ki * MMA_K + lane_id % 16;
        int col = threadIdx.y * WARP_N + ni * (MMA_N * 2) + lane_id / 16 * 8;
        int shm_col = col ^ ((row & 7) << 3);
        uint32_t shm_B_lane_addr = __cvta_generic_to_shared(shm_B + row * BLK_N + shm_col);
        LDMATRIX_X4_T(reg_B[ki * NUM_MMA_N * 2 + ni * 4], reg_B[ki * NUM_MMA_N * 2 + ni * 4 + 1], 
                    reg_B[ki * NUM_MMA_N * 2 + ni * 4 + 2], reg_B[ki * NUM_MMA_N * 2 + ni * 4 + 3], 
                    shm_B_lane_addr);
    }
}

__device__ void store_C(uint32_t* reg_C, half* C, int M, int N) {
    int lane_id = threadIdx.x;
    for (int m = 0; m < NUM_MMA_M; m++) {
        for (int n = 0; n < NUM_MMA_N; n++) {
            int shm_row = threadIdx.z * WARP_M + m * MMA_M + lane_id / 4;
            int shm_col = threadIdx.y * WARP_N + n * MMA_N + (lane_id % 4) * 2;
            int row = shm_row + blockIdx.x * BLK_M;
            int col = shm_col + blockIdx.y * BLK_N;
            C[row * N + col] = __float2half(*(float*)&reg_C[m * NUM_MMA_N * 4 + n * 4]);
            C[row * N + col + 1] = __float2half(*(float*)&reg_C[m * NUM_MMA_N * 4 + n * 4 + 1]);
            C[(row + 8) * N + col] = __float2half(*(float*)&reg_C[m * NUM_MMA_N * 4 + n * 4 + 2]);
            C[(row + 8) * N + col + 1] = __float2half(*(float*)&reg_C[m * NUM_MMA_N * 4 + n * 4 + 3]);
        }
    }
}

__device__ void pipe_load(half* shm_A, half* shm_B, half* A, half* B, int M ,int N, int K, int ko) {
    shm_A += (ko & 1) * BLK_M * BLK_K;
    shm_B += (ko & 1) * BLK_K * BLK_N;
    load_shm_A(shm_A, A, M, K, ko);
    load_shm_B(shm_B, B, K, N, ko);
}

__device__ void pipe_calc(half* shm_A, half* shm_B, uint32_t* reg_A, uint32_t* reg_B, uint32_t* reg_C, int ko) {
    shm_A += (ko & 1) * BLK_M * BLK_K;
    shm_B += (ko & 1) * BLK_K * BLK_N;
    for (int ki = 0; ki < BLK_K / MMA_K; ki++) {
        load_reg_A(reg_A, shm_A, ki);
        load_reg_B(reg_B, shm_B, ki);
    }
    for (int m = 0; m < NUM_MMA_M; m++) {
        for (int ki = 0; ki < NUM_MMA_K; ki++) {
            for (int n = 0; n < NUM_MMA_N; n++) {
                int idx = m * NUM_MMA_N + n;
                HMMA16816(reg_C[idx * 4], reg_C[idx * 4 + 1], reg_C[idx * 4 + 2], reg_C[idx * 4 + 3],
                          reg_A[ki * NUM_MMA_M * 4 + m * 4], reg_A[ki * NUM_MMA_M * 4 + m * 4 + 1], reg_A[ki * NUM_MMA_M * 4 + m * 4 + 2], reg_A[ki * NUM_MMA_M * 4 + m * 4 + 3],
                          reg_B[ki * NUM_MMA_N * 2 + n * 2], reg_B[ki * NUM_MMA_N * 2 + n * 2 + 1],
                          reg_C[idx * 4], reg_C[idx * 4 + 1], reg_C[idx * 4 + 2], reg_C[idx * 4 + 3]);
            }
        }
    }
}

__global__ void matmul_kernel(int M, int N, int K, half* d_A, half* d_B, half* d_C) {
    __shared__ half shm_A[2 * BLK_M * BLK_K];
    __shared__ half shm_B[2 * BLK_K * BLK_N];

    uint32_t reg_A[NUM_MMA_K * NUM_MMA_M * 4];
    uint32_t reg_B[NUM_MMA_K * NUM_MMA_N * 2];
    uint32_t reg_C[NUM_MMA_M * NUM_MMA_N * 4] = {0};

    pipe_load(shm_A, shm_B, d_A, d_B, M, N, K, 0);
    __pipeline_commit();

    for (int k = 1; k < K / BLK_K; k++) {
        pipe_load(shm_A, shm_B, d_A, d_B, M, N, K, k);
        __pipeline_wait_prior(0);
        __pipeline_commit();
        pipe_calc(shm_A, shm_B, reg_A, reg_B, reg_C, k-1);
    }

    __pipeline_wait_prior(0);
    pipe_calc(shm_A, shm_B, reg_A, reg_B, reg_C, K / BLK_K - 1);

    store_C(reg_C, d_C, M, N);
}
}

void matmul(int M, int N, int K, half* h_A, half* h_B, half* h_C) {
    half* d_A;
    half* d_B;
    half* d_C;
    struct timeval tv;
    double start, end;

    cudaMalloc(&d_A, M * K * sizeof(half));
    cudaMalloc(&d_B, K * N * sizeof(half));
    cudaMalloc(&d_C, M * N * sizeof(half));
    cudaMemcpy(d_A, h_A, M * K * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, K * N * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, h_C, M * N * sizeof(half), cudaMemcpyHostToDevice);

    gettimeofday(&tv, nullptr);
    start = tv.tv_sec + tv.tv_usec / 1.0e6;

    m128n64k64::matmul_kernel<<<dim3(cdiv(M, BLK_M), cdiv(N, BLK_N)), dim3(WARP_SIZE, NUM_WARP_N, NUM_WARP_M)>>>(M, N, K, d_A, d_B, d_C);

    cudaDeviceSynchronize();
    gettimeofday(&tv, nullptr);
    end = tv.tv_sec + tv.tv_usec / 1.0e6;
    printf("matmul time: %lf\n", end - start);

    cudaMemcpy(h_C, d_C, M * N * sizeof(half), cudaMemcpyDeviceToHost);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}