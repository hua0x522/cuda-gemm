/*
    single-warp single-tile 2-stage
*/
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

#define BLK_M 16
#define BLK_N 8
#define BLK_K 16
#define WARP_M 16
#define WARP_N 8
#define WARP_K 16
#define MMA_M 16
#define MMA_N 8
#define MMA_K 16

#define BLK_SIZE (32 * BLK_N / WARP_N * BLK_M / WARP_M)
#define OFFSET(ptr, x, y, ld) ((ptr) + (x) * (ld) + (y))

__device__ void load_shm_A(half* shm_A, half* A, int M, int K, int ko) {
    /*
    d_A => shm_A
    d_A[bid.x * BLK_M : (bid.x+1) * BLK_M, kid * BLK_K : (kid+1) * BLK_K]
    shm_A[0 : BLK_M, 0 : BLK_K]
    */
    int tid = threadIdx.z * 32 * BLK_M / WARP_M + threadIdx.y * 32 + threadIdx.x;
    half* blk_A = OFFSET(A, blockIdx.x * BLK_M, ko * BLK_K, K);

    for (int t = tid; t < BLK_M * BLK_K / 8; t += BLK_SIZE) {
        int i = t / (BLK_K / 8);
        int j = t % (BLK_K / 8);
        // *((float4*)shm_A + t) = *(OFFSET((float4*)blk_A, i, j, K / 8));
        __pipeline_memcpy_async((float4*)shm_A + t, OFFSET((float4*)blk_A, i, j, K / 8), 16);
    }
}

__device__ void load_shm_B(half* shm_B, half* B, int N, int K, int ko) {
    /*
    d_B => shm_B
    d_B[bid.y * BLK_N : (bid.y+1) * BLK_N, kid * BLK_K : (kid+1) * BLK_K]
    shm_B[0 : BLK_N, 0 : BLK_K]
    */
    int tid = threadIdx.z * 32 * BLK_M / WARP_M + threadIdx.y * 32 + threadIdx.x;
    half* blk_B = OFFSET(B, blockIdx.y * BLK_N, ko * BLK_K, K);

    for (int t = tid; t < BLK_N * BLK_K / 8; t += BLK_SIZE) {
        int i = t / (BLK_K / 8);
        int j = t % (BLK_K / 8);
        // *((float4*)shm_B + t) = *(OFFSET((float4*)blk_B, i, j, K / 8));
        __pipeline_memcpy_async((float4*)shm_B + t, OFFSET((float4*)blk_B, i, j, K / 8), 16);
    }
}

__device__ void store_shm_C(half* shm_C, half* C, int M, int N) {
    int tid = threadIdx.z * 32 * BLK_M / WARP_M + threadIdx.y * 32 + threadIdx.x;
    half* blk_C = OFFSET(C, blockIdx.x * BLK_M, blockIdx.y * BLK_N, N);

    for (int t = tid; t < BLK_M * BLK_N / 8; t += BLK_SIZE) {
        int i = t / (BLK_N / 8);
        int j = t % (BLK_N / 8);
        *(OFFSET((float4*)blk_C, i, j, N / 8)) = *((float4*)shm_C + t);
    }
} 

__device__ void load_reg_A(uint32_t* reg_A, half* shm_A) {
    int lane_id = threadIdx.x;
    uint32_t shm_A_lane_addr = __cvta_generic_to_shared(OFFSET(shm_A, lane_id % 16, lane_id / 16 * 8, BLK_K));
    LDMATRIX_X4(reg_A[0], reg_A[1], reg_A[2], reg_A[3], shm_A_lane_addr);
}

__device__ void load_reg_B(uint32_t* reg_B, half* shm_B) {
    int lane_id = threadIdx.x;
    uint32_t shm_B_lane_addr = __cvta_generic_to_shared(OFFSET(shm_B, lane_id % 8, ((lane_id / 8) % 2) * 8, BLK_K));
    LDMATRIX_X2(reg_B[0], reg_B[1], shm_B_lane_addr);
}

__device__ void store_reg_C(uint32_t* reg_C, half* shm_C) {
    int lane_id = threadIdx.x;
    *OFFSET(shm_C, lane_id / 4, (lane_id % 4) * 2, BLK_N) = __float2half(*(float*)(reg_C + 0));
    *OFFSET(shm_C, lane_id / 4, (lane_id % 4) * 2 + 1, BLK_N) = __float2half(*(float*)(reg_C + 1));
    *OFFSET(shm_C, lane_id / 4 + 8, (lane_id % 4) * 2, BLK_N) = __float2half(*(float*)(reg_C + 2));
    *OFFSET(shm_C, lane_id / 4 + 8, (lane_id % 4) * 2 + 1, BLK_N) = __float2half(*(float*)(reg_C + 3));
}

__device__ void pipe_load(half* d_A, half* d_B, half* shm_A, half* shm_B, int M, int N, int K, int ko) {
    shm_A += (ko & 3) * BLK_M * BLK_K;
    shm_B += (ko & 3) * BLK_N * BLK_K;

    load_shm_A(shm_A, d_A, M, K, ko);
    load_shm_B(shm_B, d_B, N, K, ko);
}

__device__ void pipe_calc(half* shm_A, half* shm_B, uint32_t* reg_A, uint32_t* reg_B, uint32_t* reg_C, int ko) {
    shm_A += (ko & 3) * BLK_M * BLK_K;
    shm_B += (ko & 3) * BLK_N * BLK_K;
    
    load_reg_A(reg_A, shm_A);
    load_reg_B(reg_B, shm_B);

    HMMA16816(reg_C[0], reg_C[1], reg_C[2], reg_C[3],
                  reg_A[0], reg_A[1], reg_A[2], reg_A[3],
                  reg_B[0], reg_B[1], 
                  reg_C[0], reg_C[1], reg_C[2], reg_C[3]);
}

__global__ void matmul_kernel(int M, int N, int K, half* d_A, half* d_B, half* d_C) {
    __shared__ half shm_A[4 * BLK_M * BLK_K];
    __shared__ half shm_B[4 * BLK_N * BLK_K];
    __shared__ half shm_C[BLK_M * BLK_N];

    uint32_t reg_A[4];
    uint32_t reg_B[2];
    uint32_t reg_C[4] = {0};

    pipe_load(d_A, d_B, shm_A, shm_B, M, N, K, 0);
    __pipeline_commit();
    pipe_load(d_A, d_B, shm_A, shm_B, M, N, K, 1);
    __pipeline_commit();
    pipe_load(d_A, d_B, shm_A, shm_B, M, N, K, 2);
    __pipeline_commit();

    for (int k = 3; k < K / MMA_K; k++) {
        pipe_load(d_A, d_B, shm_A, shm_B, M, N, K, k);
        __pipeline_commit();
        __pipeline_wait_prior(3);
        pipe_calc(shm_A, shm_B, reg_A, reg_B, reg_C, k - 3);
    }
    __pipeline_wait_prior(2);
    pipe_calc(shm_A, shm_B, reg_A, reg_B, reg_C, K / MMA_K - 3);
    __pipeline_wait_prior(1);
    pipe_calc(shm_A, shm_B, reg_A, reg_B, reg_C, K / MMA_K - 2);
    __pipeline_wait_prior(0);
    pipe_calc(shm_A, shm_B, reg_A, reg_B, reg_C, K / MMA_K - 1);

    store_reg_C(reg_C, shm_C);

    __syncthreads();

    store_shm_C(shm_C, d_C, M, N);
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

    matmul_kernel<<<dim3(cdiv(M, BLK_M), cdiv(N, BLK_N)), dim3(32, BLK_N/WARP_N, BLK_M/WARP_M)>>>(M, N, K, d_A, d_B, d_C);

    cudaDeviceSynchronize();
    gettimeofday(&tv, nullptr);
    end = tv.tv_sec + tv.tv_usec / 1.0e6;
    printf("matmul time: %lf\n", end - start);

    cudaMemcpy(h_C, d_C, M * N * sizeof(half), cudaMemcpyDeviceToHost);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}