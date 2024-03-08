/*
    single-warp multi-tile
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

#define BLK_M 32
#define BLK_N 16
#define BLK_K 16
#define WARP_M 32
#define WARP_N 16
#define WARP_K 16
#define MMA_M 16
#define MMA_N 8
#define MMA_K 16
#define stage 2

#define BLK_SIZE (32 * BLK_N / WARP_N * BLK_M / WARP_M)
#define OFFSET(ptr, x, y, ld) ((ptr) + (x) * (ld) + (y))
#define MAX(X, Y) (((X) > (Y)) ? (X) : (Y))

using Reg4 = uint32_t[4];
using Reg2 = uint32_t[2];

__device__ void load_shm_A(half* shm_A, half* A, int M, int K, int ko) {
    /*
    d_A => shm_A
    d_A[bid.x * BLK_M : (bid.x+1) * BLK_M, kid * BLK_K : (kid+1) * BLK_K]
    shm_A[0 : BLK_M, 0 : BLK_K]
    */
    int tid = threadIdx.z * 32 * BLK_M / WARP_M + threadIdx.y * 32 + threadIdx.x;
    half* blk_A = OFFSET(A, blockIdx.x * BLK_M, ko * BLK_K, K);

    #pragma unroll
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

    #pragma unroll
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

    #pragma unroll
    for (int t = tid; t < BLK_M * BLK_N / 8; t += BLK_SIZE) {
        int i = t / (BLK_N / 8);
        int j = t % (BLK_N / 8);
        *(OFFSET((float4*)blk_C, i, j, N / 8)) = *((float4*)shm_C + t);
    }
} 

__device__ void load_reg_A(Reg4* reg_A, half* shm_A) {
    int lane_id = threadIdx.x;
    #pragma unroll
    for (int m = 0; m < WARP_M / MMA_M; m++) {
        uint32_t shm_A_lane_addr = __cvta_generic_to_shared(OFFSET(shm_A, m * MMA_M + lane_id % 16, lane_id / 16 * 8, BLK_K));
        LDMATRIX_X4(reg_A[m][0], reg_A[m][1], reg_A[m][2], reg_A[m][3], shm_A_lane_addr);
    }
}

__device__ void load_reg_B(Reg2* reg_B, half* shm_B) {
    int lane_id = threadIdx.x;
    #pragma unroll
    for (int n = 0; n < WARP_N / MMA_N; n++) {
        uint32_t shm_B_lane_addr = __cvta_generic_to_shared(OFFSET(shm_B, n * MMA_N + lane_id % 8, ((lane_id / 8) % 2) * 8, BLK_K));
        LDMATRIX_X2(reg_B[n][0], reg_B[n][1], shm_B_lane_addr);
    }
}

__device__ void store_reg_C(Reg4* reg_C, half* shm_C) {
    int lane_id = threadIdx.x;
    #pragma unroll
    for (int m = 0; m < WARP_M / MMA_M; m++) {
        #pragma unroll
        for (int n = 0; n < WARP_N / MMA_N; n++) {
            int idx = m * WARP_N / MMA_N + n;
            *OFFSET(shm_C, m * MMA_M + lane_id / 4, n * MMA_N + (lane_id % 4) * 2, BLK_N) = __float2half(*(float*)(&reg_C[idx][0]));
            *OFFSET(shm_C, m * MMA_M + lane_id / 4, n * MMA_N + (lane_id % 4) * 2 + 1, BLK_N) = __float2half(*(float*)(&reg_C[idx][1]));
            *OFFSET(shm_C, m * MMA_M + lane_id / 4 + 8, n * MMA_N + (lane_id % 4) * 2, BLK_N) = __float2half(*(float*)(&reg_C[idx][2]));
            *OFFSET(shm_C, m * MMA_M + lane_id / 4 + 8, n * MMA_N + (lane_id % 4) * 2 + 1, BLK_N) = __float2half(*(float*)(&reg_C[idx][3]));
        }
    }
}

__device__ void pipe_load(half* d_A, half* d_B, half* shm_A, half* shm_B, int M, int N, int K, int ko) {
    shm_A += (ko % stage) * BLK_M * BLK_K;
    shm_B += (ko % stage) * BLK_N * BLK_K;

    load_shm_A(shm_A, d_A, M, K, ko);
    load_shm_B(shm_B, d_B, N, K, ko);
}

__device__ void pipe_calc(half* shm_A, half* shm_B, Reg4* reg_A, Reg2* reg_B, Reg4* reg_C, int ko) {
    shm_A += (ko % stage) * BLK_M * BLK_K;
    shm_B += (ko % stage) * BLK_N * BLK_K;
    
    load_reg_A(reg_A, shm_A);
    load_reg_B(reg_B, shm_B);

    #pragma unroll
    for (int m = 0; m < WARP_M / MMA_M; m++) {
        #pragma unroll
        for (int n = 0; n < WARP_N / MMA_N; n++) {
            int idx = m * WARP_N / MMA_N + n;
            HMMA16816(reg_C[idx][0], reg_C[idx][1], reg_C[idx][2], reg_C[idx][3],
                      reg_A[m][0], reg_A[m][1], reg_A[m][2], reg_A[m][3],
                      reg_B[n][0], reg_B[n][1], 
                      reg_C[idx][0], reg_C[idx][1], reg_C[idx][2], reg_C[idx][3]);       
        } 
    }
}

__global__ void matmul_kernel(int M, int N, int K, half* d_A, half* d_B, half* d_C) {
    // __shared__ half shm_A[BLK_M * BLK_K];
    // __shared__ half shm_B[BLK_N * BLK_K];
    __shared__ half shm_C[stage * MAX(BLK_M * BLK_N, (BLK_M + BLK_N) * BLK_K)];
    half* shm_A = shm_C;
    half* shm_B = shm_A + stage * BLK_M * BLK_K;

    Reg4 reg_A[WARP_M/MMA_M];
    Reg2 reg_B[WARP_N/MMA_N];
    Reg4 reg_C[WARP_M/MMA_M * WARP_N/MMA_N] = {0};

    #pragma unroll
    for (int i = 0; i < stage - 1; i++) {
        pipe_load(d_A, d_B, shm_A, shm_B, M, N, K, i);
        __pipeline_commit();
    }

    #pragma unroll
    for (int k = stage - 1; k < K / MMA_K; k++) {
        pipe_load(d_A, d_B, shm_A, shm_B, M, N, K, k);
        __pipeline_commit();
        __pipeline_wait_prior(stage - 1);
        pipe_calc(shm_A, shm_B, reg_A, reg_B, reg_C, k - stage + 1);
    }

    #pragma unroll
    for (int i = stage - 1; i > 0; i--) {
        __pipeline_wait_prior(i-1);
        pipe_calc(shm_A, shm_B, reg_A, reg_B, reg_C, K / MMA_K - i);
    }

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