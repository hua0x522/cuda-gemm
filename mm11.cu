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
/*
    block size: [128, 64]
    warp  size: [64, 32]
    warp  num : [ 2,  2]
    k : 32
*/

__device__ void load_shm_A(half* shm_A, half* A, int M, int K, int ko) {
    // global layout: [128, 32]
    // shared layout: [64, 64]
    int tid = threadIdx.z * 64 + threadIdx.y * 32 + threadIdx.x;
    for (int i = 0; i < 4; i++) {
        int row = i * 32 + tid / 4;
        int col = tid % 4 * 8;
        int shm_row = row / 2;
        int shm_col = col + (row & 1) * 32;
        shm_col = shm_col ^ ((shm_row & 3) << 3);
        __pipeline_memcpy_async(
            &shm_A[shm_row * 64 + shm_col],
            &A[(blockIdx.x * 128 + row) * K + ko * 32 + col],
            16
        );
        // *(float4*)&shm_A[shm_row * 64 + shm_col] = *(float4*)&A[(blockIdx.x * 128 + row) * K + ko * 32 + col];
    }
    __syncthreads();
}

__device__ void load_shm_B(half* shm_B, half* B, int K, int N, int ko) {
    // layout: [32, 64]
    int tid = threadIdx.z * 64 + threadIdx.y * 32 + threadIdx.x;
    for (int i = 0; i < 2; i++) {
        int row = i * 16 + tid / 8;
        int col = tid % 8 * 8;
        int shm_row = row;
        int shm_col = col ^ ((shm_row & 7) << 3);
        __pipeline_memcpy_async(
            &shm_B[shm_row * 64 + shm_col],
            &B[(ko * 32 + row) * N + blockIdx.y * 64 + col],
            16
        );
        // *(float4*)&shm_B[row * 72 + col] = *(float4*)&B[(ko * 32 + row) * N + blockIdx.y * 64 + col];
    }
    __syncthreads();
}

__device__ void load_reg_A(uint32_t* reg_A, half* shm_A, int ki) {
    for (int m = 0; m < 4; m++) {
        int lane_id = threadIdx.x;
        int row = threadIdx.z * 64 + m * 16 + lane_id % 16;
        int col = ki * 16 + lane_id / 16 * 8;
        int shm_row = row / 2;
        int shm_col = col + (row & 1) * 32;
        shm_col = shm_col ^ ((shm_row & 3) << 3);
        uint32_t shm_A_lane_addr = __cvta_generic_to_shared(shm_A + shm_row * 64 + shm_col);
        LDMATRIX_X4(reg_A[m * 4], reg_A[m * 4 + 1], reg_A[m * 4 + 2], reg_A[m * 4 + 3], shm_A_lane_addr);
    }
    __syncthreads();
}

__device__ void load_reg_B(uint32_t* reg_B, half* shm_B, int ki) {
    int lane_id = threadIdx.x;
    for (int ni = 0; ni < 2; ni++) {
        int row = ki * 16 + lane_id % 16;
        int col = threadIdx.y * 32 + ni * 16 + lane_id / 16 * 8;
        int shm_row = row;
        int shm_col = col ^ ((shm_row & 7) << 3);
        uint32_t shm_B_lane_addr = __cvta_generic_to_shared(shm_B + shm_row * 64 + shm_col);
        LDMATRIX_X4_T(reg_B[ni * 4], reg_B[ni * 4 + 1], reg_B[ni * 4 + 2], reg_B[ni * 4 + 3], shm_B_lane_addr);
    }
}

__device__ void store_C(uint32_t* reg_C, half* C, int M, int N) {
    int lane_id = threadIdx.x;
    for (int m = 0; m < 4; m++) {
        for (int n = 0; n < 4; n++) {
            int shm_row = threadIdx.z * 64 + m * 16 + lane_id / 4;
            int shm_col = threadIdx.y * 32 + n * 8 + (lane_id % 4) * 2;
            int row = shm_row + blockIdx.x * 128;
            int col = shm_col + blockIdx.y * 64;
            C[row * N + col] = __float2half(*(float*)&reg_C[m * 16 + n * 4]);
            C[row * N + col + 1] = __float2half(*(float*)&reg_C[m * 16 + n * 4 + 1]);
            C[(row + 8) * N + col] = __float2half(*(float*)&reg_C[m * 16 + n * 4 + 2]);
            C[(row + 8) * N + col + 1] = __float2half(*(float*)&reg_C[m * 16 + n * 4 + 3]);
        }
    }
}

__device__ void pipe_load(half* shm_A, half* shm_B, half* A, half* B, int M ,int N, int K, int ko) {
    shm_A += (ko % 3) * 64 * 64;
    shm_B += (ko % 3) * 32 * 72;
    load_shm_A(shm_A, A, M, K, ko);
    load_shm_B(shm_B, B, K, N, ko);
}

__device__ void pipe_calc(half* shm_A, half* shm_B, uint32_t* reg_A, uint32_t* reg_B, uint32_t* reg_C, int ko) {
    shm_A += (ko % 3) * 64 * 64;
    shm_B += (ko % 3) * 32 * 72;
    for (int ki = 0; ki < 2; ki++) {
        load_reg_A(reg_A, shm_A, ki);
        load_reg_B(reg_B, shm_B, ki);

        for (int m = 0; m < 4; m++) {
            for (int n = 0; n < 4; n++) {
                int idx = m * 4 + n;
                HMMA16816(reg_C[idx * 4], reg_C[idx * 4 + 1], reg_C[idx * 4 + 2], reg_C[idx * 4 + 3],
                          reg_A[m * 4], reg_A[m * 4 + 1], reg_A[m * 4 + 2], reg_A[m * 4 + 3],
                          reg_B[n * 2], reg_B[n * 2 + 1],
                          reg_C[idx * 4], reg_C[idx * 4 + 1], reg_C[idx * 4 + 2], reg_C[idx * 4 + 3]);
            }
        }
    }
}

__global__ void matmul_kernel(int M, int N, int K, half* d_A, half* d_B, half* d_C) {
    __shared__ half shm_A[3 * 64 * 64];
    __shared__ half shm_B[3 * 32 * 64];

    uint32_t reg_A[4 * 4];
    uint32_t reg_B[4 * 2];
    uint32_t reg_C[4 * 4 * 4] = {0};

    pipe_load(shm_A, shm_B, d_A, d_B, M, N, K, 0);
    __pipeline_commit();
    pipe_load(shm_A, shm_B, d_A, d_B, M, N, K, 1);
    __pipeline_commit();

    for (int k = 2; k < K / 32; k++) {
        pipe_load(shm_A, shm_B, d_A, d_B, M, N, K, k);
        __pipeline_commit();
        __pipeline_wait_prior(2);
        pipe_calc(shm_A, shm_B, reg_A, reg_B, reg_C, k - 2);
    }
    __pipeline_wait_prior(1);
    pipe_calc(shm_A, shm_B, reg_A, reg_B, reg_C, K / 32 - 2);
    __pipeline_wait_prior(0);
    pipe_calc(shm_A, shm_B, reg_A, reg_B, reg_C, K / 32 - 1);
    store_C(reg_C, d_C, M, N);
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

    matmul_kernel<<<dim3(cdiv(M, 128), cdiv(N, 64)), dim3(32, 2, 2)>>>(M, N, K, d_A, d_B, d_C);

    cudaDeviceSynchronize();
    gettimeofday(&tv, nullptr);
    end = tv.tv_sec + tv.tv_usec / 1.0e6;
    printf("matmul time: %lf\n", end - start);

    cudaMemcpy(h_C, d_C, M * N * sizeof(half), cudaMemcpyDeviceToHost);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}