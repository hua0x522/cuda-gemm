/*
    double-warp multi-tile
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

#define cdiv(x, y) (((x) + (y) - 1) / (y))

/*
    block size: [64, 64]
    warp  size: [64, 32]
    warp  num : [ 1,  2]
    k : 64
*/

__device__ void load_shm_A(half* shm_A, half* A, int M, int K, int ko) {
    // layout: [64, 64]
    int tid = threadIdx.y * 32 + threadIdx.x;
    for (int i = 0; i < 8; i++) {
        int row = i * 8 + tid / 8;
        int col = tid % 8 * 8;
        int shm_row = row;
        int shm_col = col ^ ((shm_row & 3) << 3);
        *(float4*)&shm_A[shm_row * 64 + shm_col] = *(float4*)&A[(blockIdx.x * 64 + row) * K + ko * 64 + col];
    }
    __syncthreads();
}

__device__ void load_shm_B(half* shm_B, half* B, int K, int N, int ko) {
    // layout: [64, 64]
    int tid = threadIdx.y * 32 + threadIdx.x;
    for (int i = 0; i < 8; i++) {
        int row = i * 8 + tid / 8;
        int col = tid % 8 * 8;
        int shm_row = row;
        int shm_col = col ^ ((shm_row & 3) << 3);
        *(float4*)&shm_B[shm_row * 64 + shm_col] = *(float4*)&B[(ko * 64 + row) * N + blockIdx.y * 64 + col];
    }
    __syncthreads();
}

__device__ void store_shm_C(float* shm_C, half* C, int M, int N) {
    // layout: [64, 64]
    int tid = threadIdx.y * 32 + threadIdx.x;
    for (int i = 0; i < 64; i++) {
        int row = i;
        int col = tid;
        col = col ^ ((row & 3) << 3);
        C[(blockIdx.x * 64 + row) * N + blockIdx.y * 64 + col] = __float2half(shm_C[row * 64 + col]);
    }
    __syncthreads();
}

__device__ void load_reg_A(uint32_t* reg_A, half* shm_A, int mi, int ki) {
    int lane_id = threadIdx.x;
    int row = mi * 16 + lane_id % 16;
    int col = ki * 16 + lane_id / 16 * 8;
    col = col ^ ((row & 3) << 3);
    uint32_t shm_A_lane_addr = __cvta_generic_to_shared(shm_A + row * 64 + col);
    LDMATRIX_X4(reg_A[0], reg_A[1], reg_A[2], reg_A[3], shm_A_lane_addr);
    __syncthreads();

}

__device__ void load_reg_B(uint32_t* reg_B, half* shm_B, int ki) {
    int lane_id = threadIdx.x;
    int row = ki * 16 + lane_id % 16;
    int col = threadIdx.y * 32;
    col = col ^ ((row & 3) << 3);
    for (int ni = 0; ni < 4; ni++) {
        uint32_t shm_B_lane_addr = __cvta_generic_to_shared(shm_B + row * 64 + col + ni * 8);
        LDMATRIX_X2_T(reg_B[ki * 8 + ni * 2], reg_B[ki * 8 + ni * 2 + 1], shm_B_lane_addr);
    }
}

__device__ void store_reg_C(uint32_t* reg_C, float* shm_C, int mi) {
    int lane_id = threadIdx.x;

    for (int ni = 0; ni < 4; ni++) {
        int row = mi * 16 + lane_id / 4;
        int col = threadIdx.y * 32 + ni * 8 + (lane_id % 4) * 2;
        col = col ^ ((row & 3) << 3);
        shm_C[row * 64 + col] += *(float*)(&reg_C[ni * 4]);
        shm_C[row * 64 + col + 1] += *(float*)(&reg_C[ni * 4 + 1]);
        shm_C[(row + 8) * 64 + col] += *(float*)(&reg_C[ni * 4 + 2]);
        shm_C[(row + 8) * 64 + col + 1] += *(float*)(&reg_C[ni * 4 + 3]);
    }
}

__device__ void clear_shm_C(float* shm_C) {
    int tid = threadIdx.y * 32 + threadIdx.x;
    for (int i = 0; i < 64; i++) {
        shm_C[i * 64 + tid] = 0;
    }
}

__device__ void clear_reg_C(uint32_t* reg_C) {
    for (int i = 0; i < 16; i++) {
        reg_C[i] = 0;
    }
}

__global__ void matmul_kernel(int M, int N, int K, half* d_A, half* d_B, half* d_C) {
    __shared__ half shm_A[64 * 16];
    __shared__ half shm_B[16 * 64];
    __shared__ float shm_C[64 * 64];

    uint32_t reg_A[4];
    uint32_t reg_B[4 * 4 * 2];
    uint32_t reg_C[4 * 4];
    clear_shm_C(shm_C);

    for (int k = 0; k < K / 64; k++) {
        load_shm_A(shm_A, d_A, M, K, k);
        load_shm_B(shm_B, d_B, K, N, k);
        __syncthreads();

        for (int ki = 0; ki < 4; ki++) {
            load_reg_B(reg_B, shm_B, ki);
        }

        for (int m = 0; m < 4; m++) {
            for (int ki = 0; ki < 4; ki++) {
                load_reg_A(reg_A, shm_A, m, ki);
                __syncthreads();

                for (int n = 0; n < 4; n++) {
                    HMMA16816(reg_C[n * 4], reg_C[n * 4 + 1], reg_C[n * 4 + 2], reg_C[n * 4 + 3],
                              reg_A[0], reg_A[1], reg_A[2], reg_A[3],
                              reg_B[n * 2], reg_B[n * 2 + 1],
                              reg_C[n * 4], reg_C[n * 4 + 1], reg_C[n * 4 + 2], reg_C[n * 4 + 3]);
                }
            }
            __syncthreads();
            store_reg_C(reg_C, shm_C, m);
        }
    }
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

    matmul_kernel<<<dim3(cdiv(M, 64), cdiv(N, 64)), dim3(32, 2)>>>(M, N, K, d_A, d_B, d_C);

    cudaDeviceSynchronize();
    gettimeofday(&tv, nullptr);
    end = tv.tv_sec + tv.tv_usec / 1.0e6;
    printf("matmul time: %lf\n", end - start);

    cudaMemcpy(h_C, d_C, M * N * sizeof(half), cudaMemcpyDeviceToHost);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}