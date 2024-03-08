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

#define cdiv(x, y) (((x) + (y) - 1) / (y))

#define BLK_M 64
#define BLK_N 32
#define BLK_K 16
#define WARP_M 64
#define WARP_N 32
#define WARP_K 16
#define MMA_M 16
#define MMA_N 8
#define MMA_K 16

#define OFFSET(ptr, x, y, ld) ((ptr) + (x) * (ld) + (y))

using Reg4 = uint32_t[4];
using Reg2 = uint32_t[2];

__device__ void load_shm_A(half* shm_A, half* A, int M, int K, int ki, int mi) {
    *((float4*)OFFSET(shm_A, mi * 32 + threadIdx.x, 0, 8)) = 
        *(float4*)(OFFSET(A, blockIdx.x * BLK_M + mi * 16 + threadIdx.x % 16, ki * BLK_K + threadIdx.x / 16 * 8, K));
}

__device__ void load_shm_B(half* shm_B, half* B, int N, int K, int ki, int ni) {
    ni = ni / 2;
    *((float4*)OFFSET(shm_B, ni * 32 + threadIdx.x, 0, 8)) = 
        *(float4*)(OFFSET(B, ki * BLK_K + threadIdx.x % 16, blockIdx.y * BLK_N + ni * 16 + threadIdx.x / 16 * 8, N));
}

__device__ void load_reg_A(Reg4* reg_A, half* shm_A, int mi) {
    uint32_t shm_A_lane_addr = __cvta_generic_to_shared(OFFSET(shm_A, mi * 32 + threadIdx.x, 0, 8));
    LDMATRIX_X4(reg_A[mi][0], reg_A[mi][1], reg_A[mi][2], reg_A[mi][3], shm_A_lane_addr);
}

__device__ void load_reg_B(Reg2* reg_B, half* shm_B, int ni) {
    uint32_t shm_B_lane_addr = __cvta_generic_to_shared(OFFSET(shm_B, ni * 16 + threadIdx.x % 16, 0, 8));
    LDMATRIX_X2_T(reg_B[ni][0], reg_B[ni][1], shm_B_lane_addr);
}

__device__ void store_C(Reg4* reg_C, half* C, int N) {
    for (int mi = 0; mi < WARP_M / MMA_M; mi++) {
        for (int ni = 0; ni < WARP_N / MMA_N; ni++) {
            int idx = mi * WARP_N / MMA_N + ni;
            *OFFSET(C, blockIdx.x * BLK_M + mi * MMA_M + threadIdx.x / 4, blockIdx.y * BLK_N + ni * MMA_N + (threadIdx.x % 4) * 2, N) = 
                __float2half(*(float*)(&reg_C[idx][0]));
            *OFFSET(C, blockIdx.x * BLK_M + mi * MMA_M + threadIdx.x / 4, blockIdx.y * BLK_N + ni * MMA_N + (threadIdx.x % 4) * 2 + 1, N) = 
                __float2half(*(float*)(&reg_C[idx][1]));
            *OFFSET(C, blockIdx.x * BLK_M + mi * MMA_M + threadIdx.x / 4 + 8, blockIdx.y * BLK_N + ni * MMA_N + (threadIdx.x % 4) * 2, N) = 
                __float2half(*(float*)(&reg_C[idx][2]));
            *OFFSET(C, blockIdx.x * BLK_M + mi * MMA_M + threadIdx.x / 4 + 8, blockIdx.y * BLK_N + ni * MMA_N + (threadIdx.x % 4) * 2 + 1, N) = 
                __float2half(*(float*)(&reg_C[idx][3]));          
        }
    }
}

__global__ void matmul_kernel(int M, int N, int K, half* d_A, half* d_B, half* d_C) {
    __shared__ half shm_A[BLK_M * BLK_K]; // [2 * BLK_M, 8]
    __shared__ half shm_B[BLK_N * BLK_K]; // [2 * BLK_N, 8]

    Reg4 reg_A[WARP_M/MMA_M];
    Reg2 reg_B[WARP_N/MMA_N];
    Reg4 reg_C[WARP_M/MMA_M * WARP_N/MMA_N] = {0};

    for (int k = 0; k < K / MMA_K; k++) {
        for (int mi = 0; mi < WARP_M / MMA_M; mi++) {
            load_shm_A(shm_A, d_A, M, K, k, mi);
        }

        for (int ni = 0; ni < WARP_N / MMA_N; ni += 2) {
            load_shm_B(shm_B, d_B, N, K, k, ni);
        }

        __syncthreads();

        for (int mi = 0; mi < WARP_M / MMA_M; mi++) {
            load_reg_A(reg_A, shm_A, mi);
        }

        for (int ni = 0; ni < WARP_N / MMA_N; ni++) {
            load_reg_B(reg_B, shm_B, ni);
        }

        for (int m = 0; m < WARP_M / MMA_M; m++) {
            for (int n = 0; n < WARP_N / MMA_N; n++) {
                int idx = m * WARP_N / MMA_N + n;
                HMMA16816(reg_C[idx][0], reg_C[idx][1], reg_C[idx][2], reg_C[idx][3],
                          reg_A[m][0], reg_A[m][1], reg_A[m][2], reg_A[m][3],
                          reg_B[n][0], reg_B[n][1], 
                          reg_C[idx][0], reg_C[idx][1], reg_C[idx][2], reg_C[idx][3]);       
            } 
        }
        __syncthreads();
    }

    store_C(reg_C, d_C, N);
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