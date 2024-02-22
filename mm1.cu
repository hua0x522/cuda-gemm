#include <cstdio>
#include <cstdlib>
#include <string>
#include <ctime>
#include <sys/time.h>
#include <cmath>
#include "cuda_fp16.h"
#include <mma.h>
#include <cuda.h>
#include "ptx.h"

#define cdiv(x, y) (((x) + (y) - 1) / (y))

#define MMA_M 16
#define MMA_N 8
#define MMA_K 16
#define WARP_SIZE 32

__global__ void matmul_kernel(int M, int N, int K, half* d_A, half* d_B, half* d_C) {
    const int k_tiles = cdiv(K, MMA_K);
    const int warp_row = blockIdx.x * MMA_M;
    const int warp_col = blockIdx.y * MMA_N;
    const int lane_id = (threadIdx.x + threadIdx.y * blockDim.x) % WARP_SIZE;

    __shared__ half A_smem[MMA_M][MMA_K];
    __shared__ half B_smem[MMA_N][MMA_K];
    __shared__ half C_smem[MMA_M][MMA_N];

    uint32_t RA[4];
    uint32_t RB[2];
    uint32_t RC[4] = {0, 0, 0, 0};

    for (size_t i = 0; i < k_tiles; i++) {
        *((int4*)(&A_smem[lane_id / 2][0]) + lane_id % 2) = 
            *((int4*)(&d_A[(warp_row + lane_id / 2) * K + i * MMA_K]) + lane_id % 2);

        if (lane_id < MMA_N * 2) {
            *((int4*)(&B_smem[lane_id / 2][0]) + lane_id % 2) = 
                *((int4*)(&d_B[i * MMA_K + (warp_col + lane_id / 2) * K]) + lane_id % 2);
        }

        __syncthreads();
    
        uint32_t A_smem_lane_addr = __cvta_generic_to_shared(&A_smem[lane_id % 16][(lane_id / 16) * 8]);
        LDMATRIX_X4(RA[0], RA[1], RA[2], RA[3], A_smem_lane_addr);

        uint32_t B_smem_lane_addr = __cvta_generic_to_shared(&B_smem[lane_id % 8][((lane_id / 8) % 2) * 8]);
        LDMATRIX_X2(RB[0], RB[1], B_smem_lane_addr);

        HMMA16816(RC[0], RC[1], RC[2], RC[3], RA[0], RA[1], RA[2], RA[3], RB[0], RB[1], RC[0], RC[1], RC[2], RC[3]);

        __syncthreads();
    }
    C_smem[lane_id / 4][(lane_id % 4) * 2] = __float2half(*((float*)&RC[0]));
    C_smem[lane_id / 4][(lane_id % 4) * 2 + 1] = __float2half(*((float*)&RC[1]));
    C_smem[lane_id / 4 + 8][(lane_id % 4) * 2] = __float2half(*((float*)&RC[2]));
    C_smem[lane_id / 4 + 8][(lane_id % 4) * 2 + 1] = __float2half(*((float*)&RC[3]));

    __syncthreads();

    if (lane_id < MMA_M) {
        *((int4 *)(&d_C[(warp_row + lane_id) * N + warp_col])) = *((int4 *)(&C_smem[lane_id][0]));
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

    matmul_kernel<<<dim3(cdiv(M, 16), cdiv(N, 8)), dim3(8, 4)>>>(M, N, K, d_A, d_B, d_C);

    cudaDeviceSynchronize();
    gettimeofday(&tv, nullptr);
    end = tv.tv_sec + tv.tv_usec / 1.0e6;
    printf("matmul time: %lf\n", end - start);

    cudaMemcpy(h_C, d_C, M * N * sizeof(half), cudaMemcpyDeviceToHost);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}