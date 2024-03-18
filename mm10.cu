#include <cuda_fp16.h>
#include <mma.h>
#include <cuda.h>
#include <ctime>
#include <sys/time.h>
#include <cstdio>
#include <cuda_pipeline.h>

#define BLK_M 128
#define BLK_N 64
#define BLK_K 32
#define WARP_M 64
#define WARP_N 64
#define MMA_M 16
#define MMA_N 16
#define MMA_K 16
#define num_threads (32 * (BLK_M / WARP_M) * (BLK_N / WARP_N))

#define cdiv(x, y) (((x) + (y) - 1) / (y))

using FA = nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, MMA_M, MMA_N, MMA_K, half, nvcuda::wmma::row_major>;
using FB = nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, MMA_M, MMA_N, MMA_K, half, nvcuda::wmma::row_major>;
using FC = nvcuda::wmma::fragment<nvcuda::wmma::accumulator, MMA_M, MMA_N, MMA_K, float>;

__device__ void loadSmemA(half *smem, half *A, int M, int K, int ko)
{
    // load BLK_M * BLK_K
    int tid = threadIdx.z * blockDim.y * blockDim.x + threadIdx.y * blockDim.x + threadIdx.x;
    for (int i = 0; i < (BLK_M * BLK_K) / (8 * num_threads); ++i)
    {
        int row = i * (8 * num_threads / BLK_K) + tid / (BLK_K / 8);
        int col = tid % (BLK_K / 8) * 8;
        // layout: [row_out, col_out, row_in, col_in] = [BLK_M / 16, BLK_K / 16, 16, 16]

        void *ptr = (void *)(smem + row / 16 * ((BLK_K / 16) * 16 * 16) + col / 16 * (16 * 16) + row % 16 * 16 + col % 16);

        __pipeline_memcpy_async(
            ptr,
            &A[(blockIdx.y * BLK_M + row) * K + (ko * BLK_K + col)],
            16
        );
    }
}

__device__ void loadSmemB(half *smem, half *B, int N, int K, int ko)
{
    // load BLK_K * BLK_N
    int tid = threadIdx.z * blockDim.y * blockDim.x + threadIdx.y * blockDim.x + threadIdx.x;
    for (int i = 0; i < (BLK_K * BLK_N) / (8 * num_threads); ++i)
    {
        int row = i * (8 * num_threads / BLK_N) + tid / (BLK_N / 8);
        int col = tid % (BLK_N / 8) * 8;
        // layout: [row_out, col_out, row_in, col_in] = [BLK_K / 16, BLK_N / 16, 16, 16]

        void *ptr = (void *)(smem + row / 16 * ((BLK_N / 16) * 16 * 16) + col / 16 * (16 * 16) + row % 16 * 16 + col % 16);

        __pipeline_memcpy_async(
            ptr,
            &B[(ko * BLK_K + row) * N + (blockIdx.x * BLK_N + col)],
            16
        );
        __syncthreads();
    }
}

__device__ void storeSmemC(half *C, float *smem, int M, int N)
{
    // load BLK_M * BLK_N
    int tid = threadIdx.z * blockDim.y * blockDim.x + threadIdx.y * blockDim.x + threadIdx.x;
    for (int i = 0; i < (BLK_M * BLK_N) / num_threads; i++) {
        int row = i * (num_threads / BLK_N) + tid / BLK_N;
        int col = tid % BLK_N;
        C[(blockIdx.y * BLK_M + row) * N + blockIdx.x * BLK_N + col] = 
            (half)smem[row / 16 * ((BLK_N / 16) * 16 * 16) + col / 16 * (16 * 16) + row % 16 * 16 + col % 16];
    }
}

__device__ void loadFragA(FA *frag, half *smem, int ki)
{
    // load WARP_M * WARP_K
    for (int i = 0; i < WARP_M / MMA_M; ++i)
    {
        int row = threadIdx.z * WARP_M + i * MMA_M;
        int col = ki * MMA_K;
        nvcuda::wmma::load_matrix_sync(frag[i], smem + row / 16 * ((BLK_K / 16) * 16 * 16) + col / 16 * (16 * 16), 16);
    }
}

__device__ void loadFragB(FB *frag, half *smem, int ki)
{
    // load WARP_K * WARP_N
    for (int i = 0; i < WARP_N / MMA_N; ++i)
    {
        int row = ki * MMA_K; 
        int col = threadIdx.y * WARP_N + i * MMA_N;
        nvcuda::wmma::load_matrix_sync(frag[i], smem + row / 16 * ((BLK_N / 16) * 16 * 16) + col / 16 * (16 * 16), 16);
    }
    __syncthreads();
}

__device__ void storeAccum(float *ptr, FC *frag)
{
    // store 64x64
    for (int i = 0; i < WARP_M / MMA_M; ++i)
    {
        for (int j = 0; j < WARP_N / MMA_N; ++j)
        {
            int row = threadIdx.z * WARP_M + i * MMA_M;
            int col = threadIdx.y * WARP_N + j * MMA_N;
            // layout: [WARP_M / MMA_M, WARP_N / MMA_N, MMA_M, MMA_N]
            nvcuda::wmma::store_matrix_sync(ptr + row / 16 * ((BLK_N / 16) * 16 * 16) + col / 16 * (16 * 16), 
                                            frag[i * (WARP_N / MMA_N) + j], 16, nvcuda::wmma::mem_row_major);
        }
    }
}

__device__ void pipe_load(half* shm_A, half* shm_B, half* A, half* B, int M, int N, int K, int ko) {
    shm_A += (ko % 4) * BLK_M * BLK_K;
    shm_B += (ko % 4) * BLK_N * BLK_K;
    loadSmemA(shm_A, A, M, K, ko);
    loadSmemB(shm_B, B, N, K, ko);
}

__device__ void pipe_calc(FA* FragA, FB* FragB, FC* Accum, half* shm_A, half* shm_B, int ko) {
    shm_A += (ko % 4) * BLK_M * BLK_K;
    shm_B += (ko % 4) * BLK_N * BLK_K;
    for (int ki = 0; ki < BLK_K / MMA_K; ki += 1)
    {
        // 64x64x16 mma for each warp
        loadFragA(FragA, shm_A, ki);
        loadFragB(FragB, shm_B, ki);
        for (int mii = 0; mii < WARP_M / MMA_M; mii += 1)
        {
            for (int nii = 0; nii < WARP_N / MMA_N; nii += 1)
            {
                // 16x16x16 for each wmma
                nvcuda::wmma::mma_sync(Accum[mii * (WARP_N / MMA_N) + nii], FragA[mii], FragB[nii], Accum[mii * (WARP_N / MMA_N) + nii]);
            }
        }
    }
}

__global__ void matmul_kernel(int M, int N, int K, half *A, half *B, half *C)
{
    // A is row-major
    // B is col-major
    // 128 threads [x, y, z] = [32, 2, 2]
    // threadblock mma: 128x128x32
    // warp mma: 64x64x16
    extern __shared__ uint8_t shared_storage[];
    half *shm_A = reinterpret_cast<half *>(shared_storage);
    half* shm_B = shm_A + 4 * BLK_M * BLK_K;
    float *SC = reinterpret_cast<float *>(shared_storage);

    FA FragA[WARP_M / MMA_M];
    FB FragB[WARP_N / MMA_N];
    FC Accum[WARP_M / MMA_M * WARP_N / MMA_N];

    for (int mii = 0; mii < WARP_M / MMA_M; mii += 1)
    {
        for (int nii = 0; nii < WARP_N / MMA_N; nii += 1)
        {
            nvcuda::wmma::fill_fragment(Accum[mii * (WARP_N / MMA_N) + nii], 0.0);
        }
    }

    pipe_load(shm_A, shm_B, A, B, M, N, K, 0);
    __pipeline_commit();

    pipe_load(shm_A, shm_B, A, B, M, N, K, 1);
    __pipeline_commit();

    pipe_load(shm_A, shm_B, A, B, M, N, K, 2);
    __pipeline_commit();

    for (int ko = 3; ko < K / BLK_K; ko++) {
        pipe_load(shm_A, shm_B, A, B, M, N, K, ko);
        __pipeline_commit();
        __pipeline_wait_prior(3);
        pipe_calc(FragA, FragB, Accum, shm_A, shm_B, ko - 3);
        __syncthreads();
    }

    __pipeline_wait_prior(2);
    pipe_calc(FragA, FragB, Accum, shm_A, shm_B, K / BLK_K - 3);
    __syncthreads();
    __pipeline_wait_prior(1);
    pipe_calc(FragA, FragB, Accum, shm_A, shm_B, K / BLK_K - 2);
    __syncthreads();
    __pipeline_wait_prior(0);
    pipe_calc(FragA, FragB, Accum, shm_A, shm_B, K / BLK_K - 1);
    __syncthreads();

    storeAccum(SC, Accum);
    __syncthreads();
    storeSmemC(C, SC, M, N);
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
    int smem_size = max(4 * (BLK_M + BLK_N) * BLK_K * 2, BLK_M * BLK_N * 4);
    if (smem_size >= (48 << 10))
    {
        cudaFuncSetAttribute(matmul_kernel,
                            cudaFuncAttributeMaxDynamicSharedMemorySize,
                            smem_size);
    }
    matmul_kernel<<<dim3(cdiv(N, BLK_N), cdiv(M, BLK_M)), dim3(32, BLK_N / WARP_N, BLK_M / WARP_M), smem_size, nullptr>>>(M, N, K, d_A, d_B, d_C);

    cudaDeviceSynchronize();
    gettimeofday(&tv, nullptr);
    end = tv.tv_sec + tv.tv_usec / 1.0e6;
    printf("matmul time: %lf\n", end - start);

    cudaMemcpy(h_C, d_C, M * N * sizeof(half), cudaMemcpyDeviceToHost);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}