#include <cuda_fp16.h>
#include <mma.h>
#include <cuda.h>
#include <ctime>
#include <sys/time.h>
#include <cstdio>
#include <cuda_pipeline.h>

const int MI = 128;
const int NI = 128;
const int KI = 32;
const int MII = 64;
const int NII = 64;
const int KII = 16;
const int wmmaM = 16;
const int wmmaN = 16;
const int wmmaK = 16;

#define cdiv(x, y) (((x) + (y) - 1) / (y))

using FA = nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, wmmaM, wmmaN, wmmaK, half, nvcuda::wmma::row_major>;
using FB = nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, wmmaM, wmmaN, wmmaK, half, nvcuda::wmma::row_major>;
using FC = nvcuda::wmma::fragment<nvcuda::wmma::accumulator, wmmaM, wmmaN, wmmaK, float>;

__device__ void loadSmemA(half *smem, half *A, int M, int K, int ko)
{
    // load 128 * 32
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int tz = threadIdx.z;
    int tid = tz * 64 + ty * 32 + tx;
    for (int i = 0; i < 4; ++i)
    {
        int row = i * 32 + tid / 4;
        int col = tid % 4 * 8;
        // layout: [row_out, col_out, row_in, col_in] = [8, 2, 16, 16]

        void *ptr = (void *)(smem + row / 16 * (2 * 16 * 16) + col / 16 * (16 * 16) + row % 16 * 16 + col % 16);

        __pipeline_memcpy_async(
            ptr,
            &A[(by * 128 + row) * K + (ko * KI + col)],
            16
        );
    }
}

__device__ void loadSmemB(half *smem, half *B, int N, int K, int ko)
{
    // load 32 * 128
    int bx = blockIdx.x;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int tz = threadIdx.z;
    int tid = tz * 64 + ty * 32 + tx;
    for (int i = 0; i < 4; ++i)
    {
        int row = i * 8 + tid / 16;
        int col = tid % 16 * 8;
        // layout: [row_out, col_out, row_in, col_in] = [8, 2, 16, 16]

        void *ptr = (void *)(smem + row / 16 * (8 * 16 * 16) + col / 16 * (16 * 16) + row % 16 * 16 + col % 16);

        __pipeline_memcpy_async(
            ptr,
            &B[(ko * KI + row) * N + (bx * 128 + col)],
            16
        );
    }
}

__device__ void loadSmemC(float *smem, half *C, int M, int N)
{
    // load 128 * 128
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int tz = threadIdx.z;
    int tid = tz * 64 + ty * 32 + tx;
    for (int i = 0; i < 128; ++i)
    {
        int row = i;
        int col = tid;
        // layout: [row_out, col_out, row_in, col_in] = [8, 8, 16, 16]
        smem[row / 16 * (8 * 16 * 16) + col / 16 * (16 * 16) + row % 16 * 16 + col % 16] = (float)(C[(by * 128 + row) * N + bx * 128 + col]);
    }
}

__device__ void storeSmemC(half *C, float *smem, int M, int N)
{
    // load 128 * 128
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int tz = threadIdx.z;
    int tid = tz * 64 + ty * 32 + tx;
    for (int i = 0; i < 128; ++i)
    {
        int row = i;
        int col = tid;
        // layout: [row_out, col_out, row_in, col_in] = [8, 8, 16, 16]
        (C[(by * 128 + row) * N + bx * 128 + col]) = (half)smem[row / 16 * (8 * 16 * 16) + col / 16 * (16 * 16) + row % 16 * 16 + col % 16];
    }
}

__device__ void loadFragA(FA *frag, half *smem, int ki)
{
    // load 64x16
    int tz = threadIdx.z;
    for (int i = 0; i < 4; ++i)
    {
        int row = tz * 64 + i * 16;
        int col = ki * KII;
        nvcuda::wmma::load_matrix_sync(frag[i], smem + row / 16 * (2 * 16 * 16) + col / 16 * (16 * 16), 16);
    }
}

__device__ void loadFragB(FB *frag, half *smem, int ki)
{
    // load 16x64
    int ty = threadIdx.y;
    for (int i = 0; i < 4; ++i)
    {
        int row = ki * KII; 
        int col = ty * 64 + i * 16;
        nvcuda::wmma::load_matrix_sync(frag[i], smem + row / 16 * (8 * 16 * 16) + col / 16 * (16 * 16), 16);
    }
}

__device__ void storeAccum(float *ptr, FC *frag)
{
    // store 64x64
    int ty = threadIdx.y;
    int tz = threadIdx.z;
    for (int i = 0; i < 4; ++i)
    {
        for (int j = 0; j < 4; ++j)
        {
            int row = tz * 64 + i * 16;
            int col = ty * 64 + j * 16;
            // laoyut: [8, 8, 16, 16]
            nvcuda::wmma::store_matrix_sync(ptr + row / 16 * (8 * 16 * 16) + col / 16 * (16 * 16), frag[i * 4 + j], 16, nvcuda::wmma::mem_row_major);
        }
    }
}

__device__ void pipe_load(half* shm_A, half* shm_B, half* A, half* B, int M, int N, int K, int ko) {
    shm_A += (ko % 4) * MI * KI;
    shm_B += (ko % 4) * NI * KI;
    loadSmemA(shm_A, A, M, K, ko);
    loadSmemB(shm_B, B, N, K, ko);
}

__device__ void pipe_calc(FA* FragA, FB* FragB, FC* Accum, half* shm_A, half* shm_B, int ko) {
    shm_A += (ko % 4) * MI * KI;
    shm_B += (ko % 4) * NI * KI;
    for (int ki = 0; ki < KI / KII; ki += 1)
    {
        // 64x64x16 mma for each warp
        loadFragA(FragA, shm_A, ki);
        loadFragB(FragB, shm_B, ki);
        for (int mii = 0; mii < MII / wmmaM; mii += 1)
        {
            for (int nii = 0; nii < NII / wmmaN; nii += 1)
            {
                // 16x16x16 for each wmma
                nvcuda::wmma::mma_sync(Accum[mii * (NII / wmmaN) + nii], FragA[mii], FragB[nii], Accum[mii * (NII / wmmaN) + nii]);
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
    half* shm_B = shm_A + 4 * MI * KI;
    float *SC = reinterpret_cast<float *>(shared_storage);

    FA FragA[MII / wmmaM];
    FB FragB[NII / wmmaN];
    FC Accum[MII / wmmaM * NII / wmmaN];

    for (int mii = 0; mii < MII / wmmaM; mii += 1)
    {
        for (int nii = 0; nii < NII / wmmaN; nii += 1)
        {
            nvcuda::wmma::fill_fragment(Accum[mii * (NII / wmmaN) + nii], 0.0);
        }
    }

    pipe_load(shm_A, shm_B, A, B, M, N, K, 0);
    __pipeline_commit();

    pipe_load(shm_A, shm_B, A, B, M, N, K, 1);
    __pipeline_commit();

    pipe_load(shm_A, shm_B, A, B, M, N, K, 2);
    __pipeline_commit();

    for (int ko = 3; ko < K / KI; ko++) {
        pipe_load(shm_A, shm_B, A, B, M, N, K, ko);
        __pipeline_commit();
        __pipeline_wait_prior(3);
        pipe_calc(FragA, FragB, Accum, shm_A, shm_B, ko - 3);
        __syncthreads();
    }

    __pipeline_wait_prior(2);
    pipe_calc(FragA, FragB, Accum, shm_A, shm_B, K / KI - 3);
    __syncthreads();
    __pipeline_wait_prior(1);
    pipe_calc(FragA, FragB, Accum, shm_A, shm_B, K / KI - 2);
    __syncthreads();
    __pipeline_wait_prior(0);
    pipe_calc(FragA, FragB, Accum, shm_A, shm_B, K / KI - 1);
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
    int smem_size = 128 * 128 *4;
    if (smem_size >= (48 << 10))
    {
        cudaFuncSetAttribute(matmul_kernel,
                            cudaFuncAttributeMaxDynamicSharedMemorySize,
                            smem_size);
    }
    matmul_kernel<<<dim3(cdiv(N, 128), cdiv(M, 128)), dim3(32, 2, 2), smem_size, nullptr>>>(M, N, K, d_A, d_B, d_C);

    cudaDeviceSynchronize();
    gettimeofday(&tv, nullptr);
    end = tv.tv_sec + tv.tv_usec / 1.0e6;
    printf("matmul time: %lf\n", end - start);

    cudaMemcpy(h_C, d_C, M * N * sizeof(half), cudaMemcpyDeviceToHost);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}