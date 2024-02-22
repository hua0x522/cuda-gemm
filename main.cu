#include <cstdio>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <ctime>
#include <sys/time.h>
#include <cmath>
#include "cuda_fp16.h"

#define cdiv(x, y) (((x) + (y) - 1) / (y))

half* random_data(int size) {
    half* handle = (half*)malloc(size * sizeof(half));
    float sign = 1;
    for (int i = 0; i < size; i++) {
        handle[i] = sign * (1.0 * (i % 10)) / 10.0;
        sign = -sign;
    }
    return handle;
}

half* empty_data(int size) {
    half* handle = (half*)malloc(size * sizeof(half));
    for (int i = 0; i < size; i++) {
        handle[i] = 0;
    }
    return handle;
}

half* copy_data(half* data, int size) {
    half* handle = (half*)malloc(size * sizeof(half));
    for (int i = 0; i < size; i++) {
        handle[i] = data[i];
    }
    return handle;
}

void transpose(half* matrix, int m, int n) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < i; j++) {
            half temp = matrix[i * n + j];
            matrix[i * n + j] = matrix[j * n + i];
            matrix[j * n + i] = temp;
        }
    }
}

void check(half* A, half* B, int size) {
    for (int i = 0; i < size; i++) {
        float a = __half2float(A[i]);
        float b = __half2float(B[i]);
        if (fabs(a - b) / a >= 1e-3) {
            printf("error at %d, %lf, %lf\n", i, a, b);
            return;
        }
    }
    printf("check success\n");
}

extern void baseline(int M, int N, int K, half* h_A, half* h_B, half* h_C);
extern void matmul(int M, int N, int K, half* h_A, half* h_B, half* h_C);

int main(int argc, char* argv[]) {
    if (argc != 4) {
        printf("usage: ./main m n k\n");
        return 0;
    }

    int m = std::stoi(argv[1]);
    int n = std::stoi(argv[2]);
    int k = std::stoi(argv[3]);

    half* h_A = random_data(m * k);
    half* h_B = random_data(k * n);
    half* h_C = empty_data(m * n);
    half* B = copy_data(h_B, k * n);
    transpose(B, k, n);
    half* C = empty_data(m * n);

    baseline(m, n, k, h_A, h_B, h_C);
    matmul(m, n, k, h_A, B, C);

    check(h_C, C, m * n);
    return 0;
}