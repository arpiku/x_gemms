#ifndef STRASSEN_CUDA_COMMON_H
#define STRASSEN_CUDA_COMMON_H

#include <cuda_runtime.h>

__global__ void strassen_gemm_naive_kernel(const float* A, const float* B, float* C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < N && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < N; ++k) {
            sum += A[row * N + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

__global__ void strassen_matrix_add_kernel(const float* A, const float* B, float* C, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * N;
    if (idx < total) {
        C[idx] = A[idx] + B[idx];
    }
}

__global__ void strassen_matrix_sub_kernel(const float* A, const float* B, float* C, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * N;
    if (idx < total) {
        C[idx] = A[idx] - B[idx];
    }
}

__global__ void strassen_extract_submatrix_kernel(const float* src, float* dst, int src_N, int dst_N, int row_offset, int col_offset) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < dst_N && col < dst_N) {
        dst[row * dst_N + col] = src[(row + row_offset) * src_N + (col + col_offset)];
    }
}

__global__ void strassen_combine_submatrix_kernel(const float* src, float* dst, int src_N, int dst_N, int row_offset, int col_offset) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < src_N && col < src_N) {
        dst[(row + row_offset) * dst_N + (col + col_offset)] = src[row * src_N + col];
    }
}

#endif
