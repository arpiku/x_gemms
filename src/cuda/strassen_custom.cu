#ifndef STRASSEN_CUSTOM_H
#define STRASSEN_CUSTOM_H

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include "strassen_common.cuh"

constexpr int STRASSEN_CUSTOM_THRESHOLD = 64;

class StrassenCustomContext {
public:
    void gemm_custom_device(const float* d_A, const float* d_B, float* d_C, int N) {
        dim3 block(16, 16);
        dim3 grid((N + 15) / 16, (N + 15) / 16);
        strassen_gemm_naive_kernel<<<grid, block>>>(d_A, d_B, d_C, N);
    }
    
    void matrix_add_device(const float* d_A, const float* d_B, float* d_C, int N) {
        int total = N * N;
        int block = 256;
        int grid = (total + block - 1) / block;
        strassen_matrix_add_kernel<<<grid, block>>>(d_A, d_B, d_C, N);
    }
    
    void matrix_sub_device(const float* d_A, const float* d_B, float* d_C, int N) {
        int total = N * N;
        int block = 256;
        int grid = (total + block - 1) / block;
        strassen_matrix_sub_kernel<<<grid, block>>>(d_A, d_B, d_C, N);
    }
    
    void extract_submatrix(const float* d_src, float* d_dst, int src_N, int dst_N, int row_offset, int col_offset) {
        dim3 block(16, 16);
        dim3 grid((dst_N + 15) / 16, (dst_N + 15) / 16);
        strassen_extract_submatrix_kernel<<<grid, block>>>(d_src, d_dst, src_N, dst_N, row_offset, col_offset);
    }
    
    void combine_submatrix(const float* d_src, float* d_dst, int src_N, int dst_N, int row_offset, int col_offset) {
        dim3 block(16, 16);
        dim3 grid((dst_N + 15) / 16, (dst_N + 15) / 16);
        strassen_combine_submatrix_kernel<<<grid, block>>>(d_src, d_dst, src_N, dst_N, row_offset, col_offset);
    }
};

void strassen_custom_impl(StrassenCustomContext& ctx, const float* d_A, const float* d_B, float* d_C, int N, int threshold);

void strassen_custom_impl(StrassenCustomContext& ctx, const float* d_A, const float* d_B, float* d_C, int N, int threshold) {
    if (N <= threshold || N % 2 != 0) {
        ctx.gemm_custom_device(d_A, d_B, d_C, N);
        return;
    }
    
    int mid = N / 2;
    size_t sub_size = mid * mid * sizeof(float);
    
    float *d_A11, *d_A12, *d_A21, *d_A22;
    float *d_B11, *d_B12, *d_B21, *d_B22;
    float *d_M1, *d_M2, *d_M3, *d_M4, *d_M5, *d_M6, *d_M7;
    float *d_T1, *d_T2;
    float *d_C11, *d_C12, *d_C21, *d_C22;
    
    cudaMalloc(&d_A11, sub_size);
    cudaMalloc(&d_A12, sub_size);
    cudaMalloc(&d_A21, sub_size);
    cudaMalloc(&d_A22, sub_size);
    cudaMalloc(&d_B11, sub_size);
    cudaMalloc(&d_B12, sub_size);
    cudaMalloc(&d_B21, sub_size);
    cudaMalloc(&d_B22, sub_size);
    
    cudaMalloc(&d_M1, sub_size);
    cudaMalloc(&d_M2, sub_size);
    cudaMalloc(&d_M3, sub_size);
    cudaMalloc(&d_M4, sub_size);
    cudaMalloc(&d_M5, sub_size);
    cudaMalloc(&d_M6, sub_size);
    cudaMalloc(&d_M7, sub_size);
    
    cudaMalloc(&d_T1, sub_size);
    cudaMalloc(&d_T2, sub_size);
    
    cudaMalloc(&d_C11, sub_size);
    cudaMalloc(&d_C12, sub_size);
    cudaMalloc(&d_C21, sub_size);
    cudaMalloc(&d_C22, sub_size);
    
    ctx.extract_submatrix(d_A, d_A11, N, mid, 0, 0);
    ctx.extract_submatrix(d_A, d_A12, N, mid, 0, mid);
    ctx.extract_submatrix(d_A, d_A21, N, mid, mid, 0);
    ctx.extract_submatrix(d_A, d_A22, N, mid, mid, mid);
    
    ctx.extract_submatrix(d_B, d_B11, N, mid, 0, 0);
    ctx.extract_submatrix(d_B, d_B12, N, mid, 0, mid);
    ctx.extract_submatrix(d_B, d_B21, N, mid, mid, 0);
    ctx.extract_submatrix(d_B, d_B22, N, mid, mid, mid);
    
    cudaDeviceSynchronize();
    
    ctx.matrix_add_device(d_A11, d_A22, d_T1, mid);
    ctx.matrix_add_device(d_B11, d_B22, d_T2, mid);
    strassen_custom_impl(ctx, d_T1, d_T2, d_M1, mid, threshold);
    
    ctx.matrix_add_device(d_A21, d_A22, d_T1, mid);
    strassen_custom_impl(ctx, d_T1, d_B11, d_M2, mid, threshold);
    
    ctx.matrix_sub_device(d_B12, d_B22, d_T1, mid);
    strassen_custom_impl(ctx, d_A11, d_T1, d_M3, mid, threshold);
    
    ctx.matrix_sub_device(d_B21, d_B11, d_T1, mid);
    strassen_custom_impl(ctx, d_A22, d_T1, d_M4, mid, threshold);
    
    ctx.matrix_add_device(d_A11, d_A12, d_T1, mid);
    strassen_custom_impl(ctx, d_T1, d_B22, d_M5, mid, threshold);
    
    ctx.matrix_sub_device(d_A21, d_A11, d_T1, mid);
    ctx.matrix_add_device(d_B11, d_B12, d_T2, mid);
    strassen_custom_impl(ctx, d_T1, d_T2, d_M6, mid, threshold);
    
    ctx.matrix_sub_device(d_A12, d_A22, d_T1, mid);
    ctx.matrix_add_device(d_B21, d_B22, d_T2, mid);
    strassen_custom_impl(ctx, d_T1, d_T2, d_M7, mid, threshold);
    
    cudaDeviceSynchronize();
    
    ctx.matrix_add_device(d_M1, d_M4, d_C11, mid);
    ctx.matrix_sub_device(d_C11, d_M5, d_C11, mid);
    ctx.matrix_add_device(d_C11, d_M7, d_C11, mid);
    
    ctx.matrix_add_device(d_M3, d_M5, d_C12, mid);
    
    ctx.matrix_add_device(d_M2, d_M4, d_C21, mid);
    
    ctx.matrix_sub_device(d_M1, d_M2, d_C22, mid);
    ctx.matrix_add_device(d_C22, d_M3, d_C22, mid);
    ctx.matrix_add_device(d_C22, d_M6, d_C22, mid);
    
    cudaDeviceSynchronize();
    
    ctx.combine_submatrix(d_C11, d_C, mid, N, 0, 0);
    ctx.combine_submatrix(d_C12, d_C, mid, N, 0, mid);
    ctx.combine_submatrix(d_C21, d_C, mid, N, mid, 0);
    ctx.combine_submatrix(d_C22, d_C, mid, N, mid, mid);
    
    cudaDeviceSynchronize();
    
    cudaFree(d_A11); cudaFree(d_A12); cudaFree(d_A21); cudaFree(d_A22);
    cudaFree(d_B11); cudaFree(d_B12); cudaFree(d_B21); cudaFree(d_B22);
    cudaFree(d_M1); cudaFree(d_M2); cudaFree(d_M3); cudaFree(d_M4);
    cudaFree(d_M5); cudaFree(d_M6); cudaFree(d_M7);
    cudaFree(d_T1); cudaFree(d_T2);
    cudaFree(d_C11); cudaFree(d_C12); cudaFree(d_C21); cudaFree(d_C22);
}

static int g_strassen_custom_threshold = STRASSEN_CUSTOM_THRESHOLD;

void gemm_strassen_custom_wrapper(const float* A, const float* B, float* C, int N) {
    StrassenCustomContext ctx;
    
    float *d_A, *d_B, *d_C;
    size_t size = N * N * sizeof(float);
    
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);
    
    cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice);
    
    strassen_custom_impl(ctx, d_A, d_B, d_C, N, g_strassen_custom_threshold);
    
    cudaMemcpy(C, d_C, size, cudaMemcpyDeviceToHost);
    
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

#endif
