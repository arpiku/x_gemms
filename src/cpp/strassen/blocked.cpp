#ifndef STRASSEN_BLOCKED_H
#define STRASSEN_BLOCKED_H

#include <cstddef>
#include <cstring>
#include <cstdlib>
#include <algorithm>
#include "common.h"

constexpr size_t STRASSEN_THRESHOLD_BLOCKED = 64;
constexpr size_t BLOCK_SIZE = 64;

template <typename T>
void gemm_blocked_base(const T* A, const T* B, T* C, size_t N, size_t block_size = BLOCK_SIZE) {
    for (size_t ii = 0; ii < N; ii += block_size) {
        for (size_t jj = 0; jj < N; jj += block_size) {
            for (size_t kk = 0; kk < N; kk += block_size) {
                size_t i_max = std::min(ii + block_size, N);
                size_t j_max = std::min(jj + block_size, N);
                size_t k_max = std::min(kk + block_size, N);
                
                for (size_t i = ii; i < i_max; ++i) {
                    for (size_t k = kk; k < k_max; ++k) {
                        T a_ik = A[i * N + k];
                        for (size_t j = jj; j < j_max; ++j) {
                            C[i * N + j] += a_ik * B[k * N + j];
                        }
                    }
                }
            }
        }
    }
}

template <typename T>
void strassen_blocked_impl(const T* A, const T* B, T* C, size_t N, size_t threshold, size_t block_size) {
    if (N <= threshold) {
        gemm_blocked_base(A, B, C, N, block_size);
        return;
    }
    
    if (N % 2 != 0) {
        gemm_blocked_base(A, B, C, N, block_size);
        return;
    }
    
    size_t mid = N / 2;
    size_t sub_size = mid * mid;
    
    T* A11 = (T*)malloc(sub_size * sizeof(T));
    T* A12 = (T*)malloc(sub_size * sizeof(T));
    T* A21 = (T*)malloc(sub_size * sizeof(T));
    T* A22 = (T*)malloc(sub_size * sizeof(T));
    T* B11 = (T*)malloc(sub_size * sizeof(T));
    T* B12 = (T*)malloc(sub_size * sizeof(T));
    T* B21 = (T*)malloc(sub_size * sizeof(T));
    T* B22 = (T*)malloc(sub_size * sizeof(T));
    
    for (size_t i = 0; i < mid; ++i) {
        for (size_t j = 0; j < mid; ++j) {
            A11[i * mid + j] = A[i * N + j];
            A12[i * mid + j] = A[i * N + mid + j];
            A21[i * mid + j] = A[(mid + i) * N + j];
            A22[i * mid + j] = A[(mid + i) * N + mid + j];
            
            B11[i * mid + j] = B[i * N + j];
            B12[i * mid + j] = B[i * N + mid + j];
            B21[i * mid + j] = B[(mid + i) * N + j];
            B22[i * mid + j] = B[(mid + i) * N + mid + j];
        }
    }
    
    T* T1 = (T*)malloc(sub_size * sizeof(T));
    T* T2 = (T*)malloc(sub_size * sizeof(T));
    
    T* M1 = (T*)malloc(sub_size * sizeof(T));
    T* M2 = (T*)malloc(sub_size * sizeof(T));
    T* M3 = (T*)malloc(sub_size * sizeof(T));
    T* M4 = (T*)malloc(sub_size * sizeof(T));
    T* M5 = (T*)malloc(sub_size * sizeof(T));
    T* M6 = (T*)malloc(sub_size * sizeof(T));
    T* M7 = (T*)malloc(sub_size * sizeof(T));
    
    strassen_add_matrix(A11, A22, T1, mid);
    strassen_add_matrix(B11, B22, T2, mid);
    strassen_blocked_impl(T1, T2, M1, mid, threshold, block_size);
    
    strassen_add_matrix(A21, A22, T1, mid);
    strassen_blocked_impl(T1, B11, M2, mid, threshold, block_size);
    
    strassen_sub_matrix(B12, B22, T1, mid);
    strassen_blocked_impl(A11, T1, M3, mid, threshold, block_size);
    
    strassen_sub_matrix(B21, B11, T1, mid);
    strassen_blocked_impl(A22, T1, M4, mid, threshold, block_size);
    
    strassen_add_matrix(A11, A12, T1, mid);
    strassen_blocked_impl(T1, B22, M5, mid, threshold, block_size);
    
    strassen_sub_matrix(A21, A11, T1, mid);
    strassen_add_matrix(B11, B12, T2, mid);
    strassen_blocked_impl(T1, T2, M6, mid, threshold, block_size);
    
    strassen_sub_matrix(A12, A22, T1, mid);
    strassen_add_matrix(B21, B22, T2, mid);
    strassen_blocked_impl(T1, T2, M7, mid, threshold, block_size);
    
    T* C11 = (T*)malloc(sub_size * sizeof(T));
    T* C12 = (T*)malloc(sub_size * sizeof(T));
    T* C21 = (T*)malloc(sub_size * sizeof(T));
    T* C22 = (T*)malloc(sub_size * sizeof(T));
    
    strassen_add_matrix(M1, M4, C11, mid);
    strassen_sub_matrix(C11, M5, C11, mid);
    strassen_add_matrix(C11, M7, C11, mid);
    
    strassen_add_matrix(M3, M5, C12, mid);
    
    strassen_add_matrix(M2, M4, C21, mid);
    
    strassen_sub_matrix(M1, M2, C22, mid);
    strassen_add_matrix(C22, M3, C22, mid);
    strassen_add_matrix(C22, M6, C22, mid);
    
    for (size_t i = 0; i < mid; ++i) {
        for (size_t j = 0; j < mid; ++j) {
            C[i * N + j] = C11[i * mid + j];
            C[i * N + mid + j] = C12[i * mid + j];
            C[(mid + i) * N + j] = C21[i * mid + j];
            C[(mid + i) * N + mid + j] = C22[i * mid + j];
        }
    }
    
    free(A11); free(A12); free(A21); free(A22);
    free(B11); free(B12); free(B21); free(B22);
    free(T1); free(T2);
    free(M1); free(M2); free(M3); free(M4); free(M5); free(M6); free(M7);
    free(C11); free(C12); free(C21); free(C22);
}

template <typename T>
void gemm_strassen_blocked(const T* A, const T* B, T* C, size_t N, size_t threshold = STRASSEN_THRESHOLD_BLOCKED, size_t block_size = BLOCK_SIZE) {
    std::memset(C, 0, N * N * sizeof(T));
    strassen_blocked_impl(A, B, C, N, threshold, block_size);
}

template void gemm_strassen_blocked<float>(const float*, const float*, float*, size_t, size_t, size_t);
template void gemm_strassen_blocked<double>(const double*, const double*, double*, size_t, size_t, size_t);

#endif
